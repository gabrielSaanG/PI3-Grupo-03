"""Sprint 6: shared setup for the ResNet50 v2 checks (Adebayo, shift test, pleura).

Loads the 559-patient split through the notebook, attaches the v2 manifest
(consensus label, ambiguous flag, physical nodule) and the v2 predictions, and
rebuilds the 64 mm crop and the reader mask exactly as scripts/sprint5_xai_v2.py,
so every Sprint 6 analysis runs on the same cases, crops and map thresholds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from torchvision import models

from sprint5_crop_preview import contour_v2
from sprint5_predictions import OUT as OUT5, ROOT, SEED, run_notebook_until_split
from sprint5_retrain_resnet50 import CROP_MM, HU_MAX, HU_MIN, IMAGE_SIZE
from sprint5_xai_v2 import MAP_NAMES, PG_TOLERANCE_PX, center_map, normalise, to_input
from src.xai.gradcam import GradCAM

OUT = ROOT / "outputs" / "sprint6"
FIG_DIR = ROOT / "docs" / "sprint6" / "figures"
V2_CKPT = ROOT / "models" / "resnet50_v2" / "best_model.pth"
MM_PER_PX = CROP_MM / IMAGE_SIZE
TAGS = {"train": "train", "validation": "val", "test": "test"}
NS_KEYS = {"train": "train_df", "validation": "val_df", "test": "test_df"}
N_BOOT = 1000
LUNG_HU = -400.0
LUNG_CLOSING_MM = 5.0
LUNG_MIN_AREA_PX = 200


def load_frames(names=("validation", "test")) -> dict[str, pd.DataFrame]:
    ns = run_notebook_until_split()
    frames = {n: ns[NS_KEYS[n]].reset_index(drop=True) for n in names}
    annotations = ns["matched_annotations"][["sop_uid", "dicom_path", "points"]].copy()
    del ns

    manifest = pd.read_csv(OUT5 / "split_manifest_v2.csv")
    for name, frame in frames.items():
        frame.drop(columns=["image", "mask"], inplace=True, errors="ignore")
        man = manifest[manifest["split"] == name].reset_index(drop=True)
        if not (man["sop_uid"].astype(str).to_numpy() == frame["sop_uid"].astype(str).to_numpy()).all():
            raise SystemExit(f"{name}: rows not aligned with split_manifest_v2.csv")
        for col in ("label", "is_ambiguous", "consensus_malignancy", "physical_nodule"):
            frame[col] = man[col].to_numpy()
        preds = OUT5 / f"preds_{TAGS[name]}_resnet50_v2.csv"
        if preds.exists():
            frame["prob_v2"] = pd.read_csv(preds)["prob_malignant"].to_numpy()
        ann = annotations.loc[frame["annotation_index"]]
        if not (ann["sop_uid"].astype(str).to_numpy() == frame["sop_uid"].astype(str).to_numpy()).all():
            raise SystemExit(f"{name}: annotation_index does not point to the same slice")
        points = [np.asarray(p, dtype=np.float32) for p in ann["points"]]
        frame["dicom_path"] = ann["dicom_path"].to_numpy()
        frame["cx"] = [p[:, 0].mean() for p in points]
        frame["cy"] = [p[:, 1].mean() for p in points]
        frame["num_points"] = [len(p) for p in points]
        frame["points"] = pd.Series(points, index=frame.index, dtype=object)
    return frames


def one_patch_per_nodule(frame: pd.DataFrame) -> pd.DataFrame:
    """Same selection as sprint5_xai_v2.pick_cases: the largest ROI of each physical nodule."""
    return frame.sort_values("num_points", ascending=False).drop_duplicates("physical_nodule").reset_index(drop=True)


def check_against_sprint5(cases: pd.DataFrame) -> None:
    ref = pd.read_csv(OUT5 / "xai_v2_per_case.csv", dtype={"sop_uid": str})
    ref = ref[ref["split"] == "test"]
    key = ["sop_uid", "reading_session", "nodule_id"]
    a = set(map(tuple, cases[key].astype(str).to_numpy()))
    b = set(map(tuple, ref[key].astype(str).to_numpy()))
    print(f"Sample check vs outputs/sprint5/xai_v2_per_case.csv: {len(a & b)}/{len(b)} test cases identical", flush=True)


def validation_thresholds() -> dict[str, float]:
    sweep = pd.read_csv(OUT5 / "xai_v2_threshold_sweep.csv")
    val = sweep[sweep["split"] == "validation"]
    return {key: float(val[val["mapa"] == label].sort_values("IoU").iloc[-1]["limiar"])
            for key, label in MAP_NAMES.items()}


def read_slice(path: str) -> tuple[np.ndarray, tuple[float, float]]:
    ds = pydicom.dcmread(path)
    hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) \
        + float(getattr(ds, "RescaleIntercept", 0))
    return hu, tuple(float(v) for v in ds.PixelSpacing)


def crop_geometry(array: np.ndarray, cx: float, cy: float, spacing: tuple[float, float],
                  fill: float, interpolation: int) -> np.ndarray:
    """Same window as sprint5_retrain_resnet50.extract_crop, for any array on the slice grid."""
    row_mm, col_mm = spacing
    half_x = int(round(CROP_MM / 2 / col_mm))
    half_y = int(round(CROP_MM / 2 / row_mm))
    pad = max(half_x, half_y) + 2
    padded = np.pad(array, pad, mode="constant", constant_values=fill)
    x, y = int(round(cx)) + pad, int(round(cy)) + pad
    crop = padded[y - half_y:y + half_y, x - half_x:x + half_x]
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interpolation)


def crop_ct(hu: np.ndarray, cx: float, cy: float, spacing) -> np.ndarray:
    crop = crop_geometry(hu, cx, cy, spacing, HU_MIN, cv2.INTER_LINEAR)
    return ((np.clip(crop, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float16).astype(np.float32)


def crop_mask(points: np.ndarray, cx: float, cy: float, spacing) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    cv2.fillPoly(mask, [np.round(contour_v2(points, cx, cy, spacing)).astype(np.int32)], 1)
    return mask


def lung_mask(hu: np.ndarray, spacing) -> np.ndarray:
    """Air below -400 HU not connected to the image border, two largest parts, closed and hole-filled.

    Juxtapleural nodules stay outside the raw mask; the closing (5 mm) re-attaches small ones,
    so the nodule itself is always handled through the reader mask, not through this mask.
    """
    air = hu < LUNG_HU
    lab, n = ndimage.label(air)
    if n == 0:
        return np.zeros_like(air)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    sizes = ndimage.sum(air, lab, index=np.arange(1, n + 1))
    keep = [i + 1 for i in np.argsort(sizes)[::-1] if i + 1 not in border and sizes[i] >= LUNG_MIN_AREA_PX][:2]
    lung = np.isin(lab, keep)
    r = max(1, int(round(LUNG_CLOSING_MM / spacing[0])))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    lung = ndimage.binary_closing(lung, structure=(xx ** 2 + yy ** 2) <= r ** 2)
    return ndimage.binary_fill_holes(lung)


def load_resnet(path: Path, device) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


class Explainer:
    """Grad-CAM (layer4) and Integrated Gradients (air baseline, 32 steps) as in Sprint 5."""

    def __init__(self, model: nn.Module, device):
        self.model, self.device = model, device
        self.gradcam = GradCAM(model, model.layer4, apply_sigmoid=False)
        self.ig = IntegratedGradients(model)
        self.baseline = to_input(np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)).to(device)

    @torch.no_grad()
    def probs(self, crops: list[np.ndarray], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(crops), batch):
            x = torch.cat([to_input(c) for c in crops[i:i + batch]]).to(self.device)
            out.append(torch.softmax(self.model(x), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def maps(self, crop: np.ndarray, target: int, with_ig: bool = True) -> dict[str, np.ndarray]:
        x = to_input(crop).to(self.device)
        out = {"gradcam": normalise(self.gradcam.generate(x, target_index=target).cam)}
        if with_ig:
            attr = self.ig.attribute(x.clone().requires_grad_(True), baselines=self.baseline, target=target,
                                     n_steps=32, internal_batch_size=8)
            out["ig"] = normalise(attr.abs().sum(dim=1)[0].detach().cpu().numpy())
        return out

    def close(self) -> None:
        self.gradcam.close()


def localisation(maps: dict[str, np.ndarray], mask: np.ndarray, thresholds: dict[str, float]) -> dict:
    """Pointing Game (mask dilated by 15 px), IoU at the validation threshold, peak distances in mm."""
    dilated = cv2.dilate(mask, np.ones((2 * PG_TOLERANCE_PX + 1,) * 2, np.uint8))
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    centre = (IMAGE_SIZE - 1) / 2
    out = {}
    for name, m in maps.items():
        py, px = np.unravel_index(np.argmax(m), m.shape)
        binary = m >= thresholds[name]
        union = np.logical_or(binary, mask > 0).sum()
        out[f"{name}_pg_hit"] = int(dilated[py, px] > 0)
        out[f"{name}_iou"] = float(np.logical_and(binary, mask > 0).sum() / union) if union else 0.0
        out[f"{name}_peak_dist_mm"] = float(np.hypot(py - cy, px - cx) * MM_PER_PX)
        out[f"{name}_peak_centre_mm"] = float(np.hypot(py - centre, px - centre) * MM_PER_PX)
    return out


def _patient_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    return [np.flatnonzero(frame["patient_id"].to_numpy() == p) for p in frame["patient_id"].unique()]


def ci_mean(frame: pd.DataFrame, column: str, seed: int = SEED) -> tuple[float, float]:
    groups, values = _patient_groups(frame), frame[column].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    stats = [np.nanmean(values[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])])
             for _ in range(N_BOOT)]
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def ci_auc(frame: pd.DataFrame, score: str, label: str = "label", seed: int = SEED) -> tuple[float, float]:
    groups = _patient_groups(frame)
    y, s = frame[label].to_numpy(), frame[score].to_numpy()
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) == 2:
            stats.append(roc_auc_score(y[idx], s[idx]))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def fmt_ci(value: float, ci: tuple[float, float], digits: int = 3) -> str:
    return f"{value:.{digits}f} [{ci[0]:.{digits}f}; {ci[1]:.{digits}f}]"
