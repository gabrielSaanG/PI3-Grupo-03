"""Sprint 5 (minimal, provisional): explainability of ResNet50 v2 against reader masks.

For one patch per physical nodule (the largest ROI) in validation and test:
  * maps: Grad-CAM (layer4), Integrated Gradients, random (smoothed noise) and
    centre (Gaussian at the crop centre), all for the predicted class;
  * metrics vs the reader ROI mask: IoU and Dice after binarising at a threshold
    on the min-max normalised map, Pointing Game (peak inside the mask dilated by
    15 px ~ 4.3 mm) and distance from the peak to the mask centroid in mm;
  * the threshold of each map is the one with the best mean IoU on validation,
    applied once to test.

  .venv\\Scripts\\python.exe scripts/sprint5_xai_v2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from scipy.ndimage import gaussian_filter
from torchvision import models

from sprint5_crop_preview import contour_v2
from sprint5_predictions import IMAGENET_MEAN, IMAGENET_STD, OUT, ROOT, SEED, run_notebook_until_split
from sprint5_retrain_resnet50 import CROP_MM, IMAGE_SIZE, extract_crop
from src.xai.gradcam import GradCAM

CKPT = ROOT / "models" / "resnet50_v2" / "best_model.pth"
FIG_DIR = ROOT / "docs" / "entrega" / "figures"
MAPS = ["gradcam", "ig", "random", "center"]
MAP_NAMES = {"gradcam": "Grad-CAM", "ig": "Integrated Gradients", "random": "Aleatório", "center": "Centrado"}
THRESHOLDS = np.round(np.arange(0.1, 0.95, 0.1), 2)
PG_TOLERANCE_PX = 15
MM_PER_PX = CROP_MM / IMAGE_SIZE
SIZE_BINS = [0, 6, 10, 20, np.inf]
SIZE_LABELS = ["<6 mm", "6-10 mm", "10-20 mm", ">=20 mm"]
N_BOOT = 1000


def normalise(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(m)


def to_input(crop: np.ndarray) -> torch.Tensor:
    image = (np.repeat(crop.astype(np.float32)[None], 3, axis=0) - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image.astype(np.float32))[None]


def center_map() -> np.ndarray:
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
    c = (IMAGE_SIZE - 1) / 2
    return np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * (IMAGE_SIZE / 8) ** 2)).astype(np.float32)


def case_metrics(maps: dict, mask: np.ndarray) -> dict:
    out = {}
    dilated = cv2.dilate(mask, np.ones((2 * PG_TOLERANCE_PX + 1,) * 2, np.uint8))
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    for name, m in maps.items():
        py, px = np.unravel_index(np.argmax(m), m.shape)
        out[f"{name}_pg_hit"] = int(dilated[py, px] > 0)
        out[f"{name}_peak_in_mask"] = int(mask[py, px] > 0)
        out[f"{name}_peak_dist_mm"] = float(np.hypot(py - cy, px - cx) * MM_PER_PX)
        for t in THRESHOLDS:
            binary = m >= t
            inter = np.logical_and(binary, mask > 0).sum()
            union = np.logical_or(binary, mask > 0).sum()
            out[f"{name}_iou@{t}"] = inter / union if union else 0.0
            out[f"{name}_dice@{t}"] = 2 * inter / (binary.sum() + mask.sum()) if binary.sum() + mask.sum() else 0.0
    return out


def pick_cases(split: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    split = split.copy()
    split["num_points"] = annotations.loc[split["annotation_index"], "points"].map(len).to_numpy()
    return split.sort_values("num_points", ascending=False).drop_duplicates("physical_nodule").reset_index(drop=True)


def patient_bootstrap(frame: pd.DataFrame, column: str, rng: np.random.Generator) -> tuple[float, float]:
    groups = [g[column].to_numpy() for _, g in frame.groupby("patient_id")]
    stats = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(groups), len(groups))
        stats.append(np.concatenate([groups[i] for i in idx]).mean())
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def summary_rows(frame: pd.DataFrame, best_t: dict, group: str, with_ci: bool, rng) -> list[dict]:
    rows = []
    for name in MAPS:
        t = best_t[name]
        cols = {"IoU": f"{name}_iou@{t}", "Dice": f"{name}_dice@{t}",
                "Pointing Game": f"{name}_pg_hit", "Pico dentro da máscara": f"{name}_peak_in_mask",
                "Distância do pico (mm)": f"{name}_peak_dist_mm"}
        row = {"grupo": group, "mapa": MAP_NAMES[name], "limiar": t, "n_casos": len(frame),
               "n_pacientes": frame["patient_id"].nunique()}
        for label, col in cols.items():
            row[label] = frame[col].mean() if len(frame) else np.nan
            if with_ci and len(frame):
                lo, hi = patient_bootstrap(frame, col, rng)
                row[f"{label} IC"] = f"[{lo:.3f}; {hi:.3f}]"
        rows.append(row)
    return rows


def main() -> None:
    ns = run_notebook_until_split()
    frames = {"validation": ns["val_df"].reset_index(drop=True), "test": ns["test_df"].reset_index(drop=True)}
    annotations = ns["matched_annotations"][["sop_uid", "dicom_path", "points"]].copy()
    del ns

    manifest = pd.read_csv(OUT / "split_manifest_v2.csv")
    tags = {"validation": "val", "test": "test"}
    for name, frame in frames.items():
        man = manifest[manifest["split"] == name].reset_index(drop=True)
        preds = pd.read_csv(OUT / f"preds_{tags[name]}_resnet50_v2.csv")
        if not (man["sop_uid"].astype(str).to_numpy() == frame["sop_uid"].astype(str).to_numpy()).all():
            raise SystemExit(f"{name}: rows not aligned with split_manifest_v2.csv")
        for col in ("label", "is_ambiguous", "consensus_malignancy", "physical_nodule"):
            frame[col] = man[col].to_numpy()
        frame["prob_malignant"] = preds["prob_malignant"].to_numpy()
        frame.drop(columns=["image", "mask"], inplace=True, errors="ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model = model.to(device).eval()
    gradcam = GradCAM(model, model.layer4, apply_sigmoid=False)
    ig = IntegratedGradients(model)
    baseline = to_input(np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)).to(device)
    center = center_map()
    rng = np.random.default_rng(SEED)

    records, examples = [], {}
    for name, frame in frames.items():
        cases = pick_cases(frame, annotations)
        start = time.time()
        for i, row in cases.iterrows():
            ann = annotations.loc[row["annotation_index"]]
            points = np.asarray(ann["points"], dtype=np.float32)
            cx, cy = points[:, 0].mean(), points[:, 1].mean()
            spacing = tuple(float(v) for v in pydicom.dcmread(ann["dicom_path"], stop_before_pixels=True).PixelSpacing)
            crop = extract_crop(ann["dicom_path"], cx, cy).astype(np.float32)
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
            cv2.fillPoly(mask, [np.round(contour_v2(points, cx, cy, spacing)).astype(np.int32)], 1)
            if mask.sum() == 0:
                continue

            x = to_input(crop).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(x), dim=1)[0, 1].item()
            pred = int(prob >= 0.5)
            cam = gradcam.generate(x, target_index=pred).cam
            attr = ig.attribute(x.clone().requires_grad_(True), baselines=baseline, target=pred,
                                n_steps=32, internal_batch_size=8)
            ig_map = normalise(attr.abs().sum(dim=1)[0].detach().cpu().numpy())
            random_map = normalise(gaussian_filter(rng.random((IMAGE_SIZE, IMAGE_SIZE)), sigma=8))
            maps = {"gradcam": normalise(cam), "ig": ig_map, "random": random_map, "center": center}

            size_mm = max((points[:, 0].max() - points[:, 0].min()) * spacing[1],
                          (points[:, 1].max() - points[:, 1].min()) * spacing[0])
            rec = {"split": name, "patient_id": row["patient_id"], "sop_uid": row["sop_uid"],
                   "reading_session": row["reading_session"], "nodule_id": row["nodule_id"],
                   "physical_nodule": int(row["physical_nodule"]), "label": int(row["label"]),
                   "is_ambiguous": int(row["is_ambiguous"]), "consensus_malignancy": row["consensus_malignancy"],
                   "prob_malignant": prob, "pred": pred, "size_mm": size_mm,
                   "mask_area_frac": float(mask.mean()), **case_metrics(maps, mask)}
            records.append(rec)
            if name == "test":
                examples[len(records) - 1] = (crop, mask, maps)
            if (i + 1) % 50 == 0:
                print(f"  {name}: {i + 1}/{len(cases)} cases ({time.time() - start:.0f}s)", flush=True)
        print(f"{name}: {len(cases)} physical nodules in {time.time() - start:.0f}s", flush=True)
    gradcam.close()

    per_case = pd.DataFrame(records)
    per_case.to_csv(OUT / "xai_v2_per_case.csv", index=False)

    sweep = []
    for split in ("validation", "test"):
        part = per_case[per_case["split"] == split]
        for name in MAPS:
            for t in THRESHOLDS:
                sweep.append({"split": split, "mapa": MAP_NAMES[name], "limiar": t,
                              "IoU": part[f"{name}_iou@{t}"].mean(), "Dice": part[f"{name}_dice@{t}"].mean()})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "xai_v2_threshold_sweep.csv", index=False)
    val_sweep = sweep[sweep["split"] == "validation"]
    best_t = {name: float(val_sweep[val_sweep["mapa"] == MAP_NAMES[name]].sort_values("IoU").iloc[-1]["limiar"])
              for name in MAPS}
    print("Thresholds chosen on validation (best mean IoU):", best_t, flush=True)

    test = per_case[per_case["split"] == "test"].copy()
    test["acerto"] = np.where(test["is_ambiguous"] == 1, "ambíguo",
                              np.where(test["pred"] == test["label"], "acerto", "erro"))
    test["tamanho"] = pd.cut(test["size_mm"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    rows = summary_rows(test, best_t, "teste (todos)", True, rng)
    for group, part in test.groupby("acerto"):
        rows += summary_rows(part, best_t, f"teste: {group}", False, rng)
    for group, part in test.groupby("tamanho", observed=True):
        rows += summary_rows(part, best_t, f"teste: {group}", False, rng)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "xai_v2_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False), flush=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name in MAPS:
        for split, style in (("validation", "--"), ("test", "-")):
            s = sweep[(sweep["split"] == split) & (sweep["mapa"] == MAP_NAMES[name])]
            ax[0].plot(s["limiar"], s["IoU"], style, label=f"{MAP_NAMES[name]} ({'val' if split == 'validation' else 'teste'})")
            ax[1].plot(s["limiar"], s["Dice"], style)
    ax[0].set_title("IoU médio por limiar"), ax[1].set_title("Dice médio por limiar")
    for a in ax:
        a.set_xlabel("limiar no mapa normalizado"), a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xai_v2_iou_limiar.png", dpi=110)

    t_cam = best_t["gradcam"]
    chosen = (
        [("Grad-CAM com maior IoU", i) for i in test.sort_values(f"gradcam_iou@{t_cam}").index[-2:]]
        + [("Grad-CAM fora do nódulo", i) for i in test[test["gradcam_pg_hit"] == 0].sample(
            min(2, int((test["gradcam_pg_hit"] == 0).sum())), random_state=SEED).index]
        + [("Erro do modelo", i) for i in test[test["acerto"] == "erro"].sample(
            min(2, int((test["acerto"] == "erro").sum())), random_state=SEED).index]
    )
    fig, axes = plt.subplots(len(chosen), 5, figsize=(14, 2.9 * len(chosen)))
    for r, (title, idx) in enumerate(chosen):
        crop, mask, maps = examples[idx]
        row = per_case.loc[idx]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        panels = [("TC + máscara", None)] + [(MAP_NAMES[n], n) for n in MAPS]
        for c, (label, key) in enumerate(panels):
            a = axes[r, c]
            a.imshow(crop, cmap="gray", vmin=0, vmax=1)
            if key is not None:
                a.imshow(maps[key], cmap="jet", alpha=0.45, vmin=0, vmax=1)
                py, px = np.unravel_index(np.argmax(maps[key]), maps[key].shape)
                a.plot(px, py, "w+", ms=10, mew=2)
                t = best_t[key]
                label = f"{label}\nIoU={row[f'{key}_iou@{t}']:.2f} PG={'sim' if row[f'{key}_pg_hit'] else 'não'}"
            for cnt in contours:
                a.plot(cnt[:, 0, 0], cnt[:, 0, 1], color="lime", lw=1)
            a.set_title(label, fontsize=8)
            a.set_xticks([]), a.set_yticks([])
        verdict = "ambíguo" if row["is_ambiguous"] else ("maligno" if row["label"] else "benigno")
        axes[r, 0].set_ylabel(f"{title}\n{row['patient_id']}\n{verdict}, p={row['prob_malignant']:.2f}\n"
                              f"~{row['size_mm']:.0f} mm", fontsize=8)
    fig.suptitle("ResNet50 v2 (provisório): mapas no teste; verde = ROI do leitor; + = pico do mapa", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xai_v2_exemplos.png", dpi=100)
    print("Wrote", OUT / "xai_v2_per_case.csv", OUT / "xai_v2_summary.csv", OUT / "xai_v2_threshold_sweep.csv")
    print("Wrote", FIG_DIR / "xai_v2_iou_limiar.png", FIG_DIR / "xai_v2_exemplos.png")


if __name__ == "__main__":
    main()
