"""Generate Grad-CAM overlays for the trained fusion model (offline smoke)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai import GradCAM, plot_gradcam_grid  # noqa: E402

DEVICE = torch.device("cpu")
FUSION_DIR = ROOT / "models" / "fusion_cnn_mlp"
GRADCAM_DIR = ROOT / "gradcam" / "fusion"
SEMANTIC_COLS = [
    "subtlety_mean",
    "internalStructure_mean",
    "calcification_mean",
    "sphericity_mean",
    "margin_mean",
    "lobulation_mean",
    "spiculation_mean",
    "texture_mean",
]


class MultimodalFusion(nn.Module):
    def __init__(self, dropout_branch=0.3, dropout_head=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.img_proj = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_branch),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_branch),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(64, 1),
        )

    def forward(self, img, sem):
        img_feat = self.img_proj(self.cnn(img))
        sem_feat = self.mlp(sem)
        fused = torch.cat([img_feat, sem_feat], dim=1)
        return self.classifier(fused).squeeze(1)


def fix_crop_path(path: str) -> str:
    raw = str(path).replace("\\", "/")
    raw = raw.replace("/outputs/crops_fixed_v2/", "/outputs/crops/")
    p = Path(raw)
    if p.exists():
        return str(p)
    hits = list((ROOT / "outputs" / "crops").rglob(Path(raw).name))
    if hits:
        return str(hits[0])
    return str(ROOT / "outputs" / "crops" / Path(raw).name)


def assign_case_type(row) -> str:
    return {(0, 0): "TN", (1, 1): "TP", (0, 1): "FP", (1, 0): "FN"}[
        (int(row["label"]), int(row["pred"]))
    ]


def main() -> None:
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    model = MultimodalFusion().to(DEVICE)
    state = torch.load(
        FUSION_DIR / "best_fusion_cnn_mlp.pth",
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    img_mean = float(np.load(FUSION_DIR / "fusion_image_mean.npy"))
    img_std = float(np.load(FUSION_DIR / "fusion_image_std.npy"))

    scaler_path = FUSION_DIR / "fusion_semantic_scaler.pkl"
    scaler = None
    try:
        import joblib

        scaler = joblib.load(scaler_path)
    except Exception:
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            scaler = None

    if scaler is None:
        from sklearn.preprocessing import StandardScaler

        train_df = pd.read_csv(ROOT / "outputs" / "splits" / "train_df.csv")
        scaler = StandardScaler().fit(train_df[SEMANTIC_COLS].astype(np.float32))
        print("[warn] scaler.pkl unreadable; fitted StandardScaler on train_df")

    preds = pd.read_csv(FUSION_DIR / "fusion_cnn_mlp_test_predictions.csv")
    test_df = pd.read_csv(ROOT / "outputs" / "splits" / "test_df.csv")
    test_df = test_df.copy()
    test_df["crop_path"] = test_df["crop_path"].map(fix_crop_path)
    preds = preds.copy()
    preds["crop_path"] = preds["crop_path"].map(fix_crop_path)

    df = preds.merge(
        test_df[["patient_id", *SEMANTIC_COLS]].drop_duplicates("patient_id"),
        on="patient_id",
        how="left",
    )

    if "fusion_prob" not in df.columns:
        raise SystemExit("fusion_prob missing from predictions CSV")

    # Ensure crops exist
    df = df[df["crop_path"].map(lambda p: Path(str(p)).exists())].copy()
    if df.empty:
        raise SystemExit("No test rows with existing crop_path")

    threshold = 0.5
    df["pred"] = (df["fusion_prob"] >= threshold).astype(int)
    df["case_type"] = df.apply(assign_case_type, axis=1)

    picked = []
    for case_type in ["TN", "TP", "FP", "FN"]:
        sub = df[df["case_type"] == case_type]
        if case_type in {"TN", "FN"}:
            sub = sub.sort_values("fusion_prob", ascending=True)
        else:
            sub = sub.sort_values("fusion_prob", ascending=False)
        if len(sub):
            picked.append(sub.iloc[0])
    if not picked:
        picked = [df.iloc[0]]

    cases = pd.DataFrame(picked).reset_index(drop=True)
    print(cases[["case_type", "patient_id", "label", "fusion_prob", "pred"]])

    def prepare(row):
        raw_img = np.load(row["crop_path"]).astype(np.float32)
        img = (raw_img - img_mean) / img_std
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sem_raw = row[SEMANTIC_COLS].values.astype(np.float32).reshape(1, -1)
        sem_tensor = torch.tensor(scaler.transform(sem_raw).astype(np.float32))
        return img_tensor, sem_tensor, raw_img

    out_png = GRADCAM_DIR / "fusion_gradcam_selected_cases.png"
    with GradCAM(
        model,
        model.cnn[8],
        forward_fn=lambda m, img, sem: m(img, sem),
    ) as grad_cam:
        plot_gradcam_grid(
            cases,
            prepare_fn=prepare,
            generate_fn=grad_cam.generate,
            title="Grad-CAM Explanations for Fusion CNN-MLP Test Cases",
            save_path=str(out_png),
            show=False,
        )

    cases.to_csv(GRADCAM_DIR / "gradcam_selected_cases.csv", index=False)
    print("saved", out_png, "bytes", out_png.stat().st_size)


if __name__ == "__main__":
    main()
