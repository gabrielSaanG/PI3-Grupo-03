"""Sprint 5: provisional per-patch predictions with case IDs on the 559-patient split.

Runs the ResNet50 notebook cells up to the patient split (seed 42), checks that the
split matches outputs/sprint3/split_summary.csv, writes the split manifest, trains a
simple CNN baseline on the same split and, if the full-run checkpoint exists, runs
the ResNet50 on validation and test.

  .venv\\Scripts\\python.exe scripts/sprint5_predictions.py
  .venv\\Scripts\\python.exe scripts/sprint5_predictions.py --skip-cnn
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "03-resnet50-sprint3.ipynb"
SPLIT_SUMMARY = ROOT / "outputs" / "sprint3" / "split_summary.csv"
DICOM_INDEX = ROOT / "outputs" / "sprint3" / "dicom_index.csv"
RESNET_CKPT = ROOT / "models" / "resnet50" / "resnet50_sprint3_best.pth"
CNN_DIR = ROOT / "models" / "custom_cnn_559"
OUT = ROOT / "outputs" / "sprint5"
STOP_MARKER = "VERIFICAÇÃO DE DATA LEAKAGE"
SEED = 42
ID_COLS = ["patient_id", "study_uid", "sop_uid", "reading_session", "nodule_id"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_notebook_until_split() -> dict:
    os.chdir(ROOT / "notebooks")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns: dict = {"__name__": "__main__"}
    plt.show = lambda *args, **kwargs: None
    ns["display"] = lambda *args, **kwargs: None

    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if "dicom_records = []" in source and DICOM_INDEX.exists():
            print(f"Cell {index}: loading cached DICOM index {DICOM_INDEX}", flush=True)
            ns["dicom_index"] = pd.read_csv(DICOM_INDEX, dtype=str).assign(
                instance_number=lambda d: d["instance_number"].astype(int)
            )
            continue
        if "dicom_index.to_csv(" in source:
            continue
        print(f"Executing cell {index}", flush=True)
        try:
            exec(source, ns, ns)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        plt.close("all")
        if STOP_MARKER in source:
            return ns
    print("Stop marker not found.", flush=True)
    raise SystemExit(1)


def check_split(splits: dict[str, pd.DataFrame]) -> None:
    expected = pd.read_csv(SPLIT_SUMMARY).set_index("split")
    for name, frame in splits.items():
        got = (len(frame), frame["patient_id"].nunique(), int(frame["label"].sum()))
        want = tuple(int(expected.loc[name, c]) for c in ("patches", "patients", "malignant"))
        print(f"{name}: patches/patients/malignant = {got} (expected {want})", flush=True)
        if got != want:
            raise SystemExit(f"Split {name} does not match {SPLIT_SUMMARY}")


def write_manifest(splits: dict[str, pd.DataFrame]) -> None:
    cols = ID_COLS + ["annotation_index", "malignancy", "label"]
    parts = []
    for name, frame in splits.items():
        part = frame[cols].reset_index(drop=True)
        part.insert(0, "row_idx", np.arange(len(part)))
        part.insert(0, "split", name)
        parts.append(part)
    path = OUT / "split_manifest.csv"
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    print("Wrote", path, flush=True)


class PatchDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, imagenet: bool):
        self.images = frame["image"].tolist()
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        self.imagenet = imagenet

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = np.clip(self.images[idx].astype(np.float32), 0, 1)[None]
        if self.imagenet:
            image = (np.repeat(image, 3, axis=0) - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(image), torch.tensor(self.labels[idx])


class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x)).squeeze(1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device, softmax: bool) -> np.ndarray:
    model.eval()
    probs = []
    for images, _ in loader:
        logits = model(images.to(device, non_blocking=True))
        p = torch.softmax(logits, dim=1)[:, 1] if softmax else torch.sigmoid(logits)
        probs.append(p.cpu().numpy())
    return np.concatenate(probs)


def save_predictions(frame: pd.DataFrame, probs: np.ndarray, path: Path, meta: dict) -> None:
    out = frame[ID_COLS + ["label"]].reset_index(drop=True)
    out.insert(0, "row_idx", np.arange(len(out)))
    out["prob_malignant"] = probs
    for key, value in meta.items():
        out[key] = value
    out.to_csv(path, index=False)
    print("Wrote", path, flush=True)


def train_cnn(splits, device, epochs: int, batch_size: int) -> None:
    set_seed()
    train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]
    train_loader = DataLoader(
        PatchDataset(train_df, imagenet=False), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(PatchDataset(val_df, imagenet=False), batch_size=batch_size)
    test_loader = DataLoader(PatchDataset(test_df, imagenet=False), batch_size=batch_size)

    n_pos = int(train_df["label"].sum())
    n_neg = len(train_df) - n_pos
    model = ImageCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    CNN_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CNN_DIR / "best_model.pth"
    history, best_auc, best_epoch = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        val_auc = roc_auc_score(val_df["label"], predict(model, val_loader, device, softmax=False))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_df), "val_auc": val_auc})
        print(f"CNN epoch {epoch}/{epochs} loss={history[-1]['train_loss']:.4f} "
              f"val_auc={val_auc:.4f} ({time.time() - start:.0f}s)", flush=True)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save(model.state_dict(), ckpt)

    pd.DataFrame(history).to_csv(CNN_DIR / "training_history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    meta = {"model": "cnn_simple", "seed": SEED, "best_epoch": best_epoch, "status": "provisorio"}
    save_predictions(val_df, predict(model, val_loader, device, softmax=False),
                     OUT / "preds_val_cnn_simple.csv", meta)
    save_predictions(test_df, predict(model, test_loader, device, softmax=False),
                     OUT / "preds_test_cnn_simple.csv", meta)
    (CNN_DIR / "config.json").write_text(json.dumps({
        "architecture": "ImageCNN (scripts/train_from_crops.py), 1 channel, 224x224",
        "epochs": epochs, "batch_size": batch_size, "lr": 1e-3, "weight_decay": 1e-4,
        "pos_weight": n_neg / n_pos, "seed": SEED, "best_epoch": best_epoch,
        "best_val_auc_patch": best_auc, "split": "559 patients (391/84/84), seed 42",
    }, indent=2), encoding="utf-8")


def run_resnet(splits, device, batch_size: int) -> None:
    if not RESNET_CKPT.exists():
        print(f"[warn] {RESNET_CKPT} not found; skipping ResNet50.", flush=True)
        return
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(RESNET_CKPT, map_location=device))
    model = model.to(device)
    meta = {"model": "resnet50_single", "seed": SEED, "checkpoint": RESNET_CKPT.name,
            "status": "provisorio"}
    for name, tag in (("validation", "val"), ("test", "test")):
        frame = splits[name]
        loader = DataLoader(PatchDataset(frame, imagenet=True), batch_size=batch_size)
        probs = predict(model, loader, device, softmax=True)
        save_predictions(frame, probs, OUT / f"preds_{tag}_resnet50.csv", meta)
        if name == "test":
            y, pred = frame["label"].to_numpy(), (probs >= 0.5).astype(int)
            tn = int(((y == 0) & (pred == 0)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            tp = int(((y == 1) & (pred == 1)).sum())
            print(f"ResNet50 test patch confusion TN={tn} FP={fp} FN={fn} TP={tp} "
                  "(outputs/sprint3/test_results_resnet50.csv: TN=528 FP=496 FN=756 TP=2383)",
                  flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-resnet", action="store_true")
    args = parser.parse_args()

    ns = run_notebook_until_split()
    splits = {
        "train": ns["train_df"].reset_index(drop=True),
        "validation": ns["val_df"].reset_index(drop=True),
        "test": ns["test_df"].reset_index(drop=True),
    }
    for frame in splits.values():
        frame.drop(columns=["mask"], inplace=True, errors="ignore")
    del ns

    check_split(splits)
    OUT.mkdir(parents=True, exist_ok=True)
    write_manifest(splits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    if not args.skip_resnet:
        run_resnet(splits, device, args.batch_size)
    if not args.skip_cnn:
        train_cnn(splits, device, args.epochs, args.batch_size)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
