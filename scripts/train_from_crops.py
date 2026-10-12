"""Train semantic / image / fusion baselines from the expanded crop table.

No Jupyter cell hunting. After expand_lidc_dataset.py:

  python scripts/train_from_crops.py

Optional:
  python scripts/train_from_crops.py --models semantic,image,fusion --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
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
TARGET_COL = "label"
IMAGE_COL = "crop_path"
BATCH_SIZE = 32


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_paths(root: Path = ROOT) -> dict[str, Path]:
    output_root = root / "outputs"
    models_root = root / "models"
    paths = {
        "root": root,
        "crops_csv": output_root / "preprocessing" / "lidc_model_table_fixed_crops.csv",
        "splits": output_root / "splits",
        "semantic": models_root / "semantic_mlp",
        "image": models_root / "image_cnn",
        "fusion": models_root / "fusion_cnn_mlp",
    }
    for key in ("splits", "semantic", "image", "fusion"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": float(acc),
        "auc": float(auc) if auc == auc else float("nan"),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_youden_threshold(y_true, y_prob) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    if len(thresholds) == 0:
        return 0.5, float("nan"), float("nan")
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    return float(thresholds[idx]), float(tpr[idx]), float(1 - fpr[idx])


def patient_split(df: pd.DataFrame, seed: int = SEED):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, df[TARGET_COL], groups=df["patient_id"]))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)
    train_idx, val_idx = next(
        gss2.split(train_val_df, train_val_df[TARGET_COL], groups=train_val_df["patient_id"])
    )
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df, test_df


class SemanticDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ImageOnlyDataset(Dataset):
    def __init__(self, df, mean, std, augment=False):
        self.df = df.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.load(row[IMAGE_COL]).astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
        img = (img - self.mean) / self.std
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(row[TARGET_COL], dtype=torch.float32)
        return img, label


class MultimodalDataset(Dataset):
    def __init__(self, df, X_sem, y, mean, std, augment=False):
        self.df = df.reset_index(drop=True)
        self.X_sem = X_sem.astype(np.float32)
        self.y = y.astype(np.float32)
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.load(row[IMAGE_COL]).astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
            img = np.rot90(img, np.random.choice([0, 1, 2, 3])).copy()
        img = (img - self.mean) / self.std
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        sem = torch.tensor(self.X_sem[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return img_t, sem, label


class SemanticMLP(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.30),
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, x):
        return self.classifier(self.encoder(x)).squeeze(1)


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


class MultimodalFusion(nn.Module):
    def __init__(self, dropout_branch=0.3, dropout_head=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
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
        fused = torch.cat([self.img_proj(self.cnn(img)), self.mlp(sem)], dim=1)
        return self.classifier(fused).squeeze(1)


def run_epoch_single(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            probs = torch.sigmoid(logits)
            total_loss += loss.item() * X_batch.size(0)
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(y_batch.detach().cpu().numpy())
    return total_loss / max(len(loader.dataset), 1), np.array(all_probs), np.array(all_labels)


def run_epoch_fusion(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    with torch.set_grad_enabled(is_train):
        for img, sem, y_batch in loader:
            img = img.to(device)
            sem = sem.to(device)
            y_batch = y_batch.to(device)
            logits = model(img, sem)
            loss = criterion(logits, y_batch)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            probs = torch.sigmoid(logits)
            total_loss += loss.item() * img.size(0)
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(y_batch.detach().cpu().numpy())
    return total_loss / max(len(loader.dataset), 1), np.array(all_probs), np.array(all_labels)


def _score_for_early_stop(val_auc: float, val_loss: float) -> float:
    if val_auc == val_auc:  # not NaN
        return float(val_auc)
    return float(-val_loss)


def train_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs: int,
    patience: int,
    best_path: Path,
    fusion: bool = False,
):
    best_score = -np.inf
    best_auc = float("nan")
    wait = 0
    history = []
    epoch_fn = run_epoch_fusion if fusion else run_epoch_single

    for epoch in range(1, epochs + 1):
        train_loss, train_probs, train_labels = epoch_fn(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_probs, val_labels = epoch_fn(
            model, val_loader, criterion, None, device
        )
        train_m = compute_metrics(train_labels, train_probs)
        val_m = compute_metrics(val_labels, val_probs)
        score = _score_for_early_stop(val_m["auc"], val_loss)
        scheduler.step(score if score == score else -val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_auc": train_m["auc"],
                "val_auc": val_m["auc"],
                "train_f1": train_m["f1"],
                "val_f1": val_m["f1"],
            }
        )

        improved = score > best_score
        if improved:
            best_score = score
            best_auc = val_m["auc"]
            wait = 0
            torch.save(model.state_dict(), best_path)
            tag = " [SAVED]"
        else:
            wait += 1
            tag = ""

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"  epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"train_auc={train_m['auc']:.4f} | val_auc={val_m['auc']:.4f}{tag}"
            )

        if wait >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    return pd.DataFrame(history), best_auc


def make_criterion(y_train, device):
    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    if num_pos == 0:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), float(pos_weight.item())


def load_and_validate(paths: dict[str, Path]) -> pd.DataFrame:
    csv_path = paths["crops_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}\nRun: python scripts/expand_lidc_dataset.py run --batch-size 25 --max-gb 8"
        )
    df = pd.read_csv(csv_path)
    required = set(SEMANTIC_COLS + [TARGET_COL, IMAGE_COL, "patient_id", "series_uid"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Crop table missing columns: {missing}")

    # Drop rows whose crop files disappeared
    ok = df[IMAGE_COL].map(lambda p: Path(str(p)).exists())
    if (~ok).any():
        print(f"Dropping {int((~ok).sum())} rows with missing .npy crops")
        df = df.loc[ok].reset_index(drop=True)

    n_patients = df["patient_id"].nunique()
    print(f"Loaded {len(df)} nodules / {n_patients} patients from {csv_path}")
    print(df[TARGET_COL].value_counts().to_string())
    if n_patients < 20:
        print(
            "WARNING: few patients — metrics will be noisy. "
            "Run expand_lidc_dataset.py again when you can."
        )
    return df


def train_semantic(train_df, val_df, test_df, paths, device, epochs, patience):
    print("\n=== Semantic MLP ===")
    out = paths["semantic"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[SEMANTIC_COLS].values)
    X_val = scaler.transform(val_df[SEMANTIC_COLS].values)
    X_test = scaler.transform(test_df[SEMANTIC_COLS].values)
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    y_val = val_df[TARGET_COL].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    with open(out / "semantic_scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    train_loader = DataLoader(SemanticDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SemanticDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SemanticDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = SemanticMLP(input_dim=len(SEMANTIC_COLS)).to(device)
    criterion, pw = make_criterion(y_train, device)
    print(f"pos_weight={pw:.3f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    best_path = out / "best_semantic_mlp.pth"

    history, best_auc = train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs, patience, best_path, fusion=False,
    )
    history.to_csv(out / "semantic_mlp_history.csv", index=False)

    model.load_state_dict(torch.load(best_path, map_location=device))
    _, test_probs, test_labels = run_epoch_single(model, test_loader, criterion, None, device)
    metrics = compute_metrics(test_labels, test_probs)
    result = pd.DataFrame([{"model": "Semantic-only MLP", **metrics}])
    result.to_csv(out / "semantic_only_mlp_results.csv", index=False)
    print("Test:", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    print(f"Best val AUC: {best_auc}")
    return metrics


def _image_stats(train_df):
    imgs = [np.load(p).astype(np.float32) for p in train_df[IMAGE_COL].values]
    stacked = np.stack(imgs, axis=0)
    return float(stacked.mean()), float(stacked.std() + 1e-8)


def train_image(train_df, val_df, test_df, paths, device, epochs, patience):
    print("\n=== Image CNN ===")
    out = paths["image"]
    img_mean, img_std = _image_stats(train_df)
    with open(out / "image_stats.json", "w", encoding="utf-8") as fh:
        json.dump({"mean": img_mean, "std": img_std}, fh, indent=2)

    y_train = train_df[TARGET_COL].values.astype(np.float32)
    train_loader = DataLoader(
        ImageOnlyDataset(train_df, img_mean, img_std, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        ImageOnlyDataset(val_df, img_mean, img_std, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        ImageOnlyDataset(test_df, img_mean, img_std, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ImageCNN().to(device)
    criterion, pw = make_criterion(y_train, device)
    print(f"pos_weight={pw:.3f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    best_path = out / "best_image_cnn.pth"

    history, best_auc = train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs, patience, best_path, fusion=False,
    )
    history.to_csv(out / "image_cnn_history.csv", index=False)

    model.load_state_dict(torch.load(best_path, map_location=device))
    _, test_probs, test_labels = run_epoch_single(model, test_loader, criterion, None, device)
    metrics = compute_metrics(test_labels, test_probs)
    result = pd.DataFrame([{"model": "Image-only CNN", **metrics}])
    result.to_csv(out / "image_only_cnn_results.csv", index=False)
    print("Test:", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    print(f"Best val AUC: {best_auc}")
    return metrics


def train_fusion(train_df, val_df, test_df, paths, device, epochs, patience):
    print("\n=== Fusion CNN-MLP ===")
    out = paths["fusion"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[SEMANTIC_COLS].values).astype(np.float32)
    X_val = scaler.transform(val_df[SEMANTIC_COLS].values).astype(np.float32)
    X_test = scaler.transform(test_df[SEMANTIC_COLS].values).astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    y_val = val_df[TARGET_COL].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    with open(out / "fusion_scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    img_mean, img_std = _image_stats(train_df)
    with open(out / "fusion_image_stats.json", "w", encoding="utf-8") as fh:
        json.dump({"mean": img_mean, "std": img_std}, fh, indent=2)

    train_loader = DataLoader(
        MultimodalDataset(train_df, X_train, y_train, img_mean, img_std, True),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        MultimodalDataset(val_df, X_val, y_val, img_mean, img_std, False),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        MultimodalDataset(test_df, X_test, y_test, img_mean, img_std, False),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MultimodalFusion().to(device)
    criterion, pw = make_criterion(y_train, device)
    print(f"pos_weight={pw:.3f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    best_path = out / "best_fusion_cnn_mlp.pth"

    history, best_auc = train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs, patience, best_path, fusion=True,
    )
    history.to_csv(out / "fusion_training_history.csv", index=False)

    model.load_state_dict(torch.load(best_path, map_location=device))
    _, test_probs, test_labels = run_epoch_fusion(model, test_loader, criterion, None, device)
    metrics_05 = compute_metrics(test_labels, test_probs, threshold=0.5)

    _, val_probs, val_labels = run_epoch_fusion(model, val_loader, criterion, None, device)
    thr, _, _ = get_youden_threshold(val_labels, val_probs)
    metrics_youden = compute_metrics(test_labels, test_probs, threshold=thr)

    fusion_results = pd.DataFrame(
        [
            {"model": "Fusion CNN-MLP", "threshold_type": "default_0.5", "threshold": 0.5, **metrics_05},
            {
                "model": "Fusion CNN-MLP",
                "threshold_type": "validation_youden_j",
                "threshold": thr,
                **metrics_youden,
            },
        ]
    )
    fusion_results.to_csv(out / "fusion_cnn_mlp_test_results.csv", index=False)
    print("Test@0.5:", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics_05.items()})
    print(f"Youden threshold={thr:.4f}")
    print(f"Best val AUC: {best_auc}")
    return metrics_05


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train baselines from expanded LIDC crop table")
    parser.add_argument(
        "--models",
        default="semantic,image,fusion",
        help="Comma list: semantic,image,fusion",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    paths = discover_paths(ROOT)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    df = load_and_validate(paths)
    train_df, val_df, test_df = patient_split(df, seed=args.seed)
    print(
        f"Split -> train={len(train_df)} val={len(val_df)} test={len(test_df)} | "
        f"patients train/val/test="
        f"{train_df.patient_id.nunique()}/{val_df.patient_id.nunique()}/{test_df.patient_id.nunique()}"
    )
    train_df.to_csv(paths["splits"] / "train_df.csv", index=False)
    val_df.to_csv(paths["splits"] / "val_df.csv", index=False)
    test_df.to_csv(paths["splits"] / "test_df.csv", index=False)

    wanted = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    if "semantic" in wanted:
        train_semantic(train_df, val_df, test_df, paths, device, args.epochs, args.patience)
    if "image" in wanted:
        train_image(train_df, val_df, test_df, paths, device, args.epochs, args.patience)
    if "fusion" in wanted:
        train_fusion(train_df, val_df, test_df, paths, device, args.epochs, args.patience)

    print("\nDone. Weights/results under models/{semantic_mlp,image_cnn,fusion_cnn_mlp}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
