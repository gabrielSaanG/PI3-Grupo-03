"""Sprint 5: ResNet50 v2 retrained on the same 559-patient split.

Changes against the notebook ResNet50 (v1), all decided before looking at test:
  * label = mean malignancy of the readers who annotated the same physical nodule,
    malignant if mean >= 3.5; ambiguous (2.5 < mean < 3.5) left out of train/val;
  * crop of fixed physical size (64 mm) around the ROI centre, resized to 224;
  * augmentation (rot90, flips, shift);
  * checkpoint at the best validation AUC (nodule level), early stopping.

Physical nodules are found by linking reader ROIs whose bounding boxes overlap on
the same slice (union-find inside each patient). Writes the v2 manifest, the v1
predictions relabelled with the v2 labels and the v2 predictions with IDs.

  .venv\\Scripts\\python.exe scripts/sprint5_retrain_resnet50.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from sprint5_predictions import (
    ID_COLS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OUT,
    ROOT,
    SEED,
    check_split,
    run_notebook_until_split,
    set_seed,
)

MODEL_DIR = ROOT / "models" / "resnet50_v2"
V1_PREDS = {"validation": OUT / "preds_val_resnet50.csv", "test": OUT / "preds_test_resnet50.csv"}
CROP_MM = 64.0
IMAGE_SIZE = 224
HU_MIN, HU_MAX = -1000.0, 400.0
MALIGNANT_MIN_MEAN = 3.5
AMBIG_LOW, AMBIG_HIGH = 2.5, 3.5
NODULE_COLS = ["patient_id", "study_uid", "reading_session", "nodule_id"]
TAGS = {"train": "train", "validation": "val", "test": "test"}


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def reader_nodule_key(frame: pd.DataFrame) -> pd.Series:
    return frame[NODULE_COLS].astype(str).agg("|".join, axis=1)


def consensus_labels(rows: pd.DataFrame) -> pd.DataFrame:
    """rows: one per patch with rn_key, reading_session, sop_uid, bbox, malignancy."""
    uf = UnionFind()
    for key in rows["rn_key"].unique():
        uf.find(key)
    for _, group in rows.groupby("sop_uid"):
        if group["reading_session"].nunique() < 2:
            continue
        g = group[["rn_key", "reading_session", "bx0", "bx1", "by0", "by1"]].to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if g[i][1] == g[j][1] or g[i][0] == g[j][0]:
                    continue
                if g[i][2] <= g[j][3] and g[j][2] <= g[i][3] and g[i][4] <= g[j][5] and g[j][4] <= g[i][5]:
                    uf.union(g[i][0], g[j][0])

    per_rn = rows.groupby("rn_key").agg(
        malignancy=("malignancy", "mean"), reading_session=("reading_session", "first")
    ).reset_index()
    per_rn["cluster"] = per_rn["rn_key"].map(uf.find)
    per_cluster = per_rn.groupby("cluster").agg(
        consensus_malignancy=("malignancy", "mean"), n_readers=("reading_session", "nunique")
    )
    per_rn = per_rn.join(per_cluster, on="cluster")
    per_rn["physical_nodule"] = per_rn["cluster"].astype("category").cat.codes
    return per_rn.set_index("rn_key")[["physical_nodule", "consensus_malignancy", "n_readers"]]


def extract_crop(dicom_path: str, cx: float, cy: float) -> np.ndarray:
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) \
        + float(getattr(ds, "RescaleIntercept", 0))
    row_mm, col_mm = (float(v) for v in ds.PixelSpacing)
    half_x = int(round(CROP_MM / 2 / col_mm))
    half_y = int(round(CROP_MM / 2 / row_mm))
    pad = max(half_x, half_y) + 2
    padded = np.pad(image, pad, mode="constant", constant_values=HU_MIN)
    x, y = int(round(cx)) + pad, int(round(cy)) + pad
    crop = padded[y - half_y:y + half_y, x - half_x:x + half_x]
    crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    crop = (np.clip(crop, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    return crop.astype(np.float16)


class CropDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool, max_shift: int = 12):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.max_shift = max_shift

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx].astype(np.float32)
        if self.augment:
            image = np.rot90(image, np.random.randint(4))
            if np.random.rand() < 0.5:
                image = image[:, ::-1]
            dx, dy = np.random.randint(-self.max_shift, self.max_shift + 1, size=2)
            shifted = np.zeros_like(image)
            h, w = image.shape
            shifted[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
                image[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
            image = shifted
        image = (np.repeat(image[None], 3, axis=0) - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)), torch.tensor(self.labels[idx])


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for images, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def nodule_auc(frame: pd.DataFrame, probs: np.ndarray) -> float:
    nodules = frame.assign(prob=probs).groupby("rn_key").agg(label=("label", "max"), prob=("prob", "mean"))
    return roc_auc_score(nodules["label"], nodules["prob"])


def save_predictions(frame: pd.DataFrame, probs: np.ndarray, path: Path, meta: dict) -> None:
    out = frame[ID_COLS + ["label", "is_ambiguous", "consensus_malignancy", "physical_nodule"]].reset_index(drop=True)
    out.insert(0, "row_idx", np.arange(len(out)))
    out["prob_malignant"] = probs
    for key, value in meta.items():
        out[key] = value
    out.to_csv(path, index=False)
    print("Wrote", path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    ns = run_notebook_until_split()
    annotations = ns["matched_annotations"]
    splits = {
        "train": ns["train_df"].reset_index(drop=True),
        "validation": ns["val_df"].reset_index(drop=True),
        "test": ns["test_df"].reset_index(drop=True),
    }
    for frame in splits.values():
        frame.drop(columns=["image", "mask"], inplace=True, errors="ignore")
    annotations = annotations[["sop_uid", "dicom_path", "points"]].copy()
    del ns
    gc.collect()
    check_split(splits)

    for name, frame in splits.items():
        ann = annotations.loc[frame["annotation_index"]]
        if not (ann["sop_uid"].astype(str).to_numpy() == frame["sop_uid"].astype(str).to_numpy()).all():
            raise SystemExit(f"{name}: annotation_index does not point to the same slice")
        frame["dicom_path"] = ann["dicom_path"].to_numpy()
        pts = [np.asarray(p, dtype=np.float32) for p in ann["points"]]
        frame["cx"] = [p[:, 0].mean() for p in pts]
        frame["cy"] = [p[:, 1].mean() for p in pts]
        frame["bx0"] = [p[:, 0].min() for p in pts]
        frame["bx1"] = [p[:, 0].max() for p in pts]
        frame["by0"] = [p[:, 1].min() for p in pts]
        frame["by1"] = [p[:, 1].max() for p in pts]
        frame["rn_key"] = reader_nodule_key(frame)
    del annotations

    all_rows = pd.concat(splits.values(), ignore_index=True)
    consensus = consensus_labels(all_rows)
    for name, frame in splits.items():
        for col in ("physical_nodule", "consensus_malignancy", "n_readers"):
            frame[col] = frame["rn_key"].map(consensus[col])
        frame["label_v1"] = frame["label"]
        frame["label"] = (frame["consensus_malignancy"] >= MALIGNANT_MIN_MEAN).astype(int)
        frame["is_ambiguous"] = ((frame["consensus_malignancy"] > AMBIG_LOW)
                                 & (frame["consensus_malignancy"] < AMBIG_HIGH)).astype(int)
        nod = frame.drop_duplicates("rn_key")
        phys = frame.drop_duplicates("physical_nodule")
        print(f"{name}: patches={len(frame)} reader-nodules={len(nod)} physical={len(phys)} "
              f"malignant_patches={int(frame['label'].sum())} ambiguous_patches={int(frame['is_ambiguous'].sum())} "
              f"label_changed_vs_v1={int((frame['label'] != frame['label_v1']).sum())}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    v1_manifest = pd.read_csv(OUT / "split_manifest.csv", dtype=str)
    parts = []
    for name, frame in splits.items():
        part = frame[ID_COLS + ["annotation_index", "malignancy", "label", "label_v1", "is_ambiguous",
                                "consensus_malignancy", "n_readers", "physical_nodule"]].reset_index(drop=True)
        part.insert(0, "row_idx", np.arange(len(part)))
        part.insert(0, "split", name)
        ref = v1_manifest[v1_manifest["split"] == name].reset_index(drop=True)
        for col in ID_COLS:
            if not (part[col].astype(str) == ref[col]).all():
                raise SystemExit(f"{name}: row order differs from split_manifest.csv in {col}")
        parts.append(part)
    pd.concat(parts, ignore_index=True).to_csv(OUT / "split_manifest_v2.csv", index=False)
    print("Wrote", OUT / "split_manifest_v2.csv", flush=True)

    for name in ("validation", "test"):
        v1 = pd.read_csv(V1_PREDS[name], dtype={c: str for c in ID_COLS})
        frame = splits[name]
        if not (v1[ID_COLS].astype(str).to_numpy() == frame[ID_COLS].astype(str).to_numpy()).all():
            raise SystemExit(f"{V1_PREDS[name]} is not aligned with the {name} split")
        save_predictions(frame, v1["prob_malignant"].to_numpy(),
                         OUT / f"preds_{TAGS[name]}_resnet50_v1_labelv2.csv",
                         {"model": "resnet50_v1", "status": "provisorio", "labels": "v2 consensus"})

    images = {}
    for name, frame in splits.items():
        start = time.time()
        arr = np.empty((len(frame), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float16)
        for i, (path, cx, cy) in enumerate(zip(frame["dicom_path"], frame["cx"], frame["cy"])):
            arr[i] = extract_crop(path, cx, cy)
            if (i + 1) % 2000 == 0:
                print(f"  {name}: {i + 1}/{len(frame)} crops ({time.time() - start:.0f}s)", flush=True)
        images[name] = arr
        print(f"{name}: {len(frame)} crops of {CROP_MM:.0f} mm in {time.time() - start:.0f}s", flush=True)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    keep = {n: (splits[n]["is_ambiguous"] == 0).to_numpy() for n in ("train", "validation")}
    train_frame = splits["train"][keep["train"]].reset_index(drop=True)
    val_frame = splits["validation"][keep["validation"]].reset_index(drop=True)
    train_loader = DataLoader(
        CropDataset(images["train"][keep["train"]], train_frame["label"].to_numpy(), augment=True),
        batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED),
        pin_memory=True,
    )
    val_loader = DataLoader(
        CropDataset(images["validation"][keep["validation"]], val_frame["label"].to_numpy(), augment=False),
        batch_size=64, pin_memory=True,
    )
    print(f"Training on {len(train_frame)} patches (ambiguous excluded), validating on {len(val_frame)}",
          flush=True)

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_frame["label"].to_numpy())
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODEL_DIR / "best_model.pth"
    history, best_auc, best_epoch, stale = [], -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        for images_b, labels_b in train_loader:
            images_b, labels_b = images_b.to(device, non_blocking=True), labels_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(images_b), labels_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels_b)
        scheduler.step()
        val_probs = predict(model, val_loader, device)
        val_auc = nodule_auc(val_frame, val_probs)
        val_auc_patch = roc_auc_score(val_frame["label"], val_probs)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_frame),
                        "val_auc_nodule": val_auc, "val_auc_patch": val_auc_patch,
                        "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - start})
        marker = ""
        if val_auc > best_auc:
            best_auc, best_epoch, stale, marker = val_auc, epoch, 0, " BEST"
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1
        print(f"epoch {epoch}/{args.epochs} loss={history[-1]['train_loss']:.4f} "
              f"val_auc_nodule={val_auc:.4f} val_auc_patch={val_auc_patch:.4f} "
              f"({history[-1]['seconds']:.0f}s){marker}", flush=True)
        pd.DataFrame(history).to_csv(MODEL_DIR / "training_history.csv", index=False)
        if stale >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.", flush=True)
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    meta = {"model": "resnet50_v2", "seed": SEED, "best_epoch": best_epoch, "status": "provisorio",
            "labels": "v2 consensus"}
    for name in ("validation", "test"):
        loader = DataLoader(CropDataset(images[name], splits[name]["label"].to_numpy(), augment=False),
                            batch_size=64)
        save_predictions(splits[name], predict(model, loader, device),
                         OUT / f"preds_{TAGS[name]}_resnet50_v2.csv", meta)

    (MODEL_DIR / "config.json").write_text(json.dumps({
        "base": "torchvision resnet50, ImageNet weights, fc -> 2",
        "split": "559 patients (391/84/84), seed 42 (same as notebook)",
        "label": f"mean malignancy of readers on the same physical nodule >= {MALIGNANT_MIN_MEAN}",
        "physical_nodule": "reader ROIs with overlapping bounding boxes on the same slice, union-find",
        "ambiguous": f"{AMBIG_LOW} < mean < {AMBIG_HIGH}, excluded from train and validation",
        "crop": f"{CROP_MM} mm square around ROI centre, resized to {IMAGE_SIZE}, window [{HU_MIN}, {HU_MAX}] HU",
        "augmentation": "rot90, horizontal flip, shift up to 12 px",
        "loss": "cross-entropy with balanced class weights", "class_weights": weights.tolist(),
        "optimizer": "AdamW", "lr": args.lr, "weight_decay": 1e-4, "scheduler": "cosine",
        "batch_size": args.batch_size, "amp": device.type == "cuda",
        "epochs_max": args.epochs, "patience": args.patience,
        "selection": "best validation AUC at reader-nodule level", "best_epoch": best_epoch,
        "best_val_auc_nodule": best_auc, "train_patches": len(train_frame), "val_patches": len(val_frame),
        "seed": SEED, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }, indent=2), encoding="utf-8")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
