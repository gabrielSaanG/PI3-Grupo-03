"""Sprint 6: Adebayo sanity checks on ResNet50 v2 (provisional, single seed).

1. Cascading weight randomization: from fc down to conv1, each stage re-initialises
   one more block of the trained v2 (reset_parameters, fixed seed). On the 275 test
   physical nodules of Sprint 5 (largest ROI), Grad-CAM and IG are recomputed for the
   class predicted by the original v2 and compared with the original maps (Spearman)
   and with the reader mask (Pointing Game, IoU at the validation threshold).
2. Label randomization: the consensus labels of train and validation are permuted
   across physical nodules (every patch of a nodule keeps the same drawn label, class
   balance is kept). A new ResNet50 is trained with the v2 recipe for a fixed number
   of epochs (no selection on the random labels); its test predictions are scored
   against the true labels and its maps are compared with the v2 maps.

  .venv\\Scripts\\python.exe scripts/sprint6_v2_adebayo.py
  .venv\\Scripts\\python.exe scripts/sprint6_v2_adebayo.py --skip-label
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import models

from sprint5_predictions import set_seed
from sprint5_retrain_resnet50 import CropDataset, extract_crop, predict, save_predictions
from sprint6_v2_common import (
    FIG_DIR, IMAGE_SIZE, OUT, ROOT, SEED, TAGS, V2_CKPT, Explainer, check_against_sprint5, ci_auc, ci_mean,
    crop_ct, crop_mask, fmt_ci, load_frames, load_resnet, localisation, one_patch_per_nodule, read_slice,
    validation_thresholds,
)

CASCADE = [("fc", ["fc"]), ("layer4", ["layer4"]), ("layer3", ["layer3"]), ("layer2", ["layer2"]),
           ("layer1", ["layer1"]), ("conv1", ["conv1", "bn1"])]
LABELRAND_DIR = ROOT / "models" / "resnet50_v2_labelrand"
MAPS = ["gradcam", "ig"]
EXAMPLE_STAGES = ["original", "fc", "layer4", "layer2", "conv1", "rótulos aleatórios"]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def prepare(cases: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    items, keep = [], []
    for i, row in enumerate(cases.itertuples(index=False)):
        hu, spacing = read_slice(row.dicom_path)
        mask = crop_mask(row.points, row.cx, row.cy, spacing)
        if mask.sum() == 0:
            continue
        items.append((crop_ct(hu, row.cx, row.cy, spacing), mask))
        keep.append(i)
    return cases.iloc[keep].reset_index(drop=True), items


def stage_records(stage: str, blocks: str, cases, items, maps_list, probs, originals, thresholds) -> list[dict]:
    rows = []
    for i, ((crop, mask), maps) in enumerate(zip(items, maps_list)):
        row = cases.iloc[i]
        rec = {"stage": stage, "randomized_blocks": blocks, "patient_id": row["patient_id"],
               "sop_uid": row["sop_uid"], "reading_session": row["reading_session"], "nodule_id": row["nodule_id"],
               "physical_nodule": int(row["physical_nodule"]), "label": int(row["label"]),
               "is_ambiguous": int(row["is_ambiguous"]), "prob": float(probs[i])}
        rec.update(localisation(maps, mask, thresholds))
        for name in MAPS:
            rec[f"{name}_empty"] = int(maps[name].max() == 0)
            rec[f"{name}_spearman_vs_v2"] = spearman(maps[name], originals[i][name])
            rec[f"{name}_spearman_vs_image"] = spearman(maps[name], crop)
        rows.append(rec)
    return rows


def weight_cascade(model_v2, cases, items, thresholds, device, example_idx):
    base = Explainer(model_v2, device)
    probs = base.probs([c for c, _ in items])
    targets = (probs >= 0.5).astype(int)
    start = time.time()
    originals = [base.maps(crop, int(t)) for (crop, _), t in zip(items, targets)]
    base.close()
    print(f"original v2 maps: {len(items)} cases in {time.time() - start:.0f}s", flush=True)
    records = stage_records("original", "", cases, items, originals, probs, originals, thresholds)
    examples = {"original": [originals[i] for i in example_idx]}

    model_r = copy.deepcopy(model_v2)
    explainer = Explainer(model_r, device)
    done = []
    for k, (stage, blocks) in enumerate(CASCADE, start=1):
        torch.manual_seed(SEED + k)
        for block in blocks:
            for module in getattr(model_r, block).modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        model_r.eval()
        done += blocks
        start = time.time()
        stage_probs = explainer.probs([c for c, _ in items])
        maps_list = [explainer.maps(crop, int(t)) for (crop, _), t in zip(items, targets)]
        records += stage_records(stage, ", ".join(done), cases, items, maps_list, stage_probs, originals, thresholds)
        examples[stage] = [maps_list[i] for i in example_idx]
        print(f"stage {stage} (random: {', '.join(done)}): {time.time() - start:.0f}s", flush=True)
    explainer.close()
    del model_r
    torch.cuda.empty_cache()
    return records, targets, originals, examples


def build_crops(frame: pd.DataFrame, name: str) -> np.ndarray:
    start = time.time()
    arr = np.empty((len(frame), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float16)
    for i, (path, cx, cy) in enumerate(zip(frame["dicom_path"], frame["cx"], frame["cy"])):
        arr[i] = extract_crop(path, cx, cy)
    print(f"{name}: {len(frame)} crops in {time.time() - start:.0f}s", flush=True)
    return arr


def permute_by_nodule(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    nodules = frame.drop_duplicates("physical_nodule")
    drawn = dict(zip(nodules["physical_nodule"], rng.permutation(nodules["label"].to_numpy())))
    return frame["physical_nodule"].map(drawn).to_numpy().astype(int)


def train_label_randomized(frames, args, device):
    rng = np.random.default_rng(SEED)
    data = {}
    for name in ("train", "validation"):
        clear = frames[name][frames[name]["is_ambiguous"] == 0].reset_index(drop=True)
        clear["label_random"] = permute_by_nodule(clear, rng)
        nod = clear.drop_duplicates("physical_nodule")
        print(f"{name}: {len(clear)} patches, {len(nod)} physical nodules, malignant nodules "
              f"{int(nod['label'].sum())} (random {int(nod['label_random'].sum())}), "
              f"random label = true label for {(nod['label'] == nod['label_random']).mean():.1%} of nodules",
              flush=True)
        data[name] = (clear, build_crops(clear, name))

    set_seed()
    train_frame, train_crops = data["train"]
    val_frame, val_crops = data["validation"]
    train_loader = DataLoader(CropDataset(train_crops, train_frame["label_random"].to_numpy(), augment=True),
                              batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED), pin_memory=True)
    val_loader = DataLoader(CropDataset(val_crops, val_frame["label_random"].to_numpy(), augment=False),
                            batch_size=64, pin_memory=True)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_frame["label_random"].to_numpy())
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.label_epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    for epoch in range(1, args.label_epochs + 1):
        model.train()
        start, total_loss, correct = time.time(), 0.0, 0
        for images_b, labels_b in train_loader:
            images_b, labels_b = images_b.to(device, non_blocking=True), labels_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images_b)
                loss = criterion(logits, labels_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels_b)
            correct += int((logits.argmax(1) == labels_b).sum())
        scheduler.step()
        val_probs = predict(model, val_loader, device)
        history.append({
            "epoch": epoch, "train_loss": total_loss / len(train_frame),
            "train_acc_random_labels": correct / len(train_frame),
            "val_auc_patch_random_labels": roc_auc_score(val_frame["label_random"], val_probs),
            "val_auc_patch_true_labels": roc_auc_score(val_frame["label"], val_probs),
            "seconds": time.time() - start,
        })
        h = history[-1]
        print(f"label-rand epoch {epoch}/{args.label_epochs} loss={h['train_loss']:.4f} "
              f"train_acc(random)={h['train_acc_random_labels']:.3f} val_auc(random)={h['val_auc_patch_random_labels']:.3f} "
              f"val_auc(true)={h['val_auc_patch_true_labels']:.3f} ({h['seconds']:.0f}s)", flush=True)

    LABELRAND_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LABELRAND_DIR / "final_model.pth")
    pd.DataFrame(history).to_csv(LABELRAND_DIR / "training_history.csv", index=False)
    (LABELRAND_DIR / "config.json").write_text(json.dumps({
        "purpose": "Adebayo label randomization for resnet50_v2 (provisional)",
        "labels": "v2 consensus labels permuted across physical nodules inside train and inside validation "
                  "(ambiguous excluded, as in v2); all patches of a nodule share the drawn label",
        "permutation_seed": SEED, "recipe": "same as models/resnet50_v2 (crop 64 mm, augmentation, "
                                            "weighted CE, AdamW, cosine)",
        "epochs": args.label_epochs, "selection": "none (last epoch)", "lr": args.lr, "batch_size": args.batch_size,
        "train_patches": len(train_frame), "val_patches": len(val_frame),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }, indent=2), encoding="utf-8")
    del data, train_loader, val_loader, train_crops, val_crops
    gc.collect()
    return model.eval()


def summarise(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    original_pred = records.loc[records["stage"] == "original", "prob"].to_numpy() >= 0.5
    for stage, part in records.groupby("stage", sort=False):
        clear = part[part["is_ambiguous"] == 0]
        auc = roc_auc_score(clear["label"], clear["prob"])
        flips = ((part["prob"].to_numpy() >= 0.5) != original_pred).mean()
        row = {"stage": stage, "randomized_blocks": part["randomized_blocks"].iloc[0], "n_casos": len(part),
               "n_pacientes": part["patient_id"].nunique(),
               "AUC (nódulos não ambíguos)": fmt_ci(auc, ci_auc(clear, "prob")),
               "classe prevista diferente do v2": round(float(flips), 3)}
        for name, label in (("gradcam", "Grad-CAM"), ("ig", "IG")):
            rho = part[f"{name}_spearman_vs_v2"]
            row[f"{label} Spearman vs v2 (mediana [IQR])"] = (
                f"{rho.median():.3f} [{rho.quantile(0.25):.3f}; {rho.quantile(0.75):.3f}]" if rho.notna().any() else "")
            row[f"{label} mapas vazios"] = round(float(part[f"{name}_empty"].mean()), 3)
            row[f"{label} Pointing Game"] = fmt_ci(part[f"{name}_pg_hit"].mean(), ci_mean(part, f"{name}_pg_hit"))
            row[f"{label} IoU"] = round(float(part[f"{name}_iou"].mean()), 3)
            row[f"{label} Spearman vs imagem (mediana)"] = round(float(part[f"{name}_spearman_vs_image"].median()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def plot(records: pd.DataFrame, examples: dict, items, cases, example_idx) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stages = list(dict.fromkeys(records["stage"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(stages))
    for name, label, color in (("gradcam", "Grad-CAM", "tab:red"), ("ig", "Integrated Gradients", "tab:blue")):
        g = records.groupby("stage", sort=False)
        med = g[f"{name}_spearman_vs_v2"].median().reindex(stages)
        q1 = g[f"{name}_spearman_vs_v2"].quantile(0.25).reindex(stages)
        q3 = g[f"{name}_spearman_vs_v2"].quantile(0.75).reindex(stages)
        ax[0].plot(x, med, "o-", color=color, label=label)
        ax[0].fill_between(x, q1, q3, color=color, alpha=0.15)
        ax[1].plot(x, g[f"{name}_pg_hit"].mean().reindex(stages) * 100, "o-", color=color, label=label)
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_title("Spearman entre o mapa e o mapa do v2 (mediana e IQR)")
    ax[1].axhline(100, color="green", ls="--", lw=1, label="Centrado (não aprende nada)")
    ax[1].set_title("Pointing Game no teste (%)")
    labels = ["v2" if s == "original" else ("rótulos\naleatórios" if s == "rótulos aleatórios" else f"até\n{s}")
              for s in stages]
    for a in ax:
        a.set_xticks(x, labels, fontsize=8)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("ResNet50 v2 (provisório): teste de sanidade de Adebayo — randomização em cascata e de rótulos",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "adebayo_v2_curvas.png", dpi=110)
    plt.close(fig)

    shown = [s for s in EXAMPLE_STAGES if s in examples]
    cols = 1 + 2 * len(shown)
    fig, axes = plt.subplots(len(example_idx), cols, figsize=(1.9 * cols, 2.1 * len(example_idx)))
    for r, idx in enumerate(example_idx):
        crop, mask = items[idx]
        row = cases.iloc[idx]
        axes[r, 0].imshow(crop, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].contour(mask, levels=[0.5], colors="lime", linewidths=0.8)
        axes[r, 0].set_ylabel(f"{row['patient_id']}\n{'maligno' if row['label'] else 'benigno'}"
                              f"{' (amb.)' if row['is_ambiguous'] else ''}", fontsize=7)
        axes[r, 0].set_title("TC + ROI" if r == 0 else "", fontsize=8)
        for k, stage in enumerate(shown):
            for j, (name, label) in enumerate((("gradcam", "Grad-CAM"), ("ig", "IG"))):
                a = axes[r, 1 + 2 * k + j]
                a.imshow(crop, cmap="gray", vmin=0, vmax=1)
                a.imshow(examples[stage][r][name], cmap="jet", alpha=0.45, vmin=0, vmax=1)
                a.contour(mask, levels=[0.5], colors="lime", linewidths=0.6)
                if r == 0:
                    a.set_title(f"{label}\n{'v2' if stage == 'original' else stage}", fontsize=7)
    for a in axes.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle("Mapas do v2 e dos modelos aleatorizados (verde = ROI do leitor)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "adebayo_v2_exemplos.png", dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-label", action="store_true", help="only the weight cascade (no training)")
    parser.add_argument("--label-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    names = ("validation", "test") if args.skip_label else ("train", "validation", "test")
    frames = load_frames(names)
    cases = one_patch_per_nodule(frames["test"])
    check_against_sprint5(cases)
    cases, items = prepare(cases)
    thresholds = validation_thresholds()
    print("Map thresholds (best validation IoU, Sprint 5):", thresholds, flush=True)
    rng = np.random.default_rng(SEED)
    clear_idx = np.flatnonzero(cases["is_ambiguous"].to_numpy() == 0)
    example_idx = sorted(rng.choice(clear_idx, size=3, replace=False).tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v2 = load_resnet(V2_CKPT, device)
    records, targets, originals, examples = weight_cascade(model_v2, cases, items, thresholds, device, example_idx)
    del model_v2
    torch.cuda.empty_cache()

    if not args.skip_label:
        model_lr = train_label_randomized(frames, args, device)
        for name in ("validation", "test"):
            frame = frames[name]
            loader = DataLoader(CropDataset(build_crops(frame, name), frame["label"].to_numpy(), augment=False),
                                batch_size=64)
            save_predictions(frame, predict(model_lr, loader, device),
                             OUT / f"preds_{TAGS[name]}_resnet50_v2_labelrand.csv",
                             {"model": "resnet50_v2_labelrand", "seed": SEED, "status": "provisorio",
                              "labels": "v2 consensus (true); model trained on permuted labels"})
            del loader
            gc.collect()
        explainer = Explainer(model_lr, device)
        probs = explainer.probs([c for c, _ in items])
        maps_list = [explainer.maps(crop, int(t)) for (crop, _), t in zip(items, targets)]
        explainer.close()
        records += stage_records("rótulos aleatórios", "treino com rótulos permutados por nódulo físico",
                                 cases, items, maps_list, probs, originals, thresholds)
        examples["rótulos aleatórios"] = [maps_list[i] for i in example_idx]

    records = pd.DataFrame(records)
    records.to_csv(OUT / "adebayo_v2_per_case.csv", index=False)
    summary = summarise(records)
    summary.to_csv(OUT / "adebayo_v2_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summary.T.to_string(), flush=True)
    plot(records, examples, items, cases, example_idx)
    print("Wrote", OUT / "adebayo_v2_per_case.csv", OUT / "adebayo_v2_summary.csv")
    print("Wrote", FIG_DIR / "adebayo_v2_curvas.png", FIG_DIR / "adebayo_v2_exemplos.png")


if __name__ == "__main__":
    main()
