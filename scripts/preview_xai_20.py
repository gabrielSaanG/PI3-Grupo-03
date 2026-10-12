"""Build 20 Grad-CAM / Integrated Gradients panels from the saved ResNet50.

Reuses the notebook cells up to Grad-CAM setup, skips training, and does not
run the full-test explanation loops.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import torch
from captum.attr import IntegratedGradients

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "03-resnet50-sprint3.ipynb"
OUT_DIR = ROOT / "outputs" / "sprint3" / "preview20"
STOP_CELL_ID = "733fc2c0"
N_CASES = 20
IG_STEPS = 32


def main() -> None:
    os.chdir(ROOT / "notebooks")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns: dict = {"__name__": "__main__"}
    plt.show = lambda *args, **kwargs: None

    def display(*args, **kwargs):
        return None

    ns["display"] = display

    for index, cell in enumerate(nb.cells):
        if cell.get("id") == STOP_CELL_ID:
            print(f"Stopping before visual-test cell {index}")
            break
        if cell.cell_type != "code":
            continue
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if "for epoch in range" in source:
            print(f"Skipping training cell {index}; loading checkpoint")
            loader = (
                "model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))\n"
                "model.eval()\n"
                "print('Checkpoint carregado:', CHECKPOINT_PATH)\n"
            )
            exec(loader, ns, ns)
            continue
        print(f"Executing cell {index}")
        try:
            exec(source, ns, ns)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)

    test_df = ns["test_df"].reset_index(drop=True)
    model = ns["model"]
    gradcam = ns["gradcam"]
    device = ns["device"]
    model.eval()

    rng = np.random.default_rng(42)
    indices = rng.choice(len(test_df), size=min(N_CASES, len(test_df)), replace=False)
    indices = np.sort(indices)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def prepare(image_np: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.repeat(1, 3, 1, 1).to(device)
        return (tensor - mean) / std

    ig = IntegratedGradients(model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    for order, idx in enumerate(indices, start=1):
        row = test_df.iloc[int(idx)]
        image_np = np.asarray(row["image"], dtype=np.float32)
        mask_np = np.asarray(row["mask"], dtype=np.uint8)
        true_label = int(row["label"])
        image_model = prepare(image_np)

        with torch.no_grad():
            logits = model(image_model)
            predicted = int(torch.argmax(logits, dim=1).item())

        cam, _ = gradcam.generate(image_model)
        cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_rgb = np.stack([image_np] * 3, axis=-1)
        overlay = np.clip(0.5 * image_rgb + 0.5 * heatmap, 0, 1)

        baseline = torch.zeros_like(image_model)
        attributions = ig.attribute(image_model, baselines=baseline, target=predicted, n_steps=IG_STEPS)
        attr = attributions.detach().abs().sum(dim=1).squeeze(0).cpu().numpy()
        if attr.max() > 0:
            attr = attr / attr.max()
        attr_resized = cv2.resize(attr, (image_np.shape[1], image_np.shape[0]))
        ig_heat = cv2.applyColorMap(np.uint8(attr_resized * 255), cv2.COLORMAP_JET)
        ig_heat = cv2.cvtColor(ig_heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ig_overlay = np.clip(0.5 * image_rgb + 0.5 * ig_heat, 0, 1)

        gt = image_rgb.copy()
        gt[mask_np > 0] = [1.0, 0.0, 0.0]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = [
            (image_np, "CT", "gray"),
            (gt, "Ground truth", None),
            (overlay, "Grad-CAM", None),
            (ig_overlay, "Integrated Gradients", None),
        ]
        for ax, (img, title, cmap) in zip(axes, panels):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(
            f"Caso {order}  idx={int(idx)}  real={true_label}  predito={predicted}"
        )
        fig.tight_layout()
        path = OUT_DIR / f"caso_{order:02d}_idx{int(idx)}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)
        print(f"Saved {path.name}")

    sheet_cols = 4
    sheet_rows = int(np.ceil(len(saved) / sheet_cols))
    sheet, axes = plt.subplots(sheet_rows, sheet_cols, figsize=(16, 4 * sheet_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, path in zip(axes, saved):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem, fontsize=8)
        ax.axis("off")
    sheet.tight_layout()
    sheet_path = OUT_DIR / "preview20_sheet.png"
    sheet.savefig(sheet_path, dpi=100)
    plt.close(sheet)
    print("SHEET", sheet_path)


if __name__ == "__main__":
    main()
