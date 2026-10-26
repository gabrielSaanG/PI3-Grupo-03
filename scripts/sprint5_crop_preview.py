"""Sprint 5: visual check of v1 (stretched bounding box) vs v2 (fixed 64 mm) crops.

Picks test cases by category (correct malignant, correct benign, v2 error,
ambiguous), one patch per physical nodule (the largest ROI), and draws the
reader contour on both crops. Labels and v2 probabilities come from the saved
manifest and predictions; nothing is retrained.

  .venv\\Scripts\\python.exe scripts/sprint5_crop_preview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

from sprint5_predictions import OUT, ROOT, SEED, run_notebook_until_split
from sprint5_retrain_resnet50 import CROP_MM, IMAGE_SIZE, extract_crop

FIG_DIR = ROOT / "docs" / "sprint5" / "figures"
CASES_PER_ROW = 3
V1_MARGIN = 20


def pick_cases(test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    test = test.copy()
    test["pred"] = (test["prob_malignant"] >= 0.5).astype(int)
    largest = (test.sort_values("num_points", ascending=False)
               .drop_duplicates("physical_nodule"))
    clear = largest[largest["is_ambiguous"] == 0]
    groups = {
        "Maligno, v2 acertou": clear[(clear["label"] == 1) & (clear["pred"] == 1)],
        "Benigno, v2 acertou": clear[(clear["label"] == 0) & (clear["pred"] == 0)],
        "Erro do v2 (FN ou FP)": clear[clear["label"] != clear["pred"]],
        "Ambíguo (fora do treino)": largest[largest["is_ambiguous"] == 1],
    }
    return {name: g.sample(min(CASES_PER_ROW, len(g)), random_state=SEED) for name, g in groups.items()}


def contour_v1(points: np.ndarray, row: pd.Series) -> np.ndarray:
    sx = IMAGE_SIZE / (row["x_max"] - row["x_min"])
    sy = IMAGE_SIZE / (row["y_max"] - row["y_min"])
    return np.c_[(points[:, 0] - row["x_min"]) * sx, (points[:, 1] - row["y_min"]) * sy]


def contour_v2(points: np.ndarray, cx: float, cy: float, spacing: tuple[float, float]) -> np.ndarray:
    row_mm, col_mm = spacing
    half_x = round(CROP_MM / 2 / col_mm)
    half_y = round(CROP_MM / 2 / row_mm)
    x0, y0 = round(cx) - half_x, round(cy) - half_y
    return np.c_[(points[:, 0] - x0) * IMAGE_SIZE / (2 * half_x),
                 (points[:, 1] - y0) * IMAGE_SIZE / (2 * half_y)]


def main() -> None:
    ns = run_notebook_until_split()
    test = ns["test_df"].reset_index(drop=True)
    annotations = ns["matched_annotations"]
    del ns

    manifest = pd.read_csv(OUT / "split_manifest_v2.csv")
    manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)
    preds = pd.read_csv(OUT / "preds_test_resnet50_v2.csv")
    if not (manifest["sop_uid"].astype(str).to_numpy() == test["sop_uid"].astype(str).to_numpy()).all():
        raise SystemExit("test rows are not aligned with split_manifest_v2.csv")
    for col in ("label", "is_ambiguous", "consensus_malignancy", "physical_nodule"):
        test[col] = manifest[col].to_numpy()
    test["prob_malignant"] = preds["prob_malignant"].to_numpy()
    test["num_points"] = annotations.loc[test["annotation_index"], "points"].map(len).to_numpy()

    chosen = pick_cases(test)
    fig, axes = plt.subplots(len(chosen), 2 * CASES_PER_ROW, figsize=(3.0 * 2 * CASES_PER_ROW, 3.5 * len(chosen)))
    rows_out = []
    for r, (group, cases) in enumerate(chosen.items()):
        for c in range(CASES_PER_ROW):
            ax_v1, ax_v2 = axes[r, 2 * c], axes[r, 2 * c + 1]
            if c >= len(cases):
                ax_v1.axis("off"), ax_v2.axis("off")
                continue
            row = cases.iloc[c]
            ann = annotations.loc[row["annotation_index"]]
            points = np.asarray(ann["points"], dtype=np.float32)
            cx, cy = points[:, 0].mean(), points[:, 1].mean()
            ds = pydicom.dcmread(ann["dicom_path"], stop_before_pixels=True)
            spacing = tuple(float(v) for v in ds.PixelSpacing)
            size_mm = max((points[:, 0].max() - points[:, 0].min()) * spacing[1],
                          (points[:, 1].max() - points[:, 1].min()) * spacing[0])
            box_mm = (row["x_max"] - row["x_min"]) * spacing[1]

            v1 = np.clip(np.asarray(row["image"], dtype=np.float32), 0, 1)
            v2 = extract_crop(ann["dicom_path"], cx, cy).astype(np.float32)
            for ax, img, contour, title in (
                (ax_v1, v1, contour_v1(points, row), f"v1: caixa de {box_mm:.0f} mm"),
                (ax_v2, v2, contour_v2(points, cx, cy, spacing), f"v2: {CROP_MM:.0f} mm"),
            ):
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.plot(*np.vstack([contour, contour[:1]]).T, color="lime", lw=1.0)
                ax.set_title(title, fontsize=9)
                ax.set_xticks([]), ax.set_yticks([])
            verdict = "maligno" if row["label"] == 1 else "benigno"
            if row["is_ambiguous"]:
                verdict = "ambíguo"
            ax_v1.set_xlabel(f"{row['patient_id']} | nódulo ~{size_mm:.0f} mm", fontsize=8)
            ax_v2.set_xlabel(f"média {row['consensus_malignancy']:.2f} ({verdict}) | p={row['prob_malignant']:.2f}",
                             fontsize=8)
            rows_out.append({"grupo": group, "patient_id": row["patient_id"], "sop_uid": row["sop_uid"],
                             "reading_session": row["reading_session"], "nodule_id": row["nodule_id"],
                             "physical_nodule": int(row["physical_nodule"]), "tamanho_mm": round(size_mm, 1),
                             "caixa_v1_mm": round(box_mm, 1), "media_malignidade": row["consensus_malignancy"],
                             "rotulo_v2": int(row["label"]), "ambiguo": int(row["is_ambiguous"]),
                             "prob_v2": round(float(row["prob_malignant"]), 3)})
        axes[r, 0].set_ylabel(group, fontsize=10)

    fig.suptitle("Recortes do teste: v1 (caixa da ROI esticada para 224 px) x v2 (64 mm fixos); "
                 "contorno verde = ROI do leitor", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "recortes_v1_v2.png"
    fig.savefig(fig_path, dpi=110)
    cases_path = FIG_DIR / "recortes_v1_v2_casos.csv"
    pd.DataFrame(rows_out).to_csv(cases_path, index=False)
    print("Wrote", fig_path)
    print("Wrote", cases_path)
    print(pd.DataFrame(rows_out).to_string(index=False))


if __name__ == "__main__":
    main()
