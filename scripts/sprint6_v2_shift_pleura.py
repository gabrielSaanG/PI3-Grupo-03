"""Sprint 6: shift test and pleura / chest-wall context check on ResNet50 v2 (provisional).

Same 275 test physical nodules (largest ROI) and map thresholds as Sprint 5.

1. Shift test: the 64 mm crop centre is moved 8 and 16 mm right, left, down and up,
   so the nodule is no longer at the centre. Maps that follow the nodule keep their
   Pointing Game; the centred map (Gaussian at the crop centre) loses it. The v2
   probability is re-scored too (AUC on non-ambiguous nodules, mean |delta p|).
2. Pleura: lungs are segmented on the full slice (sprint6_v2_common.lung_mask). Each
   crop is split into nodule (reader ROI dilated by the Pointing Game tolerance), lung
   and outside the lung (pleura, chest wall, mediastinum). Per map: where the peak
   falls and the share of map mass in each region against the region area.
   Juxtapleural = reader ROI within 2 mm of the outside-lung region.
3. Occlusion: the v2 is re-scored after filling with lung-like tissue (-850 HU)
   (a) everything outside the lung, (b) the nodule, (c) everything but the nodule.

  .venv\\Scripts\\python.exe scripts/sprint6_v2_shift_pleura.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from sprint5_retrain_resnet50 import HU_MAX, HU_MIN
from sprint6_v2_common import (
    FIG_DIR, IMAGE_SIZE, MM_PER_PX, OUT, PG_TOLERANCE_PX, SEED, V2_CKPT, Explainer, center_map,
    check_against_sprint5, ci_auc, ci_mean, crop_ct, crop_geometry, crop_mask, fmt_ci, load_frames, load_resnet,
    localisation, lung_mask, normalise, one_patch_per_nodule, read_slice, validation_thresholds,
)

SHIFTS_MM = [8.0, 16.0]
DIRECTIONS = {"direita": (1, 0), "esquerda": (-1, 0), "baixo": (0, 1), "cima": (0, -1)}
JUXTAPLEURAL_MM = 2.0
FILL = (-850.0 - HU_MIN) / (HU_MAX - HU_MIN)
OCCLUSION_MARGIN_PX = 7
REGIONS = ["nodulo", "pulmao", "fora_pulmao", "fora_imagem"]
REGION_NAMES = {"nodulo": "nódulo (±4,3 mm)", "pulmao": "pulmão", "fora_pulmao": "fora do pulmão",
                "fora_imagem": "fora da imagem"}
OCCLUSIONS = {"sem_fora_pulmao": "sem o que está fora do pulmão", "sem_nodulo": "sem o nódulo",
              "so_nodulo": "só o nódulo"}
MAP_LABELS = {"gradcam": "Grad-CAM", "ig": "Integrated Gradients", "random": "Aleatório", "center": "Centrado"}


def regions_for(crop_lung: np.ndarray, crop_inside: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    zone = cv2.dilate(mask, np.ones((2 * PG_TOLERANCE_PX + 1,) * 2, np.uint8)) > 0
    return {"nodulo": zone, "pulmao": crop_lung & ~zone & crop_inside,
            "fora_pulmao": ~crop_lung & ~zone & crop_inside, "fora_imagem": ~crop_inside & ~zone}


def occluded(crop, mask, crop_lung, crop_inside) -> dict[str, np.ndarray]:
    keep = cv2.dilate(mask, np.ones((2 * OCCLUSION_MARGIN_PX + 1,) * 2, np.uint8)) > 0
    return {"sem_fora_pulmao": np.where(~crop_lung & crop_inside & ~keep, FILL, crop).astype(np.float32),
            "sem_nodulo": np.where(keep, FILL, crop).astype(np.float32),
            "so_nodulo": np.where(keep, crop, FILL).astype(np.float32)}


def pleura_record(maps, regions, mask, crop_lung, crop_inside) -> dict:
    rec = {f"area_{r}": float(regions[r].mean()) for r in REGIONS}
    outside = ~crop_lung & crop_inside
    if outside.any():
        dist = ndimage.distance_transform_edt(~outside) * MM_PER_PX
        rec["pleura_dist_mm"] = float(dist[mask > 0].min())
    else:
        rec["pleura_dist_mm"] = np.nan
    rec["juxtapleural"] = int(rec["pleura_dist_mm"] <= JUXTAPLEURAL_MM) if not np.isnan(rec["pleura_dist_mm"]) else 0
    for name, m in maps.items():
        py, px = np.unravel_index(np.argmax(m), m.shape)
        rec[f"{name}_peak_region"] = next(r for r in REGIONS if regions[r][py, px])
        total = float(m.sum()) or 1.0
        for r in REGIONS:
            rec[f"{name}_mass_{r}"] = float(m[regions[r]].sum()) / total
    return rec


def run(cases, explainer, thresholds):
    rng = np.random.default_rng(SEED)
    center = center_map()
    clear_idx = np.flatnonzero(cases["is_ambiguous"].to_numpy() == 0)
    shift_examples = set(rng.choice(clear_idx, size=3, replace=False).tolist())
    configs = [(0.0, "centro")] + [(d, k) for d in SHIFTS_MM for k in DIRECTIONS]
    shift_rows, pleura_rows, stored, examples = [], [], {}, {}
    start = time.time()
    for i, row in enumerate(cases.itertuples(index=False)):
        hu, spacing = read_slice(row.dicom_path)
        lung_full = lung_mask(hu, spacing).astype(np.uint8)
        ids = {"case": i, "patient_id": row.patient_id, "sop_uid": row.sop_uid,
               "reading_session": row.reading_session, "nodule_id": row.nodule_id,
               "physical_nodule": int(row.physical_nodule), "label": int(row.label),
               "is_ambiguous": int(row.is_ambiguous), "prob_v2_sprint5": float(row.prob_v2)}
        for shift_mm, direction in configs:
            dx, dy = DIRECTIONS.get(direction, (0, 0))
            ccx, ccy = row.cx + dx * shift_mm / spacing[1], row.cy + dy * shift_mm / spacing[0]
            crop = crop_ct(hu, ccx, ccy, spacing)
            mask = crop_mask(row.points, ccx, ccy, spacing)
            if mask.sum() == 0:
                continue
            prob = float(explainer.probs([crop])[0])
            maps = explainer.maps(crop, int(prob >= 0.5))
            maps["center"] = center
            shift_rows.append({**ids, "shift_mm": shift_mm, "direction": direction, "prob": prob,
                               **localisation(maps, mask, thresholds)})
            if i in shift_examples and direction in ("centro", "direita"):
                examples[(i, shift_mm)] = (crop, mask, maps["gradcam"], maps["ig"], prob)
            if shift_mm:
                continue
            maps["random"] = normalise(gaussian_filter(rng.random((IMAGE_SIZE, IMAGE_SIZE)), sigma=8))
            crop_lung = crop_geometry(lung_full, ccx, ccy, spacing, 0, cv2.INTER_NEAREST) > 0
            crop_inside = crop_geometry(np.ones_like(lung_full), ccx, ccy, spacing, 0, cv2.INTER_NEAREST) > 0
            regions = regions_for(crop_lung, crop_inside, mask)
            occ = occluded(crop, mask, crop_lung, crop_inside)
            occ_probs = explainer.probs(list(occ.values()))
            pleura_rows.append({**ids, "prob": prob, "pred": int(prob >= 0.5),
                                **pleura_record(maps, regions, mask, crop_lung, crop_inside),
                                **{f"prob_{k}": float(p) for k, p in zip(occ, occ_probs)}})
            stored[i] = (crop.astype(np.float16), mask, crop_lung, crop_inside, maps["gradcam"].astype(np.float16))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(cases)} cases ({time.time() - start:.0f}s)", flush=True)
    return pd.DataFrame(shift_rows), pd.DataFrame(pleura_rows), stored, examples


def shift_summary(shift: pd.DataFrame) -> pd.DataFrame:
    base = shift[shift["shift_mm"] == 0].set_index("case")["prob"]
    shift = shift.assign(delta_p=(shift["prob"] - shift["case"].map(base)).abs(),
                         flip=((shift["prob"] >= 0.5) != (shift["case"].map(base) >= 0.5)).astype(int))
    rows = []
    for d, part in shift.groupby("shift_mm"):
        clear = part[part["is_ambiguous"] == 0]
        row = {"deslocamento_mm": d, "n_linhas": len(part), "n_nodulos": part["case"].nunique(),
               "n_pacientes": part["patient_id"].nunique(),
               "AUC (não ambíguos)": fmt_ci(roc_auc_score(clear["label"], clear["prob"]), ci_auc(clear, "prob")),
               "|Δp| médio vs 0 mm": round(float(part["delta_p"].mean()), 3),
               "classe trocada vs 0 mm": round(float(part["flip"].mean()), 3)}
        for name in ("gradcam", "ig", "center"):
            label = MAP_LABELS[name]
            row[f"{label} Pointing Game"] = fmt_ci(part[f"{name}_pg_hit"].mean(), ci_mean(part, f"{name}_pg_hit"))
            row[f"{label} IoU"] = round(float(part[f"{name}_iou"].mean()), 3)
            row[f"{label} pico ao nódulo (mm)"] = round(float(part[f"{name}_peak_dist_mm"].mean()), 1)
            row[f"{label} pico ao centro do recorte (mm)"] = round(float(part[f"{name}_peak_centre_mm"].mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def peak_region_summary(pleura: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, label in MAP_LABELS.items():
        row = {"mapa": label, "n_casos": len(pleura)}
        for r in REGIONS:
            row[f"pico em {REGION_NAMES[r]} (%)"] = round(100 * float((pleura[f"{name}_peak_region"] == r).mean()), 1)
        for r in ("nodulo", "pulmao", "fora_pulmao"):
            mass, area = pleura[f"{name}_mass_{r}"].mean(), pleura[f"area_{r}"].mean()
            row[f"massa em {REGION_NAMES[r]} (%)"] = round(100 * float(mass), 1)
            row[f"área de {REGION_NAMES[r]} (%)"] = round(100 * float(area), 1)
            row[f"enriquecimento {REGION_NAMES[r]} (massa/área)"] = round(float(mass / area), 2) if area else np.nan
        rows.append(row)
    pleura = pleura.assign(gradcam_peak_outside=(pleura["gradcam_peak_region"] == "fora_pulmao").astype(int))
    row = {"mapa": "Grad-CAM, pico fora do pulmão (IC 95% por paciente)", "n_casos": len(pleura),
           "pico em fora do pulmão (%)": fmt_ci(100 * pleura["gradcam_peak_outside"].mean(),
                                                tuple(100 * v for v in ci_mean(pleura, "gradcam_peak_outside")), 1)}
    rows.append(row)
    return pd.DataFrame(rows)


def context_summary(pleura: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pleura.assign(
        grupo=np.where(pleura["is_ambiguous"] == 1, "ambíguo",
                       np.where(pleura["pred"] == pleura["label"], "acerto", "erro")),
        gradcam_peak_outside=(pleura["gradcam_peak_region"] == "fora_pulmao").astype(int),
        ig_peak_outside=(pleura["ig_peak_region"] == "fora_pulmao").astype(int),
    )
    rows = []

    def add(group: str, part: pd.DataFrame) -> None:
        clear = part[part["is_ambiguous"] == 0]
        benign, malignant = clear[clear["label"] == 0], clear[clear["label"] == 1]
        rows.append({
            "grupo": group, "n": len(part), "n_não_ambíguos": len(clear),
            "malignos entre não ambíguos (%)": round(100 * clear["label"].mean(), 1) if len(clear) else np.nan,
            "Grad-CAM pico fora do pulmão (%)": round(100 * part["gradcam_peak_outside"].mean(), 1),
            "IG pico fora do pulmão (%)": round(100 * part["ig_peak_outside"].mean(), 1),
            "Grad-CAM massa fora do pulmão (%)": round(100 * part["gradcam_mass_fora_pulmao"].mean(), 1),
            "área fora do pulmão no recorte (%)": round(100 * part["area_fora_pulmao"].mean(), 1),
            "AUC do v2": (round(roc_auc_score(clear["label"], clear["prob"]), 3)
                          if clear["label"].nunique() == 2 else np.nan),
            "benignos marcados malignos (%)": round(100 * (benign["pred"] == 1).mean(), 1) if len(benign) else np.nan,
            "malignos perdidos (%)": round(100 * (malignant["pred"] == 0).mean(), 1) if len(malignant) else np.nan,
            "p médio benignos": round(benign["prob"].mean(), 3) if len(benign) else np.nan,
            "p médio malignos": round(malignant["prob"].mean(), 3) if len(malignant) else np.nan,
        })

    add("todos", p)
    for jp, part in p.groupby("juxtapleural"):
        add("justapleural (≤ 2 mm)" if jp else "longe da pleura (> 2 mm)", part)
    for g, part in p.groupby("grupo"):
        add(f"v2: {g}", part)
    for lab, part in p[p["is_ambiguous"] == 0].groupby("label"):
        add("maligno" if lab else "benigno", part)
    table = pd.DataFrame(rows)

    clear = p[p["is_ambiguous"] == 0]
    shortcut = pd.DataFrame([
        {"preditor (sem olhar o nódulo)": "proximidade da pleura (−distância em mm)",
         "AUC": fmt_ci(roc_auc_score(clear["label"], -clear["pleura_dist_mm"].fillna(99)),
                       ci_auc(clear.assign(s=-clear["pleura_dist_mm"].fillna(99)), "s"))},
        {"preditor (sem olhar o nódulo)": "área fora do pulmão no recorte",
         "AUC": fmt_ci(roc_auc_score(clear["label"], clear["area_fora_pulmao"]), ci_auc(clear, "area_fora_pulmao"))},
        {"preditor (sem olhar o nódulo)": "justapleural (sim/não)",
         "AUC": fmt_ci(roc_auc_score(clear["label"], clear["juxtapleural"]), ci_auc(clear, "juxtapleural"))},
        {"preditor (sem olhar o nódulo)": "massa do Grad-CAM fora do pulmão",
         "AUC": fmt_ci(roc_auc_score(clear["label"], clear["gradcam_mass_fora_pulmao"]),
                       ci_auc(clear, "gradcam_mass_fora_pulmao"))},
    ])
    return table, shortcut


def occlusion_summary(pleura: pd.DataFrame) -> pd.DataFrame:
    clear = pleura[pleura["is_ambiguous"] == 0]
    rows = []
    for key, label in [("prob", "original")] + [(f"prob_{k}", v) for k, v in OCCLUSIONS.items()]:
        row = {"condição": label, "n_não_ambíguos": len(clear), "n_pacientes": clear["patient_id"].nunique(),
               "AUC": fmt_ci(roc_auc_score(clear["label"], clear[key]), ci_auc(clear, key)),
               "p médio malignos": round(clear.loc[clear["label"] == 1, key].mean(), 3),
               "p médio benignos": round(clear.loc[clear["label"] == 0, key].mean(), 3),
               "|Δp| médio vs original": round((clear[key] - clear["prob"]).abs().mean(), 3),
               "classe trocada vs original (%)": round(100 * ((clear[key] >= 0.5) != (clear["prob"] >= 0.5)).mean(), 1)}
        for jp, part in clear.groupby("juxtapleural"):
            tag = "justapleurais" if jp else "longe da pleura"
            row[f"Δp médio {tag} (n={len(part)})"] = round((part[key] - part["prob"]).mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_shift(shift: pd.DataFrame, examples: dict) -> None:
    g = shift.groupby("shift_mm")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, color in (("gradcam", "tab:red"), ("ig", "tab:blue"), ("center", "tab:green")):
        ax[0].plot(g[f"{name}_pg_hit"].mean().index, 100 * g[f"{name}_pg_hit"].mean(), "o-", color=color,
                   label=MAP_LABELS[name])
    ax[0].set_title("Pointing Game com o nódulo deslocado (%)")
    clear = shift[shift["is_ambiguous"] == 0]
    aucs = clear.groupby("shift_mm").apply(lambda p: roc_auc_score(p["label"], p["prob"]), include_groups=False)
    ax[1].plot(aucs.index, aucs.to_numpy(), "o-", color="black")
    ax[1].set_title("AUC do v2 com o nódulo deslocado (não ambíguos)")
    for a in ax:
        a.set_xlabel("deslocamento do nódulo em relação ao centro (mm)")
        a.set_xticks([0] + SHIFTS_MM)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "deslocamento_v2_curvas.png", dpi=110)
    plt.close(fig)

    cases = sorted({i for i, _ in examples})
    shifts = [0.0] + SHIFTS_MM
    fig, axes = plt.subplots(len(cases), 2 * len(shifts), figsize=(2.3 * 2 * len(shifts), 2.5 * len(cases)))
    for r, i in enumerate(cases):
        for c, d in enumerate(shifts):
            if (i, d) not in examples:
                continue
            crop, mask, cam, ig, prob = examples[(i, d)]
            for j, (m, label) in enumerate(((cam, "Grad-CAM"), (ig, "IG"))):
                a = axes[r, 2 * c + j]
                a.imshow(crop, cmap="gray", vmin=0, vmax=1)
                a.imshow(m, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                a.contour(mask, levels=[0.5], colors="lime", linewidths=0.8)
                py, px = np.unravel_index(np.argmax(m), m.shape)
                a.plot(px, py, "w+", ms=9, mew=2)
                a.plot((IMAGE_SIZE - 1) / 2, (IMAGE_SIZE - 1) / 2, "c.", ms=5)
                a.set_title(f"{label}, {d:.0f} mm, p={prob:.2f}", fontsize=8)
    for a in axes.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle("Teste de deslocamento (v2, provisório): nódulo movido para a esquerda do centro; "
                 "verde = ROI, + = pico do mapa, ponto ciano = centro do recorte", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "deslocamento_v2_exemplos.png", dpi=100)
    plt.close(fig)


def plot_pleura(pleura: pd.DataFrame, stored: dict) -> None:
    outside = pleura[pleura["gradcam_peak_region"] == "fora_pulmao"]
    jp = pleura[(pleura["juxtapleural"] == 1) & (pleura["gradcam_peak_region"] != "fora_pulmao")]
    chosen = ([("pico do Grad-CAM fora do pulmão", c) for c in
               outside.sample(min(3, len(outside)), random_state=SEED)["case"]]
              + [("justapleural, pico em outra região", c) for c in
                 jp.sample(min(1, len(jp)), random_state=SEED)["case"]])
    fig, axes = plt.subplots(len(chosen), 4, figsize=(11, 2.9 * len(chosen)), squeeze=False)
    for r, (title, case) in enumerate(chosen):
        crop, mask, crop_lung, crop_inside, cam = stored[case]
        crop = crop.astype(np.float32)
        row = pleura[pleura["case"] == case].iloc[0]
        occ = occluded(crop, mask, crop_lung, crop_inside)
        panels = [("TC: pulmão (ciano) e ROI (verde)", crop, None),
                  (f"Grad-CAM, p={row['prob']:.2f}", crop, cam.astype(np.float32)),
                  (f"sem o que está fora do pulmão\np={row['prob_sem_fora_pulmao']:.2f}", occ["sem_fora_pulmao"], None),
                  (f"sem o nódulo\np={row['prob_sem_nodulo']:.2f}", occ["sem_nodulo"], None)]
        for c, (label, image, overlay) in enumerate(panels):
            a = axes[r, c]
            a.imshow(image, cmap="gray", vmin=0, vmax=1)
            if overlay is not None:
                a.imshow(overlay, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                py, px = np.unravel_index(np.argmax(overlay), overlay.shape)
                a.plot(px, py, "w+", ms=10, mew=2)
            if c < 2:
                a.contour(crop_lung, levels=[0.5], colors="cyan", linewidths=0.7)
            a.contour(mask, levels=[0.5], colors="lime", linewidths=0.8)
            a.set_title(label, fontsize=8)
            a.set_xticks([]), a.set_yticks([])
        verdict = "ambíguo" if row["is_ambiguous"] else ("maligno" if row["label"] else "benigno")
        axes[r, 0].set_ylabel(f"{title}\n{row['patient_id']} | {verdict}\npleura a {row['pleura_dist_mm']:.1f} mm",
                              fontsize=7)
    fig.suptitle("Pleura e parede torácica (v2, provisório): onde o Grad-CAM cai e o que muda ao remover o contexto",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pleura_v2_exemplos.png", dpi=100)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_frames(("test",))
    cases = one_patch_per_nodule(frames["test"])
    check_against_sprint5(cases)
    thresholds = validation_thresholds()
    print("Map thresholds (best validation IoU, Sprint 5):", thresholds, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explainer = Explainer(load_resnet(V2_CKPT, device), device)
    shift, pleura, stored, examples = run(cases, explainer, thresholds)
    explainer.close()
    gap = (pleura["prob"] - pleura["prob_v2_sprint5"]).abs().max()
    print(f"Max |p - p(sprint5 predictions)| on the unshifted crops: {gap:.2e}", flush=True)

    shift.to_csv(OUT / "deslocamento_v2_per_case.csv", index=False)
    pleura.to_csv(OUT / "pleura_v2_per_case.csv", index=False)
    tables = {
        "deslocamento_v2_summary.csv": shift_summary(shift),
        "pleura_v2_pico_regioes.csv": peak_region_summary(pleura),
        "pleura_v2_oclusao.csv": occlusion_summary(pleura),
    }
    context, shortcut = context_summary(pleura)
    tables["pleura_v2_contexto.csv"] = context
    tables["pleura_v2_atalho.csv"] = shortcut
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for name, table in tables.items():
        table.to_csv(OUT / name, index=False)
        print(f"\n== {name}\n{table.T.to_string() if len(table.columns) > 8 else table.to_string(index=False)}",
              flush=True)
    plot_shift(shift, examples)
    plot_pleura(pleura, stored)
    print("Wrote", *(OUT / n for n in tables), sep="\n  ")
    print("Wrote", FIG_DIR / "deslocamento_v2_curvas.png", FIG_DIR / "deslocamento_v2_exemplos.png",
          FIG_DIR / "pleura_v2_exemplos.png")


if __name__ == "__main__":
    main()
