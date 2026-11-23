"""Sprint 5: nodule-level metrics with patient-level bootstrap 95% CIs.

Each --pred file needs: patient_id, study_uid, reading_session, nodule_id, label,
prob_malignant (one row per patch). Patches are averaged per nodule; patients are
resampled with replacement. "Always malignant" is added on the same nodules.
With --val, the Youden threshold is chosen on validation nodules only and applied
to test as an extra row (name@val_youden), next to the fixed threshold.

  .venv\\Scripts\\python.exe scripts/sprint5_evaluate.py ^
      --pred resnet50=outputs/sprint5/preds_test_resnet50.csv ^
      --pred cnn_simple=outputs/sprint5/preds_test_cnn_simple.csv ^
      --val resnet50=outputs/sprint5/preds_val_resnet50.csv ^
      --val cnn_simple=outputs/sprint5/preds_val_cnn_simple.csv ^
      --status provisorio
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sprint5"
MANIFEST = OUT / "split_manifest.csv"
LABEL_RANDOMIZATION = ROOT / "outputs" / "sprint3" / "adebayo_label_randomization_results.csv"
NODULE_KEY = ["patient_id", "study_uid", "reading_session", "nodule_id"]
REQUIRED = NODULE_KEY + ["label", "prob_malignant"]
PRIMARY = ["auc", "sensitivity", "specificity", "balanced_accuracy"]
SECONDARY = ["accuracy", "precision", "f1"]
ALWAYS_MALIGNANT = "always_malignant"


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {
        "auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2,
        "accuracy": (tp + tn) / len(y),
        "precision": prec,
        "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def load_predictions(name: str, path: Path, manifest: pd.DataFrame | None,
                     split: str = "test") -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={c: str for c in NODULE_KEY})
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise SystemExit(f"{name}: missing columns {missing} in {path}")
    if manifest is not None:
        test = manifest.loc[manifest["split"] == split]
        if len(frame) != len(test):
            raise SystemExit(f"{name}: {len(frame)} rows, manifest {split} has {len(test)}")
        if set(frame["patient_id"]) != set(test["patient_id"]):
            raise SystemExit(f"{name}: patient set differs from manifest {split} split")
        if "row_idx" in frame.columns:
            merged = frame.merge(test, on="row_idx", suffixes=("", "_m"))
            for col in NODULE_KEY + ["label"]:
                if not (merged[col].astype(str) == merged[f"{col}_m"].astype(str)).all():
                    raise SystemExit(f"{name}: column {col} does not match manifest by row_idx")
    return frame


def to_nodules(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(NODULE_KEY, sort=True)
    nodules = grouped.agg(
        label=("label", "max"),
        label_min=("label", "min"),
        prob_malignant=("prob_malignant", "mean"),
        n_patches=("label", "size"),
    ).reset_index()
    conflicts = int((nodules["label"] != nodules["label_min"]).sum())
    if conflicts:
        print(f"[warn] {conflicts} nodules have mixed patch labels; using max label.")
    return nodules.drop(columns="label_min")


def youden_threshold(nodules: pd.DataFrame) -> float:
    fpr, tpr, thresholds = roc_curve(nodules["label"], nodules["prob_malignant"])
    return float(thresholds[np.argmax(tpr - fpr)])


def bootstrap(nodules: pd.DataFrame, variants: dict, n_boot: int, seed: int):
    patients = nodules["patient_id"].to_numpy()
    unique_patients = np.unique(patients)
    rows_by_patient = [np.flatnonzero(patients == p) for p in unique_patients]
    y = nodules["label"].to_numpy()
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in variants}
    single_class = 0
    for _ in range(n_boot):
        chosen = rng.integers(0, len(unique_patients), len(unique_patients))
        idx = np.concatenate([rows_by_patient[i] for i in chosen])
        if len(np.unique(y[idx])) < 2:
            single_class += 1
        for name, (score, threshold) in variants.items():
            samples[name].append(metrics(y[idx], score[idx], threshold))
    return {name: pd.DataFrame(rows) for name, rows in samples.items()}, single_class


def summarize_label_randomization() -> dict | None:
    if not LABEL_RANDOMIZATION.exists():
        return None
    df = pd.read_csv(LABEL_RANDOMIZATION)
    rho = df["spearman_correlation"]
    majority = max(df["true_label"].mean(), 1 - df["true_label"].mean())
    return {
        "file": str(LABEL_RANDOMIZATION.relative_to(ROOT)),
        "file_modified": datetime.fromtimestamp(LABEL_RANDOMIZATION.stat().st_mtime).isoformat(),
        "n_cases": int(len(df)),
        "max_test_index": int(df["index"].max()),
        "spearman_median": float(rho.median()),
        "spearman_q1": float(rho.quantile(0.25)),
        "spearman_q3": float(rho.quantile(0.75)),
        "share_spearman_above_0_5": float((rho > 0.5).mean()),
        "original_model_accuracy": float((df["original_prediction"] == df["true_label"]).mean()),
        "random_label_model_accuracy": float((df["random_prediction"] == df["true_label"]).mean()),
        "majority_class_rate": float(majority),
        "prediction_agreement_original_vs_random": float(
            (df["original_prediction"] == df["random_prediction"]).mean()
        ),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", action="append", required=True, help="name=path (test)")
    parser.add_argument("--val", action="append", default=[],
                        help="name=path (validation); adds name@val_youden with the threshold chosen there")
    parser.add_argument("--status", choices=["provisorio", "oficial"], required=True)
    parser.add_argument("--manifest", default=str(MANIFEST.relative_to(ROOT)))
    parser.add_argument("--exclude-ambiguous", action="store_true",
                        help="drop rows with is_ambiguous == 1 (test and validation) after the manifest checks")
    parser.add_argument("--tag", default="", help="suffix for the output files")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    manifest = (pd.read_csv(manifest_path, dtype={c: str for c in NODULE_KEY})
                if manifest_path.exists() else None)
    if manifest is None:
        print(f"[warn] {manifest_path} not found; skipping manifest checks.")
    suffix = f"{args.status}_{args.tag}" if args.tag else args.status

    def drop_ambiguous(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        if not args.exclude_ambiguous:
            return frame
        if "is_ambiguous" not in frame.columns:
            raise SystemExit(f"{name}: --exclude-ambiguous needs an is_ambiguous column")
        return frame[frame["is_ambiguous"] == 0].reset_index(drop=True)

    inputs, nodule_tables, patch_rows = {}, {}, []
    for spec in args.pred:
        name, path = spec.split("=", 1)
        path = (ROOT / path) if not Path(path).is_absolute() else Path(path)
        frame = drop_ambiguous(load_predictions(name, path, manifest), name)
        inputs[name] = {"path": str(path.relative_to(ROOT)), "sha256_16": sha256(path)}
        patch_rows.append({"model": name, **metrics(
            frame["label"].to_numpy(), frame["prob_malignant"].to_numpy(), args.threshold)})
        nodule_tables[name] = to_nodules(frame)

    reference = next(iter(nodule_tables.values()))
    for name, table in nodule_tables.items():
        if not table[NODULE_KEY + ["label"]].equals(reference[NODULE_KEY + ["label"]]):
            raise SystemExit(f"{name}: nodule set or labels differ from the first prediction file")

    y = reference["label"].to_numpy()
    variants = {name: (table["prob_malignant"].to_numpy(), args.threshold)
                for name, table in nodule_tables.items()}

    val_thresholds = {}
    for spec in args.val:
        name, path = spec.split("=", 1)
        if name not in nodule_tables:
            raise SystemExit(f"--val {name}: no matching --pred")
        path = (ROOT / path) if not Path(path).is_absolute() else Path(path)
        val_nodules = to_nodules(drop_ambiguous(
            load_predictions(name, path, manifest, split="validation"), name))
        threshold = youden_threshold(val_nodules)
        val_thresholds[name] = {"threshold": threshold, "val_nodules": int(len(val_nodules)),
                                "path": str(path.relative_to(ROOT)), "sha256_16": sha256(path)}
        variants[f"{name}@val_youden"] = (variants[name][0], threshold)
        print(f"{name}: Youden threshold on validation nodules = {threshold:.4f}")

    variants[ALWAYS_MALIGNANT] = (np.ones(len(y)), 0.5)

    boot, single_class = bootstrap(reference, variants, args.n_boot, args.seed)
    rows = []
    for name, (score, threshold) in variants.items():
        point = metrics(y, score, threshold)
        for metric in PRIMARY + SECONDARY:
            values = boot[name][metric].dropna()
            rows.append({
                "status": args.status,
                "unit": "nodule (patient|study|reader|nodule_id)",
                "model": name,
                "threshold": threshold,
                "metric": metric,
                "role": "primary" if metric in PRIMARY else "secondary",
                "value": point[metric],
                "ci_lower": values.quantile(0.025) if len(values) else np.nan,
                "ci_upper": values.quantile(0.975) if len(values) else np.nan,
                "n_boot_valid": len(values),
            })
        rows.append({"status": args.status, "unit": "nodule", "model": name, "threshold": threshold,
                     "metric": "confusion_tn_fp_fn_tp", "role": "count",
                     "value": f"{point['tn']}/{point['fp']}/{point['fn']}/{point['tp']}"})
    table = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT / f"metrics_{suffix}.csv"
    table.to_csv(metrics_path, index=False)

    patch_table = pd.DataFrame(patch_rows)
    patch_table.insert(0, "status", args.status)
    patch_table.to_csv(OUT / f"metrics_patch_reference_{suffix}.csv", index=False)

    label_rand = summarize_label_randomization()
    meta = {
        "status": args.status,
        "created": datetime.now().isoformat(timespec="seconds"),
        "unit": "nodule = patient_id|study_uid|reading_session|nodule_id; score = mean patch prob",
        "bootstrap": "patient-level cluster bootstrap, percentile 95% CI",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "threshold": args.threshold,
        "val_youden_thresholds": val_thresholds,
        "n_patients": int(reference["patient_id"].nunique()),
        "n_nodules": int(len(reference)),
        "n_patches": int(reference["n_patches"].sum()),
        "nodule_prevalence_malignant": float(y.mean()),
        "bootstrap_single_class_resamples": single_class,
        "inputs": inputs,
        "manifest": args.manifest if manifest is not None else None,
        "exclude_ambiguous": args.exclude_ambiguous,
        "label_randomization": label_rand,
    }
    meta_path = OUT / f"run_meta_{suffix}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    view = table[table["role"] != "count"].copy()
    view["cell"] = view.apply(
        lambda r: f"{r['value']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]", axis=1)
    print(f"\nStatus: {suffix} | exclude_ambiguous={args.exclude_ambiguous} | patients={meta['n_patients']} nodules={meta['n_nodules']} "
          f"prevalence={meta['nodule_prevalence_malignant']:.3f} n_boot={args.n_boot}")
    print(view.pivot(index="metric", columns="model", values="cell")
          .reindex(PRIMARY + SECONDARY).to_string())
    print("\nConfusion (TN/FP/FN/TP) per nodule:")
    print(table[table["role"] == "count"][["model", "value"]].to_string(index=False))
    print("\nPatch-level reference (no CI):")
    print(patch_table[["model"] + PRIMARY + SECONDARY].round(3).to_string(index=False))
    if label_rand:
        print("\nLabel randomization summary:")
        for key, value in label_rand.items():
            print(f"  {key}: {value}")
    print("\nWrote", metrics_path)
    print("Wrote", meta_path)


if __name__ == "__main__":
    main()
