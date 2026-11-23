"""Recompute the notebook split and audit patient/study/series/SOP overlap.

Stops before training. Uses the same notebook cells and random_state=42.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "03-resnet50-sprint3.ipynb"
OUT = ROOT / "outputs" / "sprint3"
STOP_MARKER = "VERIFICAÇÃO DE DATA LEAKAGE"


def _ids(frame: pd.DataFrame, column: str) -> set:
    if column not in frame.columns:
        return set()
    return set(frame[column].dropna().astype(str))


def main() -> None:
    os.chdir(ROOT / "notebooks")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns: dict = {"__name__": "__main__"}
    plt.show = lambda *args, **kwargs: None
    ns["display"] = lambda *args, **kwargs: None

    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        print(f"Executing cell {index}", flush=True)
        try:
            exec(source, ns, ns)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        if STOP_MARKER in source:
            break
    else:
        print("Stop marker not found.", flush=True)
        raise SystemExit(1)

    train_df = ns["train_df"].copy()
    val_df = ns["val_df"].copy()
    test_df = ns["test_df"].copy()
    dicom_index = ns["dicom_index"]
    sop_to_series = (
        dicom_index.dropna(subset=["sop_uid"])
        .drop_duplicates("sop_uid")
        .set_index("sop_uid")["series_uid"]
        .astype(str)
    )
    for frame in (train_df, val_df, test_df):
        frame["series_uid"] = frame["sop_uid"].map(sop_to_series)
        frame["nodule_key"] = (
            frame["patient_id"].astype(str) + "|" + frame["nodule_id"].astype(str)
        )

    splits = {"train": train_df, "validation": val_df, "test": test_df}
    summary_rows = []
    for name, frame in splits.items():
        summary_rows.append(
            {
                "split": name,
                "patches": len(frame),
                "patients": frame["patient_id"].nunique(),
                "studies": frame["study_uid"].nunique(),
                "series": frame["series_uid"].nunique(),
                "slices": frame["sop_uid"].nunique(),
                "nodules": frame["nodule_id"].nunique() if "nodule_id" in frame.columns else None,
                "benign": int((frame["label"] == 0).sum()),
                "malignant": int((frame["label"] == 1).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    keys = ["patient_id", "study_uid", "series_uid", "sop_uid", "nodule_key"]
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    overlap_rows = []
    id_sets = {name: {key: _ids(frame, key) for key in keys} for name, frame in splits.items()}
    for left, right in pairs:
        for key in keys:
            shared = id_sets[left][key] & id_sets[right][key]
            overlap_rows.append(
                {
                    "left": left,
                    "right": right,
                    "identifier": key,
                    "shared_count": len(shared),
                    "shared_ids": "" if not shared else ";".join(sorted(shared)[:20]),
                }
            )
    overlap = pd.DataFrame(overlap_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "split_summary.csv"
    overlap_path = OUT / "split_intersection.csv"
    summary.to_csv(summary_path, index=False)
    overlap.to_csv(overlap_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print(overlap.to_string(index=False), flush=True)
    failed = overlap.loc[overlap["shared_count"] > 0]
    if failed.empty:
        print("AUDIT OK: no shared patient, study, series, slice, or nodule id.", flush=True)
    else:
        print("AUDIT FAILED", flush=True)
        print(failed.to_string(index=False), flush=True)
        raise SystemExit(2)
    print("Wrote", summary_path, flush=True)
    print("Wrote", overlap_path, flush=True)


if __name__ == "__main__":
    main()
