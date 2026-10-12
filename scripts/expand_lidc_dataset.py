"""Expand LIDC-IDRI without downloading the full 133 GB collection.

Downloads CT series in small patient batches from TCIA, extracts the same
64×64 nodule crops used by the notebook, then deletes DICOMs (XML is kept).

Examples:
  python scripts/expand_lidc_dataset.py plan --batch-size 25
  python scripts/expand_lidc_dataset.py run --batch-size 25 --max-gb 8
  python scripts/expand_lidc_dataset.py extract
  python scripts/expand_lidc_dataset.py cleanup --keep-xai-patients
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _bind_runtime_imports() -> None:
    """Import pandas / OpenCV / pydicom only after argparse has handled --help."""
    import pandas as pd
    from lidc.preprocess import (
        ProjectPaths,
        build_nodule_table,
        cleanup_patient_dicoms,
        cluster_annotations,
        collect_xml_rows,
        discover_paths,
        extract_new_crops,
        flag_and_clean,
        index_local_dicoms,
        local_patient_ids_with_dicom,
        merge_tables,
        prepare_annotation_table,
        processed_patient_ids,
        read_csv_if_exists,
        xai_keep_patients,
    )
    from lidc.tcia import (
        download_series_zip,
        download_xml_archive,
        estimate_series_bytes,
        extract_series_zip,
        list_ct_series,
        series_by_patient,
        write_tcia_manifest,
    )

    globals().update({key: value for key, value in locals().items()})


def _gb(n_bytes: int | float) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GB"


def _log(paths: ProjectPaths, **row: object) -> None:
    paths.batch_log_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "action", "patient_id", "series_uid", "status", "detail"]
    exists = paths.batch_log_csv.exists()
    with paths.batch_log_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        payload = {name: "" for name in fieldnames}
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        payload.update({k: row.get(k, "") for k in fieldnames if k != "timestamp"})
        writer.writerow(payload)


def _sort_patient_ids(ids: list[str]) -> list[str]:
    def key(pid: str) -> tuple[int, str]:
        digits = "".join(ch for ch in pid if ch.isdigit())
        return (int(digits) if digits else 10**9, pid)

    return sorted(ids, key=key)


def inventory(paths: ProjectPaths) -> dict[str, object]:
    local_dicom = local_patient_ids_with_dicom(paths.base)
    already_cropped = processed_patient_ids(paths.crops_only_csv)
    print("Querying TCIA for LIDC-IDRI CT series...")
    series = list_ct_series()
    grouped = series_by_patient(series)
    all_patients = _sort_patient_ids(list(grouped))
    done = local_dicom | already_cropped
    remaining = [pid for pid in all_patients if pid not in done]
    return {
        "series": series,
        "grouped": grouped,
        "all_patients": all_patients,
        "local_dicom": local_dicom,
        "already_cropped": already_cropped,
        "remaining": remaining,
    }


def select_batch(
    remaining: list[str],
    grouped: dict[str, list[dict]],
    batch_size: int,
    max_bytes: int | None,
) -> tuple[list[str], list[dict], int]:
    chosen_patients: list[str] = []
    chosen_series: list[dict] = []
    total = 0
    for pid in remaining:
        rows = grouped.get(pid, [])
        batch_bytes = sum(estimate_series_bytes(row) for row in rows)
        if max_bytes is not None and chosen_patients and total + batch_bytes > max_bytes:
            break
        chosen_patients.append(pid)
        chosen_series.extend(rows)
        total += batch_bytes
        if len(chosen_patients) >= batch_size:
            break
    return chosen_patients, chosen_series, total


def cmd_plan(args: argparse.Namespace, paths: ProjectPaths) -> int:
    info = inventory(paths)
    remaining: list[str] = info["remaining"]  # type: ignore[assignment]
    grouped: dict = info["grouped"]  # type: ignore[assignment]
    max_bytes = int(args.max_gb * (1024 ** 3)) if args.max_gb else None
    patients, series, est = select_batch(remaining, grouped, args.batch_size, max_bytes)

    print()
    print(f"TCIA CT patients:          {len(info['all_patients'])}")
    print(f"Local patients with DICOM: {len(info['local_dicom'])}")
    print(f"Patients already cropped:  {len(info['already_cropped'])}")
    print(f"Patients still to fetch:   {len(remaining)}")
    print(f"Next batch patients:       {len(patients)}")
    print(f"Next batch CT series:      {len(series)}")
    print(f"Estimated download size:   {_gb(est)}")
    if patients:
        print("Patient IDs:", ", ".join(patients[:12]) + (" ..." if len(patients) > 12 else ""))

    plan_rows = []
    for row in series:
        plan_rows.append(
            {
                "patient_id": row.get("PatientID"),
                "series_uid": row.get("SeriesInstanceUID"),
                "study_uid": row.get("StudyInstanceUID"),
                "image_count": row.get("ImageCount"),
                "estimated_bytes": estimate_series_bytes(row),
            }
        )
    pd.DataFrame(plan_rows).to_csv(paths.batch_plan_csv, index=False)
    write_tcia_manifest(
        [r["series_uid"] for r in plan_rows if r.get("series_uid")],
        paths.tcia_manifest,
    )
    print()
    print("Wrote", paths.batch_plan_csv)
    print("Wrote", paths.tcia_manifest)
    print("You can open the .tcia file with the NBIA Data Retriever if you prefer.")
    return 0


def cmd_download_xml(args: argparse.Namespace, paths: ProjectPaths) -> int:
    if getattr(args, "dry_run", False):
        print(f"Would download XML archive to {paths.xml_official_dir}")
        return 0
    print("Downloading official LIDC XML annotations (~8.6 MB)...")
    dest = download_xml_archive(paths.xml_official_dir)
    n_xml = len(list(dest.rglob("*.xml")))
    print(f"XML files available: {n_xml} under {dest}")
    _log(paths, action="download_xml", status="ok", detail=str(n_xml))
    return 0


def _download_one_series(paths: ProjectPaths, patient_id: str, series_uid: str) -> Path:
    dest_dir = paths.base / patient_id / series_uid
    existing = list(dest_dir.glob("*.dcm")) if dest_dir.exists() else []
    if existing:
        print(f"  skip existing {patient_id} / {series_uid[-12:]}")
        return dest_dir
    zip_path = paths.tmp_dir / f"{series_uid}.zip"
    print(f"  downloading {patient_id} {series_uid[-16:]} ...")

    def progress(written: int, total: int | None) -> None:
        if total:
            pct = 100.0 * written / total
            print(f"\r    {pct:6.1f}%  {_gb(written)} / {_gb(total)}", end="", flush=True)
        else:
            print(f"\r    {_gb(written)}", end="", flush=True)

    download_series_zip(series_uid, zip_path, progress=progress)
    print()
    n_files = extract_series_zip(zip_path, dest_dir)
    try:
        zip_path.unlink()
    except OSError:
        pass
    print(f"    extracted {n_files} files -> {dest_dir}")
    _log(paths, action="download", patient_id=patient_id, series_uid=series_uid, status="ok", detail=n_files)
    return dest_dir


def cmd_download(args: argparse.Namespace, paths: ProjectPaths) -> int:
    info = inventory(paths)
    remaining: list[str] = info["remaining"]  # type: ignore[assignment]
    grouped: dict = info["grouped"]  # type: ignore[assignment]
    max_bytes = int(args.max_gb * (1024 ** 3)) if args.max_gb else None
    patients, series, est = select_batch(remaining, grouped, args.batch_size, max_bytes)
    if not patients:
        print("Nothing to download. Local DICOMs + existing crops already cover TCIA, or batch is empty.")
        return 0
    print(f"Downloading {len(patients)} patients / {len(series)} CT series (est. {_gb(est)})")
    if args.dry_run:
        for pid in patients:
            print(" ", pid)
        return 0
    failures = 0
    for row in series:
        pid = str(row.get("PatientID") or "")
        uid = str(row.get("SeriesInstanceUID") or "")
        if not pid or not uid:
            continue
        try:
            _download_one_series(paths, pid, uid)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAILED {pid} {uid}: {exc}")
            _log(paths, action="download", patient_id=pid, series_uid=uid, status="error", detail=str(exc))
    print(f"Download finished. Failures: {failures}")
    return 1 if failures else 0


def cmd_extract(args: argparse.Namespace, paths: ProjectPaths) -> int:
    print("Indexing local CT DICOMs...")
    df_dcm = index_local_dicoms(paths.base)
    df_dcm.to_csv(paths.dicom_index_csv, index=False)
    print(f"  CT slices: {len(df_dcm)}  patients: {df_dcm['patient_id_dicom'].nunique() if not df_dcm.empty else 0}")
    if df_dcm.empty:
        print("No local CT DICOMs found. Run `download` first.")
        return 1

    print("Parsing XML annotations...")
    df_xml = collect_xml_rows(paths.base, xml_official_dir=paths.xml_official_dir)
    print(f"  annotation rows: {len(df_xml)}")
    if df_xml.empty:
        print("No XML found. Run `download-xml` (and/or download CT series, which often include XML).")
        return 1

    df_valid = prepare_annotation_table(df_xml, df_dcm)
    df_clustered = cluster_annotations(df_valid)
    nodule_df = build_nodule_table(df_clustered)
    print(f"  clustered nodules with binary label: {len(nodule_df)}")

    existing_clean = read_csv_if_exists(paths.crops_only_csv)
    existing_full = read_csv_if_exists(paths.full_record_csv)
    print("Extracting new crops (skips nodules already in the model table)...")
    new_full = extract_new_crops(nodule_df, df_dcm, paths.crop_dir, existing_clean=existing_clean)
    new_clean = flag_and_clean(new_full)

    full_df = merge_tables(existing_full, new_full)
    clean_df = merge_tables(existing_clean, new_clean)

    # Refuse to shrink an on-disk table (e.g. accidental empty extract).
    if (
        existing_clean is not None
        and not existing_clean.empty
        and len(clean_df) < len(existing_clean)
    ):
        print(
            f"REFUSED to shrink crop table ({len(existing_clean)} -> {len(clean_df)}). "
            "Keeping the larger on-disk CSV."
        )
        clean_df = existing_clean
        if existing_full is not None and not existing_full.empty:
            full_df = existing_full

    full_df.to_csv(paths.full_record_csv, index=False)
    clean_df.to_csv(paths.crops_only_csv, index=False)

    n_new = int(new_clean["crop_ok"].sum()) if not new_clean.empty and "crop_ok" in new_clean.columns else len(new_clean)
    print(f"  new clean crops this run: {n_new}")
    print(f"  model table now: {len(clean_df)} nodules / {clean_df['patient_id'].nunique()} patients")
    if "label" in clean_df.columns:
        print("  labels:\n", clean_df["label"].value_counts().to_string())
    print("Saved", paths.crops_only_csv)
    _log(paths, action="extract", status="ok", detail=f"new={n_new};total={len(clean_df)}")
    print()
    print("Re-run the notebook from the train/val/test split cell so the new crops enter training.")
    return 0


def _cleanup_targets(paths: ProjectPaths) -> list[str]:
    cropped = processed_patient_ids(paths.crops_only_csv)
    local = local_patient_ids_with_dicom(paths.base)
    return _sort_patient_ids(list(cropped & local))


def _keep_set(args: argparse.Namespace, paths: ProjectPaths) -> set[str]:
    keep: set[str] = set()
    if getattr(args, "keep_xai_patients", False):
        keep |= xai_keep_patients(paths.gradcam_dir)
    extra = getattr(args, "keep_patients", "") or ""
    for item in extra.split(","):
        item = item.strip()
        if item:
            keep.add(item)
    return keep


def cmd_cleanup(args: argparse.Namespace, paths: ProjectPaths) -> int:
    targets = _cleanup_targets(paths)
    keep = _keep_set(args, paths)
    print(f"Patients with crops + local DICOM: {len(targets)}")
    print(f"Keeping DICOM for: {', '.join(_sort_patient_ids(list(keep))) or '(none)'}")
    stats = cleanup_patient_dicoms(paths.base, targets, keep=keep, dry_run=args.dry_run)
    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"{verb} {stats['deleted_files']} DICOM files ({_gb(stats['bytes_freed'])}). XML files are kept.")
    _log(
        paths,
        action="cleanup",
        status="dry_run" if args.dry_run else "ok",
        detail=f"files={stats['deleted_files']};bytes={stats['bytes_freed']}",
    )
    return 0


def cmd_run(args: argparse.Namespace, paths: ProjectPaths) -> int:
    cmd_download_xml(args, paths)
    local = local_patient_ids_with_dicom(paths.base)
    cropped = processed_patient_ids(paths.crops_only_csv)
    pending_local = local - cropped
    if pending_local:
        print(f"Extracting {len(pending_local)} local patients not yet in the crop table...")
        rc = cmd_extract(args, paths)
        if rc != 0:
            return rc
    rc = 0
    if args.batch_size > 0:
        rc = cmd_download(args, paths)
        if rc != 0 and not args.dry_run:
            print("Download had failures; extracting whatever arrived...")
    elif args.dry_run:
        print("batch-size is 0; skipping download.")
    if not args.dry_run:
        rc_ex = cmd_extract(args, paths)
        if rc_ex != 0:
            return rc_ex
        if args.no_cleanup:
            print("Skipping cleanup (--no-cleanup).")
        else:
            args.keep_xai_patients = True
            cmd_cleanup(args, paths)
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download LIDC-IDRI CT in small batches, extract crops, delete DICOMs."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_batch_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--batch-size", type=int, default=25, help="How many new patients to fetch (default: 25)")
        p.add_argument("--max-gb", type=float, default=8.0, help="Stop adding patients once estimated size exceeds this")
        p.add_argument("--dry-run", action="store_true", help="Print actions without downloading/deleting")

    def add_keep_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--keep-xai-patients",
            action="store_true",
            help="Do not delete DICOMs for patients listed in gradcam/*/gradcam_selected_cases.csv",
        )
        p.add_argument(
            "--keep-patients",
            default="",
            help="Comma-separated extra PatientIDs to keep, e.g. LIDC-IDRI-0005,LIDC-IDRI-0045",
        )

    p_plan = sub.add_parser("plan", help="List next patients and write a TCIA manifest")
    add_batch_flags(p_plan)
    p_plan.set_defaults(func=cmd_plan)

    p_xml = sub.add_parser("download-xml", help="Download the official 8.6 MB XML annotation archive")
    p_xml.set_defaults(func=cmd_download_xml)

    p_dl = sub.add_parser("download", help="Download the next CT batch from TCIA")
    add_batch_flags(p_dl)
    p_dl.set_defaults(func=cmd_download)

    p_ex = sub.add_parser("extract", help="Parse XML, cluster nodules, write new 64x64 crops")
    p_ex.set_defaults(func=cmd_extract)

    p_cl = sub.add_parser("cleanup", help="Delete DICOMs for patients already in the crop table")
    p_cl.add_argument("--dry-run", action="store_true")
    add_keep_flags(p_cl)
    p_cl.set_defaults(func=cmd_cleanup)

    p_run = sub.add_parser("run", help="XML + download + extract + cleanup in one go")
    add_batch_flags(p_run)
    add_keep_flags(p_run)
    p_run.add_argument("--no-cleanup", action="store_true", help="Keep DICOMs after extracting crops")
    p_run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _bind_runtime_imports()
    paths = discover_paths(ROOT)  # type: ignore[name-defined]
    return int(args.func(args, paths) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
