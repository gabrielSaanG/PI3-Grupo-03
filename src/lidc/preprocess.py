"""LIDC-IDRI XML → nodule cluster → 64×64 crop pipeline (matches the project notebook)."""

from __future__ import annotations

import hashlib
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.cluster import DBSCAN

HU_MIN, HU_MAX = -1000, 400
RAW_CROP_SIZE = 96
FINAL_SIZE = 64
CLUSTER_EPS_MM = 10.0
Z_WARN = 5.0
Z_DROP = 15.0
SEMANTIC_FIELDS = [
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "malignancy",
]
NODULE_KEY = ["series_uid", "spatial_cluster"]
PATIENT_FOLDER_RE = re.compile(r"LIDC-IDRI-\d+")


@dataclass
class ProjectPaths:
    root: Path
    base: Path
    output_root: Path
    preprocess_dir: Path
    crop_dir: Path
    splits_dir: Path
    gradcam_dir: Path
    xml_official_dir: Path
    tmp_dir: Path
    full_record_csv: Path
    crops_only_csv: Path
    dicom_index_csv: Path
    batch_plan_csv: Path
    batch_log_csv: Path
    tcia_manifest: Path

    def ensure(self) -> None:
        for folder in (
            self.base,
            self.preprocess_dir,
            self.crop_dir,
            self.splits_dir,
            self.xml_official_dir,
            self.tmp_dir,
            self.gradcam_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)


def discover_paths(root: Optional[Path] = None) -> ProjectPaths:
    if root is None:
        root = Path(__file__).resolve().parents[2]
    base = root / "data" / "LIDC-IDRI"
    output_root = root / "outputs"
    preprocess_dir = output_root / "preprocessing"
    paths = ProjectPaths(
        root=root,
        base=base,
        output_root=output_root,
        preprocess_dir=preprocess_dir,
        crop_dir=output_root / "crops",
        splits_dir=output_root / "splits",
        gradcam_dir=root / "gradcam",
        xml_official_dir=base / "_xml_official",
        tmp_dir=base / "_tmp_downloads",
        full_record_csv=preprocess_dir / "lidc_crop_extraction_full_record.csv",
        crops_only_csv=preprocess_dir / "lidc_model_table_fixed_crops.csv",
        dicom_index_csv=preprocess_dir / "lidc_dicom_index.csv",
        batch_plan_csv=preprocess_dir / "lidc_next_batch_plan.csv",
        batch_log_csv=preprocess_dir / "lidc_batch_log.csv",
        tcia_manifest=preprocess_dir / "lidc_next_batch.tcia",
    )
    paths.ensure()
    return paths


def patient_id_from_path(path: os.PathLike | str) -> Optional[str]:
    p = Path(path)
    for part in p.parts:
        if PATIENT_FOLDER_RE.fullmatch(part):
            return part
    stem = p.stem
    if PATIENT_FOLDER_RE.fullmatch(stem):
        return stem
    match = PATIENT_FOLDER_RE.search(p.name)
    return match.group(0) if match else None


def _is_aux_path(path: Path) -> bool:
    return any(part.startswith("_") for part in path.parts)


def local_patient_ids_with_dicom(base: Path) -> set[str]:
    found: set[str] = set()
    if not base.exists():
        return found
    for dcm in base.rglob("*.dcm"):
        try:
            if _is_aux_path(dcm.relative_to(base)):
                continue
        except ValueError:
            pass
        pid = patient_id_from_path(dcm)
        if pid:
            found.add(pid)
    return found


def processed_patient_ids(crops_csv: Path) -> set[str]:
    if not crops_csv.exists():
        return set()
    df = pd.read_csv(crops_csv)
    if "patient_id" not in df.columns:
        return set()
    return set(df["patient_id"].dropna().astype(str))


def xai_keep_patients(gradcam_dir: Path) -> set[str]:
    """Collect patient IDs from any slot under gradcam/*/gradcam_selected_cases.csv."""
    keep: set[str] = set()
    if not gradcam_dir.exists():
        return keep
    for csv_path in gradcam_dir.glob("*/gradcam_selected_cases.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "patient_id" not in df.columns:
            continue
        keep |= set(df["patient_id"].dropna().astype(str))
    return keep


def next_crop_index(crop_dir: Path) -> int:
    max_idx = -1
    for path in crop_dir.glob("nodule_*.npy"):
        match = re.match(r"nodule_(\d+)_", path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _clean_tag(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _get_child_text(parent: Optional[ET.Element], target_tag: str) -> Optional[str]:
    if parent is None:
        return None
    for child in parent:
        if _clean_tag(child.tag) == target_tag:
            return child.text
    return None


def parse_lidc_xml(xml_path: str | Path) -> list[dict[str, Any]]:
    xml_path = str(xml_path)
    rows: list[dict[str, Any]] = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return rows

    study_uid = None
    series_uid = None
    for elem in root.iter():
        tag = _clean_tag(elem.tag)
        if tag.lower() == "studyinstanceuid":
            study_uid = elem.text
        elif tag.lower() == "seriesinstanceuid":
            series_uid = elem.text

    patient_id = patient_id_from_path(xml_path)
    reader_id = 0
    for session in root.iter():
        if _clean_tag(session.tag) != "readingSession":
            continue
        reader_id += 1
        for nodule in session:
            if _clean_tag(nodule.tag) != "unblindedReadNodule":
                continue
            nodule_id = _get_child_text(nodule, "noduleID")
            row: dict[str, Any] = {
                "xml_file": os.path.basename(xml_path),
                "xml_path": xml_path,
                "patient_id": patient_id,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "reader_id": reader_id,
                "nodule_id": nodule_id,
            }
            char_node = None
            for child in nodule:
                if _clean_tag(child.tag) == "characteristics":
                    char_node = child
                    break
            for field in SEMANTIC_FIELDS:
                value = _get_child_text(char_node, field) if char_node is not None else None
                row[field] = int(value) if value is not None else np.nan

            xs, ys, zs, sop_uids = [], [], [], []
            for roi in nodule:
                if _clean_tag(roi.tag) != "roi":
                    continue
                z_pos = _get_child_text(roi, "imageZposition")
                sop_uid = _get_child_text(roi, "imageSOP_UID")
                for edge in roi:
                    if _clean_tag(edge.tag) != "edgeMap":
                        continue
                    x = _get_child_text(edge, "xCoord")
                    y = _get_child_text(edge, "yCoord")
                    if x is None or y is None:
                        continue
                    xs.append(float(x))
                    ys.append(float(y))
                    if z_pos is not None:
                        zs.append(float(z_pos))
                    if sop_uid is not None:
                        sop_uids.append(sop_uid)
            row["centroid_x"] = np.mean(xs) if xs else np.nan
            row["centroid_y"] = np.mean(ys) if ys else np.nan
            row["z_position"] = np.mean(zs) if zs else np.nan
            row["num_contour_points"] = len(xs)
            row["sop_uids"] = "|".join(sorted(set(sop_uids))) if sop_uids else None
            rows.append(row)
    return rows


def collect_xml_rows(base: Path, xml_official_dir: Optional[Path] = None) -> pd.DataFrame:
    local_xml = sorted(p for p in base.rglob("*.xml") if not _is_aux_path(p.relative_to(base)))
    official_xml: list[Path] = []
    if xml_official_dir is not None and xml_official_dir.exists():
        official_xml = sorted(xml_official_dir.rglob("*.xml"))

    seen_hashes: set[str] = set()
    seen_series: set[str] = set()
    all_rows: list[dict[str, Any]] = []

    def ingest(files: list[Path], skip_known_series: bool = False) -> None:
        for xml_path in files:
            digest = hashlib.md5(xml_path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            parsed = parse_lidc_xml(xml_path)
            if not parsed:
                continue
            series_uids = {str(row.get("series_uid") or "") for row in parsed}
            series_uids.discard("")
            if skip_known_series and series_uids and series_uids <= seen_series:
                continue
            seen_hashes.add(digest)
            seen_series.update(series_uids)
            all_rows.extend(parsed)

    ingest(local_xml, skip_known_series=False)
    ingest(official_xml, skip_known_series=True)
    return pd.DataFrame(all_rows)


def index_local_dicoms(base: Path) -> pd.DataFrame:
    dcm_rows: list[dict[str, Any]] = []
    for path in base.rglob("*.dcm"):
        try:
            if _is_aux_path(path.relative_to(base)):
                continue
        except ValueError:
            pass
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(ds, "Modality", "")) != "CT":
            continue
        image_position = getattr(ds, "ImagePositionPatient", None)
        z_pos = float(image_position[2]) if image_position is not None else np.nan
        pixel_spacing = getattr(ds, "PixelSpacing", [np.nan, np.nan])
        row_spacing = float(pixel_spacing[0]) if pixel_spacing is not None else np.nan
        col_spacing = float(pixel_spacing[1]) if pixel_spacing is not None else np.nan
        patient_id = str(getattr(ds, "PatientID", "") or "") or patient_id_from_path(path)
        dcm_rows.append(
            {
                "dicom_path": str(path),
                "patient_id_dicom": patient_id,
                "study_uid_dicom": str(getattr(ds, "StudyInstanceUID", "")),
                "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
                "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
                "instance_number": getattr(ds, "InstanceNumber", np.nan),
                "z_position_dicom": z_pos,
                "row_spacing": row_spacing,
                "col_spacing": col_spacing,
                "slice_thickness": getattr(ds, "SliceThickness", np.nan),
                "modality": str(getattr(ds, "Modality", "")),
            }
        )
    return pd.DataFrame(dcm_rows)


def prepare_annotation_table(df_xml: pd.DataFrame, df_dcm: pd.DataFrame) -> pd.DataFrame:
    if df_xml.empty:
        return df_xml
    df_valid = df_xml.dropna(subset=SEMANTIC_FIELDS).copy()
    df_valid = df_valid[(df_valid[SEMANTIC_FIELDS] > 0).all(axis=1)].copy()
    if df_valid.empty or df_dcm.empty:
        return df_valid

    series_info = (
        df_dcm.dropna(subset=["series_uid"])
        .groupby("series_uid")
        .agg(
            {
                "row_spacing": "median",
                "col_spacing": "median",
                "slice_thickness": "median",
                "patient_id_dicom": "first",
            }
        )
        .reset_index()
    )
    df_valid = df_valid.merge(series_info, on="series_uid", how="left")
    df_valid["patient_id"] = df_valid["patient_id"].fillna(df_valid["patient_id_dicom"])
    df_valid = df_valid.dropna(subset=["row_spacing", "col_spacing"]).copy()
    df_valid["centroid_x_mm"] = df_valid["centroid_x"] * df_valid["col_spacing"]
    df_valid["centroid_y_mm"] = df_valid["centroid_y"] * df_valid["row_spacing"]
    df_valid["centroid_z_mm"] = df_valid["z_position"]
    return df_valid


def cluster_annotations(df_valid: pd.DataFrame, eps_mm: float = CLUSTER_EPS_MM) -> pd.DataFrame:
    cluster_input = df_valid.dropna(
        subset=["series_uid", "centroid_x_mm", "centroid_y_mm", "centroid_z_mm"]
    ).copy()
    if cluster_input.empty:
        return cluster_input

    parts = []
    for _, group in cluster_input.groupby("series_uid"):
        group = group.copy()
        coords = group[["centroid_x_mm", "centroid_y_mm", "centroid_z_mm"]].values
        group["spatial_cluster"] = DBSCAN(eps=eps_mm, min_samples=1).fit_predict(coords)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def _collect_sop_uids(values: pd.Series) -> str:
    sop_list: list[str] = []
    for value in values.dropna():
        for item in str(value).split("|"):
            item = item.strip()
            if item and item.lower() not in {"nan", "none"}:
                sop_list.append(item)
    return "|".join(sorted(set(sop_list)))


def build_nodule_table(df_clustered: pd.DataFrame) -> pd.DataFrame:
    if df_clustered.empty:
        return df_clustered
    group_cols = NODULE_KEY
    agg_base = {
        "patient_id": "first",
        "study_uid": "first",
        "centroid_x": "mean",
        "centroid_y": "mean",
        "z_position": "mean",
        "centroid_x_mm": "mean",
        "centroid_y_mm": "mean",
        "centroid_z_mm": "mean",
        "reader_id": "nunique",
        "num_contour_points": "sum",
        "row_spacing": "median",
        "col_spacing": "median",
        "slice_thickness": "median",
        "sop_uids": _collect_sop_uids,
    }
    nodules = df_clustered.groupby(group_cols).agg(agg_base).reset_index()
    nodules = nodules.rename(columns={"reader_id": "num_readers"})
    for col in SEMANTIC_FIELDS:
        stats = (
            df_clustered.groupby(group_cols)[col]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": f"{col}_mean", "std": f"{col}_std", "count": f"{col}_count"})
        )
        nodules = nodules.merge(stats, on=group_cols, how="left")
    nodules["label"] = np.where(
        nodules["malignancy_mean"] <= 2,
        0,
        np.where(nodules["malignancy_mean"] >= 4, 1, np.nan),
    )
    nodules = nodules.dropna(subset=["label"]).copy()
    nodules["label"] = nodules["label"].astype(int)
    return nodules


def parse_sop_uids(raw_sop: Any) -> list[str]:
    if pd.isna(raw_sop):
        return []
    raw = str(raw_sop).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return [
        item.strip()
        for item in re.split(r"[|,]", raw)
        if item.strip() and item.strip().lower() not in {"nan", "none"}
    ]


def load_hu_image(dicom_path: str) -> tuple[np.ndarray, str]:
    ds = pydicom.dcmread(dicom_path, force=True)
    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    return img * slope + intercept, photometric


def window_hu(img: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    img = np.clip(img, hu_min, hu_max)
    img = (img - hu_min) / (hu_max - hu_min)
    return img.astype(np.float32)


def extract_crop(
    img: np.ndarray,
    cx: float,
    cy: float,
    raw_size: int = RAW_CROP_SIZE,
    final_size: int = FINAL_SIZE,
) -> tuple[Optional[np.ndarray], bool]:
    h, w = img.shape
    cx_i = int(round(float(cx)))
    cy_i = int(round(float(cy)))
    if cx_i < 0 or cx_i >= w or cy_i < 0 or cy_i >= h:
        return None, False
    half = raw_size // 2
    x1, x2 = cx_i - half, cx_i + half
    y1, y2 = cy_i - half, cy_i + half
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w)
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    img_pad = np.pad(
        img,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top
    crop = img_pad[y1:y2, x1:x2]
    crop = cv2.resize(crop, (final_size, final_size), interpolation=cv2.INTER_AREA)
    return crop, True


def choose_best_slice(
    row: dict[str, Any],
    sop_to_row: dict[str, dict[str, Any]],
    series_to_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    target_z = row.get("z_position", np.nan)
    series_uid = str(row.get("series_uid", "")).strip()
    sop_list = parse_sop_uids(row.get("sop_uids", ""))
    valid_sops = [s for s in sop_list if s in sop_to_row]
    if valid_sops:
        def sop_z_diff(sop: str) -> float:
            z = sop_to_row[sop].get("z_position_dicom", np.nan)
            if pd.isna(z) or pd.isna(target_z):
                return np.inf
            return abs(float(z) - float(target_z))

        best_sop = min(valid_sops, key=sop_z_diff)
        meta = sop_to_row[best_sop]
        matched_z = meta.get("z_position_dicom", np.nan)
        z_diff = (
            abs(float(matched_z) - float(target_z))
            if pd.notna(matched_z) and pd.notna(target_z)
            else np.nan
        )
        return {
            "chosen_dicom_path": meta["dicom_path"],
            "chosen_sop_uid": best_sop,
            "chosen_z": matched_z,
            "chosen_z_diff": z_diff,
            "match_method": "sop",
        }

    if series_uid in series_to_rows and pd.notna(target_z):
        slices = series_to_rows[series_uid]
        if slices:
            best = min(
                slices,
                key=lambda item: abs(float(item["z_position_dicom"]) - float(target_z))
                if pd.notna(item.get("z_position_dicom"))
                else np.inf,
            )
            matched_z = best.get("z_position_dicom", np.nan)
            z_diff = abs(float(matched_z) - float(target_z)) if pd.notna(matched_z) else np.nan
            return {
                "chosen_dicom_path": best["dicom_path"],
                "chosen_sop_uid": best.get("sop_uid"),
                "chosen_z": matched_z,
                "chosen_z_diff": z_diff,
                "match_method": "nearest_z",
            }

    return {
        "chosen_dicom_path": None,
        "chosen_sop_uid": None,
        "chosen_z": np.nan,
        "chosen_z_diff": np.nan,
        "match_method": "none",
    }


def _lookup_maps(
    df_dcm: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    df_dcm = df_dcm.copy()
    df_dcm["sop_uid"] = df_dcm["sop_uid"].astype(str).str.strip()
    df_dcm["series_uid"] = df_dcm["series_uid"].astype(str).str.strip()
    df_dcm["z_position_dicom"] = pd.to_numeric(df_dcm["z_position_dicom"], errors="coerce")
    sop_to_row = df_dcm.dropna(subset=["sop_uid"]).set_index("sop_uid").to_dict("index")
    series_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in df_dcm.dropna(subset=["series_uid"]).iterrows():
        series_to_rows[row["series_uid"]].append(row.to_dict())
    return sop_to_row, series_to_rows


def _existing_keys(df: Optional[pd.DataFrame]) -> set[tuple[str, int]]:
    if df is None or df.empty or not set(NODULE_KEY).issubset(df.columns):
        return set()
    keys = set()
    for _, row in df.iterrows():
        try:
            keys.add((str(row["series_uid"]), int(row["spatial_cluster"])))
        except (TypeError, ValueError):
            continue
    return keys


def extract_new_crops(
    nodule_df: pd.DataFrame,
    df_dcm: pd.DataFrame,
    crop_dir: Path,
    existing_clean: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    crop_dir.mkdir(parents=True, exist_ok=True)
    sop_to_row, series_to_rows = _lookup_maps(df_dcm)
    skip_keys = _existing_keys(existing_clean)
    crop_idx = next_crop_index(crop_dir)
    results: list[dict[str, Any]] = []

    for _, row in nodule_df.iterrows():
        key = (str(row["series_uid"]), int(row["spatial_cluster"]))
        if key in skip_keys:
            continue
        row_dict = row.to_dict()
        match_info = choose_best_slice(row_dict, sop_to_row, series_to_rows)
        out = {**row_dict, **match_info, "crop_path": None, "crop_ok": False, "crop_error": None}
        if match_info["chosen_dicom_path"] is None:
            out["crop_error"] = "no_dicom_match"
            results.append(out)
            continue
        try:
            hu_img, photometric = load_hu_image(match_info["chosen_dicom_path"])
            img = window_hu(hu_img)
            if "MONOCHROME1" in photometric.upper():
                img = 1.0 - img
            crop, valid = extract_crop(img, cx=row["centroid_x"], cy=row["centroid_y"])
            if not valid:
                out["crop_error"] = "centroid_off_image"
                results.append(out)
                continue
            label = int(row["label"])
            mal = float(row["malignancy_mean"])
            crop_filename = f"nodule_{crop_idx:05d}_label_{label}_mal_{mal:.2f}.npy"
            crop_path = crop_dir / crop_filename
            np.save(crop_path, crop)
            out["crop_path"] = str(crop_path)
            out["crop_ok"] = True
            crop_idx += 1
        except Exception as exc:  # noqa: BLE001 — keep going on per-nodule failures
            out["crop_error"] = str(exc)
        results.append(out)
    return pd.DataFrame(results)


def flag_and_clean(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty:
        return result_df
    ok_df = result_df[result_df["crop_ok"]].copy()
    if ok_df.empty:
        return ok_df
    ok_df["chosen_z_diff"] = pd.to_numeric(ok_df["chosen_z_diff"], errors="coerce")
    ok_df["z_flag"] = "ok"
    ok_df.loc[ok_df["chosen_z_diff"] > Z_WARN, "z_flag"] = "warn"
    ok_df.loc[ok_df["chosen_z_diff"] > Z_DROP, "z_flag"] = "drop"
    return ok_df[ok_df["z_flag"] != "drop"].copy()


def merge_tables(existing: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new.reset_index(drop=True)
    if new is None or new.empty:
        return existing.reset_index(drop=True)
    combined = pd.concat([existing, new], ignore_index=True)
    if set(NODULE_KEY).issubset(combined.columns):
        combined = combined.drop_duplicates(subset=NODULE_KEY, keep="first")
    return combined.reset_index(drop=True)


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def cleanup_patient_dicoms(
    base: Path,
    patients: Iterable[str],
    keep: Iterable[str],
    dry_run: bool = False,
) -> dict[str, int]:
    keep_set = set(keep)
    deleted = 0
    skipped_keep = 0
    bytes_freed = 0
    for patient_id in patients:
        if patient_id in keep_set:
            skipped_keep += 1
            continue
        folder = base / patient_id
        if not folder.exists():
            continue
        for dcm in folder.rglob("*.dcm"):
            size = dcm.stat().st_size if dcm.exists() else 0
            if dry_run:
                deleted += 1
                bytes_freed += size
                continue
            try:
                dcm.unlink()
                deleted += 1
                bytes_freed += size
            except OSError:
                continue
    return {"deleted_files": deleted, "kept_patients": skipped_keep, "bytes_freed": bytes_freed}
