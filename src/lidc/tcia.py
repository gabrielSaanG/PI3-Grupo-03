"""TCIA / NBIA helpers for LIDC-IDRI CT series (public REST API, no login)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
XML_ONLY_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/LIDC-XML-only.zip"
COLLECTION = "LIDC-IDRI"
USER_AGENT = "PI3-Grupo-03-lidc-batch/1.0"
BYTES_PER_SLICE_ESTIMATE = 420_000  # ~0.4 MB; used only for disk planning


def _request(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _request_with_retries(url: str, timeout: int, retries: int = 3) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return _request(url, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = exc
            if attempt == retries:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed GET {url} after {retries} tries: {last_err}") from last_err


def list_ct_series(timeout: int = 180) -> list[dict[str, Any]]:
    """All public LIDC-IDRI CT series metadata (PatientID, SeriesInstanceUID, ImageCount)."""
    query = urllib.parse.urlencode({"Collection": COLLECTION, "Modality": "CT", "format": "json"})
    payload = _request_with_retries(f"{NBIA_BASE}/getSeries?{query}", timeout=timeout)
    series = json.loads(payload.decode("utf-8"))
    if not isinstance(series, list):
        raise RuntimeError("Unexpected TCIA getSeries payload")
    return series


def estimate_series_bytes(row: dict[str, Any]) -> int:
    count = row.get("ImageCount") or row.get("imageCount") or 0
    try:
        n = int(count)
    except (TypeError, ValueError):
        n = 0
    return n * BYTES_PER_SLICE_ESTIMATE


def series_by_patient(series: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in series:
        pid = str(row.get("PatientID") or "").strip()
        if not pid:
            continue
        grouped.setdefault(pid, []).append(row)
    return grouped


def download_series_zip(
    series_uid: str,
    dest_zip: Path,
    timeout: int = 600,
    retries: int = 3,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
) -> int:
    """Download one series as ZIP. Returns bytes written."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    url = f"{NBIA_BASE}/getImage?{urllib.parse.urlencode({'SeriesInstanceUID': series_uid})}"
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = resp.headers.get("Content-Length")
                total_n = int(total) if total and total.isdigit() else None
                tmp = dest_zip.with_suffix(dest_zip.suffix + ".part")
                written = 0
                with tmp.open("wb") as fh:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        fh.write(chunk)
                        written += len(chunk)
                        if progress:
                            progress(written, total_n)
                tmp.replace(dest_zip)
                return written
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_err = exc
            part = dest_zip.with_suffix(dest_zip.suffix + ".part")
            if part.exists():
                part.unlink()
            if attempt == retries:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download series {series_uid}: {last_err}") from last_err


def extract_series_zip(zip_path: Path, dest_dir: Path) -> int:
    """Extract a TCIA series ZIP into dest_dir. Returns number of files written."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name or name.startswith("."):
                continue
            target = dest_dir / name
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
            n_files += 1
    return n_files


def download_xml_archive(dest_dir: Path, timeout: int = 180) -> Path:
    """Download the official 8.6 MB LIDC XML-only zip and extract it."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "LIDC-XML-only.zip"
    if not zip_path.exists():
        req = urllib.request.Request(XML_ONLY_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp, zip_path.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
    marker = dest_dir / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        marker.write_text("ok\n", encoding="utf-8")
    return dest_dir


def write_tcia_manifest(series_uids: Iterable[str], dest: Path) -> None:
    """NBIA Data Retriever manifest (.tcia) if you prefer the official client."""
    uids = [u.strip() for u in series_uids if str(u).strip()]
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "downloadServerUrl=https://nbia.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet",
        "includeAnnotation=true",
        "noOfrRetry=4",
        f"databasketId={dest.name}",
        "manifestVersion=3.0",
        "",
        "ListOfSeriesToDownload=",
        *uids,
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
