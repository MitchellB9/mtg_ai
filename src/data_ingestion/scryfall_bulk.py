from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import requests
from tqdm import tqdm

from src.config.settings import paths, settings
from src.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class BulkInfo:
    bulk_type: str
    name: str
    download_uri: str
    updated_at: str


def _get_bulk_index() -> dict:
    resp = requests.get(settings.scryfall_bulk_index_url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def get_bulk_info(bulk_type: str) -> BulkInfo:
    data = _get_bulk_index()
    for item in data.get("data", []):
        if item.get("type") == bulk_type:
            return BulkInfo(
                bulk_type=bulk_type,
                name=item.get("name", bulk_type),
                download_uri=item["download_uri"],
                updated_at=item.get("updated_at", ""),
            )
    raise ValueError(f"Could not find Scryfall bulk type='{bulk_type}' in index.")


def download_file(url: str, dest_path: Path) -> None:
    ensure_dir(dest_path.parent)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0")) or None
        with dest_path.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest_path.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    if total is not None:
                        pbar.update(len(chunk))


def fetch_latest_scryfall_bulk(bulk_type: str | None = None) -> Path:
    """
    Downloads the latest Scryfall bulk file into data/raw/
    and writes a metadata json alongside it.

    Keeps only the most recent file for the given bulk_type.
    """
    selected_bulk_type = bulk_type or settings.scryfall_bulk_type
    info = get_bulk_info(selected_bulk_type)

    raw_dir = paths.data_raw
    ensure_dir(raw_dir)

    safe_stamp = info.updated_at.replace(":", "").replace("-", "")
    out_json = raw_dir / f"{info.bulk_type}_{safe_stamp}.json"
    meta_path = raw_dir / f"{info.bulk_type}_{safe_stamp}.meta.json"

    # If latest already exists, return it
    if out_json.exists() and not settings.always_redownload_bulk:
        return out_json

    # 🔹 Delete older versions of this bulk type
    for file in raw_dir.glob(f"{info.bulk_type}_*.json"):
        if file != out_json:
            file.unlink(missing_ok=True)

    for file in raw_dir.glob(f"{info.bulk_type}_*.meta.json"):
        if file != meta_path:
            file.unlink(missing_ok=True)

    # Download new file
    download_file(info.download_uri, out_json)

    write_json(
        meta_path,
        {
            "bulk_type": info.bulk_type,
            "name": info.name,
            "download_uri": info.download_uri,
            "updated_at": info.updated_at,
            "saved_as": out_json.name,
        },
    )

    return out_json
