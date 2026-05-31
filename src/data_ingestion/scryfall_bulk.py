from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from src.config.settings import paths, settings
from src.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class BulkInfo:
    """
    Metadata for one Scryfall bulk data entry.
    """

    bulk_type: str
    name: str
    download_uri: str
    updated_at: str


def get_bulk_index() -> dict:
    """
    Input:
        Scryfall bulk index URL from settings.

    Logic:
        Requests the current Scryfall bulk-data index.

    Output:
        Parsed JSON response.
    """
    response = requests.get(settings.scryfall_bulk_index_url, timeout=60)
    response.raise_for_status()

    return response.json()


def get_bulk_info(bulk_type: str) -> BulkInfo:
    """
    Input:
        Scryfall bulk type, such as all_cards, default_cards, or oracle_cards.

    Logic:
        Finds the matching bulk-data entry in the Scryfall bulk index.

    Output:
        BulkInfo for the requested bulk type.
    """
    data = get_bulk_index()

    for item in data.get("data", []):
        if item.get("type") == bulk_type:
            return BulkInfo(
                bulk_type=bulk_type,
                name=item.get("name", bulk_type),
                download_uri=item["download_uri"],
                updated_at=item.get("updated_at", ""),
            )

    raise ValueError(f"Could not find Scryfall bulk type='{bulk_type}' in index.")


def delete_old_bulk_files(raw_dir: Path, bulk_type: str, keep_json: Path, keep_meta: Path) -> None:
    """
    Input:
        Raw data directory, bulk type, and latest json/meta paths.

    Logic:
        Deletes older downloaded files for the same bulk type.

    Output:
        Older raw JSON/meta files removed from disk.
    """
    for file in raw_dir.glob(f"{bulk_type}_*.json"):
        if file != keep_json and not file.name.endswith(".meta.json"):
            file.unlink(missing_ok=True)

    for file in raw_dir.glob(f"{bulk_type}_*.meta.json"):
        if file != keep_meta:
            file.unlink(missing_ok=True)


def download_file(url: str, dest_path: Path) -> None:
    """
    Input:
        Download URL and destination path.

    Logic:
        Streams the file to disk in chunks with a progress bar.

    Output:
        Downloaded file saved to dest_path.
    """
    ensure_dir(dest_path.parent)

    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()

        total = int(response.headers.get("Content-Length", "0")) or None

        with dest_path.open("wb") as file, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest_path.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))


def fetch_latest_scryfall_bulk(bulk_type: str | None = None) -> Path:
    """
    Input:
        Optional Scryfall bulk type.

    Logic:
        Downloads the latest requested Scryfall bulk file into data/raw,
        writes metadata, and keeps only the latest local file for that bulk type.

    Output:
        Path to the local bulk JSON file.
    """
    selected_bulk_type = bulk_type or settings.scryfall_bulk_type
    info = get_bulk_info(selected_bulk_type)

    raw_dir = paths.data_raw
    ensure_dir(raw_dir)

    safe_stamp = info.updated_at.replace(":", "").replace("-", "")
    out_json = raw_dir / f"{info.bulk_type}_{safe_stamp}.json"
    meta_path = raw_dir / f"{info.bulk_type}_{safe_stamp}.meta.json"

    if out_json.exists() and not settings.always_redownload_bulk:
        return out_json

    delete_old_bulk_files(
        raw_dir=raw_dir,
        bulk_type=info.bulk_type,
        keep_json=out_json,
        keep_meta=meta_path,
    )

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


# TODO Phase 4:
# Explore more robust ingestion approaches:
#   - cache and compare Scryfall updated_at before downloading
#   - support multiple retained raw versions for rollback/debugging
#   - add checksum or file-size validation after download
#   - support resumable downloads for very large bulk files
#   - log ingestion metadata into a pipeline run manifest
