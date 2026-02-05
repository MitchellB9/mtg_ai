from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_external: Path = PROJECT_ROOT / "data" / "external"
    artifacts: Path = PROJECT_ROOT / "artifacts"
    artifacts_models: Path = PROJECT_ROOT / "artifacts" / "models"
    artifacts_vectorizers: Path = PROJECT_ROOT / "artifacts" / "vectorizers"
    artifacts_encoders: Path = PROJECT_ROOT / "artifacts" / "encoders"


@dataclass(frozen=True)
class Settings:
    # Scryfall bulk endpoint (we’ll discover the latest bulk file from here)
    scryfall_bulk_index_url: str = "https://api.scryfall.com/bulk-data"

    # Which bulk dataset we want (usually "oracle_cards" for oracle text)
    scryfall_bulk_type: str = os.getenv("SCRYFALL_BULK_TYPE", "oracle_cards")

    # Safety: if you want to avoid re-downloading every run, set this false
    always_redownload_bulk: bool = (
        os.getenv("ALWAYS_REDOWNLOAD_BULK", "true").lower() == "true"
    )


paths = Paths()
settings = Settings()
