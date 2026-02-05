from __future__ import annotations

import pandas as pd

from src.config.settings import paths
from src.data_ingestion.scryfall_bulk import fetch_latest_scryfall_bulk
from src.utils.io import ensure_dir

def main() -> None:
    ensure_dir(paths.data_processed)

    bulk_path = fetch_latest_scryfall_bulk()
    print(f"Loaded bulk file: {bulk_path}")

    # Scryfall bulk json is a list of card objects -> pandas can read it directly
    df = pd.read_json(bulk_path)

    # Minimal “stage 0” cleanup: keep a core subset (expand later)
    core_cols = [
        "id", "oracle_id", "name", "released_at", "lang",
        "layout", "type_line", "oracle_text", "mana_cost",
        "colors", "color_identity", "keywords",
        "legalities", "set", "set_name", "rarity",
        "cmc", "power", "toughness"
    ]
    keep = [c for c in core_cols if c in df.columns]
    df_core = df[keep].copy()

    out_path = paths.data_processed / "cards_core.parquet"
    df_core.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} (rows={len(df_core)}, cols={len(df_core.columns)})")

if __name__ == "__main__":
    main()
