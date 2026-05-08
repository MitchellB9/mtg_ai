from __future__ import annotations
from typing import Any
import pandas as pd
from tqdm import tqdm
import json
import ijson
from pathlib import Path
from src.config.settings import paths
from src.data_ingestion.scryfall_bulk import fetch_latest_scryfall_bulk
from src.utils.io import ensure_dir
from src.utils.reports import write_json_report


BASE_PARQUET_DIR = paths.data_processed.parent / "base_parquets"

KNOWN_SUPERTYPES = {"Basic", "Legendary", "Snow", "World"}

KNOWN_TYPES = {
    "Artifact",
    "Creature",
    "Enchantment",
    "Instant",
    "Land",
    "Planeswalker",
    "Sorcery",
    "Battle",
    "Kindred",
}

IDENTITY_COLS = [
    "id",
    "oracle_id",
    "name",
]

COLUMN_GROUPS: dict[str, list[str]] = {
    "game_features": [
        "mana_cost",
        "cmc",
        "type_line",
        "oracle_text",
        "colors",
        "color_identity",
        "color_indicator",
        "keywords",
        "produced_mana",
        "game_changer",
        "power",
        "toughness",
        "loyalty",
        "defense",
        "attraction_lights",
        "card_faces",
        "all_parts",
        "legalities",
        "games",
        "rulings_uri",
    ],
    "every_printing": [
        "released_at",
        "lang",
        "prices",
        "rarity",
        "set_id",
        "set",
        "set_name",
        "set_type",
        "reprint",
        "highres_image",
        "image_status",
        "reserved",
        "foil",
        "nonfoil",
        "finishes",
        "oversized",
        "promo",
        "variation",
        "collector_number",
        "digital",
        "card_back_id",
        "artist",
        "artist_ids",
        "illustration_id",
        "border_color",
        "frame",
        "full_art",
        "textless",
        "layout",
        "booster",
        "story_spotlight",
        "prints_search_uri",
        "tcgplayer_etched_id",
        "flavor_text",
        "purchase_uris",
        "printed_name",
        "printed_type_line",
        "printed_text",
        "security_stamp",
        "promo_types",
        "watermark",
        "frame_effects",
        "preview",
        "variation_of",
        "content_warning",
        "flavor_name",
    ],
    "links_and_ids": [
        "multiverse_ids",
        "mtgo_id",
        "arena_id",
        "tcgplayer_id",
        "cardmarket_id",
        "mtgo_foil_id",
        "tcgplayer_etched_id",
        "card_back_id",
        "illustration_id",
        "resource_id",
        "uri",
        "scryfall_uri",
        "image_uris",
        "set_uri",
        "set_search_uri",
        "scryfall_set_uri",
        "rulings_uri",
        "prints_search_uri",
        "purchase_uris",
        "related_uris",
        "variation_of",
    ],
    "misc": [
        "edhrec_rank",
        "penny_rank",
        "life_modifier",
        "hand_modifier",
    ],
}

GROUP_LEVELS: dict[str, str] = {
    "game_features": "oracle",
    "every_printing": "printing",
    "links_and_ids": "printing_deduped",
    "misc": "oracle",
}

HASHABLE_SOURCE_COLS = [
    "colors",
    "color_identity",
    "keywords",
    "legalities",
    "games",
    "produced_mana",
    "all_parts",
    "card_faces",
    "artist_ids",
    "frame_effects",
    "promo_types",
    "finishes",
    "supertypes",
    "types",
    "subtypes",
]

NON_MAIN_GAME_LAYOUTS = {
    "art_series",
    "token",
    "planar",
    "vanguard",
    "scheme",
    "emblem",
    "double_faced_token",
}

NON_MAIN_GAME_TYPE_TERMS = {
    "Token",
    "Tolkien",
    "Emblem",
    "Card",
    "Ongoing",
    "Elite",
    "Host",
}


def make_json_key(value: Any) -> str | None:
    """
    Input:
        A raw Scryfall cell value.

    Logic:
        Converts lists/dicts/scalars into a stable JSON string.

    Output:
        Parquet-safe comparison key.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return json.dumps(value, sort_keys=True, default=str)


def add_hashable_companion_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        Raw Scryfall DataFrame.

    Logic:
        Preserves original complex columns and adds parquet-safe *_key columns
        for comparison, grouping, and duplicate checks.

    Output:
        DataFrame with additional *_key columns.
    """
    df = df.copy()

    for col in tqdm(HASHABLE_SOURCE_COLS, desc="Adding key columns"):
        if col in df.columns:
            df[f"{col}_key"] = df[col].apply(make_json_key)

    return df


def available_columns(df: pd.DataFrame, desired_cols: list[str]) -> list[str]:
    """
    Input:
        DataFrame and desired column list.

    Logic:
        Keeps only columns that exist in the current Scryfall bulk data.

    Output:
        Existing columns in their requested order.
    """
    return [col for col in desired_cols if col in df.columns]


def get_misc_columns(df: pd.DataFrame) -> list[str]:
    """
    Input:
        DataFrame after known column groups are defined.

    Logic:
        Finds raw columns not already assigned to a named base parquet group.
        Identity columns and generated tuple columns are excluded.

    Output:
        Miscellaneous raw columns.
    """
    assigned = set(IDENTITY_COLS)

    for cols in COLUMN_GROUPS.values():
        assigned.update(cols)

    return [
        col for col in df.columns if col not in assigned and not col.endswith("_tuple")
    ]


def dedupe_to_oracle_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        Printing-level Scryfall DataFrame.

    Logic:
        Sorts by release date, then keeps one representative row per oracle_id.
        Rows without oracle_id fall back to id so they are not accidentally dropped.

    Output:
        Oracle-level DataFrame.
    """
    df = df.copy()

    if "released_at" in df.columns:
        df = df.sort_values("released_at", ascending=True)

    if "oracle_id" in df.columns and "id" in df.columns:
        identity_key = df["oracle_id"].fillna(df["id"])
        return df.loc[~identity_key.duplicated()].copy()

    return df.drop_duplicates().copy()


def build_dedupe_subset_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        DataFrame that may contain unhashable list/dict columns.

    Logic:
        Converts every cell to a parquet-safe/string-safe comparison value.

    Output:
        DataFrame safe to use with duplicated/drop_duplicates logic.
    """
    safe_df = pd.DataFrame(index=df.index)

    for col in tqdm(df.columns, desc="Building dedupe keys"):
        safe_df[col] = df[col].apply(make_json_key)

    return safe_df


def build_group_parquet(
    df: pd.DataFrame,
    group_name: str,
    cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Input:
        Main-game Scryfall DataFrame, group name, and desired non-identity columns.

    Logic:
        Adds shared identity columns automatically, selects available group columns,
        adds hashable companion columns, and applies the group's row-level rule:
            - oracle: one row per oracle_id
            - printing: every printing
            - printing_deduped: every printing, then duplicate rows removed

    Output:
        Group DataFrame and summary report section.
    """
    identity_cols = available_columns(df, IDENTITY_COLS)
    existing_cols = available_columns(df, cols)

    key_cols = [f"{col}_key" for col in existing_cols if f"{col}_key" in df.columns]

    selected_cols = identity_cols + existing_cols + key_cols
    group_df = df[selected_cols].copy()

    if group_name == "game_features":
        group_df = add_type_line_features(group_df)

    level = GROUP_LEVELS.get(group_name, "oracle")

    starting_rows = len(group_df)

    if level == "oracle":
        group_df = dedupe_to_oracle_level(group_df)
    elif level == "printing_deduped":
        safe_dedupe_df = build_dedupe_subset_frame(group_df)
        keep_mask = ~safe_dedupe_df.duplicated()
        group_df = group_df.loc[keep_mask].copy()
    elif level == "printing":
        pass
    else:
        raise ValueError(f"Unknown group level for {group_name}: {level}")

    report = {
        "group_name": group_name,
        "level": level,
        "starting_rows": starting_rows,
        "final_rows": len(group_df),
        "columns": len(group_df.columns),
        "included_columns": list(group_df.columns),
        "missing_requested_columns": [col for col in cols if col not in df.columns],
    }

    return group_df, report


def write_base_parquets(df: pd.DataFrame) -> dict[str, Any]:
    """
    Input:
        Raw Scryfall DataFrame with hashable companion columns.

    Logic:
        Separates non-main-game pieces, then writes grouped base parquet files
        from the main-game card pool.

    Output:
        Report describing generated base parquet files and sequestered objects.
    """
    ensure_dir(BASE_PARQUET_DIR)

    main_df, non_main_df, split_report = split_main_and_non_main_game_pieces(df)

    non_main_df = drop_all_null_columns(non_main_df)

    non_main_path = BASE_PARQUET_DIR / "non_main_game_pieces.parquet"
    non_main_df.to_parquet(non_main_path, index=False)

    report: dict[str, Any] = {
        "source_rows": len(df),
        "source_columns": len(df.columns),
        "output_directory": str(BASE_PARQUET_DIR),
        "non_main_game_split": {
            **split_report,
            "output_path": str(non_main_path),
        },
        "groups": {},
    }

    all_groups = dict(COLUMN_GROUPS)

    for group_name, cols in tqdm(
        all_groups.items(),
        desc="Writing base parquet groups",
        total=len(all_groups),
    ):
        group_df, group_report = build_group_parquet(main_df, group_name, cols)

        out_path = BASE_PARQUET_DIR / f"{group_name}.parquet"
        group_df.to_parquet(out_path, index=False)

        group_report["output_path"] = str(out_path)
        report["groups"][group_name] = group_report

        tqdm.write(
            f"Saved {group_name}: "
            f"rows={len(group_df):,}, cols={len(group_df.columns):,}"
        )

        tqdm.write(
            f"Saved non_main_game_pieces: "
            f"rows={len(non_main_df):,}, cols={len(non_main_df.columns):,}"
        )

    return report


def split_main_and_non_main_game_pieces(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Input:
        Raw Scryfall DataFrame.

    Logic:
        Separates obvious non-main-game pieces before base parquet grouping.
        This keeps tokens, planes, schemes, art cards, emblems, vanguard cards,
        and explicit Token/Card type lines out of the main card tables while
        preserving them in their own sequestered parquet.

    Output:
        Main-game DataFrame, non-main-game DataFrame, and split report.
    """
    layout_mask = (
        df["layout"].isin(NON_MAIN_GAME_LAYOUTS)
        if "layout" in df.columns
        else pd.Series(False, index=df.index)
    )

    type_mask = (
        df["type_line"]
        .fillna("")
        .str.contains(
            r"\b(?:" + "|".join(NON_MAIN_GAME_TYPE_TERMS) + r")\b",
            case=False,
            regex=True,
        )
        if "type_line" in df.columns
        else pd.Series(False, index=df.index)
    )

    non_main_mask = layout_mask | type_mask

    main_df = df[~non_main_mask].copy()
    non_main_df = df[non_main_mask].copy()

    report = {
        "source_rows": len(df),
        "main_game_rows": len(main_df),
        "non_main_game_rows": len(non_main_df),
        "removed_by_layout": int(layout_mask.sum()),
        "removed_by_type_line": int(type_mask.sum()),
        "removed_total_unique_rows": int(non_main_mask.sum()),
        "non_main_layout_counts": (
            df.loc[layout_mask, "layout"].value_counts().to_dict()
            if "layout" in df.columns
            else {}
        ),
    }

    return main_df, non_main_df, report


def parse_type_line(type_line: str | None) -> tuple[list[str], list[str], list[str]]:
    """
    Input:
        A Scryfall type_line string.

    Logic:
        Splits the type line into supertypes, types, and subtypes.

    Output:
        (supertypes, types, subtypes)
    """
    if not isinstance(type_line, str) or not type_line:
        return [], [], []

    if "—" in type_line:
        left, right = type_line.split("—", 1)
        subtypes = [s.strip() for s in right.strip().split()]
    else:
        left = type_line
        subtypes = []

    left_parts = [p.strip() for p in left.strip().split()]

    supertypes = [p for p in left_parts if p in KNOWN_SUPERTYPES]
    types = [p for p in left_parts if p in KNOWN_TYPES]

    return supertypes, types, subtypes


def add_type_line_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        DataFrame containing type_line.

    Logic:
        Parses type_line into structured components.

    Output:
        DataFrame with added columns:
            - supertypes
            - types
            - subtypes
    """
    if "type_line" not in df.columns:
        return df

    df = df.copy()

    parsed = df["type_line"].apply(parse_type_line)

    df["supertypes"] = parsed.apply(lambda x: x[0])
    df["types"] = parsed.apply(lambda x: x[1])
    df["subtypes"] = parsed.apply(lambda x: x[2])

    return df


def drop_object_column_if_constant_card(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        Raw Scryfall DataFrame.

    Logic:
        Drops object if every row has the value 'card'.

    Output:
        DataFrame without object when it carries no useful information.
    """
    return df.drop(columns=["object"])


def drop_all_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        Any DataFrame.

    Logic:
        Removes columns where every value is null.

    Output:
        Smaller DataFrame with empty columns removed.
    """
    return df.dropna(axis=1, how="all")


def read_scryfall_json_streamed(json_path: Path) -> pd.DataFrame:
    """
    Input:
        Large Scryfall bulk JSON path.

    Logic:
        Streams the top-level JSON array one card at a time to avoid loading
        the entire file into memory as raw JSON text.

    Output:
        DataFrame containing all card objects.
    """
    rows = []

    with open(json_path, "rb") as f:
        for card in ijson.items(f, "item"):
            rows.append(card)

    return pd.DataFrame(rows)


def main() -> None:
    """
    Input:
        Latest Scryfall bulk JSON.

    Logic:
        Loads raw Scryfall data, preserves broad raw fields, adds hashable
        companion columns, and writes grouped base parquet files.

    Output:
        data/base_parquets/*.parquet
        data/base_parquets/base_parquets_report.json
    """
    ensure_dir(BASE_PARQUET_DIR)

    bulk_path = fetch_latest_scryfall_bulk(bulk_type="all_cards")
    print(f"Loaded bulk file: {bulk_path}")

    print("Reading Scryfall bulk JSON...")
    df_raw = read_scryfall_json_streamed(bulk_path)
    print(f"Loaded raw rows: {len(df_raw):,}, columns: {len(df_raw.columns):,}")

    print("Dropping constant object column if applicable...")
    df_raw = drop_object_column_if_constant_card(df_raw)

    print("Adding JSON-safe comparison key columns...")
    df_raw = add_hashable_companion_columns(df_raw)

    print("Splitting non-main game pieces...")
    split_main_and_non_main_game_pieces(df_raw)

    print("Writing base parquet groups...")
    report = write_base_parquets(df_raw)

    report_path = BASE_PARQUET_DIR / "base_parquets_report.json"
    write_json_report(report, report_path)


if __name__ == "__main__":
    main()

    # TODO Phase 4:
        # Explore additional raw/base dataset structures and normalization:
        #   - split non_main_game_pieces into tokens/emblems/planes/schemes/etc.
        #   - normalize nested URI/image/legality structures into relational tables
        #   - add incremental parquet update pipeline instead of full rebuilds
        #   - create dedicated legality-by-format dataset
        #   - investigate memory-efficient chunked processing for very large bulk files
        #   - evaluate DuckDB/Polars for large-scale preprocessing performance
        #   - add schema/version metadata tracking for parquet outputs
