from __future__ import annotations

import pandas as pd

from src.config.settings import paths
from src.utils.io import ensure_dir
from src.preprocessing.text_normalization import (
    normalize_oracle_text,
    TextNormalizationOptions,
)
from src.preprocessing.oracle_tokenizer import tokenize_oracle_text, TokenizeOptions


BASE_PARQUET_DIR = paths.data_processed.parent / "base_parquets"

INPUT_PATH = BASE_PARQUET_DIR / "game_features.parquet"

ORACLE_TOKENS_OUT = paths.data_processed / "oracle_tokens.parquet"
CARDS_ENRICHED_OUT = paths.data_processed / "cards_enriched.parquet"


def validate_input(df: pd.DataFrame) -> None:
    """
    Input:
        game_features DataFrame.

    Logic:
        Confirms required columns exist before downstream text/type processing.

    Output:
        Raises ValueError if the input dataset is not usable.
    """
    required_cols = ["id", "oracle_id", "name", "oracle_text"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"game_features is missing required columns: {missing}")

    duplicate_oracle_ids = df["oracle_id"].duplicated().sum()

    if duplicate_oracle_ids:
        raise ValueError(
            f"Expected one row per oracle_id, found duplicates: {duplicate_oracle_ids}"
        )


def add_normalized_oracle_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        game_features DataFrame with oracle_text and name.

    Logic:
        Normalizes oracle text for consistent tokenization and downstream NLP.

    Output:
        Copy of the DataFrame with oracle_text_norm added.
    """
    norm_opts = TextNormalizationOptions(
        strip_reminder_text=False,
        replace_card_name=True,
    )

    df = df.copy()

    df["oracle_text_norm"] = [
        normalize_oracle_text(text, name, norm_opts)
        for text, name in zip(df["oracle_text"], df["name"])
    ]

    return df


def build_oracle_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        DataFrame with normalized oracle text.

    Logic:
        Tokenizes normalized oracle text.

    Output:
        oracle_tokens DataFrame.
    """
    tok_opts = TokenizeOptions(
        keep_newlines=True,
        lowercase=True,
    )

    return pd.DataFrame(
        {
            "id": df["id"],
            "oracle_id": df["oracle_id"],
            "name": df["name"],
            "oracle_text_norm": df["oracle_text_norm"],
            "oracle_tokens": df["oracle_text_norm"].apply(
                lambda text: tokenize_oracle_text(text, tok_opts)
            ),
        }
    )


def main() -> None:
    """
    Input:
        data/base_parquets/game_features.parquet

    Logic:
        Adds normalized oracle text, builds oracle token data, and saves an
        enriched card dataset for downstream vectorization and clustering.

    Output:
        data/processed/oracle_tokens.parquet
        data/processed/cards_enriched.parquet
    """
    ensure_dir(paths.data_processed)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run: python -m src.data_processing.build_base_parquets"
        )

    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded: {INPUT_PATH} (rows={len(df):,}, cols={len(df.columns):,})")

    validate_input(df)

    df = add_normalized_oracle_text(df)

    df_tokens = build_oracle_tokens(df)
    df_tokens.to_parquet(ORACLE_TOKENS_OUT, index=False)
    print(f"Saved: {ORACLE_TOKENS_OUT} (rows={len(df_tokens):,})")

    df_enriched = df.copy()
    df_enriched.to_parquet(CARDS_ENRICHED_OUT, index=False)
    print(f"Saved: {CARDS_ENRICHED_OUT} (rows={len(df_enriched):,})")

    print("Done.")


if __name__ == "__main__":
    main()

    # TODO Phase 4:
        # Experiment with additional oracle-text preprocessing approaches:
        #   - optional reminder-text removal modes
        #   - mana-symbol canonicalization strategies
        #   - sentence-level oracle segmentation
        #   - stemming vs lemmatization evaluation
        #   - mechanic/action extraction pipelines
        #   - structured ability parsing
        #   - named-entity replacement beyond card names
        #   - multilingual preprocessing support
        #   - evaluate preserving vs removing punctuation/newlines
