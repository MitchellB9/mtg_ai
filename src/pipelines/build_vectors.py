from __future__ import annotations

import pandas as pd

from src.config.settings import paths
from src.utils.io import ensure_dir
from src.features.embedding_features import (
    fit_transform_tfidf,
    save_tfidf_artifacts,
    TfidfConfig,
)

INPUT_PATH = paths.data_processed / "cards_enriched.parquet"
OUTPUT_DIR = paths.artifacts_vectorizers / "tfidf_oracle_v1"


def validate_input(df: pd.DataFrame) -> None:
    """
    Input:
        cards_enriched DataFrame.

    Logic:
        Confirms required columns exist before vectorization.

    Output:
        Raises ValueError if required fields are missing.
    """
    required_cols = ["id", "oracle_id", "name", "oracle_text_norm"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"cards_enriched is missing required columns: {missing}")


def main() -> None:
    """
    Input:
        data/processed/cards_enriched.parquet

    Logic:
        Builds a TF-IDF feature matrix from normalized oracle text.

    Output:
        artifacts/vectorizers/tfidf_oracle_v1/
            vectorizer artifact
            sparse TF-IDF matrix
            card id index
    """
    ensure_dir(OUTPUT_DIR)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run: python -m src.data_processing.preprocess_text"
        )

    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded: {INPUT_PATH} (rows={len(df):,}, cols={len(df.columns):,})")

    validate_input(df)

    texts = df["oracle_text_norm"].fillna("")
    ids = df["id"].to_numpy()

    cfg = TfidfConfig(
        min_df=2,
        max_df=0.97,
        ngram_range=(1, 2),
        max_features=100_000,
        sublinear_tf=True,
        lowercase=True,
    )

    print("TF-IDF config:")
    print(f"  min_df={cfg.min_df}")
    print(f"  max_df={cfg.max_df}")
    print(f"  ngram_range={cfg.ngram_range}")
    print(f"  max_features={cfg.max_features}")
    print(f"  sublinear_tf={cfg.sublinear_tf}")
    print(f"  lowercase={cfg.lowercase}")

    vectorizer, matrix = fit_transform_tfidf(texts, cfg)

    save_tfidf_artifacts(OUTPUT_DIR, vectorizer, matrix, ids)

    print(f"Saved TF-IDF artifacts to: {OUTPUT_DIR}")
    print(f"Matrix shape: {matrix.shape}")


if __name__ == "__main__":
    main()

    # TODO Phase 4:
    # Experiment with additional text/vector representations after the current
    # TF-IDF pipeline is stable:
    #   - CountVectorizer baseline
    #   - character n-grams for oracle wording patterns
    #   - keyword/mechanic-weighted TF-IDF
    #   - sentence-transformer embeddings
    #   - hybrid vectors: oracle text + type/color/mana features
    #   - dimensionality reduction before clustering: SVD/LSA, UMAP
