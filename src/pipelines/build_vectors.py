from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config.settings import paths
from src.utils.io import ensure_dir
from src.features.embedding_features import (
    fit_transform_tfidf,
    save_tfidf_artifacts,
    TfidfConfig,
)


def main() -> None:
    ensure_dir(paths.artifacts_vectorizers)

    in_path = paths.data_processed / "cards_enriched.parquet"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing {in_path}. Run: python -m src.pipelines.preprocess_text"
        )

    df = pd.read_parquet(in_path)
    if "oracle_text_norm" not in df.columns:
        raise ValueError("cards_enriched.parquet is missing oracle_text_norm column.")

    texts = df["oracle_text_norm"].fillna("")
    ids = df["id"].to_numpy()

    cfg = TfidfConfig(
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 2),
        max_features=250_000,
        sublinear_tf=True,
        lowercase=True,
    )

    vec, X = fit_transform_tfidf(texts, cfg)
    out_dir = paths.artifacts_vectorizers / "tfidf_oracle_v1"
    save_tfidf_artifacts(out_dir, vec, X, ids)

    print(f"Saved TF-IDF artifacts to: {out_dir}")
    print(f"Matrix shape: {X.shape}")


if __name__ == "__main__":
    main()
