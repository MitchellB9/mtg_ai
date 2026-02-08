from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from src.config.settings import paths
from src.utils.io import ensure_dir
from src.features.embedding_features import load_tfidf_artifacts
from src.models.clustering import cluster_with_svd_kmeans, ClusterConfig


def main() -> None:
    ensure_dir(paths.artifacts_models)
    ensure_dir(paths.data_processed)

    vec_dir = paths.artifacts_vectorizers / "tfidf_oracle_v1"
    if not vec_dir.exists():
        raise FileNotFoundError(
            f"Missing {vec_dir}. Run: python -m src.pipelines.build_vectors"
        )

    vec, X, row_ids = load_tfidf_artifacts(vec_dir)

    cfg = ClusterConfig(
        n_clusters=200,
        n_components=256,
        random_state=42,
        n_init=10,
    )

    labels, svd, km = cluster_with_svd_kmeans(X, cfg)

    # Save models
    model_dir = paths.artifacts_models / "svd_kmeans_tfidf_v1"
    ensure_dir(model_dir)
    joblib.dump(svd, model_dir / "svd.joblib", compress=3)
    joblib.dump(km, model_dir / "kmeans.joblib", compress=3)

    # Save label mapping
    out_df = pd.DataFrame(
        {
            "id": row_ids,
            "cluster": labels,
        }
    )
    out_path = paths.data_processed / "oracle_clusters.parquet"
    out_df.to_parquet(out_path, index=False)

    print(f"Saved clustering models to: {model_dir}")
    print(f"Saved cluster labels to: {out_path}")
    print(f"Clusters: {cfg.n_clusters} | Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
