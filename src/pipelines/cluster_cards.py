from __future__ import annotations

import joblib
import pandas as pd

from src.config.settings import paths
from src.utils.io import ensure_dir
from src.features.embedding_features import load_tfidf_artifacts
from src.models.clustering import cluster_with_svd_kmeans, ClusterConfig


VECTOR_DIR = paths.artifacts_vectorizers / "tfidf_oracle_v1"
MODEL_DIR = paths.artifacts_models / "svd_kmeans_tfidf_v1"
CLUSTERS_OUT = paths.data_processed / "oracle_clusters.parquet"


def validate_vector_artifacts(matrix, row_ids) -> None:
    """
    Input:
        TF-IDF matrix and row id array.

    Logic:
        Confirms the vector matrix and row ids align before clustering.

    Output:
        Raises ValueError if artifact shapes are inconsistent.
    """
    if matrix.shape[0] != len(row_ids):
        raise ValueError(
            f"Vector row mismatch: matrix rows={matrix.shape[0]}, ids={len(row_ids)}"
        )

    if matrix.shape[0] == 0:
        raise ValueError("Cannot cluster an empty vector matrix.")


def build_cluster_labels(row_ids, labels) -> pd.DataFrame:
    """
    Input:
        Card ids from the vector artifact and cluster labels from KMeans.

    Logic:
        Creates a simple id-to-cluster mapping for downstream joins.

    Output:
        DataFrame with id and cluster columns.
    """
    return pd.DataFrame(
        {
            "id": row_ids,
            "cluster": labels,
        }
    )


def save_cluster_models(svd, kmeans) -> None:
    """
    Input:
        Trained SVD reducer and KMeans model.

    Logic:
        Saves fitted clustering artifacts for reuse and inspection.

    Output:
        svd.joblib and kmeans.joblib in the model artifact directory.
    """
    ensure_dir(MODEL_DIR)

    joblib.dump(svd, MODEL_DIR / "svd.joblib", compress=3)
    joblib.dump(kmeans, MODEL_DIR / "kmeans.joblib", compress=3)


def main() -> None:
    """
    Input:
        artifacts/vectorizers/tfidf_oracle_v1/

    Logic:
        Loads TF-IDF vectors, reduces dimensionality with SVD, clusters cards
        with KMeans, and saves both the fitted models and card cluster labels.

    Output:
        artifacts/models/svd_kmeans_tfidf_v1/
            svd.joblib
            kmeans.joblib

        data/processed/oracle_clusters.parquet
    """
    ensure_dir(paths.artifacts_models)
    ensure_dir(paths.data_processed)

    if not VECTOR_DIR.exists():
        raise FileNotFoundError(
            f"Missing {VECTOR_DIR}. Run: python -m src.data_processing.build_vectors"
        )

    vectorizer, matrix, row_ids = load_tfidf_artifacts(VECTOR_DIR)
    validate_vector_artifacts(matrix, row_ids)

    cfg = ClusterConfig(
        n_clusters=200,
        n_components=256,
        random_state=42,
        n_init=10,
    )

    print("Clustering config:")
    print(f"  n_clusters={cfg.n_clusters}")
    print(f"  n_components={cfg.n_components}")
    print(f"  random_state={cfg.random_state}")
    print(f"  n_init={cfg.n_init}")

    labels, svd, kmeans = cluster_with_svd_kmeans(matrix, cfg)

    save_cluster_models(svd, kmeans)

    out_df = build_cluster_labels(row_ids, labels)
    out_df.to_parquet(CLUSTERS_OUT, index=False)

    print(f"Saved clustering models to: {MODEL_DIR}")
    print(f"Saved cluster labels to: {CLUSTERS_OUT}")
    print(f"Clusters: {cfg.n_clusters} | Rows: {len(out_df):,}")

    # TODO Phase 4:
    # Experiment with additional clustering approaches and evaluation methods:
    #   - compare KMeans cluster counts with silhouette/inertia diagnostics
    #   - evaluate MiniBatchKMeans for faster iteration
    #   - test HDBSCAN or agglomerative clustering for non-spherical groups
    #   - compare clustering on TF-IDF vs SVD-reduced vs future embeddings
    #   - add cluster quality reports with representative cards and top terms
    #   - version clustering outputs by config instead of fixed folder names
