from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

from src.config.settings import paths
from src.features.embedding_features import load_tfidf_artifacts


def main() -> None:
    vec_dir = paths.artifacts_vectorizers / "tfidf_oracle_v1"
    clusters_path = paths.data_processed / "oracle_clusters.parquet"

    if not vec_dir.exists():
        raise FileNotFoundError(
            f"Missing {vec_dir}. Run: python -m src.pipelines.build_vectors"
        )

    if not clusters_path.exists():
        raise FileNotFoundError(
            f"Missing {clusters_path}. Run: python -m src.pipelines.cluster_cards"
        )

    _, X, row_ids = load_tfidf_artifacts(vec_dir)
    clusters = pd.read_parquet(clusters_path)

    # Map row index -> cluster label
    cluster_map = dict(zip(clusters["id"].astype(str), clusters["cluster"].astype(int)))
    labels = [cluster_map.get(str(cid), -1) for cid in row_ids]

    # 2D projection
    svd2 = TruncatedSVD(n_components=2, random_state=42)
    X2 = svd2.fit_transform(X)

    df_plot = pd.DataFrame(
        {
            "x": X2[:, 0],
            "y": X2[:, 1],
            "cluster": labels,
        }
    )

    # Downsample for speed if huge (optional)
    max_points = 20000
    if len(df_plot) > max_points:
        df_plot = df_plot.sample(n=max_points, random_state=42)

    plt.figure(figsize=(10, 7))
    plt.scatter(df_plot["x"], df_plot["y"], c=df_plot["cluster"], s=6, alpha=0.7)
    plt.title("MTG Oracle Text Clusters (TF-IDF → SVD2)")
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.show()


if __name__ == "__main__":
    main()
