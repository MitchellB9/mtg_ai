from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

from src.config.settings import paths
from src.features.embedding_features import load_tfidf_artifacts


VECTOR_DIR = paths.artifacts_vectorizers / "tfidf_oracle_v1"
CLUSTERS_PATH = paths.data_processed / "oracle_clusters.parquet"

MAX_POINTS = 20_000
RANDOM_STATE = 42


def validate_inputs(clusters: pd.DataFrame, row_ids) -> None:
    """
    Input:
        oracle_clusters DataFrame and TF-IDF row ids.

    Logic:
        Confirms cluster labels can be safely mapped to vector rows by id.

    Output:
        Raises ValueError if cluster data is missing required fields or ids.
    """
    required_cols = ["id", "cluster"]

    missing = [col for col in required_cols if col not in clusters.columns]
    if missing:
        raise ValueError(f"oracle_clusters is missing required columns: {missing}")

    if clusters["id"].duplicated().any():
        raise ValueError("oracle_clusters contains duplicate id values.")

    cluster_ids = set(clusters["id"])
    missing_ids = [card_id for card_id in row_ids if card_id not in cluster_ids]

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids):,} vector row ids are missing cluster labels. "
            f"Examples: {missing_ids[:5]}"
        )


def build_plot_dataframe(
    matrix,
    row_ids,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Input:
        TF-IDF matrix, vector row ids, and oracle cluster labels.

    Logic:
        Projects the TF-IDF matrix into two SVD dimensions and aligns each point
        to its cluster label by card id.

    Output:
        DataFrame with x, y, id, and cluster columns.
    """
    cluster_lookup = clusters.set_index("id")["cluster"]

    labels = cluster_lookup.loc[row_ids].to_numpy()

    svd2 = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    matrix_2d = svd2.fit_transform(matrix)

    return pd.DataFrame(
        {
            "id": row_ids,
            "x": matrix_2d[:, 0],
            "y": matrix_2d[:, 1],
            "cluster": labels,
        }
    )


def downsample_plot_data(df_plot: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        Full plotting DataFrame.

    Logic:
        Downsamples large plots for faster rendering while preserving randomness.

    Output:
        Plotting DataFrame with at most MAX_POINTS rows.
    """
    if len(df_plot) <= MAX_POINTS:
        return df_plot

    return df_plot.sample(n=MAX_POINTS, random_state=RANDOM_STATE)


def plot_clusters(df_plot: pd.DataFrame) -> None:
    """
    Input:
        Plotting DataFrame with x, y, and cluster columns.

    Logic:
        Draws a simple 2D cluster scatter plot.

    Output:
        Matplotlib plot window.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(
        df_plot["x"],
        df_plot["y"],
        c=df_plot["cluster"],
        s=6,
        alpha=0.7,
    )
    plt.title("MTG Oracle Text Clusters (TF-IDF → SVD2)")
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.show()


def main() -> None:
    """
    Input:
        artifacts/vectorizers/tfidf_oracle_v1/
        data/processed/oracle_clusters.parquet

    Logic:
        Loads TF-IDF vectors, projects them to 2D with SVD, maps cluster labels
        by card id, and displays a scatter plot.

    Output:
        Console/interactive Matplotlib cluster visualization.
    """
    if not VECTOR_DIR.exists():
        raise FileNotFoundError(
            f"Missing {VECTOR_DIR}. Run: python -m src.data_processing.build_vectors"
        )

    if not CLUSTERS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CLUSTERS_PATH}. Run: python -m src.data_processing.cluster_cards"
        )

    _, matrix, row_ids = load_tfidf_artifacts(VECTOR_DIR)
    clusters = pd.read_parquet(CLUSTERS_PATH)

    validate_inputs(clusters, row_ids)

    df_plot = build_plot_dataframe(matrix, row_ids, clusters)
    df_plot = downsample_plot_data(df_plot)

    plot_clusters(df_plot)

    # TODO Phase 4:
    # Experiment with richer cluster visualization methods:
    #   - compare SVD, UMAP, t-SNE, and graph layouts for 2D projections
    #   - build interactive plots with hoverable card names and labels
    #   - color points by label, color identity, card type, or cluster quality
    #   - save plot-ready parquet outputs for notebook/dashboard inspection
    #   - visualize relationships between clusters as a similarity graph


if __name__ == "__main__":
    main()
