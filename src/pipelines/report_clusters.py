from __future__ import annotations

from collections import Counter
import json

import joblib
import numpy as np
import pandas as pd

from src.config.settings import paths
from src.features.embedding_features import load_tfidf_artifacts


CARDS_PATH = paths.data_processed / "cards_enriched.parquet"
CLUSTERS_PATH = paths.data_processed / "oracle_clusters.parquet"

VECTORIZER_DIR = paths.artifacts_vectorizers / "tfidf_oracle_v1"
MODEL_DIR = paths.artifacts_models / "svd_kmeans_tfidf_v1"

CLUSTER_REPORT_OUT = paths.data_processed / "cluster_report.parquet"
CLUSTER_REPRESENTATIVES_OUT = paths.data_processed / "cluster_representatives.parquet"


def validate_inputs(
    cards: pd.DataFrame,
    clusters: pd.DataFrame,
    matrix,
    row_ids,
) -> None:
    """
    Input:
        Card metadata, cluster labels, TF-IDF matrix, and vector row ids.

    Logic:
        Confirms required columns exist and vector rows align with row ids.

    Output:
        Raises ValueError if required inputs are missing or inconsistent.
    """
    required_card_cols = [
        "id",
        "name",
        "type_line",
        "cmc",
        "oracle_text_norm",
        "types",
        "supertypes",
        "subtypes",
    ]

    missing_cards = [col for col in required_card_cols if col not in cards.columns]
    if missing_cards:
        raise ValueError(f"cards_enriched is missing columns: {missing_cards}")

    required_cluster_cols = ["id", "cluster"]
    missing_clusters = [col for col in required_cluster_cols if col not in clusters.columns]
    if missing_clusters:
        raise ValueError(f"oracle_clusters is missing columns: {missing_clusters}")

    if matrix.shape[0] != len(row_ids):
        raise ValueError(
            f"Vector row mismatch: matrix rows={matrix.shape[0]}, ids={len(row_ids)}"
        )


def top_items(series: pd.Series, top_n: int = 10) -> list[tuple[str, int]]:
    """
    Input:
        Series containing list-like values.

    Logic:
        Counts repeated values across all lists in the series.

    Output:
        Top item/count pairs.
    """
    counter = Counter()

    for value in series:
        if isinstance(value, list):
            counter.update([str(item) for item in value if str(item).strip()])

    return counter.most_common(top_n)


def representative_cards(
    cluster_df: pd.DataFrame,
    cluster_center: np.ndarray,
    svd_matrix_cluster: np.ndarray,
    n: int = 10,
) -> pd.DataFrame:
    """
    Input:
        Cards in one cluster, that cluster's centroid, and reduced SVD vectors.

    Logic:
        Finds cards closest to the cluster centroid in reduced vector space.

    Output:
        Representative cards with centroid_distance added.
    """
    if len(cluster_df) == 0:
        return cluster_df.head(0)

    distances = np.linalg.norm(svd_matrix_cluster - cluster_center, axis=1)
    order = np.argsort(distances)[:n]

    reps = cluster_df.iloc[order].copy()
    reps["centroid_distance"] = distances[order]

    return reps


def top_cluster_terms(
    matrix_cluster,
    feature_names: np.ndarray,
    top_n: int = 15,
) -> list[tuple[str, float]]:
    """
    Input:
        TF-IDF rows for a cluster and vectorizer feature names.

    Logic:
        Computes the mean TF-IDF weight per term within the cluster.

    Output:
        Highest-weighted cluster terms.
    """
    if matrix_cluster.shape[0] == 0:
        return []

    mean_scores = np.asarray(matrix_cluster.mean(axis=0)).ravel()

    if mean_scores.size == 0:
        return []

    top_idx = np.argsort(mean_scores)[::-1][:top_n]

    return [
        (str(feature_names[i]), float(mean_scores[i]))
        for i in top_idx
        if mean_scores[i] > 0
    ]


def build_cluster_input(
    cards: pd.DataFrame,
    clusters: pd.DataFrame,
    row_ids,
) -> pd.DataFrame:
    """
    Input:
        cards_enriched, oracle_clusters, and TF-IDF artifact row ids.

    Logic:
        Joins card metadata to cluster labels and maps each card id back to
        its matrix row.

    Output:
        Cluster analysis DataFrame sorted by matrix_row.
    """
    row_map = pd.DataFrame(
        {
            "matrix_row": np.arange(len(row_ids)),
            "id": row_ids,
        }
    )

    df = cards.merge(clusters, on="id", how="left")
    df = df.merge(row_map, on="id", how="inner")

    return df.sort_values("matrix_row").reset_index(drop=True)


def build_cluster_reports(
    df: pd.DataFrame,
    matrix,
    reduced_matrix: np.ndarray,
    feature_names: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input:
        Cluster analysis DataFrame, TF-IDF matrix, reduced matrix, and terms.

    Logic:
        Builds one summary row per cluster and one representative-card table.

    Output:
        cluster_report DataFrame and cluster_representatives DataFrame.
    """
    cluster_reports: list[dict] = []
    representative_rows: list[dict] = []

    for cluster_id in sorted(df["cluster"].dropna().unique()):
        cluster_id = int(cluster_id)

        cluster_df = df[df["cluster"] == cluster_id].copy()
        cluster_rows = cluster_df["matrix_row"].to_numpy()

        matrix_cluster = matrix[cluster_rows]
        reduced_cluster = reduced_matrix[cluster_rows]
        cluster_center = reduced_cluster.mean(axis=0)

        top_types = top_items(cluster_df["types"], top_n=8)
        top_supertypes = top_items(cluster_df["supertypes"], top_n=8)
        top_subtypes = top_items(cluster_df["subtypes"], top_n=12)
        top_terms = top_cluster_terms(matrix_cluster, feature_names, top_n=15)

        reps = representative_cards(
            cluster_df=cluster_df,
            cluster_center=cluster_center,
            svd_matrix_cluster=reduced_cluster,
            n=10,
        )

        rep_cards = reps[
            [
                "name",
                "type_line",
                "cmc",
                "oracle_text_norm",
                "centroid_distance",
            ]
        ].to_dict("records")

        cluster_reports.append(
            {
                "cluster": cluster_id,
                "cluster_size": int(len(cluster_df)),
                "avg_cmc": (
                    float(cluster_df["cmc"].dropna().mean())
                    if cluster_df["cmc"].notna().any()
                    else None
                ),
                "median_cmc": (
                    float(cluster_df["cmc"].dropna().median())
                    if cluster_df["cmc"].notna().any()
                    else None
                ),
                "top_types": json.dumps(top_types),
                "top_supertypes": json.dumps(top_supertypes),
                "top_subtypes": json.dumps(top_subtypes),
                "top_terms": json.dumps(top_terms),
                "representative_cards": json.dumps(rep_cards),
            }
        )

        for rank, (_, row) in enumerate(reps.iterrows(), start=1):
            representative_rows.append(
                {
                    "cluster": cluster_id,
                    "representative_rank": rank,
                    "id": row["id"],
                    "name": row["name"],
                    "type_line": row.get("type_line"),
                    "cmc": row.get("cmc"),
                    "oracle_text_norm": row.get("oracle_text_norm"),
                    "centroid_distance": row.get("centroid_distance"),
                }
            )

    report_df = pd.DataFrame(cluster_reports).sort_values(
        ["cluster_size", "cluster"],
        ascending=[False, True],
    ).reset_index(drop=True)

    representatives_df = pd.DataFrame(representative_rows).sort_values(
        ["cluster", "representative_rank"],
    ).reset_index(drop=True)

    return report_df, representatives_df


def main() -> None:
    """
    Input:
        data/processed/cards_enriched.parquet
        data/processed/oracle_clusters.parquet
        artifacts/vectorizers/tfidf_oracle_v1/
        artifacts/models/svd_kmeans_tfidf_v1/

    Logic:
        Joins card metadata to cluster labels, aligns rows to the vector matrix,
        computes top terms/types per cluster, and identifies representative cards.

    Output:
        data/processed/cluster_report.parquet
        data/processed/cluster_representatives.parquet
    """
    for path in [CARDS_PATH, CLUSTERS_PATH, VECTORIZER_DIR, MODEL_DIR]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    cards = pd.read_parquet(CARDS_PATH)
    clusters = pd.read_parquet(CLUSTERS_PATH)

    vectorizer, matrix, row_ids = load_tfidf_artifacts(VECTORIZER_DIR)
    svd = joblib.load(MODEL_DIR / "svd.joblib")

    validate_inputs(cards, clusters, matrix, row_ids)

    df = build_cluster_input(cards, clusters, row_ids)

    if len(df) != matrix.shape[0]:
        raise ValueError(
            f"Row alignment mismatch: df rows={len(df)}, matrix rows={matrix.shape[0]}"
        )

    feature_names = np.array(vectorizer.get_feature_names_out())
    reduced_matrix = svd.transform(matrix)

    report_df, representatives_df = build_cluster_reports(
        df=df,
        matrix=matrix,
        reduced_matrix=reduced_matrix,
        feature_names=feature_names,
    )

    report_df.to_parquet(CLUSTER_REPORT_OUT, index=False)
    representatives_df.to_parquet(CLUSTER_REPRESENTATIVES_OUT, index=False)

    print(f"Saved: {CLUSTER_REPORT_OUT} (rows={len(report_df):,})")
    print(f"Saved: {CLUSTER_REPRESENTATIVES_OUT} (rows={len(representatives_df):,})")


if __name__ == "__main__":
    main()

    # TODO Phase 4:
    # Explore alternative cluster interpretation methods:
    #   - compare centroid-nearest representatives with medoids
    #   - generate LLM-assisted cluster names/summaries from representative cards
    #   - produce human-review cluster dashboards
    #   - compare cluster themes against rule labels and mechanic extraction
    #   - create graph-based summaries of cluster relationships
