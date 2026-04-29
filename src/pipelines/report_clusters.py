from __future__ import annotations

from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd
import joblib

from src.config.settings import paths
from src.features.embedding_features import load_tfidf_artifacts


def _safe_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if pd.notna(v)]
    return []


def _top_items(series: pd.Series, top_n: int = 10) -> list[tuple[str, int]]:
    counter = Counter()
    for value in series:
        if isinstance(value, list):
            counter.update([str(v) for v in value if str(v).strip()])
    return counter.most_common(top_n)


def _representative_cards(
    cluster_df: pd.DataFrame,
    cluster_center: np.ndarray,
    svd_matrix_cluster: np.ndarray,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the most representative cards in a cluster based on distance to centroid.
    """
    if len(cluster_df) == 0:
        return cluster_df.head(0)

    distances = np.linalg.norm(svd_matrix_cluster - cluster_center, axis=1)
    order = np.argsort(distances)[:n]

    reps = cluster_df.iloc[order].copy()
    reps["centroid_distance"] = distances[order]
    return reps


def _top_cluster_terms(
    X_cluster,
    feature_names: np.ndarray,
    top_n: int = 15,
) -> list[tuple[str, float]]:
    """
    Mean TF-IDF weight within cluster, sorted descending.
    """
    if X_cluster.shape[0] == 0:
        return []

    mean_scores = np.asarray(X_cluster.mean(axis=0)).ravel()
    if mean_scores.size == 0:
        return []

    top_idx = np.argsort(mean_scores)[::-1][:top_n]
    return [
        (str(feature_names[i]), float(mean_scores[i]))
        for i in top_idx
        if mean_scores[i] > 0
    ]


def main() -> None:
    processed_dir = paths.data_processed
    vectorizer_dir = paths.artifacts_vectorizers / "tfidf_oracle_v1"
    model_dir = paths.artifacts_models / "svd_kmeans_tfidf_v1"

    cards_path = processed_dir / "cards_enriched.parquet"
    types_path = processed_dir / "type_features.parquet"
    clusters_path = processed_dir / "oracle_clusters.parquet"

    if not cards_path.exists():
        raise FileNotFoundError(f"Missing {cards_path}")
    if not types_path.exists():
        raise FileNotFoundError(f"Missing {types_path}")
    if not clusters_path.exists():
        raise FileNotFoundError(f"Missing {clusters_path}")
    if not vectorizer_dir.exists():
        raise FileNotFoundError(f"Missing {vectorizer_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing {model_dir}")

    # Load processed data
    cards = pd.read_parquet(cards_path)
    types = pd.read_parquet(types_path)
    clusters = pd.read_parquet(clusters_path)

    # Merge card metadata
    df = (
        cards.merge(
            types[["id", "basic_types", "super_types", "sub_types"]],
            on="id",
            how="left",
            suffixes=("", "_type_features")
        )
        .merge(clusters, on="id", how="left")
    )

    # Load vector artifacts
    vectorizer, X, row_ids = load_tfidf_artifacts(vectorizer_dir)
    svd = joblib.load(model_dir / "svd.joblib")

    # Row alignment from TF-IDF matrix back to cards
    row_map = pd.DataFrame({
        "matrix_row": np.arange(len(row_ids)),
        "id": row_ids,
    })

    df = df.merge(row_map, on="id", how="inner").sort_values("matrix_row").reset_index(drop=True)

    if len(df) != X.shape[0]:
        raise ValueError(
            f"Row alignment mismatch: merged df has {len(df)} rows but TF-IDF matrix has {X.shape[0]} rows."
        )

    feature_names = np.array(vectorizer.get_feature_names_out())
    X_reduced = svd.transform(X)

    cluster_reports: list[dict] = []
    representative_rows: list[dict] = []

    for cluster_id in sorted(df["cluster"].dropna().unique()):
        cluster_id = int(cluster_id)
        cluster_df = df[df["cluster"] == cluster_id].copy()
        cluster_rows = cluster_df["matrix_row"].to_numpy()

        X_cluster = X[cluster_rows]
        X_reduced_cluster = X_reduced[cluster_rows]

        cluster_center = X_reduced_cluster.mean(axis=0)

        basic_top = _top_items(cluster_df["basic_types"], top_n=8)
        super_top = _top_items(cluster_df["super_types"], top_n=8)
        sub_top = _top_items(cluster_df["sub_types"], top_n=12)
        term_top = _top_cluster_terms(X_cluster, feature_names, top_n=15)

        reps = _representative_cards(
            cluster_df=cluster_df,
            cluster_center=cluster_center,
            svd_matrix_cluster=X_reduced_cluster,
            n=10,
        )

        rep_cards = reps[["name", "type_line", "cmc", "oracle_text_norm", "centroid_distance"]].to_dict("records")

        cluster_reports.append({
            "cluster": cluster_id,
            "cluster_size": int(len(cluster_df)),
            "avg_cmc": float(cluster_df["cmc"].dropna().mean()) if cluster_df["cmc"].notna().any() else None,
            "median_cmc": float(cluster_df["cmc"].dropna().median()) if cluster_df["cmc"].notna().any() else None,
            "top_basic_types": json.dumps(basic_top),
            "top_super_types": json.dumps(super_top),
            "top_sub_types": json.dumps(sub_top),
            "top_terms": json.dumps(term_top),
            "representative_cards": json.dumps(rep_cards),
        })

        for rank, (_, row) in enumerate(reps.iterrows(), start=1):
            representative_rows.append({
                "cluster": cluster_id,
                "representative_rank": rank,
                "id": row["id"],
                "name": row["name"],
                "type_line": row.get("type_line"),
                "cmc": row.get("cmc"),
                "oracle_text_norm": row.get("oracle_text_norm"),
                "centroid_distance": row.get("centroid_distance"),
            })

    report_df = pd.DataFrame(cluster_reports).sort_values(
        ["cluster_size", "cluster"], ascending=[False, True]
    ).reset_index(drop=True)

    representatives_df = pd.DataFrame(representative_rows).sort_values(
        ["cluster", "representative_rank"]
    ).reset_index(drop=True)

    report_out = processed_dir / "cluster_report.parquet"
    reps_out = processed_dir / "cluster_representatives.parquet"

    report_df.to_parquet(report_out, index=False)
    representatives_df.to_parquet(reps_out, index=False)

    print(f"Saved: {report_out}")
    print(f"Saved: {reps_out}")

    # TODO: Don't like this output here. Save for notebook
    print("\nTop 10 clusters by size:\n")
    print(report_df[["cluster", "cluster_size", "avg_cmc", "top_basic_types", "top_terms"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
