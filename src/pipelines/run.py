from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.config.settings import paths


PIPELINE_STEPS = [
    ("Build core dataset", "src.pipelines.build_dataset"),
    ("Preprocess text and types", "src.pipelines.preprocess_text"),
    ("Build TF-IDF vectors", "src.pipelines.build_vectors"),
    ("Cluster cards", "src.pipelines.cluster_cards"),
    ("Report clusters", "src.pipelines.report_clusters"),
    ("Label cards", "src.pipelines.label_cards"),
]


EXPECTED_OUTPUTS = [
    paths.data_processed / "cards_core.parquet",
    paths.data_processed / "cards_enriched.parquet",
    paths.data_processed / "type_features.parquet",
    paths.data_processed / "oracle_tokens.parquet",
    paths.data_processed / "oracle_clusters.parquet",
    paths.data_processed / "cluster_report.parquet",
    paths.data_processed / "cluster_representatives.parquet",
    paths.data_processed / "card_labels.parquet",
    paths.artifacts_vectorizers / "tfidf_oracle_v1" / "tfidf_vectorizer.joblib",
    paths.artifacts_vectorizers / "tfidf_oracle_v1" / "tfidf_matrix.npz",
    paths.artifacts_vectorizers / "tfidf_oracle_v1" / "row_ids.joblib",
    paths.artifacts_models / "svd_kmeans_tfidf_v1" / "svd.joblib",
    paths.artifacts_models / "svd_kmeans_tfidf_v1" / "kmeans.joblib",
]


def run_step(step_name: str, module_name: str) -> None:
    print("\n" + "=" * 90)
    print(f"Running: {step_name}")
    print("=" * 90)

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=paths.root,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline step failed: {step_name}")


def validate_outputs() -> None:
    print("\n" + "=" * 90)
    print("Validating MVP outputs")
    print("=" * 90)

    missing = [path for path in EXPECTED_OUTPUTS if not path.exists()]

    if missing:
        print("\nMissing outputs:")
        for path in missing:
            print(f"- {path}")
        raise FileNotFoundError("MVP pipeline completed, but expected outputs are missing.")

    print("All expected MVP outputs exist.")


def print_summary() -> None:
    import pandas as pd

    labels_path = paths.data_processed / "card_labels.parquet"
    clusters_path = paths.data_processed / "oracle_clusters.parquet"

    labels = pd.read_parquet(labels_path)
    clusters = pd.read_parquet(clusters_path)

    print("\n" + "=" * 90)
    print("MVP Summary")
    print("=" * 90)

    print(f"Cards labeled: {len(labels):,}")
    print(f"Unique primary labels: {labels['primary_label'].nunique():,}")
    print(f"Clusters generated: {clusters['cluster'].nunique():,}")

    if "needs_review" in labels.columns:
        print(f"Needs review: {labels['needs_review'].mean():.2%}")

    print("\nTop primary labels:")
    print(labels["primary_label"].value_counts().head(20).to_string())


def main() -> None:
    for step_name, module_name in PIPELINE_STEPS:
        run_step(step_name, module_name)

    validate_outputs()
    print_summary()

    print("\nMVP pipeline complete.")


if __name__ == "__main__":
    main()
