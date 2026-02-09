from __future__ import annotations

import pandas as pd

from src.config.settings import paths
from src.features.embedding_features import load_tfidf_artifacts
from src.models.similarity import fit_nn_index, query_neighbors


def main() -> None:
    # Change this to any card name you want to test
    query_name = "Lightning Bolt"
    k = 10

    df = pd.read_parquet(paths.data_processed / "cards_enriched.parquet")
    vec_dir = paths.artifacts_vectorizers / "tfidf_oracle_v1"
    vec, X, row_ids = load_tfidf_artifacts(vec_dir)

    # Find row index by name (first match)
    matches = df.index[df["name"].fillna("").str.lower() == query_name.lower()].tolist()
    if not matches:
        raise ValueError(f"No card found with name '{query_name}' in cards_enriched.parquet")
    qi = matches[0]

    nn = fit_nn_index(X, metric="cosine")
    res = query_neighbors(nn, X, query_index=qi, k=k, include_self=False)

    print(f"\nQuery: {df.loc[qi, 'name']} — {df.loc[qi, 'type_line']}\n")
    for idx, score in zip(res.neighbor_indices, res.neighbor_scores):
        print(f"{score:0.3f} | {df.loc[idx, 'name']} — {df.loc[idx, 'type_line']}")


if __name__ == "__main__":
    main()
