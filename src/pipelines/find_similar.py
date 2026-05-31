from __future__ import annotations

import pandas as pd

from src.config.settings import paths
from src.features.embedding_features import load_tfidf_artifacts
from src.models.similarity import fit_nn_index, query_neighbors


CARDS_PATH = paths.data_processed / "cards_enriched.parquet"
VECTOR_DIR = paths.artifacts_vectorizers / "tfidf_oracle_v1"


def validate_inputs(cards: pd.DataFrame, row_ids) -> None:
    """
    Input:
        cards_enriched DataFrame and TF-IDF row ids.

    Logic:
        Confirms required card columns exist and that every vector row id can
        be matched to card metadata.

    Output:
        Raises ValueError if card metadata and vector ids cannot be aligned.
    """
    required_cols = ["id", "name", "type_line"]

    missing = [col for col in required_cols if col not in cards.columns]
    if missing:
        raise ValueError(f"cards_enriched is missing required columns: {missing}")

    if cards["id"].duplicated().any():
        raise ValueError("cards_enriched contains duplicate id values.")

    card_ids = set(cards["id"])
    missing_ids = [card_id for card_id in row_ids if card_id not in card_ids]

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids):,} vector row ids are missing from cards_enriched. "
            f"Examples: {missing_ids[:5]}"
        )


def align_cards_to_vector_rows(
    cards: pd.DataFrame,
    row_ids,
) -> pd.DataFrame:
    """
    Input:
        cards_enriched DataFrame and TF-IDF row ids.

    Logic:
        Reorders card metadata to exactly match the TF-IDF matrix row order.

    Output:
        DataFrame indexed from 0..n where row number matches matrix row number.
    """
    card_lookup = cards.set_index("id", drop=False)

    aligned = card_lookup.loc[row_ids].reset_index(drop=True)

    return aligned


def find_card_index_by_name(cards: pd.DataFrame, query_name: str) -> int:
    """
    Input:
        Vector-aligned cards DataFrame and card name.

    Logic:
        Finds the first exact case-insensitive name match.

    Output:
        Matrix row index for the matched card.
    """
    matches = cards.index[
        cards["name"].fillna("").str.lower() == query_name.lower()
    ].tolist()

    if not matches:
        raise ValueError(f"No card found with name '{query_name}'.")

    return matches[0]


def print_neighbor_results(
    cards: pd.DataFrame,
    query_index: int,
    neighbor_indices,
    neighbor_scores,
) -> None:
    """
    Input:
        Vector-aligned card metadata, query index, neighbor indices, and scores.

    Logic:
        Prints readable nearest-neighbor results.

    Output:
        Console output only.
    """
    print(
        f"\nQuery: {cards.loc[query_index, 'name']} — "
        f"{cards.loc[query_index, 'type_line']}\n"
    )

    for idx, score in zip(neighbor_indices, neighbor_scores):
        print(
            f"{score:0.3f} | "
            f"{cards.loc[idx, 'name']} — {cards.loc[idx, 'type_line']}"
        )


def main() -> None:
    """
    Input:
        data/processed/cards_enriched.parquet
        artifacts/vectorizers/tfidf_oracle_v1/

    Logic:
        Loads card metadata and TF-IDF vectors, aligns metadata to vector row ids,
        finds one query card by name, and prints nearest cards by cosine similarity.

    Output:
        Console similarity results.
    """
    query_name = "Lightning Bolt"
    k = 10

    if not CARDS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CARDS_PATH}. Run: python -m src.data_processing.preprocess_text"
        )

    if not VECTOR_DIR.exists():
        raise FileNotFoundError(
            f"Missing {VECTOR_DIR}. Run: python -m src.data_processing.build_vectors"
        )

    cards = pd.read_parquet(CARDS_PATH)
    vectorizer, matrix, row_ids = load_tfidf_artifacts(VECTOR_DIR)

    validate_inputs(cards, row_ids)

    cards_aligned = align_cards_to_vector_rows(cards, row_ids)

    query_index = find_card_index_by_name(cards_aligned, query_name)

    nn = fit_nn_index(matrix, metric="cosine")
    result = query_neighbors(
        nn,
        matrix,
        query_index=query_index,
        k=k,
        include_self=False,
    )

    print_neighbor_results(
        cards=cards_aligned,
        query_index=query_index,
        neighbor_indices=result.neighbor_indices,
        neighbor_scores=result.neighbor_scores,
    )

    # TODO Phase 4:
    # Turn this script into a richer similarity exploration tool:
    #   - accept command-line card names and k values
    #   - compare similarity across TF-IDF, embeddings, and hybrid features
    #   - save similarity results as parquet for inspection notebooks
    #   - support similarity modes for rules text, deck role, and theme/flavor
    #   - expose similarity search through a small app or agent tool


if __name__ == "__main__":
    main()
