from __future__ import annotations

import json

import pandas as pd

from src.config.settings import paths
from src.labeling.rule_labeler import label_card


CARDS_PATH = paths.data_processed / "cards_enriched.parquet"
CLUSTERS_PATH = paths.data_processed / "oracle_clusters.parquet"
LABELS_OUT = paths.data_processed / "card_labels.parquet"

REVIEW_CONFIDENCE_THRESHOLD = 0.60


def validate_cards_input(cards: pd.DataFrame) -> None:
    """
    Input:
        cards_enriched DataFrame.

    Logic:
        Confirms required fields exist before rule labeling.

    Output:
        Raises ValueError if required card fields are missing.
    """
    required_cols = ["id", "oracle_id", "name", "type_line", "cmc", "oracle_text_norm"]

    missing = [col for col in required_cols if col not in cards.columns]

    if missing:
        raise ValueError(f"cards_enriched is missing required columns: {missing}")


def attach_clusters(cards: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        cards_enriched DataFrame.

    Logic:
        Adds cluster labels when oracle_clusters.parquet exists.
        If clustering has not been run, preserves the labeling flow with null clusters.

    Output:
        Copy of cards DataFrame with a cluster column.
    """
    cards = cards.copy()

    if CLUSTERS_PATH.exists():
        clusters = pd.read_parquet(CLUSTERS_PATH)

        if not {"id", "cluster"}.issubset(clusters.columns):
            raise ValueError("oracle_clusters.parquet must contain id and cluster columns.")

        return cards.merge(clusters, on="id", how="left")

    cards["cluster"] = None
    return cards


def build_label_row(row: pd.Series) -> dict:
    """
    Input:
        One card row from cards_enriched, optionally with cluster.

    Logic:
        Applies rule-based labels and converts match metadata into output fields.

    Output:
        Dictionary representing one row for card_labels.parquet.
    """
    row_dict = row.to_dict()
    matches = label_card(row_dict)

    labels = [match.label for match in matches]
    confidences = {match.label: match.confidence for match in matches}
    sources = {match.label: match.source for match in matches}
    reasons = {match.label: match.reason for match in matches}

    primary_label = labels[0] if labels else "needs_review"
    max_confidence = max(confidences.values()) if confidences else 0.0

    needs_review = (
        primary_label == "needs_review"
        or max_confidence < REVIEW_CONFIDENCE_THRESHOLD
    )

    return {
        "id": row_dict.get("id"),
        "oracle_id": row_dict.get("oracle_id"),
        "name": row_dict.get("name"),
        "type_line": row_dict.get("type_line"),
        "cmc": row_dict.get("cmc"),
        "cluster": row_dict.get("cluster"),
        "primary_label": primary_label,
        "labels": labels,
        "label_confidences": json.dumps(confidences),
        "label_sources": json.dumps(sources),
        "label_reasons": json.dumps(reasons),
        "max_confidence": max_confidence,
        "needs_review": needs_review,
    }


def build_card_labels(cards: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        cards_enriched DataFrame with optional cluster column.

    Logic:
        Applies rule-based labeling to every card.

    Output:
        card_labels DataFrame.
    """
    rows = [build_label_row(row) for _, row in cards.iterrows()]
    return pd.DataFrame(rows)


def main() -> None:
    """
    Input:
        data/processed/cards_enriched.parquet
        data/processed/oracle_clusters.parquet, when available

    Logic:
        Applies rule-based semantic labels to each card and marks low-confidence
        or unlabeled cards for review.

    Output:
        data/processed/card_labels.parquet
    """
    if not CARDS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CARDS_PATH}. Run: python -m src.data_processing.preprocess_text"
        )

    cards = pd.read_parquet(CARDS_PATH)
    print(f"Loaded: {CARDS_PATH} (rows={len(cards):,}, cols={len(cards.columns):,})")

    validate_cards_input(cards)

    cards = attach_clusters(cards)

    labels_df = build_card_labels(cards)
    labels_df.to_parquet(LABELS_OUT, index=False)

    print(f"Saved: {LABELS_OUT} (rows={len(labels_df):,})")
    print(f"Needs review: {labels_df['needs_review'].mean():.2%}")


if __name__ == "__main__":
    main()

    # TODO Phase 4:
    # Experiment with alternative labeling approaches beyond rule-based matching:
    #   - weak supervision using multiple heuristic label sources
    #   - supervised mechanic/archetype classifiers trained from reviewed labels
    #   - LLM-assisted label suggestions with human approval
    #   - multi-label confidence calibration
    #   - compare rule labels against cluster themes and card similarity groups
