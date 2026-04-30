
from __future__ import annotations

import json
import pandas as pd

from src.config.settings import paths
from src.labeling.rule_labeler import label_card


def main() -> None:
    cards_path = paths.data_processed / "cards_enriched.parquet"
    clusters_path = paths.data_processed / "oracle_clusters.parquet"

    if not cards_path.exists():
        raise FileNotFoundError(f"Missing {cards_path}. Run preprocess pipeline first.")

    cards = pd.read_parquet(cards_path)

    if clusters_path.exists():
        clusters = pd.read_parquet(clusters_path)
        cards = cards.merge(clusters, on="id", how="left")
    else:
        cards["cluster"] = None

    rows = []

    for _, row in cards.iterrows():
        row_dict = row.to_dict()
        matches = label_card(row_dict)

        labels = [m.label for m in matches]
        confidences = {m.label: m.confidence for m in matches}
        sources = {m.label: m.source for m in matches}
        reasons = {m.label: m.reason for m in matches}

        primary_label = labels[0] if labels else "needs_review"
        max_confidence = max(confidences.values()) if confidences else 0.0
        needs_review = "needs_review" in labels or max_confidence < 0.60

        rows.append({
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
        })

    labels_df = pd.DataFrame(rows)

    out_path = paths.data_processed / "card_labels.parquet"
    labels_df.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(labels_df)}")
    print("\nTop primary labels:")
    print(labels_df["primary_label"].value_counts().head(25).to_string())
    print(f"\nNeeds review: {labels_df['needs_review'].mean():.2%}")


if __name__ == "__main__":
    main()
