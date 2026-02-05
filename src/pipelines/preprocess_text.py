from __future__ import annotations

import pandas as pd

from src.config.settings import paths
from src.utils.io import ensure_dir
from src.preprocessing.type_parser import parse_type_line
from src.preprocessing.text_normalization import (
    normalize_oracle_text,
    TextNormalizationOptions,
)
from src.preprocessing.oracle_tokenizer import tokenize_oracle_text, TokenizeOptions


def main() -> None:
    ensure_dir(paths.data_processed)

    in_path = paths.data_processed / "cards_core.parquet"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing {in_path}. Run: python -m src.pipelines.build_dataset"
        )

    df = pd.read_parquet(in_path)
    print(f"Loaded: {in_path} (rows={len(df)}, cols={len(df.columns)})")

    # ---- Type features ----
    parsed = df["type_line"].apply(parse_type_line)

    df_types = pd.DataFrame(
        {
            "id": df["id"],
            "oracle_id": df.get("oracle_id"),
            "name": df["name"],
            "type_line": df["type_line"],
            "basic_types": parsed.apply(lambda p: p.basic_types),
            "super_types": parsed.apply(lambda p: p.super_types),
            "sub_types": parsed.apply(lambda p: p.sub_types),
            "type_faces": parsed.apply(lambda p: p.faces),
        }
    )

    type_out = paths.data_processed / "type_features.parquet"
    df_types.to_parquet(type_out, index=False)
    print(f"Saved: {type_out}")

    # ---- Text normalization + tokenization ----
    norm_opts = TextNormalizationOptions(
        strip_reminder_text=False,  # set True later if you want
        replace_card_name=True,
    )
    tok_opts = TokenizeOptions(
        keep_newlines=True,
        lowercase=True,
    )

    # Normalize
    df["oracle_text_norm"] = df.apply(
        lambda r: normalize_oracle_text(r.get("oracle_text"), r.get("name"), norm_opts),
        axis=1,
    )

    # Tokenize
    df_tokens = pd.DataFrame(
        {
            "id": df["id"],
            "oracle_id": df.get("oracle_id"),
            "name": df["name"],
            "oracle_text_norm": df["oracle_text_norm"],
            "oracle_tokens": df["oracle_text_norm"].apply(
                lambda t: tokenize_oracle_text(t, tok_opts)
            ),
        }
    )

    tok_out = paths.data_processed / "oracle_tokens.parquet"
    df_tokens.to_parquet(tok_out, index=False)
    print(f"Saved: {tok_out}")

    # Optional: save an enriched unified dataset for downstream pipelines
    df_enriched = df.merge(
        df_types[["id", "basic_types", "super_types", "sub_types"]],
        on="id",
        how="left",
    )
    enriched_out = paths.data_processed / "cards_enriched.parquet"
    df_enriched.to_parquet(enriched_out, index=False)
    print(f"Saved: {enriched_out}")

    print("Done.")


if __name__ == "__main__":
    main()
