from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config.settings import paths

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)


def show_df(path: Path, name: str, n: int = 5) -> None:
    print("\n" + "=" * 90)
    print(f"{name}: {path}")
    print("=" * 90)

    df = pd.read_parquet(path)
    print(f"shape: {df.shape}")
    print("\ncolumns:")
    print(list(df.columns))

    print("\ndtypes:")
    print(df.dtypes)

    print(f"\nhead({n}):")
    print(df.head(n))


def main() -> None:
    show_df(paths.data_processed / "cards_core.parquet", "cards_core")
    show_df(paths.data_processed / "cards_enriched.parquet", "cards_enriched")
    show_df(paths.data_processed / "type_features.parquet", "type_features")
    show_df(paths.data_processed / "oracle_tokens.parquet", "oracle_tokens")

    clusters_path = paths.data_processed / "oracle_clusters.parquet"
    if clusters_path.exists():
        show_df(clusters_path, "oracle_clusters")


if __name__ == "__main__":
    main()
