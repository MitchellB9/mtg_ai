from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.io import ensure_dir

TFIDF_VECTORIZER_FILE = "tfidf_vectorizer.joblib"
TFIDF_ROW_IDS_FILE = "row_ids.joblib"
TFIDF_MATRIX_FILE = "tfidf_matrix.npz"


@dataclass(frozen=True)
class TfidfConfig:
    """
    Configuration for oracle-text TF-IDF vectorization.
    """

    min_df: int = 2
    max_df: float = 0.97
    ngram_range: tuple[int, int] = (1, 2)
    max_features: Optional[int] = 100_000
    sublinear_tf: bool = True
    lowercase: bool = True


def build_tfidf_vectorizer(cfg: TfidfConfig) -> TfidfVectorizer:
    """
    Input:
        TfidfConfig.

    Logic:
        Creates a scikit-learn TF-IDF vectorizer configured for normalized
        oracle text.

    Output:
        Unfitted TfidfVectorizer.
    """
    return TfidfVectorizer(
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        sublinear_tf=cfg.sublinear_tf,
        lowercase=cfg.lowercase,
        token_pattern=r"(?u)\b\w+\b",
    )


def prepare_texts(texts: pd.Series) -> pd.Series:
    """
    Input:
        Text Series that may contain nulls or non-string values.

    Logic:
        Converts text values into safe strings for vectorization.

    Output:
        Clean text Series.
    """
    return texts.fillna("").astype(str)


def fit_transform_tfidf(
    texts: pd.Series,
    cfg: Optional[TfidfConfig] = None,
) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """
    Input:
        Normalized oracle text and optional TF-IDF config.

    Logic:
        Fits a TF-IDF vectorizer and transforms the input text.

    Output:
        Fitted vectorizer and sparse TF-IDF matrix.
    """
    cfg = cfg or TfidfConfig()

    vectorizer = build_tfidf_vectorizer(cfg)
    matrix = vectorizer.fit_transform(prepare_texts(texts))

    return vectorizer, matrix


def transform_tfidf(
    vectorizer: TfidfVectorizer,
    texts: pd.Series,
) -> sparse.csr_matrix:
    """
    Input:
        Fitted TF-IDF vectorizer and text Series.

    Logic:
        Transforms new text using an existing vectorizer.

    Output:
        Sparse TF-IDF matrix.
    """
    return vectorizer.transform(prepare_texts(texts))


def save_tfidf_artifacts(
    out_dir: Path,
    vectorizer: TfidfVectorizer,
    matrix: sparse.csr_matrix,
    ids: np.ndarray,
) -> None:
    """
    Input:
        Output directory, fitted vectorizer, TF-IDF matrix, and row ids.

    Logic:
        Saves vectorization artifacts needed by clustering and similarity steps.

    Output:
        Vectorizer, row ids, and sparse matrix files written to disk.
    """
    ensure_dir(out_dir)

    joblib.dump(vectorizer, out_dir / TFIDF_VECTORIZER_FILE, compress=3)
    joblib.dump(ids, out_dir / TFIDF_ROW_IDS_FILE, compress=3)
    sparse.save_npz(out_dir / TFIDF_MATRIX_FILE, matrix)


def load_tfidf_artifacts(
    in_dir: Path,
) -> tuple[TfidfVectorizer, sparse.csr_matrix, np.ndarray]:
    """
    Input:
        Directory containing saved TF-IDF artifacts.

    Logic:
        Loads vectorizer, sparse matrix, and row ids.

    Output:
        Fitted vectorizer, TF-IDF matrix, and row id array.
    """
    vectorizer_path = in_dir / TFIDF_VECTORIZER_FILE
    row_ids_path = in_dir / TFIDF_ROW_IDS_FILE
    matrix_path = in_dir / TFIDF_MATRIX_FILE

    missing = [
        path
        for path in [vectorizer_path, row_ids_path, matrix_path]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(f"Missing TF-IDF artifacts: {missing}")

    vectorizer = joblib.load(vectorizer_path)
    ids = joblib.load(row_ids_path)
    matrix = sparse.load_npz(matrix_path)

    return vectorizer, matrix, ids


# TODO Phase 4:
# Experiment with alternative feature representations beyond TF-IDF:
#   - CountVectorizer baseline
#   - BM25-style weighting
#   - sentence-transformer embeddings
#   - card-specific hybrid vectors: text + type + color + mana + keywords
#   - graph embeddings from card similarity/synergy relationships
#   - topic-model vectors such as NMF-derived components
