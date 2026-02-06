from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.io import ensure_dir


@dataclass(frozen=True)
class TfidfConfig:
    min_df: int = 3
    max_df: float = 0.90
    ngram_range: Tuple[int, int] = (1, 2)
    max_features: Optional[int] = 250_000
    sublinear_tf: bool = True
    lowercase: bool = True


def build_tfidf_vectorizer(cfg: TfidfConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        sublinear_tf=cfg.sublinear_tf,
        lowercase=cfg.lowercase,
        token_pattern=r"(?u)\b\w+\b",  # keep simple, includes underscores
    )


def fit_transform_tfidf(
    texts: pd.Series,
    cfg: Optional[TfidfConfig] = None,
) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    cfg = cfg or TfidfConfig()
    vec = build_tfidf_vectorizer(cfg)
    X = vec.fit_transform(texts.fillna("").astype(str))
    return vec, X


def transform_tfidf(
    vec: TfidfVectorizer,
    texts: pd.Series,
) -> sparse.csr_matrix:
    return vec.transform(texts.fillna("").astype(str))


def save_tfidf_artifacts(
    out_dir: Path,
    vectorizer: TfidfVectorizer,
    X: sparse.csr_matrix,
    ids: np.ndarray,
) -> None:
    ensure_dir(out_dir)

    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib", compress=3)
    joblib.dump(ids, out_dir / "row_ids.joblib", compress=3)

    # Sparse matrix: store as .npz
    sparse.save_npz(out_dir / "tfidf_matrix.npz", X)


def load_tfidf_artifacts(
    in_dir: Path,
) -> tuple[TfidfVectorizer, sparse.csr_matrix, np.ndarray]:
    vec = joblib.load(in_dir / "tfidf_vectorizer.joblib")
    ids = joblib.load(in_dir / "row_ids.joblib")
    X = sparse.load_npz(in_dir / "tfidf_matrix.npz")
    return vec, X, ids
