from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer


@dataclass(frozen=True)
class ClusterConfig:
    """
    Configuration for the Phase 2 baseline clustering model.
    """

    n_clusters: int = 200
    n_components: int = 256
    random_state: int = 42
    n_init: int = 10


def validate_cluster_input(
    matrix: sparse.csr_matrix,
    cfg: ClusterConfig,
) -> None:
    """
    Input:
        Sparse TF-IDF matrix and clustering config.

    Logic:
        Confirms the matrix and config are valid before dimensionality reduction.

    Output:
        Raises ValueError if clustering cannot safely run.
    """
    if matrix.shape[0] == 0:
        raise ValueError("Cannot cluster an empty matrix.")

    if cfg.n_clusters <= 1:
        raise ValueError("n_clusters must be greater than 1.")

    if cfg.n_clusters > matrix.shape[0]:
        raise ValueError(
            f"n_clusters={cfg.n_clusters} cannot exceed rows={matrix.shape[0]}."
        )

    if cfg.n_components <= 0:
        raise ValueError("n_components must be positive.")

    max_components = min(matrix.shape) - 1

    if cfg.n_components > max_components:
        raise ValueError(
            f"n_components={cfg.n_components} is too high for matrix shape "
            f"{matrix.shape}. Max allowed is {max_components}."
        )


def reduce_with_svd(
    matrix: sparse.csr_matrix,
    cfg: ClusterConfig,
) -> tuple[np.ndarray, TruncatedSVD]:
    """
    Input:
        Sparse TF-IDF matrix and clustering config.

    Logic:
        Reduces high-dimensional sparse text vectors into dense SVD components.

    Output:
        Reduced dense matrix and fitted SVD model.
    """
    svd = TruncatedSVD(
        n_components=cfg.n_components,
        random_state=cfg.random_state,
    )

    reduced_matrix = svd.fit_transform(matrix)

    return reduced_matrix, svd


def normalize_reduced_matrix(reduced_matrix: np.ndarray) -> np.ndarray:
    """
    Input:
        Dense SVD-reduced matrix.

    Logic:
        Normalizes rows so KMeans clusters by direction/semantic pattern more
        than raw vector magnitude.

    Output:
        Normalized dense matrix.
    """
    normalizer = Normalizer(copy=False)

    return normalizer.fit_transform(reduced_matrix)


def fit_kmeans(
    reduced_matrix: np.ndarray,
    cfg: ClusterConfig,
) -> tuple[np.ndarray, KMeans]:
    """
    Input:
        Normalized reduced matrix and clustering config.

    Logic:
        Fits KMeans and assigns each row to a cluster.

    Output:
        Cluster labels and fitted KMeans model.
    """
    kmeans = KMeans(
        n_clusters=cfg.n_clusters,
        random_state=cfg.random_state,
        n_init=cfg.n_init,
    )

    labels = kmeans.fit_predict(reduced_matrix)

    return labels, kmeans


def cluster_with_svd_kmeans(
    matrix: sparse.csr_matrix,
    cfg: Optional[ClusterConfig] = None,
) -> tuple[np.ndarray, TruncatedSVD, KMeans]:
    """
    Input:
        Sparse TF-IDF matrix.

    Logic:
        Runs the Phase 2 baseline clustering pipeline:
            TF-IDF matrix -> SVD reduction -> row normalization -> KMeans.

    Output:
        Cluster labels, fitted SVD model, and fitted KMeans model.
    """
    cfg = cfg or ClusterConfig()

    validate_cluster_input(matrix, cfg)

    reduced_matrix, svd = reduce_with_svd(matrix, cfg)
    normalized_matrix = normalize_reduced_matrix(reduced_matrix)

    labels, kmeans = fit_kmeans(normalized_matrix, cfg)

    return labels, svd, kmeans


# TODO Phase 4:
# Experiment with clustering methods beyond fixed-k KMeans:
#   - HDBSCAN for discovering cluster count from density structure
#   - DBSCAN/OPTICS for density-based grouping and noise detection
#   - agglomerative clustering for hierarchical card relationships
#   - spectral clustering for graph-like similarity structures
#   - community detection on card similarity graphs
#   - topic modeling approaches such as NMF or LDA as cluster alternatives
