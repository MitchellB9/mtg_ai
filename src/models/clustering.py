from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


@dataclass(frozen=True)
class ClusterConfig:
    n_clusters: int = 200
    n_components: int = 256  # SVD dims
    random_state: int = 42
    n_init: int = 10


def cluster_with_svd_kmeans(
    X: sparse.csr_matrix,
    cfg: Optional[ClusterConfig] = None,
) -> tuple[np.ndarray, TruncatedSVD, KMeans]:
    cfg = cfg or ClusterConfig()

    svd = TruncatedSVD(n_components=cfg.n_components, random_state=cfg.random_state)
    norm = Normalizer(copy=False)

    # SVD -> normalize -> KMeans
    X_reduced = svd.fit_transform(X)
    X_reduced = norm.fit_transform(X_reduced)

    km = KMeans(
        n_clusters=cfg.n_clusters,
        random_state=cfg.random_state,
        n_init=cfg.n_init,
    )
    labels = km.fit_predict(X_reduced)

    return labels, svd, km
