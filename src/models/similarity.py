from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class NeighborResult:
    query_index: int
    neighbor_indices: np.ndarray
    neighbor_scores: np.ndarray  # cosine similarity scores (1 = identical)


def fit_nn_index(X: sparse.csr_matrix, metric: str = "cosine") -> NearestNeighbors:
    # brute on sparse is fine for a first pass; can optimize later
    nn = NearestNeighbors(metric=metric, algorithm="brute")
    nn.fit(X)
    return nn


def query_neighbors(
    nn: NearestNeighbors,
    X: sparse.csr_matrix,
    query_index: int,
    k: int = 10,
    include_self: bool = False,
) -> NeighborResult:
    # sklearn returns distances; for cosine distance, similarity = 1 - distance
    q = X[query_index]
    distances, indices = nn.kneighbors(q, n_neighbors=k + (0 if include_self else 1))

    indices = indices[0]
    distances = distances[0]

    if not include_self:
        # drop the first neighbor if it is the query itself
        if indices.size > 0 and indices[0] == query_index:
            indices = indices[1:]
            distances = distances[1:]
        else:
            indices = indices[:k]
            distances = distances[:k]
    else:
        indices = indices[:k]
        distances = distances[:k]

    scores = 1.0 - distances
    return NeighborResult(
        query_index=query_index, neighbor_indices=indices, neighbor_scores=scores
    )
