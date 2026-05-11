from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class NeighborResult:
    """
    Result object for one nearest-neighbor query.
    """

    query_index: int
    neighbor_indices: np.ndarray
    neighbor_scores: np.ndarray  # cosine similarity scores (1 = identical)


def validate_neighbor_input(
    matrix: sparse.csr_matrix,
    query_index: int | None = None,
    k: int | None = None,
) -> None:
    """
    Input:
        Sparse feature matrix, optional query index, and optional neighbor count.

    Logic:
        Confirms the matrix and query parameters are safe for nearest-neighbor search.

    Output:
        Raises ValueError if inputs are invalid.
    """
    if matrix.shape[0] == 0:
        raise ValueError("Cannot fit/query nearest neighbors on an empty matrix.")

    if query_index is not None:
        if query_index < 0 or query_index >= matrix.shape[0]:
            raise ValueError(
                f"query_index={query_index} is outside matrix row range "
                f"0-{matrix.shape[0] - 1}."
            )

    if k is not None and k <= 0:
        raise ValueError("k must be positive.")


def fit_nn_index(
    matrix: sparse.csr_matrix,
    metric: str = "cosine",
) -> NearestNeighbors:
    """
    Input:
        Sparse card feature matrix.

    Logic:
        Fits a brute-force nearest-neighbor index. Brute force is acceptable for
        the current sparse TF-IDF MVP and avoids approximate-index complexity.

    Output:
        Fitted NearestNeighbors model.
    """
    validate_neighbor_input(matrix)

    nn = NearestNeighbors(metric=metric, algorithm="brute")
    nn.fit(matrix)

    return nn


def query_neighbors(
    nn: NearestNeighbors,
    matrix: sparse.csr_matrix,
    query_index: int,
    k: int = 10,
    include_self: bool = False,
) -> NeighborResult:
    """
    Input:
        Fitted nearest-neighbor index, feature matrix, query row index, and k.

    Logic:
        Finds nearest neighbors for one card. For cosine distance, converts
        distance into similarity with similarity = 1 - distance.

    Output:
        NeighborResult with neighbor row indices and cosine similarity scores.
    """
    validate_neighbor_input(matrix, query_index=query_index, k=k)

    n_neighbors = k if include_self else k + 1
    n_neighbors = min(n_neighbors, matrix.shape[0])

    query_vector = matrix[query_index]
    distances, indices = nn.kneighbors(query_vector, n_neighbors=n_neighbors)

    indices = indices[0]
    distances = distances[0]

    if not include_self:
        keep_mask = indices != query_index
        indices = indices[keep_mask][:k]
        distances = distances[keep_mask][:k]
    else:
        indices = indices[:k]
        distances = distances[:k]

    scores = 1.0 - distances

    return NeighborResult(
        query_index=query_index,
        neighbor_indices=indices,
        neighbor_scores=scores,
    )


# TODO Phase 4:
# Experiment with alternative similarity methods:
#   - approximate nearest-neighbor indexes for larger vector spaces
#   - similarity over sentence-transformer or hybrid card embeddings
#   - graph-based similarity using shared labels, types, colors, and synergies
#   - compare cosine similarity against learned metric approaches
#   - build separate similarity modes for rules text, deck role, and flavor/theme
