from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype="float32").reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array.copy()
    return (array / norm).astype("float32")


def pool_embeddings(list_of_embeddings: Sequence[np.ndarray]) -> np.ndarray:
    usable = [np.asarray(embedding, dtype="float32").reshape(-1) for embedding in list_of_embeddings if embedding is not None]
    if not usable:
        raise ValueError("At least one family embedding is required for pooling.")
    dimensions = {embedding.shape[0] for embedding in usable}
    if len(dimensions) != 1:
        raise ValueError("All family embeddings must have the same dimension before pooling.")
    return np.mean(np.vstack(usable), axis=0, dtype="float32")
