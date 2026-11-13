"""Principal component analysis for spectra (deterministic SVD-based)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PCAResult:
    """Result of :func:`pca`; all arrays are float32."""

    scores: NDArray[np.float32]  # (n_samples, k)
    components: NDArray[np.float32]  # (k, n_bands)
    explained_variance_ratio: NDArray[np.float32]  # (k,)
    mean: NDArray[np.float32]  # (n_bands,)
    singular_values: NDArray[np.float32]  # (k,)

    def transform(self, spectra: NDArray[np.floating]) -> NDArray[np.float32]:
        x = np.asarray(spectra, dtype=np.float64) - self.mean.astype(np.float64)
        return (x @ self.components.astype(np.float64).T).astype(np.float32)


def pca(
    spectra: NDArray[np.floating],
    n_components: int | None = None,
    *,
    center: bool = True,
) -> PCAResult:
    """PCA via thin SVD on the column-centred data matrix.

    Components are returned as rows ordered by decreasing explained variance.
    """
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (samples, bands), got shape {x.shape}")
    n_samples, n_bands = x.shape
    mean = x.mean(axis=0) if center else np.zeros(n_bands, dtype=np.float64)
    centered = x - mean

    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(s > 0))
    max_k = min(n_samples, n_bands)
    if n_components is not None and int(n_components) > max_k:
        raise ValueError(f"n_components={n_components} exceeds maximum {max_k} for shape {x.shape}")
    k = rank if n_components is None else max(1, min(int(n_components), max(rank, 1), max_k))

    total = float(np.sum(s**2))
    evr = (s[:k] ** 2 / total) if total > 0 else np.zeros(k, dtype=np.float64)
    scores = u[:, :k] * s[:k]
    return PCAResult(
        scores=scores.astype(np.float32),
        components=vt[:k].astype(np.float32),
        explained_variance_ratio=evr.astype(np.float32),
        mean=np.asarray(mean, dtype=np.float32),
        singular_values=s[:k].astype(np.float32),
    )
