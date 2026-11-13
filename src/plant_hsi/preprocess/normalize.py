"""Normalisation transforms for spectra.

Column-wise transforms (:func:`mean_center`, :func:`standardize`,
:func:`min_max`) treat samples as rows and bands as columns, matching the
``(n_samples, n_bands)`` convention used across the project.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _matrix(x: NDArray[np.floating]) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D (samples, bands), got shape {arr.shape}")
    return arr


def mean_center(spectra: NDArray[np.floating]) -> NDArray[np.float32]:
    """Subtract the per-band mean across samples."""
    x = _matrix(spectra)
    return (x - x.mean(axis=0, keepdims=True)).astype(np.float32)


def standardize(
    spectra: NDArray[np.floating],
    *,
    eps: float = 1e-8,
) -> NDArray[np.float32]:
    """Per-band z-score: ``(x - mean) / std`` across samples."""
    x = _matrix(spectra)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return np.asarray((x - mean) / (std + eps), dtype=np.float32)


def min_max(
    spectra: NDArray[np.floating],
    *,
    feature_range: tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-8,
) -> NDArray[np.float32]:
    """Scale each band into ``feature_range``."""
    x = _matrix(spectra)
    lo = x.min(axis=0, keepdims=True)
    hi = x.max(axis=0, keepdims=True)
    a, b = float(feature_range[0]), float(feature_range[1])
    scaled = (x - lo) / (hi - lo + eps) * (b - a) + a
    return np.asarray(scaled, dtype=np.float32)


def vector_normalize(
    spectra: NDArray[np.floating],
    *,
    eps: float = 1e-8,
) -> NDArray[np.float32]:
    """Row-wise L2 normalisation (unit vector per spectrum)."""
    x = _matrix(spectra)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.asarray(x / (norm + eps), dtype=np.float32)
