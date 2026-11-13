"""Scatter correction: standard normal variate (SNV) and multiplicative
scatter correction (MSC).

Both take a ``(n_samples, n_bands)`` matrix and return float32 with the same
shape.  Pure functions — no project / Qt imports.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def snv(
    spectra: NDArray[np.floating],
    *,
    eps: float = 1e-8,
) -> NDArray[np.float32]:
    """Standard Normal Variate: row-wise ``(x - mean) / std``.

    Removes multiplicative scatter and baseline offset per spectrum.
    """
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (samples, bands), got shape {x.shape}")
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    out = (x - mean) / (std + eps)
    return out.astype(np.float32)


def msc(
    spectra: NDArray[np.floating],
    reference: NDArray[np.floating] | None = None,
    *,
    eps: float = 1e-8,
) -> NDArray[np.float32]:
    """Multiplicative Scatter Correction.

    Each spectrum is regressed on a reference spectrum (default: the mean of
    the input) with ``x = a * ref + b``; the corrected spectrum is
    ``(x - b) / a``.
    """
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (samples, bands), got shape {x.shape}")
    ref = x.mean(axis=0) if reference is None else np.asarray(reference, dtype=np.float64).ravel()
    if ref.size != x.shape[1]:
        raise ValueError(f"reference length {ref.size} != bands {x.shape[1]}")
    ref = ref - ref.mean()
    design = np.column_stack([ref, np.ones_like(ref)])

    out = np.empty_like(x)
    for i, row in enumerate(x):
        coef, *_ = np.linalg.lstsq(design, row, rcond=None)
        a = float(coef[0])
        b = float(coef[1])
        if abs(a) < eps:
            a = eps if a >= 0 else -eps
        out[i] = (row - b) / a
    return out.astype(np.float32)
