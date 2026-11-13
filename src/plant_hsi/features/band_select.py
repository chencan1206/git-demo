"""Successive Projections Algorithm (SPA) for band selection.

When a reference vector ``y`` is given the candidate chains are scored by the
RMSE of an ordinary least-squares model; without ``y`` the algorithm simply
returns the chain with maximal successive projection (unsupervised).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _projection_chain(
    x: NDArray[np.float64],
    start: int,
    n_select: int,
) -> list[int]:
    cols = [int(start)]
    for _ in range(n_select - 1):
        basis = x[:, cols]
        coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
        residual = x - basis @ coef
        norms = np.linalg.norm(residual, axis=0)
        norms[cols] = -1.0
        nxt = int(np.argmax(norms))
        if norms[nxt] <= 1e-12:
            break
        cols.append(nxt)
    return cols


def _ols_rmse(x: NDArray[np.float64], y: NDArray[np.float64], cols: list[int]) -> float:
    design = np.column_stack([x[:, cols], np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def spa(
    spectra: NDArray[np.floating],
    y: NDArray[np.floating] | None = None,
    *,
    n_select: int = 10,
    n_candidates: int = 64,
) -> list[int]:
    """Select ``n_select`` band indices with SPA.

    Parameters
    ----------
    spectra : (n_samples, n_bands)
    y : optional (n_samples,) reference values used to score candidate chains.
    n_select : number of bands to keep.
    n_candidates : maximum number of starting bands to evaluate (evenly spaced).
    """
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (samples, bands), got shape {x.shape}")
    n_samples, n_bands = x.shape
    k = int(n_select)
    if k < 1:
        raise ValueError(f"n_select must be >= 1, got {n_select}")
    k = min(k, n_bands)

    if n_bands <= n_candidates:
        starts = list(range(n_bands))
    else:
        starts = np.linspace(0, n_bands - 1, n_candidates).round().astype(int).tolist()

    if y is None:
        variances = x.var(axis=0)
        start = int(np.argmax(variances))
        return _projection_chain(x, start, k)

    target = np.asarray(y, dtype=np.float64).ravel()
    if target.size != n_samples:
        raise ValueError(f"y length {target.size} != samples {n_samples}")

    best: list[int] | None = None
    best_rmse = np.inf
    for start in starts:
        cols = _projection_chain(x, start, k)
        if len(cols) < k:
            continue
        rmse = _ols_rmse(x, target, cols)
        if rmse < best_rmse:
            best_rmse = rmse
            best = cols
    if best is None:
        raise ValueError("SPA could not build a candidate chain (degenerate spectra)")
    return best
