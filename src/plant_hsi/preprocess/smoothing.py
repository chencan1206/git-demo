"""Smoothing filters for spectra (Savitzky–Golay and moving average)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter


def _check_matrix(spectra: NDArray[np.floating]) -> np.ndarray:
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (samples, bands), got shape {x.shape}")
    return x


def savitzky_golay(
    spectra: NDArray[np.floating],
    *,
    window: int = 7,
    polyorder: int = 2,
    deriv: int = 0,
    delta: float = 1.0,
) -> NDArray[np.float32]:
    """Savitzky–Golay smoothing / differentiation along the band axis.

    ``window`` is automatically reduced to the largest odd value that fits the
    band count, so short spectra do not raise.
    """
    x = _check_matrix(spectra)
    bands = x.shape[1]
    win = int(window)
    if win % 2 == 0:
        win += 1
    win = max(win, int(polyorder) + 1 + (int(polyorder) % 2 == 0))
    if win % 2 == 0:
        win += 1
    win = min(win, bands if bands % 2 == 1 else bands - 1)
    if win <= int(polyorder):
        raise ValueError(f"window {window} too small for polyorder {polyorder} with {bands} bands")
    out = savgol_filter(x, win, int(polyorder), deriv=int(deriv), delta=float(delta), axis=1)
    return np.asarray(out, dtype=np.float32)


def moving_average(
    spectra: NDArray[np.floating],
    *,
    window: int = 5,
) -> NDArray[np.float32]:
    """Centred moving average with edge replication."""
    x = _check_matrix(spectra)
    k = max(1, int(window))
    if k == 1:
        return x.astype(np.float32)
    if k > x.shape[1]:
        k = x.shape[1]
    kernel = np.ones(k, dtype=np.float64) / k
    pad = k // 2
    padded = np.pad(x, ((0, 0), (pad, k - 1 - pad)), mode="edge")
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    return out.astype(np.float32)
