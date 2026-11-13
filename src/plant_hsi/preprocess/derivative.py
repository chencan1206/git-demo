"""Derivatives, including the Grünwald–Letnikov fractional-order derivative.

The GL definition is

    D^a f(x) ≈ 1/h^a * sum_{j>=0} w_j f(x - j*h),
    w_0 = 1,  w_j = w_{j-1} * (1 - (a + 1) / j).

For ``a = 1`` the weights collapse to ``[1, -1, 0, 0, ...]``, i.e. the
backward first difference — which is exactly what :func:`integer_derivative`
produces for ``order=1``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _gl_weights(order: float, memory: int) -> np.ndarray:
    weights = np.empty(memory + 1, dtype=np.float64)
    weights[0] = 1.0
    w = 1.0
    for j in range(1, memory + 1):
        w *= 1.0 - (order + 1.0) / j
        weights[j] = w
    return weights


def fractional_derivative(
    spectra: NDArray[np.floating],
    *,
    order: float = 1.0,
    memory: int | None = None,
    delta: float = 1.0,
) -> NDArray[np.float32]:
    """Grünwald–Letnikov fractional derivative along the band axis.

    ``memory`` is the number of past samples kept (default: all bands - 1).
    ``order=0`` is the identity; ``order=1`` reproduces the backward first
    difference for all columns after the first.
    """
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (samples, bands), got shape {x.shape}")
    bands = x.shape[1]
    alpha = float(order)
    if alpha < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    mem = bands - 1 if memory is None else int(memory)
    mem = max(0, min(mem, bands - 1))

    weights = _gl_weights(alpha, mem)
    padded = np.pad(x, ((0, 0), (mem, 0)))
    out = np.zeros_like(x)
    scale = float(delta) ** alpha
    for j, w in enumerate(weights):
        out += w * padded[:, mem - j : mem - j + bands]
    return np.asarray(out / scale, dtype=np.float32)


def integer_derivative(
    spectra: NDArray[np.floating],
    *,
    order: int = 1,
    delta: float = 1.0,
) -> NDArray[np.float32]:
    """Repeated backward difference, same shape as the input (edge padded)."""
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (samples, bands), got shape {x.shape}")
    out = x
    for _ in range(int(order)):
        # Zero history at the first band keeps the boundary convention used by
        # :func:`fractional_derivative` (out[:, 0] == x[:, 0] for order 1).
        out = np.diff(out, axis=1, prepend=np.zeros_like(out[:, :1]))
    if order:
        out = out / float(delta) ** int(order)
    return out.astype(np.float32)
