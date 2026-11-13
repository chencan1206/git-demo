"""Segmentation algorithms — spectral angle (RA / SAM), divergence (DE / SID),
thresholding, morphological cleanup, and connected-component labelling.

All functions take a cube ``(H, W, bands)`` as float32 ndarray and return
plain ndarrays or dataclass results — no side effects, no Qt / project
imports.  The intent is that :mod:`plant_hsi.segment.step` wraps these
into :class:`Step` implementations while keeping the algorithm layer pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

# -----------------------------------------------------------------------------
# Similarity transforms
# -----------------------------------------------------------------------------


def spectral_angle(
    cube: NDArray[np.floating],
    ref: NDArray[np.floating],
    *,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """Spectral Angle Mapper (SAM / RA).

    For every pixel compute the angle (in radians, ``[0, pi/2]``) between the
    pixel spectrum and ``ref``, then return ``1 - angle/pi`` so that higher
    values mean *more similar*.

    Parameters
    ----------
    cube : ndarray (H, W, B) float
    ref  : ndarray (B,)  float   — reference spectrum (mean ROI, endmember, …)
    normalize : bool — L2-normalise each spectrum before taking the dot
                       product.  Set False only when you *know* the input
                       is already unit-norm (e.g. from PCA endmembers).

    Returns
    -------
    similarity : ndarray (H, W) float32 in [0, 1]
    """
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    if ref.size != b:
        raise ValueError(f"ref length {ref.size} != cube bands {b}")

    flat_f = flat.astype(np.float64, copy=False)
    ref_f = ref.astype(np.float64, copy=False)

    if normalize:
        flat_f = flat_f / (np.linalg.norm(flat_f, axis=1, keepdims=True) + 1e-12)
        ref_f = ref_f / (np.linalg.norm(ref_f) + 1e-12)

    cos_ang = flat_f @ ref_f
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    angle = np.arccos(cos_ang)
    similarity = 1.0 - angle / np.pi
    return np.asarray(similarity.reshape(h, w), dtype=np.float32)


def spectral_divergence(
    cube: NDArray[np.floating],
    ref: NDArray[np.floating],
    *,
    eps: float = 1e-8,
) -> NDArray[np.float32]:
    """Spectral Information Divergence (SID / DE).

    Treats each spectrum as a probability mass function and returns the
    symmetric KL divergence.  Lower = more similar; we negate-and-shift
    so the return is in ``[0, 1]`` with 1 = perfect match, same convention
    as :func:`spectral_angle`.
    """
    h, w, b = cube.shape
    flat = cube.reshape(-1, b).astype(np.float64)
    ref_f = np.asarray(ref, dtype=np.float64).reshape(-1)
    if ref_f.size != b:
        raise ValueError(f"ref length {ref_f.size} != cube bands {b}")

    # Make both into PMFs (non-negative, sum to 1)
    p = np.clip(flat, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    q = np.clip(ref_f, eps, None)
    q = q / q.sum()

    def kl(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        return (a * np.log(a / b)).sum(axis=1)

    sid = kl(p, q) + kl(q[None, :], p)  # (N,)
    # Max plausible SID for a b-bin PMF = 2 * log(b)
    max_sid = 2.0 * np.log(max(b, 2))
    similarity = 1.0 - np.clip(sid / max_sid, 0.0, 1.0)
    return np.asarray(similarity.reshape(h, w), dtype=np.float32)


SimilarityMethod = Literal["ra", "de"]


def similarity_map(
    cube: NDArray[np.floating],
    ref: NDArray[np.floating],
    method: SimilarityMethod = "ra",
    **kw: object,
) -> NDArray[np.float32]:
    """Dispatch :func:`spectral_angle` / :func:`spectral_divergence`."""
    if method == "ra":
        return spectral_angle(cube, ref, **kw)  # type: ignore[arg-type]
    if method == "de":
        return spectral_divergence(cube, ref, **kw)  # type: ignore[arg-type]
    raise ValueError(f"unknown method {method!r}; expected 'ra' or 'de'")


# -----------------------------------------------------------------------------
# Thresholding
# -----------------------------------------------------------------------------


def binarize(
    sim: NDArray[np.floating],
    *,
    method: Literal["otsu", "percentile", "fixed"] = "otsu",
    threshold: float | None = None,
) -> tuple[NDArray[np.uint8], float]:
    """Threshold a similarity map into a binary mask.

    Returns ``(mask_uint8_0_or_1, threshold_used_as_float_in_original_units)``.
    """
    sim = np.asarray(sim, dtype=np.float32)
    if sim.ndim != 2:
        raise ValueError(f"sim must be 2-D, got shape {sim.shape}")

    if method == "otsu":
        # Normalize to uint8 for Otsu
        lo, hi = float(np.min(sim)), float(np.max(sim))
        if hi - lo < 1e-9:
            thr = 1.0
            mask = np.zeros_like(sim, dtype=np.uint8)
            return mask, thr
        sim_u8 = ((sim - lo) / (hi - lo) * 255.0).astype(np.uint8)
        thr_u8, bin_u8 = cv2.threshold(sim_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Map threshold back to original similarity units
        thr = lo + (thr_u8 / 255.0) * (hi - lo)
        mask = (bin_u8 > 0).astype(np.uint8)
        return mask, float(thr)

    if method == "percentile":
        if threshold is None or not (0.0 < threshold < 100.0):
            raise ValueError("percentile requires 0 < threshold < 100")
        thr = float(np.percentile(sim, threshold))
        mask = (sim >= thr).astype(np.uint8)
        return mask, thr

    if method == "fixed":
        if threshold is None:
            raise ValueError("fixed threshold requires threshold value")
        thr = float(threshold)
        mask = (sim >= thr).astype(np.uint8)
        return mask, thr

    raise ValueError(f"unknown method {method!r}")


def morphological_cleanup(
    mask: NDArray[np.uint8],
    *,
    kernel_size: int = 3,
    open_iter: int = 1,
    close_iter: int = 2,
) -> NDArray[np.uint8]:
    """Open (remove small bright noise) then close (fill small dark holes)."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    out = mask.copy()
    out2: NDArray[np.uint8] = out
    if open_iter > 0:
        out2 = cv2.morphologyEx(out2, cv2.MORPH_OPEN, kernel, iterations=open_iter)  # type: ignore[assignment]
    if close_iter > 0:
        out2 = cv2.morphologyEx(out2, cv2.MORPH_CLOSE, kernel, iterations=close_iter)  # type: ignore[assignment]
    return out2


# -----------------------------------------------------------------------------
# Connected components
# -----------------------------------------------------------------------------


@dataclass
class ComponentResult:
    """Output of :func:`connected_components`."""

    labels: NDArray[np.int32]  # (H, W) label map; 0 = background
    n_components: int  # number of foreground ROIs
    areas: NDArray[np.int32]  # (n,) area in pixels
    centroids: NDArray[np.float64]  # (n, 2) centroids as (col, row)
    bboxes: NDArray[np.int32]  # (n, 4) as (left, top, width, height)


def connected_components(
    mask: NDArray[np.uint8],
    *,
    connectivity: int = 8,
    min_area: int = 0,
) -> ComponentResult:
    """Label connected regions and filter by ``min_area``.

    Returns a :class:`ComponentResult` where label IDs are contiguous
    ``1..n`` after min-area filtering (background stays 0).
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)

    n, labels0, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA
    areas_all = stats[1:, cv2.CC_STAT_AREA]
    keep = areas_all >= min_area

    if keep.sum() == 0:
        h, w = mask.shape
        return ComponentResult(
            labels=np.zeros((h, w), dtype=np.int32),
            n_components=0,
            areas=np.zeros(0, dtype=np.int32),
            centroids=np.zeros((0, 2), dtype=np.float64),
            bboxes=np.zeros((0, 4), dtype=np.int32),
        )

    # Relabel: old label 1..n-1 → new label 1..k
    labels = np.zeros_like(labels0)
    areas: list[int] = []
    cents_xy: list[list[float]] = []
    bboxes: list[list[int]] = []
    new_id = 1
    for old_id in range(1, n):
        if not keep[old_id - 1]:
            continue
        areas.append(int(stats[old_id, cv2.CC_STAT_AREA]))
        # centroids from cv2 are (col, row)
        cents_xy.append([float(cents[old_id, 0]), float(cents[old_id, 1])])
        bboxes.append(
            [
                int(stats[old_id, cv2.CC_STAT_LEFT]),
                int(stats[old_id, cv2.CC_STAT_TOP]),
                int(stats[old_id, cv2.CC_STAT_WIDTH]),
                int(stats[old_id, cv2.CC_STAT_HEIGHT]),
            ]
        )
        labels[labels0 == old_id] = new_id
        new_id += 1

    return ComponentResult(
        labels=labels.astype(np.int32),
        n_components=len(areas),
        areas=np.asarray(areas, dtype=np.int32),
        centroids=np.asarray(cents_xy, dtype=np.float64),
        bboxes=np.asarray(bboxes, dtype=np.int32),
    )


# -----------------------------------------------------------------------------
# Reference-spectrum helpers
# -----------------------------------------------------------------------------


def mean_spectrum_in_box(
    cube: NDArray[np.floating],
    *,
    col: int,
    row: int,
    width: int,
    height: int,
) -> NDArray[np.float32]:
    """Return the mean spectrum of a rectangular patch (auto-clipped to cube)."""
    h, w, _ = cube.shape
    x0 = max(0, col - width // 2)
    x1 = min(w, x0 + width)
    y0 = max(0, row - height // 2)
    y1 = min(h, y0 + height)
    patch = cube[y0:y1, x0:x1, :]
    if patch.size == 0:
        raise ValueError("empty patch after clipping")
    return np.asarray(patch.mean(axis=(0, 1)), dtype=np.float32)


def reference_from_component(
    cube: NDArray[np.floating],
    labels: NDArray[np.integer],
    label_id: int,
) -> NDArray[np.float32]:
    """Mean spectrum of all pixels where ``labels == label_id``."""
    mask = labels == label_id
    if not mask.any():
        raise ValueError(f"label {label_id} has no pixels")
    return np.asarray(cube[mask].mean(axis=0), dtype=np.float32)
