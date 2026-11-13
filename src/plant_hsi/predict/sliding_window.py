"""Patch-wise sliding-window inference for full hyperspectral cubes.

Overlapping windows are averaged in probability space, which removes hard
stitching seams without any post-hoc blending filter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plant_hsi.dl.bundle import predict_bundle

__all__ = ["predict_full_image", "seam_ratio", "window_positions"]


def window_positions(size: int, patch: int, stride: int) -> list[int]:
    """Top-left coordinates covering ``[0, size)`` including the far edge."""
    if size <= patch:
        return [0]
    stride = max(1, int(stride))
    pos = list(range(0, size - patch + 1, stride))
    if pos[-1] != size - patch:
        pos.append(size - patch)
    return pos


def predict_full_image(
    cube: np.ndarray,
    bundle: dict[str, Any],
    patch_size: int,
    stride: int,
    band_indices: list[int] | None = None,
    batch_size: int = 64,
    roi_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify every pixel of ``(H, W, B)``.

    Returns ``(label_map, prob_map)`` with shapes ``(H, W)`` and ``(H, W, C)``.
    ``roi_mask`` (bool) limits accumulation to foreground pixels.
    """
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"expected (H, W, B) cube, got {cube.shape}")
    if band_indices is not None:
        cube = cube[:, :, np.asarray(band_indices, dtype=int)]
    h, w, _ = cube.shape
    pad = patch_size // 2
    padded = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    rows = window_positions(h, patch_size, stride)
    cols = window_positions(w, patch_size, stride)
    positions = [(r, c) for r in rows for c in cols]

    n_classes = len(bundle["classes"])
    prob_sum = np.zeros((h, w, n_classes), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    mask = None if roi_mask is None else np.asarray(roi_mask, dtype=bool)
    # Hann blending suppresses window-edge contributions; with uniform weights
    # a patch that saw only reflected border padding can otherwise leave a
    # visible step at the seam.
    taper = 0.1 + 0.9 * np.hanning(patch_size)
    window = np.outer(taper, taper).astype(np.float32)

    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        patches = np.stack(
            [
                np.transpose(padded[r : r + patch_size, c : c + patch_size, :], (2, 0, 1))
                for r, c in chunk
            ]
        ).astype(np.float32)
        _, probs = predict_bundle(bundle, patches)
        for (r, c), prob in zip(chunk, probs, strict=False):
            if mask is not None:
                sub_mask = mask[r : r + patch_size, c : c + patch_size].astype(np.float32)
                weights = window * sub_mask
                prob_sum[r : r + patch_size, c : c + patch_size, :] += (
                    prob[None, None, :] * weights[:, :, None]
                )
                weight[r : r + patch_size, c : c + patch_size] += weights
            else:
                prob_sum[r : r + patch_size, c : c + patch_size, :] += (
                    prob[None, None, :] * window[:, :, None]
                )
                weight[r : r + patch_size, c : c + patch_size] += window

    safe = np.where(weight > 0, weight, 1.0)[:, :, None]
    prob_map = (prob_sum / safe).astype(np.float32)
    labels = prob_map.argmax(axis=2).astype(np.int32)
    if mask is not None:
        labels = np.where(mask, labels, -1)
    return labels, prob_map


def seam_ratio(
    prob_map: np.ndarray,
    patch_size: int,
    stride: int,
) -> float:
    """Mean probability jump at window boundaries ÷ interior jump.

    ~1.0 means the stitch is indistinguishable from ordinary image texture;
    large values indicate visible seams.
    """
    prob = np.asarray(prob_map, dtype=np.float64)
    w = prob.shape[1]
    cols = set(window_positions(w, patch_size, stride)[1:])
    if prob.shape[2] > 1:
        diff = np.abs(np.diff(prob, axis=1)).mean(axis=2)
    else:
        diff = np.abs(np.diff(prob, axis=1)).mean(axis=2)
    boundary: list[float] = []
    interior: list[float] = []
    for x in range(diff.shape[1]):
        (boundary if (x + 1) in cols else interior).append(float(diff[:, x].mean()))
    if not interior or not boundary:
        return 1.0
    return float(np.mean(boundary) / max(1e-9, np.mean(interior)))
