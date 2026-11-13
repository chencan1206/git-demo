"""Wavelength importance for DC²Net (attention gates + gradient saliency)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "attention_importance",
    "combine_importance",
    "gradient_saliency",
    "importance_svg",
    "top_bands",
    "write_importance_csv",
]


def _batches(n: int, batch: int) -> list[slice]:
    return [slice(s, min(s + batch, n)) for s in range(0, n, max(1, batch))]


def attention_importance(
    bundle: dict[str, Any], patches: np.ndarray, batch_size: int = 64
) -> np.ndarray:
    """Average band gates over all patches (shape ``(bands,)``)."""
    model = bundle["model"]
    mean = bundle["band_mean_array"]
    std = bundle["band_std_array"]
    norm = (np.asarray(patches, dtype=np.float32) - mean[:, None, None]) / std[:, None, None]
    total: np.ndarray | None = None
    count = 0
    with torch.no_grad():
        for sl in _batches(len(norm), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(norm[sl]))
            _, attn = model(xb, return_attention=True)
            vec = attn.mean(dim=0).numpy()
            total = vec if total is None else total + vec
            count += 1
    if total is None:
        return np.zeros(norm.shape[1], dtype=np.float32)
    return (total / max(1, count)).astype(np.float32)


def gradient_saliency(
    bundle: dict[str, Any], patches: np.ndarray, batch_size: int = 32
) -> np.ndarray:
    """Mean absolute input gradient of the winning logit, per band."""
    model = bundle["model"]
    mean = bundle["band_mean_array"]
    std = bundle["band_std_array"]
    norm = (np.asarray(patches, dtype=np.float32) - mean[:, None, None]) / std[:, None, None]
    acc = np.zeros(norm.shape[1], dtype=np.float64)
    total = 0
    for sl in _batches(len(norm), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(norm[sl])).requires_grad_(True)
        logits = model(xb)
        score = logits.max(dim=1).values.sum()
        grad = torch.autograd.grad(score, xb)[0]
        per_band = grad.abs().mean(dim=(0, 2, 3)).detach().numpy()
        acc += per_band.astype(np.float64) * xb.shape[0]
        total += int(xb.shape[0])
    return (acc / max(1, total)).astype(np.float32)


def _normalise(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.abs(arr)
    if arr.sum() <= 1e-12:
        return np.full(arr.shape, 1.0 / max(1, arr.size))
    return arr / arr.sum()


def combine_importance(attn: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    """50/50 blend of the two normalised signals (sums to 1)."""
    return (0.5 * _normalise(attn) + 0.5 * _normalise(saliency)).astype(np.float64)


def top_bands(importance: np.ndarray, k: int) -> list[int]:
    order = np.argsort(-np.asarray(importance, dtype=float), kind="stable")
    return [int(v) for v in order[: max(1, min(k, order.size))]]


def write_importance_csv(
    path: Path,
    band_indices: list[int],
    wavelength_nm: list[float],
    attention: np.ndarray,
    saliency: np.ndarray,
    importance: np.ndarray,
) -> Path:
    rank = np.argsort(-np.asarray(importance), kind="stable")
    order = {int(b): i + 1 for i, b in enumerate(rank)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["band_index", "wavelength_nm", "attention", "saliency", "importance", "rank"]
        )
        for i, band in enumerate(band_indices):
            wl = wavelength_nm[i] if i < len(wavelength_nm) else ""
            writer.writerow(
                [
                    band,
                    wl,
                    float(attention[i]),
                    float(saliency[i]),
                    float(importance[i]),
                    order[i],
                ]
            )
    return path


def importance_svg(
    wavelength_nm: list[float],
    importance: np.ndarray,
    top: list[int],
    width: int = 720,
    height: int = 300,
) -> str:
    arr = np.asarray(importance, dtype=float)
    n = arr.size
    if n == 0:
        return "<p>无重要性数据。</p>"
    left, right, top_pad, bottom = 54, 18, 22, 42
    inner_w = width - left - right
    inner_h = height - top_pad - bottom
    hi = float(arr.max()) or 1.0
    step = inner_w / max(1, n - 1)
    pts = [
        f"{left + i * step:.1f},{top_pad + inner_h - (v / hi) * inner_h:.1f}"
        for i, v in enumerate(arr)
    ]
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="波长重要性">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top_pad + inner_h}" x2="{width - right}" '
        f'y2="{top_pad + inner_h}" stroke="#c8d2df"/>',
        f'<polyline fill="none" stroke="#2f6fdb" stroke-width="2" points="{" ".join(pts)}"/>',
    ]
    for idx in top:
        x = left + idx * step
        y = top_pad + inner_h - (arr[idx] / hi) * inner_h
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#d9534f"/>')
    if wavelength_nm and len(wavelength_nm) >= n:
        lo, hi_wl = float(wavelength_nm[0]), float(wavelength_nm[-1])
        parts.append(
            f'<text x="{left}" y="{height - 14}" font-size="12" fill="#5c6b7f">{lo:.0f} nm</text>'
        )
        parts.append(
            f'<text x="{width - right - 60}" y="{height - 14}" font-size="12" '
            f'fill="#5c6b7f">{hi_wl:.0f} nm</text>'
        )
    parts.append(f'<text x="8" y="{top_pad + 4}" font-size="12" fill="#5c6b7f">重要性</text>')
    parts.append("</svg>")
    return "".join(parts)
