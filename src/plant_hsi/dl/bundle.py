"""DC²Net weight bundles: save / load / predict across processes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from plant_hsi.dl.models.dc2net import DC2Net, DC2NetConfig, count_parameters

__all__ = ["BUNDLE_FORMAT", "load_bundle", "predict_bundle", "save_bundle"]

BUNDLE_FORMAT = "phs-dc2net/1"


def save_bundle(
    path: Path,
    model: DC2Net,
    *,
    classes: list[str],
    band_mean: np.ndarray,
    band_std: np.ndarray,
    card: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a self-describing checkpoint (weights + everything inference needs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "model_config": model.config.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "classes": [str(c) for c in classes],
        "band_mean": [float(v) for v in np.asarray(band_mean).ravel()],
        "band_std": [float(v) for v in np.asarray(band_std).ravel()],
        "card": card,
        "history": history or [],
        "torch_version": str(torch.__version__),
        "n_parameters": count_parameters(model),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_bundle(path: Path) -> dict[str, Any]:
    """Load a bundle and instantiate the model in eval mode (CPU)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"not a DC²Net bundle: {path}")
    config = DC2NetConfig.from_dict(payload["model_config"])
    model = DC2Net(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    payload["model"] = model
    payload["band_mean_array"] = np.asarray(payload["band_mean"], dtype=np.float32)
    payload["band_std_array"] = np.asarray(payload["band_std"], dtype=np.float32)
    return payload


def predict_bundle(
    bundle: dict[str, Any],
    patches: np.ndarray,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(labels, probabilities)`` for raw (unnormalised) patches."""
    model: DC2Net = bundle["model"]
    mean = bundle["band_mean_array"]
    std = bundle["band_std_array"]
    arr = np.asarray(patches, dtype=np.float32)
    norm = (arr - mean[:, None, None]) / std[:, None, None]
    labels: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(norm), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(norm[start : start + batch_size]))
            logits = model(xb)
            p = torch.softmax(logits, dim=1).numpy()
            probs.append(p)
            labels.append(p.argmax(axis=1))
    if not probs:
        return np.array([], dtype=int), np.zeros((0, len(bundle["classes"])), dtype=np.float32)
    return np.concatenate(labels), np.concatenate(probs)
