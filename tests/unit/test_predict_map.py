"""Sliding-window inference tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from plant_hsi.dl.bundle import save_bundle
from plant_hsi.dl.models.dc2net import DC2Net, DC2NetConfig
from plant_hsi.predict.sliding_window import (
    predict_full_image,
    seam_ratio,
    window_positions,
)


def _bundle(tmp_path: Path, bands: int = 12, patch: int = 9) -> dict[str, object]:
    torch.manual_seed(0)
    model = DC2Net(
        DC2NetConfig(in_bands=bands, num_classes=2, spectral_channels=4, feat_channels=16)
    )
    path = tmp_path / "model.pt"
    save_bundle(
        path,
        model,
        classes=["healthy", "diseased"],
        band_mean=np.zeros(bands, dtype=np.float32),
        band_std=np.ones(bands, dtype=np.float32),
        card={"patch_size": patch},
    )
    from plant_hsi.dl.bundle import load_bundle

    return load_bundle(path)


def test_window_positions_cover_edges() -> None:
    assert window_positions(30, 9, 4)[0] == 0
    assert window_positions(30, 9, 4)[-1] == 21
    assert window_positions(5, 9, 4) == [0]


def test_full_image_shape_matches_cube(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    cube = np.random.default_rng(0).normal(0.3, 0.03, (30, 41, 12)).astype(np.float32)
    labels, probs = predict_full_image(cube, bundle, patch_size=9, stride=4)
    assert labels.shape == (30, 41)
    assert probs.shape == (30, 41, 2)
    assert np.allclose(probs.sum(axis=2), 1.0, atol=1e-5)


def test_probabilities_are_smooth_across_seams(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    cube = np.random.default_rng(1).normal(0.3, 0.03, (31, 37, 12)).astype(np.float32)
    _, probs = predict_full_image(cube, bundle, patch_size=9, stride=4)
    ratio = seam_ratio(probs, 9, 4)
    assert 0.2 < ratio < 3.0


def test_roi_mask_restricts_labels(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    cube = np.random.default_rng(2).normal(0.3, 0.03, (20, 20, 12)).astype(np.float32)
    roi = np.zeros((20, 20), dtype=bool)
    roi[5:15, 5:15] = True
    labels, _ = predict_full_image(cube, bundle, patch_size=9, stride=4, roi_mask=roi)
    assert np.all(labels[~roi] == -1)
    assert np.all(labels[roi] >= 0)
