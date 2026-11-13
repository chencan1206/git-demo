"""Wavelength importance tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from plant_hsi.dl.bundle import load_bundle, save_bundle
from plant_hsi.dl.models.dc2net import DC2Net, DC2NetConfig
from plant_hsi.explain.wavelength import (
    attention_importance,
    combine_importance,
    gradient_saliency,
    importance_svg,
    top_bands,
    write_importance_csv,
)


def _bundle(tmp_path: Path, bands: int = 10) -> dict[str, object]:
    torch.manual_seed(0)
    model = DC2Net(
        DC2NetConfig(in_bands=bands, num_classes=2, spectral_channels=4, feat_channels=16)
    )
    path = tmp_path / "model.pt"
    save_bundle(
        path,
        model,
        classes=["a", "b"],
        band_mean=np.zeros(bands, dtype=np.float32),
        band_std=np.ones(bands, dtype=np.float32),
        card={},
    )
    return load_bundle(path)


def test_importance_shapes_and_normalisation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    patches = np.random.default_rng(0).normal(0.0, 0.1, (6, 10, 9, 9)).astype(np.float32)
    attn = attention_importance(bundle, patches)
    sal = gradient_saliency(bundle, patches)
    combined = combine_importance(attn, sal)
    assert attn.shape == (10,)
    assert sal.shape == (10,)
    assert combined.shape == (10,)
    assert abs(float(combined.sum()) - 1.0) < 1e-6
    assert np.all(combined >= 0)


def test_top_bands_sorted_descending() -> None:
    importance = np.array([0.1, 0.5, 0.2, 0.05, 0.15])
    assert top_bands(importance, 3) == [1, 2, 4]
    assert top_bands(importance, 99) == [1, 2, 4, 0, 3]


def test_write_csv_contains_ranks(tmp_path: Path) -> None:
    attn = np.array([0.1, 0.4, 0.2, 0.3])
    sal = np.array([0.0, 0.2, 0.1, 0.1])
    importance = combine_importance(attn, sal)
    path = write_importance_csv(
        tmp_path / "imp.csv", [0, 1, 2, 3], [450.0, 500.0, 550.0, 600.0], attn, sal, importance
    )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    assert list(frame.columns) == [
        "band_index",
        "wavelength_nm",
        "attention",
        "saliency",
        "importance",
        "rank",
    ]
    assert sorted(frame["rank"].tolist()) == [1, 2, 3, 4]
    assert frame.loc[frame["rank"] == 1, "band_index"].item() == 1


def test_svg_marks_top_bands() -> None:
    importance = np.array([0.05, 0.1, 0.6, 0.2, 0.05])
    svg = importance_svg([400, 450, 500, 550, 600], importance, [2])
    assert svg.startswith("<svg")
    assert "<polyline" in svg
    assert "<circle" in svg
    assert "600 nm" in svg
