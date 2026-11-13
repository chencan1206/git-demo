"""Unit tests for the DC²Net model and memory guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from plant_hsi.dl.bundle import load_bundle, predict_bundle, save_bundle
from plant_hsi.dl.models.dc2net import DC2Net, DC2NetConfig, count_parameters
from plant_hsi.dl.trainer import (
    MemoryBudgetError,
    check_memory_budget,
    estimate_peak_memory_bytes,
)


@pytest.fixture()
def model() -> DC2Net:
    torch.manual_seed(0)
    return DC2Net(DC2NetConfig(in_bands=12, num_classes=3, spectral_channels=4))


def test_forward_shapes(model: DC2Net) -> None:
    x = torch.randn(2, 12, 15, 15)
    logits = model(x)
    assert logits.shape == (2, 3)
    logits2, attn = model(x, return_attention=True)
    assert logits2.shape == (2, 3)
    assert attn.shape == (2, 12)
    assert torch.all(attn > 0) and torch.all(attn < 1)


def test_forward_is_deterministic(model: DC2Net) -> None:
    model.eval()
    x = torch.randn(3, 12, 15, 15, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.allclose(a, b)


def test_parameter_count_is_stable(model: DC2Net) -> None:
    n = count_parameters(model)
    assert n == count_parameters(model)
    assert 5_000 < n < 5_000_000


def test_gradients_flow(model: DC2Net) -> None:
    x = torch.randn(2, 12, 15, 15)
    model(x).sum().backward()
    assert model.deform1.weight.grad is not None
    assert float(model.deform1.weight.grad.abs().sum()) > 0


def test_memory_guard_raises_with_suggestion() -> None:
    total = 24 * 1024**3
    check_memory_budget(total, batch_size=4, bands=40, patch_size=15, spectral_channels=8)
    with pytest.raises(MemoryBudgetError, match="batch_size"):
        check_memory_budget(total, batch_size=4096, bands=306, patch_size=32, spectral_channels=8)


def test_memory_estimate_grows_with_batch() -> None:
    small = estimate_peak_memory_bytes(4, 20, 15, 8)
    large = estimate_peak_memory_bytes(64, 20, 15, 8)
    assert large > small * 10


def test_bundle_roundtrip_predicts_identically(tmp_path: Path, model: DC2Net) -> None:
    rng = np.random.default_rng(0)
    mean = rng.normal(0.3, 0.02, 12).astype(np.float32)
    std = np.abs(rng.normal(0.05, 0.01, 12)).astype(np.float32)
    patches = rng.normal(0.3, 0.05, (7, 12, 15, 15)).astype(np.float32)
    path = tmp_path / "dc2net.pt"
    save_bundle(path, model, classes=["a", "b", "c"], band_mean=mean, band_std=std, card={})

    first_labels, first_probs = predict_bundle(load_bundle(path), patches)
    second = load_bundle(path)
    second_labels, second_probs = predict_bundle(second, patches)
    assert np.array_equal(first_labels, second_labels)
    assert np.allclose(first_probs, second_probs, atol=1e-6)
    assert first_probs.shape == (7, 3)
    assert np.allclose(first_probs.sum(axis=1), 1.0, atol=1e-5)
