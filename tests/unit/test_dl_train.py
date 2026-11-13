"""Training loop tests: overfit, determinism, resume and early stopping."""

from __future__ import annotations

import numpy as np
import pytest

from plant_hsi.dl.trainer import DLConfig, train_dc2net


def _tiny_dataset(
    n_per_class: int = 6, bands: int = 10, patch: int = 9
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    rng = np.random.default_rng(3)
    patches = rng.normal(0.0, 0.05, (n_per_class * 2, bands, patch, patch)).astype(np.float32)
    labels = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    # Class signal: a spectral tilt plus a checkerboard pattern.
    sign = np.where(labels == 0, 1.0, -1.0).astype(np.float32)
    tilt = np.linspace(-1.0, 1.0, bands, dtype=np.float32)
    patches += sign[:, None, None, None] * tilt[None, :, None, None] * 0.4
    checker = np.indices((patch, patch)).sum(axis=0) % 2
    patches += sign[:, None, None, None] * checker[None, None, :, :] * 0.4
    splits = {
        "train": np.arange(n_per_class * 2 - 1),
        "val": np.array([n_per_class * 2 - 1]),
        "test": np.arange(n_per_class * 2),
    }
    return patches, labels, splits, ["a", "b"]


def _config(**overrides: object) -> DLConfig:
    base = DLConfig(
        epochs=40,
        batch_size=11,
        lr=1e-2,
        patience=100,
        spectral_channels=4,
        feat_channels=16,
        hidden_channels=32,
        dropout=0.0,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_overfits_small_dataset() -> None:
    patches, labels, splits, classes = _tiny_dataset()
    result = train_dc2net(patches, labels, splits, classes, _config(), seed=5)
    assert result.train_accuracy > 0.95
    assert result.history[-1]["train_loss"] < 0.2


def test_same_seed_reproduces_loss_curve() -> None:
    patches, labels, splits, classes = _tiny_dataset()
    first = train_dc2net(patches, labels, splits, classes, _config(epochs=6), seed=9)
    second = train_dc2net(patches, labels, splits, classes, _config(epochs=6), seed=9)
    first_losses = [row["train_loss"] for row in first.history]
    second_losses = [row["train_loss"] for row in second.history]
    assert first_losses == second_losses
    assert first.best_epoch == second.best_epoch


def test_different_seed_changes_curve() -> None:
    patches, labels, splits, classes = _tiny_dataset()
    first = train_dc2net(patches, labels, splits, classes, _config(epochs=4), seed=1)
    second = train_dc2net(patches, labels, splits, classes, _config(epochs=4), seed=2)
    assert [r["train_loss"] for r in first.history] != [r["train_loss"] for r in second.history]


def test_resume_continues_history() -> None:
    patches, labels, splits, classes = _tiny_dataset()
    first = train_dc2net(patches, labels, splits, classes, _config(epochs=3), seed=4)
    resumed = train_dc2net(
        patches,
        labels,
        splits,
        classes,
        _config(epochs=2),
        seed=4,
        resume={
            "state_dict": first.model.state_dict(),
            "history": first.history,
            "epoch": len(first.history),
        },
    )
    assert len(resumed.history) == len(first.history) + 2
    assert resumed.history[-1]["epoch"] == len(first.history) + 2


def test_empty_train_split_raises() -> None:
    patches, labels, _, classes = _tiny_dataset()
    with pytest.raises(ValueError, match="empty"):
        train_dc2net(
            patches,
            labels,
            {"train": np.array([], dtype=int), "val": np.array([0]), "test": np.array([1])},
            classes,
            _config(epochs=1),
            seed=0,
        )
