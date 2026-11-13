"""Deterministic train / validation / test splitting.

The split is index-based so callers can keep sample metadata (sample ids,
labels, ROI coordinates) alongside the spectra matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class DataSplit:
    """Indices for the three subsets (``val`` may be empty)."""

    train: NDArray[np.int64]
    test: NDArray[np.int64]
    val: NDArray[np.int64]
    seed: int = 0
    stratified: bool = False
    test_fraction: float = 0.2
    val_fraction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_train": int(self.train.size),
            "n_val": int(self.val.size),
            "n_test": int(self.test.size),
            "seed": int(self.seed),
            "stratified": bool(self.stratified),
            "test_fraction": float(self.test_fraction),
            "val_fraction": float(self.val_fraction),
            "train": self.train.tolist(),
            "val": self.val.tolist(),
            "test": self.test.tolist(),
        }


def _count(n: int, fraction: float, *, minimum: int = 1) -> int:
    """Turn a fraction into a sample count, clamped to ``[minimum, n-1]``."""
    if fraction <= 0:
        return 0
    k = int(round(n * fraction))
    return max(minimum, min(k, n - 1))


def _can_stratify(y: NDArray[Any], indices: NDArray[np.int64], n_requested: int) -> bool:
    """Stratification needs >=2 samples per class and enough test samples."""
    values, counts = np.unique(y[indices], return_counts=True)
    if values.size < 2:
        return False
    if np.any(counts < 2):
        return False
    return n_requested >= values.size


def split_indices(
    n_samples: int,
    *,
    test_size: float = 0.2,
    val_size: float = 0.0,
    seed: int = 0,
    stratify: NDArray[Any] | None = None,
) -> DataSplit:
    """Split ``range(n_samples)`` into train / test (and optional validation).

    ``stratify`` (labels) preserves class proportions when possible; the
    function falls back to a plain random split for degenerate cases.
    """
    n = int(n_samples)
    if n < 2:
        raise ValueError(f"need at least 2 samples to split, got {n}")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not 0 <= val_size < 1:
        raise ValueError(f"val_size must be in [0, 1), got {val_size}")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")

    from sklearn.model_selection import train_test_split

    all_idx = np.arange(n, dtype=np.int64)
    y = None if stratify is None else np.asarray(stratify)
    if y is not None and y.shape[0] != n:
        raise ValueError(f"stratify length {y.shape[0]} != n_samples {n}")

    n_test = _count(n, test_size)
    strat = y if y is not None and _can_stratify(y, all_idx, n_test) else None
    train_val, test = train_test_split(
        all_idx,
        test_size=n_test,
        random_state=seed,
        stratify=strat,
    )

    val = np.empty(0, dtype=np.int64)
    train = np.sort(np.asarray(train_val, dtype=np.int64))
    if val_size > 0:
        n_val = _count(train.size, val_size / max(1.0 - test_size, 1e-9))
        n_val = max(1, min(n_val, train.size - 1))
        sub_y = None if y is None else y[train_val]
        sub_strat = (
            sub_y
            if sub_y is not None and _can_stratify(sub_y, np.arange(sub_y.size), n_val)
            else None
        )
        train, val = train_test_split(
            np.asarray(train_val, dtype=np.int64),
            test_size=n_val,
            random_state=seed,
            stratify=sub_strat,
        )
        train = np.sort(np.asarray(train, dtype=np.int64))
        val = np.sort(np.asarray(val, dtype=np.int64))

    stratified = strat is not None
    return DataSplit(
        train=train,
        test=np.sort(np.asarray(test, dtype=np.int64)),
        val=val,
        seed=int(seed),
        stratified=stratified,
        test_fraction=float(test_size),
        val_fraction=float(val_size),
    )
