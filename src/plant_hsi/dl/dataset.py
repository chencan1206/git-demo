"""Patch dataset with deterministic augmentation for DC²Net training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["PatchDataset"]


class PatchDataset(Dataset):  # type: ignore[type-arg]
    """In-memory patches with optional flips / rotations / noise.

    Augmentation randomness is derived from ``(seed, epoch, sample_index)`` so
    two training runs with the same seed replay the exact same augmentations,
    which is what makes the loss curves bit-for-bit comparable.
    """

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.patches = np.asarray(patches, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        if self.patches.shape[0] != self.labels.shape[0]:
            raise ValueError("patches / labels length mismatch")
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[index]
        if self.augment:
            rng = np.random.default_rng((self.seed + 1) * 1_000_003 + self.epoch * 9_973 + index)
            if rng.random() < 0.5:
                patch = patch[:, :, ::-1]
            if rng.random() < 0.5:
                patch = patch[:, ::-1, :]
            if rng.random() < 0.5:
                patch = np.rot90(patch, k=int(rng.integers(1, 4)), axes=(1, 2))
            if rng.random() < 0.3:
                patch = patch + rng.normal(0.0, 0.01, patch.shape).astype(np.float32)
            patch = np.ascontiguousarray(patch)
        return torch.from_numpy(patch), torch.tensor(int(self.labels[index]))


def make_loader(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    """DataLoader with a seeded generator (``num_workers=0`` for determinism)."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
