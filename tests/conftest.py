"""Shared fixtures for the plant-hsi-studio test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the package under test is on sys.path once for every test session.
ROOT = Path(__file__).resolve().parents[1]  # plant-hsi-studio/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def synthetic_scene() -> dict[str, object]:
    """Small two-class scene: 12 leaves (6 per class) on 20 bands."""
    rng = np.random.default_rng(7)
    height, width, bands = 120, 160, 20
    cube = rng.normal(0.3, 0.02, (height, width, bands)).astype(np.float32)
    labels = np.zeros((height, width), dtype=np.int32)
    class_map: dict[int, str] = {}
    leaf = 1
    for row in (10, 48, 86):
        for col in (10, 50, 90, 130):
            healthy = col < 80
            cube[row : row + 24, col : col + 24, :] += 0.25 if healthy else -0.15
            labels[row : row + 24, col : col + 24] = leaf
            class_map[leaf] = "healthy" if healthy else "diseased"
            leaf += 1
    return {
        "cube": cube,
        "labels": labels,
        "class_map": class_map,
        "wavelength": [400.0 + i * 20.0 for i in range(bands)],
    }
