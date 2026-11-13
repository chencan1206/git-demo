"""Feature reduction, band selection and patch-sample construction."""

from __future__ import annotations

from plant_hsi.features.band_select import spa
from plant_hsi.features.patchset import (
    PatchSet,
    assert_no_group_leakage,
    extract_patches,
    grouped_split,
    normalize_patches,
)
from plant_hsi.features.reduce import PCAResult, pca

__all__ = [
    "PCAResult",
    "PatchSet",
    "assert_no_group_leakage",
    "extract_patches",
    "grouped_split",
    "normalize_patches",
    "pca",
    "spa",
]
