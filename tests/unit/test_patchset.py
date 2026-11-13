"""Unit tests for :mod:`plant_hsi.features.patchset`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plant_hsi.features.patchset import (
    PatchSet,
    assert_no_group_leakage,
    extract_patches,
    grouped_split,
)


def _build(scene: dict[str, object], **kwargs: object) -> PatchSet:
    return extract_patches(
        scene["cube"],  # type: ignore[arg-type]
        scene["labels"],  # type: ignore[arg-type]
        class_map=scene["class_map"],  # type: ignore[arg-type]
        wavelength_nm=scene["wavelength"],  # type: ignore[arg-type]
        patch_size=int(kwargs.pop("patch_size", 15)),
        stride=int(kwargs.pop("stride", 8)),
        **kwargs,  # type: ignore[arg-type]
    )


def test_extract_shapes_and_meta(synthetic_scene: dict[str, object]) -> None:
    ps = _build(synthetic_scene)
    assert ps.patches.ndim == 4
    assert ps.patches.shape[1] == 20
    assert ps.patches.shape[2] == ps.patches.shape[3] == 15
    assert ps.patches.dtype == np.float32
    assert set(ps.classes) == {"healthy", "diseased"}
    assert ps.meta["n_leaves"] == 12
    assert sum(ps.meta["split_counts"].values()) == ps.patches.shape[0]


def test_grouped_split_has_no_leaf_leakage(synthetic_scene: dict[str, object]) -> None:
    ps = _build(synthetic_scene)
    assert_no_group_leakage(ps.index)
    splits = ps.meta["leaf_splits"]
    assert sets_disjoint(splits["train"], splits["val"])
    assert sets_disjoint(splits["train"], splits["test"])
    assert sets_disjoint(splits["val"], splits["test"])
    assert all(ps.meta["split_counts"][name] > 0 for name in ("train", "val", "test"))


def sets_disjoint(a: list[int], b: list[int]) -> bool:
    return not set(a) & set(b)


def test_grouped_split_is_deterministic() -> None:
    leaves = np.repeat(np.arange(1, 13), 5)
    classes = np.array(["a"] * 30 + ["b"] * 30)
    first = grouped_split(leaves, classes, seed=11)
    second = grouped_split(leaves, classes, seed=11)
    other = grouped_split(leaves, classes, seed=12)
    assert first == second
    assert first != other


def test_band_subset_and_wavelengths(synthetic_scene: dict[str, object]) -> None:
    ps = _build(synthetic_scene, band_indices=[0, 5, 10])
    assert ps.patches.shape[1] == 3
    assert ps.band_indices == [0, 5, 10]
    assert ps.wavelength_nm == pytest.approx(
        [synthetic_scene["wavelength"][i] for i in (0, 5, 10)]  # type: ignore[index]
    )
    assert len(ps.meta["band_mean"]) == 3


def test_band_stats_ignore_non_train_rows(synthetic_scene: dict[str, object]) -> None:
    ps = _build(synthetic_scene)
    train = ps.split_indices()["train"]
    mean = np.asarray(ps.meta["band_mean"])
    expected = ps.patches[train].mean(axis=(0, 2, 3))
    assert mean == pytest.approx(expected, rel=1e-5)


def test_rejects_even_patch_size(synthetic_scene: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="odd"):
        extract_patches(synthetic_scene["cube"], synthetic_scene["labels"], patch_size=14)  # type: ignore[arg-type]


def test_rejects_empty_mask(synthetic_scene: dict[str, object]) -> None:
    empty = np.zeros_like(synthetic_scene["labels"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no labelled pixels"):
        extract_patches(synthetic_scene["cube"], empty)  # type: ignore[arg-type]


def test_save_and_load_roundtrip(tmp_path: Path, synthetic_scene: dict[str, object]) -> None:
    ps = _build(synthetic_scene, patch_size=9, stride=6)
    ps.save(tmp_path)
    loaded = PatchSet.load(tmp_path)
    assert np.array_equal(loaded.patches, ps.patches)
    assert loaded.index.shape == ps.index.shape
    assert loaded.classes == ps.classes
