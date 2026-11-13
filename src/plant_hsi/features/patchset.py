"""3-D patch sample sets for the DC²Net branch.

A :class:`PatchSet` is ``(N, bands, patch, patch)`` float32 patches plus an
index table that records where each patch came from (row/col), which object
(``leaf_id``) it belongs to, its class label and its split (train/val/test).

The split is **grouped by leaf** so patches cut from the same object never sit
in two different splits (no spatial leakage).  Normalisation statistics are
computed on the training split only and stored in ``meta.json`` so inference
can replay exactly the same scaling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------- split


def grouped_split(
    leaf_ids: np.ndarray,
    classes: np.ndarray,
    ratios: dict[str, float] | None = None,
    seed: int = 0,
) -> dict[int, str]:
    """Assign whole leaves to ``train`` / ``val`` / ``test``.

    Splitting happens per class (stratified) at the *group* level: every leaf
    of a class is shuffled with ``seed`` and handed out by target proportions.
    If a split would end up empty it steals one group from the largest split,
    so small datasets still produce a usable three-way split.
    """
    ratios = dict(ratios or {"train": 0.7, "val": 0.15, "test": 0.15})
    leaves = np.asarray(leaf_ids)
    # Class names may be strings; stratify on their integer codes.
    _, cls = np.unique(np.asarray(classes).astype(str), return_inverse=True)
    rng = np.random.default_rng(int(seed))

    groups: dict[Any, list[int]] = {}
    for leaf, label in zip(leaves, cls, strict=False):
        groups.setdefault(int(leaf), []).append(int(label))
    leaf_class = {leaf: int(np.argmax(np.bincount(vals))) for leaf, vals in groups.items()}

    assignment: dict[int, str] = {}
    for label in sorted({int(v) for v in leaf_class.values()}):
        members = sorted(leaf for leaf, lab in leaf_class.items() if lab == label)
        order = rng.permutation(len(members))
        members = [members[i] for i in order]
        n = len(members)
        n_test = int(round(n * ratios.get("test", 0.15))) if n >= 3 else 0
        n_val = int(round(n * ratios.get("val", 0.15))) if n >= 4 else 0
        if n >= 4:
            n_val = max(1, n_val)
        if n >= 3:
            n_test = max(1, n_test)
        n_train = n - n_val - n_test
        while n_train < 1 and (n_val + n_test) > 0:
            if n_test:
                n_test -= 1
            else:
                n_val -= 1
            n_train += 1
        chunks = [
            ("test", members[:n_test]),
            ("val", members[n_test : n_test + n_val]),
            ("train", members[n_test + n_val :]),
        ]
        for split, chunk in chunks:
            for leaf in chunk:
                assignment[leaf] = split

    # Guarantee every split is populated when there are enough groups.
    if len(assignment) >= 3:
        for split in SPLITS:
            if any(v == split for v in assignment.values()):
                continue
            counts = {s: sum(v == s for v in assignment.values()) for s in SPLITS}
            donor = max(counts, key=lambda s: counts[s])
            if counts[donor] > 1:
                candidate = next(leaf for leaf, s in assignment.items() if s == donor)
                assignment[candidate] = split
    return assignment


def assert_no_group_leakage(index: pd.DataFrame, group_column: str = "leaf_id") -> None:
    """Raise if the same ``group_column`` value appears in two splits."""
    seen: dict[int, str] = {}
    for leaf, split in zip(index[group_column].to_numpy(), index["split"].to_numpy(), strict=False):
        key = int(leaf)
        if key in seen and seen[key] != split:
            raise ValueError(f"group {key} leaked across splits: {seen[key]} / {split}")
        seen[key] = str(split)


# ---------------------------------------------------------------- extraction


@dataclass
class PatchSet:
    patches: np.ndarray  # (N, bands, patch, patch) float32
    index: pd.DataFrame
    band_indices: list[int]
    wavelength_nm: list[float]
    meta: dict[str, Any]

    @property
    def classes(self) -> list[str]:
        return [str(v) for v in sorted(self.index["label"].unique())]

    def split_indices(self) -> dict[str, np.ndarray]:
        return {split: np.flatnonzero(self.index["split"].to_numpy() == split) for split in SPLITS}

    def subset(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.patches[np.asarray(indices, dtype=int)], dtype=np.float32)

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "patches.npy", self.patches.astype(np.float32))
        self.index.to_csv(out_dir / "index.csv", index=False, encoding="utf-8")
        (out_dir / "meta.json").write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out_dir

    @classmethod
    def load(cls, path: Path) -> PatchSet:
        patches = np.load(path / "patches.npy").astype(np.float32)
        index = pd.read_csv(path / "index.csv")
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        return cls(
            patches=patches,
            index=index,
            band_indices=[int(v) for v in meta.get("band_indices", [])],
            wavelength_nm=[float(v) for v in meta.get("wavelength_nm", [])],
            meta=meta,
        )


def extract_patches(
    cube: np.ndarray,
    label_map: np.ndarray,
    patch_size: int = 15,
    stride: int = 8,
    band_indices: list[int] | None = None,
    wavelength_nm: list[float] | None = None,
    image_id: str = "image_0",
    class_map: dict[int, str] | None = None,
    split_ratios: dict[str, float] | None = None,
    seed: int = 0,
) -> PatchSet:
    """Cut patches around every labelled pixel of ``label_map``.

    ``label_map`` holds connected-component IDs (> 0 = foreground); each ID is
    treated as one leaf/object (``leaf_id``).  ``class_map`` translates those
    IDs to class names; without it the IDs themselves become the labels.
    """
    cube = np.asarray(cube, dtype=np.float32)
    labels = np.asarray(label_map).astype(np.int32)
    if cube.ndim != 3 or labels.shape != cube.shape[:2]:
        raise ValueError(f"cube {cube.shape} / label_map {labels.shape} mismatch")
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd and >= 3, got {patch_size}")
    stride = max(1, int(stride))
    if band_indices is not None:
        cube = cube[:, :, np.asarray(band_indices, dtype=int)]

    h, w, bands = cube.shape
    pad = patch_size // 2
    padded = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    rows: list[dict[str, Any]] = []
    patches: list[np.ndarray] = []
    for r in range(0, h, stride):
        for c in range(0, w, stride):
            leaf = int(labels[r, c])
            if leaf <= 0:
                continue
            patch = padded[r : r + patch_size, c : c + patch_size, :]
            patches.append(np.ascontiguousarray(np.transpose(patch, (2, 0, 1))))
            rows.append(
                {
                    "sample_id": f"{image_id}_r{r}_c{c}",
                    "image_id": image_id,
                    "row": r,
                    "col": c,
                    "leaf_id": leaf,
                    "label": (
                        str(class_map[leaf]) if class_map and leaf in class_map else str(leaf)
                    ),
                }
            )

    if not patches:
        raise ValueError("no labelled pixels found — check the mask/label map")

    index = pd.DataFrame(rows)
    classes = index["label"].to_numpy()
    assignment = grouped_split(index["leaf_id"].to_numpy(), classes, ratios=split_ratios, seed=seed)
    index["split"] = [assignment[int(v)] for v in index["leaf_id"]]
    assert_no_group_leakage(index)

    arr = np.stack(patches).astype(np.float32)
    train_idx = np.flatnonzero(index["split"].to_numpy() == "train")
    train_patches = arr[train_idx]
    mean = train_patches.mean(axis=(0, 2, 3)).astype(np.float32)
    std = train_patches.std(axis=(0, 2, 3)).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    used_bands = [int(v) for v in band_indices] if band_indices is not None else list(range(bands))
    source_wl = [float(v) for v in (wavelength_nm or [])]
    wl = [source_wl[i] for i in used_bands if 0 <= i < len(source_wl)]
    meta: dict[str, Any] = {
        "patch_size": int(patch_size),
        "stride": int(stride),
        "band_indices": used_bands,
        "wavelength_nm": wl,
        "image_id": image_id,
        "n_samples": int(arr.shape[0]),
        "n_bands": int(arr.shape[1]),
        "classes": [str(v) for v in sorted(index["label"].unique())],
        "n_leaves": int(index["leaf_id"].nunique()),
        "split_counts": {split: int((index["split"] == split).sum()) for split in SPLITS},
        "leaf_splits": {
            split: sorted({int(v) for v in index.loc[index["split"] == split, "leaf_id"].unique()})
            for split in SPLITS
        },
        "band_mean": mean.tolist(),
        "band_std": std.tolist(),
        "seed": int(seed),
    }
    return PatchSet(
        patches=arr,
        index=index,
        band_indices=used_bands,
        wavelength_nm=wl,
        meta=meta,
    )


def band_stats(patches: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std over ``patches[indices]`` (no data leakage)."""
    subset = np.asarray(patches)[np.asarray(indices, dtype=int)]
    mean = subset.mean(axis=(0, 2, 3)).astype(np.float32)
    std = subset.std(axis=(0, 2, 3)).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def normalize_patches(patches: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    scaled = (np.asarray(patches, dtype=np.float32) - mean[:, None, None]) / std[:, None, None]
    return np.asarray(scaled, dtype=np.float32)
