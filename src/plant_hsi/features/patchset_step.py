"""``features.patchset``: build a grouped 3-D patch sample set.

Lives outside :mod:`plant_hsi.dl` so the step is available even when the
optional PyTorch dependency is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.registry import Step
from plant_hsi.features.patchset import extract_patches
from plant_hsi.modeling.step import fingerprint, jsonable, relative_uri
from plant_hsi.segment.core import connected_components

__all__ = ["BuildPatchSetStep"]


def _read_cube(art: ArtifactRef, root: Path) -> tuple[np.ndarray, list[float]]:
    zarr_dir = art.resolve(root)
    cube = np.asarray(zarr.open_array(str(zarr_dir), mode="r")[:]).astype(np.float32)
    wavelength: list[float] = []
    meta_path = zarr_dir / ".phs_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            wavelength = [float(v) for v in meta.get("wavelength_nm", [])]
        except (json.JSONDecodeError, OSError):
            wavelength = []
    return cube, wavelength


def _read_label_map(path: Path, min_area: int) -> np.ndarray:
    arr = np.asarray(zarr.open_array(str(path), mode="r")[:])
    labels = arr.astype(np.int32)
    unique = np.unique(labels)
    if unique.size <= 2 and set(unique.tolist()) <= {0, 1}:
        comp = connected_components(labels.astype(np.uint8), connectivity=8, min_area=min_area)
        return comp.labels.astype(np.int32)
    return labels


class BuildPatchSetStep(Step):
    """Cut a grouped, split-annotated 3-D patch set from a cube + label map.

    Params
        patch_size : int (odd, default 15)
        stride : int (default 8)
        band_indices : list[int] | None — subset of bands (lightweight retrain)
        class_map : dict[str, str] — mask label id → class name
        split_ratios : {"train": .7, "val": .15, "test": .15}
        min_area : int — used when the mask is binary (default 20)
        image_id : str
    """

    name = "features.patchset"
    input_kinds = [ArtifactKind.CUBE, ArtifactKind.MASK]
    output_kinds = [ArtifactKind.OTHER]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        cube_art = next((a for a in inputs if a.kind == ArtifactKind.CUBE), None)
        mask_art = next((a for a in inputs if a.kind == ArtifactKind.MASK), None)
        if cube_art is None:
            raise ValueError("features.patchset expects a CUBE input")
        if mask_art is None:
            raise ValueError("features.patchset expects a MASK/label-map input")

        root = out_dir.parent
        cube, wavelength = _read_cube(cube_art, root)
        labels = _read_label_map(mask_art.resolve(root), int(params.get("min_area", 20)))
        if labels.shape != cube.shape[:2]:
            raise ValueError(f"mask {labels.shape} != cube spatial {cube.shape[:2]}")

        class_map_raw = params.get("class_map") or {}
        class_map = {int(k): str(v) for k, v in class_map_raw.items()}
        patch_size = int(params.get("patch_size", 15))
        stride = int(params.get("stride", 8))
        band_indices = params.get("band_indices")
        band_idx = [int(v) for v in band_indices] if band_indices else None

        ps = extract_patches(
            cube,
            labels,
            patch_size=patch_size,
            stride=stride,
            band_indices=band_idx,
            wavelength_nm=wavelength,
            image_id=str(params.get("image_id", "image_0")),
            class_map=class_map or None,
            split_ratios=params.get("split_ratios"),
            seed=seed,
        )
        fp = fingerprint(self.name, inputs, params, seed)
        base = f"{cube_art.name}_patchset_{fp}"
        ps.save(out_dir)

        meta = dict(ps.meta)
        meta.update(
            {
                "source_cube": cube_art.uri,
                "source_mask": mask_art.uri,
                "class_map": {str(k): v for k, v in class_map.items()},
                "seed": seed,
            }
        )
        (out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return [
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=base,
                uri=relative_uri(out_dir, root),
                parents=[cube_art.id, mask_art.id],
                metadata=jsonable(
                    {
                        "format": "patchset",
                        "patch_size": patch_size,
                        "stride": stride,
                        "n_samples": int(ps.patches.shape[0]),
                        "n_bands": int(ps.patches.shape[1]),
                        "band_indices": ps.band_indices,
                        "wavelength_nm": ps.wavelength_nm,
                        "classes": ps.classes,
                        "split_counts": meta["split_counts"],
                        "leaf_splits": meta["leaf_splits"],
                        "n_leaves": meta["n_leaves"],
                        "band_mean": meta["band_mean"],
                        "band_std": meta["band_std"],
                        "index_uri": relative_uri(out_dir / "index.csv", root),
                        "seed": seed,
                    }
                ),
            )
        ]
