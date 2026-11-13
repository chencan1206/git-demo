"""Segmentation :class:`Step` implementations for the default registry.

Two steps are registered:

* ``segment.threshold``          —  similarity map → binary mask
* ``segment.connected_components``  —  binary mask → labelled ROI map + CSV
* ``segment.full_pipeline``      —  CUBE → similarity → mask → label map + CSV
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.registry import Step
from plant_hsi.segment.core import (
    binarize,
    connected_components,
    mean_spectrum_in_box,
    morphological_cleanup,
    similarity_map,
)
from plant_hsi.segment.roi import ROISet, compute_mean_spectra

# ------------------------------------------------------------------- helpers


def _json_dumps(d: dict[str, Any]) -> str:
    class _Np(json.JSONEncoder):
        def default(self, o: object) -> object:
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, tuple):
                return list(o)
            return super().default(o)

    return json.dumps(d, indent=2, cls=_Np)


def _read_cube(art: ArtifactRef, project_root: Path) -> tuple[np.ndarray, list[float] | None]:
    """Open a CUBE artifact's Zarr store and return ``(cube, wavelengths)``."""
    import zarr  # noqa: F811 — already imported above but make explicit

    zarr_dir = art.resolve(project_root)
    arr = zarr.open_array(str(zarr_dir), mode="r")
    cube = np.asarray(arr[:]).astype(np.float32)
    wl: list[float] | None = None
    meta_path = zarr_dir / ".phs_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            wl = [float(v) for v in meta.get("wavelength_nm", [])]
        except Exception:  # noqa: BLE001
            wl = None
    return cube, wl


def _write_zarr_mask(path: Path, arr: np.ndarray, dtype: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = dtype or arr.dtype.name
    z = zarr.open_array(str(path), mode="w", zarr_format=2, shape=arr.shape, dtype=dt)
    z[:] = arr.astype(dt)
    return path


# ----------------------------------------------------------- threshold step


class ThresholdStep(Step):
    """Compute a similarity map and threshold it into a binary mask.

    Inputs
        * 1 CUBE artifact (the hyperspectral cube)

    Params (all optional except ref_col/row/width/height if ref_source=box)
        method : "ra" (SAM) | "de" (SID) — default "ra"
        threshold_method : "otsu" | "percentile" | "fixed" — default "otsu"
        threshold_value : float — required for percentile / fixed
        ref_source : "box" | "spectra" — how to get the reference spectrum
        ref_col, ref_row, ref_width, ref_height : int — bounding-box source
        ref_spectrum : list[float] — explicit reference spectrum
        morph_kernel : int — default 3
        morph_open_iter : int — default 1
        morph_close_iter : int — default 2

    Outputs
        * 1 MASK artifact — binary mask as uint8 Zarr
    """

    name = "segment.threshold"
    input_kinds = [ArtifactKind.CUBE]
    output_kinds = [ArtifactKind.MASK]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        if len(inputs) != 1:
            raise ValueError(f"expected 1 CUBE input, got {len(inputs)}")

        cube, _ = _read_cube(inputs[0], out_dir.parent)
        h, w, b = cube.shape

        # 1. Reference spectrum
        ref_source = params.get("ref_source", "box")
        if ref_source == "spectra":
            ref = np.asarray(params["ref_spectrum"], dtype=np.float32)
            if ref.size != b:
                raise ValueError(f"ref_spectrum length {ref.size} != bands {b}")
        elif ref_source == "box":
            ref = mean_spectrum_in_box(
                cube,
                col=int(params.get("ref_col", w // 2)),
                row=int(params.get("ref_row", h // 2)),
                width=int(params.get("ref_width", 10)),
                height=int(params.get("ref_height", 10)),
            )
        else:
            raise ValueError(f"unknown ref_source {ref_source!r}")

        # 2. Similarity map
        sim_method = params.get("method", "ra")
        sim = similarity_map(cube, ref, method=sim_method)

        # 3. Threshold
        thr_method = params.get("threshold_method", "otsu")
        thr_value = params.get("threshold_value")
        binary, thr_used = binarize(sim, method=thr_method, threshold=thr_value)

        # 4. Morphology
        kernel = int(params.get("morph_kernel", 3))
        open_iter = int(params.get("morph_open_iter", 1))
        close_iter = int(params.get("morph_close_iter", 2))
        mask = morphological_cleanup(
            binary, kernel_size=kernel, open_iter=open_iter, close_iter=close_iter
        )

        # 5. Persist
        out_dir.mkdir(parents=True, exist_ok=True)
        base = inputs[0].name
        mask_path = out_dir / f"{base}_mask.zarr"
        _write_zarr_mask(mask_path, mask, dtype="uint8")

        meta_json = out_dir / f"{base}_mask.meta.json"
        meta_json.write_text(
            _json_dumps(
                {
                    "source_cube": inputs[0].uri,
                    "sim_method": sim_method,
                    "threshold_method": thr_method,
                    "threshold_value_used": thr_used,
                    "morph_kernel": kernel,
                    "morph_open_iter": open_iter,
                    "morph_close_iter": close_iter,
                    "foreground_pixels": int(mask.sum()),
                    "total_pixels": int(mask.size),
                    "similarity_min": float(np.min(sim)),
                    "similarity_max": float(np.max(sim)),
                    "similarity_mean": float(np.mean(sim)),
                    "ref_source": ref_source,
                    "ref_spectrum": ref.tolist(),
                }
            ),
            encoding="utf-8",
        )

        art = ArtifactRef(
            id="",
            kind=ArtifactKind.MASK,
            name=f"{base}_mask",
            uri=str(mask_path.relative_to(out_dir.parent)).replace("\\", "/"),
            parents=[inputs[0].id],
            metadata={
                "shape": list(mask.shape),
                "foreground_pixels": int(mask.sum()),
                "threshold": float(thr_used),
                "method": sim_method,
            },
        )
        return [art]


# ------------------------------------------------ connected components step


class ConnectedComponentsStep(Step):
    """Turn a binary MASK into a labelled ROI map + label_map.csv.

    Inputs
        * 1 MASK artifact
        * 1 CUBE artifact (optional, needed for per-ROI mean spectra)

    Params
        min_area : int — default 20
        connectivity : int — 4 or 8, default 8
        sort_by : str — default "area_desc"

    Outputs
        * 1 MASK artifact — label map as int32 Zarr
        * 1 SPECTRA artifact — label_map.csv
    """

    name = "segment.connected_components"
    input_kinds = [ArtifactKind.MASK]
    output_kinds = [ArtifactKind.MASK, ArtifactKind.SPECTRA]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        if len(inputs) < 1:
            raise ValueError("expected at least a MASK input")

        project_root = out_dir.parent
        mask_art = inputs[0]
        mask_dir = mask_art.resolve(project_root)
        mask_arr = np.asarray(zarr.open_array(str(mask_dir), mode="r")[:]).astype(np.uint8)

        # Optional CUBE for mean spectra
        cube: np.ndarray | None = None
        wl: list[float] | None = None
        for art in inputs[1:]:
            if art.kind == ArtifactKind.CUBE:
                cube, wl = _read_cube(art, project_root)
                break

        min_area = int(params.get("min_area", 20))
        connectivity = int(params.get("connectivity", 8))
        sort_by = params.get("sort_by", "area_desc")

        comp = connected_components(mask_arr, connectivity=connectivity, min_area=min_area)
        roi_set = ROISet.from_components(comp, sort_by=sort_by, labels=comp.labels)

        # Write outputs
        out_dir.mkdir(parents=True, exist_ok=True)
        base = mask_art.name.removesuffix("_mask") or "seg"

        label_zarr = out_dir / f"{base}_labels.zarr"
        _write_zarr_mask(label_zarr, comp.labels, dtype="int32")

        mean_spectra: np.ndarray | None = None
        if cube is not None and len(roi_set) > 0:
            ids_for_spec = [r.id for r in roi_set]
            mean_spectra = compute_mean_spectra(cube, comp.labels, ids_for_spec)

        csv_path = out_dir / f"{base}_label_map.csv"
        roi_set.write_csv(csv_path, mean_spectra=mean_spectra, wavelength_nm=wl)

        meta_json = out_dir / f"{base}_cc.meta.json"
        meta_json.write_text(
            _json_dumps(
                {
                    "source_mask": mask_art.uri,
                    "min_area": min_area,
                    "connectivity": connectivity,
                    "sort_by": sort_by,
                    "n_components": comp.n_components,
                    "areas": comp.areas.tolist(),
                }
            ),
            encoding="utf-8",
        )

        out: list[ArtifactRef] = []
        out.append(
            ArtifactRef(
                id="",
                kind=ArtifactKind.MASK,
                name=f"{base}_labels",
                uri=str(label_zarr.relative_to(project_root)).replace("\\", "/"),
                parents=[mask_art.id],
                metadata={
                    "shape": list(comp.labels.shape),
                    "n_components": comp.n_components,
                    "label_type": "roi_labels",
                },
            )
        )
        out.append(
            ArtifactRef(
                id="",
                kind=ArtifactKind.SPECTRA,
                name=f"{base}_label_map",
                uri=str(csv_path.relative_to(project_root)).replace("\\", "/"),
                parents=[mask_art.id],
                metadata={
                    "n_rois": len(roi_set),
                    "has_mean_spectra": mean_spectra is not None,
                    "csv_columns": list(roi_set.to_dataframe().columns),
                },
            )
        )
        return out
