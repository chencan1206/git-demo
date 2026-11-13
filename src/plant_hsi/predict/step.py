"""``predict.map``: sliding-window classification map for a whole cube."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.registry import Step
from plant_hsi.dl.bundle import load_bundle
from plant_hsi.modeling.step import fingerprint, jsonable, relative_uri
from plant_hsi.predict.sliding_window import predict_full_image, seam_ratio

__all__ = ["InferMapStep"]


def _read_cube(art: ArtifactRef, root: Path) -> np.ndarray:
    arr = zarr.open_array(str(art.resolve(root)), mode="r")
    return np.asarray(arr[:]).astype(np.float32)


class InferMapStep(Step):
    """Classify every pixel of a CUBE with a trained DC²Net MODEL.

    Params
        stride : int — default ``patch_size // 2``
        batch_size : int — default 64

    Outputs
        MASK  — label map (int32 Zarr)
        OTHER — confidence GeoTIFF, class-count CSV, inference metadata JSON
    """

    name = "predict.map"
    input_kinds = [ArtifactKind.MODEL, ArtifactKind.CUBE]
    output_kinds = [ArtifactKind.MASK, ArtifactKind.OTHER]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        model_art = next((a for a in inputs if a.kind == ArtifactKind.MODEL), None)
        cube_art = next((a for a in inputs if a.kind == ArtifactKind.CUBE), None)
        if model_art is None or cube_art is None:
            raise ValueError("predict.map expects a MODEL and a CUBE input")

        root = out_dir.parent
        bundle = load_bundle(model_art.resolve(root))
        cube = _read_cube(cube_art, root)
        card = bundle.get("card") or {}
        patch_size = int(card.get("patch_size") or 15)
        band_indices = card.get("band_indices")
        stride = int(params.get("stride", max(1, patch_size // 2)))
        classes: list[str] = [str(c) for c in bundle["classes"]]

        labels, prob_map = predict_full_image(
            cube,
            bundle,
            patch_size=patch_size,
            stride=stride,
            band_indices=band_indices,
            batch_size=int(params.get("batch_size", 64)),
        )
        confidence = prob_map.max(axis=2).astype(np.float32)
        ratio = seam_ratio(prob_map, patch_size, stride)

        fp = fingerprint(self.name, inputs, params, seed)
        base = f"{cube_art.name}_{model_art.name}_{fp}"
        label_dir = out_dir / f"{base}_classmap.zarr"
        z = zarr.open_array(
            str(label_dir), mode="w", zarr_format=2, shape=labels.shape, dtype="int32"
        )
        z[:] = labels.astype(np.int32)

        conf_path = out_dir / f"{base}_confidence.tif"
        from plant_hsi.export.raster import write_geotiff

        write_geotiff(conf_path, confidence)

        counts = {
            (classes[i] if i < len(classes) else str(i)): int((labels == i).sum())
            for i in range(len(classes))
        }
        summary_path = out_dir / f"{base}_class_counts.csv"
        pd.DataFrame({"class": list(counts), "pixels": list(counts.values())}).to_csv(
            summary_path, index=False, encoding="utf-8-sig"
        )

        meta_path = out_dir / f"{base}_inference.json"
        meta_path.write_text(
            json.dumps(
                jsonable(
                    {
                        "model": model_art.name,
                        "cube": cube_art.name,
                        "patch_size": patch_size,
                        "stride": stride,
                        "seam_ratio": ratio,
                        "class_counts": counts,
                        "shape": list(labels.shape),
                    }
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return [
            ArtifactRef(
                id="",
                kind=ArtifactKind.MASK,
                name=f"{base}_classmap",
                uri=relative_uri(label_dir, root),
                parents=[model_art.id, cube_art.id],
                metadata=jsonable(
                    {
                        "format": "classification_map",
                        "shape": list(labels.shape),
                        "classes": classes,
                        "class_counts": counts,
                        "patch_size": patch_size,
                        "stride": stride,
                        "seam_ratio": ratio,
                        "confidence_uri": relative_uri(conf_path, root),
                        "summary_uri": relative_uri(summary_path, root),
                    }
                ),
            ),
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{base}_confidence",
                uri=relative_uri(conf_path, root),
                parents=[model_art.id, cube_art.id],
                metadata={"format": "geotiff"},
            ),
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{base}_inference",
                uri=relative_uri(meta_path, root),
                parents=[model_art.id, cube_art.id],
                metadata={"format": "json", "seam_ratio": ratio},
            ),
        ]
