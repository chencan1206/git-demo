"""Export steps: GeoTIFF rasters and portable UTF-8 CSV tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.registry import Step
from plant_hsi.export.raster import write_geotiff
from plant_hsi.modeling.step import fingerprint, jsonable, relative_uri
from plant_hsi.modeling.table import load_table


class ExportGeoTiffStep(Step):
    """Write a MASK artifact (Zarr label map) as a GeoTIFF.

    Params
        dtype : optional output dtype ("uint8" / "uint16" / "int32" / ...)
    """

    name = "export.geotiff"
    input_kinds = [ArtifactKind.MASK]
    output_kinds = [ArtifactKind.OTHER]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        mask_art = next((a for a in inputs if a.kind == ArtifactKind.MASK), None)
        if mask_art is None:
            raise ValueError("export.geotiff expects a MASK input")
        root = out_dir.parent

        import zarr

        arr = np.asarray(zarr.open_array(str(mask_art.resolve(root)), mode="r")[:])
        fp = fingerprint(self.name, inputs, params, seed)
        out_path = out_dir / f"{mask_art.name}_{fp}.tif"
        write_geotiff(out_path, arr, dtype=params.get("dtype"))

        return [
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{mask_art.name}_{fp}",
                uri=relative_uri(out_path, root),
                parents=[mask_art.id],
                metadata=jsonable(
                    {
                        "format": "geotiff",
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                        "source_mask": mask_art.id,
                    }
                ),
            )
        ]


class ExportCsvStep(Step):
    """Convert a SPECTRA / CSV / Parquet artifact into a UTF-8-BOM CSV.

    Excel and pandas both open UTF-8-BOM CSVs without mojibake, which matters
    for Chinese column headers.
    """

    name = "export.csv"
    input_kinds = [ArtifactKind.SPECTRA, ArtifactKind.OTHER]
    output_kinds = [ArtifactKind.OTHER]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        if len(inputs) != 1:
            raise ValueError(f"export.csv expects 1 input, got {len(inputs)}")
        art = inputs[0]
        root = out_dir.parent
        df = load_table(art.resolve(root))
        fp = fingerprint(self.name, inputs, params, seed)
        out_path = out_dir / f"{art.name}_{fp}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return [
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{art.name}_{fp}",
                uri=relative_uri(out_path, root),
                parents=[art.id],
                metadata=jsonable(
                    {
                        "format": "csv",
                        "n_rows": int(len(df)),
                        "n_columns": int(df.shape[1]),
                        "encoding": "utf-8-sig",
                    }
                ),
            )
        ]
