"""Import a raw dataset (MAT / ENVI / GeoTIFF) into the project's Zarr workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.registry import Step
from plant_hsi.io.spec import open_stream, write_zarr_stream


class ImportDatasetStep(Step):
    """Import one raw hyperspectral file into the project.

    Inputs:
        * Exactly one ``RAW`` artifact whose ``uri`` points to the source file.

    Params:
        ``wavelength_nm`` (optional, list[float]): override wavelengths when
        the source carries none (common with the Zenodo Javidan dump).
    """

    name = "io.import_dataset"
    input_kinds = [ArtifactKind.RAW]
    output_kinds = [ArtifactKind.CUBE]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict,
        seed: int,
    ) -> list[ArtifactRef]:
        if len(inputs) != 1:
            raise ValueError(f"expected 1 input RAW artifact, got {len(inputs)}")

        src = inputs[0].resolve(out_dir.parent)
        if not src.exists():
            raise FileNotFoundError(f"source missing: {src}")

        wl_override = params.get("wavelength_nm")
        # Stream in bounded row blocks: ENVI / uncompressed GeoTIFF are
        # memory-mapped, so a 10 GB cube never has to be fully resident.
        stream = open_stream(src, wavelength_nm=wl_override)
        spec = stream.spec

        # Write Zarr workspace + JSON side-by-side with the output artifact.
        zarr_dir = out_dir / f"{src.stem}.zarr"
        _, hrep = write_zarr_stream(stream, zarr_dir)

        # Also write dataset spec + health report JSONs alongside for the UI.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{src.stem}.spec.json").write_text(
            _json_dumps(spec.to_dict()), encoding="utf-8"
        )
        (out_dir / f"{src.stem}.health.json").write_text(
            _json_dumps(hrep.to_dict()), encoding="utf-8"
        )

        art = ArtifactRef(
            id="",
            kind=ArtifactKind.CUBE,
            name=src.stem,
            uri=str(zarr_dir.relative_to(out_dir.parent)).replace("\\", "/"),
            parents=[inputs[0].id],
            metadata={
                "shape": list(spec.shape),
                "format": spec.format,
                "health_ok": hrep.ok,
                "notes": spec.notes,
            },
        )
        return [art]


def _json_dumps(d: dict) -> str:
    import json

    class _Np(json.JSONEncoder):
        def default(self, o):  # noqa: ANN001
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, tuple):
                return list(o)
            return super().default(o)

    return json.dumps(d, indent=2, cls=_Np)
