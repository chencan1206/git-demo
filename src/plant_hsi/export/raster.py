"""Raster export helpers (GeoTIFF via tifffile)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_geotiff(
    path: Path | str,
    array: np.ndarray,
    *,
    dtype: str | None = None,
    compression: str | None = "zlib",
) -> Path:
    """Write a 2-D label / mask array as a plain (unreferenced) GeoTIFF.

    The file has no CRS / geotransform (laboratory imagery usually has none);
    QGIS and ENVI open it as a raster with pixel size 1.  Georeferencing can be
    layered on later without changing the array layout.
    """
    import tifffile

    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D raster, got shape {arr.shape}")
    if dtype:
        arr = arr.astype(dtype)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out), arr, photometric="minisblack", compression=compression)
    return out
