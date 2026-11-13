"""IO layer: read raw data in multiple formats and produce :class:`DatasetSpec`.

All readers return the same on-disk schema: 3-D float32 NumPy array
``(y, x, band)`` plus wavelength / fwhm metadata. The heavy lifting of
Zarr conversion and the health report lives in :mod:`plant_hsi.io.spec`.
"""

from __future__ import annotations

from plant_hsi.io.spec import DatasetSpec, HealthReport, read, write_zarr

__all__ = ["DatasetSpec", "HealthReport", "read", "write_zarr"]
