"""Common data structures for hyperspectral cubes + IO dispatch.

``DatasetSpec`` is the shared schema across all readers. ``read`` is the
single entry point that dispatches on extension. ``write_zarr`` converts
to the Zarr workspace format (float32).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------- types


@dataclass
class DatasetSpec:
    """All metadata we keep about an imported hyperspectral cube."""

    source_path: str  # absolute, original file path
    format: str  # "mat" / "envi" / "geotiff"
    shape: tuple[int, int, int]  # (y, x, band)
    wavelength_nm: list[float]  # per-band center wavelength
    fwhm_nm: float | None = None  # single value, None if unknown
    dtype: str = "float32"  # dtype of workspace array
    sensor: str = ""
    white_ref_hash: str = ""  # empty if not available
    dark_ref_hash: str = ""
    raw_hash: str = ""  # sha256 of source file
    notes: str = ""  # e.g. "indeterminate wavelengths, assumed 400-950 1nm"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HealthReport:
    """Per-quality-check results for a cube."""

    source_path: str = ""
    shape: tuple[int, int, int] = (0, 0, 0)
    pixel_count: int = 0
    nan_count: int = 0
    inf_count: int = 0
    out_of_range_count: int = 0  # 反射率通常应为 [0, 1]
    all_zero_bands: list[int] = field(default_factory=list)
    all_zero_rows: list[int] = field(default_factory=list)  # 空间 y 方向
    all_zero_cols: list[int] = field(default_factory=list)  # 空间 x 方向
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0

    @property
    def ok(self) -> bool:
        return (
            self.nan_count == 0
            and self.inf_count == 0
            and self.out_of_range_count == 0
            and not self.all_zero_bands
            and not self.all_zero_rows
            and not self.all_zero_cols
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d


# ---------------------------------------------------------------- read


def read(
    path: str | Path, wavelength_nm: list[float] | None = None
) -> tuple[np.ndarray, DatasetSpec]:
    """Dispatch on file extension; return ``(cube, spec)``.

    ``wavelength_nm`` is an optional override when the source file doesn't
    carry wavelength metadata (Zenodo Javidan data is one example).

    This materialises the whole cube; use :func:`open_stream` for large files
    that should be imported block-by-block.
    """
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".mat":
        cube, spec = _read_mat(p)
    elif suf in (".hdr", ".raw", ".bil", ".bsq"):
        cube, spec = _read_envi(p)
    elif suf in (".tif", ".tiff"):
        cube, spec = _read_geotiff(p)
    else:
        raise ValueError(f"unsupported extension: {suf}")

    _apply_wavelength_override(spec, cube.shape[2], wavelength_nm)
    return cube.astype(np.float32, copy=False), spec


def _apply_wavelength_override(
    spec: DatasetSpec, n_bands: int, wavelength_nm: list[float] | None
) -> None:
    """Apply an explicit wavelength list, and fall back to a 400–950 nm guess
    when the band count still doesn't line up."""
    if wavelength_nm is not None:
        spec.wavelength_nm = list(wavelength_nm)
    if len(spec.wavelength_nm) != n_bands:
        spec.wavelength_nm = list(np.linspace(400, 950, n_bands))
        spec.notes = (spec.notes + " auto-assumed 400-950 nm; confirm wavelengths;").strip()


# --------------------------------------------------------------- health report


def health_report(cube: np.ndarray, source_path: str = "") -> HealthReport:
    """Scan a 3-D ``(y, x, band)`` cube for common defects."""
    assert cube.ndim == 3, f"expected 3-D cube, got {cube.ndim}-D"
    h, w, bands = cube.shape
    arr = np.asarray(cube)
    total = arr.size

    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    bad_mask = np.isnan(arr) | np.isinf(arr)
    finite = arr[~bad_mask]
    oor = int(np.sum((finite < 0) | (finite > 1))) if finite.size else 0

    all_zero_bands = [int(b) for b in range(bands) if np.all(arr[:, :, b] == 0)]
    all_zero_rows = [int(y) for y in range(h) if np.all(arr[y, :, :] == 0)]
    all_zero_cols = [int(x) for x in range(w) if np.all(arr[:, x, :] == 0)]

    return HealthReport(
        source_path=source_path,
        shape=(h, w, bands),
        pixel_count=int(total),
        nan_count=int(nan_mask.sum()),
        inf_count=int(inf_mask.sum()),
        out_of_range_count=oor,
        all_zero_bands=all_zero_bands,
        all_zero_rows=all_zero_rows,
        all_zero_cols=all_zero_cols,
        min=float(finite.min()) if finite.size else 0.0,
        max=float(finite.max()) if finite.size else 0.0,
        mean=float(finite.mean()) if finite.size else 0.0,
    )


# ---------------------------------------------------------- wavelength util


def parse_wavelength_from_text(text: str, n_bands: int) -> list[float]:
    """Parse ENVI-style ``wavelength = {400.0, 410.0, ...}`` or a plain list
    and return ``n_bands`` floats. Falls back to 400-950 linear."""
    nums = re.findall(r"-?\d+\.?\d*", text)
    vals = [float(v) for v in nums]
    if len(vals) >= n_bands:
        return vals[:n_bands]
    if len(vals) >= 2:
        return list(np.linspace(vals[0], vals[-1], n_bands))
    return list(np.linspace(400, 950, n_bands))


def guess_wavelength_from_range(start_nm: float, end_nm: float, n_bands: int) -> list[float]:
    return list(np.linspace(start_nm, end_nm, n_bands))


# ------------------------------------------------------------- block streaming


@dataclass
class CubeStream:
    """A cube that can be read in bounded-memory row blocks.

    ``_reader(y0, y1)`` must return the ``(y1-y0, width, band)`` slab. For
    memory-mapped formats (ENVI, uncompressed GeoTIFF) the returned array is
    backed by the file; for ``.mat`` it is a view of the already-materialised
    cube (MAT v5 compression prevents random access).
    """

    spec: DatasetSpec
    shape: tuple[int, int, int]
    _reader: Callable[[int, int], np.ndarray]

    def iter_blocks(self, target_bytes: int = 32 << 20) -> Iterator[tuple[int, np.ndarray]]:
        h, w, bands = self.shape
        row_bytes = max(1, w * bands * 4)
        rows = max(1, target_bytes // row_bytes)
        for y0 in range(0, h, rows):
            y1 = min(y0 + rows, h)
            yield y0, np.asarray(self._reader(y0, y1), dtype=np.float32)


def open_stream(path: str | Path, wavelength_nm: list[float] | None = None) -> CubeStream:
    """Open ``path`` without materialising the whole cube (where possible)."""
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".mat":
        cube, spec = _read_mat(p)
        stream = CubeStream(spec, cube.shape, lambda y0, y1: cube[y0:y1])
    elif suf in (".hdr", ".raw", ".bil", ".bsq"):
        spec, reader, shape = _open_envi_stream(p)
        stream = CubeStream(spec, shape, reader)
    elif suf in (".tif", ".tiff"):
        spec, reader, shape = _open_geotiff_stream(p)
        stream = CubeStream(spec, shape, reader)
    else:
        raise ValueError(f"unsupported extension: {suf}")

    _apply_wavelength_override(stream.spec, stream.shape[2], wavelength_nm)
    return stream


class HealthAccumulator:
    """Incremental equivalent of :func:`health_report` for block streams.

    Holds only O(width + height + bands) state, so a 10 GB cube can be
    inspected with a handful of megabytes of RAM.
    """

    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.shape = shape
        h, w, b = shape
        self.nan_count = 0
        self.inf_count = 0
        self.out_of_range_count = 0
        self._sum = 0.0
        self._finite = 0
        self._min = np.inf
        self._max = -np.inf
        self._band_nonzero = np.zeros(b, dtype=bool)
        self._row_nonzero = np.zeros(h, dtype=bool)
        self._col_nonzero = np.zeros(w, dtype=bool)

    def update(self, y0: int, block: np.ndarray) -> None:
        block = np.asarray(block)
        y1 = y0 + block.shape[0]
        self.nan_count += int(np.isnan(block).sum())
        self.inf_count += int(np.isinf(block).sum())
        finite = block[np.isfinite(block)]
        if finite.size:
            self.out_of_range_count += int(np.count_nonzero((finite < 0) | (finite > 1)))
            self._min = min(self._min, float(finite.min()))
            self._max = max(self._max, float(finite.max()))
            self._sum += float(finite.sum())
            self._finite += int(finite.size)
        nonzero = block != 0
        self._band_nonzero |= np.any(nonzero, axis=(0, 1))
        self._row_nonzero[y0:y1] |= np.any(nonzero, axis=(1, 2))
        self._col_nonzero |= np.any(nonzero, axis=(0, 2))

    def finish(self, source_path: str = "") -> HealthReport:
        h, w, b = self.shape
        return HealthReport(
            source_path=source_path,
            shape=self.shape,
            pixel_count=h * w * b,
            nan_count=self.nan_count,
            inf_count=self.inf_count,
            out_of_range_count=self.out_of_range_count,
            all_zero_bands=[int(i) for i in np.flatnonzero(~self._band_nonzero)],
            all_zero_rows=[int(i) for i in np.flatnonzero(~self._row_nonzero)],
            all_zero_cols=[int(i) for i in np.flatnonzero(~self._col_nonzero)],
            min=self._min if self._finite else 0.0,
            max=self._max if self._finite else 0.0,
            mean=(self._sum / self._finite) if self._finite else 0.0,
        )


# ------------------------------------------------------------- zarr convert


def write_zarr(
    cube: np.ndarray, out_path: Path, spec: DatasetSpec, health: HealthReport | None = None
) -> Path:
    """Write an in-memory ``cube`` to a Zarr v2 store; return the directory.

    Kept as the convenience wrapper for small arrays. Large imports should go
    through :func:`write_zarr_stream` + :func:`open_stream` so the cube is
    never fully resident.
    """
    stream = CubeStream(spec=spec, shape=cube.shape, _reader=lambda y0, y1: cube[y0:y1])
    write_zarr_stream(stream, out_path, health=health)
    return Path(out_path)


def write_zarr_stream(
    stream: CubeStream,
    out_path: Path,
    target_bytes: int = 32 << 20,
    health: HealthReport | None = None,
) -> tuple[DatasetSpec, HealthReport]:
    """Write ``stream`` to Zarr block-by-block; return ``(spec, health)``.

    The store is built in a sibling temp directory and atomically renamed into
    place, so an interrupted import never leaves a half-written ``.zarr`` that
    looks complete.
    """
    import shutil

    import numcodecs
    import zarr  # lazy import — not used elsewhere

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.tmp-{uuid.uuid4().hex[:8]}")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

    h, w, bands = stream.shape
    if min(stream.shape) <= 0:
        raise ValueError(f"cannot write empty cube with shape {stream.shape}")

    arr = zarr.open_array(
        str(tmp),
        mode="w",
        zarr_format=2,
        shape=(h, w, bands),
        chunks=(min(h, 64), min(w, 64), min(bands, 256)),
        dtype=np.float32,
        compressor=numcodecs.Blosc(cname="lz4", clevel=5),
    )

    acc = HealthAccumulator(stream.shape)
    for y0, block in stream.iter_blocks(target_bytes):
        arr[y0 : y0 + block.shape[0]] = block
        if health is None:
            acc.update(y0, block)
    hrep = health if health is not None else acc.finish(stream.spec.source_path)

    meta = {
        "schema": "phs-dataset-1.0",
        "dims": ("y", "x", "band"),
        "wavelength_nm": stream.spec.wavelength_nm,
        "fwhm_nm": stream.spec.fwhm_nm,
        "format": stream.spec.format,
        "source_path": stream.spec.source_path,
        "raw_hash": stream.spec.raw_hash,
        "health": hrep.to_dict(),
        "notes": stream.spec.notes,
    }
    (tmp / ".phs_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _publish_dir(tmp, out_path)
    return stream.spec, hrep


def _publish_dir(tmp: Path, final: Path) -> None:
    """Atomically move ``tmp`` to ``final``, replacing any previous store."""
    import shutil

    if not final.exists():
        os.replace(tmp, final)
        return
    old = final.with_name(f".{final.name}.old-{uuid.uuid4().hex[:8]}")
    os.replace(final, old)
    try:
        os.replace(tmp, final)
    finally:
        shutil.rmtree(old, ignore_errors=True)


# ------------------------------------------------------- format-specific readers


def _read_mat(path: Path) -> tuple[np.ndarray, DatasetSpec]:
    """Read MATLAB v5 (scipy) or v7.3 (h5py). Auto-detects the real 3-D
    numeric variable — Zenodo Javidan files call it ``Day_one_1`` etc."""
    from plant_hsi.io.mat_reader import load_mat_cube  # local import

    cube, varname, meta = load_mat_cube(path)
    h, w, b = cube.shape

    # Try to pull wavelength / fwhm out of any metadata variables; otherwise
    # leave wavelength_nm empty and let caller override or guess.
    wl: list[float] = []
    fwhm: float | None = None
    for k, v in meta.items():
        kl = k.lower()
        if ("wave" in kl or "band" in kl) and hasattr(v, "__len__") and len(v) >= b:
            arr = np.asarray(v).ravel()[:b]
            wl = [float(x) for x in arr]
            break
        if "fwhm" in kl and hasattr(v, "__len__"):
            fwhm = float(np.asarray(v).ravel()[0])

    if not wl:
        wl = guess_wavelength_from_range(400, 950, b)

    raw_hash = _sha256(path)
    notes = "MATLAB .mat cube; variable=" + varname
    if not wl:
        notes += "; wavelengths unknown, defaulted to 400-950 nm 1 nm"

    spec = DatasetSpec(
        source_path=str(path.resolve()),
        format="mat",
        shape=(h, w, b),
        wavelength_nm=wl,
        fwhm_nm=fwhm,
        raw_hash=raw_hash,
        notes=notes,
    )
    return cube, spec


def _read_envi(path: Path) -> tuple[np.ndarray, DatasetSpec]:
    spec, reader, shape = _open_envi_stream(path)
    return reader(0, shape[0]).astype(np.float32, copy=False), spec


def _open_envi_stream(
    path: Path,
) -> tuple[DatasetSpec, Callable[[int, int], np.ndarray], tuple[int, int, int]]:
    """Open an ENVI file as a memmap-backed image (no ``.load()``)."""
    import spectral.io.envi as envi  # noqa: WPS433

    p = path
    hdr = p if p.suffix.lower() == ".hdr" else Path(str(p) + ".hdr")
    img = envi.open(str(hdr))
    h, w, b = (int(v) for v in img.shape[:3])

    text = hdr.read_text(encoding="utf-8", errors="ignore")
    wl = parse_wavelength_from_text(_envi_get(text, "wavelength", ""), b)
    fwhm_str = _envi_get(text, "fwhm", "")
    fwhm_vals = re.findall(r"-?\d+\.?\d*", fwhm_str)
    fwhm = float(fwhm_vals[0]) if fwhm_vals else None

    spec = DatasetSpec(
        source_path=str(hdr.resolve()),
        format="envi",
        shape=(h, w, b),
        wavelength_nm=wl,
        fwhm_nm=fwhm,
        raw_hash=_sha256(hdr),
    )

    def _reader(y0: int, y1: int) -> np.ndarray:
        return np.asarray(img.read_subregion((y0, y1), (0, w)))

    return spec, _reader, (h, w, b)


def _read_geotiff(path: Path) -> tuple[np.ndarray, DatasetSpec]:
    spec, reader, shape = _open_geotiff_stream(path)
    return reader(0, shape[0]).astype(np.float32, copy=False), spec


def _open_geotiff_stream(
    path: Path,
) -> tuple[DatasetSpec, Callable[[int, int], np.ndarray], tuple[int, int, int]]:
    """Open a GeoTIFF, preferring ``tifffile.memmap`` (uncompressed only)."""
    import tifffile  # noqa: WPS433

    arr: np.ndarray
    try:
        arr = tifffile.memmap(str(path))
    except Exception:  # noqa: BLE001 — compressed / tiled files can't be mapped
        arr = tifffile.imread(str(path))

    # tifffile returns whatever layout was stored. We accept:
    #   (H, W)     → single-band
    #   (H, W, B)  → bands last (tifffile default for multi-band write)
    #   (B, H, W)  → bands first (GDAL-style, rare but possible)
    if arr.ndim == 2:
        h, w = arr.shape
        b = 1

        def _reader2(y0: int, y1: int) -> np.ndarray:
            return np.asarray(arr[y0:y1])[:, :, None]

        reader: Callable[[int, int], np.ndarray] = _reader2
    elif arr.ndim == 3:
        s0, s1, s2 = arr.shape
        # Heuristic: band count is almost always the smallest axis when we
        # have many more spatial pixels than bands. If axis-0 is the
        # smallest and strictly smaller than the others, treat it as band.
        if s0 < s1 and s0 < s2:
            b, h, w = s0, s1, s2

            def _reader_first(y0: int, y1: int) -> np.ndarray:
                return np.moveaxis(np.asarray(arr[:, y0:y1, :]), 0, -1)

            reader = _reader_first
        else:
            h, w, b = s0, s1, s2

            def _reader_last(y0: int, y1: int) -> np.ndarray:
                return np.asarray(arr[y0:y1])

            reader = _reader_last
    else:
        raise ValueError(f"GeoTIFF must be 2D or 3D, got {arr.shape}")

    wl = guess_wavelength_from_range(400, 950, b)
    spec = DatasetSpec(
        source_path=str(path.resolve()),
        format="geotiff",
        shape=(h, w, b),
        wavelength_nm=wl,
        raw_hash=_sha256(path),
    )
    return spec, reader, (h, w, b)


# ------------------------------------------------------------------ helpers


def _mat_header(path: Path) -> bytes:
    """Read just the first 128 bytes of a .mat file so we can sniff the
    version without loading the full (possibly large) file."""
    with path.open("rb") as f:
        return f.read(128)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _envi_get(text: str, key: str, default: str = "") -> str:
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.*)", text, re.MULTILINE)
    if not m:
        return default
    val = m.group(1).strip()
    if val.startswith("{"):
        # Collect closing brace across lines
        close = val.find("}")
        if close >= 0:
            val = val[:close]
        # If it's a list of numbers, flatten
        return val
    return val.split("#")[0].strip()
