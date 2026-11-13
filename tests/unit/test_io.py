"""Stage-1 IO tests using small synthetic datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.project import Project
from plant_hsi.io.spec import (
    DatasetSpec,
    HealthAccumulator,
    health_report,
    open_stream,
    read,
    write_zarr,
)


def _cube(shape: tuple[int, int, int] = (4, 5, 3)) -> np.ndarray:
    return np.linspace(0.1, 0.9, np.prod(shape), dtype=np.float32).reshape(shape)


def _write_envi(path: Path, cube: np.ndarray) -> Path:
    """Write a little-endian BSQ cube and its ENVI header."""
    rows, cols, bands = cube.shape
    raw = path.with_suffix(".raw")
    hdr = Path(f"{raw}.hdr")
    raw.write_bytes(cube.transpose(2, 0, 1).astype("<f4").tobytes())
    hdr.write_text(
        "\n".join(
            [
                "ENVI",
                f"samples = {cols}",
                f"lines = {rows}",
                f"bands = {bands}",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 4",
                "interleave = bsq",
                "byte order = 0",
                "wavelength = {450.0, 550.0, 650.0}",
                "fwhm = {5.0, 5.0, 5.0}",
            ]
        ),
        encoding="utf-8",
    )
    return hdr


def test_read_mat_selects_largest_cube_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "scene.mat"
    cube = _cube()
    savemat(
        path,
        {
            "fwhm": np.array([[5.0]]),
            "wavelength": np.array([450.0, 550.0, 650.0]),
            "decoy": np.ones((2, 2, 2), dtype=np.float32),
            "scene": cube,
        },
    )

    loaded, spec = read(path)

    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, cube)
    assert spec.format == "mat"
    assert spec.shape == cube.shape
    assert spec.wavelength_nm == [450.0, 550.0, 650.0]
    assert spec.fwhm_nm == 5.0
    assert "variable=scene" in spec.notes
    assert spec.source_path == str(path.resolve())
    assert len(spec.raw_hash) == 64


def test_read_mat_wavelength_override_and_fallback(tmp_path: Path) -> None:
    path = tmp_path / "scene.mat"
    savemat(path, {"scene": _cube()})

    _, overridden = read(path, wavelength_nm=[500.0, 600.0, 700.0])
    assert overridden.wavelength_nm == [500.0, 600.0, 700.0]

    _, fallback = read(path, wavelength_nm=[500.0])
    np.testing.assert_allclose(fallback.wavelength_nm, np.linspace(400, 950, 3))
    assert "auto-assumed 400-950 nm" in fallback.notes


def test_read_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "scene.xyz"
    path.write_bytes(b"not a hyperspectral cube")
    with pytest.raises(ValueError, match="unsupported extension"):
        read(path)


@pytest.mark.parametrize("input_suffix", [".hdr", ".raw"])
def test_read_envi_from_header_or_raw(tmp_path: Path, input_suffix: str) -> None:
    pytest.importorskip("spectral")
    cube = _cube()
    hdr = _write_envi(tmp_path / "scene.img", cube)
    source = hdr if input_suffix == ".hdr" else hdr.with_suffix("")

    loaded, spec = read(source)

    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, cube, rtol=1e-6)
    assert spec.format == "envi"
    assert spec.shape == cube.shape
    assert spec.wavelength_nm == [450.0, 550.0, 650.0]
    assert spec.fwhm_nm == 5.0
    assert spec.source_path == str(hdr.resolve())


@pytest.mark.parametrize("band_first", [False, True])
def test_read_geotiff_normalizes_band_axis(tmp_path: Path, band_first: bool) -> None:
    tifffile = pytest.importorskip("tifffile")
    cube = _cube((6, 7, 3))
    stored = np.moveaxis(cube, -1, 0) if band_first else cube
    path = tmp_path / f"scene_{band_first}.tif"
    tifffile.imwrite(path, stored, photometric="minisblack")

    loaded, spec = read(path)

    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, cube)
    assert spec.format == "geotiff"
    assert spec.shape == cube.shape
    np.testing.assert_allclose(spec.wavelength_nm, np.linspace(400, 950, 3))


def test_read_geotiff_single_band(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    image = np.arange(42, dtype=np.float32).reshape(6, 7) / 100
    path = tmp_path / "single.tiff"
    tifffile.imwrite(path, image)

    loaded, spec = read(path)

    assert loaded.shape == (6, 7, 1)
    np.testing.assert_array_equal(loaded[:, :, 0], image)
    assert spec.wavelength_nm == [400.0]


def test_health_report_accepts_valid_cube() -> None:
    cube = np.linspace(0, 1, 24, dtype=np.float32).reshape(2, 3, 4)

    report = health_report(cube, source_path="synthetic")

    assert report.ok
    assert report.source_path == "synthetic"
    assert report.shape == cube.shape
    assert report.pixel_count == cube.size
    assert report.nan_count == report.inf_count == report.out_of_range_count == 0
    assert report.all_zero_bands == []
    assert report.all_zero_rows == []
    assert report.all_zero_cols == []
    assert report.min == 0.0
    assert report.max == 1.0
    assert report.mean == pytest.approx(0.5)
    assert report.to_dict()["ok"] is True


def test_health_report_counts_nonfinite_and_out_of_range_values() -> None:
    cube = np.full((2, 2, 2), 0.5, dtype=np.float32)
    cube[0, 0, 0] = np.nan
    cube[0, 1, 1] = np.inf
    cube[1, 0, 0] = -0.2
    cube[1, 1, 1] = 1.2

    report = health_report(cube)

    assert not report.ok
    assert report.nan_count == 1
    assert report.inf_count == 1
    assert report.out_of_range_count == 2
    assert report.min == pytest.approx(-0.2)
    assert report.max == pytest.approx(1.2)


def test_health_report_finds_all_zero_bands_rows_and_cols() -> None:
    cube = np.full((3, 4, 3), 0.25, dtype=np.float32)
    cube[:, :, 1] = 0
    cube[1, :, :] = 0
    cube[:, 2, :] = 0

    report = health_report(cube)

    assert not report.ok
    assert report.all_zero_bands == [1]
    assert report.all_zero_rows == [1]
    assert report.all_zero_cols == [2]


def test_write_zarr_round_trip_with_metadata(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    cube = _cube()
    spec = DatasetSpec(
        source_path="synthetic",
        format="mat",
        shape=cube.shape,
        wavelength_nm=[450.0, 550.0, 650.0],
        fwhm_nm=5.0,
    )
    health = health_report(cube)
    out = tmp_path / "nested" / "scene.zarr"

    assert write_zarr(cube, out, spec, health) == out

    stored = zarr.open_array(str(out), mode="r")
    assert stored.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(stored[:], cube)
    meta = json.loads((out / ".phs_meta.json").read_text(encoding="utf-8"))
    assert meta["schema"] == "phs-dataset-1.0"
    assert meta["dims"] == ["y", "x", "band"]
    assert meta["wavelength_nm"] == spec.wavelength_nm
    assert meta["fwhm_nm"] == 5.0
    assert meta["health"]["ok"] is True


def test_health_accumulator_matches_full_scan() -> None:
    cube = np.full((6, 5, 4), 0.4, dtype=np.float32)
    cube[:, :, 2] = 0
    cube[3, :, :] = 0
    cube[:, 1, :] = 0
    cube[0, 0, 0] = np.nan
    cube[0, 1, 1] = np.inf
    cube[5, 4, 3] = 1.5

    acc = HealthAccumulator(cube.shape)
    for y0 in range(0, cube.shape[0], 2):
        acc.update(y0, cube[y0 : y0 + 2])
    chunked = acc.finish("synthetic")
    full = health_report(cube, "synthetic")

    assert chunked.shape == full.shape
    assert chunked.pixel_count == full.pixel_count == cube.size
    assert chunked.nan_count == full.nan_count == 1
    assert chunked.inf_count == full.inf_count == 1
    assert chunked.out_of_range_count == full.out_of_range_count == 1
    assert chunked.all_zero_bands == full.all_zero_bands == [2]
    assert chunked.all_zero_rows == full.all_zero_rows == [3]
    assert chunked.all_zero_cols == full.all_zero_cols
    assert not chunked.ok and not full.ok
    # Float32 accumulation order differs between the two passes.
    assert chunked.min == pytest.approx(full.min)
    assert chunked.max == pytest.approx(full.max)
    assert chunked.mean == pytest.approx(full.mean)


def test_open_stream_envi_blocks_match_full_read(tmp_path: Path) -> None:
    pytest.importorskip("spectral")
    cube = _cube((5, 4, 3))
    hdr = _write_envi(tmp_path / "scene.img", cube)

    stream = open_stream(hdr)
    assert stream.shape == cube.shape
    blocks = [block for _, block in stream.iter_blocks(target_bytes=1)]
    assert len(blocks) == cube.shape[0]  # one row per block
    np.testing.assert_allclose(np.concatenate(blocks, axis=0), cube, rtol=1e-6)


def test_open_stream_geotiff_blocks_match_full_read(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    cube = _cube((6, 7, 3))
    path = tmp_path / "scene.tif"
    tifffile.imwrite(path, cube, photometric="minisblack")

    stream = open_stream(path)
    assert stream.shape == cube.shape
    blocks = [block for _, block in stream.iter_blocks(target_bytes=1)]
    np.testing.assert_array_equal(np.concatenate(blocks, axis=0), cube)


def test_import_dataset_step_creates_artifacts_and_hits_cache(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    cube = _cube()
    prj = Project.create(tmp_path / "project", "io-test")
    try:
        src = prj.root / "datasets" / "scene.mat"
        savemat(src, {"scene": cube})
        raw = ArtifactRef(
            id="",
            kind=ArtifactKind.RAW,
            name=src.name,
            uri="datasets/scene.mat",
        )
        prj.add_artifact(raw)
        params = {"wavelength_nm": [450.0, 550.0, 650.0]}

        first = prj.run_step("io.import_dataset", [raw], params=params)

        assert first.ok, first.run.error
        assert not first.skipped
        assert len(first.outputs) == 1
        output = first.outputs[0]
        assert output.kind == ArtifactKind.CUBE
        assert output.parents == [raw.id]
        assert output.metadata["shape"] == list(cube.shape)
        assert output.metadata["format"] == "mat"
        assert output.metadata["health_ok"] is True
        assert output.resolve(prj.root).is_dir()

        spec = json.loads((prj.root / "runs" / "scene.spec.json").read_text(encoding="utf-8"))
        health = json.loads((prj.root / "runs" / "scene.health.json").read_text(encoding="utf-8"))
        assert spec["wavelength_nm"] == params["wavelength_nm"]
        assert health["ok"] is True

        second = prj.run_step("io.import_dataset", [raw], params=params)
        assert second.ok
        assert second.skipped
        assert second.outputs == []
        cached = prj.find_cached_outputs("io.import_dataset", [raw], params=params)
        assert [art.id for art in cached] == [output.id]
        cubes = [art for art in prj.list_artifacts() if art.kind == ArtifactKind.CUBE]
        assert len(cubes) == 1
    finally:
        prj.close()
