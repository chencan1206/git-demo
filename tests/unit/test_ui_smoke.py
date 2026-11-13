"""Stage-6 UI smoke: the real main window + every dock, headless."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("zarr")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef  # noqa: E402
from plant_hsi.core.project import Project  # noqa: E402
from plant_hsi.io.spec import DatasetSpec, write_zarr  # noqa: E402
from plant_hsi.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _demo_project(tmp_path) -> tuple[Project, ArtifactRef]:
    prj = Project.create(tmp_path / "prj", "ui-smoke")
    rng = np.random.default_rng(0)
    cube = (rng.random((10, 12, 8), dtype=np.float32) * 0.6 + 0.2).astype(np.float32)
    spec = DatasetSpec(
        source_path="synthetic",
        format="mat",
        shape=cube.shape,
        wavelength_nm=list(np.linspace(400, 950, cube.shape[2])),
    )
    write_zarr(cube, prj.root / "artifacts" / "demo.zarr", spec)
    art = prj.add_artifact(
        ArtifactRef(
            id="",
            kind=ArtifactKind.CUBE,
            name="demo",
            uri="artifacts/demo.zarr",
            metadata={"shape": list(cube.shape)},
        )
    )
    return prj, art


def _find_cube_item(window: MainWindow) -> object:
    project_item = window.tree.topLevelItem(0)
    datasets = project_item.child(0)
    assert datasets.childCount() == 1
    return datasets.child(0)


def test_main_window_smoke_with_project(qapp: QApplication, tmp_path) -> None:
    prj, art = _demo_project(tmp_path)
    window = MainWindow()
    try:
        window._project = prj
        window._refresh_resource_tree()
        assert window.tree.topLevelItemCount() == 1
        assert window.tree.topLevelItem(0).text(0) == "ui-smoke"

        window._open_dataset_for_item(_find_cube_item(window))
        qapp.processEvents()
        assert window.tabs.count() == 1
        assert window.seg_panel._project is prj

        spectra = np.random.default_rng(1).random((6, 8), dtype=np.float32)
        wl = list(np.linspace(400, 950, 8))
        window.preprocess_panel.bind(spectra, wl)
        window.model_panel.bind(prj, spectra, wl)
        assert window.preprocess_panel._spectra is not None
        assert window.model_panel._spectra is not None

        if window.dl_panel is not None:
            window.dl_panel.bind(prj, art)
            assert "demo" in window.dl_panel.header.text()

        window.resize(1400, 900)
        window.show()
        qapp.processEvents()
        assert window.isVisible()
        window.close()
    finally:
        prj.close()
