"""Stage-5 UI tests: DC²Net panel + main-window wiring (headless Qt)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("torch")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef  # noqa: E402
from plant_hsi.core.project import Project  # noqa: E402
from plant_hsi.ui.dl_panel import DeepLearningPanel  # noqa: E402
from plant_hsi.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def project(tmp_path):
    prj = Project.create(tmp_path / "prj", "dl-project")
    yield prj
    prj.close()


def _cube_artifact(prj: Project) -> ArtifactRef:
    return prj.add_artifact(
        ArtifactRef(
            id="",
            kind=ArtifactKind.CUBE,
            name="synthetic_cube",
            uri="artifacts/synthetic.zarr",
            metadata={"shape": [10, 10, 5]},
        )
    )


def test_panel_disabled_until_bound(qapp: QApplication, project: Project) -> None:
    panel = DeepLearningPanel()
    assert not panel.btn_train.isEnabled()
    assert not panel.btn_infer.isEnabled()

    cue = _cube_artifact(project)
    panel.bind(project, cue)
    assert panel.btn_patch.isEnabled()
    assert not panel.btn_train.isEnabled()  # needs a patch set first
    assert "synthetic_cube" in panel.header.text()

    panel.unbind()
    assert not panel.btn_patch.isEnabled()


def test_class_map_parser(qapp: QApplication) -> None:
    panel = DeepLearningPanel()
    panel.class_map_edit.setText("1-3:healthy, 4:diseased")
    assert panel._class_map() == {
        "1": "healthy",
        "2": "healthy",
        "3": "healthy",
        "4": "diseased",
    }
    panel.class_map_edit.setText("")
    assert panel._class_map() == {}


def test_main_window_has_dl_dock(qapp: QApplication, project: Project) -> None:
    window = MainWindow()
    assert window.dl_panel is not None
    assert window.findChild(DeepLearningPanel) is not None
    window.close()
