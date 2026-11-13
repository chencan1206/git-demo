"""Stage-2 UI tests: overlay rendering + segmentation panel (headless Qt)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("zarr")

import zarr  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef  # noqa: E402
from plant_hsi.core.project import Project  # noqa: E402
from plant_hsi.ui.hsi_view import HsiImageView  # noqa: E402
from plant_hsi.ui.segment_panel import SegmentPanel  # noqa: E402

ROWS, COLS, BANDS = 40, 50, 60
WL = [400.0 + 5.0 * i for i in range(BANDS)]


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------- fixtures
def _synthetic_cube() -> np.ndarray:
    """Two bright rectangles on a dim background, spectrally distinct."""
    rng = np.random.default_rng(0)
    cube = np.full((ROWS, COLS, BANDS), 0.15, dtype=np.float32)
    cube += rng.normal(0.0, 0.01, cube.shape).astype(np.float32)
    cube[5:15, 5:20] = 0.75
    cube[25:35, 30:45] = 0.55
    return cube


def _make_project(tmp_path: Path) -> tuple[Project, ArtifactRef, np.ndarray]:
    prj = Project.create(tmp_path / "prj", "ui-test")
    cube = _synthetic_cube()
    path = prj.root / "artifacts" / "synth.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    z = zarr.open_array(str(path), mode="w", zarr_format=2, shape=cube.shape, dtype="float32")
    z[:] = cube
    (path / ".phs_meta.json").write_text(json.dumps({"wavelength_nm": WL}), encoding="utf-8")
    art = ArtifactRef(id="", kind=ArtifactKind.CUBE, name="synth", uri="artifacts/synth.zarr")
    prj.add_artifact(art)
    return prj, art, cube


# ------------------------------------------------------------------ overlay API
def test_overlay_toggle_and_shape_check(qapp: QApplication) -> None:
    view = HsiImageView(_synthetic_cube(), wavelength_nm=WL)
    assert view.overlay_visible is False

    mask = np.zeros((ROWS, COLS), dtype=np.uint8)
    mask[5:15, 5:20] = 1
    view.set_overlay(mask, kind="mask", alpha=0.5)
    assert view.overlay_visible is True
    assert view.image_label.pixmap() is not None

    view.clear_overlay()
    assert view.overlay_visible is False

    with pytest.raises(ValueError):
        view.set_overlay(np.zeros((3, 3), dtype=np.uint8))


def test_overlay_labels_and_alpha_zero(qapp: QApplication) -> None:
    view = HsiImageView(_synthetic_cube(), wavelength_nm=WL)
    labels = np.zeros((ROWS, COLS), dtype=np.int32)
    labels[5:15, 5:20] = 1
    labels[25:35, 30:45] = 2
    view.set_overlay(labels, kind="labels", alpha=0.4)
    assert view.overlay_visible

    view.set_overlay_alpha(0.0)
    assert view.overlay_visible is False
    view.set_overlay_alpha(1.5)
    assert view.overlay_visible is True


def test_cursor_and_box_spectrum(qapp: QApplication) -> None:
    view = HsiImageView(_synthetic_cube(), wavelength_nm=WL)
    view.set_cursor(8, 10)
    assert view.cursor_position() == (8, 10)
    view.set_cursor(-5, 999)
    assert view.cursor_position() == (0, COLS - 1)
    spec = view.spectrum_at_box(8, 10, 5, 5)
    assert spec.shape == (BANDS,)
    assert float(spec.mean()) > 0.5  # inside a bright blob


# ----------------------------------------------------------------- panel basics
def test_panel_disabled_until_bound(qapp: QApplication, tmp_path: Path) -> None:
    panel = SegmentPanel()
    assert not panel.btn_preview.isEnabled()
    assert panel.bound_artifact_id is None

    prj, art, cube = _make_project(tmp_path)
    view = HsiImageView(cube, wavelength_nm=WL)
    panel.bind(prj, art, view, WL)
    assert panel.btn_preview.isEnabled()
    assert panel.bound_artifact_id == art.id

    panel.unbind()
    assert panel.bound_artifact_id is None
    assert not panel.btn_preview.isEnabled()


def test_panel_preview_and_threshold_roundtrip(qapp: QApplication, tmp_path: Path) -> None:
    prj, art, cube = _make_project(tmp_path)
    view = HsiImageView(cube, wavelength_nm=WL)
    panel = SegmentPanel()
    panel.bind(prj, art, view, WL)

    panel._on_preview()
    assert panel._sim is not None and panel._sim.shape == (ROWS, COLS)
    assert panel._mask is not None and panel._mask.sum() > 0
    assert view.overlay_visible

    otsu_thr = panel._threshold_used
    panel.thr_method.setCurrentIndex(2)  # fixed
    panel.thr_value.setValue(0.9)
    assert panel._threshold_used == pytest.approx(0.9)

    panel.thr_method.setCurrentIndex(0)  # back to otsu
    assert panel._threshold_used == pytest.approx(otsu_thr)


def test_similarity_method_switch_recomputes(qapp: QApplication, tmp_path: Path) -> None:
    prj, art, cube = _make_project(tmp_path)
    view = HsiImageView(cube, wavelength_nm=WL)
    panel = SegmentPanel()
    panel.bind(prj, art, view, WL)

    panel._on_preview()
    ra_sim = panel._sim.copy()
    panel.sim_method.setCurrentIndex(1)  # DE
    panel._on_preview()
    assert panel._sim is not None
    assert not np.allclose(ra_sim, panel._sim)


# --------------------------------------------------------------- mask + ROI flow
def test_save_mask_creates_artifact_and_caches(qapp: QApplication, tmp_path: Path) -> None:
    prj, art, cube = _make_project(tmp_path)
    view = HsiImageView(cube, wavelength_nm=WL)
    panel = SegmentPanel()
    panel.bind(prj, art, view, WL)

    seen: list[int] = []
    panel.artifacts_created.connect(lambda: seen.append(1))

    mask_id = panel._save_mask(quiet=False)
    assert mask_id is not None
    stored = prj.get_artifact(mask_id)
    assert stored is not None and stored.kind == ArtifactKind.MASK
    assert seen == [1]

    # identical params ⇒ cache hit, same artifact, no duplicate
    assert panel._save_mask(quiet=True) == mask_id
    masks = [a for a in prj.list_artifacts() if a.kind == ArtifactKind.MASK]
    assert len(masks) == 1


def test_find_cached_outputs_recovers_artifacts(qapp: QApplication, tmp_path: Path) -> None:
    prj, art, cube = _make_project(tmp_path)
    view = HsiImageView(cube, wavelength_nm=WL)
    panel = SegmentPanel()
    panel.bind(prj, art, view, WL)
    panel._on_preview()
    params = panel._step_params()

    result = prj.run_step("segment.threshold", [art], params=params)
    assert result.ok and not result.skipped
    produced = result.outputs[0].id

    hit = prj.run_step("segment.threshold", [art], params=params)
    assert hit.skipped
    assert prj.find_cached_outputs("segment.threshold", [art], params)[0].id == produced
    # unknown params must not resurrect anything
    assert prj.find_cached_outputs("segment.threshold", [art], {"method": "de"}) == []


def test_extract_rois_fills_table(qapp: QApplication, tmp_path: Path) -> None:
    prj, art, cube = _make_project(tmp_path)
    view = HsiImageView(cube, wavelength_nm=WL)
    panel = SegmentPanel()
    panel.bind(prj, art, view, WL)
    panel._on_preview()
    panel.min_area.setValue(20)

    panel._on_extract()
    assert panel.table.rowCount() == 2  # two synthetic blobs
    assert panel._labels is not None and int(panel._labels.max()) == 2
    assert [a for a in prj.list_artifacts() if a.kind == ArtifactKind.SPECTRA]

    first_area = panel.table.item(0, 1).text().replace(",", "")
    assert first_area.isdigit()
    # rows are sorted by area descending
    areas = [int(panel.table.item(r, 1).text().replace(",", "")) for r in range(2)]
    assert areas == sorted(areas, reverse=True)

    before = view.cursor_position()
    panel.table.selectRow(0)
    assert view.cursor_position() != before


def test_export_writes_csv(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prj, art, cube = _make_project(tmp_path)
    view = HsiImageView(cube, wavelength_nm=WL)
    panel = SegmentPanel()
    panel.bind(prj, art, view, WL)
    panel._on_preview()

    out = tmp_path / "label_map.csv"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "csv"))
    )
    panel._on_export()

    assert out.exists()
    import pandas as pd

    df = pd.read_csv(out)
    for col in ("label", "area_px", "row_centroid", "col_centroid"):
        assert col in df.columns
    assert len([c for c in df.columns if c.startswith("mean_")]) == BANDS


# ------------------------------------------------------------- main window glue
def test_main_window_binds_panel_to_open_tab(qapp: QApplication, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt

    from plant_hsi.ui.main_window import MainWindow

    prj, art, _cube = _make_project(tmp_path)
    win = MainWindow()
    win._project = prj
    win._refresh_resource_tree()
    assert win.seg_dock.widget() is win.seg_panel

    def walk(item):
        found = [item] if item.data(0, Qt.ItemDataRole.UserRole) else []
        for i in range(item.childCount()):
            found += walk(item.child(i))
        return found

    rows = []
    for i in range(win.tree.topLevelItemCount()):
        rows += walk(win.tree.topLevelItem(i))
    cube_rows = [r for r in rows if r.text(1) == "cube"]
    assert len(cube_rows) == 1

    win._open_dataset_for_item(cube_rows[0])
    assert win.tabs.count() == 1
    assert win.seg_panel.bound_artifact_id == art.id

    win._close_tab(0)
    assert win.seg_panel.bound_artifact_id is None
    win.close()
