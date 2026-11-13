"""Stage-3 UI tests: preprocessing panel + main-window wiring (headless Qt)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plant_hsi.ui.main_window import MainWindow  # noqa: E402
from plant_hsi.ui.preprocess_panel import PreprocessPanel  # noqa: E402

ROWS, BANDS = 6, 80
WL = [400.0 + 5.0 * i for i in range(BANDS)]


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _spectra() -> np.ndarray:
    rng = np.random.default_rng(0)
    base = np.linspace(0.2, 0.8, BANDS, dtype=np.float32)
    return np.asarray([base + rng.normal(0, 0.01, BANDS) for _ in range(ROWS)], dtype=np.float32)


def test_panel_disabled_until_bound(qapp: QApplication) -> None:
    panel = PreprocessPanel()
    assert not panel.btn_apply.isEnabled()
    assert panel.plot.has_data is False

    panel.bind(_spectra(), WL)
    assert panel.btn_apply.isEnabled()
    assert "6 条光谱" in panel.header.text()
    assert panel.plot.has_data is True

    panel.unbind()
    assert not panel.btn_apply.isEnabled()
    assert panel.plot.has_data is False


def test_chain_add_apply_and_stats(qapp: QApplication) -> None:
    panel = PreprocessPanel()
    panel.bind(_spectra(), WL)

    panel.add_operator("snv")
    panel.add_operator("sg", window=9, polyorder=2)
    panel.add_operator("fractional_derivative", order=0.8, memory=12)
    assert len(panel.chain) == 3
    assert panel.op_list.count() == 3

    out = panel.apply_preview()
    assert out is not None
    assert out.shape == (ROWS, BANDS)
    assert np.isfinite(out).all()
    assert panel.processed is not None
    assert "ms" in panel.stats.text()


def test_chain_editing_ui(qapp: QApplication) -> None:
    panel = PreprocessPanel()
    panel.bind(_spectra(), WL)
    panel.add_operator("snv")
    panel.add_operator("vector_normalize")

    panel.op_list.setCurrentRow(1)
    panel.btn_up.click()
    assert [op.kind for op in panel.chain] == ["vector_normalize", "snv"]

    panel.btn_remove.click()
    assert len(panel.chain) == 1

    panel.btn_clear.click()
    assert len(panel.chain) == 0
    assert panel.apply_preview() is None


def test_empty_chain_keeps_original_curve(qapp: QApplication) -> None:
    panel = PreprocessPanel()
    spectra = _spectra()
    panel.bind(spectra, WL)
    assert panel.apply_preview() is None
    assert panel.processed is None
    assert "链为空" in panel.stats.text()


def test_main_window_binds_panel_from_segment_signal(qapp: QApplication) -> None:
    win = MainWindow()
    assert win.preprocess_dock.widget() is win.preprocess_panel
    assert win.preprocess_panel.btn_apply.isEnabled() is False

    spectra = _spectra()
    win.seg_panel.spectra_changed.emit(spectra, WL)
    assert win.preprocess_panel.btn_apply.isEnabled() is True
    assert "6 条光谱" in win.preprocess_panel.header.text()

    win.seg_panel.spectra_changed.emit(None, None)
    assert win.preprocess_panel.btn_apply.isEnabled() is False
    win.close()
