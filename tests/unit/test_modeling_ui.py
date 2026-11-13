"""Stage-4 UI tests: modeling panel + main-window wiring (headless Qt)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QSpinBox  # noqa: E402

from plant_hsi.core.project import Project  # noqa: E402
from plant_hsi.ui.main_window import MainWindow  # noqa: E402
from plant_hsi.ui.model_panel import ModelingPanel  # noqa: E402
from plant_hsi.ui.preprocess_panel import PreprocessPanel  # noqa: E402

ROWS, BANDS = 30, 40
WL = [400.0 + 5.0 * i for i in range(BANDS)]


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _spectra_and_labels() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 1.0, (3, BANDS))
    rows, labels = [], []
    for c in range(3):
        rows.append(centers[c] + rng.normal(0, 0.2, (10, BANDS)))
        labels += [f"class_{c}"] * 10
    return np.vstack(rows).astype(np.float32), labels


@pytest.fixture
def project(tmp_path):
    prj = Project.create(tmp_path / "prj", "ui-project")
    yield prj
    prj.close()


def _quiet_dialogs(monkeypatch) -> None:
    monkeypatch.setattr("plant_hsi.ui.model_panel.QMessageBox.critical", lambda *a, **k: None)
    monkeypatch.setattr("plant_hsi.ui.model_panel.QMessageBox.warning", lambda *a, **k: None)


def test_panel_disabled_until_bound(qapp: QApplication, project: Project) -> None:
    panel = ModelingPanel()
    assert not panel.btn_train.isEnabled()
    assert panel.model_combo.count() == 4  # classification default

    x, _ = _spectra_and_labels()
    panel.bind(project, x, WL)
    assert panel.btn_train.isEnabled()
    assert panel.table.rowCount() == ROWS
    assert "30 条光谱" in panel.header.text()

    panel.unbind()
    assert not panel.btn_train.isEnabled()
    assert panel.table.rowCount() == 0


def test_task_switch_lists_the_right_models(qapp: QApplication, project: Project) -> None:
    panel = ModelingPanel()
    x, _ = _spectra_and_labels()
    panel.bind(project, x, WL)

    panel.task_combo.setCurrentIndex(1)  # regression
    keys = [panel.model_combo.itemData(i) for i in range(panel.model_combo.count())]
    assert keys == ["plsr", "rfr", "svr", "lr"]

    panel.task_combo.setCurrentIndex(0)
    keys = [panel.model_combo.itemData(i) for i in range(panel.model_combo.count())]
    assert keys == ["rf", "svm", "lda", "knn"]


def test_label_helpers(qapp: QApplication, project: Project) -> None:
    panel = ModelingPanel()
    x, _ = _spectra_and_labels()
    panel.bind(project, x, WL)

    panel._fill_sequential()
    assert panel.labels()[0] == "1"
    assert panel.labels()[-1] == str(ROWS)

    panel.bulk_edit.setText("healthy")
    panel._fill_bulk()
    assert set(panel.labels()) == {"healthy"}


def test_train_and_report_from_panel(qapp: QApplication, project: Project, monkeypatch) -> None:
    _quiet_dialogs(monkeypatch)
    panel = ModelingPanel()
    x, labels = _spectra_and_labels()
    panel.bind(project, x, WL)
    for row, label in enumerate(labels):
        panel.table.item(row, 1).setText(label)

    estimator_spin = panel._param_widgets["n_estimators"]
    assert isinstance(estimator_spin, QSpinBox)
    estimator_spin.setValue(30)
    panel.cv_spin.setValue(3)
    panel.seed_spin.setValue(42)

    panel._on_train()
    assert panel.model_artifact is not None
    text = panel.result.toPlainText()
    assert "Accuracy" in text
    assert "CV f1_macro" in text

    panel._on_report()
    assert panel.report_artifact is not None
    html = panel.report_artifact.resolve(project.root).read_text(encoding="utf-8")
    assert "混淆矩阵" in html
    assert "run_0001" in html


def test_panel_records_preprocess_chain(qapp: QApplication, project: Project, monkeypatch) -> None:
    _quiet_dialogs(monkeypatch)
    x, labels = _spectra_and_labels()
    pre = PreprocessPanel()
    pre.bind(x, WL)
    pre.add_operator("snv")

    panel = ModelingPanel()
    panel.bind(project, x, WL)
    panel.set_preprocess_chain_provider(lambda: pre.chain)
    for row, label in enumerate(labels):
        panel.table.item(row, 1).setText(label)
    estimator_spin = panel._param_widgets["n_estimators"]
    assert isinstance(estimator_spin, QSpinBox)
    estimator_spin.setValue(20)

    panel._on_train()
    assert panel.model_artifact is not None
    card = panel.model_artifact.metadata["card"]
    assert card["preprocess_chain"]["ops"][0]["kind"] == "snv"


def test_main_window_exposes_modeling_dock(qapp: QApplication) -> None:
    window = MainWindow()
    assert window.model_panel is not None
    assert window.model_dock.windowTitle().startswith("建模")
    titles = [a.text() for a in window.menuBar().actions()]
    assert "&Model" in titles
    window.close()
