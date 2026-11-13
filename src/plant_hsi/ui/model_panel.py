"""Modeling dock: sample labels, model choice, training, report and export.

The panel consumes the ROI mean spectra emitted by :class:`SegmentPanel`
(the same matrix the preprocessing panel receives), optionally replays the
preprocessing chain, and drives the ``ml.train`` / ``report.build`` steps.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.project import Project
from plant_hsi.modeling.specs import CLASSIFICATION, MODELS, models_for
from plant_hsi.preprocess import PreprocessChain


class ModelingPanel(QWidget):
    """Traditional-modeling workspace for one ROI spectra matrix."""

    message = Signal(str)
    model_trained = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._spectra: np.ndarray | None = None
        self._wavelength: list[float] | None = None
        self._model_art: ArtifactRef | None = None
        self._report_art: ArtifactRef | None = None
        self._report_path: Path | None = None
        self._param_widgets: dict[str, QWidget] = {}
        self._chain_provider: Callable[[], PreprocessChain] | None = None

        root = QVBoxLayout(self)
        self.header = QLabel("未绑定光谱（先在分割面板提取 ROI 均值谱）")
        self.header.setWordWrap(True)
        root.addWidget(self.header)

        root.addWidget(self._build_label_box())
        root.addWidget(self._build_model_box())
        root.addWidget(self._build_split_box())

        action_row = QHBoxLayout()
        self.btn_train = QPushButton("训练模型")
        self.btn_train.clicked.connect(self._on_train)
        self.btn_report = QPushButton("生成报告")
        self.btn_report.clicked.connect(self._on_report)
        self.btn_open_report = QPushButton("打开报告")
        self.btn_open_report.clicked.connect(self._on_open_report)
        self.btn_export = QPushButton("导出预测 CSV…")
        self.btn_export.clicked.connect(self._on_export_predictions)
        for button in (self.btn_train, self.btn_report, self.btn_open_report, self.btn_export):
            action_row.addWidget(button)
        root.addLayout(action_row)

        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setMinimumHeight(160)
        self.result.setPlaceholderText("训练结果、指标与产物路径会显示在这里。")
        root.addWidget(self.result, stretch=1)

        self._on_task_changed()
        self._set_enabled(False)

    # ------------------------------------------------------------- building
    def _build_label_box(self) -> QGroupBox:
        box = QGroupBox("样本标签（可直接编辑单元格）")
        layout = QVBoxLayout(box)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["样本", "标签"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setMinimumHeight(130)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        self.btn_sequential = QPushButton("按序号填 1..N")
        self.btn_sequential.clicked.connect(self._fill_sequential)
        row.addWidget(self.btn_sequential)
        self.bulk_edit = QLineEdit()
        self.bulk_edit.setPlaceholderText("统一标签，如 healthy")
        row.addWidget(self.bulk_edit, stretch=1)
        self.btn_bulk = QPushButton("统一填写")
        self.btn_bulk.clicked.connect(self._fill_bulk)
        row.addWidget(self.btn_bulk)
        layout.addLayout(row)
        return box

    def _build_model_box(self) -> QGroupBox:
        box = QGroupBox("模型与超参数")
        form = QFormLayout(box)
        self.task_combo = QComboBox()
        self.task_combo.addItem("分类", CLASSIFICATION)
        self.task_combo.addItem("回归", "regression")
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        form.addRow("任务类型", self.task_combo)

        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("模型", self.model_combo)

        self.param_host = QWidget()
        self.param_form = QFormLayout(self.param_host)
        self.param_form.setContentsMargins(0, 0, 0, 0)
        form.addRow("超参数", self.param_host)
        return box

    def _build_split_box(self) -> QGroupBox:
        box = QGroupBox("划分与验证")
        form = QFormLayout(box)
        self.test_spin = QSpinBox()
        self.test_spin.setRange(10, 50)
        self.test_spin.setValue(20)
        self.test_spin.setSuffix(" %")
        form.addRow("测试集比例", self.test_spin)

        self.cv_spin = QSpinBox()
        self.cv_spin.setRange(2, 10)
        self.cv_spin.setValue(5)
        form.addRow("交叉验证折数", self.cv_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setValue(42)
        form.addRow("随机种子", self.seed_spin)

        self.chk_preprocess = QCheckBox("应用预处理链（若已配置）")
        self.chk_preprocess.setChecked(True)
        form.addRow("", self.chk_preprocess)
        return box

    # ------------------------------------------------------------ public API
    def bind(
        self,
        project: Project,
        spectra: np.ndarray,
        wavelength_nm: list[float] | None = None,
    ) -> None:
        arr = np.asarray(spectra, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"expected 2-D spectra, got shape {arr.shape}")
        self._project = project
        self._spectra = arr
        self._wavelength = list(wavelength_nm) if wavelength_nm else None
        self._model_art = None
        self._report_art = None
        self._report_path = None

        self.table.setRowCount(arr.shape[0])
        for i in range(arr.shape[0]):
            sample = QTableWidgetItem(f"ROI {i + 1}")
            sample.setFlags(sample.flags() & ~Qt.ItemFlag.ItemIsEditable)
            label = QTableWidgetItem(str(i + 1))
            self.table.setItem(i, 0, sample)
            self.table.setItem(i, 1, label)

        wl = ""
        if self._wavelength and len(self._wavelength) == arr.shape[1]:
            wl = f"  {self._wavelength[0]:.1f}–{self._wavelength[-1]:.1f} nm"
        self.header.setText(f"<b>{arr.shape[0]} 条光谱 × {arr.shape[1]} 波段</b>{wl}")
        self.result.clear()
        self.btn_report.setEnabled(False)
        self.btn_open_report.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._set_enabled(True)
        self.message.emit(f"建模面板已绑定 {arr.shape[0]} 条光谱")

    def unbind(self) -> None:
        self._project = None
        self._spectra = None
        self._wavelength = None
        self._model_art = None
        self._report_art = None
        self._report_path = None
        self.table.setRowCount(0)
        self.header.setText("未绑定光谱（先在分割面板提取 ROI 均值谱）")
        self.result.clear()
        self._set_enabled(False)

    def set_preprocess_chain_provider(self, provider: Callable[[], PreprocessChain] | None) -> None:
        self._chain_provider = provider

    @property
    def model_artifact(self) -> ArtifactRef | None:
        return self._model_art

    @property
    def report_artifact(self) -> ArtifactRef | None:
        return self._report_art

    # -------------------------------------------------------------- labels
    def labels(self) -> list[str]:
        values: list[str] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            text = item.text().strip() if item is not None else ""
            values.append(text or str(row + 1))
        return values

    def _fill_sequential(self) -> None:
        for row in range(self.table.rowCount()):
            self.table.setItem(row, 1, QTableWidgetItem(str(row + 1)))

    def _fill_bulk(self) -> None:
        text = self.bulk_edit.text().strip()
        if not text:
            return
        for row in range(self.table.rowCount()):
            self.table.setItem(row, 1, QTableWidgetItem(text))

    # -------------------------------------------------------------- models
    def _on_task_changed(self) -> None:
        task = str(self.task_combo.currentData() or CLASSIFICATION)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for spec in models_for(task):
            self.model_combo.addItem(spec.label, spec.key)
        self.model_combo.blockSignals(False)
        self._on_model_changed()

    def _on_model_changed(self) -> None:
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self._param_widgets = {}
        key = self.model_combo.currentData()
        if not key or key not in MODELS:
            return
        spec = MODELS[str(key)]
        for name, schema in spec.params.items():
            widget = self._param_widget(schema)
            if widget is None:
                continue
            self._param_widgets[name] = widget
            self.param_form.addRow(str(schema.get("label", name)), widget)
        if not self._param_widgets:
            self.param_form.addRow(QLabel("该模型无超参数"))

    @staticmethod
    def _param_widget(schema: dict[str, Any]) -> QWidget | None:
        kind = schema.get("type")
        if kind == "int":
            spin = QSpinBox()
            spin.setRange(int(schema.get("min", 0)), int(schema.get("max", 9999)))
            spin.setSingleStep(int(schema.get("step", 1)))
            spin.setValue(int(schema.get("default", 0)))
            return spin
        if kind == "float":
            spin_f = QDoubleSpinBox()
            spin_f.setDecimals(4)
            spin_f.setRange(float(schema.get("min", 0.0)), float(schema.get("max", 1e9)))
            spin_f.setSingleStep(float(schema.get("step", 0.1)))
            spin_f.setValue(float(schema.get("default", 0.0)))
            return spin_f
        if kind == "choice":
            combo = QComboBox()
            combo.addItems([str(v) for v in schema.get("choices", [])])
            default = str(schema.get("default", ""))
            if default:
                combo.setCurrentText(default)
            return combo
        return None

    def _collect_model_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            if isinstance(widget, QSpinBox):
                params[name] = int(widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                params[name] = float(widget.value())
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
        return params

    def _chain_snapshot(self) -> dict[str, Any] | None:
        if not self.chk_preprocess.isChecked() or self._chain_provider is None:
            return None
        chain = self._chain_provider()
        if chain is None or len(chain) == 0:
            return None
        return chain.to_dict()

    # ------------------------------------------------------------ training
    def _on_train(self) -> None:
        if self._spectra is None or self._project is None:
            return
        task = str(self.task_combo.currentData() or CLASSIFICATION)
        algorithm = str(self.model_combo.currentData() or "rf")
        labels = self.labels()
        if task == CLASSIFICATION and len(set(labels)) < 2:
            QMessageBox.warning(
                self, "标签不足", "分类任务至少需要 2 个不同标签；请在表格中修改标签。"
            )
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            spectra_art = self._ensure_spectra_artifact(labels)
            params = {
                "task": task,
                "algorithm": algorithm,
                "label_column": "label",
                "model_params": self._collect_model_params(),
                "test_size": self.test_spin.value() / 100.0,
                "val_size": 0.0,
                "cv_folds": self.cv_spin.value(),
                "preprocess_chain": self._chain_snapshot(),
            }
            seed = int(self.seed_spin.value())
            result = self._project.run_step("ml.train", [spectra_art], params=params, seed=seed)
            if not result.ok:
                raise RuntimeError(result.run.error or "训练失败")
            outputs = result.outputs
            if result.skipped:
                outputs = self._project.find_cached_outputs(
                    "ml.train", [spectra_art], params=params, seed=seed
                )
            model_art = next((o for o in outputs if o.kind == ArtifactKind.MODEL), None)
            if model_art is None:
                raise RuntimeError("训练未产生 MODEL 产物")
            self._model_art = model_art
            self._report_art = None
            self._report_path = None
            self._show_metrics(model_art)
            self.btn_report.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.btn_open_report.setEnabled(False)
            self.model_trained.emit(model_art)
            self.message.emit(f"训练完成：{model_art.name}")
        except Exception as exc:  # noqa: BLE001 - surface any algorithm error to the user
            QMessageBox.critical(self, "训练失败", str(exc))
            self.message.emit(f"训练失败：{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _ensure_spectra_artifact(self, labels: list[str]) -> ArtifactRef:
        assert self._project is not None and self._spectra is not None
        prj = self._project
        x = self._spectra
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())
        digest.update("\n".join(labels).encode("utf-8"))
        name = f"modeling_spectra_{digest.hexdigest()[:16]}"
        uri = f"artifacts/modeling/{name}.csv"
        for art in prj.list_artifacts():
            if art.kind == ArtifactKind.SPECTRA and art.uri == uri:
                return art

        frame: dict[str, Any] = {
            "sample_id": [f"roi_{i + 1}" for i in range(x.shape[0])],
            "label": labels,
        }
        for column in range(x.shape[1]):
            frame[f"b{column}"] = x[:, column]
        path = prj.root / uri
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(frame).to_csv(path, index=False, encoding="utf-8")
        art = ArtifactRef(
            id="",
            kind=ArtifactKind.SPECTRA,
            name=name,
            uri=uri,
            metadata={
                "n_samples": int(x.shape[0]),
                "n_features": int(x.shape[1]),
                "source": "modeling_panel",
            },
        )
        return prj.add_artifact(art)

    def _show_metrics(self, model_art: ArtifactRef) -> None:
        meta = model_art.metadata or {}
        metrics = meta.get("metrics") or {}
        cv = meta.get("cv") or {}
        task = str(meta.get("task") or "")
        lines = [
            f"模型：{model_art.name}",
            f"任务 / 算法：{task} / {meta.get('algorithm', '')}",
            f"训练 / 测试样本：{meta.get('n_train', '')} / {meta.get('n_test', '')}",
            "",
        ]
        if task == CLASSIFICATION:
            lines += [
                f"Accuracy       : {_fmt(metrics.get('accuracy'))}",
                f"F1 (macro)     : {_fmt(metrics.get('f1_macro'))}",
                f"Precision macro: {_fmt(metrics.get('precision_macro'))}",
                f"Recall macro   : {_fmt(metrics.get('recall_macro'))}",
            ]
        else:
            lines += [
                f"R²   : {_fmt(metrics.get('r2'))}",
                f"RMSE : {_fmt(metrics.get('rmse'))}",
                f"MAE  : {_fmt(metrics.get('mae'))}",
                f"MSE  : {_fmt(metrics.get('mse'))}",
            ]
        cv_metrics = cv.get("metrics") or {}
        if cv_metrics:
            primary = "f1_macro" if task == CLASSIFICATION else "r2"
            item = cv_metrics.get(primary)
            if item:
                lines.append(
                    f"CV {primary} ({cv.get('folds')} 折): "
                    f"{_fmt(item.get('mean'))} ± {_fmt(item.get('std'))}"
                )
        lines += ["", "产物："]
        for key in ("metrics_uri", "predictions_uri", "confusion_uri", "card_uri"):
            if meta.get(key):
                lines.append(f"  {key}: {meta[key]}")
        self.result.setPlainText("\n".join(lines))

    # -------------------------------------------------------------- report
    def _on_report(self) -> None:
        if self._project is None or self._model_art is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            params = {"title": f"{self._project.meta.name} · 传统建模报告"}
            result = self._project.run_step("report.build", [self._model_art], params=params)
            if not result.ok:
                raise RuntimeError(result.run.error or "报告生成失败")
            outputs = result.outputs
            if result.skipped:
                outputs = self._project.find_cached_outputs(
                    "report.build", [self._model_art], params=params
                )
            report = next((o for o in outputs if o.kind == ArtifactKind.REPORT), None)
            if report is None:
                raise RuntimeError("报告 Step 未产生 REPORT 产物")
            self._report_art = report
            self._report_path = report.resolve(self._project.root)
            self.btn_open_report.setEnabled(True)
            self.result.append(f"\n报告已生成：{self._report_path}")
            self.message.emit(f"报告已生成：{self._report_path.name}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "报告失败", str(exc))
            self.message.emit(f"报告失败：{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_open_report(self) -> None:
        if self._report_path is None or not self._report_path.exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._report_path)))

    def _on_export_predictions(self) -> None:
        if self._project is None or self._model_art is None:
            return
        uri = (self._model_art.metadata or {}).get("predictions_uri")
        if not uri:
            return
        source = self._project.root / str(uri)
        if not source.exists():
            QMessageBox.warning(self, "导出", f"预测文件不存在：{source}")
            return
        target, _ = QFileDialog.getSaveFileName(
            self,
            "导出预测 CSV",
            str(Path.home() / "predictions.csv"),
            "CSV (*.csv)",
        )
        if not target:
            return
        pd.read_csv(source).to_csv(target, index=False, encoding="utf-8-sig")
        self.message.emit(f"预测已导出：{target}")

    # ------------------------------------------------------------- helpers
    def _set_enabled(self, enabled: bool) -> None:
        for widget in (
            self.table,
            self.btn_sequential,
            self.bulk_edit,
            self.btn_bulk,
            self.task_combo,
            self.model_combo,
            self.param_host,
            self.test_spin,
            self.cv_spin,
            self.seed_spin,
            self.chk_preprocess,
            self.btn_train,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self.btn_report.setEnabled(False)
            self.btn_open_report.setEnabled(False)
            self.btn_export.setEnabled(False)


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"
