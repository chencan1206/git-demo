"""Segmentation dock panel.

Wires the Stage-2 algorithms into a usable GUI flow:

    1. pick a reference spectrum (cursor box or image centre box)
    2. compute a similarity map (RA / DE)  → cached in memory
    3. drag a threshold (Otsu / percentile / fixed) → instant mask + overlay
    4. persist the mask through the ``segment.threshold`` step
    5. extract ROIs through ``segment.connected_components`` → table + CSV

Similarity is the only expensive call, so it is computed once and re-used
while the user experiments with thresholds — threshold changes only cost a
few milliseconds.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from PySide6.QtCore import Qt, Signal
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
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from plant_hsi.core.artifact import ArtifactRef
from plant_hsi.core.project import Project
from plant_hsi.segment.core import (
    ComponentResult,
    binarize,
    connected_components,
    morphological_cleanup,
    similarity_map,
)
from plant_hsi.segment.roi import ROISet, compute_mean_spectra
from plant_hsi.ui.hsi_view import HsiImageView

_TABLE_COLS = ["ROI", "面积 px", "行", "列"]

SimMethod = Literal["ra", "de"]
ThrMethod = Literal["otsu", "percentile", "fixed"]


class SegmentPanel(QWidget):
    """Segmentation controls + ROI table for one bound CUBE view."""

    artifacts_created = Signal()
    message = Signal(str)
    #: ``(mean_spectra, wavelength_nm)`` for the ROIs just extracted, or
    #: ``(None, None)`` when the panel is unbound.
    spectra_changed = Signal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._cube_art: ArtifactRef | None = None
        self._view: HsiImageView | None = None
        self._wl: list[float] | None = None

        self._ref: np.ndarray | None = None
        self._sim: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._roiset: ROISet | None = None
        self._threshold_used: float = 0.0
        self._mask_art_id: str | None = None

        root = QVBoxLayout(self)
        self.header = QLabel("未绑定数据集")
        self.header.setWordWrap(True)
        root.addWidget(self.header)

        root.addWidget(self._build_similarity_box())
        root.addWidget(self._build_threshold_box())
        root.addWidget(self._build_roi_box())
        root.addWidget(self._build_export_box())

        self.stats = QLabel("—")
        self.stats.setWordWrap(True)
        root.addWidget(self.stats)
        root.addStretch(1)

        self._set_enabled(False)

    # ------------------------------------------------------------- building
    def _build_similarity_box(self) -> QGroupBox:
        box = QGroupBox("1 · 相似度")
        form = QFormLayout(box)

        self.sim_method = QComboBox()
        self.sim_method.addItem("RA — 光谱夹角 (SAM)", "ra")
        self.sim_method.addItem("DE — 光谱散度 (SID)", "de")
        form.addRow("算法", self.sim_method)

        self.ref_source = QComboBox()
        self.ref_source.addItem("光标处小框", "cursor")
        self.ref_source.addItem("图像中心小框", "center")
        form.addRow("参考光谱", self.ref_source)

        size_row = QHBoxLayout()
        self.ref_w = QSpinBox()
        self.ref_w.setRange(1, 201)
        self.ref_w.setValue(10)
        self.ref_w.setSuffix(" px 宽")
        self.ref_h = QSpinBox()
        self.ref_h.setRange(1, 201)
        self.ref_h.setValue(10)
        self.ref_h.setSuffix(" px 高")
        size_row.addWidget(self.ref_w)
        size_row.addWidget(self.ref_h)
        form.addRow("小框尺寸", size_row)

        self.cursor_label = QLabel("y=-  x=-")
        form.addRow("光标", self.cursor_label)

        self.btn_preview = QPushButton("计算相似度预览")
        self.btn_preview.clicked.connect(self._on_preview)
        form.addRow(self.btn_preview)
        return box

    def _build_threshold_box(self) -> QGroupBox:
        box = QGroupBox("2 · 阈值与形态学")
        form = QFormLayout(box)

        self.thr_method = QComboBox()
        self.thr_method.addItem("Otsu (自动)", "otsu")
        self.thr_method.addItem("百分位", "percentile")
        self.thr_method.addItem("固定值", "fixed")
        self.thr_method.currentIndexChanged.connect(self._on_thr_method_changed)
        form.addRow("阈值方法", self.thr_method)

        self.thr_value = QDoubleSpinBox()
        self.thr_value.setDecimals(4)
        self.thr_value.setRange(0.0, 1.0)
        self.thr_value.setSingleStep(0.01)
        self.thr_value.setValue(0.95)
        self.thr_value.valueChanged.connect(self._apply_threshold)
        form.addRow("阈值", self.thr_value)

        self.percentile = QDoubleSpinBox()
        self.percentile.setDecimals(1)
        self.percentile.setRange(1.0, 99.0)
        self.percentile.setValue(70.0)
        self.percentile.setSuffix(" %")
        self.percentile.setEnabled(False)
        self.percentile.valueChanged.connect(self._apply_threshold)
        form.addRow("百分位", self.percentile)

        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 15)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setValue(3)
        self.kernel_spin.valueChanged.connect(self._apply_threshold)
        form.addRow("形态学核", self.kernel_spin)

        iter_row = QHBoxLayout()
        self.open_spin = QSpinBox()
        self.open_spin.setRange(0, 5)
        self.open_spin.setValue(1)
        self.open_spin.setPrefix("开 ")
        self.open_spin.valueChanged.connect(self._apply_threshold)
        self.close_spin = QSpinBox()
        self.close_spin.setRange(0, 5)
        self.close_spin.setValue(2)
        self.close_spin.setPrefix("闭 ")
        self.close_spin.valueChanged.connect(self._apply_threshold)
        iter_row.addWidget(self.open_spin)
        iter_row.addWidget(self.close_spin)
        form.addRow("迭代", iter_row)

        self.show_overlay = QCheckBox("显示叠加层")
        self.show_overlay.setChecked(True)
        self.show_overlay.toggled.connect(self._update_overlay)
        form.addRow(self.show_overlay)

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(45)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        form.addRow("透明度", self.alpha_slider)

        self.btn_save_mask = QPushButton("保存 Mask 为数据集")
        self.btn_save_mask.clicked.connect(lambda: self._save_mask(quiet=False))
        form.addRow(self.btn_save_mask)
        return box

    def _build_roi_box(self) -> QGroupBox:
        box = QGroupBox("3 · ROI 提取")
        outer = QVBoxLayout(box)

        form = QFormLayout()
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 100_000)
        self.min_area.setValue(20)
        self.min_area.setSuffix(" px")
        form.addRow("最小面积", self.min_area)

        self.sort_box = QComboBox()
        for label, key in (
            ("面积 大→小", "area_desc"),
            ("面积 小→大", "area_asc"),
            ("按行扫描", "row_col"),
            ("按列扫描", "col_row"),
        ):
            self.sort_box.addItem(label, key)
        form.addRow("排序", self.sort_box)
        outer.addLayout(form)

        self.btn_extract = QPushButton("提取 ROI（连通域）")
        self.btn_extract.clicked.connect(self._on_extract)
        outer.addWidget(self.btn_extract)

        self.table = QTableWidget(0, len(_TABLE_COLS))
        self.table.setHorizontalHeaderLabels(_TABLE_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(160)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        outer.addWidget(self.table)
        return box

    def _build_export_box(self) -> QGroupBox:
        box = QGroupBox("4 · 导出")
        lay = QVBoxLayout(box)
        self.btn_export = QPushButton("导出 label_map.csv…")
        self.btn_export.clicked.connect(self._on_export)
        lay.addWidget(self.btn_export)
        return box

    # ------------------------------------------------------------- binding
    def bind(
        self,
        project: Project,
        cube_art: ArtifactRef,
        view: HsiImageView,
        wavelength_nm: list[float] | None = None,
    ) -> None:
        """Attach the panel to a project + a rendered CUBE view."""
        self._project = project
        self._cube_art = cube_art
        self._view = view
        self._wl = wavelength_nm
        self._ref = None
        self._sim = None
        self._mask = None
        self._labels = None
        self._roiset = None
        self._mask_art_id = None

        view.cursor_moved.connect(self._on_cursor_moved)
        self._on_cursor_moved(*view.cursor_position())
        self.header.setText(f"<b>{cube_art.name}</b><br/>{view.nrows}×{view.ncols}×{view.nbands}")
        self.table.setRowCount(0)
        self.stats.setText("—")
        self._set_enabled(True)
        self.message.emit(f"分割面板已绑定：{cube_art.name}")

    def unbind(self) -> None:
        self._project = None
        self._cube_art = None
        self._view = None
        self._ref = self._sim = self._mask = self._labels = None
        self._roiset = None
        self.header.setText("未绑定数据集")
        self.table.setRowCount(0)
        self.stats.setText("—")
        self._set_enabled(False)
        self.spectra_changed.emit(None, None)

    @property
    def bound_artifact_id(self) -> str | None:
        return self._cube_art.id if self._cube_art is not None else None

    def _set_enabled(self, on: bool) -> None:
        for w in (
            self.sim_method,
            self.ref_source,
            self.ref_w,
            self.ref_h,
            self.btn_preview,
            self.thr_method,
            self.kernel_spin,
            self.open_spin,
            self.close_spin,
            self.show_overlay,
            self.alpha_slider,
            self.btn_save_mask,
            self.min_area,
            self.sort_box,
            self.btn_extract,
            self.btn_export,
        ):
            w.setEnabled(on)
        self.thr_value.setEnabled(on and self.thr_method.currentData() != "otsu")
        self.percentile.setEnabled(on)

    # -------------------------------------------------------------- helpers
    @contextmanager
    def _busy(self, text: str) -> Iterator[None]:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.message.emit(text)
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()

    def _cube(self) -> np.ndarray | None:
        return None if self._view is None else np.asarray(self._view.cube)

    def _on_cursor_moved(self, y: int, x: int) -> None:
        self.cursor_label.setText(f"y={y}  x={x}")

    def _ref_spectrum(self) -> np.ndarray | None:
        """Reference spectrum according to the current widgets."""
        view = self._view
        if view is None:
            return None
        if self.ref_source.currentData() == "center":
            row, col = view.nrows // 2, view.ncols // 2
        else:
            row, col = view.cursor_position()
        return view.spectrum_at_box(row, col, self.ref_w.value(), self.ref_h.value())

    # ------------------------------------------------------------ similarity
    def _on_preview(self) -> None:
        self._compute_similarity(force=True)
        self._apply_threshold()

    def _compute_similarity(self, *, force: bool = False) -> bool:
        cube = self._cube()
        if cube is None:
            return False
        if self._sim is not None and not force:
            return True
        ref = self._ref_spectrum() if self._ref is None else self._ref
        if ref is None:
            return False
        method = cast(SimMethod, self.sim_method.currentData())
        with self._busy(f"正在计算 {method.upper()} 相似度…"):
            sim = similarity_map(cube, ref, method=method)
        self._ref = np.asarray(ref, dtype=np.float32)
        self._sim = sim
        self.message.emit(
            f"{method.upper()} 相似度完成  [{float(np.min(sim)):.3f}, {float(np.max(sim)):.3f}]"
        )
        return True

    # ------------------------------------------------------------- threshold
    def _on_thr_method_changed(self, _: int) -> None:
        method = self.thr_method.currentData()
        self.percentile.setEnabled(method == "percentile")
        self.thr_value.setEnabled(method != "otsu")
        self._apply_threshold()

    def _apply_threshold(self) -> None:
        if self._sim is None and not self._compute_similarity():
            return
        sim = self._sim
        assert sim is not None
        method = cast(ThrMethod, self.thr_method.currentData())
        thr: float | None = None
        if method == "fixed":
            thr = float(self.thr_value.value())
        elif method == "percentile":
            thr = float(np.percentile(sim, self.percentile.value()))
        binary, used = binarize(sim, method=method, threshold=thr)
        mask = morphological_cleanup(
            binary,
            kernel_size=int(self.kernel_spin.value()),
            open_iter=int(self.open_spin.value()),
            close_iter=int(self.close_spin.value()),
        )
        self._mask = mask.astype(np.uint8)
        self._threshold_used = float(used)
        self._labels = None
        self._roiset = None
        if method == "otsu":
            self.thr_value.blockSignals(True)
            self.thr_value.setValue(float(used))
            self.thr_value.blockSignals(False)
        self._refresh_stats()
        self._update_overlay()

    def _refresh_stats(self) -> None:
        if self._mask is None:
            self.stats.setText("—")
            return
        fg = int(self._mask.sum())
        total = int(self._mask.size)
        pct = 100.0 * fg / max(total, 1)
        n_roi = 0 if self._labels is None else int(self._labels.max())
        self.stats.setText(
            f"阈值 = {self._threshold_used:.4f}   前景 = {fg:,} px ({pct:.1f}%)   ROI = {n_roi}"
        )

    def _on_alpha_changed(self, value: int) -> None:
        if self._view is not None:
            self._view.set_overlay_alpha(value / 100.0)

    def _update_overlay(self) -> None:
        view = self._view
        if view is None:
            return
        if not self.show_overlay.isChecked():
            view.clear_overlay()
            return
        alpha = self.alpha_slider.value() / 100.0
        if self._labels is not None:
            view.set_overlay(self._labels, kind="labels", alpha=alpha)
        elif self._mask is not None:
            view.set_overlay(self._mask, kind="mask", alpha=alpha)
        else:
            view.clear_overlay()

    # ------------------------------------------------------------ persistence
    def _step_params(self) -> dict[str, Any]:
        method = cast(ThrMethod, self.thr_method.currentData())
        params: dict[str, Any] = {
            "method": str(self.sim_method.currentData()),
            "threshold_method": method,
            "ref_source": "spectra",
            "ref_spectrum": [float(v) for v in np.asarray(self._ref).ravel()],
            "morph_kernel": int(self.kernel_spin.value()),
            "morph_open_iter": int(self.open_spin.value()),
            "morph_close_iter": int(self.close_spin.value()),
        }
        if method == "fixed":
            params["threshold_value"] = float(self.thr_value.value())
        elif method == "percentile":
            sim = self._sim
            assert sim is not None
            params["threshold_value"] = float(np.percentile(sim, self.percentile.value()))
        return params

    def _save_mask(self, *, quiet: bool) -> str | None:
        """Persist the current mask through ``segment.threshold``."""
        if self._project is None or self._cube_art is None:
            return None
        if not self._compute_similarity():
            return None
        assert self._sim is not None and self._ref is not None
        params = self._step_params()
        with self._busy("正在保存 Mask…"):
            result = self._project.run_step("segment.threshold", [self._cube_art], params=params)
        if result.skipped:
            outputs = self._project.find_cached_outputs(
                "segment.threshold", [self._cube_art], params
            )
        else:
            outputs = result.outputs
        if not result.ok or not outputs:
            if not quiet:
                QMessageBox.critical(self, "保存 Mask 失败", result.run.error or "step 未产生输出")
            self.message.emit(f"保存 Mask 失败：{result.run.error or '无输出'}")
            return None
        self._mask_art_id = outputs[0].id
        self.artifacts_created.emit()
        state = "命中缓存" if result.skipped else "已生成"
        self.message.emit(f"Mask {state}：{outputs[0].name} ({outputs[0].id})")
        return self._mask_art_id

    # --------------------------------------------------------------- ROIs
    def _on_extract(self) -> None:
        if self._project is None or self._cube_art is None:
            return
        if self._mask is None and not self._compute_similarity():
            return
        mask_id = self._save_mask(quiet=True)
        if mask_id is None:
            return
        mask_art = self._project.get_artifact(mask_id)
        if mask_art is None:
            return
        params = {
            "min_area": int(self.min_area.value()),
            "connectivity": 8,
            "sort_by": str(self.sort_box.currentData()),
        }
        with self._busy("正在提取连通域…"):
            result = self._project.run_step(
                "segment.connected_components", [mask_art, self._cube_art], params=params
            )
        if result.skipped:
            outputs = self._project.find_cached_outputs(
                "segment.connected_components", [mask_art, self._cube_art], params
            )
        else:
            outputs = result.outputs
        if not result.ok:
            QMessageBox.critical(self, "ROI 提取失败", result.run.error)
            self.message.emit(f"ROI 提取失败：{result.run.error}")
            return
        self.artifacts_created.emit()
        local = self._local_rois()
        self._fill_table(outputs)
        self._update_overlay()
        self._emit_spectra(local)
        n = 0 if self._roiset is None else len(self._roiset)
        self.message.emit(f"ROI 提取完成：{n} 个区域")

    def _emit_spectra(self, local: tuple[ComponentResult, ROISet] | None) -> None:
        """Forward the ROI mean spectra to the preprocessing panel."""
        cube = self._cube()
        if local is None or cube is None or len(local[1]) == 0:
            return
        comp, roiset = local
        mean = compute_mean_spectra(cube, comp.labels, [r.id for r in roiset])
        self.spectra_changed.emit(mean, list(self._wl) if self._wl else None)

    def _local_rois(self) -> tuple[ComponentResult, ROISet] | None:
        """Recompute components in memory (fast) for overlay + export."""
        if self._mask is None:
            return None
        comp = connected_components(
            self._mask,
            connectivity=8,
            min_area=int(self.min_area.value()),
        )
        roiset = ROISet.from_components(
            comp, sort_by=str(self.sort_box.currentData()), labels=comp.labels
        )
        self._labels = comp.labels
        self._roiset = roiset
        return comp, roiset

    def _fill_table(self, outputs: list[ArtifactRef]) -> None:
        """Populate the ROI table, preferring the CSV the step just wrote."""
        rows: list[tuple[int, int, float, float]] = []
        for art in outputs:
            if not art.uri.endswith(".csv") or self._project is None:
                continue
            path = art.resolve(self._project.root)
            if not path.exists():
                continue
            try:
                import pandas as pd

                df = pd.read_csv(
                    path,
                    usecols=["label", "area_px", "row_centroid", "col_centroid"],
                )
            except Exception:  # noqa: BLE001 — fall back to in-memory result
                continue
            rows = [
                (int(r.label), int(r.area_px), float(r.row_centroid), float(r.col_centroid))
                for r in df.itertuples(index=False)
            ]
            break
        if not rows and self._roiset is not None:
            rows = [(r.id, r.area, r.row_centroid, r.col_centroid) for r in self._roiset]

        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for i, (rid, area, row, col) in enumerate(rows):
            for j, text in enumerate((str(rid), f"{area:,}", f"{row:.1f}", f"{col:.1f}")):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)
        self.table.blockSignals(False)
        self._refresh_stats()

    def _on_row_selected(self) -> None:
        if self._roiset is None or self._view is None:
            return
        items = self.table.selectedItems()
        if not items:
            return
        id_item = self.table.item(items[0].row(), 0)
        if id_item is None:
            return
        roi_id = int(id_item.text())
        roi = self._roiset.get(roi_id)
        if roi is None:
            return
        self._view.set_cursor(int(round(roi.row_centroid)), int(round(roi.col_centroid)))
        self.message.emit(
            f"ROI {roi_id}: 面积 {roi.area:,} px  "
            f"bbox=({roi.bbox[0]}, {roi.bbox[1]}, {roi.bbox[2]}×{roi.bbox[3]})"
        )

    # -------------------------------------------------------------- export
    def _on_export(self) -> None:
        cube = self._cube()
        if cube is None or self._mask is None:
            QMessageBox.information(self, "导出", "请先计算 Mask 与 ROI。")
            return
        local = self._local_rois()
        if local is None:
            return
        comp, roiset = local
        if len(roiset) == 0:
            QMessageBox.information(self, "导出", "当前 Mask 没有 ROI 可导出。")
            return
        base = self._cube_art.name if self._cube_art is not None else "rois"
        default = str(Path.home() / f"{base}_label_map.csv")
        path, _ = QFileDialog.getSaveFileName(self, "导出 label_map.csv", default, "CSV (*.csv)")
        if not path:
            return
        mean = compute_mean_spectra(cube, comp.labels, [r.id for r in roiset])
        roiset.write_csv(Path(path), mean_spectra=mean, wavelength_nm=self._wl)
        self.message.emit(f"已导出 {len(roiset)} 个 ROI → {path}")
