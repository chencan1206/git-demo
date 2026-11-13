"""Preprocessing dock: chain editor + before/after curve preview.

The panel is a thin shell over :class:`~plant_hsi.preprocess.PreprocessChain`
— every edit updates the chain snapshot, and "应用预览" runs the chain on the
bound spectra (ROI mean spectra emitted by the segmentation panel).
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from plant_hsi.preprocess import OPS, PreprocessChain, PreprocessOp

_DEFAULT_PARAMS: dict[str, dict[str, object]] = {
    "sg": {"window": 7, "polyorder": 2},
    "moving_average": {"window": 5},
    "derivative": {"order": 1},
    "fractional_derivative": {"order": 1.0, "memory": 16},
}

_BEFORE = QColor("#9aa0a6")
_AFTER = QColor("#4fc3f7")
_BG = QColor("#1e1f22")
_GRID = QColor("#3c4043")
_TEXT = QColor("#e8eaed")


class CurvePlot(QWidget):
    """Minimal two-curve plot (no matplotlib dependency)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._before: np.ndarray | None = None
        self._after: np.ndarray | None = None

    @property
    def has_data(self) -> bool:
        return self._before is not None or self._after is not None

    def set_curves(
        self,
        before: np.ndarray | None,
        after: np.ndarray | None = None,
    ) -> None:
        self._before = _as_curve(before)
        self._after = _as_curve(after)
        self.update()

    def clear(self) -> None:
        self._before = self._after = None
        self.update()

    # ------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 — Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BG)
        rect = self.rect().adjusted(40, 12, -12, -22)
        painter.setPen(QPen(_GRID, 1))
        painter.drawRect(rect)
        for i in range(1, 4):
            y = rect.top() + rect.height() * i / 4
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))

        if not self.has_data:
            painter.setPen(QPen(_TEXT))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无光谱数据")
            painter.end()
            return

        curves = [c for c in (self._before, self._after) if c is not None and c.size]
        if not curves:
            painter.end()
            return
        lo = min(float(np.min(c)) for c in curves)
        hi = max(float(np.max(c)) for c in curves)
        span = hi - lo if hi - lo > 1e-9 else 1.0

        def to_poly(curve: np.ndarray) -> QPolygonF:
            n = curve.size
            pts = []
            for i, value in enumerate(curve):
                x = rect.left() + (rect.width() * i / max(n - 1, 1))
                y = rect.bottom() - rect.height() * (float(value) - lo) / span
                pts.append(QPointF(x, y))
            return QPolygonF(pts)

        if self._before is not None:
            painter.setPen(QPen(_BEFORE, 1.2))
            painter.drawPolyline(to_poly(self._before))
        if self._after is not None:
            painter.setPen(QPen(_AFTER, 1.6))
            painter.drawPolyline(to_poly(self._after))

        painter.setPen(QPen(_BEFORE))
        painter.drawText(rect.left() + 6, rect.top() + 14, "处理前")
        painter.setPen(QPen(_AFTER))
        painter.drawText(rect.left() + 60, rect.top() + 14, "处理后")
        painter.setPen(QPen(_TEXT))
        painter.drawText(6, self.height() - 6, f"{lo:.3f} … {hi:.3f}")
        painter.end()


class PreprocessPanel(QWidget):
    """Chain editor for one bound spectra matrix."""

    message = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spectra: np.ndarray | None = None
        self._wavelength: list[float] | None = None
        self._chain = PreprocessChain()
        self._processed: np.ndarray | None = None

        root = QVBoxLayout(self)
        self.header = QLabel("未绑定光谱（先在分割面板提取 ROI）")
        self.header.setWordWrap(True)
        root.addWidget(self.header)

        root.addWidget(self._build_chain_box())
        root.addWidget(self._build_param_box())

        self.btn_apply = QPushButton("应用预览")
        self.btn_apply.clicked.connect(self.apply_preview)
        root.addWidget(self.btn_apply)

        self.plot = CurvePlot()
        root.addWidget(self.plot)

        self.stats = QLabel("—")
        self.stats.setWordWrap(True)
        root.addWidget(self.stats)
        root.addStretch(1)

        self._set_enabled(False)

    # ------------------------------------------------------------- building
    def _build_chain_box(self) -> QGroupBox:
        box = QGroupBox("预处理链（自上而下执行）")
        lay = QVBoxLayout(box)

        self.op_list = QListWidget()
        self.op_list.setMinimumHeight(110)
        self.op_list.currentRowChanged.connect(self._on_selection_changed)
        lay.addWidget(self.op_list)

        add_row = QHBoxLayout()
        self.op_combo = QComboBox()
        for kind, (label, _) in OPS.items():
            self.op_combo.addItem(label, kind)
        add_row.addWidget(self.op_combo, stretch=1)
        self.btn_add = QPushButton("添加")
        self.btn_add.clicked.connect(self._on_add)
        add_row.addWidget(self.btn_add)
        lay.addLayout(add_row)

        btn_row = QHBoxLayout()
        self.btn_up = QPushButton("上移")
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down = QPushButton("下移")
        self.btn_down.clicked.connect(lambda: self._move(1))
        self.btn_remove = QPushButton("删除")
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self._on_clear)
        for b in (self.btn_up, self.btn_down, self.btn_remove, self.btn_clear):
            btn_row.addWidget(b)
        lay.addLayout(btn_row)
        return box

    def _build_param_box(self) -> QGroupBox:
        box = QGroupBox("参数")
        outer = QVBoxLayout(box)
        self._param_host = QWidget()
        self.param_form = QFormLayout(self._param_host)
        self.param_form.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._param_host)
        return box

    # ------------------------------------------------------------- public API
    @property
    def chain(self) -> PreprocessChain:
        return self._chain

    @property
    def processed(self) -> np.ndarray | None:
        return self._processed

    def bind(
        self,
        spectra: np.ndarray,
        wavelength_nm: list[float] | None = None,
    ) -> None:
        arr = np.asarray(spectra, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"expected 2-D spectra, got shape {arr.shape}")
        self._spectra = arr
        self._wavelength = list(wavelength_nm) if wavelength_nm else None
        self._processed = None
        wl = ""
        if self._wavelength and len(self._wavelength) == arr.shape[1]:
            wl = f"  {self._wavelength[0]:.1f}–{self._wavelength[-1]:.1f} nm"
        self.header.setText(f"<b>{arr.shape[0]} 条光谱 × {arr.shape[1]} 波段</b>{wl}")
        self.plot.set_curves(arr)
        self.stats.setText("—")
        self._set_enabled(True)
        self.message.emit(f"预处理面板已绑定 {arr.shape[0]} 条光谱")

    def unbind(self) -> None:
        self._spectra = None
        self._wavelength = None
        self._processed = None
        self.header.setText("未绑定光谱（先在分割面板提取 ROI）")
        self.plot.clear()
        self.stats.setText("—")
        self._set_enabled(False)

    def add_operator(self, kind: str, **params: object) -> None:
        merged = dict(_DEFAULT_PARAMS.get(kind, {}))
        merged.update(params)
        self._chain.add(kind, **merged)
        self._refresh_list(select=len(self._chain) - 1)
        self.message.emit(f"已添加：{self._chain.ops[-1].describe()}")

    def apply_preview(self) -> np.ndarray | None:
        if self._spectra is None:
            return None
        if len(self._chain) == 0:
            self._processed = None
            self.plot.set_curves(self._spectra)
            self.stats.setText("链为空；显示原始光谱。")
            self.message.emit("预处理链为空。")
            return None
        started = time.perf_counter()
        try:
            out, report = self._chain.apply_with_report(self._spectra)
        except ValueError as exc:
            self.message.emit(f"预处理失败：{exc}")
            self.stats.setText(f"失败：{exc}")
            return None
        elapsed = (time.perf_counter() - started) * 1000.0

        self._processed = out
        self.plot.set_curves(self._spectra, out)
        self.stats.setText(
            f"{len(self._chain)} 步 · {out.shape[0]}×{out.shape[1]} · "
            f"{elapsed:.1f} ms · 越界 {report.out_of_range_after} · "
            f"NaN {report.nan_after} · Inf {report.inf_after}"
        )
        self.message.emit(
            f"预处理完成：{len(self._chain)} 步，{elapsed:.1f} ms，"
            f"越界 {report.out_of_range_after} 个值（已记录，未裁剪）"
        )
        return out

    # ------------------------------------------------------------- internals
    def _set_enabled(self, on: bool) -> None:
        for w in (
            self.op_list,
            self.op_combo,
            self.btn_add,
            self.btn_up,
            self.btn_down,
            self.btn_remove,
            self.btn_clear,
            self.btn_apply,
        ):
            w.setEnabled(on)

    def _refresh_list(self, select: int | None = None) -> None:
        self.op_list.blockSignals(True)
        self.op_list.clear()
        for op in self._chain:
            self.op_list.addItem(f"{op.label}   {_fmt_params(op)}")
        self.op_list.blockSignals(False)
        if select is not None and 0 <= select < self.op_list.count():
            self.op_list.setCurrentRow(select)
        self._on_selection_changed(self.op_list.currentRow())

    def _on_add(self) -> None:
        kind = self.op_combo.currentData()
        if kind:
            self.add_operator(str(kind))

    def _on_remove(self) -> None:
        row = self.op_list.currentRow()
        if row < 0:
            return
        removed = self._chain.remove(row)
        self.message.emit(f"已删除：{removed.describe()}")
        self._refresh_list(select=min(row, len(self._chain) - 1))

    def _on_clear(self) -> None:
        self._chain.clear()
        self._refresh_list()
        self.message.emit("预处理链已清空")

    def _move(self, delta: int) -> None:
        row = self.op_list.currentRow()
        if row < 0:
            return
        new_row = self._chain.move(row, delta)
        self._refresh_list(select=new_row)

    def _on_selection_changed(self, row: int) -> None:
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        if not 0 <= row < len(self._chain):
            self.param_form.addRow(QLabel("选择一个算子以编辑参数"))
            return
        op = self._chain.ops[row]
        self._build_param_widgets(row, op)

    def _replace_op(self, row: int, **params: object) -> None:
        if not 0 <= row < len(self._chain):
            return
        try:
            self._chain.ops[row] = PreprocessOp(self._chain.ops[row].kind, params)
        except ValueError as exc:
            self.message.emit(f"参数无效：{exc}")
            return
        item = self.op_list.item(row)
        if item is not None:
            op = self._chain.ops[row]
            item.setText(f"{op.label}   {_fmt_params(op)}")

    def _build_param_widgets(self, row: int, op: PreprocessOp) -> None:
        params = op.params

        if op.kind == "sg":
            window = QSpinBox()
            window.setRange(3, 99)
            window.setSingleStep(2)
            window.setValue(int(params.get("window", 7)))
            poly = QSpinBox()
            poly.setRange(1, 6)
            poly.setValue(int(params.get("polyorder", 2)))
            deriv = QSpinBox()
            deriv.setRange(0, 2)
            deriv.setValue(int(params.get("deriv", 0)))

            def _update_sg() -> None:
                self._replace_op(
                    row,
                    window=window.value(),
                    polyorder=poly.value(),
                    deriv=deriv.value(),
                )

            window.valueChanged.connect(lambda _: _update_sg())
            poly.valueChanged.connect(lambda _: _update_sg())
            deriv.valueChanged.connect(lambda _: _update_sg())
            self.param_form.addRow("窗口", window)
            self.param_form.addRow("多项式阶", poly)
            self.param_form.addRow("求导阶", deriv)
            return

        if op.kind == "moving_average":
            window = QSpinBox()
            window.setRange(1, 101)
            window.setValue(int(params.get("window", 5)))
            window.valueChanged.connect(lambda v: self._replace_op(row, window=v))
            self.param_form.addRow("窗口", window)
            return

        if op.kind == "derivative":
            deriv_order = QSpinBox()
            deriv_order.setRange(1, 3)
            deriv_order.setValue(int(params.get("order", 1)))
            deriv_delta = QDoubleSpinBox()
            deriv_delta.setDecimals(2)
            deriv_delta.setRange(0.1, 10.0)
            deriv_delta.setValue(float(params.get("delta", 1.0)))

            def _update_deriv() -> None:
                self._replace_op(row, order=deriv_order.value(), delta=deriv_delta.value())

            deriv_order.valueChanged.connect(lambda _: _update_deriv())
            deriv_delta.valueChanged.connect(lambda _: _update_deriv())
            self.param_form.addRow("阶数", deriv_order)
            self.param_form.addRow("步长", deriv_delta)
            return

        if op.kind == "fractional_derivative":
            fod_order = QDoubleSpinBox()
            fod_order.setDecimals(2)
            fod_order.setRange(0.0, 2.0)
            fod_order.setSingleStep(0.1)
            fod_order.setValue(float(params.get("order", 1.0)))
            fod_memory = QSpinBox()
            fod_memory.setRange(1, 200)
            fod_memory.setValue(int(params.get("memory", 16)))
            fod_delta = QDoubleSpinBox()
            fod_delta.setDecimals(2)
            fod_delta.setRange(0.1, 10.0)
            fod_delta.setValue(float(params.get("delta", 1.0)))

            def _update_fod() -> None:
                self._replace_op(
                    row,
                    order=fod_order.value(),
                    memory=fod_memory.value(),
                    delta=fod_delta.value(),
                )

            fod_order.valueChanged.connect(lambda _: _update_fod())
            fod_memory.valueChanged.connect(lambda _: _update_fod())
            fod_delta.valueChanged.connect(lambda _: _update_fod())
            self.param_form.addRow("阶数 r", fod_order)
            self.param_form.addRow("记忆长度", fod_memory)
            self.param_form.addRow("步长", fod_delta)
            return

        label = QLabel("该算子无参数")
        label.setWordWrap(True)
        self.param_form.addRow(label)


def _as_curve(data: np.ndarray | None) -> np.ndarray | None:
    """Collapse ``(n_samples, bands)`` to a mean curve for display."""
    if data is None:
        return None
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    while arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr


def _fmt_params(op: PreprocessOp) -> str:
    if not op.params:
        return ""
    return "(" + ", ".join(f"{k}={v}" for k, v in op.params.items()) + ")"
