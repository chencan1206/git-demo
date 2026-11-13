"""Hyperspectral image viewer widget (2D fake-color + band slider + spectrum).

This is a thin Qt widget that renders a 3-D ``(y, x, band)`` cube. Stage 1
focuses on:
  * fake-color RGB synthesis from 3 bands
  * band slider to inspect single-band grayscale slices
  * cross-hair cursor + live spectrum plot
  * cursor-time-band trade-off cached in-memory (no paging yet)

Stage 2 adds a translucent **overlay layer** so the segmentation panel can
paint a binary mask (red wash) or a labelled ROI map (one colour per label)
on top of whichever base image is currently rendered.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


def _normalize(x: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    lo = float(np.nanpercentile(x, p_low))
    hi = float(np.nanpercentile(x, p_high))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


#: Palette used for labelled ROI maps — 1-based label *i* takes colour ``i-1``.
LABEL_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.00, 0.25, 0.30),
    (0.20, 0.75, 1.00),
    (0.35, 0.90, 0.35),
    (1.00, 0.80, 0.20),
    (0.80, 0.40, 1.00),
    (1.00, 0.55, 0.15),
    (0.20, 0.95, 0.85),
    (0.95, 0.50, 0.75),
)


def _label_palette(n: int) -> np.ndarray:
    """Return an ``(n, 3)`` float RGB table for labels ``1..n``."""
    base = np.asarray(LABEL_COLORS, dtype=np.float32)
    reps = int(np.ceil(max(n, 1) / len(base)))
    table = np.tile(base, (reps, 1))
    return table[: max(n, 1)]


def _to_qimage_rgb(rgb: np.ndarray) -> QImage:
    """Convert a float ``(h, w, 3)`` array in [0, 1] into a QImage."""
    buf = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    h, w, _ = buf.shape
    qimg = QImage(buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return qimg.copy()  # detach from the numpy buffer


class HsiImageView(QWidget):
    """Two-pane view: image (top-left) + spectrum (bottom-right).

    Signals:
        cursor_moved(y, x) — emitted when the cross-hair cursor moves over the image.
    """

    cursor_moved = Signal(int, int)

    def __init__(
        self,
        cube: np.ndarray,
        wavelength_nm: Sequence[float] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        assert cube.ndim == 3, f"expected (y,x,band), got {cube.shape}"
        self.cube = np.asarray(cube, dtype=np.float32)
        self.nrows, self.ncols, self.nbands = self.cube.shape
        self.wl = np.asarray(wavelength_nm or np.arange(self.nbands, dtype=float))

        # Overlay state must exist before any layout-driven resize/repaint.
        self._overlay: np.ndarray | None = None
        self._overlay_kind: str = "mask"  # "mask" (binary) | "labels" (ROI map)
        self._overlay_alpha: float = 0.45
        self._overlay_color = np.asarray((1.0, 0.15, 0.15), dtype=np.float32)

        # -- layout --
        root = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Band:"))
        self.band_slider = QSlider(Qt.Orientation.Horizontal)
        self.band_slider.setRange(0, self.nbands - 1)
        self.band_slider.setValue(self.nbands // 2)
        self.band_slider.valueChanged.connect(self._on_band_changed)
        top_bar.addWidget(self.band_slider, stretch=1)
        self.band_label = QLabel()
        top_bar.addWidget(self.band_label)

        top_bar.addSpacing(16)
        top_bar.addWidget(QLabel("合成:"))
        self.mode = QComboBox()
        self.mode.addItems(["假彩色", "单波段"])
        self.mode.currentIndexChanged.connect(lambda _: self._render())
        top_bar.addWidget(self.mode)
        top_bar.addWidget(QLabel("R"))
        self.rband = self._mk_band_combo(50)
        top_bar.addWidget(self.rband)
        top_bar.addWidget(QLabel("G"))
        self.gband = self._mk_band_combo(150)
        top_bar.addWidget(self.gband)
        top_bar.addWidget(QLabel("B"))
        self.bband = self._mk_band_combo(250)
        top_bar.addWidget(self.bband)
        for c in (self.rband, self.gband, self.bband):
            c.currentIndexChanged.connect(lambda _: self._render())
        top_bar.addStretch()
        root.addLayout(top_bar)

        split = QSplitter(Qt.Orientation.Vertical)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMouseTracking(True)
        # Patch the handler — Qt forbids method override on a widget instance,
        # but monkey-patching works fine for our single-callback case.
        self.image_label.mouseMoveEvent = self._on_mouse_move  # type: ignore[method-assign]
        split.addWidget(self.image_label)

        # Spectrum plot — matplotlib embedded so no pyqtgraph dep needed here
        self._init_spectrum_plot()
        split.addWidget(self._spectrum_widget)
        split.setSizes([int(self.nrows * 3), int(self.nrows * 2)])
        root.addWidget(split, stretch=1)

        self.status_label = QLabel(
            f"{self.nrows}×{self.ncols}×{self.nbands}  band={self.band_slider.value()}"
        )
        root.addWidget(self.status_label)

        self._cursor_y = self.nrows // 2
        self._cursor_x = self.ncols // 2

        self._render()
        self._update_spectrum()

    def _mk_band_combo(self, default: int) -> QComboBox:
        cb = QComboBox()
        for b in range(self.nbands):
            cb.addItem(f"{b:3d}  {self.wl[b]:.1f} nm", b)
        cb.setCurrentIndex(max(0, min(default, self.nbands - 1)))
        return cb

    # ------------------------------------------------------------- rendering
    def _render(self) -> None:
        self._render_image_only()
        self.band_label.setText(
            f"band={self.band_slider.value()}  ({self.wl[int(self.band_slider.value())]:.1f} nm)"
        )
        self.status_label.setText(
            f"y={self._cursor_y} x={self._cursor_x}  band={self.band_slider.value()}  "
            f"wl={self.wl[int(self.band_slider.value())]:.1f} nm"
        )
        self._update_spectrum()

    def _render_image_only(self) -> None:
        mode = self.mode.currentText()
        if mode == "单波段":
            b = int(self.band_slider.value())
            gray = _normalize(self.cube[:, :, b])
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
        else:
            r = int(self.rband.currentData())
            g = int(self.gband.currentData())
            b = int(self.bband.currentData())
            rgb = np.stack(
                [
                    _normalize(self.cube[:, :, r]),
                    _normalize(self.cube[:, :, g]),
                    _normalize(self.cube[:, :, b]),
                ],
                axis=-1,
            )
        rgb = self._composite_overlay(np.ascontiguousarray(rgb, dtype=np.float32))
        pix = QPixmap.fromImage(_to_qimage_rgb(rgb))
        area = self.image_label.size()
        if area.width() > 40 and area.height() > 40:
            pix = pix.scaled(
                area,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.image_label.setPixmap(pix)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        self._render_image_only()

    def _composite_overlay(self, rgb: np.ndarray) -> np.ndarray:
        """Alpha-blend the current mask/label overlay onto an RGB base image."""
        ov = self._overlay
        if ov is None or self._overlay_alpha <= 0.0:
            return rgb
        sel = ov > 0
        if not sel.any():
            return rgb
        # Desaturate + darken the base under the overlay so mask colours read
        # clearly no matter which fake-colour bands are selected.
        out = rgb.copy()
        lum = out @ np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
        base = np.repeat((lum * 0.55)[:, :, None], 3, axis=2)
        a = self._overlay_alpha
        if self._overlay_kind == "labels":
            n = int(ov.max())
            if n <= 0:
                return rgb
            table = _label_palette(n)
            out[sel] = base[sel] * (1.0 - a) + table[ov[sel] - 1] * a
        else:
            color = np.broadcast_to(self._overlay_color, out[sel].shape)
            out[sel] = base[sel] * (1.0 - a) + color * a
        return out

    # ------------------------------------------------------------- overlay
    def set_overlay(
        self,
        data: np.ndarray | None,
        *,
        kind: str = "mask",
        alpha: float | None = None,
    ) -> None:
        """Paint ``data`` (``(rows, cols)`` mask or label map) over the image."""
        if data is None:
            self.clear_overlay()
            return
        arr = np.asarray(data)
        if arr.shape != (self.nrows, self.ncols):
            raise ValueError(f"overlay shape {arr.shape} != image ({self.nrows}, {self.ncols})")
        self._overlay = arr.astype(np.int32, copy=False) if kind == "labels" else arr
        self._overlay_kind = kind
        if alpha is not None:
            self._overlay_alpha = float(alpha)
        self._render()

    def clear_overlay(self) -> None:
        self._overlay = None
        self._render()

    def set_overlay_alpha(self, alpha: float) -> None:
        self._overlay_alpha = float(max(0.0, min(1.0, alpha)))
        self._render()

    @property
    def overlay_visible(self) -> bool:
        return self._overlay is not None and self._overlay_alpha > 0.0

    # -------------------------------------------------------------- cursor
    def cursor_position(self) -> tuple[int, int]:
        """Return the cross-hair position as ``(row, col)``."""
        return self._cursor_y, self._cursor_x

    def set_cursor(self, row: int, col: int) -> None:
        self._cursor_y = int(max(0, min(self.nrows - 1, row)))
        self._cursor_x = int(max(0, min(self.ncols - 1, col)))
        self.cursor_moved.emit(self._cursor_y, self._cursor_x)
        self._render()

    def cursor_spectrum(self) -> np.ndarray:
        """Return the spectrum at the cross-hair as a ``(bands,)`` array."""
        return np.asarray(self.cube[self._cursor_y, self._cursor_x, :])

    def spectrum_at_box(self, row: int, col: int, width: int, height: int) -> np.ndarray:
        """Mean spectrum of the ``width × height`` box centred on ``(row, col)``."""
        hh, hw = max(1, height) // 2, max(1, width) // 2
        r0, r1 = max(0, row - hh), min(self.nrows, row + hh + 1)
        c0, c1 = max(0, col - hw), min(self.ncols, col + hw + 1)
        patch = self.cube[r0:r1, c0:c1, :]
        if patch.size == 0:
            return self.cursor_spectrum()
        return np.asarray(patch.reshape(-1, self.nbands).mean(axis=0), dtype=np.float32)

    def _on_band_changed(self, _: int) -> None:
        self._render()

    # ------------------------------------------------------- cursor/spectrum
    def _on_mouse_move(self, ev) -> None:
        pm = self.image_label.pixmap()
        if pm is None:
            return
        rect = self.image_label.contentsRect()
        pm_w = pm.width()
        pm_h = pm.height()
        scale_x = self.ncols / pm_w if pm_w else 1.0
        scale_y = self.nrows / pm_h if pm_h else 1.0
        px = float(ev.position().x()) - (rect.width() - pm_w) / 2
        py = float(ev.position().y()) - (rect.height() - pm_h) / 2
        cx = int(max(0, min(self.ncols - 1, px * scale_x)))
        cy = int(max(0, min(self.nrows - 1, py * scale_y)))
        self._cursor_x = cx
        self._cursor_y = cy
        self.cursor_moved.emit(cy, cx)
        self._render()

    # ---------------------------------------------------- matplotlib pane
    def _init_spectrum_plot(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(6, 3))
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel("Wavelength (nm)")
        self._ax.set_ylabel("Reflectance")
        (self._line,) = self._ax.plot([], [], lw=1.0)
        (self._cursor_dot,) = self._ax.plot([], [], "ro", ms=3)
        self._ax.set_xlim(float(self.wl[0]), float(self.wl[-1]))
        self._ax.set_ylim(0, 1)
        self._ax.grid(alpha=0.3)
        self._spectrum_widget = FigureCanvas(self._fig)
        self._spectrum_widget.setMinimumHeight(140)

    def _update_spectrum(self) -> None:
        x = self.wl
        y = self.cube[self._cursor_y, self._cursor_x, :]
        self._line.set_data(x, y)
        self._cursor_dot.set_data(
            [self.wl[int(self.band_slider.value())]],
            [y[int(self.band_slider.value())]],
        )
        self._ax.set_ylim(max(0.0, float(y.min() - 0.05)), min(1.0, float(y.max() + 0.05)))
        self._spectrum_widget.draw_idle()
