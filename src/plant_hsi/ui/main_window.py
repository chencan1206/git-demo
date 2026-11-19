"""Main window: project shell + data menu + dataset management + HSI view.

Stage 1 keeps it intentionally small — a QDockWidget on the left lists
artifacts, a QTabWidget in the centre holds HSI viewers, and a File/Data/Help
menu wire the common flows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QAction, QKeySequence, QScreen
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from plant_hsi import __version__
from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.project import Project, ProjectError
from plant_hsi.ui.hsi_view import HsiImageView
from plant_hsi.ui.model_panel import ModelingPanel
from plant_hsi.ui.preprocess_panel import PreprocessPanel
from plant_hsi.ui.segment_panel import SegmentPanel


class MainWindow(QMainWindow):
    # Hard caps — never open a window larger than these, even on a 4K screen.
    _MAX_W = 1400
    _MAX_H = 900
    # Floor — smaller screens will get an 85 % clip of the usable area so that
    # the user can still drag/maximise manually, and tiny 1024×640 panels at
    # least keep all dock widgets accessible.
    _MIN_W = 1024
    _MIN_H = 640

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PHS Studio — plant-hsi-studio")
        self._project: Project | None = None
        self._open_views: dict[str, HsiImageView] = {}  # artifact.id → view
        self._view_wl: dict[str, list[float] | None] = {}  # artifact.id → wavelengths

        self._apply_initial_geometry()

        self._build_menu()
        self._build_central()
        self._build_segment_dock()
        self.setStatusBar(QStatusBar())
        self._refresh_resource_tree()

    # ------------------------------------------------------------------ UI
    def _apply_initial_geometry(self) -> None:
        """Pick a sensible starting size that fits on the current screen.

        ``resize(1400, 900)`` used to be hard-coded, which spilled out past the
        right/bottom edge on 1366×768 laptops and left users with a window they
        couldn't drag back — exactly the complaint we got.
        """
        screen: QScreen | None = self.screen()
        if screen is None:
            # ``self.screen()`` only returns a real screen after show(); before
            # that fall back to the primary screen via the app instance.
            app = QApplication.instance()
            screen = app.primaryScreen() if app else None
        if screen is None:
            self.resize(self._MAX_W, self._MAX_H)
            return

        avail = screen.availableGeometry()  # excludes taskbar / dock
        w = min(int(avail.width() * 0.92), self._MAX_W)
        h = min(int(avail.height() * 0.92), self._MAX_H)
        w = max(w, self._MIN_W)
        h = max(h, self._MIN_H)

        self.resize(w, h)
        # Centre the window on the usable area so nothing falls off the edge.
        self.move(
            QPoint(
                avail.x() + (avail.width() - w) // 2,
                avail.y() + (avail.height() - h) // 2,
            )
        )
        # If the geometry is still larger than the available rectangle
        # (e.g. an ultra-small secondary monitor), clamp to the screen.
        if (w, h) > (avail.width(), avail.height()):
            self.setGeometry(avail)  # type: ignore[arg-type]

        self.setMinimumSize(self._MIN_W, self._MIN_H)

    # ------------------------------------------------------------------ UI
    def _build_menu(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu("&File")
        a_new = QAction("&New Project…", self)
        a_new.setShortcut(QKeySequence(QKeySequence.StandardKey.New))
        a_new.triggered.connect(self._on_new)
        m_file.addAction(a_new)
        a_open = QAction("&Open Project…", self)
        a_open.setShortcut(QKeySequence(QKeySequence.StandardKey.Open))
        a_open.triggered.connect(self._on_open)
        m_file.addAction(a_open)
        a_save = QAction("&Save", self)
        a_save.setShortcut(QKeySequence(QKeySequence.StandardKey.Save))
        a_save.triggered.connect(self._on_save)
        m_file.addAction(a_save)
        m_file.addSeparator()
        a_quit = QAction("&Quit", self)
        a_quit.setShortcut(QKeySequence(QKeySequence.StandardKey.Quit))
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_data = mb.addMenu("&Data")
        a_import = QAction("&Import RAW Dataset…", self)
        a_import.triggered.connect(self._on_import_dataset)
        m_data.addAction(a_import)
        a_open_view = QAction("&Open Selected Dataset", self)
        a_open_view.setShortcut("Ctrl+Shift+O")
        a_open_view.triggered.connect(self._open_selected_dataset)
        m_data.addAction(a_open_view)

        m_seg = mb.addMenu("&Segment")
        a_save_mask = QAction("&Save Mask", self)
        a_save_mask.setShortcut("Ctrl+M")
        a_save_mask.triggered.connect(lambda: self.seg_panel.btn_save_mask.click())
        m_seg.addAction(a_save_mask)
        a_extract = QAction("&Extract ROIs", self)
        a_extract.setShortcut("Ctrl+R")
        a_extract.triggered.connect(lambda: self.seg_panel.btn_extract.click())
        m_seg.addAction(a_extract)
        a_export = QAction("E&xport label_map.csv…", self)
        a_export.triggered.connect(lambda: self.seg_panel.btn_export.click())
        m_seg.addAction(a_export)

        m_model = mb.addMenu("&Model")
        a_train = QAction("&Train Model", self)
        a_train.setShortcut("Ctrl+T")
        a_train.triggered.connect(lambda: self.model_panel.btn_train.click())
        m_model.addAction(a_train)
        a_report = QAction("&Build HTML Report", self)
        a_report.setShortcut("Ctrl+Shift+R")
        a_report.triggered.connect(lambda: self.model_panel.btn_report.click())
        m_model.addAction(a_report)
        a_open_report = QAction("&Open Report", self)
        a_open_report.triggered.connect(lambda: self.model_panel.btn_open_report.click())
        m_model.addAction(a_open_report)

        # View 菜单 —— 关键：把所有 dock 面板的 toggleAction 挂进来，
        # 用户一旦点了面板上的关闭按钮，还能从这里点一下把它再打开。
        self.m_view = mb.addMenu("&View")
        self._dock_actions: dict[str, QAction] = {}

        m_help = mb.addMenu("&Help")
        a_about = QAction("&About", self)
        a_about.triggered.connect(
            lambda: QMessageBox.information(self, "About", f"PHS Studio {__version__}")
        )
        m_help.addAction(a_about)

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Resource", "Kind", "Size"])
        self.tree.setRootIsDecorated(True)
        self.tree.itemDoubleClicked.connect(lambda i, _: self._open_dataset_for_item(i))
        layout.addWidget(self.tree, stretch=1)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs, stretch=2)

    def _register_dock(self, dock: QDockWidget, key: str) -> None:
        """挂到 View 菜单 + 记住它的切换动作。

        QDockWidget.toggleViewAction() 是 Qt 自带的、自动在
        显示/隐藏之间切换的 QAction——把它挂到菜单里，用户点菜单就等价于
        点一下 dock 自己的关闭按钮（再点一次就再打开）。
        """
        action = dock.toggleViewAction()
        action.setText(dock.windowTitle())
        self.m_view.addAction(action)
        self._dock_actions[key] = action

    def _build_segment_dock(self) -> None:
        self.seg_panel = SegmentPanel(self)
        self.seg_panel.artifacts_created.connect(self._refresh_resource_tree)
        self.seg_panel.message.connect(self._show_status)
        dock = QDockWidget("分割 / Segmentation", self)
        dock.setObjectName("segmentation_dock")
        dock.setWidget(self.seg_panel)
        dock.setMinimumWidth(280)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._register_dock(dock, "segment")
        self.seg_dock = dock

        self.preprocess_panel = PreprocessPanel(self)
        self.preprocess_panel.message.connect(self._show_status)
        self.seg_panel.spectra_changed.connect(self._on_spectra_ready)
        dock2 = QDockWidget("光谱预处理 / Preprocessing", self)
        dock2.setObjectName("preprocess_dock")
        dock2.setWidget(self.preprocess_panel)
        dock2.setMinimumWidth(280)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock2)
        self._register_dock(dock2, "preprocess")
        self.preprocess_dock = dock2

        self.model_panel = ModelingPanel(self)
        self.model_panel.message.connect(self._show_status)
        self.model_panel.set_preprocess_chain_provider(lambda: self.preprocess_panel.chain)
        dock3 = QDockWidget("建模与报告 / Modeling", self)
        dock3.setObjectName("modeling_dock")
        dock3.setWidget(self.model_panel)
        dock3.setMinimumWidth(280)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock3)
        self._register_dock(dock3, "model")
        self.model_dock = dock3

        self.dl_panel: Any = None
        try:
            from plant_hsi.ui.dl_panel import DeepLearningPanel
        except ImportError:  # PyTorch is an optional dependency
            pass
        else:
            self.dl_panel = DeepLearningPanel(self)
            self.dl_panel.message.connect(self._show_status)
            self.dl_panel.artifacts_created.connect(self._refresh_resource_tree)
            dock4 = QDockWidget("深度学习 / DC²Net", self)
            dock4.setObjectName("dl_dock")
            dock4.setWidget(self.dl_panel)
            dock4.setMinimumWidth(280)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock4)
            self._register_dock(dock4, "dl")
            self.dl_dock = dock4

    def _on_spectra_ready(self, spectra: object, wavelength: object) -> None:
        """Bind the preprocessing and modeling panels to fresh ROI spectra."""
        if spectra is None:
            self.preprocess_panel.unbind()
            self.model_panel.unbind()
            return
        wl = [float(v) for v in wavelength] if isinstance(wavelength, list) else None
        arr = np.asarray(spectra)
        self.preprocess_panel.bind(arr, wl)
        if self._project is not None:
            self.model_panel.bind(self._project, arr, wl)
        else:
            self.model_panel.unbind()

    def _show_status(self, text: str) -> None:
        self.statusBar().showMessage(text, 8000)

    # ---------------------------------------------------------- slots
    def _on_new(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "New project folder", str(Path.home()), "")
        if not path:
            return
        try:
            self._project = Project.create(Path(path), Path(path).name)
        except ProjectError as exc:
            QMessageBox.critical(self, "New project failed", str(exc))
            return
        self._reset_views()
        self._refresh_resource_tree()
        self.statusBar().showMessage(f"Created: {self._project.root}")

    def _on_open(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open project", str(Path.home()))
        if not path:
            return
        try:
            self._project = Project.open(Path(path))
        except ProjectError as exc:
            QMessageBox.critical(self, "Open project failed", str(exc))
            return
        self._reset_views()
        self._refresh_resource_tree()
        self.statusBar().showMessage(f"Opened: {self._project.root}")

    def _on_save(self) -> None:
        if self._project is None:
            QMessageBox.warning(self, "Save", "No project is open.")
            return
        self._project.save()
        self.statusBar().showMessage(f"Saved: {self._project.root}")

    def _on_import_dataset(self) -> None:
        if self._project is None:
            QMessageBox.warning(self, "Import", "Open a project first.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select hyperspectral files",
            str(Path.home()),
            "HSI (*.mat *.hdr *.raw *.tif *.tiff);;All (*)",
        )
        if not files:
            return
        imported = 0
        for f in files:
            try:
                imported += self._import_one_file(Path(f))
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Import failed", f"{f}\n{exc}")
        self._refresh_resource_tree()
        self.statusBar().showMessage(f"Imported {imported} new dataset(s).")

    def _import_one_file(self, src: Path) -> int:
        """Copy the raw file into the project + run `io.import_dataset`.

        Returns 1 if a new CUBE artifact was produced, 0 if the file was
        already imported (hash match).
        """
        assert self._project is not None
        prj = self._project
        # 1) Copy to datasets/
        dst = prj.root / "datasets" / src.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(src, dst)

        # 2) Register as RAW artifact (dedupe by hash)
        from plant_hsi.core.artifact import compute_file_hash

        raw_hash = compute_file_hash(dst)
        existing = next(
            (a for a in prj.list_artifacts() if a.kind == ArtifactKind.RAW and a.hash == raw_hash),
            None,
        )
        if existing is None:
            existing = ArtifactRef(
                id="",
                kind=ArtifactKind.RAW,
                name=src.name,
                uri=f"datasets/{src.name}",
                hash=raw_hash,
                size=dst.stat().st_size,
            )
            prj.add_artifact(existing)

        # 3) Run import_dataset step
        result = prj.run_step("io.import_dataset", [existing])
        if result.skipped:
            return 0
        if not result.ok:
            raise RuntimeError(result.run.error)
        return 1

    def _open_selected_dataset(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            QMessageBox.information(self, "Open", "Select a dataset artifact first.")
            return
        self._open_dataset_for_item(items[0])

    def _open_dataset_for_item(self, item: QTreeWidgetItem) -> None:
        if self._project is None:
            return
        # Walk upward until we find a QTreeWidgetItem that holds an artifact ID
        # (items carry the ID in their `data(Qt.UserRole, 0)` when populated from DB).
        art_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not art_id:
            # Try the next ancestor (resource-tree items have IDs attached to artifact rows)
            parent = item.parent()
            while parent is not None:
                art_id = parent.data(0, Qt.ItemDataRole.UserRole)
                if art_id:
                    break
                parent = parent.parent()
        if not art_id:
            QMessageBox.information(self, "Open", "Not a dataset row.")
            return

        if art_id in self._open_views:
            self.tabs.setCurrentWidget(self._open_views[art_id])
            return
        art = self._project.get_artifact(art_id)
        if art is None:
            return
        cube, wl = self._load_cube_for(art)
        if cube is None:
            QMessageBox.warning(
                self, "Open", f"Can't load Zarr at {art.resolve(self._project.root)}"
            )
            return
        view = HsiImageView(cube, wavelength_nm=wl)
        self._open_views[art_id] = view
        self._view_wl[art_id] = wl
        self.tabs.addTab(view, art.name)
        self.tabs.setCurrentWidget(view)
        self._bind_panel(view)
        self.statusBar().showMessage(
            f"Viewing {art.name}  ({art.shape if hasattr(art, 'shape') else cube.shape})"
        )

    def _on_tab_changed(self, _: int) -> None:
        self._bind_panel(self.tabs.currentWidget())

    def _bind_panel(self, widget: QWidget | None) -> None:
        """Point the per-image panels at whichever view is in front."""
        if self._project is None or widget is None:
            self.seg_panel.unbind()
            if self.dl_panel is not None:
                self.dl_panel.unbind()
            return
        for aid, view in self._open_views.items():
            if view is widget:
                art = self._project.get_artifact(aid)
                if art is not None:
                    self.seg_panel.bind(self._project, art, view, self._view_wl.get(aid))
                    if self.dl_panel is not None:
                        self.dl_panel.bind(self._project, art)
                return
        self.seg_panel.unbind()
        if self.dl_panel is not None:
            self.dl_panel.unbind()

    def _close_tab(self, idx: int) -> None:
        w = self.tabs.widget(idx)
        if w is None:
            return
        # Find matching artifact and forget it
        for aid, view in list(self._open_views.items()):
            if view is w:
                del self._open_views[aid]
                self._view_wl.pop(aid, None)
                break
        self.tabs.removeTab(idx)
        w.setParent(None)
        w.deleteLater()
        self._bind_panel(self.tabs.currentWidget())

    # ---------------------------------------------------------- helpers
    def _reset_views(self) -> None:
        """Drop every open viewer (used when the project changes under us)."""
        self.seg_panel.unbind()
        self.preprocess_panel.unbind()
        self.model_panel.unbind()
        if self.dl_panel is not None:
            self.dl_panel.unbind()
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._open_views.clear()
        self._view_wl.clear()

    def _refresh_resource_tree(self) -> None:
        self.tree.clear()
        if self._project is None:
            root = QTreeWidgetItem(self.tree, ["(no project open)", "", ""])
            root.setFlags(root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            return
        prj = self._project
        project_item = QTreeWidgetItem(
            self.tree, [prj.meta.name, prj.meta.project_id, str(prj.root)]
        )
        project_item.setFlags(project_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)

        datasets_item = QTreeWidgetItem(project_item, ["Datasets", "", ""])
        masks_item = QTreeWidgetItem(project_item, ["Masks & ROIs", "", ""])
        runs_item = QTreeWidgetItem(project_item, ["Runs", "", ""])

        for row in prj.list_artifacts():
            if row.kind == ArtifactKind.CUBE:
                parent = datasets_item
            elif row.kind in (ArtifactKind.MASK, ArtifactKind.SPECTRA):
                parent = masks_item
            else:
                parent = project_item
            item = QTreeWidgetItem(parent, [row.name, row.kind.value, _fmt_size(row.size)])
            item.setData(0, Qt.ItemDataRole.UserRole, row.id)

        for run in prj.list_runs():
            QTreeWidgetItem(runs_item, [run.id, run.step, run.status.value])

        self.tree.expandAll()

    def _load_cube_for(self, art: ArtifactRef) -> tuple[np.ndarray | None, list[float] | None]:
        """Load Zarr CUBE artifact and return ``(cube, wavelength)``."""
        assert self._project is not None
        import zarr

        path = art.resolve(self._project.root)
        if art.kind != ArtifactKind.CUBE or not path.exists():
            return None, None
        arr = zarr.open_array(str(path), mode="r")
        cube = np.asarray(arr[:]).astype(np.float32)
        wl: list[float] | None = None
        meta_path = path / ".phs_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            wl = [float(v) for v in meta.get("wavelength_nm", [])]
        return cube, wl


def _fmt_size(n: int) -> str:
    if n <= 0:
        return ""
    x: float = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"
