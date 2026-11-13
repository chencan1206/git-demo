"""Capture Stage-6 walkthrough screenshots (headless, offscreen Qt).

Outputs /workspace/reports/stage6/{main_window,segmentation,preprocess,modeling,dl}.png
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef  # noqa: E402
from plant_hsi.core.project import Project  # noqa: E402
from plant_hsi.io.spec import DatasetSpec, write_zarr  # noqa: E402
from plant_hsi.ui.main_window import MainWindow  # noqa: E402

OUT = Path("/workspace/reports/stage6")
TMP = Path("/tmp/phs_stage6_capture")


def build_project() -> tuple[Project, ArtifactRef]:
    if TMP.exists():
        shutil.rmtree(TMP)
    prj = Project.create(TMP / "demo-project", "番茄叶片 · 病害分型")
    rng = np.random.default_rng(7)
    h, w, b = 120, 160, 64
    yy, xx = np.mgrid[0:h, 0:w]
    leaf = np.exp(-(((yy - h / 2) / (h / 3)) ** 2 + ((xx - w / 2) / (w / 2.6)) ** 2))
    cube = (0.15 + 0.55 * leaf[..., None] * (0.6 + 0.4 * rng.random((h, w, b)))).astype(np.float32)
    wl = list(np.linspace(400, 950, b))
    write_zarr(
        cube,
        prj.root / "artifacts" / "tomato_day1.zarr",
        DatasetSpec(
            source_path="synthetic://tomato", format="mat", shape=cube.shape, wavelength_nm=wl
        ),
    )
    art = prj.add_artifact(
        ArtifactRef(
            id="",
            kind=ArtifactKind.CUBE,
            name="tomato_day1",
            uri="artifacts/tomato_day1.zarr",
            metadata={"shape": list(cube.shape)},
        )
    )
    return prj, art


def grab(widget, name: str) -> None:
    widget.grab().save(str(OUT / f"{name}.png"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])

    prj, art = build_project()
    win = MainWindow()
    win._project = prj
    win._refresh_resource_tree()

    item = win.tree.topLevelItem(0).child(0).child(0)
    win._open_dataset_for_item(item)

    spectra = (
        0.2
        + 0.5 * np.exp(-((np.linspace(0, 1, art.metadata["shape"][2]) - 0.45) ** 2) / 0.05)
        + 0.01 * np.random.default_rng(3).random((8, art.metadata["shape"][2]))
    ).astype(np.float32)
    wl = list(np.linspace(400, 950, spectra.shape[1]))
    win.preprocess_panel.bind(spectra, wl)
    win.model_panel.bind(prj, spectra, wl)
    if win.dl_panel is not None:
        win.dl_panel.bind(prj, art)

    win.resize(1500, 950)
    win.show()
    app.processEvents()

    grab(win, "main_window")
    for panel, name in (
        (win.seg_panel, "segmentation"),
        (win.preprocess_panel, "preprocess"),
        (win.model_panel, "modeling"),
    ):
        panel.setMinimumSize(430, panel.sizeHint().height())
        panel.show()
        app.processEvents()
        grab(panel, name)
    if win.dl_panel is not None:
        win.dl_panel.show()
        app.processEvents()
        grab(win.dl_panel, "dl")

    win.close()
    prj.close()
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"wrote {len(list(OUT.glob('*.png')))} screenshots to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
