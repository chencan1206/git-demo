"""End-to-end self test for the frozen desktop build.

Builds a tiny synthetic two-class dataset in a temp folder and drives the same
``Step`` pipeline the GUI uses: import -> spectral-angle threshold ->
connected components -> random forest -> HTML report. Running this from the
packaged executable proves that the bundled scientific stack (numpy, scipy,
sklearn, matplotlib, zarr, tifffile) actually works on the target OS, not just
that the Qt window can be constructed.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.project import Project

H, W, B = 64, 96, 48
WAVELENGTHS = np.linspace(450.0, 850.0, B).tolist()
CLASSES = ["Healthy", "Alternaria"]
LEAVES = [(22, 26, 10, 14), (44, 64, 9, 12)]


def _spectrum(cls: str) -> np.ndarray:
    wl = np.asarray(WAVELENGTHS)
    spec = (
        0.16
        + 0.09 * np.exp(-(((wl - 550) / 55) ** 2))
        + 0.20 / (1 + np.exp(-(wl - 700) / 18))
        - 0.06 * np.exp(-(((wl - 675) / 22) ** 2))
    )
    if cls == "Alternaria":
        spec = 0.88 * spec + 0.035 * np.exp(-(((wl - 680) / 30) ** 2))
    return spec.astype(np.float32)


def _cube(cls: str, variant: int, rng: np.random.Generator) -> np.ndarray:
    cube = (0.15 + rng.normal(0, 0.004, size=(H, W, B))).astype(np.float32)
    base = _spectrum(cls)
    yy, xx = np.mgrid[0:H, 0:W]
    for k, (cy, cx, ry, rx) in enumerate(LEAVES):
        cy = int(np.clip(cy + (variant - 1) * 3 + k, ry + 1, H - ry - 1))
        cx = int(np.clip(cx + (variant - 1) * 4, rx + 1, W - rx - 1))
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        cube[mask] = (
            base * (0.95 + 0.10 * rng.random())
            + rng.normal(0, 0.006, size=(int(mask.sum()), B))
        ).astype(np.float32)
    return cube


def _run(prj: Project, step: str, inputs: list[ArtifactRef], params: dict | None = None, seed=None):
    result = prj.run_step(step, inputs, params=params, seed=seed)
    if not result.ok:
        raise RuntimeError(f"{step} failed: {result.run.error}")
    return result


def run(workdir: Path | None = None) -> dict:
    """Execute the pipeline and return a summary dict. Raises on any failure."""
    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="phs_e2e_"))
    project_dir = base / "e2e-project"
    raw_dir = base / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    prj = Project.create(project_dir, "PHS Studio self test")
    rng = np.random.default_rng(11)
    frames: list[pd.DataFrame] = []
    parents: list[str] = []

    try:
        for cls in CLASSES:
            ref = _spectrum(cls)
            for variant in (0, 1):
                name = f"{cls.lower()}_{variant + 1}"
                tif = raw_dir / f"{name}.tif"
                tifffile.imwrite(tif, _cube(cls, variant, rng), photometric="minisblack")
                raw = prj.add_artifact(
                    ArtifactRef(id="", kind=ArtifactKind.RAW, name=name, uri=str(tif))
                )
                imported = _run(prj, "io.import_dataset", [raw], {"wavelength_nm": WAVELENGTHS})
                cube = next(a for a in imported.outputs if a.kind == ArtifactKind.CUBE)
                seg = _run(
                    prj,
                    "segment.threshold",
                    [cube],
                    {
                        "method": "ra",
                        "threshold_method": "otsu",
                        "ref_source": "spectra",
                        "ref_spectrum": ref.tolist(),
                        "morph_kernel": 3,
                        "morph_open_iter": 1,
                        "morph_close_iter": 2,
                    },
                )
                mask = next(a for a in seg.outputs if a.kind == ArtifactKind.MASK)
                cc = _run(
                    prj,
                    "segment.connected_components",
                    [mask, cube],
                    {"min_area": 40, "connectivity": 8, "sort_by": "area_desc"},
                )
                table = next(a for a in cc.outputs if a.kind == ArtifactKind.SPECTRA)
                df = pd.read_csv(table.resolve(prj.root), encoding="utf-8-sig")
                if df.empty:
                    raise RuntimeError(f"{name}: segmentation produced no ROIs")
                df["class"] = cls
                df["sample_id"] = [f"{name}_{int(v)}" for v in df["label"]]
                frames.append(df.drop(columns=["label"]))
                parents.append(table.id)

        combined = pd.concat(frames, ignore_index=True)
        csv_path = prj.root / "artifacts" / "e2e_spectra.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
        spectra = prj.add_artifact(
            ArtifactRef(
                id="",
                kind=ArtifactKind.SPECTRA,
                name="e2e_spectra",
                uri=str(csv_path.relative_to(prj.root)).replace("\\", "/"),
                parents=parents,
                metadata={"n_samples": int(len(combined)), "classes": CLASSES},
            )
        )
        trained = _run(
            prj,
            "ml.train",
            [spectra],
            {
                "task": "classification",
                "algorithm": "rf",
                "label_column": "class",
                "test_size": 0.34,
                "val_size": 0.0,
                "cv_folds": 2,
                "model_params": {"n_estimators": 60},
                "preprocess_chain": {
                    "version": 1,
                    "ops": [{"kind": "snv", "params": {}}],
                },
            },
            seed=11,
        )
        model = next(a for a in trained.outputs if a.kind == ArtifactKind.MODEL)
        report = _run(prj, "report.build", [model], {"title": "PHS Studio self test"})
        html = next(a for a in report.outputs if a.kind == ArtifactKind.REPORT)
        report_path = html.resolve(prj.root)
        if not report_path.exists() or report_path.stat().st_size < 1000:
            raise RuntimeError("report was not written")
        return {
            "project": str(project_dir),
            "samples": int(len(combined)),
            "report": str(report_path),
            "report_bytes": report_path.stat().st_size,
        }
    finally:
        prj.close()
