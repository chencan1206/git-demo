"""Create a small synthetic demo project with two leaf classes.

Useful for trying the GUI / modeling panels without downloading real data::

    python scripts/make_demo_project.py --out /workspace/phs_demo

The generated project contains four cubes (two per class), runs the same
``Step`` pipeline as the desktop app, and writes ``demo_report.html``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.project import Project
from plant_hsi.core.run import RunStatus

H, W, B = 96, 128, 64
WAVELENGTHS = np.linspace(450.0, 850.0, B).tolist()
CLASSES = ["Healthy", "Alternaria"]
LEAVES = [  # (cy, cx, ry, rx)
    (34, 42, 22, 30),
    (62, 92, 20, 26),
]


def leaf_spectrum(cls: str) -> np.ndarray:
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


def synth_cube(cls: str, variant: int, rng: np.random.Generator) -> np.ndarray:
    cube = np.full((H, W, B), 0.15, dtype=np.float32)
    cube += rng.normal(0, 0.004, size=cube.shape).astype(np.float32)
    base = leaf_spectrum(cls)
    yy, xx = np.mgrid[0:H, 0:W]
    for k, (cy, cx, ry, rx) in enumerate(LEAVES):
        cy = int(np.clip(cy + (variant - 0.5) * 6 + k * 2, ry + 1, H - ry - 1))
        cx = int(np.clip(cx + (variant - 0.5) * 10, rx + 1, W - rx - 1))
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        gain = 0.95 + 0.10 * rng.random()
        cube[mask] = (base * gain + rng.normal(0, 0.006, size=(int(mask.sum()), B))).astype(
            np.float32
        )
    return cube


def run(prj: Project, step: str, inputs: list[ArtifactRef], params: dict | None = None, seed=None):
    result = prj.run_step(step, inputs, params=params, seed=seed)
    if result.run.status != RunStatus.SUCCEEDED:
        raise RuntimeError(f"{step} failed: {result.run.error}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/workspace/phs_demo"))
    parser.add_argument("--force", action="store_true", help="overwrite an existing demo")
    args = parser.parse_args()

    base: Path = args.out.resolve()
    project_dir = base / "demo-project"
    if project_dir.exists():
        if not args.force:
            raise SystemExit(f"{project_dir} exists; pass --force to overwrite")
        shutil.rmtree(project_dir)
    raw_dir = base / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    prj = Project.create(project_dir, "PHS Studio 合成演示 · 健康/病害二分类")
    rng = np.random.default_rng(7)
    frames: list[pd.DataFrame] = []
    parents: list[str] = []

    for cls in CLASSES:
        ref = leaf_spectrum(cls)
        for variant in range(2):
            name = f"{cls.lower()}_{variant + 1}"
            tif = raw_dir / f"{name}.tif"
            tifffile.imwrite(tif, synth_cube(cls, variant, rng), photometric="minisblack")
            raw = prj.add_artifact(
                ArtifactRef(id="", kind=ArtifactKind.RAW, name=name, uri=str(tif))
            )
            imported = run(prj, "io.import_dataset", [raw], {"wavelength_nm": WAVELENGTHS})
            cube = next(a for a in imported.outputs if a.kind == ArtifactKind.CUBE)
            mask = next(
                a
                for a in run(
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
                ).outputs
                if a.kind == ArtifactKind.MASK
            )
            cc = run(
                prj,
                "segment.connected_components",
                [mask, cube],
                {"min_area": 40, "connectivity": 8, "sort_by": "area_desc"},
            )
            table = next(a for a in cc.outputs if a.kind == ArtifactKind.SPECTRA)
            df = pd.read_csv(table.resolve(prj.root), encoding="utf-8-sig")
            df["class"] = cls
            df["sample_id"] = [f"{name}_{int(v)}" for v in df["label"]]
            frames.append(df.drop(columns=["label"]))
            parents.append(table.id)
            print(f"  {name}: {len(df)} ROIs")

    combined = pd.concat(frames, ignore_index=True)
    csv_path = prj.root / "artifacts" / "demo_spectra.csv"
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    spectra = prj.add_artifact(
        ArtifactRef(
            id="",
            kind=ArtifactKind.SPECTRA,
            name="demo_spectra",
            uri=str(csv_path.relative_to(prj.root)).replace("\\", "/"),
            parents=parents,
            metadata={"n_samples": int(len(combined)), "classes": CLASSES},
        )
    )

    trained = run(
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
            "preprocess_chain": {
                "version": 1,
                "ops": [
                    {"kind": "snv", "params": {}},
                    {"kind": "sg", "params": {"window": 11, "polyorder": 2, "deriv": 1}},
                ],
            },
        },
        seed=7,
    )
    model = next(a for a in trained.outputs if a.kind == ArtifactKind.MODEL)
    report = next(
        a
        for a in run(prj, "report.build", [model], {"title": "PHS Studio 合成演示项目"}).outputs
        if a.kind == ArtifactKind.REPORT
    )
    report_copy = base / "demo_report.html"
    shutil.copy2(report.resolve(prj.root), report_copy)
    prj.close()

    print(f"project : {project_dir}")
    print(f"samples : {len(combined)} ROIs, classes={CLASSES}")
    print(f"report  : {report_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
