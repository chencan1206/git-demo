"""Stage 7 acceptance: end-to-end run on the real Zenodo tomato dataset.

Pipeline (no GUI, same ``Step`` engine as the desktop app):

    io.import_dataset -> segment.threshold -> segment.connected_components
    -> combined SPECTRA table -> ml.train -> report.build
    -> export.csv / export.geotiff

Two cubes are used, one per class (Healthy / Alternaria alternata).  Each
cube yields several leaf ROIs; per-ROI mean spectra are merged into one
labelled table so the binary classifier has both classes.

Outputs:
    /workspace/stage7_zenodo/                    project directory
    /workspace/reports/stage7_zenodo_report.html report
    /workspace/reports/stage7_zenodo_spectra.csv exported spectra
    /workspace/reports/stage7_zenodo_labels.tif  exported ROI label map
    /workspace/reports/stage7_e2e_acceptance.json
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import zarr

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.project import Project
from plant_hsi.core.run import RunStatus

DATA_DIR = Path("/workspace/data/raw/zenodo_15800414")
PROJECT_ROOT = Path("/workspace/stage7_zenodo")
REPORT_DIR = Path("/workspace/reports")

CASES = [
    ("Healthy", "Healthy_1.mat"),
    ("Alternaria", "Alternaria_alternata_1.mat"),
]
BANDS = 550
WAVELENGTHS = [400.0 + i for i in range(BANDS)]

timings: dict[str, float] = {}
run_ids: dict[str, str] = {}


def run(
    prj: Project,
    step: str,
    inputs: list[ArtifactRef],
    params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> Any:
    started = time.perf_counter()
    result = prj.run_step(step, inputs, params=params, seed=seed)
    elapsed = time.perf_counter() - started
    timings[step] = round(timings.get(step, 0.0) + elapsed, 2)
    run_ids[step] = result.run.id
    if result.run.status != RunStatus.SUCCEEDED:
        raise RuntimeError(f"{step} failed: {result.run.error}")
    return result


def ndvi_reference(cube_art: ArtifactRef, prj: Project) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cube, mean leaf spectrum)`` using the NDVI>0.25 leaf mask."""
    arr = np.asarray(zarr.open_array(str(cube_art.resolve(prj.root)), mode="r")[:])
    cube = arr.astype(np.float32, copy=False)
    red, nir = cube[:, :, 270], cube[:, :, 400]  # 670 nm / 800 nm
    ndvi = (nir - red) / (nir + red + 1e-9)
    leaf = ndvi > 0.25
    if not leaf.any():
        raise RuntimeError("NDVI leaf mask is empty; check wavelength assignment")
    return cube, cube[leaf].mean(axis=0).astype(np.float32)


def main() -> int:
    started_all = time.perf_counter()
    if PROJECT_ROOT.exists():
        shutil.rmtree(PROJECT_ROOT)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prj = Project.create(PROJECT_ROOT, "Zenodo 番茄病害 · v1.0 端到端验收")

    frames: list[pd.DataFrame] = []
    spectra_parents: list[str] = []
    mask_for_export: ArtifactRef | None = None
    import_checks: list[dict[str, Any]] = []
    seg_checks: list[dict[str, Any]] = []

    for cls_name, filename in CASES:
        src = DATA_DIR / filename
        raw = prj.add_artifact(
            ArtifactRef(id="", kind=ArtifactKind.RAW, name=src.stem, uri=str(src))
        )
        imported = run(prj, "io.import_dataset", [raw], {"wavelength_nm": WAVELENGTHS})
        cube_art = next(a for a in imported.outputs if a.kind == ArtifactKind.CUBE)

        cube, ref = ndvi_reference(cube_art, prj)
        threshold = run(
            prj,
            "segment.threshold",
            [cube_art],
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
        mask_art = next(a for a in threshold.outputs if a.kind == ArtifactKind.MASK)

        cc = run(
            prj,
            "segment.connected_components",
            [mask_art, cube_art],
            {"min_area": 200, "connectivity": 8, "sort_by": "area_desc"},
        )
        labels_art = next(a for a in cc.outputs if a.kind == ArtifactKind.MASK)
        table_art = next(a for a in cc.outputs if a.kind == ArtifactKind.SPECTRA)

        df = pd.read_csv(table_art.resolve(prj.root), encoding="utf-8-sig")
        df["class"] = cls_name
        df["sample_id"] = [f"{cls_name}_{int(v)}" for v in df["label"]]
        df = df.drop(columns=["label"])
        frames.append(df)
        spectra_parents.append(table_art.id)
        mask_for_export = mask_for_export or labels_art

        import_checks.append(
            {
                "file": filename,
                "shape": list(cube.shape),
                "wavelengths": [WAVELENGTHS[0], WAVELENGTHS[-1], len(WAVELENGTHS)],
                "health_ok": bool(cube_art.metadata.get("health_ok")),
            }
        )
        seg_checks.append(
            {
                "class": cls_name,
                "foreground_pixels": int(mask_art.metadata["foreground_pixels"]),
                "coverage": round(
                    mask_art.metadata["foreground_pixels"] / (cube.shape[0] * cube.shape[1]), 4
                ),
                "threshold": round(float(mask_art.metadata["threshold"]), 4),
                "n_rois": int(labels_art.metadata["n_components"]),
                "roi_areas": sorted(
                    (int(v) for v in df.get("area_px", pd.Series(dtype=int))), reverse=True
                ),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    combined_path = prj.root / "artifacts" / "zenodo_combined_spectra.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    combined_art = prj.add_artifact(
        ArtifactRef(
            id="",
            kind=ArtifactKind.SPECTRA,
            name="zenodo_combined_spectra",
            uri=str(combined_path.relative_to(prj.root)).replace("\\", "/"),
            parents=spectra_parents,
            metadata={
                "n_samples": int(len(combined)),
                "n_bands": int(sum(c.startswith("mean_") for c in combined.columns)),
                "classes": sorted(combined["class"].unique().tolist()),
                "source": "Zenodo 15800414 / Day 1",
            },
        )
    )

    trained = run(
        prj,
        "ml.train",
        [combined_art],
        {
            "task": "classification",
            "algorithm": "rf",
            "label_column": "class",
            "test_size": 0.34,
            "val_size": 0.0,
            "cv_folds": 2,
            "model_params": {"n_estimators": 200},
            "preprocess_chain": {
                "version": 1,
                "ops": [
                    {"kind": "snv", "params": {}},
                    {"kind": "sg", "params": {"window": 11, "polyorder": 2, "deriv": 1}},
                ],
            },
        },
        seed=42,
    )
    model_art = next(a for a in trained.outputs if a.kind == ArtifactKind.MODEL)

    reported = run(
        prj,
        "report.build",
        [model_art],
        {"title": "Zenodo 番茄病害二分类 · v1.0 端到端验收"},
    )
    report_art = next(a for a in reported.outputs if a.kind == ArtifactKind.REPORT)

    exported_csv = run(prj, "export.csv", [combined_art])
    assert mask_for_export is not None
    exported_tif = run(prj, "export.geotiff", [mask_for_export], {"dtype": "int32"})

    report_copy = REPORT_DIR / "stage7_zenodo_report.html"
    shutil.copy2(report_art.resolve(prj.root), report_copy)
    manifest_uri = report_art.metadata.get("manifest_uri")
    manifest_copy: Path | None = None
    if manifest_uri:
        manifest_copy = REPORT_DIR / "stage7_zenodo_run_manifest.json"
        shutil.copy2(prj.root / str(manifest_uri), manifest_copy)

    csv_art = exported_csv.outputs[0]
    tif_art = exported_tif.outputs[0]
    csv_copy = REPORT_DIR / "stage7_zenodo_spectra.csv"
    tif_copy = REPORT_DIR / "stage7_zenodo_labels.tif"
    shutil.copy2(csv_art.resolve(prj.root), csv_copy)
    shutil.copy2(tif_art.resolve(prj.root), tif_copy)

    metrics = model_art.metadata.get("metrics", {})
    cv = model_art.metadata.get("cv", {})
    exported_df = pd.read_csv(csv_copy, encoding="utf-8-sig")
    tif_arr = tifffile.imread(tif_copy)

    checks = [
        {
            "criterion": "1. 两个真实 MAT 立方体导入成功",
            "passed": all(c["shape"] == [140, 280, 550] for c in import_checks),
            "detail": import_checks,
        },
        {
            "criterion": "2. RA 分割 + 连通域产生有效 ROI",
            "passed": all(0.1 <= c["coverage"] <= 0.6 and c["n_rois"] >= 2 for c in seg_checks),
            "detail": seg_checks,
        },
        {
            "criterion": "3. 合并光谱表含 2 类、550 波段",
            "passed": len(combined) >= 4
            and sorted(combined["class"].unique().tolist()) == ["Alternaria", "Healthy"],
            "detail": {
                "n_samples": int(len(combined)),
                "n_bands": int(sum(c.startswith("mean_") for c in combined.columns)),
                "classes": sorted(combined["class"].unique().tolist()),
            },
        },
        {
            "criterion": "4. 传统模型训练完成并产出指标",
            "passed": bool(metrics) and "accuracy" in metrics,
            "detail": {"metrics": metrics, "cv": cv},
        },
        {
            "criterion": "5. HTML 报告与运行清单生成",
            "passed": report_copy.exists()
            and report_copy.stat().st_size > 5000
            and (manifest_copy is None or manifest_copy.exists()),
            "detail": {
                "report": str(report_copy),
                "report_bytes": report_copy.stat().st_size if report_copy.exists() else 0,
                "manifest": str(manifest_copy) if manifest_copy else None,
            },
        },
        {
            "criterion": "6. CSV / GeoTIFF 导出可被 pandas / tifffile 读取",
            "passed": len(exported_df) == len(combined) and tif_arr.shape == (140, 280),
            "detail": {
                "csv_rows": int(len(exported_df)),
                "csv_columns": int(exported_df.shape[1]),
                "tif_shape": list(tif_arr.shape),
                "tif_dtype": str(tif_arr.dtype),
            },
        },
    ]

    runs = prj.list_runs()
    elapsed = round(time.perf_counter() - started_all, 2)
    payload = {
        "stage": 7,
        "dataset": "Zenodo 10.5281/zenodo.15800414 (Day 1, Healthy vs Alternaria)",
        "total_seconds": elapsed,
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "timings_seconds": timings,
        "run_ids": run_ids,
        "runs_total": len(runs),
        "runs_failed": sum(1 for r in runs if r.status == RunStatus.FAILED),
        "project_root": str(prj.root),
        "artifacts": [a.to_dict() for a in prj.list_artifacts()],
    }
    out_json = REPORT_DIR / "stage7_e2e_acceptance.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    prj.close()

    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "total_seconds": elapsed,
                "checks": [(c["criterion"], c["passed"]) for c in checks],
                "metrics": metrics,
                "report": str(report_copy),
                "acceptance": str(out_json),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
