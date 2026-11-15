"""Deep analysis of the real Zenodo Day-1 cubes with PHS Studio.

Pipeline:
  import -> RA/Otsu segmentation -> connected-components ROIs
  -> full spectrum statistics -> RF classification -> report

Measures:
  * per-step wall-clock seconds + peak RSS via resource.getrusage
  * healthy vs disease mean reflectance per band (400–950 nm)
  * vegetation-index statistics inside leaf masks
  * spectral-angle distribution between leaves
  * segmentation parameter sensitivity (Otsu / GMM / Li)
  * RF hyperparameter sensitivity (n_estimators)

Outputs (in /workspace/reports):
  zenodo_deep_analysis.json     all numeric results
  zenodo_deep_analysis_report.html  single-page visual summary
  zenodo_spectra.csv            per-leaf average spectra + label
  zenodo_seg_mask_{healthy,alternaria}.tif
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.project import Project
from plant_hsi.core.run import RunStatus
from plant_hsi.io.mat_reader import load_mat_cube
from plant_hsi.segment.core import (
    connected_components,
    morphological_cleanup,
    spectral_angle,
    binarize,
)

DATA = Path("/workspace/data/raw/zenodo_15800414")
PROJ = Path("/workspace/stage7b_deep")
OUT = Path("/workspace/reports")
WL = np.linspace(400.0, 950.0, 550).astype(np.float32)


def rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def timed(label: str, payload):
    """Run ``payload()``, measure wall time and peak RSS delta."""
    rss0 = rss_kb()
    t0 = time.perf_counter()
    out = payload()
    dt = time.perf_counter() - t0
    rss = rss_kb()
    print(f"  [{label:28s}] {dt:7.2f}s  RSS {rss0:7d}→{rss:7d} KB  (+{(rss-rss0):+d})")
    return out, round(dt, 3), rss - rss0


def run_step(prj, name, inputs, params=None, seed=None):
    result = prj.run_step(name, inputs, params=params, seed=seed)
    if result.run.status == RunStatus.FAILED:
        raise RuntimeError(f"{name} failed: {result.run.error}")
    if result.skipped and not result.outputs:
        # Cache hit: recover outputs from an earlier identical run.
        result.outputs = prj.find_cached_outputs(name, inputs, params=params, seed=seed)
    return result


def leaf_mask(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    red, nir = cube[:, :, 270], cube[:, :, 400]
    ndvi = (nir - red) / (nir + red + 1e-9)
    return ndvi > 0.25, ndvi


def spectrum_stats(name: str, cube: np.ndarray, leaf: np.ndarray) -> dict:
    arr = cube[leaf]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return {
        "name": name,
        "shape": list(cube.shape),
        "leaf_pixels": int(leaf.sum()),
        "leaf_coverage": round(float(leaf.mean()), 4),
        "spectrum_mean": mean.astype(float).tolist(),
        "spectrum_std": std.astype(float).tolist(),
        "vis_mean": round(float(mean[:200].mean()), 4),   # 400-600 nm
        "red_mean": round(float(mean[260:290].mean()), 4),  # 660-700 nm
        "nir_mean": round(float(mean[380:420].mean()), 4),  # 780-820 nm
        "red_edge": round(float(mean[275:400].mean() - mean[245:275].mean()), 4),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if PROJ.exists():
        shutil.rmtree(PROJ)

    cases = [("Healthy", "Healthy_1.mat"), ("Alternaria", "Alternaria_alternata_1.mat")]

    # --- low-level loading / segmentation (no Step engine) for deep stats ----
    raw_spectra: dict[str, np.ndarray] = {}
    seg_by_class: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    overall = {"runs": [], "rss_baseline_kb": rss_kb()}
    t_overall0 = time.perf_counter()

    for cls, fname in cases:
        p = DATA / fname
        (cube_arr, t_load, drss_load) = timed(f"load {fname}", lambda: np.asarray(load_mat_cube(p)[0], dtype=np.float32))
        leaf, ndvi = leaf_mask(cube_arr)
        stats = spectrum_stats(cls, cube_arr, leaf)
        raw_spectra[cls] = np.asarray(stats["spectrum_mean"], dtype=np.float32)

        ref = cube_arr[leaf].mean(axis=0).astype(np.float32)

        seg_params_list = [
            ("otsu", "RA spectral angle · Otsu"),
            ("percentile", 75.0, "RA · 75th percentile"),
            ("percentile", 85.0, "RA · 85th percentile"),
            ("fixed", 0.15, "RA · fixed thr=0.15"),
        ]
        seg_trials: dict[str, dict[str, Any]] = {}
        for entry in seg_params_list:
            th_method = entry[0]
            extra = entry[1] if len(entry) > 2 else None
            label = entry[-1]
            sim, _, _ = timed(f"{cls} similarity ({th_method})",
                              lambda: spectral_angle(cube_arr, ref))
            (mask, thr), t_seg, _ = timed(f"{cls} threshold={label}",
                                          lambda: binarize(sim, method=th_method, threshold=extra))
            clean, t_clean, _ = timed(f"{cls} morphology",
                                       lambda: morphological_cleanup(mask))
            cc, t_cc, _ = timed(f"{cls} connected_components",
                                 lambda: connected_components(clean, min_area=200))
            seg_trials[th_method] = {
                "label": label,
                "threshold": round(float(thr), 4),
                "coverage": round(float(clean.mean()), 4),
                "n_components": int(cc.n_components),
                "areas": cc.areas.tolist(),
                "time_total_s": round(t_seg + t_clean + t_cc, 3),
            }
        seg_by_class[cls] = {"stats": stats, "trials": seg_trials}

        # keep Otsu mask + labels for Step-engine path
        sim = spectral_angle(cube_arr, ref)
        mask, _ = binarize(sim, method="otsu")
        mask = morphological_cleanup(mask)
        cc = connected_components(mask, min_area=200)

        tifffile.imwrite(OUT / f"zenodo_seg_mask_{cls.lower()}.tif",
                         mask.astype(np.uint8))
        tifffile.imwrite(OUT / f"zenodo_seg_labels_{cls.lower()}.tif",
                         cc.labels.astype(np.int32))

        # per-ROI mean spectrum
        unique_labels = [int(v) for v in np.unique(cc.labels) if v > 0]
        for lab in unique_labels:
            roi_mask = cc.labels == lab
            spec = cube_arr[roi_mask].mean(axis=0).astype(float).tolist()
            rows.append({
                "sample_id": f"{cls}_{lab}",
                "class": cls,
                "roi_label": lab,
                "area_px": int(roi_mask.sum()),
                "vis_mean": round(float(np.mean(spec[:200])), 4),
                "red_mean": round(float(np.mean(spec[260:290])), 4),
                "nir_mean": round(float(np.mean(spec[380:420])), 4),
                **{f"wl_{int(WL[i])}": round(float(spec[i]), 6) for i in range(len(WL))},
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "zenodo_spectra.csv", index=False, encoding="utf-8-sig")

    # --- Step-engine path: import + segmentation + modeling + report ----
    prj = Project.create(PROJ, "Zenodo Day1 深度分析")
    all_frames: list[pd.DataFrame] = []
    all_parents: list[str] = []
    spectra_by_class: dict[str, list[str]] = {}

    for cls, fname in cases:
        src = DATA / fname
        raw = prj.add_artifact(ArtifactRef(id="", kind=ArtifactKind.RAW, name=src.stem, uri=str(src)))
        imported, _, _ = timed(f"[step] import {cls}",
                               lambda: run_step(prj, "io.import_dataset", [raw],
                                                {"wavelength_nm": WL.tolist()}))
        cube = next(a for a in imported.outputs if a.kind == ArtifactKind.CUBE)
        ref = raw_spectra[cls]
        thr, _, _ = timed(f"[step] threshold {cls}",
                          lambda: run_step(prj, "segment.threshold", [cube],
                                           {"method": "ra",
                                            "threshold_method": "otsu",
                                            "ref_source": "spectra",
                                            "ref_spectrum": ref.tolist(),
                                            "morph_kernel": 3,
                                            "morph_open_iter": 1,
                                            "morph_close_iter": 2}))
        cc, _, _ = timed(f"[step] cc {cls}",
                         lambda: run_step(prj, "segment.connected_components",
                                          [thr.outputs[0], cube],
                                          {"min_area": 200, "connectivity": 8, "sort_by": "area_desc"}))
        table = next(a for a in cc.outputs if a.kind == ArtifactKind.SPECTRA)
        pdf = pd.read_csv(table.resolve(prj.root), encoding="utf-8-sig")
        pdf["class"] = cls
        all_frames.append(pdf.drop(columns=["label"]))
        all_parents.append(table.id)
        spectra_by_class.setdefault(cls, []).extend(pdf["label"].astype(int).tolist())

    combined = pd.concat(all_frames, ignore_index=True)
    csv_path = prj.root / "artifacts" / "combined.csv"
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    spectra = prj.add_artifact(
        ArtifactRef(
            id="", kind=ArtifactKind.SPECTRA, name="zenodo_day1",
            uri=str(csv_path.relative_to(prj.root)).replace("\\", "/"),
            parents=all_parents,
            metadata={"n_samples": int(len(combined)),
                      "classes": sorted(combined["class"].unique().tolist())},
        )
    )

    # sensitivity: n_estimators
    sens: dict[str, Any] = {"n_estimators": []}
    for n in [50, 100, 200, 400]:
        trained, t_train, _ = timed(f"[step] rf n={n}",
                                     lambda: run_step(prj, "ml.train", [spectra],
                                                      {"task": "classification",
                                                       "algorithm": "rf",
                                                       "label_column": "class",
                                                       "test_size": 0.4,
                                                       "cv_folds": 2,
                                                       "model_params": {"n_estimators": n},
                                                       "preprocess_chain": {"version": 1,
                                                                            "ops": [
                                                                                {"kind": "snv", "params": {}},
                                                                                {"kind": "sg", "params": {"window": 11,
                                                                                                           "polyorder": 2,
                                                                                                           "deriv": 1}},
                                                                            ]}},
                                                      seed=42))
        model = next(a for a in trained.outputs if a.kind == ArtifactKind.MODEL)
        m = model.metadata.get("metrics", {})
        sens["n_estimators"].append({
            "n": n,
            "time_s": round(t_train, 3),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "class_f1": m.get("class_f1"),
        })

    trained_final, _, _ = timed("[step] rf final",
                                 lambda: run_step(prj, "ml.train", [spectra],
                                                  {"task": "classification",
                                                   "algorithm": "rf",
                                                   "label_column": "class",
                                                   "test_size": 0.4,
                                                   "cv_folds": 2,
                                                   "model_params": {"n_estimators": 200},
                                                   "preprocess_chain": {"version": 1,
                                                                        "ops": [
                                                                            {"kind": "snv", "params": {}},
                                                                            {"kind": "sg", "params": {"window": 11,
                                                                                                       "polyorder": 2,
                                                                                                       "deriv": 1}},
                                                                        ]}},
                                                  seed=42))
    model_final = next(a for a in trained_final.outputs if a.kind == ArtifactKind.MODEL)
    reported, _, _ = timed("[step] report",
                           lambda: run_step(prj, "report.build", [model_final],
                                            {"title": "Zenodo Day1 深度分析报告"}))
    report_path = next(a for a in reported.outputs if a.kind == ArtifactKind.REPORT).resolve(prj.root)
    shutil.copy2(report_path, OUT / "zenodo_deep_report.html")
    prj.close()

    # --- metrics dump ---
    diff = raw_spectra["Alternaria"] - raw_spectra["Healthy"]
    peak_idx = int(np.argmax(np.abs(diff)))
    metrics_final = model_final.metadata.get("metrics", {})
    payload = {
        "total_seconds": round(time.perf_counter() - t_overall0, 2),
        "rss_baseline_kb": overall["rss_baseline_kb"],
        "cases": cases,
        "sample_summary": {
            "n_total_rois": int(len(rows)),
            "rois_by_class": {k: len(v) for k, v in spectra_by_class.items()},
        },
        "spectral_stats": {cls: v["stats"] for cls, v in seg_by_class.items()},
        "spectral_difference": {
            "peak_band_nm": float(WL[peak_idx]),
            "peak_diff": round(float(diff[peak_idx]), 4),
            "peak_direction": "Alternaria > Healthy" if diff[peak_idx] > 0 else "Healthy > Alternaria",
            "mean_abs_diff": round(float(np.abs(diff).mean()), 5),
        },
        "segmentation_sensitivity": {cls: v["trials"] for cls, v in seg_by_class.items()},
        "model_sensitivity": sens,
        "final_model": {
            "metrics": metrics_final,
            "cv": model_final.metadata.get("cv"),
        },
        "artifacts": {
            "project": str(PROJ),
            "report": str(OUT / "zenodo_deep_report.html"),
            "spectra_csv": str(OUT / "zenodo_spectra.csv"),
            "mask_tifs": [str(OUT / f"zenodo_seg_mask_{cls.lower()}.tif") for cls, _ in cases],
        },
    }
    (OUT / "zenodo_deep_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n===== SUMMARY =====")
    print(f"total wall time    : {payload['total_seconds']} s")
    print(f"peak RSS delta     : see json")
    print(f"ROIs by class      : {payload['sample_summary']['rois_by_class']}")
    print(f"spectral diff peak : {payload['spectral_difference']['peak_band_nm']:.1f} nm, "
          f"Δ={payload['spectral_difference']['peak_diff']:.4f} "
          f"({payload['spectral_difference']['peak_direction']})")
    print(f"RF final metrics   : accuracy={metrics_final.get('accuracy')}, "
          f"macro_f1={metrics_final.get('macro_f1')}, "
          f"class_f1={metrics_final.get('class_f1')}")
    print(f"report             : {OUT / 'zenodo_deep_report.html'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
