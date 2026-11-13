"""Stage-4 Step integration tests: ml.train / ml.predict / report / export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.project import Project
from plant_hsi.preprocess import PreprocessChain

BANDS = 30


def _dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, (3, BANDS))
    rows, labels = [], []
    for c in range(3):
        rows.append(centers[c] + rng.normal(0, 0.2, (12, BANDS)))
        labels += [f"class_{c}"] * 12
    return np.vstack(rows).astype(np.float32), np.asarray(labels)


def _add_spectra(prj: Project, name: str, x: np.ndarray, y: np.ndarray) -> ArtifactRef:
    frame: dict[str, object] = {
        "sample_id": [f"roi_{i}" for i in range(len(y))],
        "label": y,
    }
    for i in range(x.shape[1]):
        frame[f"b{i}"] = x[:, i]
    path = prj.root / "artifacts" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame).to_csv(path, index=False)
    return prj.add_artifact(
        ArtifactRef(
            id="",
            kind=ArtifactKind.SPECTRA,
            name=name,
            uri=f"artifacts/{name}.csv",
            metadata={"n_samples": int(len(y))},
        )
    )


@pytest.fixture
def project(tmp_path: Path):
    prj = Project.create(tmp_path / "prj", "stage4")
    yield prj
    prj.close()


@pytest.fixture
def trained(project: Project):
    x, y = _dataset()
    art = _add_spectra(project, "spectra", x, y)
    params = {
        "task": "classification",
        "algorithm": "rf",
        "label_column": "label",
        "model_params": {"n_estimators": 40},
        "test_size": 0.25,
        "cv_folds": 3,
    }
    result = project.run_step("ml.train", [art], params=params, seed=42)
    assert result.ok, result.run.error
    model = next(o for o in result.outputs if o.kind == ArtifactKind.MODEL)
    return project, art, model, params


def test_train_step_produces_all_artifacts(trained) -> None:
    prj, art, model, params = trained
    assert model.parents == [art.id]
    assert model.metadata["metrics"]["accuracy"] >= 0.8
    assert model.metadata["card"]["algorithm"] == "rf"
    assert model.metadata["card"]["split"]["stratified"] is True

    outputs = {o.name.rsplit("_", 1)[-1]: o for o in prj.list_artifacts()}
    assert "model" in outputs
    card = model.metadata["card"]
    assert card["n_train"] + card["n_test"] == 36
    assert card["feature_names"] == [f"b{i}" for i in range(BANDS)]

    metrics_path = prj.root / model.metadata["metrics_uri"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["metrics"]["accuracy"] == pytest.approx(model.metadata["metrics"]["accuracy"])

    pred = pd.read_csv(prj.root / model.metadata["predictions_uri"])
    assert len(pred) == 9
    assert {"sample_id", "y_true", "y_pred", "correct"} <= set(pred.columns)


def test_seed_is_part_of_the_cache_key(trained) -> None:
    prj, art, model, params = trained
    again = prj.run_step("ml.train", [art], params=params, seed=42)
    assert again.skipped is True
    cached = prj.find_cached_outputs("ml.train", [art], params=params, seed=42)
    assert cached and cached[0].id == model.id

    different = prj.run_step("ml.train", [art], params=params, seed=7)
    assert different.ok and not different.skipped
    other_model = next(o for o in different.outputs if o.kind == ArtifactKind.MODEL)
    assert other_model.name != model.name


def test_predict_step_batch_inference(trained) -> None:
    prj, art, model, _ = trained
    x, y = _dataset(seed=4)
    probe = _add_spectra(prj, "probe", x, y)
    result = prj.run_step("ml.predict", [model, probe], params={})
    assert result.ok, result.run.error
    out = result.outputs[0]
    frame = pd.read_csv(out.resolve(prj.root))
    assert len(frame) == len(y)
    assert {"sample_id", "prediction"} <= set(frame.columns)
    assert set(frame["prediction"]) <= {"class_0", "class_1", "class_2"}


def test_report_contains_project_params_metrics_and_run_ids(trained) -> None:
    prj, _art, model, _params = trained
    result = prj.run_step("report.build", [model], params={"title": "验收报告"})
    assert result.ok, result.run.error
    report = next(o for o in result.outputs if o.kind == ArtifactKind.REPORT)
    html = report.resolve(prj.root).read_text(encoding="utf-8")

    assert "验收报告" in html
    assert prj.meta.name in html
    assert "运行记录" in html
    assert "run_0001" in html
    assert "混淆矩阵" in html
    assert "accuracy" in html
    assert "模型卡" in html

    manifest_art = next(
        o for o in result.outputs if (o.metadata or {}).get("format") == "run_manifest"
    )
    manifest = json.loads(manifest_art.resolve(prj.root).read_text(encoding="utf-8"))
    assert manifest["paths"]["relative"] is True
    assert manifest["model"]["card"]["task"] == "classification"
    assert any(r["id"] == "run_0001" for r in manifest["runs"])
    assert all(not Path(a["uri"]).is_absolute() for a in manifest["artifacts"])


def test_preprocess_chain_is_recorded_and_replayed(trained) -> None:
    prj, art, _model, params = trained
    chain = PreprocessChain()
    chain.add("snv")
    chained = dict(params)
    chained["preprocess_chain"] = chain.to_dict()
    result = prj.run_step("ml.train", [art], params=chained, seed=3)
    assert result.ok, result.run.error
    model = next(o for o in result.outputs if o.kind == ArtifactKind.MODEL)
    card = model.metadata["card"]
    assert card["preprocess_chain"]["ops"][0]["kind"] == "snv"

    probe = _add_spectra(prj, "probe_chain", *_dataset(seed=9))
    pred = prj.run_step("ml.predict", [model, probe], params={"apply_preprocess_chain": True})
    assert pred.ok, pred.run.error


def test_export_geotiff_and_csv(trained, tmp_path: Path) -> None:
    import tifffile
    import zarr

    prj, art, _model, _params = trained
    mask_dir = prj.root / "artifacts" / "labels.zarr"
    labels = (np.arange(120).reshape(10, 12) % 4).astype("uint8")
    zarr.open_array(str(mask_dir), mode="w", shape=labels.shape, dtype="uint8")[:] = labels
    mask = prj.add_artifact(
        ArtifactRef(
            id="",
            kind=ArtifactKind.MASK,
            name="labels",
            uri="artifacts/labels.zarr",
            metadata={"shape": list(labels.shape)},
        )
    )

    tiff_result = prj.run_step("export.geotiff", [mask], params={})
    assert tiff_result.ok, tiff_result.run.error
    tiff_path = tiff_result.outputs[0].resolve(prj.root)
    assert np.array_equal(tifffile.imread(tiff_path), labels)

    csv_result = prj.run_step("export.csv", [art], params={})
    assert csv_result.ok, csv_result.run.error
    frame = pd.read_csv(csv_result.outputs[0].resolve(prj.root), encoding="utf-8-sig")
    assert frame.shape[0] == 36
    assert "label" in frame.columns
