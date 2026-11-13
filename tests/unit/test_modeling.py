"""Stage-4 unit tests: model registry, splits, metrics, training, persistence."""

from __future__ import annotations

import numpy as np
import pytest

from plant_hsi.modeling import (
    MODELS,
    ModelCard,
    load_bundle,
    predict_bundle,
    save_bundle,
    split_indices,
    train_model,
)
from plant_hsi.modeling.metrics import classification_metrics, regression_metrics
from plant_hsi.modeling.step import TrainModelStep

BANDS = 40

CLASSIFICATION_MODELS = ["rf", "svm", "lda", "knn"]
REGRESSION_MODELS = ["plsr", "rfr", "svr", "lr"]


def _blobs(
    n_per_class: int = 12, n_classes: int = 3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, (n_classes, BANDS))
    rows, labels = [], []
    for c in range(n_classes):
        rows.append(centers[c] + rng.normal(0, 0.25, (n_per_class, BANDS)))
        labels += [f"class_{c}"] * n_per_class
    return np.vstack(rows).astype(np.float64), np.asarray(labels)


def _regression_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (45, BANDS))
    weights = rng.normal(0, 1, BANDS)
    y = x @ weights * 0.4 + 1.5
    return x, y


def test_registry_has_eight_models() -> None:
    assert len(MODELS) == 8
    for key in CLASSIFICATION_MODELS:
        assert MODELS[key].task == "classification"
    for key in REGRESSION_MODELS:
        assert MODELS[key].task == "regression"


def test_classification_metrics_known_values() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["labels"] == ["0", "1"]
    assert len(metrics["per_class"]) == 2


def test_regression_metrics_known_values() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["mse"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert metrics["n"] == 3


def test_split_is_deterministic_and_stratified() -> None:
    _, labels = _blobs()
    a = split_indices(len(labels), test_size=0.25, seed=7, stratify=labels)
    b = split_indices(len(labels), test_size=0.25, seed=7, stratify=labels)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)
    assert a.stratified is True
    assert a.train.size + a.test.size == len(labels)
    assert a.test.size == 9

    train_labels = set(labels[a.train])
    test_labels = set(labels[a.test])
    assert train_labels == test_labels == {"class_0", "class_1", "class_2"}


def test_split_with_validation() -> None:
    _, labels = _blobs()
    split = split_indices(len(labels), test_size=0.2, val_size=0.2, seed=3, stratify=labels)
    assert split.val.size > 0
    assert split.train.size + split.val.size + split.test.size == len(labels)
    assert not set(split.train) & set(split.test)
    assert not set(split.val) & set(split.test)


def test_same_seed_training_is_reproducible() -> None:
    x, y = _blobs()
    first = train_model(x, y, task="classification", algorithm="rf", cv_folds=3, seed=11)
    second = train_model(x, y, task="classification", algorithm="rf", cv_folds=3, seed=11)
    assert np.array_equal(first.y_pred, second.y_pred)
    assert first.metrics["accuracy"] == pytest.approx(second.metrics["accuracy"], abs=1e-12)
    assert first.metrics["confusion_matrix"] == second.metrics["confusion_matrix"]


def test_different_seed_changes_the_forest() -> None:
    x, y = _blobs()
    first = train_model(x, y, task="classification", algorithm="rf", cv_folds=2, seed=1)
    second = train_model(x, y, task="classification", algorithm="rf", cv_folds=2, seed=2)
    thresholds_a = first.model.estimators_[0].tree_.threshold
    thresholds_b = second.model.estimators_[0].tree_.threshold
    assert not np.array_equal(thresholds_a, thresholds_b)


def test_all_classification_models_train() -> None:
    x, y = _blobs()
    for algorithm in CLASSIFICATION_MODELS:
        result = train_model(x, y, task="classification", algorithm=algorithm, cv_folds=2, seed=5)
        assert result.metrics["accuracy"] >= 0.5
        assert np.isfinite(result.y_pred.size)
        assert result.classes == ["class_0", "class_1", "class_2"]


def test_all_regression_models_train() -> None:
    x, y = _regression_data()
    for algorithm in REGRESSION_MODELS:
        result = train_model(x, y, task="regression", algorithm=algorithm, cv_folds=2, seed=5)
        assert set(result.metrics) >= {"r2", "mae", "mse", "rmse"}
        assert np.isfinite(result.metrics["rmse"])


def test_plsr_components_are_clamped() -> None:
    x, y = _regression_data()
    result = train_model(
        x, y, task="regression", algorithm="plsr", params={"n_components": 99}, seed=0
    )
    assert result.params_used["n_components"] <= x.shape[1]


def test_bundle_roundtrip_predictions(tmp_path) -> None:
    x, y = _blobs()
    result = train_model(x, y, task="classification", algorithm="svm", seed=3)
    path = tmp_path / "model.pkl"
    save_bundle(
        path,
        result.model,
        task="classification",
        algorithm="svm",
        feature_names=result.feature_names,
        classes=result.classes,
    )
    bundle = load_bundle(path)
    expected = result.model.predict(x)
    assert np.array_equal(predict_bundle(bundle, x), expected)
    assert bundle["preprocess_chain"] is None


def test_model_card_roundtrip() -> None:
    card = ModelCard(model_id="mdl_1", name="demo", task="classification", algorithm="rf")
    data = card.to_dict()
    again = ModelCard.from_dict(data)
    assert again.model_id == "mdl_1"
    assert again.algorithm == "rf"
    assert again.created_at == card.created_at


def test_train_step_is_registered() -> None:
    assert TrainModelStep.name == "ml.train"
