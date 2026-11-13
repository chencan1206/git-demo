"""Train a single traditional model on a spectra matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plant_hsi.modeling.metrics import cross_validate, score_metrics
from plant_hsi.modeling.specs import CLASSIFICATION, TASKS, build_model, clamp_params
from plant_hsi.modeling.split import DataSplit, split_indices


@dataclass
class TrainingResult:
    task: str
    algorithm: str
    model: Any
    params_used: dict[str, Any]
    split: DataSplit
    metrics: dict[str, Any]
    cv: dict[str, Any] = field(default_factory=dict)
    y_true: NDArray[Any] | None = None
    y_pred: NDArray[Any] | None = None
    probabilities: NDArray[np.float64] | None = None
    classes: list[str] | None = None
    feature_names: list[str] = field(default_factory=list)
    seed: int = 0

    @property
    def n_train(self) -> int:
        return int(self.split.train.size)

    @property
    def n_test(self) -> int:
        return int(self.split.test.size)


def train_model(
    spectra: NDArray[np.floating],
    y: NDArray[Any],
    *,
    task: str,
    algorithm: str,
    params: dict[str, Any] | None = None,
    test_size: float = 0.2,
    val_size: float = 0.0,
    cv_folds: int = 5,
    seed: int = 0,
    feature_names: list[str] | None = None,
) -> TrainingResult:
    """Split, fit, predict and score a model.

    Fully deterministic for a fixed ``(data, params, seed)``: every estimator
    receives ``seed`` as its random state and the split is seeded as well.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    x = np.asarray(spectra, dtype=np.float64)
    target = np.asarray(y)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D spectra (samples, features), got {x.shape}")
    if target.ndim != 1:
        target = target.ravel()
    if target.shape[0] != x.shape[0]:
        raise ValueError(f"y length {target.shape[0]} != samples {x.shape[0]}")
    if not np.isfinite(x).all():
        raise ValueError("spectra contain NaN/Inf")
    if task == CLASSIFICATION:
        classes = np.unique(target)
        if classes.size < 2:
            raise ValueError("classification needs at least 2 classes")

    split = split_indices(
        x.shape[0],
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        stratify=target if task == CLASSIFICATION else None,
    )
    x_train, y_train = x[split.train], target[split.train]
    x_test, y_test = x[split.test], target[split.test]

    params_used = clamp_params(algorithm, dict(params or {}), x_train.shape[0], x_train.shape[1])
    model = build_model(task, algorithm, params_used, seed=seed)
    model.fit(x_train, y_train)
    y_pred = np.asarray(model.predict(x_test)).ravel()

    label_list: list[Any] | None = None
    if task == CLASSIFICATION:
        label_list = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    metrics = score_metrics(task, y_test, y_pred, labels=label_list)

    probabilities: NDArray[np.float64] | None = None
    if task == CLASSIFICATION and hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x_test), dtype=np.float64)

    cv = cross_validate(
        task,
        algorithm,
        x_train,
        y_train,
        params=params_used,
        folds=cv_folds,
        seed=seed,
    )

    names = (
        list(feature_names) if feature_names is not None else [f"b{i}" for i in range(x.shape[1])]
    )
    classes_out = [str(v) for v in label_list] if label_list is not None else None
    return TrainingResult(
        task=task,
        algorithm=algorithm,
        model=model,
        params_used=params_used,
        split=split,
        metrics=metrics,
        cv=cv,
        y_true=y_test,
        y_pred=y_pred,
        probabilities=probabilities,
        classes=classes_out,
        feature_names=names,
        seed=int(seed),
    )
