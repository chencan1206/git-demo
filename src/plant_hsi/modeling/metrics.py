"""Metrics and cross-validation for classification / regression."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from plant_hsi.modeling.specs import CLASSIFICATION, build_model, clamp_params


def classification_metrics(
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    """Accuracy / precision / recall / F1 + confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    label_list = list(labels)

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        yt, yp, labels=label_list, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        yt, yp, labels=label_list, average="weighted", zero_division=0
    )
    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(
        yt, yp, labels=label_list, average=None, zero_division=0
    )
    cm = confusion_matrix(yt, yp, labels=label_list)

    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f_weighted),
        "confusion_matrix": cm.astype(int).tolist(),
        "labels": [str(v) for v in label_list],
        "per_class": [
            {
                "label": str(label_list[i]),
                "precision": float(p_cls[i]),
                "recall": float(r_cls[i]),
                "f1": float(f_cls[i]),
                "support": int(support[i]),
            }
            for i in range(len(label_list))
        ],
    }


def regression_metrics(y_true: NDArray[Any], y_pred: NDArray[Any]) -> dict[str, Any]:
    """R² / MAE / MSE / RMSE."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.size != yp.size:
        raise ValueError(f"y_true size {yt.size} != y_pred size {yp.size}")
    mse = float(mean_squared_error(yt, yp))
    return {
        "r2": float(r2_score(yt, yp)) if yt.size > 1 else float("nan"),
        "mae": float(mean_absolute_error(yt, yp)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "n": int(yt.size),
    }


def score_metrics(
    task: str,
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    if task == CLASSIFICATION:
        return classification_metrics(y_true, y_pred, labels=labels)
    return regression_metrics(y_true, y_pred)


def cross_validate(
    task: str,
    algorithm: str,
    spectra: NDArray[np.floating],
    y: NDArray[Any],
    *,
    params: dict[str, Any] | None = None,
    folds: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """K-fold cross-validation on the training portion.

    Returns ``{"folds", "metrics": {name: {mean, std, values}}}``.  Degenerate
    datasets (too few samples per class) yield an empty metric map instead of
    raising.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    x = np.asarray(spectra, dtype=np.float64)
    target = np.asarray(y)
    n = x.shape[0]
    k = int(folds)
    if k < 2 or n < k:
        return {"folds": 0, "metrics": {}}

    splitter: Any
    if task == CLASSIFICATION:
        _, counts = np.unique(target, return_counts=True)
        if counts.size >= 2 and int(np.min(counts)) >= 2:
            k = min(k, int(np.min(counts)))
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            split_iter = splitter.split(x, target)
        else:
            splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
            split_iter = splitter.split(x)
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        split_iter = splitter.split(x)

    metric_rows: dict[str, list[float]] = {}
    try:
        splits = list(split_iter)
    except ValueError:
        return {"folds": 0, "metrics": {}}

    for train_idx, test_idx in splits:
        fold_params = clamp_params(algorithm, dict(params or {}), train_idx.size, x.shape[1])
        model = build_model(task, algorithm, fold_params, seed=seed)
        model.fit(x[train_idx], target[train_idx])
        pred = np.asarray(model.predict(x[test_idx])).ravel()
        scores = score_metrics(task, target[test_idx], pred, labels=_labels(target))
        for key, value in scores.items():
            if isinstance(value, int | float) and np.isfinite(float(value)):
                metric_rows.setdefault(key, []).append(float(value))

    aggregated = {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": [float(v) for v in values],
        }
        for key, values in metric_rows.items()
        if values
    }
    return {"folds": len(splits), "metrics": aggregated}


def _labels(y: NDArray[Any]) -> list[Any] | None:
    values = sorted(set(np.asarray(y).tolist()))
    return values if values and isinstance(values[0], str) else None
