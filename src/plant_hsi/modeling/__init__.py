"""Traditional modeling layer: splits, models, metrics, cards, persistence."""

from __future__ import annotations

from plant_hsi.modeling.metrics import (
    classification_metrics,
    cross_validate,
    regression_metrics,
    score_metrics,
)
from plant_hsi.modeling.persistence import (
    ModelCard,
    apply_bundle_preprocess,
    load_bundle,
    predict_bundle,
    predict_proba_bundle,
    save_bundle,
)
from plant_hsi.modeling.specs import (
    CLASSIFICATION,
    MODELS,
    REGRESSION,
    TASKS,
    ModelSpec,
    build_model,
    get_spec,
    models_for,
)
from plant_hsi.modeling.split import DataSplit, split_indices
from plant_hsi.modeling.table import (
    as_feature_matrix,
    load_table,
    resolve_feature_columns,
    resolve_label_column,
)
from plant_hsi.modeling.training import TrainingResult, train_model

__all__ = [
    "CLASSIFICATION",
    "MODELS",
    "REGRESSION",
    "TASKS",
    "DataSplit",
    "ModelCard",
    "ModelSpec",
    "TrainingResult",
    "apply_bundle_preprocess",
    "as_feature_matrix",
    "build_model",
    "classification_metrics",
    "cross_validate",
    "get_spec",
    "load_bundle",
    "load_table",
    "models_for",
    "predict_bundle",
    "predict_proba_bundle",
    "regression_metrics",
    "resolve_feature_columns",
    "resolve_label_column",
    "save_bundle",
    "score_metrics",
    "split_indices",
    "train_model",
]
