"""``ml.train``: train a traditional model from a SPECTRA table.

Outputs (all under the project's ``runs/`` directory):

* ``MODEL``   — joblib bundle (``*_model.pkl``)
* ``OTHER``   — model card, metrics JSON, predictions CSV, confusion matrix
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.registry import Step
from plant_hsi.modeling.persistence import ModelCard, save_bundle, sklearn_version, write_json
from plant_hsi.modeling.specs import CLASSIFICATION
from plant_hsi.modeling.table import (
    as_feature_matrix,
    load_table,
    resolve_feature_columns,
    resolve_label_column,
    sample_ids,
)
from plant_hsi.modeling.training import train_model
from plant_hsi.preprocess import PreprocessChain


def fingerprint(
    step_name: str, inputs: list[ArtifactRef], params: dict[str, Any], seed: int
) -> str:
    payload = {
        "step": step_name,
        "inputs": sorted(i.hash or i.id for i in inputs),
        "params": params,
        "seed": seed,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def jsonable(value: Any) -> Any:
    """Convert numpy / non-finite values into strict JSON primitives."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def relative_uri(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


class TrainModelStep(Step):
    """Train one of the 8 traditional models on a spectra table.

    Params
        task : "classification" | "regression" (default classification)
        algorithm : rf|svm|lda|knn|plsr|rfr|svr|lr
        label_column : str — default auto-detect ("label")
        feature_columns : list[str] — default all numeric non-metadata columns
        model_params : dict — algorithm hyper-parameters
        test_size, val_size : float
        cv_folds : int
        preprocess_chain : dict | None — chain applied before training and
            recorded in the model card so inference can replay it.
    """

    name = "ml.train"
    input_kinds = [ArtifactKind.SPECTRA]
    output_kinds = [ArtifactKind.MODEL, ArtifactKind.OTHER]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        if len(inputs) != 1:
            raise ValueError(f"ml.train expects 1 SPECTRA input, got {len(inputs)}")
        table_art = inputs[0]
        root = out_dir.parent
        df = load_table(table_art.resolve(root))

        label_column = resolve_label_column(df, params.get("label_column"))
        feature_columns = resolve_feature_columns(df, label_column, params.get("feature_columns"))
        x = as_feature_matrix(df, feature_columns)
        y = df[label_column].to_numpy()

        chain_snapshot = params.get("preprocess_chain") or None
        if chain_snapshot:
            x = PreprocessChain.from_dict(chain_snapshot).apply(x)

        task = str(params.get("task", CLASSIFICATION))
        algorithm = str(params.get("algorithm", "rf"))
        result = train_model(
            x,
            y,
            task=task,
            algorithm=algorithm,
            params=dict(params.get("model_params") or {}),
            test_size=float(params.get("test_size", 0.2)),
            val_size=float(params.get("val_size", 0.0)),
            cv_folds=int(params.get("cv_folds", 5)),
            seed=seed,
            feature_names=feature_columns,
        )

        fp = fingerprint(self.name, inputs, params, seed)
        base = f"{table_art.name}_{algorithm}_{fp}"
        model_path = out_dir / f"{base}_model.pkl"
        card_path = out_dir / f"{base}_model_card.json"
        metrics_path = out_dir / f"{base}_metrics.json"
        pred_path = out_dir / f"{base}_predictions.csv"
        cm_path = out_dir / f"{base}_confusion_matrix.csv"

        card = ModelCard(
            model_id=f"mdl_{fp}",
            name=base,
            task=task,
            algorithm=algorithm,
            seed=seed,
            params=jsonable(result.params_used),
            split=result.split.to_dict(),
            cv=result.cv,
            metrics=result.metrics,
            feature_names=result.feature_names,
            classes=result.classes or [],
            preprocess_chain=chain_snapshot,
            n_train=result.n_train,
            n_test=result.n_test,
            env={
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit-learn": sklearn_version(),
            },
        )
        save_bundle(
            model_path,
            result.model,
            task=task,
            algorithm=algorithm,
            feature_names=result.feature_names,
            classes=result.classes,
            preprocess_chain=chain_snapshot,
        )
        write_json(card_path, jsonable(card.to_dict()))

        metrics_payload = jsonable(
            {
                "task": task,
                "algorithm": algorithm,
                "model_params": result.params_used,
                "seed": seed,
                "split": result.split.to_dict(),
                "metrics": result.metrics,
                "cv": result.cv,
                "feature_names": result.feature_names,
                "classes": result.classes or [],
                "n_features": len(result.feature_names),
                "n_train": result.n_train,
                "n_test": result.n_test,
            }
        )
        write_json(metrics_path, metrics_payload)

        pred_df = self._predictions_frame(df, result)
        pred_df.to_csv(pred_path, index=False, encoding="utf-8")

        outputs: list[ArtifactRef] = []
        outputs.append(
            ArtifactRef(
                id="",
                kind=ArtifactKind.MODEL,
                name=f"{base}_model",
                uri=relative_uri(model_path, root),
                parents=[table_art.id],
                metadata=jsonable(
                    {
                        "task": task,
                        "algorithm": algorithm,
                        "seed": seed,
                        "metrics": result.metrics,
                        "cv": result.cv,
                        "card": card.to_dict(),
                        "predictions_uri": relative_uri(pred_path, root),
                        "metrics_uri": relative_uri(metrics_path, root),
                        "card_uri": relative_uri(card_path, root),
                        "confusion_uri": relative_uri(cm_path, root)
                        if task == CLASSIFICATION
                        else None,
                        "n_train": result.n_train,
                        "n_test": result.n_test,
                        "feature_names": result.feature_names,
                        "classes": result.classes or [],
                    }
                ),
            )
        )
        outputs.append(
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{base}_model_card",
                uri=relative_uri(card_path, root),
                parents=[table_art.id],
                metadata={"format": "model_card"},
            )
        )
        outputs.append(
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{base}_metrics",
                uri=relative_uri(metrics_path, root),
                parents=[table_art.id],
                metadata={"format": "metrics_json"},
            )
        )
        outputs.append(
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{base}_predictions",
                uri=relative_uri(pred_path, root),
                parents=[table_art.id],
                metadata={"format": "csv", "n_rows": int(len(pred_df))},
            )
        )
        if task == CLASSIFICATION:
            labels = [str(v) for v in result.metrics.get("labels", [])]
            cm = result.metrics.get("confusion_matrix", [])
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_path, encoding="utf-8")
            outputs.append(
                ArtifactRef(
                    id="",
                    kind=ArtifactKind.OTHER,
                    name=f"{base}_confusion_matrix",
                    uri=relative_uri(cm_path, root),
                    parents=[table_art.id],
                    metadata={"format": "csv", "labels": labels},
                )
            )
        return outputs

    @staticmethod
    def _predictions_frame(df: pd.DataFrame, result: Any) -> pd.DataFrame:
        ids = np.asarray(sample_ids(df), dtype=object)[result.split.test]
        frame: dict[str, Any] = {
            "sample_id": ids,
            "y_true": np.asarray(result.y_true).ravel(),
            "y_pred": np.asarray(result.y_pred).ravel(),
        }
        if result.task == CLASSIFICATION:
            frame["correct"] = frame["y_true"] == frame["y_pred"]
            if result.probabilities is not None:
                prob_classes = [str(c) for c in result.model.classes_]
                for i, name in enumerate(prob_classes):
                    frame[f"prob_{name}"] = result.probabilities[:, i]
        else:
            frame["residual"] = np.asarray(result.y_true, dtype=float) - np.asarray(
                result.y_pred, dtype=float
            )
        return pd.DataFrame(frame)
