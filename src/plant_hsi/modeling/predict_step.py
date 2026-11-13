"""``ml.predict``: batch ROI inference with a saved model bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.registry import Step
from plant_hsi.modeling.persistence import (
    apply_bundle_preprocess,
    load_bundle,
    predict_bundle,
    predict_proba_bundle,
)
from plant_hsi.modeling.specs import CLASSIFICATION
from plant_hsi.modeling.step import fingerprint, jsonable, relative_uri
from plant_hsi.modeling.table import (
    as_feature_matrix,
    load_table,
    resolve_feature_columns,
    sample_ids,
)


class PredictStep(Step):
    """Apply a trained model to a spectra table (batch ROI inference).

    Inputs
        * 1 MODEL artifact
        * 1 SPECTRA artifact (CSV / Parquet / Zarr)

    Params
        apply_preprocess_chain : bool — default True; replay the chain stored
            in the bundle before predicting.  Set False when the table is
            already preprocessed.
        feature_columns : list[str] — override the bundle's feature order.
        label_column : str — optional; when present the prediction CSV also
            contains ``y_true``.
    """

    name = "ml.predict"
    input_kinds = [ArtifactKind.MODEL, ArtifactKind.SPECTRA]
    output_kinds = [ArtifactKind.OTHER]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        model_art = next((a for a in inputs if a.kind == ArtifactKind.MODEL), None)
        table_art = next((a for a in inputs if a.kind == ArtifactKind.SPECTRA), None)
        if model_art is None or table_art is None:
            raise ValueError("ml.predict expects one MODEL and one SPECTRA artifact")

        root = out_dir.parent
        bundle = load_bundle(model_art.resolve(root))
        df = load_table(table_art.resolve(root))

        expected = [str(c) for c in bundle.get("feature_names") or []]
        if expected and all(c in df.columns for c in expected):
            feature_columns = expected
        else:
            feature_columns = resolve_feature_columns(
                df, params.get("label_column") or "", params.get("feature_columns")
            )
        x = as_feature_matrix(df, feature_columns)
        if expected and len(feature_columns) != len(expected):
            raise ValueError(
                f"model expects {len(expected)} features, table provides {len(feature_columns)}"
            )

        if bool(params.get("apply_preprocess_chain", True)):
            x = apply_bundle_preprocess(bundle, x)

        task = str(bundle.get("task", CLASSIFICATION))
        preds = predict_bundle(bundle, x)
        probs = predict_proba_bundle(bundle, x)

        frame: dict[str, Any] = {
            "sample_id": sample_ids(df),
            "prediction": preds,
        }
        label_col = params.get("label_column")
        if label_col is None:
            for candidate in ("label", "class", "target", "y"):
                if candidate in df.columns:
                    label_col = candidate
                    break
        if label_col and label_col in df.columns:
            frame["y_true"] = df[label_col].to_numpy()
            frame["correct"] = frame["y_true"] == frame["prediction"]

        if task == CLASSIFICATION and probs is not None:
            classes = [str(c) for c in bundle.get("classes") or bundle["model"].classes_]
            for i, name in enumerate(classes):
                frame[f"prob_{name}"] = probs[:, i]

        pred_df = pd.DataFrame(frame)
        fp = fingerprint(self.name, inputs, params, seed)
        out_path = out_dir / f"{table_art.name}_predictions_{fp}.csv"
        pred_df.to_csv(out_path, index=False, encoding="utf-8")

        return [
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{table_art.name}_predictions_{fp}",
                uri=relative_uri(out_path, root),
                parents=[model_art.id, table_art.id],
                metadata=jsonable(
                    {
                        "format": "csv",
                        "task": task,
                        "n_rows": int(len(pred_df)),
                        "model_id": model_art.id,
                        "apply_preprocess_chain": bool(params.get("apply_preprocess_chain", True)),
                    }
                ),
            )
        ]
