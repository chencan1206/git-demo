"""Model card + joblib bundle persistence.

A bundle is a small dict holding the estimator plus everything inference
needs (task, algorithm, feature order, classes and the preprocessing chain
snapshot).  Loading it in a fresh process therefore never depends on project
state.
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

BUNDLE_VERSION = 1


@dataclass
class ModelCard:
    """Human- and machine-readable model metadata."""

    model_id: str = ""
    name: str = ""
    task: str = ""
    algorithm: str = ""
    framework: str = "scikit-learn"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    cv: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    preprocess_chain: dict[str, Any] | None = None
    n_train: int = 0
    n_test: int = 0
    env: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


def sklearn_version() -> str:
    try:
        return importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - ml extra missing
        return ""


def save_bundle(
    path: Path | str,
    model: Any,
    *,
    task: str,
    algorithm: str,
    feature_names: list[str],
    classes: list[str] | None = None,
    preprocess_chain: dict[str, Any] | None = None,
) -> Path:
    """Persist estimator + inference metadata via joblib."""
    import joblib

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "task": task,
        "algorithm": algorithm,
        "feature_names": list(feature_names),
        "classes": list(classes or []),
        "preprocess_chain": preprocess_chain,
        "model": model,
    }
    joblib.dump(bundle, out)
    return out


def load_bundle(path: Path | str) -> dict[str, Any]:
    import joblib

    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"not a PHS model bundle: {path}")
    return bundle


def predict_bundle(bundle: dict[str, Any], spectra: NDArray[np.floating]) -> NDArray[Any]:
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    expected = bundle.get("feature_names") or []
    if expected and x.shape[1] != len(expected):
        raise ValueError(
            f"feature count mismatch: model expects {len(expected)}, input has {x.shape[1]}"
        )
    return np.asarray(bundle["model"].predict(x)).ravel()


def predict_proba_bundle(
    bundle: dict[str, Any], spectra: NDArray[np.floating]
) -> NDArray[np.float64] | None:
    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        return None
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.asarray(model.predict_proba(x), dtype=np.float64)


def apply_bundle_preprocess(
    bundle: dict[str, Any], spectra: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Replay the bundle's preprocessing chain (no-op when absent)."""
    snapshot = bundle.get("preprocess_chain")
    if not snapshot:
        return np.asarray(spectra)
    from plant_hsi.preprocess import PreprocessChain

    chain = PreprocessChain.from_dict(snapshot)
    return np.asarray(chain.apply(spectra), dtype=np.float64)


def read_json(path: Path | str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def write_json(path: Path | str, data: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out
