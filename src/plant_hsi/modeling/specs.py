"""Traditional ML model registry (8 models: 4 classification + 4 regression).

scikit-learn is imported lazily inside :func:`build_model` so that importing
the modeling package (and therefore the Step registry) never requires the
optional ML extras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CLASSIFICATION = "classification"
REGRESSION = "regression"
TASKS = (CLASSIFICATION, REGRESSION)


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one algorithm."""

    key: str
    label: str
    task: str
    defaults: dict[str, Any] = field(default_factory=dict)
    #: ``name -> {label, type, min, max, step, choices, default}`` used by the
    #: GUI to build parameter widgets generically.
    params: dict[str, dict[str, Any]] = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    # ------------------------------------------------------- classification
    "rf": ModelSpec(
        key="rf",
        label="随机森林 (RF)",
        task=CLASSIFICATION,
        defaults={"n_estimators": 200},
        params={
            "n_estimators": {
                "label": "树数量",
                "type": "int",
                "min": 10,
                "max": 2000,
                "step": 10,
                "default": 200,
            }
        },
    ),
    "svm": ModelSpec(
        key="svm",
        label="支持向量机 (SVM)",
        task=CLASSIFICATION,
        defaults={"C": 1.0, "gamma": "scale"},
        params={
            "C": {
                "label": "惩罚系数 C",
                "type": "float",
                "min": 0.001,
                "max": 1000.0,
                "default": 1.0,
            },
            "gamma": {
                "label": "gamma",
                "type": "choice",
                "choices": ["scale", "auto"],
                "default": "scale",
            },
        },
    ),
    "lda": ModelSpec(key="lda", label="线性判别分析 (LDA)", task=CLASSIFICATION),
    "knn": ModelSpec(
        key="knn",
        label="K 近邻 (KNN)",
        task=CLASSIFICATION,
        defaults={"n_neighbors": 5},
        params={
            "n_neighbors": {
                "label": "邻居数 k",
                "type": "int",
                "min": 1,
                "max": 50,
                "step": 1,
                "default": 5,
            }
        },
    ),
    # ----------------------------------------------------------- regression
    "plsr": ModelSpec(
        key="plsr",
        label="偏最小二乘回归 (PLSR)",
        task=REGRESSION,
        defaults={"n_components": 2},
        params={
            "n_components": {
                "label": "主成分数",
                "type": "int",
                "min": 1,
                "max": 30,
                "step": 1,
                "default": 2,
            }
        },
    ),
    "rfr": ModelSpec(
        key="rfr",
        label="随机森林回归 (RFR)",
        task=REGRESSION,
        defaults={"n_estimators": 200},
        params={
            "n_estimators": {
                "label": "树数量",
                "type": "int",
                "min": 10,
                "max": 2000,
                "step": 10,
                "default": 200,
            }
        },
    ),
    "svr": ModelSpec(
        key="svr",
        label="支持向量回归 (SVR)",
        task=REGRESSION,
        defaults={"C": 1.0, "gamma": "scale", "epsilon": 0.1},
        params={
            "C": {
                "label": "惩罚系数 C",
                "type": "float",
                "min": 0.001,
                "max": 1000.0,
                "default": 1.0,
            },
            "gamma": {
                "label": "gamma",
                "type": "choice",
                "choices": ["scale", "auto"],
                "default": "scale",
            },
            "epsilon": {
                "label": "epsilon",
                "type": "float",
                "min": 0.0,
                "max": 10.0,
                "default": 0.1,
            },
        },
    ),
    "lr": ModelSpec(key="lr", label="线性回归 (LR)", task=REGRESSION),
}

MODEL_ORDER: dict[str, list[str]] = {
    CLASSIFICATION: ["rf", "svm", "lda", "knn"],
    REGRESSION: ["plsr", "rfr", "svr", "lr"],
}


def models_for(task: str) -> list[ModelSpec]:
    """Return model specs for ``task`` in UI order."""
    if task not in MODEL_ORDER:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    return [MODELS[key] for key in MODEL_ORDER[task]]


def get_spec(task: str, algorithm: str) -> ModelSpec:
    spec = MODELS.get(algorithm)
    if spec is None:
        raise ValueError(f"unknown algorithm {algorithm!r}; known={sorted(MODELS)}")
    if spec.task != task:
        raise ValueError(f"algorithm {algorithm!r} is for {spec.task}, not {task}")
    return spec


def clamp_params(
    algorithm: str,
    params: dict[str, Any],
    n_train: int,
    n_features: int,
) -> dict[str, Any]:
    """Keep algorithm parameters inside the feasible range for a dataset."""
    p = dict(params)
    if algorithm == "plsr":
        k = int(p.get("n_components", 2))
        p["n_components"] = max(1, min(k, n_features, n_train))
    elif algorithm == "knn":
        k = int(p.get("n_neighbors", 5))
        p["n_neighbors"] = max(1, min(k, n_train))
    return p


def build_model(
    task: str,
    algorithm: str,
    params: dict[str, Any] | None = None,
    seed: int = 0,
) -> Any:
    """Instantiate a scikit-learn estimator with deterministic seeding."""
    spec = get_spec(task, algorithm)
    p = dict(spec.defaults)
    p.update(params or {})

    if algorithm == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=seed, n_jobs=-1, **p)
    if algorithm == "svm":
        from sklearn.svm import SVC

        return SVC(kernel="rbf", probability=True, random_state=seed, **p)
    if algorithm == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        return LinearDiscriminantAnalysis(**p)
    if algorithm == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(**p)
    if algorithm == "plsr":
        from sklearn.cross_decomposition import PLSRegression

        return PLSRegression(**p)
    if algorithm == "rfr":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=seed, n_jobs=-1, **p)
    if algorithm == "svr":
        from sklearn.svm import SVR

        return SVR(kernel="rbf", **p)
    if algorithm == "lr":
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**p)
    raise ValueError(f"unhandled algorithm {algorithm!r}")
