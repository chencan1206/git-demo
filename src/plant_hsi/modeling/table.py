"""Load spectra tables and resolve label / feature columns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Columns that describe samples rather than spectral features.
META_COLUMNS = {
    "sample_id",
    "label",
    "class",
    "target",
    "y",
    "sort_order",
    "area_px",
    "row_centroid",
    "col_centroid",
    "bbox_left",
    "bbox_top",
    "bbox_w",
    "bbox_h",
    "note",
    "prediction",
    "y_true",
    "y_pred",
    "residual",
    "correct",
}

LABEL_CANDIDATES = ("label", "class", "target", "y", "category")


def load_table(path: str | Path) -> pd.DataFrame:
    """Read a spectra table from CSV / Parquet / Zarr / NPY."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"spectra table not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p, encoding="utf-8-sig")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix == ".zarr":
        import zarr

        arr = np.asarray(zarr.open_array(str(p), mode="r")[:])
        if arr.ndim != 2:
            raise ValueError(f"expected a 2-D spectra array, got shape {arr.shape}")
        return pd.DataFrame(arr, columns=[f"b{i}" for i in range(arr.shape[1])])
    if suffix in {".npy", ".npz"}:
        data = np.load(p)
        arr = np.asarray(data["spectra"] if isinstance(data, np.lib.npyio.NpzFile) else data)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2-D spectra array, got shape {arr.shape}")
        return pd.DataFrame(arr, columns=[f"b{i}" for i in range(arr.shape[1])])
    return pd.read_csv(p, encoding="utf-8-sig")


def resolve_label_column(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"label column {requested!r} not in {list(df.columns)}")
        return requested
    for name in LABEL_CANDIDATES:
        if name in df.columns:
            return name
    raise ValueError(
        f"no label column found; candidates={LABEL_CANDIDATES}, columns={list(df.columns)}"
    )


def resolve_feature_columns(
    df: pd.DataFrame,
    label_column: str,
    requested: list[str] | None = None,
) -> list[str]:
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(f"feature column(s) not found: {missing}")
        return list(requested)
    excluded = set(META_COLUMNS) | {label_column}
    cols = [
        str(c) for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        raise ValueError("no numeric feature columns found")
    return cols


def as_feature_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    x = np.asarray(df[columns].to_numpy(dtype=np.float64, copy=True))
    if not np.isfinite(x).all():
        raise ValueError("feature matrix contains NaN/Inf; clean the spectra table first")
    return x


def sample_ids(df: pd.DataFrame) -> list[str]:
    if "sample_id" in df.columns:
        return [str(v) for v in df["sample_id"].tolist()]
    return [str(i + 1) for i in range(len(df))]


def table_summary(
    df: pd.DataFrame, feature_columns: list[str], label_column: str
) -> dict[str, Any]:
    return {
        "n_samples": int(len(df)),
        "n_features": len(feature_columns),
        "feature_columns": list(feature_columns),
        "label_column": label_column,
    }
