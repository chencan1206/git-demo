"""Ordered, serialisable preprocessing chain.

A chain is a list of :class:`PreprocessOp` applied left-to-right to a
``(n_samples, n_bands)`` matrix.  The chain snapshot (``to_dict``) is JSON
safe so it can be stored next to spectra / model cards and replayed later.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from plant_hsi.preprocess.derivative import fractional_derivative, integer_derivative
from plant_hsi.preprocess.normalize import (
    mean_center,
    min_max,
    standardize,
    vector_normalize,
)
from plant_hsi.preprocess.scatter import msc, snv
from plant_hsi.preprocess.smoothing import moving_average, savitzky_golay

Applier = Callable[..., NDArray[np.float32]]

#: Public operator registry: ``kind -> (label, callable)``.
OPS: dict[str, tuple[str, Applier]] = {
    "snv": ("SNV", snv),
    "msc": ("MSC", msc),
    "sg": ("Savitzky–Golay", savitzky_golay),
    "moving_average": ("移动平均", moving_average),
    "derivative": ("整数阶导数", integer_derivative),
    "fractional_derivative": ("FOD 分数阶导数", fractional_derivative),
    "mean_center": ("均值中心", mean_center),
    "standardize": ("标准化", standardize),
    "min_max": ("Min-Max", min_max),
    "vector_normalize": ("矢量归一化", vector_normalize),
}


@dataclass(frozen=True)
class PreprocessOp:
    """One chain entry: operator id + JSON-safe keyword parameters."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in OPS:
            raise ValueError(f"unknown preprocess kind {self.kind!r}; known={sorted(OPS)}")
        _validate_params(self.kind, self.params)

    @property
    def label(self) -> str:
        return OPS[self.kind][0]

    def describe(self) -> str:
        if not self.params:
            return self.label
        body = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.label}({body})"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessOp:
        return cls(kind=str(data["kind"]), params=dict(data.get("params") or {}))


@dataclass
class ChainReport:
    """Diagnostics recorded when a chain is applied."""

    steps: list[str] = field(default_factory=list)
    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    nan_before: int = 0
    nan_after: int = 0
    inf_before: int = 0
    inf_after: int = 0
    out_of_range_before: int = 0
    out_of_range_after: int = 0

    @property
    def ok(self) -> bool:
        return self.nan_after == 0 and self.inf_after == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": list(self.steps),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "nan_before": self.nan_before,
            "nan_after": self.nan_after,
            "inf_before": self.inf_before,
            "inf_after": self.inf_after,
            "out_of_range_before": self.out_of_range_before,
            "out_of_range_after": self.out_of_range_after,
            "ok": self.ok,
        }


class PreprocessChain:
    """An ordered list of preprocessing operators."""

    def __init__(self, ops: list[PreprocessOp] | None = None) -> None:
        self.ops: list[PreprocessOp] = list(ops or [])

    # ------------------------------------------------------------ editing
    def add(self, kind: str, **params: Any) -> PreprocessOp:
        op = PreprocessOp(kind=kind, params=dict(params))
        self.ops.append(op)
        return op

    def insert(self, index: int, kind: str, **params: Any) -> PreprocessOp:
        op = PreprocessOp(kind=kind, params=dict(params))
        self.ops.insert(index, op)
        return op

    def remove(self, index: int) -> PreprocessOp:
        return self.ops.pop(index)

    def clear(self) -> None:
        self.ops.clear()

    def move(self, index: int, delta: int) -> int:
        """Move the op at ``index`` by ``delta``; return the new index."""
        target = index + delta
        if not 0 <= index < len(self.ops) or not 0 <= target < len(self.ops):
            return index
        self.ops[index], self.ops[target] = self.ops[target], self.ops[index]
        return target

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self):  # noqa: ANN204 — simple iterator proxy
        return iter(self.ops)

    # ------------------------------------------------------------ apply
    def apply(self, spectra: NDArray[np.floating]) -> NDArray[np.float32]:
        out = np.asarray(spectra, dtype=np.float32)
        if out.ndim != 2:
            raise ValueError(f"expected 2-D (samples, bands), got shape {out.shape}")
        for op in self.ops:
            out = OPS[op.kind][1](out, **op.params)
            if not np.isfinite(out).all():
                raise ValueError(f"step {op.describe()!r} produced non-finite values")
        return out.astype(np.float32, copy=False)

    def apply_with_report(
        self, spectra: NDArray[np.floating]
    ) -> tuple[NDArray[np.float32], ChainReport]:
        x = np.asarray(spectra, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"expected 2-D (samples, bands), got shape {x.shape}")
        report = ChainReport(
            steps=[op.describe() for op in self.ops],
            input_shape=(int(x.shape[0]), int(x.shape[1])),
            nan_before=int(np.isnan(x).sum()),
            inf_before=int(np.isinf(x).sum()),
            out_of_range_before=int(np.sum((x < 0) | (x > 1))),
        )
        out = self.apply(x)
        report.output_shape = (int(out.shape[0]), int(out.shape[1]))
        report.nan_after = int(np.isnan(out).sum())
        report.inf_after = int(np.isinf(out).sum())
        report.out_of_range_after = int(np.sum((out < 0) | (out > 1)))
        return out, report

    # ------------------------------------------------------------ snapshot
    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "ops": [op.to_dict() for op in self.ops]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessChain:
        return cls([PreprocessOp.from_dict(item) for item in data.get("ops", [])])

    def describe(self) -> list[str]:
        return [op.describe() for op in self.ops]


def _validate_params(kind: str, params: dict[str, Any]) -> None:
    fn = OPS[kind][1]
    allowed = {
        name
        for name, p in inspect.signature(fn).parameters.items()
        if name != "spectra" and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {sorted(unknown)} for {kind!r}; allowed={sorted(allowed)}"
        )
