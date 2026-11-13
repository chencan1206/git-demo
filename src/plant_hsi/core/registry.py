"""StepRegistry and Step base class.

A :class:`Step` is the smallest unit of reproducible work: given immutable
inputs (artifact refs), declared parameters and a random seed, it produces
deterministic output artifacts and records a :class:`RunRecord`.

Stage 0 ships one concrete step — ``demo.copy_dataset`` — solely as a
smoke-test of the registry + cache + dependency + run pipeline. Real
algorithms land in later stages under ``preprocess/``, ``segment/``,
``model/`` etc.
"""

from __future__ import annotations

import abc
import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef, compute_file_hash
from plant_hsi.core.run import RunRecord, RunStatus

# ---------------------------------------------------------------------- Step


@dataclass
class StepResult:
    """Return value of :meth:`Step.run`."""

    run: RunRecord
    outputs: list[ArtifactRef] = field(default_factory=list)
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.run.status == RunStatus.SUCCEEDED or self.skipped


class Step(abc.ABC):
    """Abstract step. Subclasses must override :meth:`_execute` and declare
    :attr:`name` + :attr:`input_kinds` + :attr:`output_kinds`.
    """

    name: str = ""
    input_kinds: list[ArtifactKind] = []
    output_kinds: list[ArtifactKind] = []

    # -- public entry point ------------------------------------------------
    def run(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> StepResult:
        params = copy.deepcopy(params or {})
        # Deterministic by default: cache key excludes seed unless an explicit
        # non-zero seed is supplied.  Algorithms that *actually* use randomness
        # (e.g. PCA warm-start, data-augmented model training) should call
        # ``run(..., seed=random.randint(...))`` explicitly to get variation.
        if seed is None:
            seed = 0
            _include_seed = False
        else:
            _include_seed = True

        # Cache key: step name + params + inputs (hash only, paths irrelevant)
        # + seed (only when explicitly varied, so deterministic steps cache).
        cache_key = self._cache_key(inputs, params, seed, include_seed=_include_seed)

        # Cache hit ⇒ short-circuit
        if self._cache_hit(out_dir, cache_key, inputs):
            run = RunRecord(
                id="", step=self.name, params=params, seed=seed, status=RunStatus.SKIPPED
            )
            run.mark_skipped()
            return StepResult(run=run, skipped=True)

        run = RunRecord(id="", step=self.name, params=params, seed=seed, status=RunStatus.RUNNING)
        run.mark_running()

        try:
            outputs = self._execute(inputs, out_dir, params, seed)
            for o in outputs:
                o.hash = o.hash or compute_file_hash(o.resolve(out_dir.parent))
            run.mark_succeeded()
            run.output_artifacts = [o.id for o in outputs]
            self._write_cache_marker(out_dir, cache_key)
            return StepResult(run=run, outputs=outputs)
        except Exception as exc:  # noqa: BLE001 — we want to record any failure
            run.mark_failed(f"{type(exc).__name__}: {exc}")
            return StepResult(run=run)

    # -- subclasses implement ---------------------------------------------
    @abc.abstractmethod
    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        """Produce new artifacts under ``out_dir``. Subclasses must create
        their own IDs via :meth:`StepRegistry.new_artifact_id` when run
        inside a project; otherwise the IDs are placeholders."""
        ...

    # -- internals ---------------------------------------------------------
    def _cache_key(
        self,
        inputs: list[ArtifactRef],
        params: dict[str, Any],
        seed: int,
        include_seed: bool = True,
    ) -> str:
        payload: dict[str, Any] = {
            "step": self.name,
            "inputs": sorted(i.hash or i.id for i in inputs),
            "params": self._canonical(params),
        }
        if include_seed:
            payload["seed"] = seed
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @staticmethod
    def _canonical(x: Any) -> Any:
        """Convert numpy arrays / tuples / Paths into JSON-safe primitives."""
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (list, tuple)):
            return [Step._canonical(v) for v in x]
        if isinstance(x, dict):
            return {k: Step._canonical(v) for k, v in x.items()}
        if hasattr(x, "tolist"):
            try:
                return Step._canonical(x.tolist())
            except Exception:  # noqa: BLE001
                return str(x)
        return x

    @staticmethod
    def _cache_hit(out_dir: Path, cache_key: str, inputs: list[ArtifactRef]) -> bool:
        marker = out_dir / ".cache" / cache_key
        if not marker.exists():
            return False
        return all(not (i.hash and not i.exists(out_dir.parent)) for i in inputs)

    @staticmethod
    def _write_cache_marker(out_dir: Path, cache_key: str) -> None:
        m = out_dir / ".cache"
        m.mkdir(parents=True, exist_ok=True)
        (m / cache_key).write_text("", encoding="utf-8")


# ---------------------------------------------------------------- Demo step


class CopyDatasetStep(Step):
    """Stage-0 demo step: copy each input artifact's file into the project.

    This exists purely to exercise the registry/cache/run pipeline with
    zero real science. The copy is byte-for-byte and uses :mod:`shutil`.
    """

    name = "demo.copy_dataset"
    input_kinds = [ArtifactKind.RAW, ArtifactKind.CUBE]
    output_kinds = [ArtifactKind.RAW]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        import shutil

        out_dir.mkdir(parents=True, exist_ok=True)
        out: list[ArtifactRef] = []
        for i, src in enumerate(inputs):
            src_path = src.resolve(out_dir.parent)
            if not src_path.exists():
                raise FileNotFoundError(f"source artifact missing: {src_path}")
            dst_path = out_dir / src.name
            shutil.copy2(src_path, dst_path)
            h = compute_file_hash(dst_path)
            out.append(
                ArtifactRef(
                    id=f"_copy_{i}",
                    kind=ArtifactKind.RAW,
                    name=dst_path.name,
                    uri=str(dst_path.relative_to(out_dir.parent)).replace("\\", "/"),
                    hash=h,
                    size=dst_path.stat().st_size,
                    parents=[src.id],
                )
            )
        return out


# ----------------------------------------------------------------- Registry


class StepRegistry:
    """Registry of known :class:`Step` implementations.

    Call :meth:`register` during app startup; query with :meth:`get`.
    The instance is *not* a global singleton so tests can construct fresh
    registries without leaking state.
    """

    def __init__(self) -> None:
        self._steps: dict[str, Step] = {}

    def register(self, step: Step) -> None:
        if not step.name:
            raise ValueError("step.name must be non-empty")
        if step.name in self._steps:
            raise KeyError(f"step {step.name!r} already registered")
        self._steps[step.name] = step

    def get(self, name: str) -> Step:
        if name not in self._steps:
            raise KeyError(f"unknown step: {name!r}. known={sorted(self._steps)}")
        return self._steps[name]

    def names(self) -> list[str]:
        return sorted(self._steps)


def default_registry() -> StepRegistry:
    """Return a :class:`StepRegistry` pre-populated with the default steps."""
    from plant_hsi.export.step import ExportCsvStep, ExportGeoTiffStep  # lazy, avoids cycle
    from plant_hsi.features.patchset_step import BuildPatchSetStep
    from plant_hsi.io.import_step import ImportDatasetStep
    from plant_hsi.modeling.predict_step import PredictStep
    from plant_hsi.modeling.step import TrainModelStep
    from plant_hsi.report.step import BuildReportStep
    from plant_hsi.segment.step import ConnectedComponentsStep, ThresholdStep

    r = StepRegistry()
    r.register(CopyDatasetStep())
    r.register(ImportDatasetStep())
    r.register(ThresholdStep())
    r.register(ConnectedComponentsStep())
    r.register(TrainModelStep())
    r.register(PredictStep())
    r.register(BuildReportStep())
    r.register(ExportGeoTiffStep())
    r.register(ExportCsvStep())
    r.register(BuildPatchSetStep())
    try:
        from plant_hsi.dl.step import TrainDC2NetStep
        from plant_hsi.explain.step import BandImportanceStep
        from plant_hsi.predict.step import InferMapStep
    except ImportError:  # PyTorch (the optional ``dl`` extra) is unavailable
        return r
    r.register(TrainDC2NetStep())
    r.register(InferMapStep())
    r.register(BandImportanceStep())
    return r
