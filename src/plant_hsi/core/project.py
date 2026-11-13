"""Project: the single top-level container for a PHS workspace.

Project owns:
  * on-disk layout (``artifacts/``, ``runs/``, ``logs/``, ``datasets/``, ``project.json``, ``project.db``)
  * the SQLite :class:`MetaStore`
  * the :class:`StepRegistry` instance (so a project can opt in/out of steps later)
  * ID generation (``art_0001``, ``run_0001``)
  * the public API for running steps and inspecting state

It does **not** render UI — that responsibility sits in :mod:`plant_hsi.ui`.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plant_hsi.core.artifact import ArtifactRef, compute_file_hash
from plant_hsi.core.db import MetaStore
from plant_hsi.core.logging_setup import get_logger
from plant_hsi.core.registry import StepRegistry, StepResult, default_registry
from plant_hsi.core.run import RunRecord, RunStatus

log = get_logger("project")


class ProjectError(Exception):
    """Raised for invariant violations inside a project (e.g. missing dirs,
    dirty lock, dependency conflict)."""


# ---------------------------------------------------------------- Layout


class ProjectState(str):
    pass  # placeholder type alias; stage 0 doesn't need richer states


LAYOUT = {
    "artifacts": "artifacts",
    "datasets": "datasets",
    "runs": "runs",
    "logs": "logs",
}


# --------------------------------------------------------------- Project meta


@dataclass
class ProjectMeta:
    schema_version: str = "1.0"
    project_id: str = ""  # e.g. "prj_0001"
    name: str = ""
    created_at: str = ""
    description: str = ""

    # ------------------------------------------------------------------ IO
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectMeta:
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


# ---------------------------------------------------------------- Project


class Project:
    """A single analysis workspace on disk."""

    JSON_FILENAME = "project.json"
    DB_FILENAME = "project.db"

    def __init__(
        self,
        root: Path,
        meta: ProjectMeta | None = None,
        registry: StepRegistry | None = None,
        _internal: bool = False,
    ) -> None:
        """Construct the handle. Callers should use :meth:`create` or
        :meth:`open` rather than this constructor."""
        self.root = root.resolve()
        self.meta = meta or ProjectMeta()
        self.registry = registry or default_registry()
        self._db: MetaStore | None = None
        self._next_art: int = 0
        self._next_run: int = 0
        if not _internal:
            self._db = MetaStore(self.root / self.DB_FILENAME)

    # -------------------------------------------------------- factory funcs
    @classmethod
    def create(
        cls,
        root: Path,
        name: str,
        description: str = "",
        registry: StepRegistry | None = None,
    ) -> Project:
        root = root.resolve()
        if root.exists():
            raise ProjectError(f"project root already exists: {root}")
        root.mkdir(parents=True, exist_ok=False)
        for sub in LAYOUT.values():
            (root / sub).mkdir(parents=True, exist_ok=False)

        pid = f"prj_{uuid.uuid4().hex[:8]}"
        meta = ProjectMeta(
            schema_version="1.0",
            project_id=pid,
            name=name,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            description=description,
        )
        prj = cls(root=root, meta=meta, registry=registry, _internal=False)
        prj._write_json()
        log.info("project created: %s (%s)", name, root)
        return prj

    @classmethod
    def open(cls, root: Path, registry: StepRegistry | None = None) -> Project:
        root = root.resolve()
        meta_path = root / cls.JSON_FILENAME
        db_path = root / cls.DB_FILENAME
        if not meta_path.exists():
            raise ProjectError(f"missing {cls.JSON_FILENAME} in {root}")
        if not db_path.exists():
            raise ProjectError(f"missing {cls.DB_FILENAME} in {root}")

        data = json.loads(meta_path.read_text(encoding="utf-8"))
        prj = cls(root=root, meta=ProjectMeta.from_dict(data), registry=registry, _internal=False)
        if not prj.db.integrity_check():
            raise ProjectError(f"project database failed integrity check: {db_path}")

        # Rehydrate counters so new IDs don't collide.
        artifacts = prj.db.list_artifacts()
        runs = prj.db.list_runs()
        prj._next_art = max((int(a["id"].split("_")[-1] or 0) for a in artifacts), default=0)
        prj._next_run = max((int(r["id"].split("_")[-1] or 0) for r in runs), default=0)
        log.info("project opened: %s (%s)", prj.meta.name, root)
        return prj

    # ------------------------------------------------------------- properties
    @property
    def db(self) -> MetaStore:
        assert self._db is not None
        return self._db

    # ------------------------------------------------------------- json save
    def save(self) -> None:
        self._write_json()
        log.info("project saved: %s", self.meta.name)

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    # ----------------------------------------------------------- ID helpers
    def new_artifact_id(self) -> str:
        self._next_art += 1
        return f"art_{self._next_art:04d}"

    def new_run_id(self) -> str:
        self._next_run += 1
        return f"run_{self._next_run:04d}"

    # ------------------------------------------------------------ artifact IO
    def add_artifact(self, art: ArtifactRef, compute_hash: bool = True) -> ArtifactRef:
        if not art.id:
            art.id = self.new_artifact_id()
        if compute_hash and not art.hash:
            p = art.resolve(self.root)
            if p.exists() and p.is_file():
                art.hash = compute_file_hash(p)
                art.size = p.stat().st_size
        self.db.upsert_artifact(_row(art))
        return art

    def list_artifacts(self) -> list[ArtifactRef]:
        return [
            ArtifactRef.from_dict(_with_json(a, {"parents", "metadata"}))
            for a in self.db.list_artifacts()
        ]

    def get_artifact(self, aid: str) -> ArtifactRef | None:
        r = self.db.get_artifact(aid)
        if r is None:
            return None
        return ArtifactRef.from_dict(_with_json(r, {"parents", "metadata"}))

    def dependents_of(self, aid: str) -> list[ArtifactRef]:
        rows = self.db.find_dependents(aid)
        return [ArtifactRef.from_dict(_with_json(r, {"parents", "metadata"})) for r in rows]

    # --------------------------------------------------------------- run API
    def run_step(
        self,
        step_name: str,
        inputs: list[ArtifactRef],
        params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> StepResult:
        step = self.registry.get(step_name)
        out_dir = self.root / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = step.run(inputs, out_dir, params=params, seed=seed)

        run = result.run
        run.id = self.new_run_id()
        self.db.upsert_run(_row(run))

        if result.ok and not result.skipped:
            ids: list[str] = []
            for art in result.outputs:
                art.id = art.id or self.new_artifact_id()
                art.created_by_run = run.id
                self.add_artifact(art, compute_hash=False)
                ids.append(art.id)
            # Refresh run.output_artifacts — Step.run() captured o.id *before*
            # project assigned the real IDs, so it was ["", ...] until now.
            run.output_artifacts = ids
            self.db.upsert_run(_row(run))
        elif result.skipped:
            log.info("run %s skipped (cache hit): %s", run.id, step_name)
        else:
            log.error("run %s failed: %s", run.id, result.run.error)
        return result

    def list_runs(self) -> list[RunRecord]:
        return [
            RunRecord.from_dict(_with_json(r, {"params", "output_artifacts"}))
            for r in self.db.list_runs()
        ]

    def find_cached_outputs(
        self,
        step_name: str,
        inputs: list[ArtifactRef],
        params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> list[ArtifactRef]:
        """Return the artifacts that an *identical* earlier run produced.

        :meth:`Step.run` short-circuits on a cache hit and returns no outputs,
        so callers that need the artifacts back (e.g. the segmentation panel
        re-running with unchanged parameters) recover them by scanning the runs
        log newest-first for the same step + params whose outputs descend from
        ``inputs``.

        When ``seed`` is given, only runs trained with that seed match — seeded
        steps (model training) must not recover another seed's outputs.
        """
        wanted = json.dumps(params or {}, sort_keys=True, default=str)
        in_ids = {i.id for i in inputs}
        for run in reversed(self.list_runs()):
            if run.step != step_name or run.status != RunStatus.SUCCEEDED:
                continue
            if seed is not None and run.seed != seed:
                continue
            if json.dumps(run.params or {}, sort_keys=True, default=str) != wanted:
                continue
            arts = [self.get_artifact(aid) for aid in run.output_artifacts]
            found = [a for a in arts if a is not None]
            if len(found) != len(run.output_artifacts) or not found:
                continue
            if any(in_ids & set(a.parents) for a in found):
                return found
        return []

    # --------------------------------------------------------- dependency guard
    def can_delete(self, art_id: str) -> tuple[bool, list[ArtifactRef]]:
        """Return ``(can_delete, dependents)``. A non-empty ``dependents`` list
        means something upstream depends on this artifact and the UI must ask
        the user to confirm before deleting (or suggest cascade)."""
        dep = self.dependents_of(art_id)
        return (len(dep) == 0, dep)

    # --------------------------------------------------------- private helpers
    def _write_json(self) -> None:
        _atomic_write_text(
            self.root / self.JSON_FILENAME,
            json.dumps(self.meta.to_dict(), indent=2),
        )


# --------------------------------------------------------- small utilities


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` such that readers only ever see the old or
    the complete new contents — never a truncated file.

    Strategy: write a sibling temp file, fsync it, then ``os.replace`` it over
    the destination (atomic on POSIX and Windows). The directory is fsynced
    afterwards so the rename itself survives a power loss.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        with contextlib.suppress(OSError):
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


def _row(obj: ArtifactRef | RunRecord, extra_json: tuple[str, ...] = ()) -> dict[str, Any]:
    """Turn a domain object into a DB row, serialising all list/dict values
    (and any explicitly requested ``extra_json`` keys) as JSON strings."""
    data = obj.to_dict()
    for k, v in data.items():
        if k in extra_json or isinstance(v, (list, dict)):
            data[k] = json.dumps(v, default=str)
    return data


def _with_json(row: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    """Inverse of :func:`_row`: parse JSON columns back into Python objects."""
    out = dict(row)
    for f in fields:
        if f in out and out[f]:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                out[f] = json.loads(out[f])
    return out
