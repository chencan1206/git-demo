"""RunRecord: one execution of a step inside a project."""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


class RunStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"  # cache hit
    CANCELLED = "cancelled"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class RunRecord:
    id: str  # "run_0001"
    step: str  # registered step name, e.g. "demo.copy_dataset"
    params: dict[str, Any] = field(default_factory=dict)  # frozen snapshot
    seed: int | None = None
    status: RunStatus = RunStatus.PENDING
    started_at: str = ""  # ISO-8601
    finished_at: str = ""  # ISO-8601, empty if still running
    output_artifacts: list[str] = field(default_factory=list)
    log_tail: str = ""  # last N lines from step log
    error: str = ""

    # ------------------------------------------------------------------ IO
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        d = dict(data)
        d["status"] = RunStatus(d["status"])
        return cls(**d)

    # --------------------------------------------------------------- helpers
    def mark_running(self) -> None:
        self.status = RunStatus.RUNNING
        self.started_at = _now_iso()

    def mark_succeeded(self) -> None:
        self.status = RunStatus.SUCCEEDED
        self.finished_at = _now_iso()

    def mark_failed(self, error: str) -> None:
        self.status = RunStatus.FAILED
        self.finished_at = _now_iso()
        self.error = error

    def mark_skipped(self) -> None:
        self.status = RunStatus.SKIPPED
        self.started_at = _now_iso()
        self.finished_at = _now_iso()
