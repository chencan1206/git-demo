"""Stage-0 StepRegistry + Step tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.registry import Step, StepRegistry, default_registry
from plant_hsi.core.run import RunStatus


def test_registry_register_and_get() -> None:
    reg = StepRegistry()
    assert reg.names() == []
    reg.register(default_registry().get("demo.copy_dataset"))
    assert "demo.copy_dataset" in reg.names()


def test_registry_rejects_duplicates() -> None:
    reg = default_registry()
    with pytest.raises(KeyError):
        reg.register(default_registry().get("demo.copy_dataset"))


def test_registry_unknown() -> None:
    reg = default_registry()
    with pytest.raises(KeyError):
        reg.get("nope.nope")


class _FailingStep(Step):
    name = "test.failing"

    def _execute(self, inputs, out_dir, params, seed):  # noqa: ANN001
        raise RuntimeError("kaboom")


def test_step_failure_is_recorded(tmp_path: Path) -> None:
    reg = StepRegistry()
    reg.register(_FailingStep())
    step = reg.get("test.failing")
    result = step.run([], tmp_path, params={}, seed=1)
    assert not result.ok
    assert result.run.status == RunStatus.FAILED
    assert "kaboom" in result.run.error


def test_step_deterministic_key(tmp_path: Path) -> None:
    reg = default_registry()
    step = reg.get("demo.copy_dataset")

    (tmp_path / "in.bin").write_bytes(b"data")
    src = ArtifactRef(
        id="art_0001",
        kind=ArtifactKind.RAW,
        name="in.bin",
        uri="in.bin",
        hash=__import__(
            "plant_hsi.core.artifact", fromlist=["compute_file_hash"]
        ).compute_file_hash(tmp_path / "in.bin"),
    )

    r1 = step.run([src], tmp_path / "o1", {"x": 1}, seed=5)
    r2 = step.run([src], tmp_path / "o2", {"x": 1}, seed=5)
    r3 = step.run([src], tmp_path / "o3", {"x": 2}, seed=5)

    assert r1.ok and r2.ok
    # Different cache dirs ⇒ caches don't collide, but params differ ⇒ r3 is NOT skipped
    assert not r3.skipped
