"""ArtifactRef and RunRecord basics."""

from __future__ import annotations

from pathlib import Path

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef, compute_file_hash
from plant_hsi.core.run import RunRecord, RunStatus


def test_compute_file_hash_missing(tmp_path: Path) -> None:
    assert compute_file_hash(tmp_path / "nope.bin") == ""


def test_compute_file_hash_matches_known(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    h = compute_file_hash(p)
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64


def test_artifact_ref_round_trip() -> None:
    a = ArtifactRef(
        id="art_0001",
        kind=ArtifactKind.RAW,
        name="foo.mat",
        uri="datasets/foo.mat",
        hash="sha256:deadbeef",
        parents=["raw_0001"],
        created_by_run="run_0001",
    )
    d = a.to_dict()
    b = ArtifactRef.from_dict(d)
    assert a == b


def test_run_record_round_trip() -> None:
    r = RunRecord(
        id="run_0042",
        step="demo.copy_dataset",
        params={"x": 1, "y": 2.0},
        seed=7,
        status=RunStatus.SUCCEEDED,
        output_artifacts=["art_0007"],
    )
    r.mark_running()
    r.mark_succeeded()
    d = r.to_dict()
    b = RunRecord.from_dict(d)
    assert b.status == RunStatus.SUCCEEDED
    assert b.started_at and b.finished_at


def test_run_record_failure() -> None:
    r = RunRecord(id="", step="seg.bad")
    r.mark_running()
    r.mark_failed("boom")
    assert r.status == RunStatus.FAILED
    assert "boom" in r.error
    assert r.finished_at


def test_artifact_resolve(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "a.txt"
    f.write_text("hi")
    art = ArtifactRef(id="a", kind=ArtifactKind.OTHER, name="a.txt", uri="sub/a.txt")
    assert art.resolve(tmp_path) == f
    assert art.exists(tmp_path)
