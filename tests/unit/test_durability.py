"""Stage-6 durability: WAL crash safety + atomic ``project.json`` writes."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import plant_hsi.core.project as project_mod
from plant_hsi.core.project import Project


def test_project_uses_wal_and_busy_timeout(tmp_path: Path) -> None:
    prj = Project.create(tmp_path / "prj", "wal-test")
    try:
        mode = prj.db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        timeout = prj.db._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert str(mode).lower() == "wal"
        assert int(timeout) == 5000
        assert prj.db.integrity_check()
    finally:
        prj.close()


def test_project_json_write_is_atomic_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prj = Project.create(tmp_path / "prj", "atomic-test")
    prj.close()
    meta_path = prj.root / "project.json"
    before = meta_path.read_text(encoding="utf-8")

    real_replace = project_mod.os.replace

    def boom(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if str(dst).endswith("project.json"):
            raise OSError("simulated power loss before rename")
        real_replace(src, dst)

    monkeypatch.setattr(project_mod.os, "replace", boom)

    reopened = Project.open(prj.root)
    try:
        reopened.meta.name = "renamed"
        with pytest.raises(OSError, match="simulated power loss"):
            reopened.save()
    finally:
        reopened.close()

    assert meta_path.read_text(encoding="utf-8") == before
    assert json.loads(meta_path.read_text(encoding="utf-8"))["name"] == "atomic-test"
    assert list(meta_path.parent.glob(".project.json.tmp-*")) == []


def test_project_survives_sigkill_mid_write(tmp_path: Path) -> None:
    """Kill a writer process while it is appending artifacts; the project must
    still open, pass SQLite's integrity check, and keep every committed row
    complete."""
    root = tmp_path / "prj"
    prj = Project.create(root, "kill-test")
    prj.close()

    child_code = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
        from plant_hsi.core.project import Project

        prj = Project.open(Path(sys.argv[1]))
        i = 0
        while True:
            i += 1
            p = prj.root / "datasets" / f"f{i}.bin"
            p.write_bytes(b"x" * (i % 4096 + 1))
            prj.add_artifact(
                ArtifactRef(
                    id="",
                    kind=ArtifactKind.RAW,
                    name=f"f{i}.bin",
                    uri=f"datasets/f{i}.bin",
                )
            )
        """
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code, str(root)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    db_path = root / "project.db"
    deadline = time.time() + 20
    seen = 0
    try:
        conn = sqlite3.connect(str(db_path), timeout=2.0)
        while time.time() < deadline:
            try:
                seen = int(conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0])
            except sqlite3.Error:
                seen = 0
            if seen >= 5:
                break
            time.sleep(0.02)
        conn.close()
    finally:
        proc.kill()
        proc.wait(timeout=10)

    assert seen >= 5, f"child never committed artifacts (stderr={proc.stderr.read()!r})"

    reopened = Project.open(root)
    try:
        assert reopened.db.integrity_check()
        arts = reopened.list_artifacts()
        assert len(arts) >= 5
        for art in arts:
            assert art.id and art.kind and art.name and art.uri
        assert (
            json.loads((root / "project.json").read_text(encoding="utf-8"))["name"] == "kill-test"
        )
    finally:
        reopened.close()
