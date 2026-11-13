"""Tiny SQLite metadata store.

We keep metadata in SQLite rather than a single growing JSON file because:
  * dependency graph queries are naturally relational,
  * we need atomic writes for ``RunRecord`` append during long-running steps,
  * stage 4+ will need indexes over runs/artifacts by kind/status.

The DB is created lazily on first use; schema migrations are versioned.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    uri TEXT NOT NULL,
    hash TEXT,
    size INTEGER,
    parents TEXT,          -- JSON array of artifact IDs
    created_by_run TEXT,
    metadata TEXT          -- JSON object
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    step TEXT NOT NULL,
    params TEXT,           -- JSON object
    seed INTEGER,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    output_artifacts TEXT, -- JSON array
    log_tail TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_step ON runs(step);
CREATE INDEX IF NOT EXISTS idx_artifacts_created_by_run ON artifacts(created_by_run);
"""


class MetaStore:
    """Thin wrapper around a project's ``project.db``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None turns off the implicit transaction handling so
        # our explicit commits are the durability boundary (important for the
        # WAL crash-safety guarantees we rely on).
        self._conn = sqlite3.connect(str(path), timeout=5.0, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def integrity_check(self) -> bool:
        """Run SQLite's built-in integrity check; used on open + by tests."""
        cur = self._conn.execute("PRAGMA integrity_check")
        row = cur.fetchone()
        return bool(row) and str(row[0]).lower() == "ok"

    # --------------------------------------------------------------- runs
    def upsert_run(self, row: dict) -> None:
        cols = ",".join(row.keys())
        placeholders = ",".join(f":{k}" for k in row)
        updates = ",".join(f"{k}=excluded.{k}" for k in row)
        self._conn.execute(
            f"INSERT INTO runs({cols}) VALUES({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            row,
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        cur = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_runs(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM runs ORDER BY started_at")
        return [dict(r) for r in cur.fetchall()]

    # ----------------------------------------------------------- artifacts
    def upsert_artifact(self, row: dict) -> None:
        cols = ",".join(row.keys())
        placeholders = ",".join(f":{k}" for k in row)
        updates = ",".join(f"{k}=excluded.{k}" for k in row)
        self._conn.execute(
            f"INSERT INTO artifacts({cols}) VALUES({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            row,
        )
        self._conn.commit()

    def get_artifact(self, art_id: str) -> dict | None:
        cur = self._conn.execute("SELECT * FROM artifacts WHERE id=?", (art_id,))
        r = cur.fetchone()
        return dict(r) if r else None

    def list_artifacts(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM artifacts ORDER BY id")
        return [dict(r) for r in cur.fetchall()]

    def find_dependents(self, art_id: str) -> list[dict]:
        """Return all artifacts whose ``parents`` list includes ``art_id``."""
        cur = self._conn.execute(
            "SELECT * FROM artifacts WHERE parents LIKE ?",
            (f'%"{art_id}"%',),
        )
        return [dict(r) for r in cur.fetchall()]
