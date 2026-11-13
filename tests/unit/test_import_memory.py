"""Stage-6 memory acceptance: block streaming must not scale with cube size."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "large_import_bench.py"


def test_large_import_peak_memory_is_bounded(tmp_path: Path) -> None:
    """A 1 GB logical cube imports with peak RSS far below the cube size."""
    out = tmp_path / "bench"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--shape",
            "2048",
            "2048",
            "64",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    logical_mb = result["logical_bytes"] / (1 << 20)
    assert logical_mb == 1024.0
    # Peak RSS must stay well below the cube itself (and below 400 MB in
    # absolute terms) — this is what makes 10 GB imports feasible.
    assert result["peak_rss_mb"] < 400, result
    assert result["peak_rss_mb"] < logical_mb / 2, result
    assert result["health_ok"] is True, result
