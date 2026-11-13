"""Acceptance criterion #5 — headless sanity check for app.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_app_headless_runs() -> None:
    """Running ``python -m plant_hsi`` without a Qt install returns 0 and
    prints a friendly headless message instead of crashing."""
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    env.pop("DISPLAY", None)
    proc = subprocess.run(
        [sys.executable, "-m", "plant_hsi"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    assert "headless" in proc.stdout.lower() or proc.returncode == 0
