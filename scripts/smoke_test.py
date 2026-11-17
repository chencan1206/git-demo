"""Smoke test for the frozen desktop app.

Runs the packaged executable with ``--self-test`` so the GUI is constructed on
the offscreen platform and then exits. This catches packaging regressions that
only appear in the frozen build (missing hidden imports, absent Qt plugins,
wrong headless detection) on every platform.

Usage:  python scripts/smoke_test.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def locate() -> pathlib.Path:
    if sys.platform == "darwin":
        candidates = [ROOT / "dist" / "PHS Studio.app" / "Contents" / "MacOS" / "PHS-Studio"]
    elif sys.platform.startswith("win"):
        candidates = [ROOT / "dist" / "PHS-Studio.exe"]
    else:
        candidates = [ROOT / "dist" / "PHS-Studio"]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"frozen binary not found, looked for: {candidates}")


def main() -> int:
    exe = locate()
    print(f"smoke test: {exe} --self-test", flush=True)
    proc = subprocess.run(
        [str(exe), "--self-test"],
        timeout=300,
        capture_output=True,
        text=True,
        errors="replace",
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print(f"exit code: {proc.returncode}")
    if proc.returncode != 0:
        print("FAILED: the frozen app could not start")
        return proc.returncode
    if "self-test OK" not in proc.stdout:
        print("FAILED: no confirmation from --self-test")
        return 1
    print("OK: frozen app starts and builds its main window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
