"""Smoke test for the frozen desktop app.

Two stages, both run from the packaged executable so they exercise the real
bundled dependencies rather than the source tree:

1. ``--self-test``  — Qt starts and the main window is constructed.
2. ``--self-test --e2e`` — the full pipeline runs: import -> spectral-angle
   segmentation -> connected components -> random forest -> HTML report.

Usage:  python scripts/smoke_test.py [--e2e]
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

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


def run(exe: pathlib.Path, args: list[str], out_file: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if out_file:
        env["PHS_E2E_OUT"] = str(out_file)
    proc = subprocess.run(
        [str(exe), *args],
        timeout=900,
        capture_output=True,
        text=True,
        errors="replace",
        env=env,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print(f"exit code: {proc.returncode}")
    return proc


def main() -> int:
    exe = locate()
    print(f"smoke test: {exe}", flush=True)

    print("\n--- stage 1: GUI construction ---", flush=True)
    proc = run(exe, ["--self-test"])
    if proc.returncode != 0 or "self-test OK" not in proc.stdout:
        print("FAILED: the frozen app could not start or build its main window")
        return proc.returncode or 1
    print("OK: window constructed")

    if "--e2e" not in sys.argv:
        return 0

    print("\n--- stage 2: end-to-end pipeline ---", flush=True)
    result_file = pathlib.Path(tempfile.gettempdir()) / "phs_e2e_result.json"
    if result_file.exists():
        result_file.unlink()
    proc = run(exe, ["--e2e"], out_file=result_file)
    payload = {}
    if result_file.exists():
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        print("result file:", json.dumps(payload, indent=2))
    if proc.returncode != 0 or not payload.get("ok"):
        print("FAILED: the end-to-end pipeline did not complete")
        return proc.returncode or 1
    print(f"OK: pipeline produced {payload.get('samples')} ROIs and a {payload.get('report_bytes')} byte report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
