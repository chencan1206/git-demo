"""Stage-6 (v1.0 hardening) acceptance runner.

Four checks, one per acceptance criterion:
  1. durability  — WAL + atomic writes survive SIGKILL (pytest subset)
  2. memory      — 10 GB logical cube imports with bounded RSS
  3. quality     — full pytest + ruff + format + mypy
  4. release     — screenshots + install/limitations docs exist

Writes /workspace/reports/stage6_acceptance.json and prints a table.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("/workspace/reports/stage6_acceptance.json")
ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src"),
    "QT_QPA_PLATFORM": "offscreen",
}


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=ENV,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def check_durability() -> dict:
    t0 = time.perf_counter()
    proc = _run([sys.executable, "-m", "pytest", "tests/unit/test_durability.py", "-q"])
    return {
        "criterion": "1. 断电/杀进程后项目不损坏（WAL + 原子写）",
        "command": "pytest tests/unit/test_durability.py -q",
        "passed": proc.returncode == 0,
        "seconds": round(time.perf_counter() - t0, 2),
        "detail": (proc.stdout or proc.stderr).strip().splitlines()[-1:],
    }


def check_memory() -> dict:
    t0 = time.perf_counter()
    proc = _run(
        [
            sys.executable,
            "scripts/large_import_bench.py",
            "--shape",
            "10000",
            "10000",
            "25",
            "--out",
            "/tmp/phs_stage6_acceptance_bench",
        ],
        timeout=900,
    )
    elapsed = round(time.perf_counter() - t0, 2)
    result: dict = {}
    if proc.returncode == 0:
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                result = json.loads(line)
                break
    peak = result.get("peak_rss_mb")
    ok = bool(result) and peak is not None and peak < 512 and result.get("health_ok") is True
    return {
        "criterion": "2. 10 GB 级立方体导入不耗尽内存（分块）",
        "command": "large_import_bench.py --shape 10000 10000 25",
        "passed": ok,
        "seconds": elapsed,
        "detail": result or (proc.stderr or "").strip().splitlines()[-3:],
    }


def check_quality() -> dict:
    t0 = time.perf_counter()
    steps = [
        ("pytest", [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings"]),
        ("ruff", ["ruff", "check", "."]),
        ("format", ["ruff", "format", "--check", "."]),
        ("mypy", [sys.executable, "-m", "mypy", "src"]),
    ]
    results = {}
    for name, cmd in steps:
        proc = _run(cmd)
        tail = (proc.stdout or proc.stderr).strip().splitlines()
        results[name] = {
            "passed": proc.returncode == 0,
            "output": tail[-1] if tail else "",
        }
    return {
        "criterion": "3. 全量测试与 UI 冒烟通过；质量门槛全绿",
        "command": "pytest / ruff / mypy",
        "passed": all(v["passed"] for v in results.values()),
        "seconds": round(time.perf_counter() - t0, 2),
        "detail": results,
    }


def check_release() -> dict:
    shots = sorted(Path("/workspace/reports/stage6").glob("*.png"))
    docs = {
        "README.md": (ROOT / "README.md").exists(),
        "KNOWN_LIMITATIONS.md": (ROOT / "KNOWN_LIMITATIONS.md").exists(),
    }
    return {
        "criterion": "4. 关键流程有截图与已知限制清单；安装文档可复现",
        "command": "artifact existence",
        "passed": len(shots) >= 4 and all(docs.values()),
        "seconds": 0.0,
        "detail": {
            "screenshots": [p.name for p in shots],
            "docs": docs,
        },
    }


def main() -> int:
    print("Stage 6 acceptance — v1.0 hardening")
    checks = [check_durability, check_memory, check_quality, check_release]
    report = []
    t0 = time.perf_counter()
    for fn in checks:
        item = fn()
        report.append(item)
        status = "PASS" if item["passed"] else "FAIL"
        print(f"  [{status}] {item['criterion']}  ({item['seconds']}s)")

    total = round(time.perf_counter() - t0, 2)
    payload = {
        "stage": 6,
        "version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_seconds": total,
        "passed": all(i["passed"] for i in report),
        "checks": report,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'ALL PASS' if payload['passed'] else 'FAILED'} in {total}s — {REPORT}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
