"""Synthetic large-cube import benchmark for Stage-6 acceptance.

Builds a logical hyperspectral cube of ``H x W x B`` float32 values *without
ever materialising it*, streams it through the same ``write_zarr_stream`` code
path used by the real importer, and reports peak RSS. This is how we verify
that a 10 GB import stays within a bounded memory envelope even on a machine
with far less RAM.

Usage:
    python scripts/large_import_bench.py --shape 10000 10000 25 --out /tmp/bench
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_hsi.io.spec import CubeStream, DatasetSpec, write_zarr_stream  # noqa: E402


def make_stream(shape: tuple[int, int, int], value: float = 0.5) -> CubeStream:
    h, w, b = shape
    spec = DatasetSpec(
        source_path=f"synthetic://{h}x{w}x{b}",
        format="synthetic",
        shape=shape,
        wavelength_nm=list(np.linspace(400, 950, b)),
        notes="synthetic block stream; no source file",
    )

    def reader(y0: int, y1: int) -> np.ndarray:
        return np.full((y1 - y0, w, b), value, dtype=np.float32)

    return CubeStream(spec, shape, reader)


def peak_rss_mb() -> float:
    """Peak resident set size of *this* process.

    ``ru_maxrss`` can be inherited from a forking parent (e.g. a pytest run
    that already imported torch), which would make the number meaningless.
    ``VmHWM`` in ``/proc/self/status`` belongs to the post-exec address space,
    so prefer it and fall back to ``ru_maxrss`` elsewhere.
    """
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, required=True, metavar=("H", "W", "B"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-mb", type=int, default=32)
    ap.add_argument("--keep", action="store_true", help="keep the produced .zarr")
    args = ap.parse_args()

    shape = (args.shape[0], args.shape[1], args.shape[2])
    logical_bytes = int(np.prod(shape)) * 4
    stream = make_stream(shape)
    zarr_dir = args.out / "bench.zarr"

    t0 = time.perf_counter()
    _, health = write_zarr_stream(stream, zarr_dir, target_bytes=args.target_mb << 20)
    elapsed = time.perf_counter() - t0

    peak = peak_rss_mb()
    stored_bytes = sum(p.stat().st_size for p in zarr_dir.rglob("*") if p.is_file())

    result = {
        "shape": list(shape),
        "logical_bytes": logical_bytes,
        "logical_gb": round(logical_bytes / 1e9, 3),
        "stored_bytes": stored_bytes,
        "peak_rss_mb": round(peak, 1),
        "elapsed_s": round(elapsed, 2),
        "health_ok": health.ok,
        "zarr": str(zarr_dir),
    }
    print(json.dumps(result))

    if not args.keep:
        shutil.rmtree(zarr_dir, ignore_errors=True)
        shutil.rmtree(args.out, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
