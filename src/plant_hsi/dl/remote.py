"""Remote-GPU hand-off: export a training package, train headless, import back.

The package is a small JSON file pointing at a patch set plus the training
configuration; it contains no absolute requirement — ``--patchset`` overrides
the recorded path so the package can be moved between machines.

Usage on the GPU server::

    python -m plant_hsi.dl.remote train package.json --device cuda --out out/

The run writes ``bundle.pt``, ``training_log.csv`` and ``result.json``;
:func:`import_result` reads them back into a model card for the project.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plant_hsi.dl.bundle import save_bundle
from plant_hsi.dl.trainer import DLConfig, train_dc2net
from plant_hsi.features.patchset import PatchSet, band_stats

__all__ = ["export_package", "import_result", "run_package"]


def export_package(
    patchset_dir: Path,
    out_dir: Path,
    config: dict[str, Any],
    seed: int,
    card: dict[str, Any] | None = None,
) -> Path:
    """Write ``package.json`` describing everything needed to train remotely."""
    out_dir.mkdir(parents=True, exist_ok=True)
    package = {
        "format": "phs-dc2net-package/1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "patchset": str(Path(patchset_dir).resolve()),
        "config": config,
        "seed": int(seed),
        "card": card or {},
    }
    path = out_dir / "package.json"
    path.write_text(json.dumps(package, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_package(
    package_path: Path,
    out_dir: Path | None = None,
    device: str | None = None,
    patchset_override: Path | None = None,
) -> Path:
    """Train from a package and return the output directory."""
    package = json.loads(Path(package_path).read_text(encoding="utf-8"))
    if package.get("format") != "phs-dc2net-package/1":
        raise ValueError(f"unsupported package: {package_path}")
    patchset_dir = Path(patchset_override or package["patchset"])
    out = Path(out_dir or (Path(package_path).parent / "run"))
    out.mkdir(parents=True, exist_ok=True)

    ps = PatchSet.load(patchset_dir)
    classes = ps.classes
    label_index = {name: i for i, name in enumerate(classes)}
    labels = np.asarray([label_index[str(v)] for v in ps.index["label"]], dtype=np.int64)
    splits = ps.split_indices()
    if splits["test"].size == 0:
        splits["test"] = splits["val"] if splits["val"].size else splits["train"]
    mean, std = band_stats(ps.patches, splits["train"])

    config = DLConfig(**{**package["config"], **({"device": device} if device else {})})
    result = train_dc2net(
        ps.patches,
        labels,
        splits,
        classes,
        config,
        seed=int(package["seed"]),
        band_mean=mean,
        band_std=std,
    )

    card = dict(package.get("card") or {})
    card.update(
        {
            "metrics": result.test_metrics,
            "train_accuracy": result.train_accuracy,
            "n_parameters": result.n_parameters,
            "peak_memory_mb": result.peak_memory_mb,
            "device": result.device,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "stop_reason": result.stop_reason,
            "patch_size": int(ps.patches.shape[2]),
            "band_indices": ps.band_indices,
            "wavelength_nm": ps.wavelength_nm,
            "classes": classes,
            "seed": int(package["seed"]),
            "config": config.to_dict(),
            "trained_remotely": True,
        }
    )
    bundle_path = out / "bundle.pt"
    save_bundle(
        bundle_path,
        result.model,
        classes=classes,
        band_mean=mean,
        band_std=std,
        card=card,
        history=result.history,
    )
    pd.DataFrame(result.history).to_csv(out / "training_log.csv", index=False, encoding="utf-8")
    (out / "result.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def import_result(out_dir: Path) -> tuple[dict[str, Any], Path]:
    """Return ``(card, bundle_path)`` for a finished remote run."""
    out = Path(out_dir)
    card = json.loads((out / "result.json").read_text(encoding="utf-8"))
    bundle = out / "bundle.pt"
    if not bundle.exists():
        raise FileNotFoundError(f"missing bundle: {bundle}")
    return card, bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DC²Net remote training helper")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train from an exported package")
    train.add_argument("package", type=Path)
    train.add_argument("--out", type=Path, default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--patchset", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "train":
        out = run_package(args.package, args.out, args.device, args.patchset)
        print(f"trained → {out}")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
