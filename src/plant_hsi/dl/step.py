"""Steps for the DC²Net branch: ``dl.train``."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.registry import Step
from plant_hsi.dl.bundle import save_bundle
from plant_hsi.dl.trainer import DLConfig, DLTrainResult, train_dc2net
from plant_hsi.features.patchset import PatchSet, band_stats
from plant_hsi.modeling.step import fingerprint, jsonable, relative_uri

__all__ = ["TrainDC2NetStep"]


class TrainDC2NetStep(Step):
    """Train DC²Net on a patch set and save weights + model card.

    Optional second input: a previous MODEL artifact to resume from.
    """

    name = "dl.train"
    input_kinds = [ArtifactKind.OTHER, ArtifactKind.MODEL]
    output_kinds = [ArtifactKind.MODEL, ArtifactKind.OTHER]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        patch_art = next(
            (a for a in inputs if (a.metadata or {}).get("format") == "patchset"), None
        )
        if patch_art is None:
            raise ValueError("dl.train expects a patchset artifact as input")
        resume_art = next((a for a in inputs if a.kind == ArtifactKind.MODEL), None)

        root = out_dir.parent
        ps = PatchSet.load(patch_art.resolve(root))
        classes = ps.classes
        label_index = {name: i for i, name in enumerate(classes)}
        labels = np.asarray([label_index[str(v)] for v in ps.index["label"]], dtype=np.int64)
        splits = ps.split_indices()
        if splits["test"].size == 0:
            splits["test"] = splits["val"] if splits["val"].size else splits["train"]

        mean, std = band_stats(ps.patches, splits["train"])
        config = DLConfig(
            epochs=int(params.get("epochs", 30)),
            batch_size=int(params.get("batch_size", 16)),
            lr=float(params.get("lr", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
            patience=int(params.get("patience", 8)),
            device=str(params.get("device", "auto")),
            spectral_channels=int(params.get("spectral_channels", 8)),
            spectral_kernel=int(params.get("spectral_kernel", 5)),
            spectral_dilation=int(params.get("spectral_dilation", 2)),
            feat_channels=int(params.get("feat_channels", 32)),
            hidden_channels=int(params.get("hidden_channels", 64)),
            dropout=float(params.get("dropout", 0.2)),
            attention_reduction=int(params.get("attention_reduction", 4)),
        )
        resume: dict[str, Any] | None = None
        if resume_art is not None:
            from plant_hsi.dl.bundle import load_bundle

            prev = load_bundle(resume_art.resolve(root))
            resume = {
                "state_dict": prev["state_dict"],
                "history": prev.get("history") or [],
                "epoch": len(prev.get("history") or []),
            }

        result = train_dc2net(
            ps.patches,
            labels,
            splits,
            classes,
            config,
            seed=seed,
            band_mean=mean,
            band_std=std,
            resume=resume,
        )

        fp = fingerprint(self.name, inputs, params, seed)
        base = f"{patch_art.name}_dc2net_{fp}"
        card = self._card(result, ps, patch_art, splits, seed, mean, std)
        model_path = out_dir / f"{base}_model.pt"
        card_path = out_dir / f"{base}_model_card.json"
        metrics_path = out_dir / f"{base}_metrics.json"
        history_path = out_dir / f"{base}_training_log.csv"
        curve_path = out_dir / f"{base}_training_curve.svg"

        save_bundle(
            model_path,
            result.model,
            classes=classes,
            band_mean=mean,
            band_std=std,
            card=card,
            history=result.history,
        )
        card_path.write_text(
            json.dumps(jsonable(card), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        metrics_payload = jsonable(
            {
                "task": "classification",
                "algorithm": "DC2Net",
                "seed": seed,
                "split": {k: int(v.size) for k, v in splits.items()},
                "metrics": result.test_metrics,
                "train_accuracy": result.train_accuracy,
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "stop_reason": result.stop_reason,
                "n_parameters": result.n_parameters,
                "peak_memory_mb": result.peak_memory_mb,
                "device": result.device,
                "classes": classes,
            }
        )
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pd.DataFrame(result.history).to_csv(history_path, index=False, encoding="utf-8")
        curve_path.write_text(_curve_svg(result.history), encoding="utf-8")

        outputs = [
            ArtifactRef(
                id="",
                kind=ArtifactKind.MODEL,
                name=f"{base}_model",
                uri=relative_uri(model_path, root),
                parents=[patch_art.id],
                metadata=jsonable(
                    {
                        "task": "classification",
                        "algorithm": "DC2Net",
                        "seed": seed,
                        "classes": classes,
                        "metrics": result.test_metrics,
                        "train_accuracy": result.train_accuracy,
                        "patch_size": int(ps.patches.shape[2]),
                        "band_indices": ps.band_indices,
                        "wavelength_nm": ps.wavelength_nm,
                        "band_mean": mean,
                        "band_std": std,
                        "n_parameters": result.n_parameters,
                        "peak_memory_mb": result.peak_memory_mb,
                        "device": result.device,
                        "best_epoch": result.best_epoch,
                        "epochs_run": result.epochs_run,
                        "stop_reason": result.stop_reason,
                        "card": card,
                        "card_uri": relative_uri(card_path, root),
                        "metrics_uri": relative_uri(metrics_path, root),
                        "history_uri": relative_uri(history_path, root),
                        "curve_uri": relative_uri(curve_path, root),
                    }
                ),
            ),
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{base}_model_card",
                uri=relative_uri(card_path, root),
                parents=[patch_art.id],
                metadata={"format": "model_card"},
            ),
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{base}_training_log",
                uri=relative_uri(history_path, root),
                parents=[patch_art.id],
                metadata={"format": "csv", "n_rows": len(result.history)},
            ),
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{base}_training_curve",
                uri=relative_uri(curve_path, root),
                parents=[patch_art.id],
                metadata={"format": "svg"},
            ),
        ]
        return outputs

    @staticmethod
    def _card(
        result: DLTrainResult,
        ps: PatchSet,
        patch_art: ArtifactRef,
        splits: dict[str, np.ndarray],
        seed: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "name": f"DC2Net · {patch_art.name}",
            "task": "classification",
            "algorithm": "DC2Net",
            "seed": seed,
            "params": result.config.to_dict(),
            "split": {
                "strategy": "grouped_by_leaf",
                "n_train": int(splits["train"].size),
                "n_val": int(splits["val"].size),
                "n_test": int(splits["test"].size),
                "leaf_splits": ps.meta.get("leaf_splits", {}),
            },
            "metrics": jsonable(result.test_metrics),
            "train_accuracy": result.train_accuracy,
            "classes": result.classes,
            "n_parameters": result.n_parameters,
            "patch_size": int(ps.patches.shape[2]),
            "n_bands": int(ps.patches.shape[1]),
            "band_indices": ps.band_indices,
            "band_mean": [float(v) for v in mean],
            "band_std": [float(v) for v in std],
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "stop_reason": result.stop_reason,
            "peak_memory_mb": result.peak_memory_mb,
            "device": result.device,
            "env": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": _torch_version(),
            },
        }


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


def _curve_svg(history: list[dict[str, Any]], width: int = 640, height: int = 260) -> str:
    if not history:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='640' height='40'></svg>"
    train = [float(r.get("train_loss", 0.0)) for r in history]
    val = [float(r.get("val_loss", 0.0)) for r in history]
    hi = max(train + val + [1e-6])
    left, right, top, bottom = 52, 16, 18, 38
    inner_w = width - left - right
    inner_h = height - top - bottom
    step = inner_w / max(1, len(train) - 1)

    def poly(values: list[float]) -> str:
        return " ".join(
            f"{left + i * step:.1f},{top + inner_h - (v / hi) * inner_h:.1f}"
            for i, v in enumerate(values)
        )

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="训练曲线">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top + inner_h}" x2="{width - right}" y2="{top + inner_h}" '
        f'stroke="#c8d2df"/>',
        f'<polyline fill="none" stroke="#2f6fdb" stroke-width="2" points="{poly(train)}"/>',
        f'<polyline fill="none" stroke="#d9534f" stroke-width="2" points="{poly(val)}"/>',
        f'<text x="{left}" y="{height - 12}" font-size="12" fill="#5c6b7f">epoch</text>',
        f'<text x="8" y="{top + 6}" font-size="12" fill="#5c6b7f">loss</text>',
        f'<text x="{width - 170}" y="16" font-size="12" fill="#2f6fdb">训练损失</text>',
        f'<text x="{width - 92}" y="16" font-size="12" fill="#d9534f">验证损失</text>',
        "</svg>",
    ]
    return "".join(parts)
