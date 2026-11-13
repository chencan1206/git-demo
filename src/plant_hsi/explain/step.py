"""``explain.wavelength``: band importance for a trained DC²Net."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.registry import Step
from plant_hsi.dl.bundle import load_bundle
from plant_hsi.explain.wavelength import (
    attention_importance,
    combine_importance,
    gradient_saliency,
    importance_svg,
    top_bands,
    write_importance_csv,
)
from plant_hsi.features.patchset import PatchSet
from plant_hsi.modeling.step import fingerprint, jsonable, relative_uri

__all__ = ["BandImportanceStep"]


class BandImportanceStep(Step):
    """Aggregate attention gates and gradient saliency into per-band scores.

    Params
        method : "both" | "attention" | "gradient" (default both)
        top_k : int — bands written to ``selected_bands.json`` (default 10)
        batch_size : int

    Outputs
        OTHER — ``wavelength_importance.csv``, ``importance.svg``,
        ``selected_bands.json``
    """

    name = "explain.wavelength"
    input_kinds = [ArtifactKind.MODEL, ArtifactKind.OTHER]
    output_kinds = [ArtifactKind.OTHER]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        model_art = next((a for a in inputs if a.kind == ArtifactKind.MODEL), None)
        patch_art = next(
            (a for a in inputs if (a.metadata or {}).get("format") == "patchset"), None
        )
        if model_art is None or patch_art is None:
            raise ValueError("explain.wavelength expects a MODEL and a patchset input")

        root = out_dir.parent
        bundle = load_bundle(model_art.resolve(root))
        ps = PatchSet.load(patch_art.resolve(root))
        bands = int(ps.patches.shape[1])
        batch_size = int(params.get("batch_size", 64))
        method = str(params.get("method", "both"))
        top_k = int(params.get("top_k", 10))
        if top_k > bands:
            top_k = bands

        attention = attention_importance(bundle, ps.patches, batch_size=batch_size)
        if method == "attention":
            saliency = np.zeros(bands, dtype=np.float32)
            importance = combine_importance(attention, attention)
        elif method == "gradient":
            saliency = gradient_saliency(bundle, ps.patches, batch_size=batch_size)
            importance = combine_importance(saliency, saliency)
        else:
            saliency = gradient_saliency(bundle, ps.patches, batch_size=batch_size)
            importance = combine_importance(attention, saliency)

        band_indices = ps.band_indices or list(range(bands))
        wavelength = ps.wavelength_nm
        selected = top_bands(importance, top_k)

        fp = fingerprint(self.name, inputs, params, seed)
        base = f"{model_art.name}_wavelength_{fp}"
        csv_path = out_dir / f"{base}_importance.csv"
        svg_path = out_dir / f"{base}_importance.svg"
        json_path = out_dir / f"{base}_selected_bands.json"

        write_importance_csv(csv_path, band_indices, wavelength, attention, saliency, importance)
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(importance_svg(wavelength, importance, selected), encoding="utf-8")
        selected_names = [
            {
                "band_index": int(band_indices[i]) if i < len(band_indices) else i,
                "wavelength_nm": float(wavelength[i]) if i < len(wavelength) else None,
                "importance": float(importance[i]),
            }
            for i in selected
        ]
        json_path.write_text(
            json.dumps(
                jsonable(
                    {
                        "method": method,
                        "top_k": top_k,
                        "bands": selected_names,
                        "selected_band_indices": [
                            int(band_indices[i]) if i < len(band_indices) else i for i in selected
                        ],
                    }
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return [
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{base}_importance",
                uri=relative_uri(csv_path, root),
                parents=[model_art.id, patch_art.id],
                metadata=jsonable(
                    {
                        "format": "wavelength_importance",
                        "method": method,
                        "top_bands": selected_names,
                        "svg_uri": relative_uri(svg_path, root),
                        "selected_uri": relative_uri(json_path, root),
                    }
                ),
            ),
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{base}_importance_svg",
                uri=relative_uri(svg_path, root),
                parents=[model_art.id, patch_art.id],
                metadata={"format": "svg"},
            ),
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{base}_selected_bands",
                uri=relative_uri(json_path, root),
                parents=[model_art.id, patch_art.id],
                metadata={"format": "json"},
            ),
        ]
