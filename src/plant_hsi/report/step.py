"""``report.build``: assemble a self-contained HTML report + run manifest."""

from __future__ import annotations

import contextlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from plant_hsi.core.artifact import ArtifactKind, ArtifactRef
from plant_hsi.core.db import MetaStore
from plant_hsi.core.registry import Step
from plant_hsi.modeling.step import fingerprint, jsonable, relative_uri
from plant_hsi.report.builder import build_report


class BuildReportStep(Step):
    """Render an HTML report for a trained MODEL artifact.

    Reads the model card / metrics / predictions produced by ``ml.train``
    (via artifact metadata, with file fallbacks), plus the project's run and
    artifact tables from SQLite.
    """

    name = "report.build"
    input_kinds = [ArtifactKind.MODEL]
    output_kinds = [ArtifactKind.REPORT, ArtifactKind.OTHER]

    def _execute(
        self,
        inputs: list[ArtifactRef],
        out_dir: Path,
        params: dict[str, Any],
        seed: int,
    ) -> list[ArtifactRef]:
        model_art = next((a for a in inputs if a.kind == ArtifactKind.MODEL), None)
        if model_art is None:
            raise ValueError("report.build expects a MODEL input")

        root = out_dir.parent
        meta = model_art.metadata or {}
        card = self._card(model_art, root, meta)
        metrics_payload = self._metrics_payload(model_art, root, meta, card)
        predictions = self._predictions(root, meta)

        project = self._project_info(root)
        runs, artifacts = self._snapshot(root)

        title = str(params.get("title") or f"{project.get('name', 'PHS')} · 传统建模报告")
        html_text = build_report(
            project=project,
            model=card,
            metrics_payload=metrics_payload,
            predictions=predictions,
            runs=runs,
            artifacts=artifacts,
            title=title,
        )

        fp = fingerprint(self.name, inputs, params, seed)
        report_path = out_dir / f"{model_art.name}_report_{fp}.html"
        report_path.write_text(html_text, encoding="utf-8")

        manifest = jsonable(
            {
                "schema_version": "1.0",
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "project": project,
                "model": {
                    "artifact_id": model_art.id,
                    "name": model_art.name,
                    "uri": model_art.uri,
                    "hash": model_art.hash,
                    "card": card,
                },
                "runs": runs,
                "artifacts": artifacts,
                "paths": {
                    "project_root": str(root),
                    "relative": True,
                    "redirect_note": (
                        "换机器复现时替换 paths.project_root，保持 artifacts[].uri "
                        "相对路径不变即可定位数据与产物。"
                    ),
                },
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
            }
        )
        manifest_path = out_dir / f"{model_art.name}_run_manifest_{fp}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        run_ids = [str(r.get("id", "")) for r in runs]
        return [
            ArtifactRef(
                id="",
                kind=ArtifactKind.REPORT,
                name=f"{model_art.name}_report_{fp}",
                uri=relative_uri(report_path, root),
                parents=[model_art.id],
                metadata=jsonable(
                    {
                        "format": "html",
                        "model_id": model_art.id,
                        "task": card.get("task"),
                        "algorithm": card.get("algorithm"),
                        "metrics": (metrics_payload.get("metrics") or {}),
                        "run_ids": run_ids,
                        "manifest_uri": relative_uri(manifest_path, root),
                    }
                ),
            ),
            ArtifactRef(
                id="",
                kind=ArtifactKind.OTHER,
                name=f"{model_art.name}_run_manifest_{fp}",
                uri=relative_uri(manifest_path, root),
                parents=[model_art.id],
                metadata={"format": "run_manifest", "run_ids": run_ids},
            ),
        ]

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _card(model_art: ArtifactRef, root: Path, meta: dict[str, Any]) -> dict[str, Any]:
        card = meta.get("card")
        if isinstance(card, dict) and card:
            return card
        card_uri = meta.get("card_uri")
        candidates: list[Path] = []
        if card_uri:
            candidates.append(root / str(card_uri))
        model_path = model_art.resolve(root)
        candidates.append(
            model_path.with_name(model_path.name.replace("_model.pkl", "_model_card.json"))
        )
        for path in candidates:
            if path.exists():
                return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        raise ValueError("model card not found in artifact metadata or alongside the model file")

    @staticmethod
    def _metrics_payload(
        model_art: ArtifactRef,
        root: Path,
        meta: dict[str, Any],
        card: dict[str, Any],
    ) -> dict[str, Any]:
        metrics_uri = meta.get("metrics_uri")
        if metrics_uri:
            path = root / str(metrics_uri)
            if path.exists():
                return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        return {
            "task": card.get("task"),
            "algorithm": card.get("algorithm"),
            "metrics": card.get("metrics") or meta.get("metrics") or {},
            "cv": card.get("cv"),
            "split": card.get("split"),
            "feature_names": card.get("feature_names"),
            "classes": card.get("classes"),
            "n_train": card.get("n_train"),
            "n_test": card.get("n_test"),
        }

    @staticmethod
    def _predictions(root: Path, meta: dict[str, Any]) -> pd.DataFrame | None:
        uri = meta.get("predictions_uri")
        if not uri:
            return None
        path = root / str(uri)
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001 - report degrades gracefully
            return None

    @staticmethod
    def _project_info(root: Path) -> dict[str, Any]:
        path = root / "project.json"
        if path.exists():
            return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        return {"project_id": "", "name": root.name, "schema_version": "1.0"}

    @staticmethod
    def _snapshot(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        db = MetaStore(root / "project.db")
        try:
            runs = [_decode(r, {"params", "output_artifacts"}) for r in db.list_runs()]
            artifacts = [_decode(a, {"parents", "metadata"}) for a in db.list_artifacts()]
        finally:
            db.close()
        return runs, artifacts


def _decode(row: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    out = dict(row)
    for field in fields:
        value = out.get(field)
        if isinstance(value, str) and value:
            with contextlib.suppress(json.JSONDecodeError):
                out[field] = json.loads(value)
    return out
