"""Artifact reference: a file or dataset produced by a step.

Stage 0 only supports file artifacts; Zarr/GeoTIFF/Parquet containers will
be added in later stages via a ``kind`` tag and a ``uri`` relative to the
project directory.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ArtifactKind(enum.StrEnum):
    RAW = "raw"  # imported external dataset (copied or referenced)
    CUBE = "cube"  # hyperspectral cube (Zarr or similar)
    MASK = "mask"  # segmentation / classification map
    SPECTRA = "spectra"  # extracted spectra table (Parquet/CSV)
    MODEL = "model"  # trained model file + model card
    REPORT = "report"  # HTML / PDF report
    OTHER = "other"


def compute_file_hash(path: Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    """Return ``algo:hexdigest`` for a file. Empty / missing files return ``""``."""
    if not path.is_file():
        return ""
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return f"{algo}:{h.hexdigest()}"


@dataclass
class ArtifactRef:
    """Immutable-ish reference to an artifact that lives under a project."""

    id: str  # stable ID, e.g. "art_0001"
    kind: ArtifactKind
    name: str
    uri: str  # relative to project root, e.g. "artifacts/ab/cd1234.zarr"
    hash: str = ""  # "algo:hex" or "" if unknown/skipped
    size: int = 0
    parents: list[str] = field(default_factory=list)  # upstream artifact IDs
    created_by_run: str = ""  # run ID that produced it
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ IO
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRef:
        d = dict(data)
        d["kind"] = ArtifactKind(d["kind"])
        return cls(**d)

    def resolve(self, project_root: Path) -> Path:
        return project_root / self.uri

    def exists(self, project_root: Path) -> bool:
        return self.resolve(project_root).exists()
