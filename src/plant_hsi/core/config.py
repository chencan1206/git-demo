"""Application-level configuration.

The config lives next to the app and does not belong to any project.
Project-specific state stays inside the project directory (project.json, project.db).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_FILENAME = "config.json"


@dataclass
class Config:
    """Persistent user preferences. Stage 0 keeps it minimal."""

    default_project_root: str = str(Path.home() / "PHS_Projects")
    recent_projects: list[str] = field(default_factory=list)
    log_level: str = "INFO"
    max_cache_bytes: int = 2 * 1024**3  # 2 GiB

    # ------------------------------------------------------------------ IO
    @classmethod
    def path(cls, root: Path | None = None) -> Path:
        if root is None:
            root = Path.home() / ".config" / "plant_hsi"
        return root / DEFAULT_FILENAME

    @classmethod
    def load(cls, root: Path | None = None) -> Config:
        p = cls.path(root)
        if not p.exists():
            cfg = cls()
            cfg.save(root)
            return cfg
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def save(self, root: Path | None = None) -> None:
        p = self.path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def load_config(root: Path | None = None) -> Config:
    """Module-level convenience wrapper around :meth:`Config.load`."""
    return Config.load(root)
