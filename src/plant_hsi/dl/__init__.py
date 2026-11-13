"""DC²Net branch: models, dataset, trainer and persistence."""

from __future__ import annotations

from plant_hsi.dl.bundle import load_bundle, predict_bundle, save_bundle
from plant_hsi.dl.models.dc2net import DC2Net, DC2NetConfig, count_parameters
from plant_hsi.dl.trainer import (
    DLConfig,
    DLTrainResult,
    MemoryBudgetError,
    check_memory_budget,
    train_dc2net,
)

__all__ = [
    "DC2Net",
    "DC2NetConfig",
    "DLConfig",
    "DLTrainResult",
    "MemoryBudgetError",
    "check_memory_budget",
    "count_parameters",
    "load_bundle",
    "predict_bundle",
    "save_bundle",
    "train_dc2net",
]
