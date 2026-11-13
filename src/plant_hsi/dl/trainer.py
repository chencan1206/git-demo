"""Deterministic training loop for DC²Net with early stopping and resume."""

from __future__ import annotations

import contextlib
import copy
import random
import resource
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from plant_hsi.dl.dataset import PatchDataset, make_loader
from plant_hsi.dl.models.dc2net import DC2Net, DC2NetConfig, count_parameters
from plant_hsi.modeling.metrics import classification_metrics

__all__ = [
    "DLConfig",
    "DLTrainResult",
    "MemoryBudgetError",
    "check_memory_budget",
    "estimate_peak_memory_bytes",
    "resolve_device",
    "train_dc2net",
]

MB = 1024 * 1024


class MemoryBudgetError(RuntimeError):
    """Raised *before* allocating when the batch is unlikely to fit in VRAM."""


@dataclass
class DLConfig:
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    device: str = "auto"
    spectral_channels: int = 8
    spectral_kernel: int = 5
    spectral_dilation: int = 2
    feat_channels: int = 32
    hidden_channels: int = 64
    dropout: float = 0.2
    attention_reduction: int = 4
    deterministic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DLTrainResult:
    model: DC2Net
    config: DLConfig
    classes: list[str]
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    test_metrics: dict[str, Any] = field(default_factory=dict)
    train_accuracy: float = 0.0
    n_parameters: int = 0
    peak_memory_mb: float = 0.0
    device: str = "cpu"
    epochs_run: int = 0
    stop_reason: str = ""
    band_attention: list[float] = field(default_factory=list)


def resolve_device(name: str) -> torch.device:
    if name in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def estimate_peak_memory_bytes(
    batch_size: int, bands: int, patch_size: int, spectral_channels: int = 8
) -> int:
    """Conservative activation estimate for the deformable branch.

    The deformable convolution materialises ``C × B × k²`` sampled maps per
    patch; we budget four copies (forward + backward) of that tensor.
    """
    taps = 9
    per_sample = spectral_channels * bands * taps * patch_size * patch_size * 4
    return int(batch_size * per_sample * 4)


def check_memory_budget(
    total_bytes: int,
    batch_size: int,
    bands: int,
    patch_size: int,
    spectral_channels: int = 8,
    safety: float = 0.85,
) -> None:
    """Raise :class:`MemoryBudgetError` with a concrete suggestion when the
    requested batch will not fit.  Never allocates — safe to call up front."""
    if total_bytes <= 0:
        return
    estimate = estimate_peak_memory_bytes(batch_size, bands, patch_size, spectral_channels)
    if estimate <= total_bytes * safety:
        return
    per_sample = max(1, estimate // max(1, batch_size))
    suggested = max(1, int(total_bytes * safety // per_sample))
    raise MemoryBudgetError(
        f"batch_size={batch_size} 预计需要 {estimate / MB:.0f} MB 显存，"
        f"超过可用 {total_bytes / MB:.0f} MB；建议 batch_size ≤ {suggested}，"
        f"或减小 patch_size / 减少波段数。"
    )


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / MB)
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss_kb / 1024.0)


def _evaluate(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    trues: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            losses.append(float(criterion(logits, yb).item()) * xb.shape[0])
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    n = max(1, sum(len(v) for v in trues))
    return sum(losses) / n, np.concatenate(trues), np.concatenate(preds)


def train_dc2net(
    patches: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    classes: list[str],
    config: DLConfig,
    seed: int = 0,
    band_mean: np.ndarray | None = None,
    band_std: np.ndarray | None = None,
    resume: dict[str, Any] | None = None,
    on_epoch: Callable[[int, dict[str, Any]], None] | None = None,
) -> DLTrainResult:
    """Train DC²Net on normalised patches.

    ``splits`` maps ``train`` / ``val`` / ``test`` to integer indices.  When
    ``val`` is empty the training loss drives early stopping and the test set
    falls back to whatever split is available.
    """
    device = resolve_device(config.device)
    n_classes = max(1, len(classes))
    bands = int(patches.shape[1])
    patch_size = int(patches.shape[2])
    mean = np.zeros(bands, dtype=np.float32) if band_mean is None else band_mean
    std = np.ones(bands, dtype=np.float32) if band_std is None else band_std
    normalised = ((patches - mean[:, None, None]) / std[:, None, None]).astype(np.float32)

    train_idx = np.asarray(splits.get("train", np.array([], dtype=int)), dtype=int)
    val_idx = np.asarray(splits.get("val", np.array([], dtype=int)), dtype=int)
    test_idx = np.asarray(splits.get("test", np.array([], dtype=int)), dtype=int)
    if train_idx.size == 0:
        raise ValueError("training split is empty")
    if test_idx.size == 0:
        test_idx = val_idx if val_idx.size else train_idx

    if device.type == "cuda":
        total = int(torch.cuda.get_device_properties(device).total_memory)
        check_memory_budget(total, config.batch_size, bands, patch_size, config.spectral_channels)
        torch.cuda.reset_peak_memory_stats(device)

    _seed_everything(int(seed), config.deterministic)

    model_config = DC2NetConfig(
        in_bands=bands,
        num_classes=n_classes,
        spectral_channels=config.spectral_channels,
        spectral_kernel=config.spectral_kernel,
        spectral_dilation=config.spectral_dilation,
        feat_channels=config.feat_channels,
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
        attention_reduction=config.attention_reduction,
    )
    model = DC2Net(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))

    history: list[dict[str, Any]] = []
    start_epoch = 0
    if resume:
        state = resume.get("state_dict")
        if state is not None:
            model.load_state_dict(state)
        opt_state = resume.get("optimizer_state_dict")
        if opt_state is not None:
            with contextlib.suppress(ValueError, KeyError):
                optimizer.load_state_dict(opt_state)
        history = list(resume.get("history") or [])
        start_epoch = int(resume.get("epoch") or len(history))

    base_dataset = PatchDataset(normalised, labels, augment=False, seed=int(seed))
    val_loader = make_loader(
        torch.utils.data.Subset(base_dataset, val_idx.tolist()),
        max(1, config.batch_size),
        shuffle=False,
        seed=int(seed),
    )
    test_loader = make_loader(
        torch.utils.data.Subset(base_dataset, test_idx.tolist()),
        max(1, config.batch_size),
        shuffle=False,
        seed=int(seed),
    )
    train_eval_loader = make_loader(
        torch.utils.data.Subset(base_dataset, train_idx.tolist()),
        max(1, config.batch_size),
        shuffle=False,
        seed=int(seed),
    )
    # Training patches get their own dataset so augmentation can be re-seeded
    # per epoch without touching the evaluation splits.
    train_dataset = PatchDataset(
        normalised[train_idx], labels[train_idx], augment=True, seed=int(seed)
    )
    train_loader = make_loader(train_dataset, config.batch_size, shuffle=True, seed=int(seed))

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = -1
    patience_left = int(config.patience)
    stop_reason = "epochs"
    epochs_run = 0

    for local_epoch in range(max(1, config.epochs)):
        epoch = start_epoch + local_epoch
        train_dataset.set_epoch(epoch)
        model.train()
        running = 0.0
        seen = 0
        started = time.perf_counter()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += float(loss.item()) * xb.shape[0]
            seen += int(xb.shape[0])
        train_loss = running / max(1, seen)
        train_acc = _accuracy(model, train_eval_loader, device)
        if val_idx.size:
            val_loss, _, _ = _evaluate(model, val_loader, criterion, device)
        else:
            val_loss = train_loss
        elapsed = time.perf_counter() - started
        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": elapsed,
        }
        history.append(row)
        epochs_run += 1
        if on_epoch is not None:
            on_epoch(epoch + 1, row)

        if val_loss < best_val - 1e-9:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = int(config.patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                stop_reason = "early_stop"
                break

    model.load_state_dict(best_state)
    _, y_true, y_pred = _evaluate(model, test_loader, criterion, device)
    metrics = classification_metrics(y_true, y_pred, labels=list(range(n_classes)))
    _, train_true, train_pred = _evaluate(model, train_eval_loader, criterion, device)
    train_accuracy = float((train_true == train_pred).mean()) if train_true.size else 0.0
    band_attn = _band_attention(model, normalised[test_idx], device)

    return DLTrainResult(
        model=model,
        config=config,
        classes=list(classes),
        history=history,
        best_epoch=best_epoch + 1,
        best_val_loss=best_val,
        test_metrics=metrics,
        train_accuracy=train_accuracy,
        n_parameters=count_parameters(model),
        peak_memory_mb=_peak_memory_mb(device),
        device=str(device),
        epochs_run=epochs_run,
        stop_reason=stop_reason,
        band_attention=[float(v) for v in band_attn],
    )


def _accuracy(model: nn.Module, loader: Any, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    model.train()
    return float(correct / total) if total else 0.0


def _band_attention(
    model: DC2Net, patches: np.ndarray, device: torch.device, batch: int = 64
) -> np.ndarray:
    model.eval()
    sums: np.ndarray | None = None
    count = 0
    with torch.no_grad():
        for start in range(0, len(patches), batch):
            xb = torch.from_numpy(patches[start : start + batch]).to(device)
            _, attn = model(xb, return_attention=True)
            arr = attn.mean(dim=0).cpu().numpy()
            sums = arr if sums is None else sums + arr
            count += 1
    model.train()
    if sums is None or count == 0:
        return np.full(patches.shape[1], 0.5, dtype=np.float32)
    return (sums / count).astype(np.float32)
