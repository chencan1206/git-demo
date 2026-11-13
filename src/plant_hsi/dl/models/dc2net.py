"""DC²Net — deformable spatial conv + dilated spectral conv + band attention.

The implementation is intentionally compact (a few hundred thousand
parameters) and pure PyTorch:

* **deformable spatial convolution** — offsets are predicted from the feature
  map and used with :func:`torch.nn.functional.grid_sample`; it degenerates to
  a plain 3×3 convolution when the offsets are zero.
* **dilated spectral convolution** — a 3-D convolution whose kernel only spans
  the band axis, with a configurable dilation to widen the spectral receptive
  field.
* **band/channel attention** — a squeeze-excite gate over the band axis; the
  gates are also exported for wavelength importance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

__all__ = ["DC2Net", "DC2NetConfig", "count_parameters"]


@dataclass
class DC2NetConfig:
    in_bands: int = 40
    num_classes: int = 2
    spectral_channels: int = 8
    spectral_kernel: int = 5
    spectral_dilation: int = 2
    feat_channels: int = 32
    hidden_channels: int = 64
    dropout: float = 0.2
    attention_reduction: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DC2NetConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class DeformableConv2d(nn.Module):
    """2-D deformable convolution (DCN v1) implemented with ``grid_sample``.

    A regular conv predicts ``(dx, dy)`` offsets for every kernel tap; the
    feature map is bilinearly sampled at the deformed positions and a 1×1
    convolution over ``in_channels × k²`` taps produces the output.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = int(kernel_size)
        self.pad = kernel_size // 2
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size, kernel_size, padding=self.pad
        )
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        # Start as a standard convolution: zero offsets, zero bias in the
        # offset branch (tanh(0) = 0 keeps the sampling grid on pixel centres).
        nn.init.zeros_(self.offset_conv.weight)
        assert self.offset_conv.bias is not None
        nn.init.zeros_(self.offset_conv.bias)

    def _base_grid(self, height: int, width: int, x: torch.Tensor) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1)  # (H, W, 2), (x, y) order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        k = self.kernel_size
        kk = k * k
        # tanh bounds the offsets to ±0.5 in normalised coordinates (≈ ±7 px
        # for a 15-px patch) which keeps sampling stable.
        offsets = 0.5 * torch.tanh(self.offset_conv(x))
        offsets = offsets.view(b, kk, 2, h, w).permute(0, 3, 4, 1, 2)  # (B,H,W,kk,2)
        base = self._base_grid(h, w, x).view(1, h, w, 1, 2)
        grid = (base + offsets).reshape(b, h * kk, w, 2)
        sampled = functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        sampled = sampled.view(b, c, kk, h, w).permute(0, 3, 4, 1, 2)
        flat = sampled.reshape(b * h * w, c * kk)
        out = flat @ self.weight.reshape(self.weight.shape[0], c * kk).t()
        out = out + self.bias
        return out.view(b, h, w, -1).permute(0, 3, 1, 2)


class BandChannelAttention(nn.Module):
    """Squeeze-excite gate over the band axis.

    Input ``(N, C, B, H, W)``; the gate is computed per (channel, band) after
    spatial average pooling and stored in :attr:`last_attention` so the
    explainability step can aggregate it into per-band importance.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(2, channels // max(1, reduction))
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(3, 4)).permute(0, 2, 1)  # (N, B, C)
        gate = torch.sigmoid(self.fc2(functional.gelu(self.fc1(pooled)))).permute(
            0, 2, 1
        )  # (N,C,B)
        self.last_attention = gate.detach()
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class DC2Net(nn.Module):
    """Hyperspectral patch classifier."""

    def __init__(self, config: DC2NetConfig) -> None:
        super().__init__()
        self.config = config
        c = config.spectral_channels
        pad = config.spectral_kernel // 2 * config.spectral_dilation
        self.spectral = nn.Conv3d(
            1,
            c,
            kernel_size=(config.spectral_kernel, 1, 1),
            padding=(pad, 0, 0),
            dilation=(config.spectral_dilation, 1, 1),
        )
        self.bn_spectral = nn.BatchNorm3d(c)
        self.attention = BandChannelAttention(c, config.attention_reduction)

        self.deform1 = DeformableConv2d(c * config.in_bands, config.feat_channels, 3)
        self.bn1 = nn.BatchNorm2d(config.feat_channels)
        self.deform2 = DeformableConv2d(config.feat_channels, config.feat_channels, 3)
        self.bn2 = nn.BatchNorm2d(config.feat_channels)
        self.conv3 = nn.Conv2d(config.feat_channels, config.hidden_channels, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(config.hidden_channels)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_channels, config.num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Any:
        n, bands, h, w = x.shape
        h3 = x.unsqueeze(1)  # (N, 1, B, H, W)
        h3 = functional.gelu(self.bn_spectral(self.spectral(h3)))
        h3 = self.attention(h3)
        attn = self.attention.last_attention
        flat = h3.reshape(n, -1, h, w)
        flat = functional.gelu(self.bn1(self.deform1(flat)))
        flat = functional.gelu(self.bn2(self.deform2(flat)))
        flat = functional.gelu(self.bn3(self.conv3(flat)))
        pooled = flat.mean(dim=(2, 3))
        logits = self.head(self.dropout(pooled))
        if return_attention:
            band_attn = attn.mean(dim=1) if attn is not None else torch.zeros(n, bands)
            return logits, band_attn
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
