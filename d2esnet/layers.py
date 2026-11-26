"""Basic layers for adaptive continuous and spiking paths."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Module):
    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class BinarySpike(torch.autograd.Function):
    """Binary forward pulse with a smooth surrogate derivative."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_output / (1.0 + x.abs()).square()


class ElasticANN(nn.Module):
    """Width-selectable continuous residual path."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw3 = nn.Parameter(torch.empty(channels, 1, 3, 3))
        self.pw3 = nn.Parameter(torch.empty(channels, channels, 1, 1))
        self.dw5 = nn.Parameter(torch.empty(channels, 1, 5, 5))
        self.pw5 = nn.Parameter(torch.empty(channels, channels, 1, 1))
        for weight in (self.dw3, self.pw3, self.dw5, self.pw5):
            nn.init.kaiming_normal_(
                weight, mode="fan_out", nonlinearity="relu"
            )

    def forward(self, x: torch.Tensor, width: int) -> torch.Tensor:
        z = F.conv2d(x, self.dw3[:width], padding=1, groups=width)
        z = F.silu(F.conv2d(z, self.pw3[:width, :width]))
        z = F.conv2d(z, self.dw5[:width], padding=2, groups=width)
        return F.silu(F.conv2d(z, self.pw5[:width, :width]))

