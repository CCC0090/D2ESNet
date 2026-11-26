"""Scattering operators shared by routing and spiking-evidence modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def channel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Channel-averaged absolute response."""
    return x.abs().mean(dim=1, keepdim=True)


def multiscale_contrast(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Local contrast versus an average-pooled neighborhood."""
    smooth = F.avg_pool2d(
        x,
        kernel,
        stride=1,
        padding=kernel // 2,
        count_include_pad=False,
    )
    return ((x - smooth).abs() / (smooth + 0.1)).clamp(max=4.0)


def directional_scattering(x: torch.Tensor) -> torch.Tensor:
    """Horizontal/vertical directional response of the magnitude map."""
    horizontal = F.avg_pool2d(
        x,
        (1, 5),
        stride=1,
        padding=(0, 2),
        count_include_pad=False,
    )
    vertical = F.avg_pool2d(
        x,
        (5, 1),
        stride=1,
        padding=(2, 0),
        count_include_pad=False,
    )
    return (
        (horizontal - vertical).abs()
        / (0.1 + horizontal + vertical)
    ).clamp(max=4.0)


def scattering_stack(x: torch.Tensor) -> torch.Tensor:
    """Four-channel multiscale scattering descriptor."""
    magnitude = channel_magnitude(x)
    return torch.cat(
        (
            magnitude,
            multiscale_contrast(magnitude, 3),
            multiscale_contrast(magnitude, 5),
            multiscale_contrast(magnitude, 7),
            directional_scattering(magnitude),
        ),
        dim=1,
    )

