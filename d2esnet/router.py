"""Scene-context-aware discrete ANN-SNN routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .ops import directional_scattering, multiscale_contrast


@dataclass
class RouteOutput:
    probabilities: torch.Tensor
    density: torch.Tensor
    difficulty: torch.Tensor
    descriptor: torch.Tensor


class SceneContextRouter(nn.Module):
    """Predict ANN-channel ratios from unlabeled scene statistics."""

    def __init__(self, ratios: Sequence[float] = (0.25, 0.5, 0.75)):
        super().__init__()
        self.ratios = tuple(float(value) for value in ratios)
        self.context = nn.Sequential(
            nn.Conv2d(4, 12, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(12, 2, 1),
            nn.Sigmoid(),
        )
        self.policy = nn.Sequential(
            nn.Linear(6, 16),
            nn.SiLU(),
            nn.Linear(16, len(self.ratios)),
        )
        nn.init.normal_(self.policy[-1].weight, std=0.02)
        nn.init.zeros_(self.policy[-1].bias)

    def forward(self, x: torch.Tensor) -> RouteOutput:
        magnitude = x.abs().mean(dim=1, keepdim=True)
        c3 = multiscale_contrast(magnitude, 3)
        c7 = multiscale_contrast(magnitude, 7)
        direction = directional_scattering(magnitude)
        maps = self.context(torch.cat((magnitude, c3, c7, direction), 1))
        density, difficulty = maps[:, 0:1], maps[:, 1:2]
        descriptor = torch.cat(
            (
                density.mean((2, 3)),
                difficulty.mean((2, 3)),
                density.amax((2, 3)),
                difficulty.amax((2, 3)),
                c3.mean((2, 3)),
                direction.mean((2, 3)),
            ),
            dim=1,
        )
        probabilities = self.policy(descriptor).softmax(dim=1)
        return RouteOutput(probabilities, density, difficulty, descriptor)


def route_regularization(
    probabilities: torch.Tensor,
    balance_weight: float = 0.01,
    entropy_weight: float = 0.001,
) -> torch.Tensor:
    """Balance route usage while discouraging premature one-hot collapse."""
    mean_probability = probabilities.mean(dim=0)
    balance = (
        mean_probability - 1.0 / probabilities.shape[1]
    ).square().sum()
    entropy = -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(dim=1).mean()
    return balance_weight * balance + entropy_weight * entropy

