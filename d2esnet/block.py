"""Adaptive hybrid ANN-SNN block."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .evidence import EvidenceStateNeuron
from .layers import ElasticANN
from .router import SceneContextRouter, route_regularization


class D2ESBlock(nn.Module):
    """Dynamically routes channels between continuous and spiking paths."""

    def __init__(
        self,
        channels: int,
        ratios: Sequence[float] = (0.25, 0.5, 0.75),
        fixed_ratio: float = 0.5,
        time_steps: int = 3,
        use_router: bool = True,
        use_evidence: bool = True,
        use_recovery: bool = True,
    ) -> None:
        super().__init__()
        if channels < 4:
            raise ValueError("channels must be at least 4")
        self.channels = channels
        self.ratios = sorted({float(ratio) for ratio in ratios})
        self.widths = tuple(
            max(1, min(channels - 1, round(channels * ratio)))
            for ratio in self.ratios
        )
        self.fixed_width = max(
            1, min(channels - 1, round(channels * fixed_ratio))
        )
        self.use_router = use_router
        self.router = SceneContextRouter(self.ratios) if use_router else None
        self.ann = ElasticANN(channels)
        self.snn = EvidenceStateNeuron(
            channels,
            time_steps=time_steps,
            use_evidence=use_evidence,
        )
        if not use_recovery:
            self.snn.weak_gain.requires_grad_(False)
            self.snn.weak_gain.data.zero_()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.last_route_probabilities = None
        self.last_route_indices = None
        self.last_selected_ratio = None

    def _run_width(self, x: torch.Tensor, width: int) -> torch.Tensor:
        continuous, spiking = x[:, :width], x[:, width:]
        spiking = self.snn(spiking, channel_offset=width)
        fused = self.fusion(torch.cat((self.ann(continuous, width), spiking), 1))
        return x + self.gamma * fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_router or self.router is None:
            output = self._run_width(x, self.fixed_width)
            self.last_selected_ratio = self.fixed_width / self.channels
            return output

        route = self.router(x)
        self.last_route_probabilities = route.probabilities
        if self.training:
            candidates = torch.stack(
                [self._run_width(x, width) for width in self.widths],
                dim=1,
            )
            output = (
                candidates * route.probabilities[:, :, None, None, None]
            ).sum(dim=1)
        else:
            index = route.probabilities.argmax(dim=1)
            if not bool((index == index[0]).all()):
                raise RuntimeError(
                    "This reference block expects a single routed width per "
                    "inference batch."
                )
            option = int(index[0])
            self.last_route_indices = index
            self.last_selected_ratio = self.widths[option] / self.channels
            output = self._run_width(x, self.widths[option])
        return output

    def route_regularization(self) -> torch.Tensor:
        if self.last_route_probabilities is None:
            raise RuntimeError("Run a training forward pass first")
        return route_regularization(self.last_route_probabilities)

