"""Stateful spiking representation driven by scattering evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .layers import BinarySpike
from .ops import directional_scattering, multiscale_contrast


@dataclass
class SpikingState:
    spikes: torch.Tensor
    membrane: torch.Tensor
    evidence: torch.Tensor
    fired: torch.Tensor
    residual: torch.Tensor
    recovery_gate: torch.Tensor
    recovery_residual: torch.Tensor
    output: torch.Tensor


def recovery_gate(
    membrane: torch.Tensor,
    fired: torch.Tensor,
    evidence: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute residual potential, effective gate, and recovery residual."""
    residual = torch.tanh(
        membrane.clamp_min(0.0) / (theta + 1e-6)
    )
    effective_gate = (1.0 - fired) * evidence.amax(dim=1)[:, None]
    return residual, effective_gate, effective_gate * residual


class EvidenceStateNeuron(nn.Module):
    """Multi-step stateful pulsed representation of selected SNN channels."""

    def __init__(
        self,
        channels: int,
        time_steps: int = 3,
        weak_gain: float = 0.15,
        use_evidence: bool = True,
    ) -> None:
        super().__init__()
        if not 1 <= time_steps <= 4:
            raise ValueError("time_steps must be in [1, 4]")
        self.channels = channels
        self.time_steps = time_steps
        self.use_evidence = use_evidence
        self.leak = 0.8
        self.spike_conv = nn.Parameter(torch.empty(channels, 1, 3, 3))
        nn.init.kaiming_normal_(
            self.spike_conv, mode="fan_out", nonlinearity="relu"
        )
        self.evidence_encoder = (
            nn.Sequential(
                nn.Conv2d(4, 12, 1),
                nn.SiLU(),
                nn.Conv2d(12, time_steps, 1),
                nn.Sigmoid(),
            )
            if use_evidence
            else None
        )
        self.raw_theta = nn.Parameter(
            torch.tensor(0.7).logit()
        )
        self.weak_gain = nn.Parameter(torch.tensor(weak_gain))
        self.last_state: SpikingState | None = None

    def _evidence(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = x.abs().mean(dim=1, keepdim=True)
        features = torch.cat(
            (
                multiscale_contrast(magnitude, 3),
                multiscale_contrast(magnitude, 5),
                multiscale_contrast(magnitude, 7),
                directional_scattering(magnitude),
            ),
            dim=1,
        )
        return self.evidence_encoder(features)

    def forward(
        self,
        x: torch.Tensor,
        channel_offset: int = 0,
    ) -> torch.Tensor:
        channels = x.shape[1]
        current = F.relu(x)
        theta = F.softplus(self.raw_theta) + 0.01
        membrane = torch.zeros_like(x)
        fired = torch.zeros_like(x)
        spikes = []
        if self.use_evidence:
            evidence = self._evidence(x)
        else:
            evidence = torch.ones(
                x.shape[0],
                self.time_steps,
                x.shape[2],
                x.shape[3],
                dtype=x.dtype,
                device=x.device,
            )
        for index in range(self.time_steps):
            step_evidence = evidence[:, index : index + 1]
            pre_reset = self.leak * membrane + current * step_evidence
            pulse = BinarySpike.apply(pre_reset - theta)
            membrane = pre_reset - theta * pulse.detach()
            fired = torch.maximum(fired, pulse.detach())
            spikes.append(pulse)
        spike_stack = torch.stack(spikes, dim=0)
        spike_feature = F.conv2d(
            spike_stack.mean(dim=0),
            self.spike_conv[channel_offset : channel_offset + channels],
            padding=1,
            groups=channels,
        )
        residual, gate, gated = recovery_gate(
            membrane, fired, evidence, theta
        )
        output = spike_feature + self.weak_gain * gated
        self.last_state = SpikingState(
            spikes=spike_stack.detach(),
            membrane=membrane.detach(),
            evidence=evidence.detach(),
            fired=fired.detach(),
            residual=residual.detach(),
            recovery_gate=gate.detach(),
            recovery_residual=(self.weak_gain * gated).detach(),
            output=output.detach(),
        )
        return output

