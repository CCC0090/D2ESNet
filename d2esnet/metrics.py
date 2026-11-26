"""Efficiency measurements specific to dynamic routing."""

from __future__ import annotations

import torch
from torch.utils.flop_counter import FlopCounterMode


def routed_flops(model: torch.nn.Module, image: torch.Tensor) -> int:
    """Count FLOPs along the route actually selected by the model."""
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        model(image)
    return int(counter.get_total_flops())


def average_snn_channel_ratio(model: torch.nn.Module) -> float:
    """Average SNN-channel share across all active D2ES blocks."""
    from .block import D2ESBlock

    blocks = [
        module for module in model.modules()
        if isinstance(module, D2ESBlock)
    ]
    if not blocks:
        raise RuntimeError("No D2ESBlock found")
    ratios = []
    for block in blocks:
        if block.last_route_indices is None:
            raise RuntimeError("Run an inference forward pass first")
        selected = int(block.last_route_indices[0].item())
        ratios.append(1.0 - block.widths[selected] / block.channels)
    return float(sum(ratios) / len(ratios))

