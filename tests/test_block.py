import torch

from d2esnet import D2ESBlock, EvidenceStateNeuron


def test_spiking_neuron_shapes_and_state():
    neuron = EvidenceStateNeuron(8, time_steps=3)
    x = torch.randn(1, 8, 32, 32)
    y = neuron(x)
    assert y.shape == x.shape
    assert neuron.last_state is not None
    assert neuron.last_state.spikes.shape[:2] == (3, 1)


def test_dynamic_block_inference():
    block = D2ESBlock(channels=8, time_steps=3).eval()
    x = torch.randn(1, 8, 32, 32)
    y = block(x)
    assert y.shape == x.shape
    assert block.last_selected_ratio in {0.25, 0.50, 0.75}

