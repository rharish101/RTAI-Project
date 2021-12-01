"""Tests for device support."""
import pytest
import torch
from networks import FullyConnected
from utils import SEED
from verifier import Verifier


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_support(device: str) -> None:
    """Test CPU/GPU support.

    GPU support is only checked if this machine has CUDA support.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA support on this machine")

    # Add a hidden layer to test the SPU activation
    net = FullyConnected(device, 28, [5, 10])
    verifier = Verifier(net, device=device)

    # Keep inputs on CPU to see if verifier automatically moves them to the GPU
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    inputs = torch.rand(784, generator=rng, device="cpu")
    true_lbl = torch.randint(10, [], generator=rng, device="cpu")
    eps = torch.rand([], generator=rng, device="cpu")

    verifier.analyze(inputs, true_lbl, eps)