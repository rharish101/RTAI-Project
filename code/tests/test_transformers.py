"""Tests for the layer transformers."""
import numpy as np
import pytest
import torch
from networks import SPU, FullyConnected
from typing_extensions import Final
from verifier import DEVICE, Verifier

# How many evenly-distributed points to consider b/w upper and lower bounds for
# testing transformers
NUM_TEST_POINTS: Final = 1000


@pytest.mark.parametrize(
    "lower_bound, upper_bound",
    [
        # Simple aka non-crossing case
        (np.array([-5.0]), np.array([-1e-4])),
        (np.array([1e-4]), np.array([1e2])),
        (np.array([-5.0, 1e-4]), np.array([-1e-4, 1e2])),
        # Crossing case
        (np.array([-1e-4]), np.array([1e2])),
        (np.array([-5.0]), np.array([1e-4])),
        (np.array([-5.0, -1e-4]), np.array([1e-4, 1e2])),
    ],
)
def test_spu(lower_bound, upper_bound) -> None:
    """Test the SPU transformer.

    Args:
        lower_bound: A 1D vector of input lower bounds
        upper_bound: A 1D vector of corresponding input upper bounds
    """
    net = FullyConnected(DEVICE, 784, [10])
    layer = SPU()
    verifier = Verifier(net)

    verifier._upper_bound = [torch.from_numpy(upper_bound)]
    verifier._lower_bound = [torch.from_numpy(lower_bound)]

    # No need to use previous constraint values, so keep them empty
    verifier._upper_constraint = []
    verifier._lower_constraint = []

    verifier._analyze_spu(layer)

    inputs_np = np.linspace(lower_bound, upper_bound, num=NUM_TEST_POINTS)
    inputs = torch.from_numpy(inputs_np)  # dim: NUM_TEST_POINTS x D
    outputs = layer(inputs)  # dim: NUM_TEST_POINTS x D

    # To compensate for floating-point precision
    eps = torch.finfo(outputs.dtype).eps

    # Test bounds
    spu_upper_bound = verifier._upper_bound[-1]  # dim: D
    spu_lower_bound = verifier._lower_bound[-1]  # dim: D
    assert (outputs >= spu_lower_bound.unsqueeze(0) - eps).all()
    assert (outputs <= spu_upper_bound.unsqueeze(0) + eps).all()

    inputs_with_bias = torch.cat(
        [inputs, torch.ones(len(inputs), 1, device=DEVICE)], dim=1
    )  # NUM_TEST_POINTS x (D+1)

    # NUM_TEST_POINTS x D = NUM_TEST_POINTS x (D+1) @ (D+1) x D
    constraint_upper_bound = (
        inputs_with_bias @ verifier._upper_constraint[-1].T
    )
    constraint_lower_bound = (
        inputs_with_bias @ verifier._lower_constraint[-1].T
    )

    # Test constraints
    assert (outputs >= constraint_lower_bound - eps).all()
    assert (outputs <= constraint_upper_bound + eps).all()
