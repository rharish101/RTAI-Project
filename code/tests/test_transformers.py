"""Tests for the layer transformers."""
import numpy as np
import pytest
import torch
from networks import SPU, Normalization
from typing_extensions import Final
from utils import DTYPE, EPS
from verifier import DEVICE, Verifier

# How many evenly-distributed points to consider b/w upper and lower bounds for
# testing transformers
NUM_TEST_POINTS: Final = int(1e5)

# Common test cases for lower and upper bounds
TEST_BOUNDS: Final = [
    # Simple aka non-crossing case
    (np.array([-5.0]), np.array([-1e-4])),
    (np.array([1e-4]), np.array([1e2])),
    (np.array([-5.0, 1e-4]), np.array([-1e-4, 1e2])),
    # Crossing case
    (np.array([-1e-4]), np.array([1e2])),
    (np.array([-5.0]), np.array([1e-4])),
    (np.array([-5.0, -1e-4]), np.array([1e-4, 1e2])),
]


def _test_helper(
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    layer: torch.nn.Module,
) -> None:
    """Test the transformer for the given layer.

    Args:
        lower_bound: A 1D vector of input lower bounds
        upper_bound: A 1D vector of corresponding input upper bounds
        layer: The layer to test
    """
    verifier = Verifier([]).to(device=DEVICE, dtype=DTYPE)

    verifier._upper_bound = [
        torch.from_numpy(upper_bound).to(device=DEVICE, dtype=DTYPE)
    ]
    verifier._lower_bound = [
        torch.from_numpy(lower_bound).to(device=DEVICE, dtype=DTYPE)
    ]

    # Norm modifies the previous constraint values, so set the previous "layer"
    # as the identity layer. For the other layers, this is ignored.
    constraint = torch.eye(
        len(upper_bound), len(upper_bound) + 1, device=DEVICE, dtype=DTYPE
    )
    verifier._upper_constraint = [constraint.clone()]
    verifier._lower_constraint = [constraint.clone()]

    if isinstance(layer, torch.nn.Linear):
        verifier._analyze_affine(layer)
    elif isinstance(layer, Normalization):
        verifier._analyze_norm(layer)
    elif isinstance(layer, SPU):
        verifier._analyze_spu(layer)

    inputs_np = np.linspace(lower_bound, upper_bound, num=NUM_TEST_POINTS)
    inputs = torch.from_numpy(inputs_np).to(
        device=DEVICE, dtype=DTYPE
    )  # dim: NUM_TEST_POINTS x D
    outputs = layer(inputs)  # dim: NUM_TEST_POINTS x D

    # Test bounds
    new_upper_bound = verifier._upper_bound[-1]  # dim: D
    new_lower_bound = verifier._lower_bound[-1]  # dim: D
    assert (outputs >= new_lower_bound.unsqueeze(0) - EPS).all()
    assert (outputs <= new_upper_bound.unsqueeze(0) + EPS).all()

    inputs_with_bias = torch.cat(
        [inputs, torch.ones(len(inputs), 1, device=DEVICE, dtype=DTYPE)], dim=1
    )  # NUM_TEST_POINTS x (D+1)

    # NUM_TEST_POINTS x D = NUM_TEST_POINTS x (D+1) @ (D+1) x D
    constraint_upper_bound = (
        inputs_with_bias @ verifier._upper_constraint[-1].T
    )
    constraint_lower_bound = (
        inputs_with_bias @ verifier._lower_constraint[-1].T
    )

    # Test constraints
    assert (outputs >= constraint_lower_bound - EPS).all()
    assert (outputs <= constraint_upper_bound + EPS).all()


@pytest.mark.parametrize("lower_bound, upper_bound", TEST_BOUNDS)
@pytest.mark.parametrize(
    "weight, bias", [(-1.0, 1e-4), (1.0, 1e2), (1e-2, 1e2), (-1e-2, -1e-4)]
)
@pytest.mark.parametrize("flag", [-1, 0, 1])
def test_affine(lower_bound, upper_bound, weight, bias, flag) -> None:
    """Test the Affine transformer.

    Args:
        lower_bound: A 1D vector of input lower bounds
        upper_bound: A 1D vector of corresponding input upper bounds
        weight: A scalar used to construct the weights for the affine layer
        bias: A scalar used to construct the bias for the affine layer
        flag: The mode of alteration for the signs of the coefficients in the
            weights and biases
    """
    layer = torch.nn.Linear(2, lower_bound.shape[0])

    weight_tmp = torch.full(
        (2, lower_bound.shape[0]), weight, dtype=DTYPE, device=DEVICE
    )
    bias_tmp = torch.full((2,), bias, dtype=DTYPE, device=DEVICE)
    for i in range(lower_bound.shape[0]):
        if i % 2 == flag:
            bias_tmp[i] *= -1
        for j in range(2):
            if (i + j) % 2 == flag:
                weight_tmp[j, i] *= -1

    layer.weight = torch.nn.Parameter(weight_tmp)
    layer.bias = torch.nn.Parameter(bias_tmp)

    _test_helper(lower_bound, upper_bound, layer)


@pytest.mark.parametrize("lower_bound, upper_bound", TEST_BOUNDS)
def test_norm(lower_bound, upper_bound) -> None:
    """Test the Normalization transformer.

    Args:
        lower_bound: A 1D vector of input lower bounds
        upper_bound: A 1D vector of corresponding input upper bounds
    """
    _test_helper(lower_bound, upper_bound, Normalization(DEVICE))


@pytest.mark.parametrize("lower_bound, upper_bound", TEST_BOUNDS)
def test_spu(lower_bound: np.ndarray, upper_bound: np.ndarray) -> None:
    """Test the SPU transformer.

    Args:
        lower_bound: A 1D vector of input lower bounds
        upper_bound: A 1D vector of corresponding input upper bounds
    """
    _test_helper(lower_bound, upper_bound, SPU())
