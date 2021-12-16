"""Tests for binary search to get tangent to sigmoid from upper bound point."""
import numpy as np
import pytest
import torch
from networks import SPU
from typing_extensions import Final
from utils import DTYPE, EPS
from verifier import DEVICE, Verifier

NUM_TEST_POINTS: Final = int(1e4)
TOLERANCE: Final = 2e-3


@pytest.mark.parametrize("lower_bound, upper_bound", [(-10.0, 5)])
def test_binary_search(lower_bound, upper_bound) -> None:
    """Test the binary search functionality for SPU transformer.

    Args:
        lower_bound: A 1D vector of input lower bounds
        upper_bound: A 1D vector of corresponding input upper bounds
    """
    verifier = Verifier([]).to(device=DEVICE, dtype=DTYPE)
    layer = SPU()

    upper_bound_inputs = np.linspace(EPS, upper_bound, num=NUM_TEST_POINTS)
    verifier._upper_bound = [
        torch.tensor(upper_bound_inputs).to(device=DEVICE, dtype=DTYPE)
    ]

    lower_bound_inputs = np.linspace(lower_bound, -EPS, num=NUM_TEST_POINTS)
    verifier._lower_bound = [
        torch.tensor(lower_bound_inputs).to(device=DEVICE, dtype=DTYPE)
    ]

    upper_x = verifier._upper_bound[-1]
    lower_x = verifier._lower_bound[-1]
    upper_y = layer(upper_x)

    def get_sigmoid_tangent_dist(x: torch.Tensor) -> torch.Tensor:
        """Get distance of (upper_x, upper_y) from tangent at x."""
        sigmoid_constraint = verifier._get_sigmoid_tangent_constr(x)
        slope = sigmoid_constraint.diagonal()
        numerator = slope * upper_x + sigmoid_constraint[:, -1] - upper_y
        denominator = (slope ** 2 + 1).sqrt()
        return numerator / denominator

    binary_search_mask_lower = get_sigmoid_tangent_dist(lower_x) > 0

    L = verifier._get_binary_search_point(lower_x, upper_x, upper_y)
    sigmoid_tangent_value = get_sigmoid_tangent_dist(L)

    assert (L >= lower_x.unsqueeze(0) - EPS).all()
    assert (L < torch.zeros_like(L) + EPS).all()
    assert (binary_search_mask_lower == (sigmoid_tangent_value > 0)).all()
    assert (sigmoid_tangent_value < TOLERANCE)[binary_search_mask_lower].all()
