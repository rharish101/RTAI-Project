"""Tests for the back substitution."""
import pytest
import torch
from networks import FullyConnected
from verifier import DEVICE, Verifier


@pytest.mark.parametrize(
    "upper_constraint, lower_constraint, "
    "upper_bound, expected_upper_bound, "
    "lower_bound, expected_lower_bound",
    [
        (
            [
                torch.Tensor([[1], [1]]),
                torch.Tensor([[1, 1, 0], [1, -1, 0]]),
                torch.Tensor([[0.5, 0, 1], [0, 0.5, 1]]),
                torch.Tensor([[1, 1, -0.5]]),
            ],
            [
                torch.Tensor([[-1], [-1]]),
                torch.Tensor([[1, 1, 0], [1, -1, 0]]),
                torch.Tensor([[0, 0, 0], [0, 0, 0]]),
                torch.Tensor([[0, 0, -0.5]]),
            ],
            [
                torch.Tensor([1, 1]),
                torch.Tensor([2, 2]),
                torch.Tensor([2, 2]),
                torch.Tensor([3.5]),
            ],
            torch.Tensor([2.5]),
            [
                torch.Tensor([-1, -1]),
                torch.Tensor([-2, -2]),
                torch.Tensor([0, 0]),
                torch.Tensor([-0.5]),
            ],
            torch.Tensor([-0.5]),
        ),
        (
            [
                torch.Tensor([[1], [1]]),
                torch.Tensor([[1, 1, 0], [1, -1, 0]]),
                torch.Tensor([[0.5, 0, 1], [0, 0.5, 1]]),
            ],
            [
                torch.Tensor([[-1], [-1]]),
                torch.Tensor([[1, 1, 0], [1, -1, 0]]),
                torch.Tensor([[0, 0, 0], [0, 0, 0]]),
            ],
            [torch.Tensor([1, 1]), torch.Tensor([2, 2]), torch.Tensor([2, 2])],
            torch.Tensor([2, 2]),
            [
                torch.Tensor([-1, -1]),
                torch.Tensor([-2, -2]),
                torch.Tensor([0, 0]),
            ],
            torch.Tensor([0, 0]),
        ),
        (
            [
                torch.Tensor([[1], [1]]),
                torch.Tensor([[1, 1, 0], [1, -1, 0]]),
                torch.Tensor([[0.5, 0, 1], [0, 0.5, 1]]),
                torch.Tensor([[1, 1, -0.5], [1, -1, 0]]),
                torch.Tensor([[5 / 6, 0, 5 / 12], [0, 0.5, 1]]),
                torch.Tensor([[-1, 1, 3], [0, 1, 0]]),
            ],
            [
                torch.Tensor([[-1], [-1]]),
                torch.Tensor([[1, 1, 0], [1, -1, 0]]),
                torch.Tensor([[0, 0, 0], [0, 0, 0]]),
                torch.Tensor([[1, 1, -0.5], [1, -1, 0]]),
                torch.Tensor([[0, 0, 0], [0, 0, 0]]),
                torch.Tensor([[-1, 1, 3], [0, 1, 0]]),
            ],
            [
                torch.Tensor([1, 1]),
                torch.Tensor([2, 2]),
                torch.Tensor([2, 2]),
                torch.Tensor([3.5, 2]),
                torch.Tensor([10 / 3, 2]),
                torch.Tensor([5, 2]),
            ],
            torch.Tensor([5, 2]),
            [
                torch.Tensor([-1, -1]),
                torch.Tensor([-2, -2]),
                torch.Tensor([0, 0]),
                torch.Tensor([-0.5, -2]),
                torch.Tensor([0, 0]),
                torch.Tensor([0.5, 0]),
            ],
            torch.Tensor([0.5, 0]),
        ),
    ],
)
def test_back_substitution(
    upper_constraint,
    lower_constraint,
    upper_bound,
    expected_upper_bound,
    lower_bound,
    expected_lower_bound,
) -> None:
    """Test back substitution functionality of the verifier.

    Dimensions reference:
        A: number of neurons in the current layer
        B: number of neurons in the previous layer
        C: number of inputs

    Args:
        upper_constraint: list of pytorch.Tensor objects,
            matrices each with dimensions Bx(A+1)
        lower_constraint: list of pytorch.Tensor objects,
            matrices each with dimensions Bx(A+1)
        upper_bound: list of pytorch.Tensor objects,
            1D vectors each with size C
        expected_upper_bound: pytorch.Tensor object,
            1D vector of size C
        lower_bound: list of pytorch.Tensor objects,
            1D vectors each with size C
        expected_lower_bound: pytorch.Tensor object,
            1D vector of size C
    """
    net = FullyConnected(DEVICE, 28, [10])
    verifier = Verifier(net)

    verifier._upper_constraint = upper_constraint
    verifier._lower_constraint = lower_constraint

    verifier._upper_bound = upper_bound
    verifier._lower_bound = lower_bound

    verifier._back_substitute()

    assert torch.allclose(verifier._upper_bound[-1], expected_upper_bound)
    assert torch.allclose(verifier._lower_bound[-1], expected_lower_bound)
