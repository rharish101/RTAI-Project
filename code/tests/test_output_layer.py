"""Tests for the custom output layer that captures the verification task."""
import pytest
import torch
from utils import DTYPE, EPS
from verifier import DEVICE, _get_output_layer


@pytest.mark.parametrize(
    "num_classes, true_lbl",
    [(10, 5), (7, 0), (5, 4)],
)
def test_output_layer(num_classes: int, true_lbl: int) -> None:
    """Test the custom verification affine layer's weight and bias."""
    layer = _get_output_layer(num_classes, true_lbl).to(
        device=DEVICE, dtype=DTYPE
    )

    curr_row_idx = 0

    for other_idx in range(num_classes):
        if other_idx == true_lbl:
            continue

        # Equation should be: y = x_true - x_other
        assert torch.allclose(
            layer.weight[curr_row_idx, true_lbl],
            torch.ones([], device=DEVICE, dtype=DTYPE),
            atol=EPS,
        )
        assert torch.allclose(
            layer.weight[curr_row_idx, other_idx],
            -torch.ones([], device=DEVICE, dtype=DTYPE),
            atol=EPS,
        )
        assert torch.allclose(
            layer.bias[curr_row_idx],
            torch.zeros([], device=DEVICE, dtype=DTYPE),
            atol=EPS,
        )

        curr_row_idx += 1
