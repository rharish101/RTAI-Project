"""Tests for the custom output layer that captures the verification task."""
import pytest
import torch
from verifier import Verifier


@pytest.mark.parametrize(
    "num_classes, true_lbl",
    [(10, 5), (7, 0), (5, 4)],
)
def test_output_layer(num_classes: int, true_lbl: int) -> None:
    """Test the custom verification affine layer's weight and bias."""
    with torch.inference_mode():
        layer = Verifier._get_output_layer(num_classes, true_lbl)

    curr_row_idx = 0

    for other_idx in range(num_classes):
        if other_idx == true_lbl:
            continue

        # Equation should be: y = x_true - x_other
        assert layer.weight[curr_row_idx, true_lbl] == 1
        assert layer.weight[curr_row_idx, other_idx] == -1
        assert layer.bias[curr_row_idx] == 0

        curr_row_idx += 1
