"""Code to verify feed-forward MNIST networks."""
import argparse

import torch
from networks import SPU, FullyConnected, Normalization
from typing_extensions import Final

DEVICE: Final = "cpu"
INPUT_SIZE: Final = 28


class Verifier:
    """Class that analyzes a network."""

    def __init__(self, net: FullyConnected):
        """Store the network."""
        self.net = net.to(DEVICE)

    def analyze(self, inputs: torch.Tensor, true_lbl: int, eps: float) -> bool:
        """Analyze the given input.

        Args:
            inputs: The 2D inputs to the model corresponding to one image
            true_label: The corresponding true label
            eps: The size of the L∞ norm ball around the inputs

        Returns:
            Whether the network is verified to be correct for the given region
        """
        # Remove the singleton batch and channel axes
        inputs = inputs.flatten()

        self._upper_bound = inputs + eps
        self._lower_bound = inputs - eps

        # Constraints must be of the shape:
        # [curr_layer_neurons, prev_layer_neurons + 1]
        # Thus, it acts as a weights + bias matrix (for convenience)
        self._upper_constraint = self._upper_bound.unsqueeze(-1)
        self._lower_constraint = self._lower_bound.unsqueeze(-1)

        for layer in self.net.layers:
            if isinstance(layer, torch.nn.Linear):
                self._analyze_affine(layer)
            elif isinstance(layer, Normalization):
                self._analyze_norm(layer)
            elif isinstance(layer, SPU):
                self._analyze_spu(layer)
            elif isinstance(layer, torch.nn.Flatten):
                pass  # Ignore flatten
            else:
                raise NotImplementedError(
                    f"Layer type {type(layer)} is not supported"
                )

        lower_bound_true_lbl = self._lower_bound[true_lbl]
        false_lbls = [
            i for i in range(len(self._upper_bound)) if i != true_lbl
        ]
        upper_bound_others = self._upper_bound[false_lbls].amax()
        return lower_bound_true_lbl > upper_bound_others

    def _analyze_affine(self, layer: torch.nn.Linear) -> None:
        """Analyze the affine layer."""
        self._upper_constraint = torch.cat(
            [layer.weight, layer.bias.unsqueeze(-1)], dim=1
        )
        self._lower_constraint = self._upper_constraint.clone()

        x = layer.weight @ self._upper_bound + layer.bias
        y = layer.weight @ self._lower_bound + layer.bias
        self._upper_bound = torch.maximum(x, y)
        self._lower_bound = torch.minimum(x, y)

    def _analyze_norm(self, layer: Normalization) -> None:
        """Analyze the normalization layer."""
        mean = layer.mean.squeeze()
        std_dev = layer.sigma.squeeze()

        self._upper_bound = (self._upper_bound - mean) / std_dev
        self._lower_bound = (self._lower_bound - mean) / std_dev

        self._upper_constraint[:, -1] -= mean
        self._upper_constraint /= std_dev

        self._lower_constraint[:, -1] -= mean
        self._lower_constraint /= std_dev

    @staticmethod
    def _get_parabola_tangent(x: torch.Tensor) -> torch.Tensor:
        slope = 2 * x
        intercept = -(x ** 2) - -0.5
        return slope, intercept

    @staticmethod
    def _get_sigmoid_tangent(x: torch.Tensor) -> torch.Tensor:
        sigmoid = 1 / (1 + torch.exp(-x))
        slope = -sigmoid * (1 - sigmoid)
        intercept = -sigmoid - slope * x
        return slope, intercept

    def _analyze_spu(self, layer: SPU) -> None:
        """Analyze the SPU layer."""
        upper_y = layer(self._upper_bound)
        lower_y = layer(self._lower_bound)

        self._upper_bound = torch.maximum(upper_y, lower_y)
        self._lower_bound = torch.minimum(upper_y, lower_y)

        joining_slope = (upper_y - lower_y) / (
            self._upper_bound - self._lower_bound
        )
        joining_intercept = (
            self._lower_bound * upper_y - self._upper_bound * lower_y
        ) / (self._upper_bound - self._lower_bound)
        joining_constraint = torch.cat(
            [joining_slope.diag(), joining_intercept.unsqueeze(1)], dim=1
        )

        # lower_bound > 0
        case_right_mask = (self._lower_bound > 0).unsqueeze(1)
        parabola_slope, parabola_intercept = self._get_parabola_tangent(
            self._lower_bound
        )
        parabola_constraint = torch.cat(
            [parabola_slope.diag(), parabola_intercept.unsqueeze(1)], dim=1
        )

        # upper_bound < 0
        case_left_mask = (self._upper_bound <= 0).unsqueeze(1)
        sigmoid_slope, sigmoid_intercept = self._get_sigmoid_tangent(
            self._lower_bound
        )
        sigmoid_constraint = torch.cat(
            [sigmoid_slope.diag(), sigmoid_intercept.unsqueeze(1)], dim=1
        )

        self._upper_constraint = (
            case_right_mask * joining_constraint
            + case_left_mask * sigmoid_constraint
        )
        self._lower_constraint = (
            case_right_mask * parabola_constraint
            + case_left_mask * joining_constraint
        )


def analyze(
    net: FullyConnected, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    """Analyze the network for the given input."""
    with torch.inference_mode():
        verifier = Verifier(net)
        return verifier.analyze(inputs, true_label, eps)


def main() -> None:
    """Run the main function."""
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation"
    )
    parser.add_argument(
        "--net",
        type=str,
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument(
        "--spec", type=str, required=True, help="Test case to verify."
    )
    args = parser.parse_args()

    with open(args.spec, "r") as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split("/")[-1].split("_")[-1])

    if args.net.endswith("fc1"):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith("fc2"):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith("fc3"):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith("fc4"):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith("fc5"):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(
            DEVICE
        )
    else:
        assert False

    net.load_state_dict(
        torch.load(
            "../mnist_nets/%s.pt" % args.net, map_location=torch.device(DEVICE)
        )
    )

    inputs = (
        torch.FloatTensor(pixel_values)
        .view(1, 1, INPUT_SIZE, INPUT_SIZE)
        .to(DEVICE)
    )
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
