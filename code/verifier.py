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

        def add_bias(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x, torch.ones(1, device=DEVICE)])

        x = self._upper_constraint @ add_bias(self._upper_bound)
        y = self._upper_constraint @ add_bias(self._lower_bound)
        self._upper_bound = torch.maximum(x, y)
        self._lower_bound = torch.minimum(x, y)

    # TODO: Harish
    def _analyze_norm(self, layer: Normalization) -> None:
        """Analyze the normalization layer."""
        raise NotImplementedError

    # TODO: Martin + Vandit
    def _analyze_spu(self, layer: SPU) -> None:
        """Analyze the SPU layer."""
        raise NotImplementedError


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
