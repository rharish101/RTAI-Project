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

        self._upper_bound = [inputs + eps]
        self._lower_bound = [inputs - eps]

        # Constraints must be of the shape:
        # [curr_layer_neurons, prev_layer_neurons + 1]
        # Thus, it acts as a weights + bias matrix (for convenience)
        self._upper_constraint = [self._upper_bound[0].unsqueeze(-1)]
        self._lower_constraint = [self._lower_bound[0].unsqueeze(-1)]

        for layer in self.net.layers:
            if isinstance(layer, torch.nn.Linear):
                self._analyze_affine(layer)
            elif isinstance(layer, Normalization):
                self._analyze_norm(layer)
            elif isinstance(layer, SPU):
                self._analyze_spu(layer)
            elif isinstance(layer, torch.nn.Flatten):
                continue  # Ignore flatten
            else:
                raise NotImplementedError(
                    f"Layer type {type(layer)} is not supported"
                )

            self._back_substitute()

        lower_bound_true_lbl = self._lower_bound[-1][true_lbl]
        false_lbls = [
            i for i in range(len(self._upper_bound[-1])) if i != true_lbl
        ]
        upper_bound_others = self._upper_bound[-1][false_lbls].amax()
        return lower_bound_true_lbl > upper_bound_others

    def _analyze_affine(self, layer: torch.nn.Linear) -> None:
        """Analyze the affine layer."""
        constraint = torch.cat([layer.weight, layer.bias.unsqueeze(-1)], dim=1)
        self._upper_constraint.append(constraint)
        self._lower_constraint.append(constraint)

        bounds_for_upper = torch.where(
            layer.weight > 0,
            self._upper_bound[-1].unsqueeze(0),
            self._lower_bound[-1].unsqueeze(0),
        )
        bounds_for_lower = torch.where(
            layer.weight > 0,
            self._lower_bound[-1].unsqueeze(0),
            self._upper_bound[-1].unsqueeze(0),
        )
        self._upper_bound.append(
            torch.sum(bounds_for_upper * layer.weight, 1) + layer.bias
        )
        self._lower_bound.append(
            torch.sum(bounds_for_lower * layer.weight, 1) + layer.bias
        )

    def _analyze_norm(self, layer: Normalization) -> None:
        """Analyze the normalization layer."""
        mean = layer.mean.squeeze()
        std_dev = layer.sigma.squeeze()

        num_neurons = len(self._upper_bound[-1])
        self._upper_bound.append((self._upper_bound[-1] - mean) / std_dev)
        self._lower_bound.append((self._lower_bound[-1] - mean) / std_dev)

        constraint = torch.eye(num_neurons, num_neurons + 1, device=DEVICE)
        constraint[:, -1] = -mean
        constraint /= std_dev

        self._upper_constraint.append(constraint)
        self._lower_constraint.append(constraint)

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
        upper_x = self._upper_bound[-1]
        lower_x = self._lower_bound[-1]
        upper_y = layer(upper_x)
        lower_y = layer(lower_x)

        self._upper_bound.append(torch.maximum(upper_y, lower_y))
        self._lower_bound.append(torch.minimum(upper_y, lower_y))

        joining_slope = (upper_y - lower_y) / (upper_x - lower_x)
        joining_intercept = upper_y - joining_slope * upper_x
        joining_constraint = torch.cat(
            [joining_slope.diag(), joining_intercept.unsqueeze(1)], dim=1
        )

        # lower_bound > 0
        case_right_mask = (lower_x > 0).unsqueeze(1)
        parabola_slope, parabola_intercept = self._get_parabola_tangent(
            lower_x
        )
        parabola_constraint = torch.cat(
            [parabola_slope.diag(), parabola_intercept.unsqueeze(1)], dim=1
        )

        # upper_bound < 0
        case_left_mask = (upper_x <= 0).unsqueeze(1)
        sigmoid_slope, sigmoid_intercept = self._get_sigmoid_tangent(lower_x)
        sigmoid_constraint = torch.cat(
            [sigmoid_slope.diag(), sigmoid_intercept.unsqueeze(1)], dim=1
        )

        # Crossing case
        crossing_mask = ~case_left_mask & ~case_right_mask

        # Set lower constraint as the line y = -0.5
        num_neurons = len(upper_x)
        lowest_point_constraint = torch.zeros(
            (num_neurons, num_neurons + 1), device=DEVICE
        )
        lowest_point_constraint[:, -1] = -0.5

        # Set upper constraint based on whether upper bound is below tangent
        # at lower bound or above
        sigmoid_tangent_value = (
            sigmoid_slope * upper_x + sigmoid_intercept - upper_y
        )
        crossing_lesser_mask = (sigmoid_tangent_value > 0).unsqueeze(1)
        # If u is less than intersection point, set upper constraint as
        # the tangent to the sigmoid part, else the joining line
        crossing_upper_constraint = (
            crossing_lesser_mask * sigmoid_constraint
            + ~crossing_lesser_mask * joining_constraint
        )

        self._upper_constraint.append(
            case_right_mask * joining_constraint
            + case_left_mask * sigmoid_constraint
            + crossing_mask * crossing_upper_constraint
        )
        self._lower_constraint.append(
            case_right_mask * parabola_constraint
            + case_left_mask * joining_constraint
            + crossing_mask * lowest_point_constraint
        )

    def _back_substitute(self) -> None:
        """Make constraints more precise with back-substitution."""
        curr_upper_constr = self._upper_constraint[-1]
        curr_lower_constr = self._lower_constraint[-1]

        for prev_upper_constr, prev_lower_constr in zip(
            self._upper_constraint[-2::-1], self._lower_constraint[-2::-1]
        ):
            # Neurons: A -> B -> C
            x = torch.where(
                (curr_upper_constr[:, :-1] > 0).unsqueeze(-1),  # CxBx1
                prev_upper_constr.unsqueeze(0),  # 1xBx(A+1)
                prev_lower_constr.unsqueeze(0),  # 1xBx(A+1)
            )  # CxBx(A+1)
            x = torch.sum(
                x * curr_upper_constr[:, :-1].unsqueeze(-1), dim=1
            )  # Cx(A+1)
            x[:, -1] += curr_upper_constr[:, -1]

            y = torch.where(
                (curr_lower_constr[:, :-1] > 0).unsqueeze(-1),  # CxBx1
                prev_lower_constr.unsqueeze(0),  # 1xBx(A+1)
                prev_upper_constr.unsqueeze(0),  # 1xBx(A+1)
            )  # CxBx(A+1)
            y = torch.sum(
                y * curr_lower_constr[:, :-1].unsqueeze(-1), dim=1
            )  # Cx(A+1)
            y[:, -1] += curr_lower_constr[:, -1]

            curr_upper_constr = x
            curr_lower_constr = y

        self._upper_bound[-1] = torch.minimum(
            self._upper_bound[-1], curr_upper_constr.squeeze(1)
        )
        self._lower_bound[-1] = torch.maximum(
            self._lower_bound[-1], curr_lower_constr.squeeze(1)
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
