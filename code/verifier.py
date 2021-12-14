"""Code to verify feed-forward MNIST networks."""
import argparse
import logging
from typing import Iterable, Optional, Union

import torch
from networks import SPU, FullyConnected, Normalization
from typing_extensions import Final

DEVICE: Final = "cpu"
DTYPE: Final = torch.float32
INPUT_SIZE: Final = 28

# Optimization constants
STEPS: Final = 50
LR: Final = 0.1
ALPHA: Final = 0.999


class Verifier(torch.nn.Module, torch.nn.modules.lazy.LazyModuleMixin):
    """Class that analyzes a network using DeepPoly."""

    BINARY_ITER: Final = 10
    SIGMOID_SCALE: Final = 5  # replace [-inf, inf] with this

    def __init__(self, layers: Iterable[torch.nn.Module]):
        """Store the network."""
        super().__init__()
        self.layers = layers
        self.params = torch.nn.ParameterDict()

    def _get_param(self, name: str, value: torch.Tensor) -> torch.nn.Parameter:
        """Get the requested param, creating it if it doesn't exist."""
        if name not in self.params:
            self.params[name] = torch.nn.Parameter(value)
        return self.params[name]

    def forward(self, inputs: torch.Tensor, eps: float) -> torch.Tensor:
        """Do an analysis forward propagation."""
        self._upper_bound = [torch.clamp(inputs + eps, max=1.0)]
        self._lower_bound = [torch.clamp(inputs - eps, min=0.0)]

        # Constraints must be of the shape:
        # [curr_layer_neurons, prev_layer_neurons + 1]
        # Thus, it acts as a weights + bias matrix (for convenience)
        self._upper_constraint = [self._upper_bound[0].unsqueeze(-1)]
        self._lower_constraint = [self._lower_bound[0].unsqueeze(-1)]

        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                self._analyze_affine(layer)
            elif isinstance(layer, Normalization):
                self._analyze_norm(layer)
                # No need for back-substitution, since this directly changes
                # the previous constraints & bounds
                continue
            elif isinstance(layer, SPU):
                self._analyze_spu(layer)
            elif isinstance(layer, torch.nn.Flatten):
                continue  # Ignore flatten
            else:
                raise NotImplementedError(
                    f"Layer type {type(layer)} is not supported"
                )

            self._back_substitute()

        # The lower bound of `y_true - y_other` for all `other` labels must be
        # positive
        return self._lower_bound[-1].min()

    def _analyze_affine(self, layer: torch.nn.Linear) -> None:
        """Analyze the affine layer."""
        weight = layer.weight
        bias = layer.bias

        constraint = torch.cat([weight, bias.unsqueeze(-1)], dim=1)
        self._upper_constraint.append(constraint)
        self._lower_constraint.append(constraint)

        bounds_for_upper = torch.where(
            weight > 0,
            self._upper_bound[-1].unsqueeze(0),
            self._lower_bound[-1].unsqueeze(0),
        )
        bounds_for_lower = torch.where(
            weight > 0,
            self._lower_bound[-1].unsqueeze(0),
            self._upper_bound[-1].unsqueeze(0),
        )
        self._upper_bound.append(
            torch.einsum("ij,ij->i", bounds_for_upper, weight) + bias
        )
        self._lower_bound.append(
            torch.einsum("ij,ij->i", bounds_for_lower, weight) + bias
        )

    def _analyze_norm(self, layer: Normalization) -> None:
        """Analyze the normalization layer."""
        mean = layer.mean.squeeze()
        std_dev = layer.sigma.squeeze()

        self._upper_bound[-1] = (self._upper_bound[-1] - mean) / std_dev
        self._lower_bound[-1] = (self._lower_bound[-1] - mean) / std_dev

        for constraints in self._upper_constraint, self._lower_constraint:
            constraints[-1][:, -1] -= mean
            constraints[-1] /= std_dev

    @staticmethod
    def _line_to_constraint(
        slope: torch.Tensor, intercept: torch.Tensor
    ) -> torch.Tensor:
        """Get the constraint matrix from the slope and intercept."""
        return torch.cat([slope.diagflat(), intercept.unsqueeze(1)], dim=1)

    @classmethod
    def _get_joining_line_constr(
        cls,
        x1: torch.Tensor,
        y1: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
    ) -> torch.Tensor:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y2 - slope * x2
        return cls._line_to_constraint(slope, intercept)

    @classmethod
    def _get_parabola_tangent_constr(cls, x: torch.Tensor) -> torch.Tensor:
        slope = 2 * x
        intercept = -(x ** 2) - 0.5
        return cls._line_to_constraint(slope, intercept)

    @classmethod
    def _get_parabola_tangent_constr_crossing_a_point(
        cls, x_1: torch.Tensor, y_1: torch.Tensor
    ) -> torch.Tensor:
        """Get a tangent line to the parabola that crosses (x_1, y_1)."""
        D = 4 * x_1 ** 2 - 4 * y_1 - 2
        # Clamp to avoid NaN
        x_2 = (2 * x_1 + D.clamp(min=0.0).sqrt()) / 2
        return cls._get_parabola_tangent_constr(x_2)

    @classmethod
    def _get_sigmoid_tangent_constr(cls, x: torch.Tensor) -> torch.Tensor:
        sigmoid = torch.sigmoid(x)
        slope = -sigmoid * (1 - sigmoid)
        intercept = -sigmoid - slope * x
        return cls._line_to_constraint(slope, intercept)

    @classmethod
    def _get_binary_search_point(
        cls,
        lower_x: torch.Tensor,
        upper_x: torch.Tensor,
        upper_y: torch.Tensor,
    ) -> torch.Tensor:
        L = lower_x
        R = torch.zeros_like(lower_x)
        for _ in range(0, cls.BINARY_ITER):
            m = (L + R) / 2
            sigmoid_constraint_mid = cls._get_sigmoid_tangent_constr(m)
            sigmoid_tangent_value = (
                sigmoid_constraint_mid.diagonal() * upper_x
                + sigmoid_constraint_mid[:, -1]
                - upper_y
            )
            binary_search_mask = sigmoid_tangent_value > 0
            # If the intersection point of tangent at m with line x = upper_x
            # is above upper_y, set L as the mid point and do not change R
            # else set R as the mid point and do not change L
            L = torch.where(binary_search_mask, m, L)
            R = torch.where(binary_search_mask, R, m)

        return L

    def _analyze_spu(self, layer: SPU) -> None:
        """Analyze the SPU layer."""
        upper_x = self._upper_bound[-1]
        lower_x = self._lower_bound[-1]
        upper_y = layer(upper_x)
        lower_y = layer(lower_x)

        layer_idx = len(self._upper_bound)

        self._upper_bound.append(torch.maximum(upper_y, lower_y))

        joining_constraint = self._get_joining_line_constr(
            lower_x, lower_y, upper_x, upper_y
        )

        # lower_bound > 0
        case_right_mask = (lower_x > 0).unsqueeze(1)
        parabola_pos = self._get_param(
            f"parabola_pos/{layer_idx}", torch.zeros_like(upper_x)
        )
        parabola_pos = (
            torch.sigmoid(parabola_pos) * (upper_x - lower_x) + lower_x
        )
        parabola_constraint = self._get_parabola_tangent_constr(parabola_pos)

        # upper_bound < 0
        case_left_mask = (upper_x <= 0).unsqueeze(1)
        sigmoid_pos = self._get_param(
            f"sigmoid_pos/{layer_idx}", torch.zeros_like(upper_x)
        )
        sigmoid_pos = (
            torch.sigmoid(sigmoid_pos) * (upper_x - lower_x) + lower_x
        )
        sigmoid_constraint_mid = self._get_sigmoid_tangent_constr(sigmoid_pos)

        # Crossing case
        crossing_mask = ~case_left_mask & ~case_right_mask
        new_lower_bound = torch.where(
            crossing_mask.squeeze(1),
            -0.5 * torch.ones_like(upper_y),
            torch.minimum(upper_y, lower_y),
        )
        self._lower_bound.append(new_lower_bound)

        # Get the tangent line to the parabola at upper_x
        parabola_tangent_line_at_upper_x = self._get_parabola_tangent_constr(
            upper_x
        )
        # Get the crossing point of the tangent line and x = lower_x,
        # use as the lower limit for crossing_lower_pos
        crossing_lower_pos_limit = (
            parabola_tangent_line_at_upper_x.diagonal() * lower_x
            + parabola_tangent_line_at_upper_x[:, -1]
        )

        crossing_lower_mask = (lower_x.abs() > upper_x).type(lower_x.dtype)
        crossing_lower_pos = self._get_param(
            f"crossing_lower_pos/{layer_idx}",
            (crossing_lower_mask - 0.5) * 2 * self.SIGMOID_SCALE,
        )
        crossing_lower_pos = (
            torch.sigmoid(crossing_lower_pos)
            * (lower_y - crossing_lower_pos_limit)
            + crossing_lower_pos_limit
        )

        # If above -0.5 use a joining line,
        # else use a tangent line to the parabola
        parabola_tangent_mask = (crossing_lower_pos < -0.5).unsqueeze(1)
        crossing_lower_constraint = torch.where(
            parabola_tangent_mask,
            self._get_parabola_tangent_constr_crossing_a_point(
                lower_x, crossing_lower_pos
            ),
            self._get_joining_line_constr(
                lower_x, crossing_lower_pos, 0, -0.5
            ),
        )

        # Set upper constraint based on whether upper bound is below tangent
        # at lower bound or above
        sigmoid_constraint_lower = self._get_sigmoid_tangent_constr(lower_x)
        sigmoid_tangent_value = (
            sigmoid_constraint_lower.diagonal() * upper_x
            + sigmoid_constraint_lower[:, -1]
            - upper_y
        )

        L = self._get_binary_search_point(lower_x, upper_x, upper_y)
        sigmoid_crossing_pos = self._get_param(
            f"sigmoid_crossing_pos/{layer_idx}",
            torch.ones_like(lower_x) * -self.SIGMOID_SCALE,
        )
        sigmoid_crossing_pos = (
            torch.sigmoid(sigmoid_crossing_pos) * (L - lower_x) + lower_x
        )
        sigmoid_constraint_crossing = self._get_sigmoid_tangent_constr(
            sigmoid_crossing_pos
        )

        # If u is less than intersection point, set upper constraint as
        # the tangent to the sigmoid part, else the joining line
        crossing_upper_constraint = torch.where(
            (sigmoid_tangent_value > 0).unsqueeze(1),
            sigmoid_constraint_crossing,
            joining_constraint,
        )

        self._upper_constraint.append(
            case_right_mask * joining_constraint
            + case_left_mask * sigmoid_constraint_mid
            + crossing_mask * crossing_upper_constraint
        )
        self._lower_constraint.append(
            case_right_mask * parabola_constraint
            + case_left_mask * joining_constraint
            + crossing_mask * crossing_lower_constraint
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
            x = torch.einsum(
                "ijk,ij->ik", x, curr_upper_constr[:, :-1]
            )  # Cx(A+1)
            x[:, -1] += curr_upper_constr[:, -1]

            y = torch.where(
                (curr_lower_constr[:, :-1] > 0).unsqueeze(-1),  # CxBx1
                prev_lower_constr.unsqueeze(0),  # 1xBx(A+1)
                prev_upper_constr.unsqueeze(0),  # 1xBx(A+1)
            )  # CxBx(A+1)
            y = torch.einsum(
                "ijk,ij->ik", y, curr_lower_constr[:, :-1]
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


def _get_output_layer(num_classes: int, true_lbl: int) -> torch.nn.Linear:
    """Get the output layer that captures the verification task.

    This returns a custom affine layer that captures `y_true - y_other` for
    each `other` label. This way, back-substitution should automatically
    be done for the output.
    """
    verification_layer = torch.nn.Linear(num_classes, num_classes - 1)
    torch.nn.init.zeros_(verification_layer.bias)
    torch.nn.init.zeros_(verification_layer.weight)

    # The i^th row should correspond to the equation `y = x_true - x_i` if
    # i < true_lbl, else `y = x_true - x_{i+1}`
    with torch.inference_mode():
        verification_layer.weight[:, true_lbl] = 1
        verification_layer.weight[:true_lbl, :true_lbl].fill_diagonal_(-1)
        verification_layer.weight[true_lbl:, true_lbl + 1 :].fill_diagonal_(-1)

    # Ensure that this layer isn't accidently trained
    for param in verification_layer.parameters():
        param.requires_grad = False

    return verification_layer


def analyze(
    net: FullyConnected,
    inputs: torch.Tensor,
    eps: float,
    true_label: int,
    device: Union[str, torch.device] = DEVICE,
) -> bool:
    """Analyze the network for the given input.

    This optimizes the verifier's parameters using SGD to improve precision.
    """
    # Remove the singleton batch and channel axes
    inputs = inputs.flatten().to(device=device, dtype=DTYPE)

    net = net.to(device=device, dtype=DTYPE)
    # Make the network un-trainable
    for param in net.parameters():
        param.requires_grad = False

    num_classes = net.layers[-1].out_features
    verification_layer = _get_output_layer(num_classes, true_label).to(
        device=device, dtype=DTYPE
    )
    verifier = Verifier(list(net.layers) + [verification_layer])
    verifier = verifier.to(device=device, dtype=DTYPE)

    best_objective = float("-inf")
    optim: Optional[torch.optim.Optimizer] = None

    for _ in range(STEPS):
        objective = verifier(inputs, eps)
        logging.debug(f"Objective: {objective:.4f}")
        best_objective = max(objective, best_objective)
        if objective > 0:
            return True

        if optim is None:
            optim = torch.optim.RMSprop(
                verifier.parameters(),
                lr=LR,
                alpha=ALPHA,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=STEPS
            )

        optim.zero_grad(set_to_none=True)
        # We have to increase the objective, but PyTorch decreases the loss
        (-objective).backward()
        optim.step()
        sched.step()

    best_objective = max(best_objective, verifier(inputs, eps))
    return best_objective > 0


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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print additional debug info.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)

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
