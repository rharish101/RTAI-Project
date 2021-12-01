"""Common utilities and constants for tests."""
import torch
from typing_extensions import Final

# For better precision
DTYPE: Final = torch.float64
# Used when comparing floats in tests
EPS: Final = 1e-10
# Seed for any RNGs
SEED: Final = 0
