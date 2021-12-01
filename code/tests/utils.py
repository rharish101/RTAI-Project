"""Common utilities and constants for tests."""
import torch

# For better precision
DTYPE = torch.float64
# Used when comparing floats in tests
EPS = torch.finfo(DTYPE).eps
