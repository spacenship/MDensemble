"""Scalar edge-feature encodings used by the geometric encoder."""
from __future__ import annotations

import torch
from torch import Tensor


def rbf_encode(distances: Tensor, num_rbf: int, min_dist: float, max_dist: float) -> Tensor:
    """Gaussian radial-basis-function encoding of scalar distances.

    Args:
        distances: [...] tensor of non-negative distances.
        num_rbf: number of basis functions.
        min_dist: center of the first basis function.
        max_dist: center of the last basis function.

    Returns:
        Tensor of shape [..., num_rbf].
    """
    centers = torch.linspace(min_dist, max_dist, num_rbf, device=distances.device, dtype=distances.dtype)
    width = (max_dist - min_dist) / max(num_rbf - 1, 1)
    diff = distances.unsqueeze(-1) - centers
    gamma = 1.0 / (2.0 * width * width + 1e-8)
    return torch.exp(-gamma * diff.pow(2))
