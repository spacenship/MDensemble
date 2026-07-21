"""Conditional flow-matching paths.

The flow time ``tau`` lives in [0, 1] and is a purely mathematical
interpolation variable -- it must never be interpreted as physical MD time.
Physical time (``physical_delta_t``) is a separate conditioning input to the
model (see :mod:`protein_flow.models.fusion`).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


def sample_tau(batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Sample one flow time tau ~ Uniform(0, 1) per batch element."""
    return torch.rand(batch_size, device=device, dtype=dtype)


class FlowPath(abc.ABC):
    """Interface for a conditional probability path between x0 and x1."""

    @abc.abstractmethod
    def sample(self, x0: Tensor, x1: Tensor, tau: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (x_tau, target_velocity) for the given endpoints and tau.

        Args:
            x0: [B, L, 3] source coordinates.
            x1: [B, L, 3] Kabsch-aligned target coordinates.
            tau: [B] flow time in [0, 1].

        Returns:
            x_tau: [B, L, 3] interpolated coordinates.
            target_velocity: [B, L, 3] velocity the model should regress to.
        """
        raise NotImplementedError


class LinearPath(FlowPath):
    """Deterministic straight-line conditional path (default, fully supported).

    x_tau = (1 - tau) * x0 + tau * x1
    target_velocity = x1 - x0
    """

    def sample(self, x0: Tensor, x1: Tensor, tau: Tensor) -> Tuple[Tensor, Tensor]:
        tau_b = tau.view(-1, 1, 1)
        x_tau = (1.0 - tau_b) * x0 + tau_b * x1
        target_velocity = x1 - x0
        return x_tau, target_velocity


class GaussianBridgePath(FlowPath):
    """Noisy bridge path around the linear mean (future-extension stub).

    x_tau = (1 - tau) * x0 + tau * x1 + sigma_min * tau * (1 - tau) * eps

    The added noise vanishes at both endpoints (tau=0 and tau=1) so x0/x1
    are recovered exactly, but the conditional target velocity here still
    uses the linear-mean derivative (x1 - x0) and ignores the noise-term
    derivative. This is a simplified approximation intended as scaffolding
    for future work, not a rigorously derived stochastic bridge -- it is
    not used by the default config (``flow.path_type: linear``).
    """

    def __init__(self, sigma_min: float = 0.0):
        self.sigma_min = sigma_min

    def sample(self, x0: Tensor, x1: Tensor, tau: Tensor) -> Tuple[Tensor, Tensor]:
        tau_b = tau.view(-1, 1, 1)
        mean = (1.0 - tau_b) * x0 + tau_b * x1
        target_velocity = x1 - x0
        if self.sigma_min > 0.0:
            noise_scale = self.sigma_min * tau_b * (1.0 - tau_b)
            x_tau = mean + noise_scale * torch.randn_like(mean)
        else:
            x_tau = mean
        return x_tau, target_velocity


def build_flow_path(path_type: str, sigma_min: float = 0.0) -> FlowPath:
    """Factory selecting a :class:`FlowPath` implementation by name."""
    if path_type == "linear":
        return LinearPath()
    if path_type == "gaussian_bridge":
        return GaussianBridgePath(sigma_min=sigma_min)
    raise ValueError(f"Unknown flow path_type: {path_type!r}")
