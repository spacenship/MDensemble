"""ODE integration for sampling trajectories from the learned velocity field.

Integrates dx/dtau = v_theta(x, tau, conditions) from tau=0 to tau=1. The
geometric k-NN graph is rebuilt from scratch at every step (and at every
Heun sub-evaluation) since it depends on the current coordinates, which
change at each step.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torch import nn


def _make_tau_batch(batch_size: int, tau_value: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.full((batch_size,), tau_value, device=device, dtype=dtype)


@torch.no_grad()
def integrate_ode(
    model: nn.Module,
    x0: Tensor,
    sequence_embedding: Tensor,
    residue_types: Tensor,
    residue_mask: Tensor,
    temperature: Tensor,
    physical_delta_t: Tensor,
    num_steps: int = 50,
    solver: str = "heun",
    return_trajectory: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Integrate the flow ODE from tau=0 (x0) to tau=1.

    Args:
        model: a DualGraphFlowModel-like module callable as
            ``model(x_tau, tau, sequence_embedding, residue_types,
            temperature, physical_delta_t, residue_mask)``.
        x0: [B, L, 3] initial coordinates.
        num_steps: number of integration steps.
        solver: "euler" or "heun".
        return_trajectory: if True, also returns intermediate states.

    Returns:
        final_coords: [B, L, 3].
        trajectory: [num_steps + 1, B, L, 3] if ``return_trajectory`` else
            a tensor containing only the final state stacked once.
    """
    if solver not in ("euler", "heun"):
        raise ValueError(f"Unknown solver: {solver!r}")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")

    device, dtype = x0.device, x0.dtype
    batch_size = x0.shape[0]
    dtau = 1.0 / num_steps

    x = x0
    states = [x0] if return_trajectory else None

    for step in range(num_steps):
        tau_start = step * dtau
        tau_end = tau_start + dtau

        tau_batch_start = _make_tau_batch(batch_size, tau_start, device, dtype)
        v0 = model(x, tau_batch_start, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask)

        if solver == "euler":
            x = x + dtau * v0
        else:  # heun
            x_euler = x + dtau * v0
            tau_batch_end = _make_tau_batch(batch_size, tau_end, device, dtype)
            v1 = model(
                x_euler, tau_batch_end, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask
            )
            x = x + dtau * 0.5 * (v0 + v1)

        if return_trajectory:
            states.append(x)

    if return_trajectory:
        trajectory = torch.stack(states, dim=0)
    else:
        trajectory = x.unsqueeze(0)
    return x, trajectory
