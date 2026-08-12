"""ODE integration for sampling trajectories from the learned velocity field.

Two integrators live here, matching the two families in
:mod:`protein_flow.flow.paths`.

:func:`integrate_ode` integrates dx/dtau = v_theta(x, tau, conditions) in
**coordinate space** from tau=0 to tau=1. The geometric k-NN graph is
rebuilt from scratch at every step (and at every Heun sub-evaluation) since
it depends on the current coordinates, which change at each step.

:func:`integrate_displacement_ode` integrates in **displacement space**,
starting from Gaussian noise and ending at a predicted displacement that is
added back to the source structure. The graph is built from the source
structure, which does not move, so every evaluation along the trajectory
sees the same (and always physically valid) geometry.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn

from protein_flow.flow.paths import sample_zero_com_noise


#: How far short of tau=1 the field may be queried. Training draws
#: ``tau ~ Uniform(0, 1)``, which never returns 1.0, and the learned field is
#: measurably wild at exactly that point: on a mid-training checkpoint the
#: flow-matching error at tau=1.00 was 1464 against 20.3 at tau=0.99, with the
#: predicted velocity 2.8x too large. Heun is the only integrator that reaches
#: it, since its corrector evaluates at tau_end and the final step lands on
#: 1.0 exactly. This keeps the query on the interval the field was fitted on.
#:
#: It is a guard, not a fix for anything currently observed: applying it left
#: the sampled structures unchanged to three decimals (moved 1.228, spread
#: 1.559, contact Jaccard 0.675 either way), so the tau=1 spike does not
#: survive one half-step of weight ``dtau/2``. Kept because querying a point
#: the model has never been trained on is indefensible on its own terms and
#: costs nothing, and because a later checkpoint's spike may be larger.
#:
#: What the same experiment *did* show, and what is easy to misread: euler at
#: 20 steps scores better than heun (moved 1.075 / spread 1.358 / contact
#: 0.717) -- but euler at 50 steps moves back toward heun (1.161 / 1.471 /
#: 0.703). Euler is not more faithful, it is under-resolved, and its
#: truncation error happens to cancel part of the model's over-dispersion.
#: Do not switch solvers to buy that; it is the model that over-disperses.
TAU_QUERY_LIMIT = 1.0 - 1e-3


def _make_tau_batch(batch_size: int, tau_value: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.full((batch_size,), min(tau_value, TAU_QUERY_LIMIT), device=device, dtype=dtype)


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
    **atom_inputs: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Integrate the flow ODE from tau=0 (x0) to tau=1.

    Args:
        model: a DualGraphFlowModel-like module callable as
            ``model(x_tau, tau, sequence_embedding, residue_types,
            temperature, physical_delta_t, residue_mask, **atom_inputs)``.
        x0: [B, N, 3] initial coordinates.
        num_steps: number of integration steps.
        solver: "euler" or "heun".
        return_trajectory: if True, also returns intermediate states.
        **atom_inputs: optional all-atom tensors (``atom_mask``,
            ``atom_residue_index``, ``atom_element``, ``ca_atom_index``)
            forwarded unchanged to every velocity evaluation. They describe
            fixed topology, so unlike the coordinates they do not change
            along the trajectory.

    Returns:
        final_coords: [B, N, 3].
        trajectory: [num_steps + 1, B, N, 3] if ``return_trajectory`` else
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
        v0 = model(
            x, tau_batch_start, sequence_embedding, residue_types,
            temperature, physical_delta_t, residue_mask, **atom_inputs,
        )

        if solver == "euler":
            x = x + dtau * v0
        else:  # heun
            x_euler = x + dtau * v0
            tau_batch_end = _make_tau_batch(batch_size, tau_end, device, dtype)
            v1 = model(
                x_euler, tau_batch_end, sequence_embedding, residue_types,
                temperature, physical_delta_t, residue_mask, **atom_inputs,
            )
            x = x + dtau * 0.5 * (v0 + v1)

        if return_trajectory:
            states.append(x)

    if return_trajectory:
        trajectory = torch.stack(states, dim=0)
    else:
        trajectory = x.unsqueeze(0)
    return x, trajectory


@torch.no_grad()
def integrate_displacement_ode(
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
    noise_scale: float = 2.667,
    smoothing_rounds: int = 0,
    knn_k: int = 16,
    remove_rigid_motion: bool = False,
    generator: Optional[torch.Generator] = None,
    **atom_inputs: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Integrate the displacement flow from noise to a generated structure.

    Starts at ``eps ~ N(0, noise_scale^2 I)`` projected onto the
    zero-centre-of-mass subspace, integrates
    ``ds/dtau = v(x0, s, tau, ...)`` over displacement space, and returns
    ``x0 + s``.

    Unlike :func:`integrate_ode` this is **stochastic**: the returned
    structure depends on the noise draw, which is the point -- calling it
    repeatedly with different draws is how an ensemble is produced. Pass
    ``generator`` to reproduce a particular draw.

    Args:
        x0: [B, N, 3] source structure. Conditioning, not an endpoint: it is
            passed to every velocity evaluation as the coordinates the graph
            is built from, and never itself integrated.
        noise_scale: per-axis standard deviation of the base distribution.
        smoothing_rounds: how many k-NN hops to smooth the noise over, which
            is what gives it the spatial correlation real protein motion has.

    Both noise arguments must match what the model was trained with
    (``flow.noise_scale``, ``flow.noise_smoothing_rounds``), or the sampler
    starts off the manifold the velocity field was fitted on.

    Returns:
        final_coords: [B, N, 3].
        trajectory: [num_steps + 1, B, N, 3] of *coordinates* (x0 + state at
            each step) if ``return_trajectory``, else the final state
            stacked once.
    """
    if solver not in ("euler", "heun"):
        raise ValueError(f"Unknown solver: {solver!r}")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")

    device, dtype = x0.device, x0.dtype
    batch_size = x0.shape[0]
    dtau = 1.0 / num_steps
    particle_mask = atom_inputs.get("atom_mask", residue_mask)

    state = sample_zero_com_noise(
        x0.shape, particle_mask, noise_scale, device, dtype, generator,
        coords=x0, smoothing_rounds=smoothing_rounds, knn_k=knn_k,
        remove_rigid_motion=remove_rigid_motion,
    )
    states = [x0 + state] if return_trajectory else None

    for step in range(num_steps):
        tau_start = step * dtau
        tau_end = tau_start + dtau

        tau_batch_start = _make_tau_batch(batch_size, tau_start, device, dtype)
        v0 = model(
            x0, tau_batch_start, sequence_embedding, residue_types,
            temperature, physical_delta_t, residue_mask,
            flow_state=state, **atom_inputs,
        )

        if solver == "euler":
            state = state + dtau * v0
        else:  # heun
            state_euler = state + dtau * v0
            tau_batch_end = _make_tau_batch(batch_size, tau_end, device, dtype)
            v1 = model(
                x0, tau_batch_end, sequence_embedding, residue_types,
                temperature, physical_delta_t, residue_mask,
                flow_state=state_euler, **atom_inputs,
            )
            state = state + dtau * 0.5 * (v0 + v1)

        if return_trajectory:
            states.append(x0 + state)

    final_coords = x0 + state
    if return_trajectory:
        trajectory = torch.stack(states, dim=0)
    else:
        trajectory = final_coords.unsqueeze(0)
    return final_coords, trajectory
