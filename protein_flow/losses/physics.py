"""Physics-informed regularization losses over C-alpha traces.

All losses correctly exclude padding residues and, for the clash term,
covalently-adjacent residue pairs. Reference bond distances / bond angles
are computed once (outside autograd) from either the source or the
Kabsch-aligned target structure, controlled by
``PhysicsLossConfig.d_ref_source``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from protein_flow.config import GraphConfig
from protein_flow.geometry.graph import build_geometric_graph
from protein_flow.utils import masked_mean


@torch.no_grad()
def compute_consecutive_distances(coords: Tensor, residue_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Reference consecutive C-alpha distances, detached from autograd.

    Returns:
        distances: [B, L-1].
        valid_pair: [B, L-1] bool, True where both i and i+1 are valid.
    """
    distances = (coords[:, 1:] - coords[:, :-1]).norm(dim=-1)
    valid_pair = residue_mask[:, :-1] & residue_mask[:, 1:]
    return distances, valid_pair


@torch.no_grad()
def compute_consecutive_cos_angles(coords: Tensor, residue_mask: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """Reference bond-angle cosines over consecutive C-alpha triplets (i-1, i, i+1).

    Returns:
        cos_angles: [B, L-2].
        valid_triplet: [B, L-2] bool, True where i-1, i, i+1 are all valid.
    """
    v1 = coords[:, :-2] - coords[:, 1:-1]
    v2 = coords[:, 2:] - coords[:, 1:-1]
    cos_angles = (v1 * v2).sum(dim=-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + eps)
    valid_triplet = residue_mask[:, :-2] & residue_mask[:, 1:-1] & residue_mask[:, 2:]
    return cos_angles, valid_triplet


def bond_distance_loss(pred_coords: Tensor, reference_distances: Tensor, valid_pair: Tensor) -> Tensor:
    """Masked MSE between predicted and reference consecutive C-alpha distances."""
    pred_distances = (pred_coords[:, 1:] - pred_coords[:, :-1]).norm(dim=-1)
    squared_error = (pred_distances - reference_distances) ** 2
    return masked_mean(squared_error, valid_pair)


def bond_angle_loss(pred_coords: Tensor, reference_cos_angles: Tensor, valid_triplet: Tensor, eps: float = 1e-8) -> Tensor:
    """Masked MSE between predicted and reference bond-angle cosines."""
    v1 = pred_coords[:, :-2] - pred_coords[:, 1:-1]
    v2 = pred_coords[:, 2:] - pred_coords[:, 1:-1]
    pred_cos_angles = (v1 * v2).sum(dim=-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + eps)
    squared_error = (pred_cos_angles - reference_cos_angles) ** 2
    return masked_mean(squared_error, valid_triplet)


def clash_loss(
    pred_coords: Tensor,
    residue_mask: Tensor,
    graph_config: GraphConfig,
    clash_threshold: float,
    clash_seq_sep: int,
    eps: float = 1e-8,
) -> Tensor:
    """Steric-clash penalty for non-covalently-adjacent residue pairs.

    Uses the same sparse k-NN construction as the geometric encoder rather
    than a full L x L pair tensor, then excludes edges whose sequence
    separation is <= ``clash_seq_sep`` (covalent / near-covalent neighbors).
    """
    graph = build_geometric_graph(
        pred_coords,
        residue_mask,
        k=graph_config.knn_k,
        use_radius_cutoff=graph_config.use_radius_cutoff,
        radius_cutoff=graph_config.radius_cutoff,
        eps=eps,
    )
    if graph.edge_index.shape[1] == 0:
        return pred_coords.new_zeros(())

    length = pred_coords.shape[1]
    src, dst = graph.edge_index
    src_local = src % length
    dst_local = dst % length
    seq_sep = (src_local - dst_local).abs()

    non_covalent = seq_sep > clash_seq_sep
    if not torch.any(non_covalent):
        return pred_coords.new_zeros(())

    distances = graph.distances[non_covalent]
    penalty = torch.relu(clash_threshold - distances) ** 2
    return penalty.mean()


@dataclass
class PhysicsLossOutputs:
    bond: Tensor
    angle: Tensor
    clash: Tensor


def compute_physics_losses(
    pred_coords: Tensor,
    reference_coords: Tensor,
    residue_mask: Tensor,
    graph_config: GraphConfig,
    clash_threshold: float,
    clash_seq_sep: int,
) -> PhysicsLossOutputs:
    """Convenience wrapper computing bond, angle, and clash losses together.

    ``reference_coords`` supplies the (detached) reference geometry for the
    bond and angle terms; ``pred_coords`` is the structure the losses are
    actually applied to (either x_tau or an Euler-stepped prediction,
    depending on ``PhysicsLossConfig.apply_to``).
    """
    ref_distances, valid_pair = compute_consecutive_distances(reference_coords, residue_mask)
    ref_cos_angles, valid_triplet = compute_consecutive_cos_angles(reference_coords, residue_mask)

    bond = bond_distance_loss(pred_coords, ref_distances, valid_pair)
    angle = bond_angle_loss(pred_coords, ref_cos_angles, valid_triplet)
    clash = clash_loss(pred_coords, residue_mask, graph_config, clash_threshold, clash_seq_sep)
    return PhysicsLossOutputs(bond=bond, angle=angle, clash=clash)


def endpoint_rmsd_loss(rollout_coords: Tensor, aligned_target: Tensor, residue_mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Masked RMSD between an ODE-rollout endpoint and the aligned target.

    Disabled by default (see ``LossConfig.endpoint_enabled``) because it
    requires a full ODE rollout inside the training step, which is far more
    expensive per-step than the other (single-shot) losses.
    """
    squared_dist = ((rollout_coords - aligned_target) ** 2).sum(dim=-1)  # [B, L]
    mask = residue_mask.to(squared_dist.dtype)
    count = mask.sum(dim=1).clamp(min=1.0)
    mean_squared = (squared_dist * mask).sum(dim=1) / count
    return torch.sqrt(mean_squared + eps).mean()


# --- All-atom (topology-driven) physics losses -------------------------------
#
# The C-alpha losses above define "bonded" as "adjacent in sequence", which is
# only meaningful when one particle == one residue. An all-atom structure has
# branched side chains whose connectivity cannot be read off atom ordering, so
# these variants take the real CHARMM covalent topology (bond/angle index
# tensors from protein_flow/data/topology.py) instead.


def topology_bond_loss(
    pred_coords: Tensor,
    reference_coords: Tensor,
    bond_index: Tensor,
    bond_mask: Tensor,
) -> Tensor:
    """Masked MSE between predicted and reference covalent bond lengths.

    Args:
        pred_coords: [B, N, 3] structure the loss is applied to.
        reference_coords: [B, N, 3] structure supplying target bond lengths.
        bond_index: [B, E, 2] atom indices of each covalent bond.
        bond_mask: [B, E] bool, True for real (non-padding) bonds.
    """
    if bond_index.shape[1] == 0:
        return pred_coords.new_zeros(())
    i, j = bond_index[..., 0], bond_index[..., 1]
    pred_lengths = _gathered_distance(pred_coords, i, j)
    with torch.no_grad():
        reference_lengths = _gathered_distance(reference_coords, i, j)
    return masked_mean((pred_lengths - reference_lengths) ** 2, bond_mask)


def topology_angle_loss(
    pred_coords: Tensor,
    reference_coords: Tensor,
    angle_index: Tensor,
    angle_mask: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Masked MSE between predicted and reference covalent bond-angle cosines.

    Args:
        angle_index: [B, E, 3] atom indices (i, centre, k) of each covalent angle.
        angle_mask: [B, E] bool, True for real angles.
    """
    if angle_index.shape[1] == 0:
        return pred_coords.new_zeros(())
    i, centre, k = angle_index[..., 0], angle_index[..., 1], angle_index[..., 2]
    pred_cos = _gathered_cosine(pred_coords, i, centre, k, eps)
    with torch.no_grad():
        reference_cos = _gathered_cosine(reference_coords, i, centre, k, eps)
    return masked_mean((pred_cos - reference_cos) ** 2, angle_mask)


def _gathered_distance(coords: Tensor, i: Tensor, j: Tensor) -> Tensor:
    xi = torch.gather(coords, 1, i.unsqueeze(-1).expand(-1, -1, 3))
    xj = torch.gather(coords, 1, j.unsqueeze(-1).expand(-1, -1, 3))
    return (xi - xj).norm(dim=-1)


def _gathered_cosine(coords: Tensor, i: Tensor, centre: Tensor, k: Tensor, eps: float) -> Tensor:
    xi = torch.gather(coords, 1, i.unsqueeze(-1).expand(-1, -1, 3))
    xc = torch.gather(coords, 1, centre.unsqueeze(-1).expand(-1, -1, 3))
    xk = torch.gather(coords, 1, k.unsqueeze(-1).expand(-1, -1, 3))
    v1, v2 = xi - xc, xk - xc
    return (v1 * v2).sum(dim=-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + eps)


def _covalent_exclusion_keys(
    bond_index: Tensor, bond_mask: Tensor, angle_index: Tensor, angle_mask: Tensor, num_atoms: int
) -> Tensor:
    """Sorted int64 keys of all atom pairs that must be exempt from the clash
    penalty: 1-2 (bonded) and 1-3 (angle end) pairs, which are legitimately
    much closer than any non-bonded contact."""
    batch_size = bond_index.shape[0]
    device = bond_index.device
    batch_offset = torch.arange(batch_size, device=device).view(-1, 1) * num_atoms * num_atoms

    def to_keys(a: Tensor, b: Tensor, mask: Tensor) -> Tensor:
        low = torch.minimum(a, b)
        high = torch.maximum(a, b)
        keys = batch_offset + low * num_atoms + high
        return keys[mask]

    parts = [to_keys(bond_index[..., 0], bond_index[..., 1], bond_mask)]
    if angle_index.shape[1] > 0:
        # (i, centre, k): all three pairs are within one or two bonds.
        parts.append(to_keys(angle_index[..., 0], angle_index[..., 2], angle_mask))
        parts.append(to_keys(angle_index[..., 0], angle_index[..., 1], angle_mask))
        parts.append(to_keys(angle_index[..., 1], angle_index[..., 2], angle_mask))
    return torch.unique(torch.cat(parts)) if parts else torch.zeros(0, dtype=torch.long, device=device)


def topology_clash_loss(
    pred_coords: Tensor,
    atom_mask: Tensor,
    graph_config: GraphConfig,
    clash_threshold: float,
    bond_index: Tensor,
    bond_mask: Tensor,
    angle_index: Tensor,
    angle_mask: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Steric-clash penalty over the sparse k-NN graph, excluding covalently
    close (1-2 and 1-3) atom pairs.

    ``clash_threshold`` should be set for the representation in use: measured
    on real mdCATH frames, no non-bonded heavy-atom pair comes closer than
    2.51 A, whereas C-alpha-only pairs are separated by much more.
    """
    graph = build_geometric_graph(
        pred_coords, atom_mask, k=graph_config.knn_k,
        use_radius_cutoff=graph_config.use_radius_cutoff,
        radius_cutoff=graph_config.radius_cutoff, eps=eps,
    )
    if graph.edge_index.shape[1] == 0:
        return pred_coords.new_zeros(())

    num_atoms = pred_coords.shape[1]
    src, dst = graph.edge_index
    batch_of_edge = src // num_atoms
    local_src, local_dst = src % num_atoms, dst % num_atoms
    low = torch.minimum(local_src, local_dst)
    high = torch.maximum(local_src, local_dst)
    edge_keys = batch_of_edge * num_atoms * num_atoms + low * num_atoms + high

    excluded = _covalent_exclusion_keys(bond_index, bond_mask, angle_index, angle_mask, num_atoms)
    keep = ~torch.isin(edge_keys, excluded)
    if not torch.any(keep):
        return pred_coords.new_zeros(())

    penalty = torch.relu(clash_threshold - graph.distances[keep]) ** 2
    return penalty.mean()


def compute_all_atom_physics_losses(
    pred_coords: Tensor,
    reference_coords: Tensor,
    atom_mask: Tensor,
    bond_index: Tensor,
    bond_mask: Tensor,
    angle_index: Tensor,
    angle_mask: Tensor,
    graph_config: GraphConfig,
    clash_threshold: float,
) -> PhysicsLossOutputs:
    """All-atom counterpart of :func:`compute_physics_losses`."""
    return PhysicsLossOutputs(
        bond=topology_bond_loss(pred_coords, reference_coords, bond_index, bond_mask),
        angle=topology_angle_loss(pred_coords, reference_coords, angle_index, angle_mask),
        clash=topology_clash_loss(
            pred_coords, atom_mask, graph_config, clash_threshold,
            bond_index, bond_mask, angle_index, angle_mask,
        ),
    )
