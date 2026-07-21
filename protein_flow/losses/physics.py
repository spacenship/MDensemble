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
