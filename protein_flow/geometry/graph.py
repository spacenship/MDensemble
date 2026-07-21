"""Sparse graph construction for the bidirectional sequence graph and the
dynamic geometric graph.

Both graphs use a flattened node-index space: node ``(b, l)`` in a batch of
shape ``[B, L]`` is addressed as ``b * L + l``. This lets sparse edges from
every batch element live in one flat ``edge_index`` tensor (the same
convention used by torch_geometric), while guaranteeing samples never share
edges since each sample's local indices only ever combine with its own
batch offset.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Edge-type ids for the bidirectional typed sequence graph. This graph is
# NOT a DAG: it contains both i -> i+1 and i+1 -> i edges.
SEQUENCE_EDGE_FORWARD = 0
SEQUENCE_EDGE_BACKWARD = 1
NUM_SEQUENCE_EDGE_TYPES = 2


@dataclass
class SequenceGraph:
    """Bidirectional typed sequence graph (peptide connectivity).

    Attributes:
        edge_index: [2, E] (src, dst) node ids in flattened batch space.
        edge_type: [E] long tensor, SEQUENCE_EDGE_FORWARD or _BACKWARD.
        num_nodes: total number of nodes (B * L), including padding nodes.
    """

    edge_index: Tensor
    edge_type: Tensor
    num_nodes: int


@dataclass
class GeometricGraph:
    """Dynamic k-NN (optionally radius-cutoff) geometric graph.

    Attributes:
        edge_index: [2, E] (src=neighbor j, dst=center i) node ids in
            flattened batch space.
        distances: [E] Euclidean distance ||x_j - x_i||.
        relative_vectors: [E, 3] x_j - x_i (used to build equivariant
            vector outputs; never fed through a plain MLP as raw xyz).
        sequence_separation_norm: [E] |i - j| / valid_length(sample), in [0, 1].
        peptide_neighbor_indicator: [E] 1.0 if |i - j| == 1 else 0.0.
        num_nodes: total number of nodes (B * L), including padding nodes.
    """

    edge_index: Tensor
    distances: Tensor
    relative_vectors: Tensor
    sequence_separation_norm: Tensor
    peptide_neighbor_indicator: Tensor
    num_nodes: int


def _empty_sequence_graph(num_nodes: int, device: torch.device) -> SequenceGraph:
    return SequenceGraph(
        edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
        edge_type=torch.zeros(0, dtype=torch.long, device=device),
        num_nodes=num_nodes,
    )


def build_sequence_graph(residue_mask: Tensor) -> SequenceGraph:
    """Build the fixed bidirectional typed peptide-connectivity graph.

    An edge i -> i+1 (forward) and i+1 -> i (backward) is created only when
    both residues i and i+1 are valid (non-padding).

    Args:
        residue_mask: [B, L] bool tensor, True for valid residues.

    Returns:
        A :class:`SequenceGraph`.
    """
    batch_size, length = residue_mask.shape
    device = residue_mask.device
    num_nodes = batch_size * length

    if length < 2:
        return _empty_sequence_graph(num_nodes, device)

    valid_consecutive = residue_mask[:, :-1] & residue_mask[:, 1:]  # [B, L-1]
    if not torch.any(valid_consecutive):
        return _empty_sequence_graph(num_nodes, device)

    batch_index = torch.arange(batch_size, device=device).view(batch_size, 1).expand(batch_size, length - 1)
    local_index = torch.arange(length - 1, device=device).view(1, length - 1).expand(batch_size, length - 1)

    node_i = (batch_index * length + local_index)[valid_consecutive]
    node_ip1 = (batch_index * length + local_index + 1)[valid_consecutive]

    forward_src, forward_dst = node_i, node_ip1
    backward_src, backward_dst = node_ip1, node_i

    edge_index = torch.stack(
        [torch.cat([forward_src, backward_src]), torch.cat([forward_dst, backward_dst])],
        dim=0,
    )
    edge_type = torch.cat(
        [
            torch.full_like(forward_src, SEQUENCE_EDGE_FORWARD),
            torch.full_like(backward_src, SEQUENCE_EDGE_BACKWARD),
        ]
    )
    return SequenceGraph(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes)


def _empty_geometric_graph(num_nodes: int, device: torch.device, dtype: torch.dtype) -> GeometricGraph:
    return GeometricGraph(
        edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
        distances=torch.zeros(0, dtype=dtype, device=device),
        relative_vectors=torch.zeros(0, 3, dtype=dtype, device=device),
        sequence_separation_norm=torch.zeros(0, dtype=dtype, device=device),
        peptide_neighbor_indicator=torch.zeros(0, dtype=dtype, device=device),
        num_nodes=num_nodes,
    )


def build_geometric_graph(
    coords: Tensor,
    residue_mask: Tensor,
    k: int,
    use_radius_cutoff: bool = False,
    radius_cutoff: float = 12.0,
    eps: float = 1e-8,
) -> GeometricGraph:
    """Build the dynamic per-sample k-NN geometric graph from current coordinates.

    Neighbors are restricted to the same batch sample (no cross-sample
    edges), exclude self-edges and padding residues, and safely handle
    ``k`` larger than the number of valid residues in a sample. The dense
    [B, L, L] pairwise-distance tensor used internally is transient and is
    not part of the returned, sparse representation.

    Args:
        coords: [B, L, 3] current (e.g. flow-time) C-alpha coordinates.
        residue_mask: [B, L] bool tensor, True for valid residues.
        k: number of nearest neighbors requested per node.
        use_radius_cutoff: if True, additionally drop edges farther than
            ``radius_cutoff``.
        radius_cutoff: distance cutoff used only if ``use_radius_cutoff``.
        eps: numerical-stability epsilon.

    Returns:
        A :class:`GeometricGraph`.
    """
    batch_size, length, _ = coords.shape
    device = coords.device
    dtype = coords.dtype
    num_nodes = batch_size * length

    k_eff = min(k, length - 1) if length > 1 else 0
    if k_eff <= 0:
        return _empty_geometric_graph(num_nodes, device, dtype)

    valid_counts = residue_mask.sum(dim=1).clamp(min=1).to(dtype)  # [B]

    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, L, L, 3], transient
    sqdist = diff.pow(2).sum(dim=-1)  # [B, L, L], transient

    valid_pair = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)  # [B, L, L]
    self_mask = torch.eye(length, dtype=torch.bool, device=device).unsqueeze(0)
    valid_pair = valid_pair & ~self_mask
    sqdist_masked = sqdist.masked_fill(~valid_pair, float("inf"))

    topk_sqdist, topk_local_idx = torch.topk(sqdist_masked, k_eff, dim=-1, largest=False)  # [B, L, k_eff]

    keep = torch.isfinite(topk_sqdist)
    keep = keep & residue_mask.unsqueeze(-1).expand(-1, -1, k_eff)

    if use_radius_cutoff:
        keep = keep & (topk_sqdist <= radius_cutoff * radius_cutoff)

    if not torch.any(keep):
        return _empty_geometric_graph(num_nodes, device, dtype)

    batch_index = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand(batch_size, length, k_eff)
    center_local_idx = torch.arange(length, device=device).view(1, length, 1).expand(batch_size, length, k_eff)

    src_flat = (batch_index * length + topk_local_idx)[keep]  # neighbor j
    dst_flat = (batch_index * length + center_local_idx)[keep]  # center i
    edge_index = torch.stack([src_flat, dst_flat], dim=0)

    coords_flat = coords.reshape(num_nodes, 3)
    relative_vectors = coords_flat[src_flat] - coords_flat[dst_flat]  # x_j - x_i
    distances = torch.sqrt(relative_vectors.pow(2).sum(dim=-1) + eps)

    local_sep = (topk_local_idx - center_local_idx).abs()[keep].to(dtype)
    batch_of_edge = batch_index[keep]
    sequence_separation_norm = local_sep / valid_counts[batch_of_edge]
    peptide_neighbor_indicator = (local_sep == 1).to(dtype)

    return GeometricGraph(
        edge_index=edge_index,
        distances=distances,
        relative_vectors=relative_vectors,
        sequence_separation_norm=sequence_separation_norm,
        peptide_neighbor_indicator=peptide_neighbor_indicator,
        num_nodes=num_nodes,
    )
