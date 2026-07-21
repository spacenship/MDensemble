"""EGNN-style E(3)-equivariant geometric encoder.

Node hidden features produced by this module are rotation/translation
**invariant** (they are built only from pairwise distances and other scalar
edge features, never from raw relative-vector components). Equivariant
vector outputs are constructed downstream in
:mod:`protein_flow.models.vector_field` by weighting the *relative*
position vectors ``x_j - x_i`` with these invariant scalars -- this module
never emits a raw xyz vector from an MLP.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from protein_flow.geometry.features import rbf_encode
from protein_flow.geometry.graph import GeometricGraph, build_geometric_graph
from protein_flow.utils import scatter_mean


def edge_scalar_dim(num_rbf: int) -> int:
    """Dimensionality of [distance, rbf(distance), seq_sep_norm, peptide_indicator]."""
    return 1 + num_rbf + 1 + 1


def build_edge_scalar_features(graph: GeometricGraph, num_rbf: int, rbf_min_dist: float, rbf_max_dist: float) -> Tensor:
    rbf = rbf_encode(graph.distances, num_rbf, rbf_min_dist, rbf_max_dist)
    return torch.cat(
        [
            graph.distances.unsqueeze(-1),
            rbf,
            graph.sequence_separation_norm.unsqueeze(-1),
            graph.peptide_neighbor_indicator.unsqueeze(-1),
        ],
        dim=-1,
    )


class EGNNLayer(nn.Module):
    """A single EGNN-style message-passing layer.

    Hidden-feature update (always invariant):
        m_ij = phi_e(h_i, h_j, edge_scalars_ij)
        h_i' = h_i + phi_h(h_i, mean_j m_ij)

    Optional coordinate update (equivariant, off by default):
        a_ij = phi_x(m_ij)
        x_i' = x_i + mean_j a_ij * (x_j - x_i) / (||x_j - x_i|| + eps)
    """

    def __init__(self, hidden_dim: int, edge_feature_dim: int, update_coordinates: bool, dropout: float, eps: float = 1e-8):
        super().__init__()
        self.update_coordinates = update_coordinates
        self.eps = eps
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        if update_coordinates:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        h: Tensor,
        coords_flat: Tensor,
        edge_index: Tensor,
        edge_scalars: Tensor,
        relative_vectors: Tensor,
        distances: Tensor,
        num_nodes: int,
    ) -> Tuple[Tensor, Tensor]:
        if edge_index.shape[1] == 0:
            return self.norm(h), coords_flat

        src, dst = edge_index
        edge_input = torch.cat([h[src], h[dst], edge_scalars], dim=-1)
        messages = self.edge_mlp(edge_input)  # [E, hidden]
        aggregated = scatter_mean(messages, dst, dim_size=num_nodes)  # invariant
        h_new = self.norm(h + self.dropout(self.node_mlp(torch.cat([h, aggregated], dim=-1))))

        new_coords_flat = coords_flat
        if self.update_coordinates:
            coeff = self.coord_mlp(messages)  # [E, 1], invariant scalar
            unit_vectors = relative_vectors / (distances.unsqueeze(-1) + self.eps)  # equivariant
            weighted = coeff * unit_vectors  # [E, 3], equivariant
            delta = scatter_mean(weighted, dst, dim_size=num_nodes)  # equivariant
            new_coords_flat = coords_flat + delta

        return h_new, new_coords_flat


class GeometricEncoder(nn.Module):
    """Dynamic-graph EGNN encoder over the current flow-time coordinates."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        update_coordinates: bool = False,
        knn_k: int = 16,
        use_radius_cutoff: bool = False,
        radius_cutoff: float = 12.0,
        num_rbf: int = 16,
        rbf_min_dist: float = 0.0,
        rbf_max_dist: float = 20.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.knn_k = knn_k
        self.use_radius_cutoff = use_radius_cutoff
        self.radius_cutoff = radius_cutoff
        self.num_rbf = num_rbf
        self.rbf_min_dist = rbf_min_dist
        self.rbf_max_dist = rbf_max_dist
        self.update_coordinates = update_coordinates
        self.eps = eps
        feature_dim = edge_scalar_dim(num_rbf)
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, feature_dim, update_coordinates, dropout, eps) for _ in range(num_layers)]
        )

    def forward(
        self,
        coords: Tensor,
        node_features: Tensor,
        residue_mask: Tensor,
    ) -> Tuple[Tensor, GeometricGraph]:
        """
        Args:
            coords: [B, L, 3] current (flow-time) C-alpha coordinates.
            node_features: [B, L, hidden_dim] initial invariant node features.
            residue_mask: [B, L] bool.

        Returns:
            h_out: [B, L, hidden_dim] rotation/translation-invariant features.
            graph: the :class:`GeometricGraph` built from the *input* coords
                (used unchanged downstream by the vector-field decoder, even
                if internal coordinate refinement is enabled).
        """
        batch_size, length, hidden_dim = node_features.shape
        num_nodes = batch_size * length

        graph = build_geometric_graph(
            coords, residue_mask, k=self.knn_k, use_radius_cutoff=self.use_radius_cutoff,
            radius_cutoff=self.radius_cutoff, eps=self.eps,
        )

        h_flat = node_features.reshape(num_nodes, hidden_dim)
        coords_flat = coords.reshape(num_nodes, 3)
        relative_vectors = graph.relative_vectors
        distances = graph.distances
        edge_scalars = build_edge_scalar_features(graph, self.num_rbf, self.rbf_min_dist, self.rbf_max_dist)

        for layer in self.layers:
            h_flat, coords_flat = layer(
                h_flat, coords_flat, graph.edge_index, edge_scalars, relative_vectors, distances, num_nodes
            )
            if self.update_coordinates and graph.edge_index.shape[1] > 0:
                src, dst = graph.edge_index
                relative_vectors = coords_flat[src] - coords_flat[dst]
                distances = torch.sqrt(relative_vectors.pow(2).sum(dim=-1) + self.eps)
                edge_scalars = torch.cat(
                    [
                        distances.unsqueeze(-1),
                        rbf_encode(distances, self.num_rbf, self.rbf_min_dist, self.rbf_max_dist),
                        graph.sequence_separation_norm.unsqueeze(-1),
                        graph.peptide_neighbor_indicator.unsqueeze(-1),
                    ],
                    dim=-1,
                )

        h_out = h_flat.reshape(batch_size, length, hidden_dim)
        h_out = h_out * residue_mask.unsqueeze(-1).to(h_out.dtype)
        return h_out, graph
