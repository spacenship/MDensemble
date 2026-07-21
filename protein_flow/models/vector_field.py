"""E(3)-equivariant vector-field decoder.

The decoder never predicts raw xyz vectors from an MLP. Instead, for every
geometric-graph edge it predicts *invariant* scalar coefficients from
invariant inputs (fused hidden states, distance, RBF features), and builds
the output velocity as a weighted sum of the *equivariant* relative
position vectors ``x_j - x_i``:

    a_ij = phi_x(h_i, h_j, edge_scalars_ij)
    v_i  = sum_j a_ij * (x_j - x_i) / (||x_j - x_i|| + eps)

This guarantees rotation-equivariance and translation-invariance of the
output by construction, as long as ``h`` is itself invariant.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from protein_flow.geometry.graph import GeometricGraph
from protein_flow.models.geometric_encoder import build_edge_scalar_features, edge_scalar_dim
from protein_flow.utils import masked_mean, scatter_mean


class VectorFieldDecoder(nn.Module):
    """Decodes fused invariant node features into an equivariant [B, L, 3] velocity.

    Combines two equivariant terms built from the same geometric-graph
    edges: a local geometric term and a second ("residual") term with
    independently-learned coefficients. Optionally removes the masked
    center-of-mass velocity from the combined output.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_rbf: int = 16,
        rbf_min_dist: float = 0.0,
        rbf_max_dist: float = 20.0,
        remove_com_velocity: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.rbf_min_dist = rbf_min_dist
        self.rbf_max_dist = rbf_max_dist
        self.remove_com_velocity = remove_com_velocity
        self.eps = eps

        feature_dim = 2 * hidden_dim + edge_scalar_dim(num_rbf)

        def make_coefficient_mlp() -> nn.Module:
            layers = [nn.Linear(feature_dim, hidden_dim), nn.SiLU()]
            for _ in range(max(num_layers - 2, 0)):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*layers)

        self.local_coeff_mlp = make_coefficient_mlp()
        self.residual_coeff_mlp = make_coefficient_mlp()

    def forward(self, h_fused: Tensor, graph: GeometricGraph, residue_mask: Tensor) -> Tensor:
        """
        Args:
            h_fused: [B, L, hidden_dim] invariant fused node features.
            graph: :class:`GeometricGraph` built from the current x_tau.
            residue_mask: [B, L] bool.

        Returns:
            [B, L, 3] predicted velocity field.
        """
        batch_size, length, hidden_dim = h_fused.shape
        num_nodes = batch_size * length
        device, dtype = h_fused.device, h_fused.dtype

        if graph.edge_index.shape[1] == 0:
            return torch.zeros(batch_size, length, 3, device=device, dtype=dtype)

        src, dst = graph.edge_index
        h_flat = h_fused.reshape(num_nodes, hidden_dim)
        edge_scalars = build_edge_scalar_features(graph, self.num_rbf, self.rbf_min_dist, self.rbf_max_dist)
        edge_input = torch.cat([h_flat[src], h_flat[dst], edge_scalars], dim=-1)

        local_coeff = self.local_coeff_mlp(edge_input)  # [E, 1], invariant
        residual_coeff = self.residual_coeff_mlp(edge_input)  # [E, 1], invariant

        unit_vectors = graph.relative_vectors / (graph.distances.unsqueeze(-1) + self.eps)  # equivariant

        combined_coeff = local_coeff + residual_coeff
        weighted = combined_coeff * unit_vectors  # [E, 3], equivariant
        velocity_flat = scatter_mean(weighted, dst, dim_size=num_nodes)  # equivariant

        velocity = velocity_flat.reshape(batch_size, length, 3)
        mask = residue_mask.unsqueeze(-1).to(dtype)
        velocity = velocity * mask

        if self.remove_com_velocity:
            com = masked_mean(velocity, residue_mask, dim=1, eps=self.eps)  # [B, 3], invariant under translation
            velocity = (velocity - com.unsqueeze(1)) * mask

        return velocity
