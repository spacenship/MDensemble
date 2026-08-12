"""EGNN-style SE(3)-equivariant geometric encoder.

Node hidden features produced by this module are rotation/translation
**invariant** (they are built only from pairwise distances and other scalar
edge features, never from raw relative-vector components). Equivariant
vector outputs are constructed downstream in
:mod:`protein_flow.models.vector_field` by weighting the *relative*
position vectors ``x_j - x_i`` with these invariant scalars -- this module
never emits a raw xyz vector from an MLP.

By default (``use_chirality_features=True``) the initial node features
also include the signed backbone-dihedral pseudo-scalar from
:mod:`protein_flow.geometry.chirality`, which is invariant under proper
rotation + translation but flips sign under reflection. Without it, every
feature used anywhere in this module is a true scalar (invariant under
reflection too), which would make the network E(3)-equivariant -- unable
to distinguish a structure from its mirror image, wrong for real chiral
proteins. With it, the network is SE(3)-equivariant: equivariant under
rotation + translation, but not under reflection.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor

from protein_flow.geometry.chirality import compute_particle_dihedral
from protein_flow.geometry.features import rbf_encode
from protein_flow.geometry.graph import GeometricGraph, build_geometric_graph
from protein_flow.utils import scatter_mean, sinusoidal_embedding


#: Number of extra edge scalars contributed by the flow state; see
#: :func:`flow_state_edge_features`.
FLOW_STATE_EDGE_FEATURES = 3


def edge_scalar_dim(num_rbf: int, with_flow_state: bool = False) -> int:
    """Dimensionality of
    [distance, rbf(distance), seq_sep_norm, peptide_indicator, same_residue_indicator]
    plus, when the flow state is carried, the three scalars from
    :func:`flow_state_edge_features`.

    ``same_residue_indicator`` is identically zero in the C-alpha
    representation (one particle per residue) and only carries signal in
    the all-atom representation, where it distinguishes intra-residue
    edges from inter-residue contacts.
    """
    base = 1 + num_rbf + 1 + 1 + 1
    return base + (FLOW_STATE_EDGE_FEATURES if with_flow_state else 0)


def flow_state_edge_features(
    flow_state_flat: Tensor,
    edge_index: Tensor,
    relative_vectors: Tensor,
    distances: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Rotation-invariant projections of the flow state onto each edge: [E, 3].

    ``flow_state_flat`` is a per-particle 3-vector living in *displacement*
    space (a point on the noise-to-delta path), so it rotates with the
    structure and is already translation-invariant. The three scalars below
    are the complete first-order invariant description of it relative to an
    edge, and are what lets the network condition on the flow state without
    ever feeding raw xyz components through an MLP:

      s_i . u_ij   -- how much the centre's own displacement points at j
      s_j . u_ij   -- how much the neighbour's displacement points at j
      ||s_j - s_i|| -- how differently the two ends are being displaced

    Caution when combining with ``flow.noise_smoothing_rounds``: the third
    scalar measures exactly what smoothing removes, so it shrinks as the base
    noise is made collective. Measured on real batches, its mean falls
    6.00 (white) -> 2.65 (4 rounds) -> 1.43 (8 rounds) while the first two are
    unchanged at ~2.64. That starves a third of the flow-state signal, and is
    the leading explanation for the correlated-noise arm learning the field
    worse (velocity cosine 0.457 versus 0.695 for white noise) despite
    scoring better on the physics terms. A scale-free form such as
    ``||s_j - s_i|| / (||s_i|| + ||s_j||)`` would decouple the two; it has not
    been tried.

    Note ``src`` is the neighbour j and ``dst`` the centre i (see
    :class:`~protein_flow.geometry.graph.GeometricGraph`).
    """
    src, dst = edge_index
    unit_vectors = relative_vectors / (distances.unsqueeze(-1) + eps)
    state_center = flow_state_flat[dst]
    state_neighbor = flow_state_flat[src]
    return torch.stack(
        [
            (state_center * unit_vectors).sum(dim=-1),
            (state_neighbor * unit_vectors).sum(dim=-1),
            (state_neighbor - state_center).norm(dim=-1),
        ],
        dim=-1,
    )


def build_edge_scalar_features(
    graph: GeometricGraph,
    num_rbf: int,
    rbf_min_dist: float,
    rbf_max_dist: float,
    flow_state_flat: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    rbf = rbf_encode(graph.distances, num_rbf, rbf_min_dist, rbf_max_dist)
    features = [
        graph.distances.unsqueeze(-1),
        rbf,
        graph.sequence_separation_norm.unsqueeze(-1),
        graph.peptide_neighbor_indicator.unsqueeze(-1),
        graph.same_residue_indicator.unsqueeze(-1),
    ]
    if flow_state_flat is not None:
        features.append(
            flow_state_edge_features(
                flow_state_flat, graph.edge_index, graph.relative_vectors, graph.distances, eps
            )
        )
    return torch.cat(features, dim=-1)


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
        use_chirality_features: bool = True,
        gradient_checkpointing: bool = False,
        use_flow_state: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.use_chirality_features = use_chirality_features
        self.gradient_checkpointing = gradient_checkpointing
        self.use_flow_state = use_flow_state
        self.knn_k = knn_k
        self.use_radius_cutoff = use_radius_cutoff
        self.radius_cutoff = radius_cutoff
        self.num_rbf = num_rbf
        self.rbf_min_dist = rbf_min_dist
        self.rbf_max_dist = rbf_max_dist
        self.update_coordinates = update_coordinates
        self.eps = eps
        self.hidden_dim = hidden_dim
        feature_dim = edge_scalar_dim(num_rbf, with_flow_state=use_flow_state)
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, feature_dim, update_coordinates, dropout, eps) for _ in range(num_layers)]
        )

    def forward(
        self,
        coords: Tensor,
        node_features: Tensor,
        residue_mask: Tensor,
        atom_residue_index: Optional[Tensor] = None,
        ca_atom_index: Optional[Tensor] = None,
        residue_level_mask: Optional[Tensor] = None,
        flow_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, GeometricGraph]:
        """
        Args:
            coords: [B, N, 3] the coordinates the graph is built from. In the
                coordinate-space flow this is the current flow-time position;
                in the displacement flow it is the (fixed) source structure.
            node_features: [B, N, hidden_dim] initial invariant node features.
            residue_mask: [B, N] bool validity of each *particle*.
            atom_residue_index: [B, N] residue index per atom; all-atom only.
            ca_atom_index: [B, L] atom index of each residue's CA; all-atom only.
            residue_level_mask: [B, L] residue validity; all-atom only.
            flow_state: [B, N, 3] point on the displacement path. Enters only
                through rotation-invariant projections -- a per-node
                magnitude embedding and the three edge scalars of
                :func:`flow_state_edge_features` -- so the encoder's output
                stays invariant. Required when the encoder was built with
                ``use_flow_state=True``.

        The three optional topology arguments are what turn this into an
        all-atom encoder: they let sequence separation, peptide adjacency, and
        the backbone chirality feature stay defined over *residues* even though
        the nodes are now individual atoms. Omitting them recovers the
        C-alpha behaviour exactly.

        Returns:
            h_out: [B, N, hidden_dim] rotation/translation-invariant features.
            graph: the :class:`GeometricGraph` built from the *input* coords
                (used unchanged downstream by the vector-field decoder, even
                if internal coordinate refinement is enabled).
        """
        batch_size, length, hidden_dim = node_features.shape
        num_nodes = batch_size * length

        if self.use_flow_state and flow_state is None:
            raise ValueError(
                "GeometricEncoder was built with use_flow_state=True but no flow_state was "
                "passed; the edge features it was sized for would be missing."
            )
        if flow_state is not None and not self.use_flow_state:
            raise ValueError(
                "flow_state was passed to a GeometricEncoder built with use_flow_state=False; "
                "its layers are sized for the smaller edge-feature vector."
            )

        if flow_state is not None:
            # Magnitude only: the direction of the displacement cannot enter a
            # node feature without breaking invariance, and reaches the
            # network through the edge scalars instead. Masked because
            # sinusoidal_embedding(0) is not zero -- its cosine half is all
            # ones -- so padding particles would otherwise pick up features,
            # matching how the chirality embedding is handled just below.
            state_magnitude = flow_state.norm(dim=-1)
            state_embedding = sinusoidal_embedding(state_magnitude, hidden_dim)
            node_features = node_features + state_embedding * residue_mask.unsqueeze(-1).to(
                node_features.dtype
            )

        if self.use_chirality_features:
            dihedral, valid = compute_particle_dihedral(
                coords, residue_mask, ca_atom_index, atom_residue_index, residue_level_mask, eps=self.eps
            )
            chirality_embedding = sinusoidal_embedding(dihedral, hidden_dim) * valid.unsqueeze(-1).to(coords.dtype)
            node_features = node_features + chirality_embedding

        chain_length = None
        if residue_level_mask is not None:
            chain_length = residue_level_mask.sum(dim=1)

        graph = build_geometric_graph(
            coords, residue_mask, k=self.knn_k, use_radius_cutoff=self.use_radius_cutoff,
            radius_cutoff=self.radius_cutoff, eps=self.eps,
            separation_index=atom_residue_index, chain_length=chain_length,
        )

        h_flat = node_features.reshape(num_nodes, hidden_dim)
        coords_flat = coords.reshape(num_nodes, 3)
        flow_state_flat = flow_state.reshape(num_nodes, 3) if flow_state is not None else None
        relative_vectors = graph.relative_vectors
        distances = graph.distances
        edge_scalars = build_edge_scalar_features(
            graph, self.num_rbf, self.rbf_min_dist, self.rbf_max_dist, flow_state_flat, self.eps
        )

        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for layer in self.layers:
            if checkpointing:
                # Each layer's [E, hidden_dim] edge activations are the single
                # largest allocation in the model at atom resolution -- with
                # k=16 neighbours over 4 particles per residue, a 512-residue
                # sample carries 32,768 edges per layer. Recomputing them in
                # the backward pass is what makes large batches fit.
                h_flat, coords_flat = torch.utils.checkpoint.checkpoint(
                    layer, h_flat, coords_flat, graph.edge_index, edge_scalars,
                    relative_vectors, distances, num_nodes, use_reentrant=False,
                )
            else:
                h_flat, coords_flat = layer(
                    h_flat, coords_flat, graph.edge_index, edge_scalars,
                    relative_vectors, distances, num_nodes,
                )
            if self.update_coordinates and graph.edge_index.shape[1] > 0:
                src, dst = graph.edge_index
                relative_vectors = coords_flat[src] - coords_flat[dst]
                distances = torch.sqrt(relative_vectors.pow(2).sum(dim=-1) + self.eps)
                refreshed = [
                    distances.unsqueeze(-1),
                    rbf_encode(distances, self.num_rbf, self.rbf_min_dist, self.rbf_max_dist),
                    graph.sequence_separation_norm.unsqueeze(-1),
                    graph.peptide_neighbor_indicator.unsqueeze(-1),
                    graph.same_residue_indicator.unsqueeze(-1),
                ]
                if flow_state_flat is not None:
                    # The edge unit vectors moved with the refined coordinates,
                    # so the flow-state projections onto them have to be redone.
                    refreshed.append(
                        flow_state_edge_features(
                            flow_state_flat, graph.edge_index, relative_vectors, distances, self.eps
                        )
                    )
                edge_scalars = torch.cat(refreshed, dim=-1)

        h_out = h_flat.reshape(batch_size, length, hidden_dim)
        h_out = h_out * residue_mask.unsqueeze(-1).to(h_out.dtype)
        return h_out, graph
