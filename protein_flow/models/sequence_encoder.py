"""Bidirectional typed sequence-graph encoder.

Note on terminology: the sequence graph contains both forward (i -> i+1)
and backward (i+1 -> i) peptide edges. Because both directions are present
it is **not** a DAG -- it is referred to throughout as a bidirectional
typed sequence graph.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from protein_flow.geometry.graph import NUM_SEQUENCE_EDGE_TYPES, build_sequence_graph
from protein_flow.utils import scatter_mean, sinusoidal_embedding


class TypedMessageLayer(nn.Module):
    """One bidirectional typed message-passing layer over the sequence graph.

    Each edge type (forward / backward) is distinguished via a learned
    edge-type embedding fed into a shared message MLP, with residual
    connection, LayerNorm, and dropout.
    """

    def __init__(self, hidden_dim: int, num_edge_types: int, dropout: float):
        super().__init__()
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: Tensor, edge_index: Tensor, edge_type: Tensor, num_nodes: int) -> Tensor:
        if edge_index.shape[1] == 0:
            return self.norm(h)
        src, dst = edge_index
        edge_embed = self.edge_type_embedding(edge_type)
        message_input = torch.cat([h[src], h[dst], edge_embed], dim=-1)
        messages = self.message_mlp(message_input)
        aggregated = scatter_mean(messages, dst, dim_size=num_nodes)
        update = self.update_mlp(torch.cat([h, aggregated], dim=-1))
        return self.norm(h + self.dropout(update))


class SequenceEncoder(nn.Module):
    """Encodes precomputed PLM embeddings over the bidirectional peptide graph.

    Complexity is O(L) in sequence length since the graph has at most
    2 * (L - 1) edges per protein.
    """

    def __init__(
        self,
        plm_dim: int,
        num_amino_acid_types: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_position_encoding: bool = True,
    ):
        super().__init__()
        if not 2 <= num_layers <= 4:
            raise ValueError("SequenceEncoder expects 2-4 message-passing layers")
        self.hidden_dim = hidden_dim
        self.use_position_encoding = use_position_encoding
        self.input_proj = nn.Linear(plm_dim, hidden_dim)
        self.aa_embedding = nn.Embedding(num_amino_acid_types, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [TypedMessageLayer(hidden_dim, NUM_SEQUENCE_EDGE_TYPES, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        sequence_embedding: Tensor,
        residue_types: Tensor,
        residue_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            sequence_embedding: [B, L, D_plm] precomputed PLM residue embeddings.
            residue_types: [B, L] long amino-acid type indices.
            residue_mask: [B, L] bool, True for valid residues.

        Returns:
            [B, L, hidden_dim] sequence representation, zeroed at padding.
        """
        batch_size, length, _ = sequence_embedding.shape
        h = self.input_proj(sequence_embedding) + self.aa_embedding(residue_types)

        if self.use_position_encoding:
            denom = max(length - 1, 1)
            positions = torch.arange(length, device=h.device, dtype=h.dtype).unsqueeze(0) / denom
            positions = positions.expand(batch_size, length)
            h = h + sinusoidal_embedding(positions, self.hidden_dim)

        h = self.input_norm(h)
        h_flat = h.reshape(batch_size * length, self.hidden_dim)

        graph = build_sequence_graph(residue_mask)
        for layer in self.layers:
            h_flat = layer(h_flat, graph.edge_index, graph.edge_type, num_nodes=batch_size * length)

        h_out = h_flat.reshape(batch_size, length, self.hidden_dim)
        return h_out * residue_mask.unsqueeze(-1).to(h_out.dtype)
