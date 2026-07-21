"""Full dual-graph conditional-flow-matching model.

Wires together:
  1. SequenceEncoder   -- bidirectional typed sequence-graph over PLM embeddings.
  2. GeometricEncoder   -- dynamic-graph EGNN over current flow-time coordinates.
  3. GatedFusion        -- residue-wise gated fusion of (1) and (2), conditioned
                           on flow time tau, temperature, and physical_delta_t.
  4. VectorFieldDecoder -- E(3)-equivariant velocity-field readout.
"""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn
from torch import Tensor

from protein_flow.config import Config
from protein_flow.flow.solver import integrate_ode
from protein_flow.models.fusion import GatedFusion
from protein_flow.models.geometric_encoder import GeometricEncoder
from protein_flow.models.sequence_encoder import SequenceEncoder
from protein_flow.models.vector_field import VectorFieldDecoder


class DualGraphFlowModel(nn.Module):
    """Predicts the conditional flow-matching velocity field v_theta(x_tau, ...)."""

    def __init__(self, config: Config):
        super().__init__()
        data_cfg = config.data
        model_cfg = config.model
        graph_cfg = model_cfg.graph

        self.sequence_encoder = SequenceEncoder(
            plm_dim=data_cfg.plm_dim,
            num_amino_acid_types=data_cfg.num_amino_acid_types,
            hidden_dim=model_cfg.sequence_encoder.hidden_dim,
            num_layers=model_cfg.sequence_encoder.num_layers,
            dropout=model_cfg.sequence_encoder.dropout,
            use_position_encoding=model_cfg.sequence_encoder.use_position_encoding,
        )

        self.geometric_aa_embedding = nn.Embedding(
            data_cfg.num_amino_acid_types, model_cfg.geometric_encoder.hidden_dim
        )
        self.geometric_encoder = GeometricEncoder(
            hidden_dim=model_cfg.geometric_encoder.hidden_dim,
            num_layers=model_cfg.geometric_encoder.num_layers,
            dropout=model_cfg.geometric_encoder.dropout,
            update_coordinates=model_cfg.geometric_encoder.update_coordinates,
            knn_k=graph_cfg.knn_k,
            use_radius_cutoff=graph_cfg.use_radius_cutoff,
            radius_cutoff=graph_cfg.radius_cutoff,
            num_rbf=graph_cfg.num_rbf,
            rbf_min_dist=graph_cfg.rbf_min_dist,
            rbf_max_dist=graph_cfg.rbf_max_dist,
        )

        self.fusion = GatedFusion(
            seq_hidden_dim=model_cfg.sequence_encoder.hidden_dim,
            geo_hidden_dim=model_cfg.geometric_encoder.hidden_dim,
            fusion_hidden_dim=model_cfg.fusion.hidden_dim,
            condition_dim=model_cfg.fusion.condition_dim,
        )

        self.decoder = VectorFieldDecoder(
            hidden_dim=model_cfg.fusion.hidden_dim,
            num_layers=model_cfg.decoder.num_layers,
            num_rbf=graph_cfg.num_rbf,
            rbf_min_dist=graph_cfg.rbf_min_dist,
            rbf_max_dist=graph_cfg.rbf_max_dist,
            remove_com_velocity=model_cfg.decoder.remove_com_velocity,
        )

    def forward(
        self,
        x_tau: Tensor,
        tau: Tensor,
        sequence_embedding: Tensor,
        residue_types: Tensor,
        temperature: Tensor,
        physical_delta_t: Tensor,
        residue_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            x_tau: [B, L, 3] current flow-time C-alpha coordinates.
            tau: [B] flow time in [0, 1] (NOT physical MD time).
            sequence_embedding: [B, L, D_plm] precomputed PLM residue embeddings.
            residue_types: [B, L] long amino-acid type indices.
            temperature: [B, 1].
            physical_delta_t: [B, 1] physical MD time gap between source/target frames.
            residue_mask: [B, L] bool, True for valid residues.

        Returns:
            predicted_velocity: [B, L, 3].
        """
        h_seq = self.sequence_encoder(sequence_embedding, residue_types, residue_mask)

        h_geo_init = self.geometric_aa_embedding(residue_types)
        h_geo_init = h_geo_init * residue_mask.unsqueeze(-1).to(h_geo_init.dtype)
        h_geo, graph = self.geometric_encoder(x_tau, h_geo_init, residue_mask)

        h_fused = self.fusion(h_seq, h_geo, tau, temperature, physical_delta_t, residue_mask)
        return self.decoder(h_fused, graph, residue_mask)

    def sample(
        self,
        source_coords: Tensor,
        sequence_embedding: Tensor,
        residue_types: Tensor,
        residue_mask: Tensor,
        temperature: Tensor,
        physical_delta_t: Tensor,
        num_steps: int = 50,
        solver: str = "heun",
        return_trajectory: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Integrate dx/dtau = v_theta(x, tau, ...) from tau=0 (source_coords) to tau=1.

        Runs under ``torch.no_grad()``. Returns ``(generated_coords, trajectory)``
        where ``trajectory`` has shape [num_steps + 1, B, L, 3] when
        ``return_trajectory`` is True.
        """
        return integrate_ode(
            self,
            source_coords,
            sequence_embedding,
            residue_types,
            residue_mask,
            temperature,
            physical_delta_t,
            num_steps=num_steps,
            solver=solver,
            return_trajectory=return_trajectory,
        )
