"""Full dual-graph conditional-flow-matching model.

Wires together:
  1. SequenceEncoder   -- bidirectional typed sequence-graph over PLM embeddings.
  2. GeometricEncoder   -- dynamic-graph EGNN over current flow-time coordinates.
  3. GatedFusion        -- per-particle gated fusion of (1) and (2), conditioned
                           on flow time tau, temperature, and physical_delta_t.
  4. VectorFieldDecoder -- SE(3)-equivariant velocity-field readout.

The sequence side is always **residue-level** while the geometric side runs
over whatever particles carry the flow: C-alpha atoms (one per residue) or
every heavy atom. When the particles are atoms, the residue-level sequence
representation is broadcast down to each atom of its residue before fusion,
so a residue's language-model context reaches all of its atoms.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from protein_flow.config import Config
from protein_flow.data.topology import NUM_ELEMENT_TYPES
from protein_flow.flow.solver import integrate_ode
from protein_flow.models.esm_encoder import EsmSequenceEncoder
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

        self.representation = data_cfg.representation

        # Optional in-graph PLM. When enabled the model produces its own
        # residue embeddings from token ids instead of consuming precomputed
        # ones, so the PLM is trained end-to-end with everything else.
        self.esm_encoder = None
        if model_cfg.esm.enabled:
            self.esm_encoder = EsmSequenceEncoder(
                model_name=model_cfg.esm.model_name,
                trainable=model_cfg.esm.trainable,
                gradient_checkpointing=model_cfg.esm.gradient_checkpointing,
                num_frozen_layers=model_cfg.esm.num_frozen_layers,
            )
            if self.esm_encoder.output_dim != data_cfg.plm_dim:
                raise ValueError(
                    f"data.plm_dim={data_cfg.plm_dim} does not match the hidden size of "
                    f"{model_cfg.esm.model_name} ({self.esm_encoder.output_dim}); set them equal."
                )
        self.geometric_aa_embedding = nn.Embedding(
            data_cfg.num_amino_acid_types, model_cfg.geometric_encoder.hidden_dim
        )
        # All-atom mode additionally distinguishes atoms *within* a residue
        # by chemical element; padding maps to index 0. Not created in the
        # C-alpha representation, where it would be a dead parameter that
        # never receives gradient.
        self.element_embedding = (
            nn.Embedding(NUM_ELEMENT_TYPES, model_cfg.geometric_encoder.hidden_dim, padding_idx=0)
            if data_cfg.representation == "heavy_atom"
            else None
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
            use_chirality_features=model_cfg.geometric_encoder.use_chirality_features,
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
        atom_mask: Optional[Tensor] = None,
        atom_residue_index: Optional[Tensor] = None,
        atom_element: Optional[Tensor] = None,
        ca_atom_index: Optional[Tensor] = None,
        esm_input_ids: Optional[Tensor] = None,
        esm_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x_tau: [B, N, 3] current flow-time particle coordinates. N == L
                in the C-alpha representation; N is the heavy-atom count in
                the all-atom representation.
            tau: [B] flow time in [0, 1] (NOT physical MD time).
            sequence_embedding: [B, L, D_plm] precomputed PLM residue embeddings.
            residue_types: [B, L] long amino-acid type indices.
            temperature: [B, 1].
            physical_delta_t: [B, 1] physical MD time gap between source/target frames.
            residue_mask: [B, L] bool, True for valid residues.
            atom_mask: [B, N] bool particle validity; all-atom only.
            atom_residue_index: [B, N] residue index per atom; all-atom only.
            atom_element: [B, N] element id per atom; all-atom only.
            ca_atom_index: [B, L] atom index of each residue's CA; all-atom only.
            esm_input_ids: [B, T] ESM token ids; required when in-graph ESM
                is enabled, in which case ``sequence_embedding`` is ignored
                and the embedding is computed (and fine-tuned) here instead.
            esm_attention_mask: [B, T] 1 for real tokens.

        Passing the four optional atom tensors switches the geometric side
        to all-atom; omitting them keeps the C-alpha behaviour.

        Returns:
            predicted_velocity: [B, N, 3].
        """
        is_all_atom = atom_residue_index is not None
        particle_mask = atom_mask if is_all_atom else residue_mask

        if self.esm_encoder is not None:
            if esm_input_ids is None:
                raise ValueError(
                    "model.esm.enabled is True but no esm_input_ids were provided; the dataset "
                    "must be built with esm_tokenizer_name set."
                )
            sequence_embedding = self.esm_encoder(
                esm_input_ids, esm_attention_mask, num_residues=residue_mask.shape[1]
            )

        # Sequence side always runs at residue level.
        h_seq = self.sequence_encoder(sequence_embedding, residue_types, residue_mask)

        if is_all_atom:
            # Broadcast residue-level features/types down to that residue's atoms.
            gather_index = atom_residue_index.unsqueeze(-1).expand(-1, -1, h_seq.shape[-1])
            h_seq_particles = torch.gather(h_seq, 1, gather_index)
            if self.element_embedding is None:
                raise ValueError(
                    "Model was built with representation='ca' but all-atom tensors were passed; "
                    "set data.representation='heavy_atom' in the config."
                )
            atom_residue_types = torch.gather(residue_types, 1, atom_residue_index)
            h_geo_init = self.geometric_aa_embedding(atom_residue_types) + self.element_embedding(atom_element)
        else:
            h_seq_particles = h_seq
            h_geo_init = self.geometric_aa_embedding(residue_types)

        h_geo_init = h_geo_init * particle_mask.unsqueeze(-1).to(h_geo_init.dtype)
        h_geo, graph = self.geometric_encoder(
            x_tau, h_geo_init, particle_mask,
            atom_residue_index=atom_residue_index,
            ca_atom_index=ca_atom_index,
            residue_level_mask=residue_mask if is_all_atom else None,
        )

        h_fused = self.fusion(
            h_seq_particles, h_geo, tau, temperature, physical_delta_t, particle_mask
        )
        return self.decoder(h_fused, graph, particle_mask)

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
        **atom_inputs: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Integrate dx/dtau = v_theta(x, tau, ...) from tau=0 (source_coords) to tau=1.

        Runs under ``torch.no_grad()``. Returns ``(generated_coords, trajectory)``
        where ``trajectory`` has shape [num_steps + 1, B, N, 3] when
        ``return_trajectory`` is True. Any all-atom tensors
        (``atom_mask``/``atom_residue_index``/``atom_element``/``ca_atom_index``)
        may be passed as keyword arguments and are forwarded to every
        velocity evaluation along the trajectory.
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
            **atom_inputs,
        )
