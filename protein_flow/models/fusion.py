"""Gated fusion of sequence-graph and geometric-graph residue representations."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from protein_flow.utils import sinusoidal_embedding


class ConditionEncoder(nn.Module):
    """Encodes flow-time tau, temperature, and physical_delta_t as one condition vector.

    tau is a purely mathematical flow-interpolation variable in [0, 1] and
    is embedded separately from physical_delta_t (the physical MD time gap
    between source and target frames) -- the two must never be conflated.
    """

    def __init__(self, condition_dim: int):
        super().__init__()
        self.condition_dim = condition_dim
        self.mix_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(self, tau: Tensor, temperature: Tensor, physical_delta_t: Tensor) -> Tensor:
        """
        Args:
            tau: [B] flow time in [0, 1].
            temperature: [B, 1].
            physical_delta_t: [B, 1] physical MD time gap (e.g. picoseconds).

        Returns:
            [B, condition_dim] combined condition embedding.
        """
        tau_embed = sinusoidal_embedding(tau, self.condition_dim)
        temperature_embed = sinusoidal_embedding(temperature.squeeze(-1), self.condition_dim)
        log_delta_t = torch.log1p(physical_delta_t.squeeze(-1).clamp(min=0.0))
        delta_t_embed = sinusoidal_embedding(log_delta_t, self.condition_dim)
        combined = tau_embed + temperature_embed + delta_t_embed
        return self.mix_mlp(combined)


class GatedFusion(nn.Module):
    """Residue-wise gated fusion of sequence and geometric representations."""

    def __init__(self, seq_hidden_dim: int, geo_hidden_dim: int, fusion_hidden_dim: int, condition_dim: int):
        super().__init__()
        self.condition_encoder = ConditionEncoder(condition_dim)
        self.proj_seq = nn.Linear(seq_hidden_dim, fusion_hidden_dim)
        self.proj_geo = nn.Linear(geo_hidden_dim, fusion_hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim + geo_hidden_dim + condition_dim, fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(
        self,
        h_seq: Tensor,
        h_geo: Tensor,
        tau: Tensor,
        temperature: Tensor,
        physical_delta_t: Tensor,
        residue_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            h_seq: [B, L, seq_hidden_dim] invariant sequence representation.
            h_geo: [B, L, geo_hidden_dim] invariant geometric representation.
            tau: [B].
            temperature: [B, 1].
            physical_delta_t: [B, 1].
            residue_mask: [B, L] bool.

        Returns:
            [B, L, fusion_hidden_dim] fused, mask-zeroed representation.
        """
        batch_size, length, _ = h_seq.shape
        condition = self.condition_encoder(tau, temperature, physical_delta_t)  # [B, condition_dim]
        condition_expanded = condition.unsqueeze(1).expand(batch_size, length, condition.shape[-1])

        gate_input = torch.cat([h_seq, h_geo, condition_expanded], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))  # [B, L, 1]

        fused = gate * self.proj_seq(h_seq) + (1.0 - gate) * self.proj_geo(h_geo)
        return fused * residue_mask.unsqueeze(-1).to(fused.dtype)
