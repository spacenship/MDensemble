"""Gated fusion of sequence-graph and geometric-graph residue representations."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from protein_flow.utils import sinusoidal_embedding


class ConditionEncoder(nn.Module):
    """Encodes flow-time tau, temperature, and physical_delta_t as one condition vector.

    tau is a purely mathematical flow-interpolation variable in [0, 1] and
    is embedded separately from physical_delta_t (the physical MD time gap
    between source and target frames) -- the two must never be conflated.

    Two details here are load-bearing, and getting either wrong silently
    produces a condition vector that carries no information at all.

    **Every scalar is rescaled onto a common [0, embedding_scale] range
    before it is embedded.** ``sinusoidal_embedding`` lays its frequencies
    out geometrically from 1.0 down to ``1 / max_period``, so it resolves
    inputs that span hundreds of units -- the diffusion-timestep convention
    of feeding t in [0, 1000]. Feeding it raw values breaks in both
    directions: tau in [0, 1] leaves nearly every band at ``sin(x) ~ 0,
    cos(x) ~ 1``, and raw Kelvin (320-450) aliases the high-frequency bands
    into noise. Measured on the 34.5k-step run that first exposed this, the
    trained encoder had collapsed to
        cosine(condition@tau=0, condition@tau=1) = 0.999977
        cosine(condition@320K,  condition@450K ) = 0.998356
    i.e. the model could see neither its own flow time nor the simulation
    temperature.

    **The three embeddings are concatenated, not summed.** Summing was the
    second half of the same failure: each embedding is dominated by the same
    near-constant "all cosines ~ 1" direction, so adding them mostly
    accumulates that shared constant and buries what little the individual
    scalars contribute.
    """

    def __init__(
        self,
        condition_dim: int,
        temperature_min: float = 320.0,
        temperature_max: float = 450.0,
        delta_t_max: float = 1000.0,
        embedding_scale: float = 1000.0,
    ):
        super().__init__()
        if temperature_max <= temperature_min:
            raise ValueError(
                f"fusion.temperature_max ({temperature_max}) must exceed "
                f"fusion.temperature_min ({temperature_min})"
            )
        if delta_t_max <= 0.0:
            raise ValueError("fusion.delta_t_max must be > 0")
        self.condition_dim = condition_dim
        self.temperature_min = temperature_min
        self.temperature_span = temperature_max - temperature_min
        self.log_delta_t_max = math.log1p(delta_t_max)
        self.embedding_scale = embedding_scale
        self.mix_mlp = nn.Sequential(
            nn.Linear(3 * condition_dim, condition_dim),
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
        scale = self.embedding_scale
        # Values outside the configured ranges stay monotonic rather than
        # being clamped: an unseen temperature should land outside the
        # training span, not on top of its nearest endpoint.
        normalized_temperature = (temperature.squeeze(-1) - self.temperature_min) / self.temperature_span
        log_delta_t = torch.log1p(physical_delta_t.squeeze(-1).clamp(min=0.0)) / self.log_delta_t_max

        tau_embed = sinusoidal_embedding(tau * scale, self.condition_dim)
        temperature_embed = sinusoidal_embedding(normalized_temperature * scale, self.condition_dim)
        delta_t_embed = sinusoidal_embedding(log_delta_t * scale, self.condition_dim)

        combined = torch.cat([tau_embed, temperature_embed, delta_t_embed], dim=-1)
        return self.mix_mlp(combined)


class GatedFusion(nn.Module):
    """Residue-wise gated fusion of sequence and geometric representations.

    The condition reaches the output through two paths. The gate chooses
    *which* representation to read, but it is a single scalar in (0, 1) per
    particle and so cannot change the scale -- let alone the sign -- of what
    the decoder eventually emits. FiLM supplies that missing degree of
    freedom by modulating the fused channels directly.

    Both are needed. The velocity this model must produce depends on the
    condition in exactly the way a gate cannot express: measured on the run
    that motivated this, the scalar alpha minimising ||alpha*pred - target||
    ran from -86 at tau=0.05 to +86 at tau=0.95. With no way to flip sign
    with tau, gradients from the two halves of the tau range cancelled and
    the predicted velocity collapsed to 0.24% of the target's magnitude.

    The FiLM projection is zero-initialised, so a freshly built model starts
    exactly at the plain gated-fusion behaviour and departs from it only as
    the condition earns its keep.
    """

    def __init__(
        self,
        seq_hidden_dim: int,
        geo_hidden_dim: int,
        fusion_hidden_dim: int,
        condition_dim: int,
        temperature_min: float = 320.0,
        temperature_max: float = 450.0,
        delta_t_max: float = 1000.0,
        embedding_scale: float = 1000.0,
        film_conditioning: bool = True,
    ):
        super().__init__()
        self.condition_encoder = ConditionEncoder(
            condition_dim,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
            delta_t_max=delta_t_max,
            embedding_scale=embedding_scale,
        )
        self.proj_seq = nn.Linear(seq_hidden_dim, fusion_hidden_dim)
        self.proj_geo = nn.Linear(geo_hidden_dim, fusion_hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim + geo_hidden_dim + condition_dim, fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(fusion_hidden_dim, 1),
        )
        self.film = None
        if film_conditioning:
            self.film = nn.Linear(condition_dim, 2 * fusion_hidden_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

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

        if self.film is not None:
            # Per-channel affine modulation shared across particles; the
            # representation stays invariant, so the decoder's equivariant
            # construction is untouched.
            scale, shift = self.film(condition).chunk(2, dim=-1)  # [B, fusion_hidden_dim] each
            fused = fused * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return fused * residue_mask.unsqueeze(-1).to(fused.dtype)
