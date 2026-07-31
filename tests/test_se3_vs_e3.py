"""Confirms the model is SE(3)-equivariant (proper rotation + translation)
but, with the default chirality feature enabled, NOT E(3)-equivariant
(reflections are no longer a symmetry) -- and that disabling the chirality
feature restores the old E(3)-equivariant (achiral) behavior.
"""
from __future__ import annotations

import torch

from protein_flow.config import Config
from protein_flow.models.dual_graph_flow import DualGraphFlowModel

SEED = 999
ATOL = 1e-4
RTOL = 1e-4


def _proper_rotation(generator: torch.Generator) -> torch.Tensor:
    a = torch.randn(3, 3, generator=generator)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diagonal(r))
    q = q * d.unsqueeze(-2)
    if torch.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


def _reflection() -> torch.Tensor:
    return torch.diag(torch.tensor([1.0, 1.0, -1.0]))


def _tie_free_coords(batch: int, length: int, generator: torch.Generator) -> torch.Tensor:
    coords = torch.randn(batch, length, 3, generator=generator)
    scale = torch.tensor([1.0, 2.7182818, 4.6692016])
    return coords * scale


def _build_model(use_chirality_features: bool) -> tuple[DualGraphFlowModel, Config]:
    config = Config()
    config.data.plm_dim = 20
    config.data.num_amino_acid_types = 22
    config.model.sequence_encoder.hidden_dim = 24
    config.model.geometric_encoder.hidden_dim = 24
    config.model.fusion.hidden_dim = 24
    config.model.fusion.condition_dim = 12
    config.model.graph.knn_k = 5
    config.model.graph.num_rbf = 10
    config.model.sequence_encoder.dropout = 0.0
    config.model.geometric_encoder.dropout = 0.0
    config.model.geometric_encoder.use_chirality_features = use_chirality_features

    torch.manual_seed(SEED)
    model = DualGraphFlowModel(config)
    model.eval()
    return model, config


def _random_inputs(config: Config, generator: torch.Generator, batch=2, length=16):
    coords = _tie_free_coords(batch, length, generator)
    sequence_embedding = torch.randn(batch, length, config.data.plm_dim, generator=generator)
    residue_types = torch.randint(0, config.data.num_amino_acid_types, (batch, length), generator=generator)
    residue_mask = torch.ones(batch, length, dtype=torch.bool)
    temperature = torch.rand(batch, 1, generator=generator)
    physical_delta_t = torch.rand(batch, 1, generator=generator)
    tau = torch.rand(batch, generator=generator)
    return coords, sequence_embedding, residue_types, residue_mask, temperature, physical_delta_t, tau


def test_proper_rotation_equivariance_holds_with_chirality_enabled():
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_model(use_chirality_features=True)
    coords, seq_emb, residue_types, mask, temperature, delta_t, tau = _random_inputs(config, generator)

    rotation = _proper_rotation(generator)
    with torch.no_grad():
        v = model(coords, tau, seq_emb, residue_types, temperature, delta_t, mask)
        v_rotated = model(coords @ rotation, tau, seq_emb, residue_types, temperature, delta_t, mask)

    torch.testing.assert_close(v_rotated, v @ rotation, atol=ATOL, rtol=RTOL)


def test_reflection_equivariance_fails_with_chirality_enabled():
    """The whole point of the chirality feature: mirror images must now be
    treated differently, so the naive v(xO) == v(x)O identity must break."""
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_model(use_chirality_features=True)
    coords, seq_emb, residue_types, mask, temperature, delta_t, tau = _random_inputs(config, generator)

    reflection = _reflection()
    with torch.no_grad():
        v = model(coords, tau, seq_emb, residue_types, temperature, delta_t, mask)
        v_reflected = model(coords @ reflection, tau, seq_emb, residue_types, temperature, delta_t, mask)

    naive_expected = v @ reflection
    max_diff = (v_reflected - naive_expected).abs().max().item()
    assert max_diff > 1e-2, f"expected reflection symmetry to be broken, but diff was only {max_diff}"


def test_reflection_equivariance_holds_when_chirality_disabled():
    """Ablation: disabling the chirality feature restores the old,
    fully-achiral E(3)-equivariant behavior (reflections ARE a symmetry)."""
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_model(use_chirality_features=False)
    coords, seq_emb, residue_types, mask, temperature, delta_t, tau = _random_inputs(config, generator)

    reflection = _reflection()
    with torch.no_grad():
        v = model(coords, tau, seq_emb, residue_types, temperature, delta_t, mask)
        v_reflected = model(coords @ reflection, tau, seq_emb, residue_types, temperature, delta_t, mask)

    torch.testing.assert_close(v_reflected, v @ reflection, atol=ATOL, rtol=RTOL)


def test_translation_invariance_still_holds_with_chirality_enabled():
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_model(use_chirality_features=True)
    coords, seq_emb, residue_types, mask, temperature, delta_t, tau = _random_inputs(config, generator)

    translation = torch.randn(3, generator=generator) * 10.0
    with torch.no_grad():
        v = model(coords, tau, seq_emb, residue_types, temperature, delta_t, mask)
        v_translated = model(coords + translation, tau, seq_emb, residue_types, temperature, delta_t, mask)

    torch.testing.assert_close(v_translated, v, atol=ATOL, rtol=RTOL)
