"""Automated E(3)-equivariance test for the full DualGraphFlowModel.

Checks, for a fixed random seed and an explicit numerical tolerance:
  1. Rotation equivariance: v(x @ R, ...) ~= v(x, ...) @ R.
  2. Translation invariance: v(x + b, ...) ~= v(x, ...).

Coordinates are generated with a large, irregular spread so that no two
pairwise distances coincide (avoiding k-NN neighbor-order ties, which
would otherwise make the two forward passes select different graph
topologies and spuriously fail the equivariance check).
"""
from __future__ import annotations

import torch

from protein_flow.config import Config
from protein_flow.models.dual_graph_flow import DualGraphFlowModel

SEED = 12345
ATOL = 1e-4
RTOL = 1e-4


def _random_rotation(generator: torch.Generator) -> torch.Tensor:
    """A random proper (det=+1) rotation matrix."""
    a = torch.randn(3, 3, generator=generator)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diagonal(r))
    q = q * d.unsqueeze(-2)
    if torch.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


def _tie_free_coords(batch: int, length: int, generator: torch.Generator) -> torch.Tensor:
    """Coordinates with irregular per-axis scaling so pairwise distances are
    (with overwhelming probability) all distinct -- avoiding k-NN ties."""
    coords = torch.randn(batch, length, 3, generator=generator)
    scale = torch.tensor([1.0, 2.7182818, 4.6692016])  # e, Feigenbaum constant: irrational-ish, avoids symmetry
    return coords * scale


def _build_small_model(generator: torch.Generator) -> tuple[DualGraphFlowModel, Config]:
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

    torch.manual_seed(SEED)
    model = DualGraphFlowModel(config)
    model.eval()  # disable dropout for a deterministic comparison
    return model, config


def test_full_model_rotation_equivariance():
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_small_model(generator)

    batch, length = 3, 14
    coords = _tie_free_coords(batch, length, generator)
    sequence_embedding = torch.randn(batch, length, config.data.plm_dim, generator=generator)
    residue_types = torch.randint(0, config.data.num_amino_acid_types, (batch, length), generator=generator)
    residue_mask = torch.ones(batch, length, dtype=torch.bool)
    temperature = torch.rand(batch, 1, generator=generator)
    physical_delta_t = torch.rand(batch, 1, generator=generator)
    tau = torch.rand(batch, generator=generator)

    rotation = _random_rotation(generator)
    coords_rotated = coords @ rotation

    with torch.no_grad():
        v = model(coords, tau, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask)
        v_rotated = model(
            coords_rotated, tau, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask
        )

    expected = v @ rotation
    torch.testing.assert_close(v_rotated, expected, atol=ATOL, rtol=RTOL)


def test_full_model_translation_invariance():
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_small_model(generator)

    batch, length = 3, 14
    coords = _tie_free_coords(batch, length, generator)
    sequence_embedding = torch.randn(batch, length, config.data.plm_dim, generator=generator)
    residue_types = torch.randint(0, config.data.num_amino_acid_types, (batch, length), generator=generator)
    residue_mask = torch.ones(batch, length, dtype=torch.bool)
    temperature = torch.rand(batch, 1, generator=generator)
    physical_delta_t = torch.rand(batch, 1, generator=generator)
    tau = torch.rand(batch, generator=generator)

    translation = torch.randn(3, generator=generator) * 10.0
    coords_translated = coords + translation

    with torch.no_grad():
        v = model(coords, tau, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask)
        v_translated = model(
            coords_translated, tau, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask
        )

    torch.testing.assert_close(v_translated, v, atol=ATOL, rtol=RTOL)


def test_full_model_combined_rotation_and_translation():
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_small_model(generator)

    batch, length = 2, 18
    coords = _tie_free_coords(batch, length, generator)
    sequence_embedding = torch.randn(batch, length, config.data.plm_dim, generator=generator)
    residue_types = torch.randint(0, config.data.num_amino_acid_types, (batch, length), generator=generator)
    residue_mask = torch.ones(batch, length, dtype=torch.bool)
    temperature = torch.rand(batch, 1, generator=generator)
    physical_delta_t = torch.rand(batch, 1, generator=generator)
    tau = torch.rand(batch, generator=generator)

    rotation = _random_rotation(generator)
    translation = torch.randn(3, generator=generator) * 10.0
    coords_transformed = coords @ rotation + translation

    with torch.no_grad():
        v = model(coords, tau, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask)
        v_transformed = model(
            coords_transformed, tau, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask
        )

    expected = v @ rotation
    torch.testing.assert_close(v_transformed, expected, atol=ATOL, rtol=RTOL)


def test_equivariance_with_padding_present():
    """Equivariance must hold even when some residues are padding."""
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_small_model(generator)

    batch, length, valid_len = 2, 16, 11
    coords = _tie_free_coords(batch, length, generator)
    sequence_embedding = torch.randn(batch, length, config.data.plm_dim, generator=generator)
    residue_types = torch.randint(0, config.data.num_amino_acid_types, (batch, length), generator=generator)
    residue_mask = torch.zeros(batch, length, dtype=torch.bool)
    residue_mask[:, :valid_len] = True
    temperature = torch.rand(batch, 1, generator=generator)
    physical_delta_t = torch.rand(batch, 1, generator=generator)
    tau = torch.rand(batch, generator=generator)

    rotation = _random_rotation(generator)
    translation = torch.randn(3, generator=generator) * 10.0
    coords_transformed = coords @ rotation + translation

    with torch.no_grad():
        v = model(coords, tau, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask)
        v_transformed = model(
            coords_transformed, tau, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask
        )

    expected = v @ rotation
    # only valid residues are meaningful; padding residues are zeroed by construction.
    torch.testing.assert_close(v_transformed[:, :valid_len], expected[:, :valid_len], atol=ATOL, rtol=RTOL)
    assert torch.all(v[:, valid_len:] == 0.0)
    assert torch.all(v_transformed[:, valid_len:] == 0.0)
