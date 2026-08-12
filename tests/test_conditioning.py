"""Conditioning: tau, temperature and physical_delta_t reaching the velocity.

These are regression tests for a silent failure that cost a 34,500-step run.
The condition vector had collapsed to a constant -- cosine 0.999977 between
tau=0 and tau=1, 0.998356 between 320 K and 450 K -- so the model could see
neither its own flow time nor the simulation temperature, and the predicted
velocity fell to 0.24% of the target's magnitude. Nothing in the loss curve
distinguished that from ordinary slow progress, which is why the properties
below are asserted directly.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_flow.config import Config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.models.fusion import ConditionEncoder, GatedFusion
from protein_flow.train import compute_losses

# mdCATH's five simulation temperatures.
TEMPERATURES = (320.0, 348.0, 379.0, 413.0, 450.0)
# Two conditions this far apart must not look alike to the network. The
# failure being guarded against sat above 0.998, so this leaves a wide
# margin while still failing loudly if the embedding scaling regresses.
MAX_COSINE = 0.95


def _condition(encoder, tau: float, temperature: float, delta_t: float = 5.0) -> torch.Tensor:
    with torch.no_grad():
        return encoder(
            torch.tensor([tau]), torch.tensor([[temperature]]), torch.tensor([[delta_t]])
        )


def test_condition_encoder_separates_flow_times():
    torch.manual_seed(0)
    encoder = ConditionEncoder(512).eval()
    start = _condition(encoder, 0.0, 379.0)
    end = _condition(encoder, 1.0, 379.0)
    assert F.cosine_similarity(start, end).item() < MAX_COSINE


def test_condition_encoder_separates_temperatures():
    torch.manual_seed(0)
    encoder = ConditionEncoder(512).eval()
    conditions = [_condition(encoder, 0.5, t) for t in TEMPERATURES]
    assert F.cosine_similarity(conditions[0], conditions[-1]).item() < MAX_COSINE
    # Adjacent temperatures must be distinguishable too, not just the extremes:
    # the model has to interpolate across all five, not merely notice hot vs cold.
    for cold, warm in zip(conditions, conditions[1:]):
        assert F.cosine_similarity(cold, warm).item() < 0.99


def test_condition_encoder_separates_physical_time_gaps():
    torch.manual_seed(0)
    encoder = ConditionEncoder(512).eval()
    near = _condition(encoder, 0.5, 379.0, delta_t=5.0)
    far = _condition(encoder, 0.5, 379.0, delta_t=50.0)
    assert F.cosine_similarity(near, far).item() < 0.99


def test_condition_encoder_is_monotonic_outside_the_configured_range():
    """An unseen temperature must land outside the training span, not on its edge."""
    torch.manual_seed(0)
    encoder = ConditionEncoder(512, temperature_min=320.0, temperature_max=450.0).eval()
    hottest = _condition(encoder, 0.5, 450.0)
    beyond = _condition(encoder, 0.5, 600.0)
    assert F.cosine_similarity(hottest, beyond).item() < 0.999


def test_condition_encoder_rejects_degenerate_ranges():
    with pytest.raises(ValueError, match="temperature_max"):
        ConditionEncoder(64, temperature_min=400.0, temperature_max=400.0)
    with pytest.raises(ValueError, match="delta_t_max"):
        ConditionEncoder(64, delta_t_max=0.0)


def _fusion(film: bool, seed: int = 0) -> GatedFusion:
    torch.manual_seed(seed)
    return GatedFusion(
        seq_hidden_dim=16, geo_hidden_dim=16, fusion_hidden_dim=16,
        condition_dim=8, film_conditioning=film,
    ).eval()


def _fusion_inputs(batch_size=2, length=5, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return dict(
        h_seq=torch.randn(batch_size, length, 16, generator=generator),
        h_geo=torch.randn(batch_size, length, 16, generator=generator),
        tau=torch.rand(batch_size, generator=generator),
        temperature=torch.full((batch_size, 1), 379.0),
        physical_delta_t=torch.full((batch_size, 1), 5.0),
        residue_mask=torch.ones(batch_size, length, dtype=torch.bool),
    )


def test_film_is_exactly_identity_at_initialization():
    """Zero-init means adding FiLM changes nothing about how a run starts.

    The FiLM projection is built last, so with a shared seed every other
    parameter is initialised identically and the two modules must agree
    bit for bit.
    """
    inputs = _fusion_inputs()
    with torch.no_grad():
        with_film = _fusion(film=True)(**inputs)
        without_film = _fusion(film=False)(**inputs)
    torch.testing.assert_close(with_film, without_film)


def test_film_can_rescale_the_fused_representation():
    """The degree of freedom the gate cannot supply: an output scale."""
    fusion = _fusion(film=True)
    inputs = _fusion_inputs()
    with torch.no_grad():
        baseline = fusion(**inputs)
        # bias = [scale=1, shift=0] -> fused * (1 + 1) + 0
        fusion.film.bias[: fusion.film.out_features // 2] = 1.0
        doubled = fusion(**inputs)
    torch.testing.assert_close(doubled, 2.0 * baseline)


class _ConstantVelocity(nn.Module):
    """Stands in for a model that has learned the flow-matching target exactly."""

    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.velocity


def _physics_config(mode: str) -> Config:
    config = Config()
    config.data.plm_dim = 8
    config.model.graph.knn_k = 3
    config.loss.endpoint_enabled = False
    config.loss.physics.enabled = True
    config.loss.physics.apply_to = "euler_step"
    config.loss.physics.step_scale_mode = mode
    return config


def _chain(num_residues: int, generator: torch.Generator, spacing: float = 3.8) -> torch.Tensor:
    """A random walk with a fixed step length, standing in for a C-alpha trace.

    Constant spacing is what makes this fixture useful: the bond term
    measures consecutive distances against the source, so any two chains
    built this way agree exactly on them and the loss floor is 0. A cloud of
    independent random points would instead start with a large, noisy bond
    loss that swamps the effect under test.
    """
    steps = torch.randn(num_residues - 1, 3, generator=generator)
    steps = spacing * steps / steps.norm(dim=-1, keepdim=True)
    return torch.cat([torch.zeros(1, 3), steps.cumsum(dim=0)], dim=0)


def _pair_batch(num_residues=6, plm_dim=8, seed=3):
    generator = torch.Generator().manual_seed(seed)
    samples = [{
        "sequence_embedding": torch.randn(num_residues, plm_dim, generator=generator),
        "source_coords": _chain(num_residues, generator),
        "target_coords": _chain(num_residues, generator),
        "residue_types": torch.randint(0, 20, (num_residues,), generator=generator),
        "temperature": torch.tensor([320.0]),
        "physical_delta_t": torch.tensor([5.0]),
    } for _ in range(2)]
    return collate_protein_batch(samples)


def _true_velocity(batch):
    aligned = masked_kabsch_align(
        batch["source_coords"], batch["target_coords"], batch["residue_mask"]
    ).aligned_target
    return aligned - batch["source_coords"]


@pytest.mark.parametrize("tau_value", [0.05, 0.5, 0.95])
def test_remaining_step_scale_never_penalizes_the_correct_velocity(tau_value):
    """x_tau + (1 - tau) * (x1 - x0) is x1 itself, at every tau.

    So the physics terms charged against the ground-truth velocity must not
    depend on tau. Under "constant" they grow with it, which is what taught
    the model to shrink its velocity toward zero.
    """
    batch = _pair_batch()
    model = _ConstantVelocity(_true_velocity(batch))
    tau = torch.full((batch["source_coords"].shape[0],), tau_value)

    losses = compute_losses(model, batch, _physics_config("remaining"), tau=tau)
    reference = compute_losses(
        model, batch, _physics_config("remaining"),
        tau=torch.zeros_like(tau),  # at tau=0 both modes agree, so this is the floor
    )
    for term in ("bond", "angle"):
        torch.testing.assert_close(losses[term], reference[term], rtol=1e-4, atol=1e-6)


def test_constant_step_scale_penalizes_the_correct_velocity_more_at_high_tau():
    """Documents the old behaviour the default moved away from."""
    batch = _pair_batch()
    model = _ConstantVelocity(_true_velocity(batch))
    config = _physics_config("constant")
    batch_size = batch["source_coords"].shape[0]

    early = compute_losses(model, batch, config, tau=torch.full((batch_size,), 0.05))
    late = compute_losses(model, batch, config, tau=torch.full((batch_size,), 0.95))
    assert late["bond"].item() > 10.0 * early["bond"].item()


def test_unknown_step_scale_mode_is_rejected():
    batch = _pair_batch()
    model = _ConstantVelocity(_true_velocity(batch))
    with pytest.raises(ValueError, match="step_scale_mode"):
        compute_losses(model, batch, _physics_config("halfway"))
