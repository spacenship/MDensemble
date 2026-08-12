"""Tests for the displacement flow: noise -> (x1 - x0), with x0 as conditioning.

Two of these are load-bearing rather than routine.

``test_decoder_can_represent_negative_flow_state`` guards the reason the
coordinate-space formulation could not work at all: at tau=0 the correct
velocity is ``delta - eps == -eps``, which is not in the span of the
geometric graph's relative vectors. If the decoder's flow-state basis terms
are ever removed, the model silently loses the ability to express its own
target and quietly collapses again -- which is exactly the failure this
reformulation exists to fix, and it took a full 67,250-step run to notice
last time.

``test_rotation_equivariance_in_flow_state`` covers the risk introduced by
adding those terms: the flow state is a raw 3-vector entering the network,
and the only thing keeping the model SE(3)-equivariant is that it enters
solely through invariant projections and equivariant basis vectors.
"""
from __future__ import annotations

import pytest
import torch

from protein_flow.config import Config, validate_config
from protein_flow.flow.paths import (
    DisplacementPath,
    build_flow_path,
    clean_displacement,
    sample_zero_com_noise,
)
from protein_flow.models.dual_graph_flow import DualGraphFlowModel

SEED = 4242
ATOL = 1e-4
RTOL = 1e-4


def _random_rotation(generator: torch.Generator) -> torch.Tensor:
    a = torch.randn(3, 3, generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(-2)
    if torch.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


def _tie_free_coords(batch: int, length: int, generator: torch.Generator) -> torch.Tensor:
    """Irregular per-axis scaling so no two pairwise distances coincide and the
    k-NN graph is identical before and after a rotation."""
    coords = torch.randn(batch, length, 3, generator=generator)
    return coords * torch.tensor([1.0, 2.7182818, 4.6692016])


def _build_small_model() -> tuple[DualGraphFlowModel, Config]:
    config = Config()
    config.flow.path_type = "displacement"
    config.flow.noise_scale = 2.667
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
    model.eval()
    return model, config


def _model_inputs(batch: int, length: int, config: Config, generator: torch.Generator) -> dict:
    return {
        "sequence_embedding": torch.randn(batch, length, config.data.plm_dim, generator=generator),
        "residue_types": torch.randint(
            0, config.data.num_amino_acid_types, (batch, length), generator=generator
        ),
        "temperature": torch.rand(batch, 1, generator=generator),
        "physical_delta_t": torch.rand(batch, 1, generator=generator),
        "residue_mask": torch.ones(batch, length, dtype=torch.bool),
    }


# --- the path itself ---------------------------------------------------------


def test_endpoints_and_target():
    generator = torch.Generator().manual_seed(SEED)
    batch, length = 3, 9
    x0 = torch.randn(batch, length, 3, generator=generator)
    x1 = torch.randn(batch, length, 3, generator=generator)
    mask = torch.ones(batch, length, dtype=torch.bool)
    delta = x1 - x0
    path = DisplacementPath(noise_scale=1.5)

    zero = torch.zeros(batch)
    state_at_zero, target_at_zero = path.sample(
        x0, x1, zero, mask, torch.Generator().manual_seed(7)
    )
    # At tau=0 the state is pure noise, so state == -(target - delta).
    torch.testing.assert_close(state_at_zero, delta - target_at_zero, atol=ATOL, rtol=RTOL)

    one = torch.ones(batch)
    state_at_one, _ = path.sample(x0, x1, one, mask, torch.Generator().manual_seed(7))
    torch.testing.assert_close(state_at_one, delta, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("tau_value", [0.0, 0.3, 0.7, 1.0])
def test_clean_displacement_recovers_delta(tau_value):
    """x_t + (1 - t) * v == delta exactly, for every t, when v is the target.

    This is what the physics terms are evaluated on, so if it drifts the
    bond/angle penalties start charging the *correct* answer -- the same
    defect step_scale_mode was introduced to fix in the coordinate-space path.
    """
    generator = torch.Generator().manual_seed(SEED)
    batch, length = 4, 11
    x0 = torch.randn(batch, length, 3, generator=generator)
    x1 = torch.randn(batch, length, 3, generator=generator)
    mask = torch.ones(batch, length, dtype=torch.bool)
    delta = x1 - x0

    tau = torch.full((batch,), tau_value)
    state, target = DisplacementPath(noise_scale=2.0).sample(x0, x1, tau, mask)
    torch.testing.assert_close(clean_displacement(state, target, tau), delta, atol=ATOL, rtol=RTOL)


def test_noise_is_zero_com_and_zero_on_padding():
    generator = torch.Generator().manual_seed(SEED)
    batch, length = 3, 12
    mask = torch.ones(batch, length, dtype=torch.bool)
    mask[0, 8:] = False
    mask[1, 5:] = False

    noise = sample_zero_com_noise(
        (batch, length, 3), mask, 2.667, torch.device("cpu"), torch.float32, generator
    )

    assert torch.all(noise[~mask] == 0.0)
    centre = (noise * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    torch.testing.assert_close(centre, torch.zeros(batch, 3), atol=1e-5, rtol=0.0)


def test_noise_scale_matches_requested_std():
    generator = torch.Generator().manual_seed(SEED)
    mask = torch.ones(2, 2000, dtype=torch.bool)
    noise = sample_zero_com_noise(
        (2, 2000, 3), mask, 2.667, torch.device("cpu"), torch.float32, generator
    )
    assert abs(noise.std().item() - 2.667) < 0.05


@pytest.mark.parametrize("rounds", [0, 4, 8])
def test_smoothing_preserves_scale_com_and_padding(rounds):
    """Smoothing must change only the *correlation* of the noise.

    It shrinks raw variance, so without the rescale the base distribution
    would silently narrow as rounds increase and noise_scale would stop
    meaning what the config says it means.
    """
    generator = torch.Generator().manual_seed(SEED)
    batch, length = 3, 64
    mask = torch.ones(batch, length, dtype=torch.bool)
    mask[0, 50:] = False
    coords = _tie_free_coords(batch, length, generator) * 4.0

    noise = sample_zero_com_noise(
        (batch, length, 3), mask, 2.667, torch.device("cpu"), torch.float32, generator,
        coords=coords, smoothing_rounds=rounds, knn_k=8,
    )

    assert torch.all(noise[~mask] == 0.0)
    centre = (noise * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    torch.testing.assert_close(centre, torch.zeros(batch, 3), atol=1e-4, rtol=0.0)
    per_axis = (noise.pow(2).sum() / (mask.sum() * 3)).sqrt()
    assert abs(float(per_axis) - 2.667) < 1e-3


def test_smoothing_creates_spatial_correlation():
    """The whole point: neighbours must end up moving together.

    Measured on real data, MD displacement direction correlation is 0.96
    within 4 A and 0.60 at 6-8 A, while i.i.d. noise is 0 everywhere. This
    checks the mechanism produces correlation at all and that it grows with
    the number of rounds -- the knob has to actually be a knob.
    """
    generator = torch.Generator().manual_seed(SEED)
    batch, length = 2, 96
    mask = torch.ones(batch, length, dtype=torch.bool)
    # A chain-like structure so spatial neighbours are meaningful.
    steps = torch.randn(length - 1, 3, generator=generator)
    chain = torch.cat([torch.zeros(1, 3), (3.8 * steps / steps.norm(dim=-1, keepdim=True)).cumsum(0)])
    coords = chain.unsqueeze(0).expand(batch, length, 3).contiguous()

    def neighbour_correlation(rounds: int) -> float:
        noise = sample_zero_com_noise(
            (batch, length, 3), mask, 2.667, torch.device("cpu"), torch.float32,
            torch.Generator().manual_seed(11),
            coords=coords, smoothing_rounds=rounds, knn_k=8,
        )
        left, right = noise[:, :-1], noise[:, 1:]
        cosine = (left * right).sum(-1) / (
            left.norm(dim=-1).clamp(min=1e-8) * right.norm(dim=-1).clamp(min=1e-8)
        )
        return float(cosine.mean())

    white = neighbour_correlation(0)
    smoothed_4 = neighbour_correlation(4)
    smoothed_8 = neighbour_correlation(8)

    assert abs(white) < 0.2, f"i.i.d. noise should be uncorrelated, got {white:.3f}"
    assert smoothed_4 > 0.5, f"4 rounds produced no collectivity ({smoothed_4:.3f})"
    assert smoothed_8 > smoothed_4, "more rounds must mean more correlation"


def test_smoothing_requires_coords():
    mask = torch.ones(1, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match="needs coords"):
        sample_zero_com_noise(
            (1, 8, 3), mask, 2.667, torch.device("cpu"), torch.float32, None, smoothing_rounds=4
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cpu_generator_seeds_a_cuda_draw():
    """A CPU generator must seed a CUDA noise draw.

    Callers naturally hold a CPU generator, and torch.randn otherwise refuses
    the mismatch -- a failure that only appears on GPU, so the CPU test suite
    is blind to it.
    """
    device = torch.device("cuda:0")
    mask = torch.ones(2, 16, dtype=torch.bool, device=device)
    first = sample_zero_com_noise(
        (2, 16, 3), mask, 2.667, device, torch.float32, torch.Generator().manual_seed(5)
    )
    second = sample_zero_com_noise(
        (2, 16, 3), mask, 2.667, device, torch.float32, torch.Generator().manual_seed(5)
    )
    assert first.device.type == "cuda"
    torch.testing.assert_close(first, second)
    # Same seed, same numbers on either device -- draws are device-independent.
    on_cpu = sample_zero_com_noise(
        (2, 16, 3), mask.cpu(), 2.667, torch.device("cpu"), torch.float32,
        torch.Generator().manual_seed(5),
    )
    torch.testing.assert_close(first.cpu(), on_cpu, atol=1e-5, rtol=1e-5)


def test_build_flow_path_dispatch():
    assert isinstance(build_flow_path("displacement", noise_scale=3.0), DisplacementPath)
    assert build_flow_path("displacement", noise_scale=3.0).flows_in_displacement_space
    assert not build_flow_path("linear").flows_in_displacement_space
    with pytest.raises(ValueError):
        DisplacementPath(noise_scale=0.0)
    with pytest.raises(ValueError, match="particle_mask"):
        DisplacementPath().sample(torch.zeros(1, 4, 3), torch.zeros(1, 4, 3), torch.zeros(1))


# --- the model ---------------------------------------------------------------


def test_rotation_equivariance_in_flow_state():
    """v(R x0, R s) == R v(x0, s). The flow state rotates with the structure."""
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_small_model()
    batch, length = 3, 14

    x0 = _tie_free_coords(batch, length, generator)
    flow_state = torch.randn(batch, length, 3, generator=generator)
    tau = torch.rand(batch, generator=generator)
    inputs = _model_inputs(batch, length, config, generator)
    rotation = _random_rotation(generator)

    with torch.no_grad():
        v = model(x0, tau, inputs["sequence_embedding"], inputs["residue_types"],
                  inputs["temperature"], inputs["physical_delta_t"], inputs["residue_mask"],
                  flow_state=flow_state)
        v_rotated = model(x0 @ rotation, tau, inputs["sequence_embedding"], inputs["residue_types"],
                          inputs["temperature"], inputs["physical_delta_t"], inputs["residue_mask"],
                          flow_state=flow_state @ rotation)

    torch.testing.assert_close(v_rotated, v @ rotation, atol=ATOL, rtol=RTOL)


def test_translation_invariance_in_flow_state():
    """Translating x0 must not move the velocity: the flow state is already a
    displacement and does not translate with the structure."""
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_small_model()
    batch, length = 3, 14

    x0 = _tie_free_coords(batch, length, generator)
    flow_state = torch.randn(batch, length, 3, generator=generator)
    tau = torch.rand(batch, generator=generator)
    inputs = _model_inputs(batch, length, config, generator)
    translation = torch.randn(3, generator=generator) * 10.0

    with torch.no_grad():
        v = model(x0, tau, inputs["sequence_embedding"], inputs["residue_types"],
                  inputs["temperature"], inputs["physical_delta_t"], inputs["residue_mask"],
                  flow_state=flow_state)
        v_translated = model(x0 + translation, tau, inputs["sequence_embedding"],
                             inputs["residue_types"], inputs["temperature"],
                             inputs["physical_delta_t"], inputs["residue_mask"],
                             flow_state=flow_state)

    torch.testing.assert_close(v_translated, v, atol=ATOL, rtol=RTOL)


def test_velocity_depends_on_flow_state():
    """A different flow state must give a different velocity.

    If it does not, the state is being ignored and the sampler is
    deterministic again -- the whole reformulation would be a no-op.
    """
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_small_model()
    batch, length = 2, 12

    x0 = _tie_free_coords(batch, length, generator)
    tau = torch.full((batch,), 0.5)
    inputs = _model_inputs(batch, length, config, generator)

    # The decoder's state heads are zero-initialised, so at init the state can
    # only reach the output through the encoder's invariant features. It must
    # still get through.
    with torch.no_grad():
        v_a = model(x0, tau, inputs["sequence_embedding"], inputs["residue_types"],
                    inputs["temperature"], inputs["physical_delta_t"], inputs["residue_mask"],
                    flow_state=torch.randn(batch, length, 3, generator=generator))
        v_b = model(x0, tau, inputs["sequence_embedding"], inputs["residue_types"],
                    inputs["temperature"], inputs["physical_delta_t"], inputs["residue_mask"],
                    flow_state=torch.randn(batch, length, 3, generator=generator))

    assert (v_a - v_b).abs().max() > 1e-6


def test_decoder_can_represent_negative_flow_state():
    """The decoder must be able to learn v = -flow_state.

    This is the target at tau=0 and it lies outside the span of the geometric
    graph's relative vectors, so it is reachable only through the decoder's
    flow-state basis terms. Checking that those terms receive gradient is the
    regression test for the structural blind spot that made the previous
    formulation unlearnable.
    """
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_small_model()
    batch, length = 2, 12

    x0 = _tie_free_coords(batch, length, generator)
    flow_state = torch.randn(batch, length, 3, generator=generator)
    flow_state = flow_state - flow_state.mean(dim=1, keepdim=True)  # zero-COM, as the sampler draws it
    tau = torch.zeros(batch)
    inputs = _model_inputs(batch, length, config, generator)

    velocity = model(x0, tau, inputs["sequence_embedding"], inputs["residue_types"],
                     inputs["temperature"], inputs["physical_delta_t"], inputs["residue_mask"],
                     flow_state=flow_state)
    loss = (velocity - (-flow_state)).pow(2).mean()
    loss.backward()

    state_head = model.decoder.state_coeff_mlp[-1]
    assert state_head.weight.grad is not None
    assert state_head.weight.grad.abs().max() > 0.0, (
        "the flow-state basis term received no gradient; without it the decoder cannot "
        "represent v = -flow_state and the model will collapse to a near-identity map"
    )


def test_flow_state_basis_is_what_makes_the_tau_zero_target_learnable():
    """Fit v = -flow_state with and without the flow-state basis terms.

    This is the reformulation's central claim, checked directly rather than
    argued: the velocity at tau=0 is orthogonal to what invariant coefficients
    on graph relative vectors can build, so a decoder without the extra basis
    terms cannot fit it however long it trains. The gap between the two
    numbers below is the gap that made the previous run's rollout move 4.7% of
    the required distance.
    """
    def fit(use_flow_state: bool) -> float:
        config = Config()
        config.flow.path_type = "displacement" if use_flow_state else "linear"
        config.data.plm_dim = 20
        config.data.num_amino_acid_types = 22
        for section in (config.model.sequence_encoder, config.model.geometric_encoder):
            section.hidden_dim, section.dropout = 24, 0.0
        config.model.fusion.hidden_dim = 24
        config.model.fusion.condition_dim = 12
        config.model.graph.knn_k = 5
        config.model.graph.num_rbf = 10

        torch.manual_seed(SEED)
        model = DualGraphFlowModel(config)
        generator = torch.Generator().manual_seed(SEED)
        batch, length = 2, 12
        x0 = _tie_free_coords(batch, length, generator)
        flow_state = torch.randn(batch, length, 3, generator=generator)
        flow_state = flow_state - flow_state.mean(dim=1, keepdim=True)
        tau = torch.zeros(batch)
        inputs = _model_inputs(batch, length, config, generator)
        target = -flow_state

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        loss = torch.tensor(float("inf"))
        for _ in range(200):
            optimizer.zero_grad()
            velocity = model(
                x0, tau, inputs["sequence_embedding"], inputs["residue_types"],
                inputs["temperature"], inputs["physical_delta_t"], inputs["residue_mask"],
                # A linear-path model has no flow-state input at all: the
                # information simply is not available to it, which is the point.
                **({"flow_state": flow_state} if use_flow_state else {}),
            )
            loss = (velocity - target).pow(2).mean()
            loss.backward()
            optimizer.step()
        return float(loss.detach())

    with_basis = fit(use_flow_state=True)
    without_basis = fit(use_flow_state=False)
    target_scale = 1.0  # unit-variance flow state, so ~1.0 is "predicting zero"

    assert with_basis < 0.1 * target_scale, (
        f"the displacement decoder failed to fit v = -flow_state (loss {with_basis:.4f}); "
        "its flow-state basis terms are not doing their job"
    )
    assert without_basis > 5 * with_basis, (
        f"a decoder without the flow-state basis reached {without_basis:.4f} vs "
        f"{with_basis:.4f} with it -- the structural gap this reformulation rests on is absent"
    )


def test_state_heads_are_zero_init():
    """A freshly built displacement decoder outputs exactly what one without
    the flow-state basis terms would, so the change is a pure extension."""
    model, _ = _build_small_model()
    for head in (model.decoder.state_coeff_mlp, model.decoder.state_neighbour_mlp):
        assert torch.all(head[-1].weight == 0.0)
        assert torch.all(head[-1].bias == 0.0)


def test_sampling_is_stochastic_and_reproducible():
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_small_model()
    batch, length = 2, 12

    x0 = _tie_free_coords(batch, length, generator)
    inputs = _model_inputs(batch, length, config, generator)
    kwargs = dict(
        sequence_embedding=inputs["sequence_embedding"],
        residue_types=inputs["residue_types"],
        residue_mask=inputs["residue_mask"],
        temperature=inputs["temperature"],
        physical_delta_t=inputs["physical_delta_t"],
        num_steps=3,
        solver="euler",
    )

    first, _ = model.sample(x0, generator=torch.Generator().manual_seed(1), **kwargs)
    second, _ = model.sample(x0, generator=torch.Generator().manual_seed(2), **kwargs)
    repeat, _ = model.sample(x0, generator=torch.Generator().manual_seed(1), **kwargs)

    assert (first - second).abs().max() > 1e-6, "different noise draws gave the same structure"
    torch.testing.assert_close(first, repeat, atol=ATOL, rtol=RTOL)


def test_generated_structure_keeps_the_source_centroid():
    """The flow lives entirely in the zero-COM subspace, so sampling must not
    translate the protein."""
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_small_model()
    batch, length = 2, 12

    x0 = _tie_free_coords(batch, length, generator)
    inputs = _model_inputs(batch, length, config, generator)
    generated, _ = model.sample(
        x0, inputs["sequence_embedding"], inputs["residue_types"], inputs["residue_mask"],
        inputs["temperature"], inputs["physical_delta_t"], num_steps=4, solver="heun",
        generator=torch.Generator().manual_seed(11),
    )
    torch.testing.assert_close(
        generated.mean(dim=1), x0.mean(dim=1), atol=1e-4, rtol=0.0
    )


def test_trajectory_entries_are_coordinates():
    generator = torch.Generator().manual_seed(SEED)
    model, config = _build_small_model()
    batch, length = 2, 10

    x0 = _tie_free_coords(batch, length, generator)
    inputs = _model_inputs(batch, length, config, generator)
    generated, trajectory = model.sample(
        x0, inputs["sequence_embedding"], inputs["residue_types"], inputs["residue_mask"],
        inputs["temperature"], inputs["physical_delta_t"], num_steps=5, solver="euler",
        return_trajectory=True, generator=torch.Generator().manual_seed(3),
    )
    assert trajectory.shape == (6, batch, length, 3)
    torch.testing.assert_close(trajectory[-1], generated, atol=ATOL, rtol=RTOL)
    # The first entry is x0 + eps, not x0 -- the flow starts at noise.
    assert (trajectory[0] - x0).abs().max() > 1e-3


# --- the training path -------------------------------------------------------


def _tiny_training_config() -> Config:
    config = Config()
    config.flow.path_type = "displacement"
    config.flow.noise_scale = 2.667
    config.loss.lambda_direction = 0.1
    config.loss.endpoint_enabled = False
    config.data.plm_dim = 16
    config.data.num_amino_acid_types = 22
    config.data.min_length = 6
    config.data.max_length = 10
    config.data.train_size = 8
    config.data.val_size = 4
    config.data.batch_size = 4
    config.model.sequence_encoder.hidden_dim = 8
    config.model.geometric_encoder.hidden_dim = 8
    config.model.fusion.hidden_dim = 8
    config.model.fusion.condition_dim = 4
    config.model.graph.knn_k = 3
    config.model.graph.num_rbf = 4
    return config


def test_compute_losses_runs_in_displacement_mode():
    """The whole objective, end to end: path, model call, physics on the
    reconstructed structure, direction term, and the collapse diagnostics."""
    from protein_flow.models.dual_graph_flow import DualGraphFlowModel
    from protein_flow.train import build_dataloaders, compute_losses

    torch.manual_seed(0)
    config = _tiny_training_config()
    train_loader, _ = build_dataloaders(config)
    model = DualGraphFlowModel(config)
    batch = next(iter(train_loader))

    diagnostics: dict = {}
    losses = compute_losses(model, batch, config, diagnostics=diagnostics)

    for name, value in losses.items():
        assert torch.isfinite(value), f"{name} loss is not finite: {value}"
    assert "direction" in losses, "lambda_direction > 0 but no direction term was added"
    assert 0.0 <= float(losses["direction"].detach()) <= 2.0
    for key in ("velocity_ratio", "velocity_cosine"):
        assert key in diagnostics and torch.isfinite(diagnostics[key])
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_physics_sees_a_real_structure_not_an_interpolant():
    """With the target velocity substituted in, the coordinates the physics
    terms score must be exactly x1 -- a genuine MD frame."""
    from protein_flow.flow.paths import DisplacementPath

    generator = torch.Generator().manual_seed(SEED)
    batch, length = 3, 10
    x0 = torch.randn(batch, length, 3, generator=generator)
    x1 = torch.randn(batch, length, 3, generator=generator)
    mask = torch.ones(batch, length, dtype=torch.bool)
    tau = torch.rand(batch, generator=generator)

    state, target = DisplacementPath(noise_scale=2.667).sample(x0, x1, tau, mask)
    reconstructed = x0 + clean_displacement(state, target, tau)
    torch.testing.assert_close(reconstructed, x1, atol=1e-4, rtol=1e-4)


# --- config guards -----------------------------------------------------------


def test_displacement_rejects_endpoint_rollout():
    config = Config()
    config.flow.path_type = "displacement"
    config.loss.endpoint_enabled = True
    with pytest.raises(ValueError, match="endpoint rollout"):
        validate_config(config)

    config.loss.endpoint_enabled = False
    config.train.val_endpoint_enabled = True
    with pytest.raises(ValueError, match="endpoint rollout"):
        validate_config(config)


def test_mismatched_flow_state_is_rejected():
    """Passing a flow state to a linear-path model (or omitting it on a
    displacement model) must fail loudly: the layer widths differ, so the
    silent version is a shape error deep inside the encoder."""
    model, config = _build_small_model()
    generator = torch.Generator().manual_seed(SEED)
    batch, length = 2, 10
    x0 = _tie_free_coords(batch, length, generator)
    tau = torch.rand(batch, generator=generator)
    inputs = _model_inputs(batch, length, config, generator)

    with pytest.raises(ValueError, match="use_flow_state=True"):
        model(x0, tau, inputs["sequence_embedding"], inputs["residue_types"],
              inputs["temperature"], inputs["physical_delta_t"], inputs["residue_mask"])

    linear_config = Config()
    linear_config.data.plm_dim = 20
    linear_config.data.num_amino_acid_types = 22
    linear_model = DualGraphFlowModel(linear_config)
    linear_model.eval()
    assert not linear_model.uses_flow_state
    with pytest.raises(ValueError, match="use_flow_state=False"):
        linear_model(
            x0, tau,
            torch.randn(batch, length, linear_config.data.plm_dim, generator=generator),
            inputs["residue_types"], inputs["temperature"], inputs["physical_delta_t"],
            inputs["residue_mask"], flow_state=torch.randn(batch, length, 3, generator=generator),
        )
