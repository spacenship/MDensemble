"""Differentiable ODE unrolling for the endpoint loss.

Covers the three mechanisms that make many-step unrolling affordable:
per-step gradient checkpointing, a training-specific step count, and
truncated backpropagation through time.
"""
from __future__ import annotations

import pytest
import torch

from protein_flow.config import Config, EndpointRolloutConfig
from protein_flow.data.collate import collate_protein_batch
from protein_flow.models.dual_graph_flow import DualGraphFlowModel
from protein_flow.train import _differentiable_rollout, compute_losses


def _small_config() -> Config:
    config = Config()
    config.data.plm_dim = 16
    for module in (
        config.model.sequence_encoder, config.model.geometric_encoder,
        config.model.fusion, config.model.decoder,
    ):
        module.hidden_dim = 16
    config.model.fusion.condition_dim = 8
    config.model.sequence_encoder.num_layers = 2
    config.model.geometric_encoder.num_layers = 2
    config.model.sequence_encoder.dropout = 0.0
    config.model.geometric_encoder.dropout = 0.0
    config.model.graph.knn_k = 4
    config.model.graph.num_rbf = 4
    return config


def _batch(num_residues=7, batch_size=2, plm_dim=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    samples = [{
        "sequence_embedding": torch.randn(num_residues, plm_dim, generator=generator),
        "source_coords": torch.randn(num_residues, 3, generator=generator) * 3.0,
        "target_coords": torch.randn(num_residues, 3, generator=generator) * 3.0,
        "residue_types": torch.randint(0, 20, (num_residues,), generator=generator),
        "temperature": torch.tensor([320.0]),
        "physical_delta_t": torch.tensor([5.0]),
    } for _ in range(batch_size)]
    return collate_protein_batch(samples)


def _rollout(model, batch, rollout_config):
    return _differentiable_rollout(
        model, batch["source_coords"], batch["sequence_embedding"], batch["residue_types"],
        batch["residue_mask"], batch["temperature"], batch["physical_delta_t"], rollout_config,
    )


def test_rollout_moves_coordinates_away_from_the_source():
    torch.manual_seed(0)
    model = DualGraphFlowModel(_small_config()).eval()
    batch = _batch()
    out = _rollout(model, batch, EndpointRolloutConfig(num_steps=4))
    displacement = (out - batch["source_coords"]).norm(dim=-1)[batch["residue_mask"]]
    assert displacement.mean() > 0.0, "rollout must actually integrate, not return x0"


def test_step_count_changes_the_integrated_trajectory():
    """A genuine ODE integration depends on its discretization; if it did
    not, the rollout would be a no-op in disguise."""
    torch.manual_seed(0)
    base = DualGraphFlowModel(_small_config()).eval()

    class Amplified(torch.nn.Module):
        """Untrained velocities are tiny relative to the structure, so scale
        them up to make the discretization difference measurable."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args, **kwargs):
            return self.inner(*args, **kwargs) * 500.0

    model = Amplified(base)
    batch = _batch()
    coarse = _rollout(model, batch, EndpointRolloutConfig(num_steps=2))
    fine = _rollout(model, batch, EndpointRolloutConfig(num_steps=16))
    assert not torch.allclose(coarse, fine, atol=1e-3)


def test_gradient_checkpointing_matches_plain_rollout_forward_and_backward():
    """Checkpointing is a memory optimization: it must not change the value
    of the rollout or the gradients it produces."""
    config = _small_config()
    batch = _batch()

    def run(use_checkpointing: bool):
        torch.manual_seed(0)
        model = DualGraphFlowModel(config)
        out = _rollout(
            model, batch,
            EndpointRolloutConfig(num_steps=4, gradient_checkpointing=use_checkpointing),
        )
        out.pow(2).sum().backward()
        grads = torch.cat([
            p.grad.flatten() for _, p in sorted(model.named_parameters()) if p.grad is not None
        ])
        return out.detach(), grads

    plain_out, plain_grads = run(False)
    checkpointed_out, checkpointed_grads = run(True)

    torch.testing.assert_close(checkpointed_out, plain_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(checkpointed_grads, plain_grads, atol=1e-4, rtol=1e-4)


def test_truncated_backprop_keeps_the_same_trajectory():
    """Truncating BPTT changes which steps receive gradient, not the
    trajectory itself, so the forward value must be unchanged."""
    config = _small_config()
    batch = _batch()

    torch.manual_seed(0)
    model_full = DualGraphFlowModel(config).eval()
    full = _rollout(model_full, batch, EndpointRolloutConfig(num_steps=6))

    torch.manual_seed(0)
    model_truncated = DualGraphFlowModel(config).eval()
    truncated = _rollout(
        model_truncated, batch, EndpointRolloutConfig(num_steps=6, backprop_last_steps=2)
    )
    torch.testing.assert_close(truncated, full, atol=1e-5, rtol=1e-5)


def test_truncated_backprop_still_produces_gradient():
    config = _small_config()
    batch = _batch()
    torch.manual_seed(0)
    model = DualGraphFlowModel(config)
    out = _rollout(model, batch, EndpointRolloutConfig(num_steps=6, backprop_last_steps=2))
    out.pow(2).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_heun_rollout_runs_and_differs_from_euler():
    torch.manual_seed(0)
    model = DualGraphFlowModel(_small_config()).eval()
    batch = _batch()
    euler = _rollout(model, batch, EndpointRolloutConfig(num_steps=3, solver="euler"))
    heun = _rollout(model, batch, EndpointRolloutConfig(num_steps=3, solver="heun"))
    assert torch.isfinite(heun).all()
    assert not torch.allclose(euler, heun, atol=1e-8)


def test_invalid_rollout_solver_and_step_count_raise():
    torch.manual_seed(0)
    model = DualGraphFlowModel(_small_config()).eval()
    batch = _batch()
    with pytest.raises(ValueError, match="solver"):
        _rollout(model, batch, EndpointRolloutConfig(solver="rk4"))
    with pytest.raises(ValueError, match="num_steps"):
        _rollout(model, batch, EndpointRolloutConfig(num_steps=0))


def test_endpoint_loss_appears_in_compute_losses_and_is_differentiable():
    config = _small_config()
    config.loss.endpoint_enabled = True
    config.loss.lambda_endpoint = 1.0
    config.loss.endpoint_rollout.num_steps = 3
    torch.manual_seed(0)
    model = DualGraphFlowModel(config)
    batch = _batch()

    losses = compute_losses(model, batch, config)
    assert "endpoint" in losses
    assert torch.isfinite(losses["endpoint"])
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_endpoint_physics_terms_are_reported_when_enabled():
    config = _small_config()
    config.loss.endpoint_enabled = True
    config.loss.lambda_endpoint = 1.0
    config.loss.endpoint_physics_enabled = True
    config.loss.endpoint_rollout.num_steps = 2
    torch.manual_seed(0)
    model = DualGraphFlowModel(config)
    batch = _batch()

    losses = compute_losses(model, batch, config)
    for name in ("endpoint_bond", "endpoint_angle", "endpoint_clash"):
        assert name in losses
        assert torch.isfinite(losses[name])


def test_endpoint_disabled_by_default():
    config = _small_config()
    assert config.loss.endpoint_enabled is False
    torch.manual_seed(0)
    model = DualGraphFlowModel(config)
    losses = compute_losses(model, _batch(), config)
    assert "endpoint" not in losses
