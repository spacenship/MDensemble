import torch
import torch.nn as nn

from protein_flow.config import Config
from protein_flow.flow.solver import integrate_ode
from protein_flow.models.dual_graph_flow import DualGraphFlowModel


class ConstantVelocityModel(nn.Module):
    """Returns a fixed velocity regardless of input; ODE integral should be exact."""

    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.velocity = velocity

    def forward(self, x_tau, tau, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask):
        return self.velocity.expand_as(x_tau)


class LinearInTauVelocityModel(nn.Module):
    """v(x, tau) = tau * c -- integral over [0,1] is c/2, exactly matched by Heun, not Euler."""

    def __init__(self, c: torch.Tensor):
        super().__init__()
        self.c = c

    def forward(self, x_tau, tau, sequence_embedding, residue_types, temperature, physical_delta_t, residue_mask):
        return tau.view(-1, 1, 1) * self.c.expand_as(x_tau)


def _dummy_conditions(batch=2, length=5, plm_dim=4):
    seq_emb = torch.zeros(batch, length, plm_dim)
    residue_types = torch.zeros(batch, length, dtype=torch.long)
    mask = torch.ones(batch, length, dtype=torch.bool)
    temperature = torch.zeros(batch, 1)
    delta_t = torch.zeros(batch, 1)
    return seq_emb, residue_types, mask, temperature, delta_t


def test_euler_matches_analytic_constant_velocity():
    batch, length = 2, 5
    velocity = torch.tensor([1.0, -2.0, 0.5])
    model = ConstantVelocityModel(velocity)
    x0 = torch.randn(batch, length, 3)
    seq_emb, residue_types, mask, temperature, delta_t = _dummy_conditions(batch, length)

    final, traj = integrate_ode(
        model, x0, seq_emb, residue_types, mask, temperature, delta_t, num_steps=10, solver="euler", return_trajectory=True
    )
    expected = x0 + velocity  # integral of constant velocity over tau in [0,1]
    torch.testing.assert_close(final, expected, atol=1e-4, rtol=1e-4)
    assert traj.shape == (11, batch, length, 3)


def test_heun_more_accurate_than_euler_for_time_varying_field():
    batch, length = 2, 4
    c = torch.tensor([2.0, 0.0, 0.0])
    model = LinearInTauVelocityModel(c)
    x0 = torch.zeros(batch, length, 3)
    seq_emb, residue_types, mask, temperature, delta_t = _dummy_conditions(batch, length)
    expected = x0 + 0.5 * c  # integral_0^1 tau*c dtau = c/2

    final_euler, _ = integrate_ode(
        model, x0, seq_emb, residue_types, mask, temperature, delta_t, num_steps=5, solver="euler"
    )
    final_heun, _ = integrate_ode(
        model, x0, seq_emb, residue_types, mask, temperature, delta_t, num_steps=5, solver="heun"
    )

    err_euler = (final_euler - expected).abs().max()
    err_heun = (final_heun - expected).abs().max()
    assert err_heun < err_euler
    torch.testing.assert_close(final_heun, expected, atol=1e-4, rtol=1e-4)


def test_sampling_runs_under_no_grad_with_real_model():
    torch.manual_seed(0)
    config = Config()
    config.data.plm_dim = 16
    config.model.sequence_encoder.hidden_dim = 8
    config.model.geometric_encoder.hidden_dim = 8
    config.model.fusion.hidden_dim = 8
    config.model.fusion.condition_dim = 4
    config.model.graph.knn_k = 3
    config.model.graph.num_rbf = 4
    model = DualGraphFlowModel(config)

    batch, length = 2, 6
    seq_emb, residue_types, mask, temperature, delta_t = _dummy_conditions(batch, length, plm_dim=16)
    seq_emb = torch.randn(batch, length, 16)
    x0 = torch.randn(batch, length, 3, requires_grad=True)

    final, traj = model.sample(
        x0, seq_emb, residue_types, mask, temperature, delta_t, num_steps=4, solver="heun", return_trajectory=True
    )
    assert not final.requires_grad
    assert traj.shape == (5, batch, length, 3)
    assert torch.all(torch.isfinite(final))


def test_invalid_solver_raises():
    import pytest

    model = ConstantVelocityModel(torch.zeros(3))
    x0 = torch.zeros(1, 3, 3)
    seq_emb, residue_types, mask, temperature, delta_t = _dummy_conditions(1, 3)
    with pytest.raises(ValueError):
        integrate_ode(model, x0, seq_emb, residue_types, mask, temperature, delta_t, solver="rk4")
