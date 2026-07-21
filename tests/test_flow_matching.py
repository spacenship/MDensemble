import torch

from protein_flow.flow.paths import LinearPath, build_flow_path, sample_tau
from protein_flow.losses.flow_matching import flow_matching_loss


def test_linear_path_endpoints():
    torch.manual_seed(0)
    x0 = torch.randn(4, 10, 3)
    x1 = torch.randn(4, 10, 3)
    path = LinearPath()

    tau0 = torch.zeros(4)
    x_tau0, vel0 = path.sample(x0, x1, tau0)
    torch.testing.assert_close(x_tau0, x0)
    torch.testing.assert_close(vel0, x1 - x0)

    tau1 = torch.ones(4)
    x_tau1, vel1 = path.sample(x0, x1, tau1)
    torch.testing.assert_close(x_tau1, x1)


def test_linear_path_midpoint():
    x0 = torch.zeros(1, 3, 3)
    x1 = torch.ones(1, 3, 3) * 2.0
    path = LinearPath()
    tau = torch.tensor([0.5])
    x_tau, vel = path.sample(x0, x1, tau)
    torch.testing.assert_close(x_tau, torch.ones(1, 3, 3))
    torch.testing.assert_close(vel, x1 - x0)


def test_sample_tau_range_and_shape():
    tau = sample_tau(1000, device=torch.device("cpu"))
    assert tau.shape == (1000,)
    assert torch.all(tau >= 0.0) and torch.all(tau <= 1.0)


def test_build_flow_path_factory():
    linear = build_flow_path("linear")
    assert linear.__class__.__name__ == "LinearPath"
    bridge = build_flow_path("gaussian_bridge", sigma_min=0.1)
    assert bridge.__class__.__name__ == "GaussianBridgePath"


def test_flow_matching_loss_zero_when_perfect():
    target = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    loss = flow_matching_loss(target, target, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_flow_matching_loss_ignores_padding():
    predicted = torch.zeros(1, 4, 3)
    target = torch.zeros(1, 4, 3)
    target[0, -1] = 1000.0  # huge error on a padding residue
    mask = torch.tensor([[True, True, True, False]])
    loss = flow_matching_loss(predicted, target, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_flow_matching_loss_matches_manual_masked_mse():
    predicted = torch.randn(2, 6, 3)
    target = torch.randn(2, 6, 3)
    mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, True, False],
        ]
    )
    loss = flow_matching_loss(predicted, target, mask)

    manual_num = 0.0
    manual_den = 0
    for b in range(2):
        for l in range(6):
            if mask[b, l]:
                manual_num += ((predicted[b, l] - target[b, l]) ** 2).sum().item()
                manual_den += 1
    manual = manual_num / manual_den
    assert abs(loss.item() - manual) < 1e-4
