import torch

from protein_flow.config import GraphConfig
from protein_flow.losses.physics import (
    bond_angle_loss,
    bond_distance_loss,
    clash_loss,
    compute_consecutive_cos_angles,
    compute_consecutive_distances,
    compute_physics_losses,
    endpoint_rmsd_loss,
)


def test_bond_distance_loss_zero_when_matching_reference():
    torch.manual_seed(0)
    coords = torch.randn(2, 6, 3)
    mask = torch.ones(2, 6, dtype=torch.bool)
    ref_dist, valid_pair = compute_consecutive_distances(coords, mask)
    loss = bond_distance_loss(coords, ref_dist, valid_pair)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_bond_distance_loss_penalizes_stretching():
    coords = torch.zeros(1, 3, 3)
    coords[0, 1, 0] = 1.0
    coords[0, 2, 0] = 2.0
    mask = torch.ones(1, 3, dtype=torch.bool)
    ref_dist, valid_pair = compute_consecutive_distances(coords, mask)  # all bonds = 1.0

    stretched = coords.clone()
    stretched[0, 2, 0] = 3.0  # bond 1-2 now length 2.0 instead of 1.0
    loss = bond_distance_loss(stretched, ref_dist, valid_pair)
    # bond 0-1 unchanged (err=0), bond 1-2 stretched by 1.0 (err=1.0) -> mean = 0.5
    assert loss.item() > 0.4


def test_bond_distance_loss_ignores_padding():
    coords = torch.randn(1, 5, 3)
    mask = torch.tensor([[True, True, True, False, False]])
    ref_dist, valid_pair = compute_consecutive_distances(coords, mask)
    assert valid_pair.tolist() == [[True, True, False, False]]

    corrupted = coords.clone()
    corrupted[0, 4] += 100.0  # corrupt a padding residue
    loss_clean = bond_distance_loss(coords, ref_dist, valid_pair)
    loss_corrupt = bond_distance_loss(corrupted, ref_dist, valid_pair)
    assert torch.isclose(loss_clean, loss_corrupt, atol=1e-6)


def test_bond_angle_loss_zero_for_straight_chain():
    # perfectly straight chain: cos(angle) at each interior residue = -1
    # (vectors to neighbors point in opposite directions)
    coords = torch.zeros(1, 4, 3)
    for i in range(4):
        coords[0, i, 0] = float(i)
    mask = torch.ones(1, 4, dtype=torch.bool)
    ref_cos, valid_triplet = compute_consecutive_cos_angles(coords, mask)
    torch.testing.assert_close(ref_cos, -torch.ones_like(ref_cos))
    loss = bond_angle_loss(coords, ref_cos, valid_triplet)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_bond_angle_loss_detects_kink():
    coords = torch.zeros(1, 3, 3)
    coords[0, 0] = torch.tensor([-1.0, 0.0, 0.0])
    coords[0, 1] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 2] = torch.tensor([1.0, 0.0, 0.0])  # straight -> cos = -1
    mask = torch.ones(1, 3, dtype=torch.bool)
    ref_cos, valid_triplet = compute_consecutive_cos_angles(coords, mask)

    kinked = coords.clone()
    kinked[0, 2] = torch.tensor([0.0, 1.0, 0.0])  # 90 degree kink -> cos = 0
    loss = bond_angle_loss(kinked, ref_cos, valid_triplet)
    assert loss.item() > 0.5


def test_clash_loss_penalizes_close_non_covalent_pairs():
    # residues 0 and 5 are far apart in sequence but placed very close in space.
    coords = torch.zeros(1, 6, 3)
    for i in range(6):
        coords[0, i, 0] = float(i) * 10.0  # covalent neighbors far apart to avoid clash there
    coords[0, 5] = coords[0, 0] + torch.tensor([0.5, 0.0, 0.0])  # residue 5 clashing with residue 0
    mask = torch.ones(1, 6, dtype=torch.bool)
    graph_config = GraphConfig(knn_k=5)

    loss = clash_loss(coords, mask, graph_config, clash_threshold=3.5, clash_seq_sep=2)
    assert loss.item() > 0.0


def test_clash_loss_ignores_covalent_neighbors():
    # a tight, physically bonded chain (~3.8 A spacing) should not be
    # penalized by the clash loss even though residues are close together.
    coords = torch.zeros(1, 5, 3)
    for i in range(5):
        coords[0, i, 0] = float(i) * 3.8
    mask = torch.ones(1, 5, dtype=torch.bool)
    graph_config = GraphConfig(knn_k=4)

    loss = clash_loss(coords, mask, graph_config, clash_threshold=3.5, clash_seq_sep=2)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_clash_loss_ignores_padding():
    torch.manual_seed(0)
    coords = torch.randn(1, 8, 3) * 20.0
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, :4] = True
    coords[0, 4:] = coords[0, 0]  # place padding right on top of a valid residue
    graph_config = GraphConfig(knn_k=4)

    loss = clash_loss(coords, mask, graph_config, clash_threshold=3.5, clash_seq_sep=2)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_compute_physics_losses_gradients_flow():
    torch.manual_seed(1)
    graph_config = GraphConfig(knn_k=4)
    reference = torch.randn(2, 8, 3)
    pred = reference.clone().detach().requires_grad_(True)
    mask = torch.ones(2, 8, dtype=torch.bool)

    outputs = compute_physics_losses(pred, reference, mask, graph_config, clash_threshold=3.5, clash_seq_sep=2)
    total = outputs.bond + outputs.angle + outputs.clash
    total.backward()
    assert pred.grad is not None


def test_endpoint_rmsd_loss_zero_when_matching():
    coords = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    loss = endpoint_rmsd_loss(coords, coords, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-3)
