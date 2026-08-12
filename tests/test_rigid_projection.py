"""The flow must stay in the subspace its target actually occupies.

``delta = x1_aligned - x0`` comes out of Kabsch, so it has no rigid-body
component: not the three translations (already handled by
``decoder.remove_com_velocity``) and not the three rotations. The first full
displacement run had nothing enforcing the second half of that and spent
17.0% of its output energy on rotation, which the evaluation's own alignment
then threw away.

These pin the projection down: that it removes what it claims to, that it
leaves untouched anything already free of rigid motion, that it does not
break SE(3)-equivariance (the whole reason the decoder is built the way it
is), and that the config flag reaches *both* the velocity and the base noise
-- projecting only one of the two would strand a rotation in the state that
nothing downstream could remove.
"""
import pytest
import torch

from protein_flow.config import Config
from protein_flow.flow.paths import DisplacementPath, sample_zero_com_noise
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.geometry.rigid import remove_rigid_motion
from protein_flow.models.dual_graph_flow import DualGraphFlowModel


def random_structure(batch=2, particles=40, seed=0):
    generator = torch.Generator().manual_seed(seed)
    coords = torch.randn(batch, particles, 3, generator=generator) * 8.0
    mask = torch.ones(batch, particles, dtype=torch.bool)
    return coords, mask, generator


def angular_momentum(field, coords, mask):
    weights = mask.unsqueeze(-1).to(field.dtype)
    centroid = (coords * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True)
    offsets = (coords - centroid) * weights
    return torch.cross(offsets, field * weights, dim=-1).sum(dim=1)


def test_projection_zeroes_linear_and_angular_momentum():
    coords, mask, generator = random_structure()
    field = torch.randn(coords.shape, generator=generator) * 3.0

    projected = remove_rigid_motion(field, coords, mask)

    assert angular_momentum(projected, coords, mask).abs().max() < 1e-3
    assert projected.mean(dim=1).abs().max() < 1e-4


def test_a_pure_rotation_is_removed_entirely():
    """The generators are exactly what the projection must annihilate."""
    coords, mask, _ = random_structure()
    centroid = coords.mean(dim=1, keepdim=True)
    omega = torch.tensor([0.3, -0.2, 0.5])
    rotation = torch.cross(omega.view(1, 1, 3).expand_as(coords), coords - centroid, dim=-1)

    projected = remove_rigid_motion(rotation, coords, mask)

    assert projected.abs().max() < 1e-3, "a pure rotation should project to zero"


def test_a_kabsch_aligned_displacement_is_left_alone():
    """The control: the training target already lives in the subspace.

    If the projection changed ``delta`` it would be removing signal, not
    rigid motion, and every flow-matching target would be quietly corrupted.
    """
    coords, mask, generator = random_structure(seed=3)
    target = coords + torch.randn(coords.shape, generator=generator) * 1.5
    aligned = masked_kabsch_align(coords, target, mask).aligned_target
    delta = aligned - coords

    projected = remove_rigid_motion(delta, coords, mask)

    kept = (projected.pow(2).sum(-1)).mean() / (delta.pow(2).sum(-1)).mean()
    assert kept > 0.999, f"projection removed {100 * (1 - kept):.2f}% of an aligned displacement"


def test_projection_is_equivariant_under_rotation_and_translation():
    coords, mask, generator = random_structure(seed=7)
    field = torch.randn(coords.shape, generator=generator) * 2.0

    angle = torch.tensor(0.7)
    rotation = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0.0],
        [torch.sin(angle), torch.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    shift = torch.tensor([4.0, -2.0, 9.0])

    direct = remove_rigid_motion(field, coords, mask) @ rotation.T
    transformed = remove_rigid_motion(field @ rotation.T, coords @ rotation.T + shift, mask)

    torch.testing.assert_close(direct, transformed, atol=1e-4, rtol=1e-4)


def test_padding_is_excluded_and_returned_as_zero():
    coords, mask, generator = random_structure(particles=30, seed=11)
    mask[:, 20:] = False
    field = torch.randn(coords.shape, generator=generator)

    projected = remove_rigid_motion(field, coords, mask)

    assert projected[:, 20:].abs().max() == 0.0
    # The momenta must be computed over valid particles only, so garbage in the
    # padded coordinates cannot move the answer.
    noisy = coords.clone()
    noisy[:, 20:] = 1e4
    torch.testing.assert_close(projected, remove_rigid_motion(field, noisy, mask),
                               atol=1e-4, rtol=1e-4)


def test_isotropic_noise_carries_only_the_expected_rotational_share():
    """Guards the reasoning that killed the first proposed fix.

    Projecting the *base noise* was the obvious remedy for a 17% rotational
    output, and it is wrong: an isotropic field on N particles has only
    ``3/(3N-3)`` of its energy in the rotation modes. If this number were
    large, the rotation could be inherited; it is not, so it is manufactured
    by the network and only projecting the velocity can remove it.
    """
    particles = 300
    coords, mask, generator = random_structure(batch=4, particles=particles, seed=5)
    noise = torch.randn(coords.shape, generator=generator)
    noise = noise - noise.mean(dim=1, keepdim=True)

    projected = remove_rigid_motion(noise, coords, mask)
    removed = 1.0 - projected.pow(2).sum() / noise.pow(2).sum()

    expected = 3.0 / (3 * particles - 3)
    assert removed < 5 * expected, (
        f"isotropic noise lost {100 * removed:.2f}% to alignment, far above the "
        f"{100 * expected:.2f}% the degree-of-freedom count predicts"
    )


def test_displacement_path_noise_is_rigid_free_only_when_asked():
    coords, mask, _ = random_structure(batch=2, particles=60, seed=13)

    plain = sample_zero_com_noise(
        coords.shape, mask, 2.667, coords.device, coords.dtype,
        torch.Generator().manual_seed(0), coords=coords,
    )
    projected = sample_zero_com_noise(
        coords.shape, mask, 2.667, coords.device, coords.dtype,
        torch.Generator().manual_seed(0), coords=coords, remove_rigid_motion=True,
    )

    assert angular_momentum(projected, coords, mask).abs().max() < 1e-2
    assert angular_momentum(plain, coords, mask).abs().max() > 1e-2
    # noise_scale keeps its meaning either way -- the rescale happens after the
    # projection, so the base distribution does not silently shrink.
    for field in (plain, projected):
        per_axis = (field.pow(2).sum() / (mask.sum() * 3)).sqrt()
        assert abs(float(per_axis) - 2.667) < 1e-3


def test_noise_projection_needs_coords():
    coords, mask, _ = random_structure(particles=20)
    with pytest.raises(ValueError, match="coords"):
        sample_zero_com_noise(
            coords.shape, mask, 2.667, coords.device, coords.dtype,
            torch.Generator().manual_seed(0), remove_rigid_motion=True,
        )


def _tiny_config(remove_rigid: bool) -> Config:
    config = Config()
    config.data.plm_dim = 16
    config.model.sequence_encoder.hidden_dim = 8
    config.model.geometric_encoder.hidden_dim = 8
    config.model.fusion.hidden_dim = 8
    config.model.fusion.condition_dim = 4
    config.model.graph.knn_k = 4
    config.model.graph.num_rbf = 4
    config.flow.path_type = "displacement"
    config.flow.remove_rigid_motion = remove_rigid
    return config


def test_the_flag_reaches_the_decoder_output():
    torch.manual_seed(0)
    batch, particles = 2, 24
    config = _tiny_config(remove_rigid=True)
    model = DualGraphFlowModel(config)

    coords = torch.randn(batch, particles, 3) * 6.0
    mask = torch.ones(batch, particles, dtype=torch.bool)
    state = torch.randn(batch, particles, 3)
    # Zero-initialised flow-state heads make the untrained decoder's output
    # independent of the state, so the coefficient MLPs are nudged first --
    # otherwise this would pass on a velocity that is trivially rigid-free.
    for head in (model.decoder.state_coeff_mlp, model.decoder.state_neighbour_mlp):
        torch.nn.init.normal_(head[-1].weight, std=0.5)
        torch.nn.init.normal_(head[-1].bias, std=0.5)

    velocity = model(
        coords, torch.rand(batch), torch.randn(batch, particles, 16),
        torch.zeros(batch, particles, dtype=torch.long), torch.full((batch, 1), 320.0),
        torch.zeros(batch, 1), mask, flow_state=state,
    )

    assert angular_momentum(velocity, coords, mask).abs().max() < 1e-2


def test_sampling_end_to_end_stays_rigid_free():
    """The invariant that matters: the *generated structure* has no rotation.

    This is the end the metric sees. If it holds, Kabsch alignment at
    evaluation time removes nothing, and none of the generated amplitude is
    spent on a mode the target cannot contain.
    """
    torch.manual_seed(1)
    batch, particles = 2, 24
    model = DualGraphFlowModel(_tiny_config(remove_rigid=True))
    coords = torch.randn(batch, particles, 3) * 6.0
    mask = torch.ones(batch, particles, dtype=torch.bool)

    generated, _ = model.sample(
        coords, torch.randn(batch, particles, 16),
        torch.zeros(batch, particles, dtype=torch.long), mask,
        torch.full((batch, 1), 320.0), torch.zeros(batch, 1),
        num_steps=5, solver="heun", generator=torch.Generator().manual_seed(0),
    )

    displacement = generated - coords
    aligned = masked_kabsch_align(coords, generated, mask).aligned_target - coords
    rigid_share = 1.0 - aligned.pow(2).sum() / displacement.pow(2).sum().clamp(min=1e-12)
    assert float(rigid_share) < 0.01, (
        f"alignment still removed {100 * float(rigid_share):.1f}% of the generated displacement"
    )


@pytest.mark.parametrize("particles", [270, 600, 2048])
def test_survives_amp_at_full_protein_size(particles):
    """The inertia tensor overflows fp16 long before the 512-residue cap.

    ``torch.einsum`` is autocast-eligible, so under AMP it runs in fp16 even
    when handed float32 inputs, and the inertia tensor sums an outer product
    over every particle. Measured, that crosses fp16's 65504 ceiling somewhere
    between 270 and 600 backbone atoms: it returns inf, ``linalg.solve`` turns
    that into NaN, and the training loop then *skips the update*. The failure
    is therefore silent and size-biased -- it drops exactly the largest
    proteins -- which is why it is pinned at several sizes and not just one.

    A 270-particle-only version of this test passes against the broken code.
    """
    if not torch.cuda.is_available():
        pytest.skip("autocast(fp16) needs a GPU")
    torch.manual_seed(0)
    extent = 12.0 * (particles / 270) ** (1 / 3)
    coords = torch.randn(2, particles, 3, device="cuda") * extent
    field = torch.randn(2, particles, 3, device="cuda") * 3.0
    mask = torch.ones(2, particles, dtype=torch.bool, device="cuda")

    for dtype in (torch.float16, torch.bfloat16):
        with torch.autocast("cuda", dtype=dtype):
            projected = remove_rigid_motion(field, coords, mask)
        assert torch.isfinite(projected).all(), (
            f"{dtype} autocast produced non-finite output at {particles} particles"
        )
        assert angular_momentum(projected.float(), coords, mask).abs().max() < 1.0


def test_default_is_off_so_the_completed_run_is_reproducible():
    assert Config().flow.remove_rigid_motion is False
    model = DualGraphFlowModel(_tiny_config(remove_rigid=False))
    assert model.decoder.remove_rigid_motion is False
    assert DisplacementPath().remove_rigid_motion is False
