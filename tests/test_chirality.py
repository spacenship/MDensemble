import torch

from protein_flow.geometry.chirality import compute_signed_dihedral


def _random_proper_rotation(generator: torch.Generator) -> torch.Tensor:
    a = torch.randn(3, 3, generator=generator)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diagonal(r))
    q = q * d.unsqueeze(-2)
    if torch.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


def test_dihedral_invariant_under_proper_rotation_and_translation():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    coords = torch.randn(2, 8, 3, generator=gen)
    mask = torch.ones(2, 8, dtype=torch.bool)

    rotation = _random_proper_rotation(gen)
    translation = torch.randn(3, generator=gen) * 5.0
    coords_transformed = coords @ rotation + translation

    dihedral, valid = compute_signed_dihedral(coords, mask)
    dihedral_t, valid_t = compute_signed_dihedral(coords_transformed, mask)

    torch.testing.assert_close(valid, valid_t)
    torch.testing.assert_close(dihedral, dihedral_t, atol=1e-4, rtol=1e-4)


def test_dihedral_sign_flips_under_reflection():
    torch.manual_seed(1)
    gen = torch.Generator().manual_seed(1)
    coords = torch.randn(2, 8, 3, generator=gen)
    mask = torch.ones(2, 8, dtype=torch.bool)

    reflection = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    coords_reflected = coords @ reflection

    dihedral, valid = compute_signed_dihedral(coords, mask)
    dihedral_r, valid_r = compute_signed_dihedral(coords_reflected, mask)

    torch.testing.assert_close(valid, valid_r)
    torch.testing.assert_close(dihedral_r[valid], -dihedral[valid], atol=1e-4, rtol=1e-4)


def test_padding_and_termini_marked_invalid():
    coords = torch.randn(1, 10, 3)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[:, :6] = True  # residues 6-9 are padding

    dihedral, valid = compute_signed_dihedral(coords, mask)
    # chain termini (residue 0 and the last 2) never have a full quadruple
    assert not valid[0, 0]
    assert not valid[0, -1]
    assert not valid[0, -2]
    # quadruples touching padding (residue >= 6) must be invalid
    assert not valid[0, 5:].any()
    # interior residues with 4 fully valid, non-padding neighbors should be valid
    assert valid[0, 1:5].any()
    # invalid positions must have dihedral exactly 0
    assert torch.all(dihedral[~valid] == 0.0)


def test_short_sequence_returns_all_invalid_without_crashing():
    coords = torch.randn(2, 3, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    dihedral, valid = compute_signed_dihedral(coords, mask)
    assert dihedral.shape == (2, 3)
    assert not valid.any()
