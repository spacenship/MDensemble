import torch

from protein_flow.geometry.kabsch import masked_kabsch_align


def _random_rotation(batch: int, generator: torch.Generator) -> torch.Tensor:
    """Random proper (det=+1) rotation matrices via QR decomposition."""
    a = torch.randn(batch, 3, 3, generator=generator)
    q, r = torch.linalg.qr(a)
    # Fix signs so diagonal of r is positive (standard QR sign convention),
    # then force det(q) = +1.
    d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * d.unsqueeze(-2)
    det = torch.det(q)
    flip = torch.sign(det)
    q[..., -1] = q[..., -1] * flip.unsqueeze(-1)
    return q


def test_pure_rigid_transform_is_fully_recovered():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    batch, length = 4, 12
    source = torch.randn(batch, length, 3, generator=gen)
    mask = torch.ones(batch, length, dtype=torch.bool)

    rotation_true = _random_rotation(batch, gen)
    translation_true = torch.randn(batch, 3, generator=gen)
    target = torch.einsum("bli,bij->blj", source, rotation_true) + translation_true.unsqueeze(1)

    result = masked_kabsch_align(source, target, mask)

    assert result.pre_rmsd.mean() > 1e-2
    assert result.post_rmsd.max() < 5e-4
    torch.testing.assert_close(
        result.aligned_target, source, atol=1e-3, rtol=1e-3
    )

    # det(R) must be +1 (no reflection).
    dets = torch.det(result.rotation)
    torch.testing.assert_close(dets, torch.ones_like(dets), atol=1e-4, rtol=1e-4)


def test_padding_residues_are_ignored():
    torch.manual_seed(1)
    gen = torch.Generator().manual_seed(1)
    batch, length, valid_len = 2, 10, 6
    source = torch.randn(batch, length, 3, generator=gen)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[:, :valid_len] = True

    rotation_true = _random_rotation(batch, gen)
    translation_true = torch.randn(batch, 3, generator=gen)
    target = torch.einsum("bli,bij->blj", source, rotation_true) + translation_true.unsqueeze(1)

    # Corrupt padding region of target with garbage; alignment must be
    # unaffected since padding is excluded from centroid/covariance.
    target_corrupted = target.clone()
    target_corrupted[:, valid_len:] = torch.randn(batch, length - valid_len, 3, generator=gen) * 1000.0

    result_clean = masked_kabsch_align(source, target, mask)
    result_corrupt = masked_kabsch_align(source, target_corrupted, mask)

    torch.testing.assert_close(result_clean.rotation, result_corrupt.rotation, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result_clean.translation, result_corrupt.translation, atol=1e-4, rtol=1e-4)
    assert result_corrupt.post_rmsd.max() < 1e-3
    # Padding region of aligned_target must be zeroed.
    assert torch.all(result_corrupt.aligned_target[:, valid_len:] == 0.0)


def test_internal_conformational_change_is_preserved_not_removed():
    """Kabsch must remove rigid-body pose but must NOT erase genuine
    internal deformation between source and target."""
    torch.manual_seed(2)
    gen = torch.Generator().manual_seed(2)
    batch, length = 3, 16
    source = torch.randn(batch, length, 3, generator=gen)

    # Internal deformation: perturb each residue independently (breaks
    # rigidity -- no single rotation/translation can undo this).
    internal_deformation = 0.5 * torch.randn(batch, length, 3, generator=gen)
    deformed = source + internal_deformation

    rotation_true = _random_rotation(batch, gen)
    translation_true = torch.randn(batch, 3, generator=gen)
    target = torch.einsum("bli,bij->blj", deformed, rotation_true) + translation_true.unsqueeze(1)

    mask = torch.ones(batch, length, dtype=torch.bool)
    result = masked_kabsch_align(source, target, mask)

    # Rigid part must be undone: post_rmsd should track the residual
    # internal-deformation magnitude, not be near zero, and should be much
    # smaller than pre_rmsd (which is dominated by the rigid transform).
    internal_rmsd = torch.sqrt((internal_deformation ** 2).sum(-1).mean(-1))
    torch.testing.assert_close(result.post_rmsd, internal_rmsd, atol=0.1, rtol=0.2)
    assert result.post_rmsd.mean() > 1e-2
    assert result.post_rmsd.mean() < result.pre_rmsd.mean()


def test_det_always_positive_even_with_reflection_prone_data():
    """Even for point sets whose best-fit unconstrained SVD solution would
    be a reflection, the returned rotation must have det = +1."""
    torch.manual_seed(3)
    gen = torch.Generator().manual_seed(3)
    batch, length = 2, 5
    source = torch.randn(batch, length, 3, generator=gen)
    # Nearly-planar target to encourage a reflection-ambiguous fit.
    target = source.clone()
    target[..., 2] *= 0.001
    target = target + 0.01 * torch.randn(batch, length, 3, generator=gen)
    mask = torch.ones(batch, length, dtype=torch.bool)

    result = masked_kabsch_align(source, target, mask)
    dets = torch.det(result.rotation)
    torch.testing.assert_close(dets, torch.ones_like(dets), atol=1e-3, rtol=1e-3)
