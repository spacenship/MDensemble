"""Masked, batched Kabsch rigid-body alignment.

This module removes the rigid-body (rotation + translation) component
between paired source/target C-alpha structures. It does **not** remove
internal conformational change -- that is the whole point of using it as
flow-matching preprocessing: the aligned target still differs from the
source by whatever intrinsic folding/backbone motion occurred, only the
overall pose has been normalized away.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KabschResult:
    """Output of :func:`masked_kabsch_align`.

    Attributes:
        aligned_target: [B, L, 3] target coordinates rotated/translated onto
            the source frame. Padding residues are zeroed out.
        rotation: [B, 3, 3] proper rotation (det = +1) applied as
            ``target @ rotation + translation``.
        translation: [B, 3] translation applied after rotation.
        pre_rmsd: [B] masked RMSD between raw target and source.
        post_rmsd: [B] masked RMSD between aligned_target and source.
    """

    aligned_target: Tensor
    rotation: Tensor
    translation: Tensor
    pre_rmsd: Tensor
    post_rmsd: Tensor


def _masked_rmsd(a: Tensor, b: Tensor, mask: Tensor, count: Tensor, eps: float) -> Tensor:
    """Masked per-batch RMSD between two [B, L, 3] point sets."""
    squared_dist = ((a - b) ** 2).sum(dim=-1)  # [B, L]
    squared_dist = squared_dist * mask
    mean_squared = squared_dist.sum(dim=1) / count.squeeze(-1)
    return torch.sqrt(mean_squared + eps)


@torch.no_grad()
def masked_kabsch_align(
    source_coords: Tensor,
    target_coords: Tensor,
    residue_mask: Tensor,
    eps: float = 1e-8,
) -> KabschResult:
    """Align ``target_coords`` onto ``source_coords`` per batch element.

    Solves, for each batch element independently,

        argmin_{R in SO(3), b} sum_i mask_i * || target_i @ R + b - source_i ||^2

    Padding residues (``residue_mask`` False) are excluded from both the
    centroid computation and the cross-covariance matrix. This runs under
    ``torch.no_grad()`` since alignment is treated as fixed preprocessing,
    not a differentiable operation in the training loop.

    Args:
        source_coords: [B, L, 3] reference structure.
        target_coords: [B, L, 3] structure to be aligned onto the reference.
        residue_mask: [B, L] bool mask, True for valid (non-padding) residues.
        eps: numerical-stability epsilon used in RMSD sqrt and count clamping.

    Returns:
        A :class:`KabschResult`.
    """
    if source_coords.shape != target_coords.shape:
        raise ValueError("source_coords and target_coords must have the same shape")
    mask = residue_mask.to(dtype=source_coords.dtype)  # [B, L]
    count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
    mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]

    source_centroid = (source_coords * mask_expanded).sum(dim=1) / count  # [B, 3]
    target_centroid = (target_coords * mask_expanded).sum(dim=1) / count  # [B, 3]

    p = (source_coords - source_centroid.unsqueeze(1)) * mask_expanded  # [B, L, 3]
    q = (target_coords - target_centroid.unsqueeze(1)) * mask_expanded  # [B, L, 3]

    # Cross-covariance H = q^T p, summed over residues, per batch element.
    cross_covariance = torch.einsum("bli,blj->bij", q, p)  # [B, 3, 3]

    u, _, v_t = torch.linalg.svd(cross_covariance)
    v = v_t.transpose(-1, -2)
    det_sign = torch.sign(torch.det(v @ u.transpose(-1, -2)))  # [B]
    ones = torch.ones_like(det_sign)
    diag_correction = torch.diag_embed(torch.stack([ones, ones, det_sign], dim=-1))  # [B, 3, 3]

    rotation = u @ diag_correction @ v_t  # [B, 3, 3], det = +1 by construction

    translation = source_centroid - torch.einsum("bi,bij->bj", target_centroid, rotation)  # [B, 3]

    aligned_target_full = torch.einsum("bli,bij->blj", target_coords, rotation) + translation.unsqueeze(1)
    aligned_target = aligned_target_full * mask_expanded

    pre_rmsd = _masked_rmsd(target_coords, source_coords, mask, count, eps)
    post_rmsd = _masked_rmsd(aligned_target, source_coords, mask, count, eps)

    return KabschResult(
        aligned_target=aligned_target,
        rotation=rotation,
        translation=translation,
        pre_rmsd=pre_rmsd,
        post_rmsd=post_rmsd,
    )
