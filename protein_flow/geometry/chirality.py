"""Signed backbone-dihedral pseudo-scalar: the chirality-breaking feature.

Every other invariant feature used elsewhere in this codebase (pairwise
distance, RBF of distance, normalized sequence separation, peptide
indicator) is built only from dot products and norms, which are identical
for a structure and its mirror image -- that is precisely what makes the
rest of the network equivariant under the *full* orthogonal group O(3)
(rotations AND reflections), not just the proper-rotation subgroup SO(3).
Real proteins are chiral (built from L-amino acids only), so an O(3) /
E(3)-equivariant model cannot, even in principle, distinguish a structure
from its mirror image.

The signed dihedral angle of four consecutive C-alpha atoms is a
*pseudo-scalar*: invariant under proper rotation and translation, but it
flips sign under any improper rotation (reflection). Adding it as a node
feature breaks the reflection symmetry while leaving proper
rotation/translation equivariance untouched, making the overall model
SE(3)-equivariant rather than E(3)-equivariant.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def compute_signed_dihedral(coords: Tensor, residue_mask: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """Signed dihedral angle of each consecutive C-alpha quadruple (i, i+1, i+2, i+3).

    The value for quadruple (i, i+1, i+2, i+3) is assigned to residue index
    i + 1. Residues without a fully-defined quadruple (chain termini, or
    any of the 4 residues being padding) get 0 with ``valid=False``.

    Args:
        coords: [B, L, 3].
        residue_mask: [B, L] bool.
        eps: numerical-stability epsilon for normalizing the central bond vector.

    Returns:
        dihedral: [B, L] signed angle in (-pi, pi], 0 where invalid.
        valid: [B, L] bool, True where the dihedral is well-defined.
    """
    batch_size, length, _ = coords.shape
    dihedral_full = coords.new_zeros(batch_size, length)
    valid_full = torch.zeros(batch_size, length, dtype=torch.bool, device=coords.device)

    if length < 4:
        return dihedral_full, valid_full

    x0 = coords[:, 0 : length - 3]
    x1 = coords[:, 1 : length - 2]
    x2 = coords[:, 2 : length - 1]
    x3 = coords[:, 3 : length]

    b1 = x1 - x0
    b2 = x2 - x1
    b3 = x3 - x2

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2_unit = b2 / (b2.norm(dim=-1, keepdim=True) + eps)
    m1 = torch.cross(n1, b2_unit, dim=-1)

    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)
    dihedral = torch.atan2(y, x)  # [B, L - 3]

    valid_quad = (
        residue_mask[:, 0 : length - 3]
        & residue_mask[:, 1 : length - 2]
        & residue_mask[:, 2 : length - 1]
        & residue_mask[:, 3 : length]
    )

    dihedral_full[:, 1 : length - 2] = dihedral * valid_quad.to(coords.dtype)
    valid_full[:, 1 : length - 2] = valid_quad

    return dihedral_full, valid_full


def compute_particle_dihedral(
    coords: Tensor,
    particle_mask: Tensor,
    ca_atom_index: Optional[Tensor] = None,
    atom_residue_index: Optional[Tensor] = None,
    residue_mask: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Per-particle signed backbone dihedral, for either representation.

    The pseudo-scalar is always defined on the **C-alpha backbone** -- that
    is what makes it a meaningful measure of backbone handedness. In the
    C-alpha representation the particles already are the backbone, so this
    reduces exactly to :func:`compute_signed_dihedral`. In the all-atom
    representation the dihedral is computed on the gathered CA trace and
    then broadcast to every atom of the owning residue, so a side-chain
    atom inherits its residue's backbone chirality signal.

    Args:
        coords: [B, N, 3] particle coordinates.
        particle_mask: [B, N] bool, True for valid particles.
        ca_atom_index: [B, L] atom index of each residue's CA (all-atom only).
        atom_residue_index: [B, N] residue index per atom (all-atom only).
        residue_mask: [B, L] bool residue validity (all-atom only).

    Returns:
        dihedral: [B, N] signed angle, 0 where invalid.
        valid: [B, N] bool.
    """
    if ca_atom_index is None or atom_residue_index is None or residue_mask is None:
        return compute_signed_dihedral(coords, particle_mask, eps=eps)

    ca_coords = torch.gather(coords, 1, ca_atom_index.unsqueeze(-1).expand(-1, -1, 3))  # [B, L, 3]
    residue_dihedral, residue_valid = compute_signed_dihedral(ca_coords, residue_mask, eps=eps)

    dihedral = torch.gather(residue_dihedral, 1, atom_residue_index)  # [B, N]
    valid = torch.gather(residue_valid, 1, atom_residue_index) & particle_mask
    return dihedral * valid.to(coords.dtype), valid
