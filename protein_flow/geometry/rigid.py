"""Projection of a per-particle vector field onto the non-rigid subspace.

The displacement this repo transports is ``delta = x1_aligned - x0`` where
``x1_aligned`` comes out of Kabsch. That construction removes the rigid-body
part **exactly**: measured over the holdout, re-aligning ``delta`` takes away
0.0% of its energy, against 74.3% for the same displacement before alignment
(``scripts/diagnostics/rigid_share.py``). The target therefore lives in the
subspace orthogonal to the six rigid-body modes -- three translations and
three rotations about the centroid -- and so does the velocity ``delta - eps``
once the base noise is projected too.

``VectorFieldDecoder`` already projects out the three translations
(``remove_com_velocity``). Nothing projected out the three rotations, and the
trained model turned out to spend **17.0% of its output energy** on them.
That is not inherited from the base distribution -- isotropic noise on N
particles carries only ``3/(3N-3)``, measured 0.4% at N~270 -- so it is
manufactured by the network and then discarded by the evaluation's own Kabsch
step. Amplitude spent there cannot score.

Projecting is safe rather than merely helpful: it is an orthogonal projection
onto a subspace that provably contains the target, so it can only reduce the
flow-matching error, never increase it.
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["remove_rigid_motion"]


def remove_rigid_motion(
    field: Tensor,
    coords: Tensor,
    mask: Tensor,
    eps: float = 1e-8,
    remove_translation: bool = True,
) -> Tensor:
    """Project ``field`` onto zero linear and angular momentum about ``coords``.

    Args:
        field: [B, N, 3] per-particle vector field (a velocity or a
            displacement).
        coords: [B, N, 3] positions the rotation is taken about. In the
            displacement flow this is the **source structure**, which is fixed
            for the whole trajectory, so the six modes being removed are the
            same at every integration step.
        mask: [B, N] bool. Padding particles are excluded from the centroid,
            the inertia tensor and the momenta, and are returned as exactly
            zero.
        remove_translation: also subtract the masked mean, i.e. project out
            the three translation modes. Left switchable because the decoder
            already does this itself.

    Returns:
        [B, N, 3] the field with its rigid-body component removed.

    Equivariance: the three rotation generators are ``e_k x (x_i - centroid)``,
    whose span rotates with the structure, so the projection commutes with a
    global rotation of ``(coords, field)`` and with a translation of
    ``coords``. This is what keeps it usable inside an SE(3)-equivariant
    decoder.
    """
    if field.shape != coords.shape:
        raise ValueError(f"field {tuple(field.shape)} and coords {tuple(coords.shape)} must match")

    # Autocast is disabled for the whole body, not just cast around it.
    # ``torch.einsum`` is autocast-eligible, so under AMP it runs in fp16 even
    # when handed float32 inputs, and the inertia tensor is a sum of outer
    # products over every particle: at 600 backbone atoms it already exceeds
    # fp16's 65504 ceiling and returns inf, which the solve turns into NaN.
    # Measured, that begins between 270 and 600 particles, so a 512-residue
    # cap crosses it comfortably -- and it would have shown up as skipped
    # updates on exactly the largest proteins in the batch.
    with torch.autocast(device_type=field.device.type, enabled=False):
        return _project(field, coords, mask, eps, remove_translation)


def _project(
    field: Tensor,
    coords: Tensor,
    mask: Tensor,
    eps: float,
    remove_translation: bool,
) -> Tensor:
    # float64 inputs keep their precision; everything else is computed in
    # float32, which is what the 3x3 solve needs to be trustworthy.
    working = torch.float64 if field.dtype == torch.float64 else torch.float32
    weights = mask.to(working).unsqueeze(-1)
    values = field.to(working) * weights
    positions = coords.to(working) * weights

    count = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    centroid = positions.sum(dim=1, keepdim=True) / count
    offsets = (coords.to(working) - centroid) * weights

    if remove_translation:
        values = values - (values.sum(dim=1, keepdim=True) / count) * weights

    # Angular momentum of the field about the centroid, with unit masses.
    angular = torch.cross(offsets, values, dim=-1).sum(dim=1)  # [B, 3]

    # Inertia tensor: sum_i (|r_i|^2 I - r_i r_i^T), padding excluded because
    # offsets are already masked to zero.
    squared = offsets.pow(2).sum(dim=-1).sum(dim=1)  # [B]
    identity = torch.eye(3, device=field.device, dtype=working).expand(field.shape[0], 3, 3)
    outer = torch.einsum("bni,bnj->bij", offsets, offsets)
    inertia = squared.view(-1, 1, 1) * identity - outer

    # A protein is never collinear, but a padded-to-one-particle row or a
    # degenerate batch entry would be, so regularise rather than risk a
    # singular solve poisoning the whole batch with NaNs.
    scale = squared.view(-1, 1, 1).clamp(min=1.0)
    inertia = inertia + eps * scale * identity

    omega = torch.linalg.solve(inertia, angular.unsqueeze(-1)).squeeze(-1)  # [B, 3]
    rotational = torch.cross(omega.unsqueeze(1).expand_as(offsets), offsets, dim=-1)

    return ((values - rotational * weights) * weights).to(field.dtype)
