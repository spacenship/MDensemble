"""Flow-matching regression loss."""
from __future__ import annotations

from torch import Tensor

from protein_flow.utils import masked_mean


def flow_matching_loss(predicted_velocity: Tensor, target_velocity: Tensor, residue_mask: Tensor) -> Tensor:
    """Masked mean-squared-error between predicted and target velocity fields.

    Args:
        predicted_velocity: [B, L, 3].
        target_velocity: [B, L, 3].
        residue_mask: [B, L] bool, True for valid (non-padding) residues.

    Returns:
        Scalar loss tensor.
    """
    squared_error = (predicted_velocity - target_velocity).pow(2).sum(dim=-1)  # [B, L]
    return masked_mean(squared_error, residue_mask)


def direction_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    residue_mask: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Masked mean of ``1 - cos(predicted, target)``, in [0, 2].

    The MSE term alone goes to its minimum by shrinking the prediction when
    the target is dominated by unpredictable noise -- which is precisely how
    the coordinate-space run collapsed to 0.24% of the target magnitude. This
    term is scale-free, so it keeps supplying gradient on the *direction*
    even where the magnitude has been given up on, and it is what stops zero
    from being an attractor.

    Args:
        predicted_velocity: [B, L, 3].
        target_velocity: [B, L, 3].
        residue_mask: [B, L] bool, True for valid (non-padding) residues.
    """
    predicted_norm = predicted_velocity.norm(dim=-1).clamp(min=eps)
    target_norm = target_velocity.norm(dim=-1).clamp(min=eps)
    cosine = (predicted_velocity * target_velocity).sum(dim=-1) / (predicted_norm * target_norm)
    return masked_mean(1.0 - cosine, residue_mask)


def velocity_diagnostics(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    residue_mask: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Return ``(magnitude_ratio, mean_cosine)`` -- the two collapse detectors.

    ``magnitude_ratio`` is RMS||v_pred|| / RMS||v_target||: 1.0 means the
    field has the right scale, and the collapsed coordinate-space checkpoint
    measured 0.0024 here while its MSE looked healthy. ``mean_cosine`` is the
    masked mean cosine, which separates "too small" from "pointing the wrong
    way" -- the distinction a rescaling sweep would otherwise be needed to
    make.
    """
    mask = residue_mask.to(dtype=predicted_velocity.dtype)
    predicted_rms = masked_mean(predicted_velocity.pow(2).sum(dim=-1), mask).sqrt()
    target_rms = masked_mean(target_velocity.pow(2).sum(dim=-1), mask).sqrt()
    cosine = (predicted_velocity * target_velocity).sum(dim=-1) / (
        predicted_velocity.norm(dim=-1).clamp(min=eps) * target_velocity.norm(dim=-1).clamp(min=eps)
    )
    return predicted_rms / target_rms.clamp(min=eps), masked_mean(cosine, mask)
