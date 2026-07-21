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
