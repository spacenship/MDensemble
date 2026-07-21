"""Checkpoint loading and ODE sampling for inference/generation."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor

from protein_flow.config import Config
from protein_flow.models.dual_graph_flow import DualGraphFlowModel
from protein_flow.train import load_checkpoint


def load_model_for_inference(checkpoint_path: str | Path, config: Config, device: torch.device) -> DualGraphFlowModel:
    """Builds a :class:`DualGraphFlowModel` from ``config`` and loads trained weights."""
    model = DualGraphFlowModel(config).to(device)
    load_checkpoint(Path(checkpoint_path), model)
    model.eval()
    return model


def generate(
    model: DualGraphFlowModel,
    source_coords: Tensor,
    sequence_embedding: Tensor,
    residue_types: Tensor,
    residue_mask: Tensor,
    temperature: Tensor,
    physical_delta_t: Tensor,
    num_steps: int = 50,
    solver: str = "heun",
    return_trajectory: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Thin wrapper around :meth:`DualGraphFlowModel.sample`."""
    return model.sample(
        source_coords, sequence_embedding, residue_types, residue_mask,
        temperature, physical_delta_t, num_steps=num_steps, solver=solver,
        return_trajectory=return_trajectory,
    )
