"""Checkpoint loading and ODE sampling for inference/generation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

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
    generator: Optional[torch.Generator] = None,
    **atom_inputs: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Thin wrapper around :meth:`DualGraphFlowModel.sample`.

    ``atom_inputs`` must be forwarded, not dropped: in the ``backbone`` and
    ``heavy_atom`` representations the model needs ``atom_mask``,
    ``atom_residue_index``, ``atom_element`` and ``ca_atom_index`` (plus
    ``esm_input_ids``/``esm_attention_mask`` when the PLM runs in-graph) to
    build the geometric graph at all. Omitting them silently reverts the
    model to its C-alpha code path, which then fails on the shape mismatch
    between residues and particles.

    ``generator`` seeds the noise draw of the displacement flow, and is
    ignored by the (deterministic) coordinate-space paths.
    """
    return model.sample(
        source_coords, sequence_embedding, residue_types, residue_mask,
        temperature, physical_delta_t, num_steps=num_steps, solver=solver,
        return_trajectory=return_trajectory, generator=generator, **atom_inputs,
    )
