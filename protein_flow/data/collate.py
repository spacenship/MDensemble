"""Pads a list of variable-length protein-trajectory samples into a batch."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_protein_batch(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate a list of per-sample dicts (see :mod:`protein_flow.data.dataset`)
    into a padded batch with an added ``residue_mask``.

    Returns a dict with keys:
        sequence_embedding: [B, L_max, D_plm]
        source_coords:      [B, L_max, 3]
        target_coords:      [B, L_max, 3]
        residue_types:      [B, L_max] long
        residue_mask:       [B, L_max] bool
        temperature:        [B, 1]
        physical_delta_t:   [B, 1]
    """
    lengths = torch.tensor([sample["residue_types"].shape[0] for sample in samples], dtype=torch.long)
    max_length = int(lengths.max().item())

    sequence_embedding = pad_sequence([s["sequence_embedding"] for s in samples], batch_first=True)
    source_coords = pad_sequence([s["source_coords"] for s in samples], batch_first=True)
    target_coords = pad_sequence([s["target_coords"] for s in samples], batch_first=True)
    residue_types = pad_sequence([s["residue_types"] for s in samples], batch_first=True, padding_value=0)

    batch_size = len(samples)
    residue_mask = torch.arange(max_length).unsqueeze(0).expand(batch_size, max_length) < lengths.unsqueeze(1)

    temperature = torch.stack([s["temperature"].reshape(1) for s in samples], dim=0)
    physical_delta_t = torch.stack([s["physical_delta_t"].reshape(1) for s in samples], dim=0)

    return {
        "sequence_embedding": sequence_embedding,
        "source_coords": source_coords,
        "target_coords": target_coords,
        "residue_types": residue_types,
        "residue_mask": residue_mask,
        "temperature": temperature,
        "physical_delta_t": physical_delta_t,
    }
