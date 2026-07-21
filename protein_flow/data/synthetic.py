"""Synthetic protein-trajectory dataset for smoke tests and CPU training.

Generates a persistent-random-walk C-alpha backbone (roughly constant
~3.8 A bond length, smooth bending) as the source structure, then derives
a target structure by (a) applying a small further smooth bending/torsion
perturbation to the direction sequence -- genuine internal conformational
change -- and (b) applying a random rigid rotation and translation on top.
Kabsch alignment (done later, at training time) removes (b) but must
leave (a) intact.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from protein_flow.config import DataConfig
from protein_flow.data.dataset import ProteinTrajectoryDataset


def _random_rotation(generator: torch.Generator) -> torch.Tensor:
    a = torch.randn(3, 3, generator=generator)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diagonal(r))
    q = q * d.unsqueeze(-2)
    if torch.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


def _persistent_random_walk_backbone(
    length: int, step: float, bend_std: float, generator: torch.Generator
) -> torch.Tensor:
    """Build an [length, 3] backbone via a persistent random walk with unit-norm
    per-step directions (so consecutive bond lengths are exactly ``step``)."""
    direction = torch.randn(3, generator=generator)
    direction = direction / (direction.norm() + 1e-8)
    coords = [torch.zeros(3)]
    for _ in range(length - 1):
        direction = direction + bend_std * torch.randn(3, generator=generator)
        direction = direction / (direction.norm() + 1e-8)
        coords.append(coords[-1] + direction * step)
    return torch.stack(coords, dim=0)


def _deformed_backbone(
    source: torch.Tensor, step: float, deform_std: float, generator: torch.Generator
) -> torch.Tensor:
    """Internal conformational change: re-walk the chain with directions
    perturbed slightly from the ones implied by ``source``, keeping bond
    lengths ~step (a genuine internal deformation, not rigid-body motion)."""
    length = source.shape[0]
    directions = source[1:] - source[:-1]
    directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
    coords = [source[0].clone()]
    for i in range(length - 1):
        direction = directions[i] + deform_std * torch.randn(3, generator=generator)
        direction = direction / (direction.norm() + 1e-8)
        coords.append(coords[-1] + direction * step)
    return torch.stack(coords, dim=0)


class SyntheticProteinTrajectoryDataset(ProteinTrajectoryDataset):
    """Generates random polymer-like trajectory pairs on the fly, deterministically per index."""

    def __init__(self, config: DataConfig, size: int, seed: int, bond_length: float = 3.8,
                 bend_std: float = 0.15, deform_std: float = 0.2):
        self.config = config
        self.size = size
        self.seed = seed
        self.bond_length = bond_length
        self.bend_std = bend_std
        self.deform_std = deform_std

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)

        length = int(
            torch.randint(self.config.min_length, self.config.max_length + 1, (1,), generator=generator).item()
        )

        source_coords = _persistent_random_walk_backbone(length, self.bond_length, self.bend_std, generator)
        deformed = _deformed_backbone(source_coords, self.bond_length, self.deform_std, generator)

        rotation = _random_rotation(generator)
        translation = torch.randn(3, generator=generator) * 5.0
        target_coords = deformed @ rotation + translation

        sequence_embedding = torch.randn(length, self.config.plm_dim, generator=generator)
        residue_types = torch.randint(
            0, self.config.num_amino_acid_types, (length,), generator=generator, dtype=torch.long
        )
        temperature = 273.0 + torch.rand(1, generator=generator) * 100.0  # Kelvin, ~[273, 373)
        physical_delta_t = torch.rand(1, generator=generator) * 10.0 + 0.1  # e.g. nanoseconds, > 0

        return {
            "sequence_embedding": sequence_embedding,
            "source_coords": source_coords,
            "target_coords": target_coords,
            "residue_types": residue_types,
            "temperature": temperature,
            "physical_delta_t": physical_delta_t,
        }
