#!/usr/bin/env python
"""How much of the displacement is predictable at all?

The model is asked for ``delta = x1 - x0``: where a protein goes over
``mdcath_frame_gap`` frames at temperature T. MD is a chaotic system under a
thermostat, so ``delta`` is a *sample* from ``p(delta | x0, T)``, not a
function of the conditioning -- mdCATH's own five replicas per (domain,
temperature) start alike and diverge. That means part of the target is
unpredictable in principle, and no architecture recovers it.

This measures the split, with no model involved:

  cos(delta_a, delta_b)  between two displacements of the *same* domain at the
                         same temperature but different starting frames. This
                         is what the conditioning cannot distinguish, so its
                         value is roughly how much direction is shared across
                         realisations -- the predictable part.
  profile r              the same comparison on per-residue *magnitude*
                         profiles. Flexibility is a property of the fold and
                         should survive where direction does not.

A near-zero direction correlation with a high profile correlation is the
signature of "which residues move is knowable, which way they move is not",
which is exactly what a generative model should reproduce and a regression
model cannot.

Usage:
    python scripts/diagnostics/predictability.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protein_flow.config import is_atom_level, load_config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.shard_manifest import ShardManifest
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.train_rotating import _build_dataset


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mdcath_backbone_rotate_displacement.yaml")
    parser.add_argument("--max-domains", type=int, default=6)
    parser.add_argument("--pairs-per-unit", type=int, default=12)
    args = parser.parse_args()

    config = load_config(args.config)
    config.data.batch_size = 1
    config.data.num_workers = 0
    # Many starting frames per trajectory, so several displacements of the same
    # (domain, temperature) can be compared against one another.
    config.data.mdcath_val_pairs_per_trajectory = args.pairs_per_unit
    atom_level = is_atom_level(config.data.representation)

    manifest = ShardManifest.load(Path(config.data.rotation.manifest_path))
    data_dir = Path(config.data.mdcath_dir)
    files = [p for p in (data_dir / Path(e.path).name for e in manifest.val_domains) if p.exists()]
    dataset = _build_dataset(config, data_dir, files, seed=config.data.seed + 1, is_validation=True)

    # Group displacements by (domain, temperature, replica-independent).
    units: dict = defaultdict(list)
    for index in range(len(dataset)):
        entry = dataset.trajectory_index[index // dataset.pairs_per_trajectory]
        key = (entry["domain"], entry["temperature"])
        if len(units) >= args.max_domains * 5 and key not in units:
            continue
        if len(units[key]) >= args.pairs_per_unit:
            continue
        units[key].append(index)

    loader_cache: dict = {}
    by_temperature = defaultdict(lambda: [0.0, 0.0, 0])

    for (domain, temperature), indices in units.items():
        if len(indices) < 2:
            continue
        displacements, profiles = [], []
        for index in indices:
            sample = dataset[index]
            batch = collate_protein_batch([sample])
            mask = batch["atom_mask"] if atom_level else batch["residue_mask"]
            x0 = batch["source_coords"]
            x1 = masked_kabsch_align(x0, batch["target_coords"], mask).aligned_target
            delta = ((x1 - x0) * mask.unsqueeze(-1))[0]
            displacements.append(delta)
            profiles.append(delta.norm(dim=-1))

        cosines, profile_correlations = [], []
        for a in range(len(displacements)):
            for b in range(a + 1, len(displacements)):
                da, db = displacements[a], displacements[b]
                # Per-atom direction agreement, averaged over atoms.
                cosine = (da * db).sum(-1) / (
                    da.norm(dim=-1).clamp(min=1e-8) * db.norm(dim=-1).clamp(min=1e-8)
                )
                cosines.append(float(cosine.mean()))
                profile_correlations.append(pearson(profiles[a], profiles[b]))

        slot = by_temperature[float(temperature)]
        slot[0] += sum(cosines) / len(cosines)
        slot[1] += sum(profile_correlations) / len(profile_correlations)
        slot[2] += 1

    if not by_temperature:
        print("error: no comparable displacement pairs found", file=sys.stderr)
        return 1

    print(f"agreement between two MD displacements of the same domain and temperature")
    print(f"(frame gap {config.data.mdcath_frame_gap}, different starting frames)\n")
    print(f"{'temperature':>12}{'direction cos':>15}{'profile r':>12}{'units':>8}")
    print("-" * 47)
    total = [0.0, 0.0, 0]
    for temperature in sorted(by_temperature):
        cosine, profile, count = by_temperature[temperature]
        print(f"{temperature:>9.0f} K {cosine / count:>15.3f}{profile / count:>12.3f}{count:>8d}")
        total[0] += cosine
        total[1] += profile
        total[2] += count
    print("-" * 47)
    print(f"{'all':>12}{total[0] / total[2]:>15.3f}{total[1] / total[2]:>12.3f}{total[2]:>8d}")
    print("\ndirection cos ~ 0 means which way the protein moves is not a function of")
    print("where it starts -- that part is the thermal noise and is irreducible.")
    print("profile r > 0 means which residues move *is* a property of the fold,")
    print("and is the part a model can and should learn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
