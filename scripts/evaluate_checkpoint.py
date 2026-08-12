#!/usr/bin/env python
"""Sample from a trained checkpoint and score the structures it generates.

Training reports flow-matching MSE, which says how well the velocity field is
regressed but not whether integrating it produces a usable structure. This
runs the actual inference path -- the full ODE rollout at
``sampling.num_steps`` -- over the held-out validation domains and asks the
questions that matter for the model's purpose:

  Does it move?          |generated - source| against |target - source|.
                         A model that has collapsed to zero velocity scores a
                         perfect-looking RMSD here by simply not moving, so
                         this column is what distinguishes "learned" from
                         "learned to sit still".
  Does it move usefully? RMSD(generated, target) against the do-nothing
                         baseline RMSD(source, target), plus the fraction of
                         samples where it beats that baseline (win rate).
  Is it still a protein? Bond-length and bond-angle deviation from the source
                         topology, and steric clashes.

Everything is broken down by simulation temperature, because the displacement
the model must produce grows ~3.9x from 320 K to 450 K and a model that
ignores its temperature conditioning shows up here as a flat magnitude column.

**The "improve" column is not a success metric for a stochastic model.**
Which particular thermal fluctuation separated two frames is not predictable
from the first one, so the conditional mean -- returning the source unchanged
-- is the RMSD-optimal answer and will always score ~0% here no matter how
good the model is. Optimising this number is how the coordinate-space run
learned to sit still. Read the ``moved gen``/``moved tgt`` ratio instead: it
says whether the model produces motion of the right *scale*, which is a real
requirement. For distributional quality (does the generated *ensemble* match
the MD ensemble) use ``scripts/evaluate_ensemble.py``, which is only
meaningful once sampling is stochastic.

Usage:
    python scripts/evaluate_checkpoint.py --checkpoint checkpoints_mdcath_backbone/best.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_flow.config import is_atom_level, load_config
from protein_flow.data.shard_manifest import ShardManifest
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.inference import generate, load_model_for_inference
from protein_flow.losses.physics import compute_all_atom_physics_losses, compute_physics_losses
from protein_flow.train import _make_val_loader
from protein_flow.train_rotating import _build_dataset

ATOM_KEYS = ("atom_mask", "atom_residue_index", "atom_element", "ca_atom_index")


def masked_rmsd(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample RMSD over valid particles: [B]."""
    squared = ((a - b) ** 2).sum(dim=-1) * mask
    return (squared.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).sqrt()


def masked_magnitude(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS displacement: [B]."""
    return ((delta ** 2).sum(dim=-1) * mask).sum(dim=1).div(mask.sum(dim=1).clamp(min=1)).sqrt()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mdcath_backbone_rotate.yaml")
    parser.add_argument("--checkpoint", default="checkpoints_mdcath_backbone/best.pt")
    parser.add_argument("--num-steps", type=int, default=None, help="ODE steps (default: sampling.num_steps)")
    parser.add_argument("--solver", default=None, help="euler | heun (default: sampling.solver)")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=None,
                        help="cap this process's VRAM, to stay out of a concurrent run's way")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.memory_fraction:
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)

    config = load_config(args.config)
    config.data.batch_size = args.batch_size
    num_steps = args.num_steps or config.sampling.num_steps
    solver = args.solver or config.sampling.solver
    atom_level = is_atom_level(config.data.representation)

    manifest = ShardManifest.load(Path(config.data.rotation.manifest_path))
    data_dir = Path(config.data.mdcath_dir)
    val_files = [p for p in (data_dir / Path(e.path).name for e in manifest.val_domains) if p.exists()]
    if not val_files:
        print(f"error: no validation shards resident under {data_dir}", file=sys.stderr)
        return 1
    dataset = _build_dataset(config, data_dir, val_files, seed=config.data.seed + 1, is_validation=True)
    loader = _make_val_loader(dataset, config)

    model = load_model_for_inference(args.checkpoint, config, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint     : {args.checkpoint} (step {checkpoint.get('step')})")
    print(f"held-out       : {len(val_files)} domain(s), {len(dataset)} pair(s)")
    print(f"sampling       : {num_steps} steps, {solver}")
    print(f"representation : {config.data.representation}\n")

    totals = defaultdict(float)
    by_temperature = defaultdict(lambda: defaultdict(float))
    seen = 0

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            residue_mask = batch["residue_mask"]
            particle_mask = batch["atom_mask"] if atom_level else residue_mask
            extra = {k: batch[k] for k in ATOM_KEYS} if atom_level else {}
            if "esm_input_ids" in batch:
                extra["esm_input_ids"] = batch["esm_input_ids"]
                extra["esm_attention_mask"] = batch["esm_attention_mask"]

            source = batch["source_coords"]
            target = masked_kabsch_align(source, batch["target_coords"], particle_mask).aligned_target

            generated, _ = generate(
                model, source, batch["sequence_embedding"], batch["residue_types"],
                residue_mask, batch["temperature"], batch["physical_delta_t"],
                num_steps=num_steps, solver=solver, **extra,
            )

            mask = particle_mask.float()
            rmsd_generated = masked_rmsd(generated, target, mask)
            rmsd_source = masked_rmsd(source, target, mask)
            moved_generated = masked_magnitude(generated - source, mask)
            moved_target = masked_magnitude(target - source, mask)

            if atom_level:
                physics = compute_all_atom_physics_losses(
                    generated, source, particle_mask,
                    batch["bond_index"], batch["bond_mask"],
                    batch["angle_index"], batch["angle_mask"],
                    config.model.graph,
                    config.loss.physics.clash_threshold_for(config.data.representation),
                )
            else:
                physics = compute_physics_losses(
                    generated, source, residue_mask, config.model.graph,
                    config.loss.physics.clash_threshold, config.loss.physics.clash_seq_sep,
                )

            batch_size = source.shape[0]
            seen += batch_size
            totals["rmsd_generated"] += rmsd_generated.sum().item()
            totals["rmsd_source"] += rmsd_source.sum().item()
            totals["moved_generated"] += moved_generated.sum().item()
            totals["moved_target"] += moved_target.sum().item()
            totals["wins"] += (rmsd_generated < rmsd_source).sum().item()
            for name, value in (("bond", physics.bond), ("angle", physics.angle), ("clash", physics.clash)):
                totals[name] += float(value) * batch_size

            for position in range(batch_size):
                key = float(batch["temperature"][position].item())
                slot = by_temperature[key]
                slot["n"] += 1
                slot["rmsd_generated"] += rmsd_generated[position].item()
                slot["rmsd_source"] += rmsd_source[position].item()
                slot["moved_generated"] += moved_generated[position].item()
                slot["moved_target"] += moved_target[position].item()
                slot["wins"] += float(rmsd_generated[position] < rmsd_source[position])

    if seen == 0:
        print("error: no samples evaluated", file=sys.stderr)
        return 1

    generated_rmsd = totals["rmsd_generated"] / seen
    source_rmsd = totals["rmsd_source"] / seen
    print(f"{'':<12}{'RMSD gen':>10}{'RMSD src':>10}{'improve':>10}{'win rate':>10}"
          f"{'moved gen':>11}{'moved tgt':>11}{'ratio':>8}")
    print("-" * 82)
    for temperature in sorted(by_temperature):
        slot = by_temperature[temperature]
        n = slot["n"]
        gen, src = slot["rmsd_generated"] / n, slot["rmsd_source"] / n
        mg, mt = slot["moved_generated"] / n, slot["moved_target"] / n
        print(f"{temperature:>7.0f} K   {gen:>10.3f}{src:>10.3f}{100 * (1 - gen / src):>9.2f}%"
              f"{100 * slot['wins'] / n:>9.1f}%{mg:>11.3f}{mt:>11.3f}{mg / mt:>8.3f}")
    moved_generated = totals["moved_generated"] / seen
    moved_target = totals["moved_target"] / seen
    print("-" * 82)
    print(f"{'all':<12}{generated_rmsd:>10.3f}{source_rmsd:>10.3f}"
          f"{100 * (1 - generated_rmsd / source_rmsd):>9.2f}%"
          f"{100 * totals['wins'] / seen:>9.1f}%{moved_generated:>11.3f}{moved_target:>11.3f}"
          f"{moved_generated / moved_target:>8.3f}")
    print(f"\nphysical validity of generated structures (0 = matches the source topology)")
    print(f"  bond  {totals['bond'] / seen:.6f}   angle {totals['angle'] / seen:.6f}"
          f"   clash {totals['clash'] / seen:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
