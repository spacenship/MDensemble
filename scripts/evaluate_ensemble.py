#!/usr/bin/env python
"""Score a sampled *ensemble* against the ground-truth MD ensemble.

Why this exists separately from ``evaluate_checkpoint.py``: for a stochastic
model, "how close is the generated structure to the target frame" is not a
measure of quality. Which particular thermal fluctuation occurred between two
frames is not predictable from the first one, so the conditional mean -- doing
nothing -- wins that comparison by construction, and a model that optimises it
learns to sit still. The coordinate-space run did exactly that: 0.00% RMSD
improvement, 4.7% of the required motion, and no rescaling of its output
helped.

What it should be judged on is whether the *distribution* it produces matches
the one MD produces. The comparison here is deliberately made against the
distribution the model is actually trained on -- the displacement over
``mdcath_frame_gap`` frames, p(x1 - x0 | x0, T) -- and not against the
pooled full-trajectory ensemble that AlphaFlow-style evaluations use. Those
are not the same object: at 450 K a full mdCATH trajectory unfolds (measured
RMSF 12.3 A about its own mean), and no single-shot 5-frame model can or
should reproduce that spread. Scoring against it would report a failure that
says nothing about the model. Reaching the long-horizon ensemble needs an
autoregressive rollout, which this model is deliberately not trained for --
that is the path where the sibling project's naive flow matching came apart
(contact Jaccard 0.115).

  moved ratio     RMS|delta| generated / ground truth. 1.0 means the model
                  produces motion of the right size; the coordinate-space run
                  scored 0.047 here, i.e. it barely moved.
  spread          mean pairwise RMSD among the generated structures, over
                  RMS|delta| of the ground truth. Exactly 0 means the sampler
                  is deterministic (the collapse this reformulation fixes);
                  sqrt(2) ~= 1.414 is what independent draws from a matching
                  displacement distribution give.
  profile r       Pearson correlation of the per-residue RMS displacement
                  profile. Says whether the model knows *which* residues move.
  JS divergence   Jensen-Shannon between the per-residue displacement
                  magnitude distributions. 0 is perfect.
  contact J       Jaccard overlap between the generated ensemble's mean
                  contact map and the source structure's, alongside the same
                  quantity measured on real MD pairs. The validity guard: a
                  model can score well on spread by tearing the protein apart.
                  Judge it against the MD column, never against 1.0 -- five
                  frames is ~5.5 ns and the fold really does reorganise
                  (measured GT: 0.786 at 320 K falling to 0.304 at 450 K).
                  Scoring *above* MD means moving less than MD, which is the
                  coordinate-space failure wearing a good-looking number.

The displacement statistics are computed over all flowing particles; the
contact map is C-alpha only, so it is comparable to the literature.

Usage:
    python scripts/evaluate_ensemble.py \
        --config configs/mdcath_backbone_rotate_displacement.yaml \
        --checkpoint checkpoints_mdcath_displacement/best.pt \
        --num-samples 100
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_flow.config import is_atom_level, load_config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.shard_manifest import ShardManifest
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.inference import generate, load_model_for_inference
from protein_flow.losses.physics import compute_all_atom_physics_losses, compute_physics_losses
from protein_flow.train_rotating import _build_dataset

CONTACT_CUTOFF = 8.0  # angstroms, the standard C-alpha contact definition
CONTACT_SEQUENCE_SEPARATION = 3  # ignore trivial near-diagonal contacts


# --- ensemble statistics -----------------------------------------------------


def align_to_mean(ensemble: torch.Tensor, iterations: int = 2) -> torch.Tensor:
    """Kabsch-align every frame to the running ensemble mean: [K, N, 3].

    Without this, rigid-body tumbling dominates every metric below -- the
    ground-truth mdCATH replicas drift by up to ~100 A of centre of mass over
    a trajectory, which would swamp the internal motion the metrics are about.
    """
    mask = torch.ones(ensemble.shape[0], ensemble.shape[1], dtype=torch.bool, device=ensemble.device)
    aligned = ensemble
    for _ in range(iterations):
        reference = aligned.mean(dim=0, keepdim=True).expand_as(aligned)
        aligned = masked_kabsch_align(reference, aligned, mask).aligned_target
    return aligned


def displacement_profile(displacements: torch.Tensor) -> torch.Tensor:
    """Per-particle RMS displacement magnitude over the sample: [N]."""
    return displacements.pow(2).sum(dim=-1).mean(dim=0).sqrt()


def displacement_rms(displacements: torch.Tensor) -> float:
    """Single RMS displacement magnitude over particles and samples."""
    return float(displacements.pow(2).sum(dim=-1).mean().sqrt())


def mean_pairwise_rmsd(ensemble: torch.Tensor, num_pairs: int, generator: torch.Generator) -> float:
    """Mean RMSD between random pairs of structures, after alignment.

    Zero to numerical precision means every draw produced the same structure,
    which is what a deterministic sampler looks like here.
    """
    aligned = align_to_mean(ensemble)
    count = aligned.shape[0]
    if count < 2:
        return 0.0
    left = torch.randint(0, count, (num_pairs,), generator=generator)
    right = torch.randint(0, count, (num_pairs,), generator=generator)
    keep = left != right
    if not keep.any():
        return 0.0
    left, right = left[keep], right[keep]
    return float((aligned[left] - aligned[right]).pow(2).sum(dim=-1).mean(dim=-1).sqrt().mean())


def jensen_shannon(left: torch.Tensor, right: torch.Tensor, num_bins: int = 50) -> float:
    """JS divergence between two samples, via a shared histogram: [0, log 2]."""
    low = float(min(left.min(), right.min()))
    high = float(max(left.max(), right.max()))
    if high <= low:
        return 0.0
    edges = torch.linspace(low, high, num_bins + 1)
    p = torch.histogram(left, bins=edges).hist
    q = torch.histogram(right, bins=edges).hist
    p = p / p.sum().clamp(min=1e-12)
    q = q / q.sum().clamp(min=1e-12)
    m = 0.5 * (p + q)

    def kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        support = a > 0
        return (a[support] * (a[support] / b[support].clamp(min=1e-12)).log()).sum()

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def mean_contact_map(ensemble: torch.Tensor, num_frames: int, generator: torch.Generator) -> torch.Tensor:
    """Contact frequency per residue pair over sampled frames: [N, N]."""
    count = ensemble.shape[0]
    index = torch.randperm(count, generator=generator)[: min(num_frames, count)]
    frames = ensemble[index]
    contacts = (torch.cdist(frames, frames) < CONTACT_CUTOFF).float().mean(dim=0)
    position = torch.arange(contacts.shape[0], device=contacts.device)
    offsets = (position.unsqueeze(0) - position.unsqueeze(1)).abs()
    return contacts * (offsets >= CONTACT_SEQUENCE_SEPARATION)


def contact_jaccard(predicted: torch.Tensor, reference: torch.Tensor, threshold: float = 0.5) -> float:
    """Jaccard overlap of the two thermally-averaged contact maps."""
    p = predicted > threshold
    q = reference > threshold
    union = (p | q).sum()
    if union == 0:
        return 1.0
    return float((p & q).sum() / union)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a * b).sum() / denominator)


# --- data --------------------------------------------------------------------


def load_ground_truth_displacements(
    entry: dict, particle_indices: np.ndarray, frame_gap: int, max_pairs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kabsch-aligned displacements over ``frame_gap`` frames, and their sources.

    Returns ``(displacements, sources)``, both [P, N, 3]. The sources are
    needed to measure what contact Jaccard MD itself scores over the same
    gap, which is the only meaningful reference for the generated one.

    This is the quantity the model is trained to produce, sampled across
    every replica of one (domain, temperature). Alignment matters: mdCATH
    trajectories drift by up to ~100 A of centre of mass, and without
    removing the rigid-body component the displacements would be dominated by
    translation the model is not asked to predict (and which the zero-COM
    formulation cannot express).
    """
    import h5py

    displacements = []
    with h5py.File(entry["path"], "r") as handle:
        temperature_group = handle[entry["domain"]][entry["temperature"]]
        replicas = sorted(temperature_group.keys())
        per_replica = max(max_pairs // max(len(replicas), 1), 1)
        for replica in replicas:
            coords = temperature_group[replica]["coords"]
            usable = coords.shape[0] - frame_gap
            if usable < 1:
                continue
            stride = max(usable // per_replica, 1)
            starts = np.arange(0, usable, stride)[:per_replica]
            for start in starts:
                pair = np.asarray(
                    coords[[int(start), int(start) + frame_gap]][:, particle_indices],
                    dtype=np.float32,
                )
                displacements.append(torch.from_numpy(pair))
    stacked = torch.stack(displacements)  # [P, 2, N, 3]
    source, target = stacked[:, 0], stacked[:, 1]
    mask = torch.ones(source.shape[0], source.shape[1], dtype=torch.bool)
    aligned_target = masked_kabsch_align(source, target, mask).aligned_target
    return aligned_target - source, source


def batch_from_sample(sample: dict, num_copies: int, device: torch.device) -> dict:
    """Replicate one dataset sample ``num_copies`` times, via the real collate.

    One forward pass over the replicated batch produces ``num_copies``
    independent structures, because the noise is drawn per batch element.
    Going through ``collate_protein_batch`` rather than expanding by hand is
    what keeps the derived tensors (``atom_mask``, ``bond_mask``,
    ``angle_mask``, ``residue_mask``) exactly as training built them.
    """
    batch = collate_protein_batch([sample] * num_copies)
    return {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}


# --- main --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mdcath_backbone_rotate_displacement.yaml")
    parser.add_argument("--checkpoint", default="checkpoints_mdcath_displacement/best.pt")
    parser.add_argument("--num-samples", type=int, default=100, help="structures generated per (domain, temperature)")
    parser.add_argument("--chunk-size", type=int, default=20, help="how many to generate per forward pass")
    parser.add_argument("--gt-frames", type=int, default=500, help="ground-truth displacement pairs pooled across replicas")
    parser.add_argument("--max-domains", type=int, default=5)
    parser.add_argument("--temperatures", default=None, help="comma-separated, e.g. 320,450 (default: all)")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--smoothing-rounds", type=int, default=None,
                        help="override flow.noise_smoothing_rounds; must match what the "
                             "checkpoint was trained with, and lets one config A/B the two")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.memory_fraction:
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)

    config = load_config(args.config)
    if args.smoothing_rounds is not None:
        config.flow.noise_smoothing_rounds = args.smoothing_rounds
    num_steps = args.num_steps or config.sampling.num_steps
    solver = args.solver or config.sampling.solver
    atom_level = is_atom_level(config.data.representation)

    manifest = ShardManifest.load(Path(config.data.rotation.manifest_path))
    data_dir = Path(config.data.mdcath_dir)
    files = [p for p in (data_dir / Path(e.path).name for e in manifest.val_domains) if p.exists()]
    if not files:
        print(f"error: no validation shards resident under {data_dir}", file=sys.stderr)
        return 1
    dataset = _build_dataset(config, data_dir, files, seed=config.data.seed + 1, is_validation=True)

    wanted = None
    if args.temperatures:
        wanted = {t.strip() for t in args.temperatures.split(",")}

    # One (domain, temperature) per evaluation unit, taking the first replica
    # as the source of x0. The ground truth pools all replicas.
    seen: set[tuple[str, str]] = set()
    units = []
    for index in range(len(dataset)):
        entry = dataset.trajectory_index[index // dataset.pairs_per_trajectory]
        key = (entry["domain"], entry["temperature"])
        if key in seen or (wanted is not None and entry["temperature"] not in wanted):
            continue
        seen.add(key)
        units.append((index, entry))
    domains = sorted({domain for domain, _ in seen})[: args.max_domains]
    units = [(i, e) for i, e in units if e["domain"] in domains]

    model = load_model_for_inference(args.checkpoint, config, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stochastic = getattr(model, "uses_flow_state", False)
    print(f"checkpoint     : {args.checkpoint} (step {checkpoint.get('step')})")
    print(f"flow           : {config.flow.path_type}"
          + (f", noise_scale={config.flow.noise_scale}"
             f", smoothing_rounds={config.flow.noise_smoothing_rounds}"
             if stochastic else " (deterministic)"))
    print(f"sampling       : {num_steps} steps, {solver}, {args.num_samples} draws per unit")
    print(f"units          : {len(units)} (domain, temperature) over {len(domains)} domain(s)\n")
    if not stochastic:
        print("warning: this checkpoint samples deterministically, so every draw is identical and\n"
              "         the diversity/JS/RMSF numbers below describe a single structure.\n")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rows = []
    by_temperature = defaultdict(list)

    for dataset_index, entry in units:
        sample = dataset[dataset_index]
        particle_indices = (
            dataset._domain_topology[entry["domain"]].particle_indices
            if atom_level
            else dataset._domain_meta[entry["domain"]][0]
        )
        ca_index = (
            sample["ca_atom_index"].numpy() if atom_level else np.arange(sample["source_coords"].shape[0])
        )

        truth_displacements, truth_sources = load_ground_truth_displacements(
            entry, particle_indices, config.data.mdcath_frame_gap, args.gt_frames
        )
        truth_displacements = truth_displacements.to(device)
        truth_sources = truth_sources.to(device)

        generated = []
        remaining = args.num_samples
        while remaining > 0:
            copies = min(args.chunk_size, remaining)
            batch = batch_from_sample(sample, copies, device)
            extra = {}
            if atom_level:
                for key in ("atom_mask", "atom_residue_index", "atom_element", "ca_atom_index"):
                    extra[key] = batch[key]
            if "esm_input_ids" in batch:
                extra["esm_input_ids"] = batch["esm_input_ids"]
                extra["esm_attention_mask"] = batch["esm_attention_mask"]
            with torch.no_grad():
                coords, _ = generate(
                    model, batch["source_coords"], batch["sequence_embedding"],
                    batch["residue_types"], batch["residue_mask"], batch["temperature"],
                    batch["physical_delta_t"], num_steps=num_steps, solver=solver,
                    generator=generator, **extra,
                )
            generated.append(coords)
            remaining -= copies
        generated = torch.cat(generated, dim=0)

        # Physics of the generated structures, at whatever resolution the model
        # flows -- a diverse ensemble of broken proteins is the failure mode
        # this guards against, and the metrics above cannot see it.
        physics_batch = min(args.chunk_size, generated.shape[0])
        physics_inputs = batch_from_sample(sample, physics_batch, device)
        if atom_level:
            physics = compute_all_atom_physics_losses(
                generated[:physics_batch], physics_inputs["source_coords"],
                physics_inputs["atom_mask"],
                physics_inputs["bond_index"], physics_inputs["bond_mask"],
                physics_inputs["angle_index"], physics_inputs["angle_mask"],
                config.model.graph,
                config.loss.physics.clash_threshold_for(config.data.representation),
            )
        else:
            physics = compute_physics_losses(
                generated[:physics_batch], physics_inputs["source_coords"],
                physics_inputs["residue_mask"], config.model.graph,
                config.loss.physics.clash_threshold, config.loss.physics.clash_seq_sep,
            )

        # The generated displacements, in the same frame as the ground-truth
        # ones: Kabsch-align each sample back onto the source before
        # differencing, so any rigid-body drift the sampler introduced does
        # not count as motion.
        source = batch["source_coords"][:1].expand_as(generated)
        particle_mask = torch.ones(
            generated.shape[0], generated.shape[1], dtype=torch.bool, device=device
        )
        aligned = masked_kabsch_align(source, generated, particle_mask).aligned_target
        generated_displacements = aligned - source

        profile_generated = displacement_profile(generated_displacements)
        profile_truth = displacement_profile(truth_displacements)
        moved_generated = displacement_rms(generated_displacements)
        moved_truth = displacement_rms(truth_displacements)
        spread = mean_pairwise_rmsd(generated[:, ca_index], 300, generator)
        js = jensen_shannon(
            generated_displacements.norm(dim=-1).reshape(-1).cpu(),
            truth_displacements.norm(dim=-1).reshape(-1).cpu(),
        )
        source_map = mean_contact_map(batch["source_coords"][:1, ca_index], 1, generator)
        jaccard = contact_jaccard(
            mean_contact_map(generated[:, ca_index], 30, generator), source_map
        )
        # What MD itself scores over the same gap. Without this the generated
        # number has no scale: 0.9 looks good and means the model did not move.
        truth_pairs = min(30, truth_sources.shape[0])
        truth_jaccard = sum(
            contact_jaccard(
                mean_contact_map((truth_sources[p] + truth_displacements[p])[ca_index].unsqueeze(0), 1, generator),
                mean_contact_map(truth_sources[p][ca_index].unsqueeze(0), 1, generator),
            )
            for p in range(truth_pairs)
        ) / max(truth_pairs, 1)

        row = {
            "domain": entry["domain"],
            "temperature": float(entry["temperature"]),
            "residues": len(ca_index),
            "moved_generated": moved_generated,
            "moved_truth": moved_truth,
            "moved_ratio": moved_generated / max(moved_truth, 1e-8),
            "spread": spread / max(moved_truth, 1e-8),
            "profile_pearson": pearson(profile_generated, profile_truth),
            "js": js,
            "contact_jaccard": jaccard,
            "contact_jaccard_truth": truth_jaccard,
            "bond": float(physics.bond),
            "angle": float(physics.angle),
            "clash": float(physics.clash),
        }
        rows.append(row)
        by_temperature[row["temperature"]].append(row)
        print(f"  {row['domain']:>10} {row['temperature']:>5.0f} K  "
              f"moved={row['moved_ratio']:.3f}  spread={row['spread']:.3f}  "
              f"profile_r={row['profile_pearson']:+.3f}  js={row['js']:.3f}  "
              f"cj={row['contact_jaccard']:.3f} (MD {row['contact_jaccard_truth']:.3f})")

    if not rows:
        print("error: no units evaluated", file=sys.stderr)
        return 1

    def mean(key: str, subset=None) -> float:
        subset = subset if subset is not None else rows
        return sum(r[key] for r in subset) / len(subset)

    columns = ("|d| gen", "|d| gt", "moved", "spread", "profile r", "JS", "contact J", "contact MD")
    print(f"\n{'':<10}" + "".join(f"{name:>11}" for name in columns))
    print("-" * (10 + 11 * len(columns)))
    for temperature in sorted(by_temperature):
        subset = by_temperature[temperature]
        print(f"{temperature:>7.0f} K "
              f"{mean('moved_generated', subset):>11.3f}{mean('moved_truth', subset):>11.3f}"
              f"{mean('moved_ratio', subset):>11.3f}{mean('spread', subset):>11.3f}"
              f"{mean('profile_pearson', subset):>+11.3f}{mean('js', subset):>11.3f}"
              f"{mean('contact_jaccard', subset):>11.3f}"
              f"{mean('contact_jaccard_truth', subset):>11.3f}")
    print("-" * (10 + 11 * len(columns)))
    print(f"{'all':<10}{mean('moved_generated'):>11.3f}{mean('moved_truth'):>11.3f}"
          f"{mean('moved_ratio'):>11.3f}{mean('spread'):>11.3f}"
          f"{mean('profile_pearson'):>+11.3f}{mean('js'):>11.3f}"
          f"{mean('contact_jaccard'):>11.3f}{mean('contact_jaccard_truth'):>11.3f}")
    print(f"\nphysical validity of generated structures (0 = matches the source topology)")
    print(f"  bond  {mean('bond'):.6f}   angle {mean('angle'):.6f}   clash {mean('clash'):.6f}")
    print("\ntargets: moved -> 1.0 (coordinate-space run scored 0.047), spread -> 1.41")
    print("         (0.0 = deterministic collapse), profile r > 0.5, JS -> 0, and")
    print("         contact J -> the contact MD column, from either side. Scoring above it")
    print("         means moving less than MD; the coordinate-space run's 0.888 was that.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
