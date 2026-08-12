#!/usr/bin/env python
"""Where along the path does the amplitude go missing?

``evaluate_ensemble.py`` reports ``moved`` -- RMS|delta| generated over RMS|delta|
from MD -- as a single number at the end of the rollout. When it lands below 1
the natural reading is "the flow does not travel far enough from the noise to
the structure". That reading is testable, because the *correct* amplitude at
every intermediate tau is known in closed form.

The training path is ``x_t = (1-t)eps + t*delta`` with ``eps`` independent of
``delta``, so writing ``a = E|eps|^2`` and ``b = E|delta|^2`` the marginal that
a perfect sampler must reproduce at every step is

    RMS|x_t| = sqrt((1-t)^2 * a + t^2 * b)

This is **not monotonic**. With a ~ b it contracts to about ``sqrt(b/2)`` near
t=0.5 and expands again, so the sampler has to shrink the noise before it can
grow the displacement. A model that simply "does not go far enough" undershoots
only near t=1; a model that over-contracts in the middle and never recovers
undershoots from t~0.5 onward. Those are different defects with different
fixes, and the end-point number cannot tell them apart.

The ratio column is the diagnostic: ``RMS|state| / RMS|x_t|``. Where it first
departs from 1.0 is where the amplitude is actually lost.

Usage:
    python scripts/diagnostics/path_amplitude.py \
        --checkpoint checkpoints_mdcath_displacement/best.pt
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
from protein_flow.inference import generate, load_model_for_inference
from protein_flow.train_rotating import _build_dataset

ATOM_KEYS = ("atom_mask", "atom_residue_index", "atom_element", "ca_atom_index")


def masked_rms(vectors: torch.Tensor, mask: torch.Tensor) -> float:
    """RMS length of a per-particle vector field over the valid particles."""
    return float(((vectors.pow(2).sum(-1) * mask).sum() / mask.sum().clamp(min=1)).sqrt())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mdcath_backbone_rotate_displacement.yaml")
    parser.add_argument("--checkpoint", default="checkpoints_mdcath_displacement/best.pt")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--solver", default="heun")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.memory_fraction:
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)

    config = load_config(args.config)
    config.data.batch_size = args.batch_size
    config.data.num_workers = 2
    atom_level = is_atom_level(config.data.representation)
    if config.flow.path_type != "displacement":
        print(f"error: needs the displacement path, config has {config.flow.path_type!r}",
              file=sys.stderr)
        return 1

    manifest = ShardManifest.load(Path(config.data.rotation.manifest_path))
    data_dir = Path(config.data.mdcath_dir)
    files = [p for p in (data_dir / Path(e.path).name for e in manifest.val_domains) if p.exists()]
    dataset = _build_dataset(config, data_dir, files, seed=config.data.seed + 1, is_validation=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                        collate_fn=collate_protein_batch)

    model = load_model_for_inference(args.checkpoint, config, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # a = E|eps|^2 is set by the base distribution, not measured: noise_scale is
    # the per-axis standard deviation, and the three axes are independent.
    noise_energy = 3.0 * config.flow.noise_scale ** 2

    tau_grid = [step / args.num_steps for step in range(args.num_steps + 1)]
    # Accumulated as energies (squared), because RMS of a pooled set is the root
    # of the pooled mean square, not the mean of the roots.
    state_energy = [0.0] * len(tau_grid)
    target_energy = 0.0
    batches = 0
    # generated (raw), ground truth, count, generated (Kabsch-aligned)
    by_temperature = defaultdict(lambda: [0.0, 0.0, 0, 0.0])

    with torch.no_grad():
        for index, raw in enumerate(loader):
            if index >= args.max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
            mask = batch["atom_mask"] if atom_level else batch["residue_mask"]
            extra = {k: batch[k] for k in ATOM_KEYS} if atom_level else {}
            if "esm_input_ids" in batch:
                extra["esm_input_ids"] = batch["esm_input_ids"]
                extra["esm_attention_mask"] = batch["esm_attention_mask"]

            x0 = batch["source_coords"]
            x1 = masked_kabsch_align(x0, batch["target_coords"], mask).aligned_target
            target_energy += masked_rms(x1 - x0, mask) ** 2

            _, trajectory = generate(
                model, x0, batch["sequence_embedding"], batch["residue_types"],
                batch["residue_mask"], batch["temperature"], batch["physical_delta_t"],
                num_steps=args.num_steps, solver=args.solver, return_trajectory=True,
                generator=torch.Generator().manual_seed(args.seed + index), **extra,
            )
            # The trajectory is returned as coordinates; the flow state is the
            # displacement from the (fixed) source structure.
            for position in range(trajectory.shape[0]):
                state_energy[position] += masked_rms(trajectory[position] - x0, mask) ** 2

            final_state = trajectory[-1] - x0
            # The evaluation harness Kabsch-aligns the generated structure to
            # the source before measuring, and so does the ground truth by
            # construction. Anything the alignment removes is rigid-body
            # motion: a global rotation about the centroid, which delta never
            # contains but the isotropic base noise does. Measuring both tells
            # us how much of the generated amplitude is spent on a mode the
            # target has no component along.
            aligned_state = masked_kabsch_align(x0, trajectory[-1], mask).aligned_target - x0
            for row in range(x0.shape[0]):
                single = mask[row:row + 1]
                slot = by_temperature[float(batch["temperature"][row].flatten()[0])]
                slot[0] += masked_rms(final_state[row:row + 1], single) ** 2
                slot[1] += masked_rms((x1 - x0)[row:row + 1], single) ** 2
                slot[2] += 1
                slot[3] += masked_rms(aligned_state[row:row + 1], single) ** 2
            batches += 1

    if batches == 0:
        print("error: no batches", file=sys.stderr)
        return 1

    delta_energy = target_energy / batches

    print(f"checkpoint : {args.checkpoint} (step {checkpoint.get('step')})")
    print(f"sampling   : {args.num_steps} steps, {args.solver}, {batches} x {args.batch_size}")
    print(f"noise_scale: {config.flow.noise_scale} per axis "
          f"-> RMS|eps| {noise_energy ** 0.5:.3f} A")
    print(f"MD         : RMS|delta| {delta_energy ** 0.5:.3f} A\n")

    print(f"{'tau':>6}{'RMS|state|':>12}{'analytic':>11}{'ratio':>8}")
    print("-" * 37)
    stride = max(1, args.num_steps // 10)
    for position, tau in enumerate(tau_grid):
        if position % stride and position != len(tau_grid) - 1:
            continue
        measured = (state_energy[position] / batches) ** 0.5
        analytic = ((1 - tau) ** 2 * noise_energy + tau ** 2 * delta_energy) ** 0.5
        print(f"{tau:>6.2f}{measured:>12.3f}{analytic:>11.3f}{measured / analytic:>8.3f}")

    print(f"\n{'temperature':>12}{'gen raw':>10}{'gen algn':>10}{'MD':>10}"
          f"{'moved':>9}{'rigid %':>9}{'|eps|/MD':>10}")
    print("-" * 70)
    pooled = [0.0, 0.0, 0.0, 0]
    for temperature in sorted(by_temperature):
        raw, truth, count, algn = by_temperature[temperature]
        pooled[0] += raw
        pooled[1] += truth
        pooled[2] += algn
        pooled[3] += count
        raw, truth, algn = (raw / count) ** 0.5, (truth / count) ** 0.5, (algn / count) ** 0.5
        rigid = 100.0 * (1.0 - (algn / raw) ** 2)
        print(f"{temperature:>9.0f} K {raw:>10.3f}{algn:>10.3f}{truth:>10.3f}"
              f"{algn / truth:>9.3f}{rigid:>8.1f}%{noise_energy ** 0.5 / truth:>10.3f}")
    raw, truth, algn = ((pooled[0] / pooled[3]) ** 0.5, (pooled[1] / pooled[3]) ** 0.5,
                        (pooled[2] / pooled[3]) ** 0.5)
    print("-" * 70)
    print(f"{'all':>12}{raw:>10.3f}{algn:>10.3f}{truth:>10.3f}{algn / truth:>9.3f}"
          f"{100.0 * (1.0 - (algn / raw) ** 2):>8.1f}%{noise_energy ** 0.5 / truth:>10.3f}")

    print("\nratio ~ 1.0 throughout and < 1 only at the end -> the flow stops short.")
    print("ratio dipping below 1 mid-path and never recovering -> it over-contracts.")
    print("rigid % is the share of generated displacement *energy* that Kabsch removes,")
    print("i.e. global rotation. The target has none of it by construction, so it is")
    print("amplitude spent on a mode that cannot score. The base noise is isotropic and")
    print("does carry it, so this is noise the flow failed to cancel.")
    print("The |eps|/MD column is how far the fixed base sits from each temperature's")
    print("target scale: the flow must supply that factor itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
