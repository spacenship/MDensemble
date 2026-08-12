#!/usr/bin/env python
"""Where along the path is the model still learning, and where has it stopped?

``fm`` is averaged over flow time, and the two ends of the path are not
remotely equally hard. At tau -> 1 the state is nearly the answer already
(``x_t = (1-t)eps + t*delta``), so predicting ``delta - eps`` is close to
trivial. At tau -> 0 the state is pure noise and the only predictable part of
the target is the noise-cancellation term; whatever remains is the thermal
fluctuation, which is not predictable from the source structure at all.

That matters because **sampling starts at tau=0**. A falling aggregate ``fm``
is compatible with two very different situations:

  healthy convergence -- every tau slice is still improving together
  saturation at the hard end -- tau~1 keeps improving and drags the average
                                down while tau~0, the slice that actually
                                governs generation, has stopped

This prints the breakdown, plus the irreducible floor for each slice. The
floor is the variance the target keeps once the observable part is removed:
at tau the state reveals a fraction of delta, so a model that knows
everything knowable still pays ``(1-tau)^2 * E|delta|^2``.

Usage:
    python scripts/diagnostics/fm_by_tau.py --checkpoint checkpoints_mdcath_displacement/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protein_flow.config import is_atom_level, load_config
from protein_flow.data.shard_manifest import ShardManifest
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.inference import load_model_for_inference
from protein_flow.losses.flow_matching import velocity_diagnostics
from protein_flow.train import _make_val_loader
from protein_flow.train_rotating import _build_dataset
from protein_flow.utils import masked_mean

TAU_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
ATOM_KEYS = ("atom_mask", "atom_residue_index", "atom_element", "ca_atom_index")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mdcath_backbone_rotate_displacement.yaml")
    parser.add_argument("--checkpoint", default="checkpoints_mdcath_displacement/best.pt")
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.memory_fraction:
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)

    config = load_config(args.config)
    config.data.batch_size = args.batch_size
    atom_level = is_atom_level(config.data.representation)
    if config.flow.path_type != "displacement":
        print(f"error: this diagnostic is for the displacement path, config has "
              f"{config.flow.path_type!r}", file=sys.stderr)
        return 1

    manifest = ShardManifest.load(Path(config.data.rotation.manifest_path))
    data_dir = Path(config.data.mdcath_dir)
    files = [p for p in (data_dir / Path(e.path).name for e in manifest.val_domains) if p.exists()]
    dataset = _build_dataset(config, data_dir, files, seed=config.data.seed + 1, is_validation=True)
    loader = _make_val_loader(dataset, config)

    model = load_model_for_inference(args.checkpoint, config, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {args.checkpoint} (step {checkpoint.get('step')})")
    print(f"batches   : {args.max_batches} x {args.batch_size}\n")

    from protein_flow.flow.paths import DisplacementPath

    path = DisplacementPath(
        noise_scale=config.flow.noise_scale,
        smoothing_rounds=config.flow.noise_smoothing_rounds,
        knn_k=config.model.graph.knn_k,
    )

    totals = {tau: [0.0, 0.0, 0.0, 0.0, 0, 0.0] for tau in TAU_GRID}  # fm, zero, floor, cos, n, ratio
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

            x0 = batch["source_coords"]
            x1 = masked_kabsch_align(x0, batch["target_coords"], particle_mask).aligned_target
            delta = (x1 - x0) * particle_mask.unsqueeze(-1).to(x0.dtype)
            delta_energy = float(masked_mean(delta.pow(2).sum(dim=-1), particle_mask))

            for tau_value in TAU_GRID:
                tau = torch.full((x0.shape[0],), tau_value, device=device, dtype=x0.dtype)
                state, target = path.sample(
                    x0, x1, tau, particle_mask, torch.Generator().manual_seed(index)
                )
                predicted = model(
                    x0, tau, batch["sequence_embedding"], batch["residue_types"],
                    batch["temperature"], batch["physical_delta_t"], residue_mask,
                    flow_state=state, **extra,
                )
                fm = float(masked_mean((predicted - target).pow(2).sum(dim=-1), particle_mask))
                zero_fm = float(masked_mean(target.pow(2).sum(dim=-1), particle_mask))
                ratio, cosine = velocity_diagnostics(predicted, target, particle_mask)
                slot = totals[tau_value]
                slot[0] += fm
                slot[1] += zero_fm
                # What an oracle still pays. With x_t = (1-t)eps + t*delta a
                # single linear combination of the two unknowns is observed, so
                # by Gaussian conditioning (writing a = E|delta|^2 = E|eps|^2,
                # which noise_scale was chosen to make true):
                #     Var(v | x_t) = a * [2 - (2t-1)^2 / ((1-t)^2 + t^2)]
                # It is *largest* at t=0.5, where x_t and v are uncorrelated and
                # nothing at all is learnable, and equals a at both endpoints --
                # the opposite of the naive "tau=1 reveals the answer" intuition.
                t = tau_value
                observable = (2 * t - 1) ** 2 / max((1 - t) ** 2 + t ** 2, 1e-9)
                slot[2] += delta_energy * (2.0 - observable)
                slot[3] += float(cosine)
                slot[5] += float(ratio)
                slot[4] += 1

    print(f"{'tau':>6}{'fm':>10}{'zero_fm':>10}{'improve':>10}{'floor':>10}"
          f"{'fm/floor':>10}{'v_cos':>9}{'|v|ratio':>10}")
    print("-" * 65)
    for tau_value in TAU_GRID:
        fm, zero_fm, floor, cosine, n, ratio = totals[tau_value]
        fm, zero_fm, floor, cosine, ratio = fm / n, zero_fm / n, floor / n, cosine / n, ratio / n
        improvement = 100.0 * (1 - fm / max(zero_fm, 1e-9))
        print(f"{tau_value:>6.2f}{fm:>10.3f}{zero_fm:>10.3f}{improvement:>9.1f}%"
              f"{floor:>10.3f}{fm / max(floor, 1e-9):>10.2f}{cosine:>9.3f}{ratio:>10.3f}")
    print("\nfm/floor -> 1.0 means that slice has extracted everything predictable.")
    print("The tau=0 row is the one that governs sampling; the tau=1 row is nearly free.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
