#!/usr/bin/env python
"""How much of a displacement field is global rotation, and where does it come from?

``path_amplitude.py`` measures that Kabsch alignment removes ~17% of the
generated displacement *energy*. The obvious explanation is that the isotropic
base noise carries a random global rotation which the flow fails to cancel.
That explanation is checkable and this script checks it, because the expected
rotational content of an isotropic field is tiny: after the centre of mass is
removed there are ``3N - 3`` degrees of freedom and only 3 of them are
rotations, so a random field should lose about ``1/(N-1)`` of its energy to
alignment -- a few tenths of a percent at backbone resolution, not 17%.

Four fields are measured with the same estimator, so the numbers are
comparable:

  base noise      what the sampler actually starts from.
  MD raw          ``x1 - x0`` before alignment. Real MD tumbles, so this is
                  the one field that should show a large rotational share.
  MD aligned      the training target. Zero by construction -- it is the
                  control that proves the estimator reports 0 when there is
                  nothing to remove.
  generated       the model's output.

If base noise is ~0 and generated is large, the rotation is *manufactured by
the model*, not inherited from the base, and projecting the noise would fix
nothing.

Usage:
    python scripts/diagnostics/rigid_share.py \
        --checkpoint checkpoints_mdcath_displacement/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protein_flow.config import is_atom_level, load_config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.shard_manifest import ShardManifest
from protein_flow.flow.paths import sample_zero_com_noise
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.inference import generate, load_model_for_inference
from protein_flow.train_rotating import _build_dataset

ATOM_KEYS = ("atom_mask", "atom_residue_index", "atom_element", "ca_atom_index")


def energy(vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample mean squared length over the valid particles."""
    return (vectors.pow(2).sum(-1) * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def rigid_share(x0: torch.Tensor, displacement: torch.Tensor, mask: torch.Tensor):
    """Fraction of displacement energy that Kabsch alignment removes.

    The displacement is applied to ``x0``, the result is aligned back onto
    ``x0``, and what the alignment took away is by definition the rigid-body
    component -- translation plus the optimal global rotation.
    """
    raw = energy(displacement, mask)
    aligned = masked_kabsch_align(x0, x0 + displacement, mask).aligned_target - x0
    kept = energy(aligned, mask)
    return raw, kept, (1.0 - kept / raw.clamp(min=1e-12))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mdcath_backbone_rotate_displacement.yaml")
    parser.add_argument("--checkpoint", default="checkpoints_mdcath_displacement/best.pt")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=12)
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

    manifest = ShardManifest.load(Path(config.data.rotation.manifest_path))
    data_dir = Path(config.data.mdcath_dir)
    files = [p for p in (data_dir / Path(e.path).name for e in manifest.val_domains) if p.exists()]
    dataset = _build_dataset(config, data_dir, files, seed=config.data.seed + 1, is_validation=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                        collate_fn=collate_protein_batch)
    model = load_model_for_inference(args.checkpoint, config, device)

    labels = ("base noise", "MD raw", "MD aligned", "generated")
    totals = {label: [0.0, 0.0, 0] for label in labels}
    particle_counts = []

    with torch.no_grad():
        for index, raw_batch in enumerate(loader):
            if index >= args.max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw_batch.items()}
            mask = batch["atom_mask"] if atom_level else batch["residue_mask"]
            extra = {k: batch[k] for k in ATOM_KEYS} if atom_level else {}
            if "esm_input_ids" in batch:
                extra["esm_input_ids"] = batch["esm_input_ids"]
                extra["esm_attention_mask"] = batch["esm_attention_mask"]

            x0 = batch["source_coords"]
            particle_counts.append(float(mask.sum(-1).float().mean()))

            fields = {}
            fields["base noise"] = sample_zero_com_noise(
                x0.shape, mask, config.flow.noise_scale, x0.device, x0.dtype,
                torch.Generator().manual_seed(args.seed + index),
                coords=x0, smoothing_rounds=config.flow.noise_smoothing_rounds,
                knn_k=config.model.graph.knn_k,
                remove_rigid_motion=config.flow.remove_rigid_motion,
            )
            # Raw MD: the target frame as it sits, only recentred so that the
            # translation part does not swamp the rotation part.
            target = batch["target_coords"]
            centre_source = (x0 * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
            centre_target = (target * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
            recentred = target - (centre_target - centre_source).unsqueeze(1)
            fields["MD raw"] = (recentred - x0) * mask.unsqueeze(-1)
            aligned_x1 = masked_kabsch_align(x0, target, mask).aligned_target
            fields["MD aligned"] = (aligned_x1 - x0) * mask.unsqueeze(-1)

            generated, _ = generate(
                model, x0, batch["sequence_embedding"], batch["residue_types"],
                batch["residue_mask"], batch["temperature"], batch["physical_delta_t"],
                num_steps=args.num_steps, solver="heun",
                generator=torch.Generator().manual_seed(args.seed + index), **extra,
            )
            fields["generated"] = (generated - x0) * mask.unsqueeze(-1)

            for label, field in fields.items():
                raw, kept, share = rigid_share(x0, field, mask)
                slot = totals[label]
                slot[0] += float(raw.sum())
                slot[1] += float(kept.sum())
                slot[2] += raw.shape[0]

    particles = sum(particle_counts) / len(particle_counts)
    print(f"checkpoint : {args.checkpoint}")
    print(f"particles  : {particles:.0f} on average -> an isotropic field should lose")
    print(f"             3/(3N-3) = {3.0 / (3 * particles - 3) * 100:.2f}% of its energy to alignment\n")
    print(f"{'field':>14}{'RMS (A)':>10}{'RMS aligned':>13}{'rigid share':>13}")
    print("-" * 50)
    for label in labels:
        raw, kept, count = totals[label]
        raw, kept = (raw / count) ** 0.5, (kept / count) ** 0.5
        print(f"{label:>14}{raw:>10.3f}{kept:>13.3f}{100.0 * (1 - (kept / raw) ** 2):>12.1f}%")
    print("\nIf 'base noise' is near the isotropic expectation and 'generated' is far")
    print("above it, the rotation is produced by the model, not inherited from the base,")
    print("and projecting the base distribution would not remove it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
