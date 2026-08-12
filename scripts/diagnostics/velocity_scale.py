#!/usr/bin/env python
"""Is the under-expansion a scale deficit or a direction deficit?

The sampler reaches only ~0.84 of the amplitude the marginal path calls for,
and the shortfall accumulates through the second half of the trajectory rather
than appearing at the end. Two causes look identical in the ``moved`` number:

  scale     the velocity field points the right way and is uniformly too
            short. Multiplying it by a constant recovers the amplitude and
            leaves the structure intact, so contact Jaccard holds.
  direction the field is wrong in a way that averaging has smoothed. Scaling
            it up amplifies the error too: amplitude rises but the protein
            comes apart, so contact Jaccard falls off a cliff.

This sweeps a constant gain on the predicted velocity at inference and reports
both, which decides whether a cheap correction exists or whether the field
itself has to get better.

The coordinate-space run's own scale sweep is the cautionary precedent: no
gain beat the do-nothing baseline there, because the problem was the
formulation rather than the magnitude. Read contact Jaccard before ``moved``.

Usage:
    python scripts/diagnostics/velocity_scale.py \
        --checkpoint checkpoints_mdcath_displacement/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protein_flow.config import is_atom_level, load_config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.shard_manifest import ShardManifest
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.inference import generate, load_model_for_inference
from protein_flow.train_rotating import _build_dataset

ATOM_KEYS = ("atom_mask", "atom_residue_index", "atom_element", "ca_atom_index")
GAINS = (1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0)
CONTACT_CUTOFF, CONTACT_SEPARATION = 8.0, 3


class ScaledField(nn.Module):
    """Wraps the model so every velocity evaluation is multiplied by ``gain``.

    Wrapping the field rather than the final displacement matters: the ODE is
    non-linear in its own state, so scaling the output once is not the same
    experiment as scaling the field the integrator follows.
    """

    def __init__(self, model: nn.Module, gain: float):
        super().__init__()
        self.model = model
        self.gain = gain
        # DualGraphFlowModel.sample() reads these off ``self``, so they have to
        # survive the wrapping or the sampler takes the wrong branch and starts
        # from the source structure instead of from noise.
        for attribute in ("uses_flow_state", "noise_scale", "noise_smoothing_rounds", "knn_k"):
            object.__setattr__(self, attribute, getattr(model, attribute))

    def forward(self, *args, **kwargs):
        return self.gain * self.model(*args, **kwargs)

    def sample(self, *args, **kwargs):
        from protein_flow.models.dual_graph_flow import DualGraphFlowModel

        return DualGraphFlowModel.sample(self, *args, **kwargs)


def contact_map(frame: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(frame, frame)
    position = torch.arange(frame.shape[0], device=frame.device)
    separation = (position.unsqueeze(0) - position.unsqueeze(1)).abs()
    return (distances < CONTACT_CUTOFF) & (separation >= CONTACT_SEPARATION)


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union > 0 else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mdcath_backbone_rotate_displacement.yaml")
    parser.add_argument("--checkpoint", default="checkpoints_mdcath_displacement/best.pt")
    parser.add_argument("--num-steps", type=int, default=50)
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

    manifest = ShardManifest.load(Path(config.data.rotation.manifest_path))
    data_dir = Path(config.data.mdcath_dir)
    files = [p for p in (data_dir / Path(e.path).name for e in manifest.val_domains) if p.exists()]
    dataset = _build_dataset(config, data_dir, files, seed=config.data.seed + 1, is_validation=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                        collate_fn=collate_protein_batch)
    model = load_model_for_inference(args.checkpoint, config, device)

    batches = [next(iter(loader))] if args.max_batches == 1 else []
    if not batches:
        for index, raw in enumerate(loader):
            if index >= args.max_batches:
                break
            batches.append(raw)

    print(f"checkpoint : {args.checkpoint}")
    print(f"sampling   : {args.num_steps} steps, heun, {len(batches)} x {args.batch_size}\n")
    print(f"{'gain':>6}{'moved':>9}{'rigid':>9}{'contactJ':>10}{'contact MD':>12}{'bond':>9}")
    print("-" * 55)


    with torch.no_grad():
        md_contact_total, md_count = 0.0, 0
        for gain in GAINS:
            field = ScaledField(model, gain)
            moved_energy, truth_energy, raw_energy = 0.0, 0.0, 0.0
            contact_total, bond_total, count = 0.0, 0.0, 0
            for index, raw_batch in enumerate(batches):
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in raw_batch.items()}
                mask = batch["atom_mask"] if atom_level else batch["residue_mask"]
                extra = {k: batch[k] for k in ATOM_KEYS} if atom_level else {}
                if "esm_input_ids" in batch:
                    extra["esm_input_ids"] = batch["esm_input_ids"]
                    extra["esm_attention_mask"] = batch["esm_attention_mask"]

                x0 = batch["source_coords"]
                x1 = masked_kabsch_align(x0, batch["target_coords"], mask).aligned_target
                generated, _ = generate(
                    field, x0, batch["sequence_embedding"], batch["residue_types"],
                    batch["residue_mask"], batch["temperature"], batch["physical_delta_t"],
                    num_steps=args.num_steps, solver="heun",
                    generator=torch.Generator().manual_seed(args.seed + index), **extra,
                )
                aligned = masked_kabsch_align(x0, generated, mask).aligned_target

                def mean_energy(vectors):
                    return float((vectors.pow(2).sum(-1) * mask).sum() / mask.sum())

                raw_energy += mean_energy(generated - x0)
                moved_energy += mean_energy(aligned - x0)
                truth_energy += mean_energy(x1 - x0)

                # Consecutive-backbone-atom bond stretch, computed inline rather
                # than through the loss module: contact Jaccard is the validity
                # guard here and this is only a coarse second opinion on it.
                separations = (aligned[:, 1:] - aligned[:, :-1]).norm(dim=-1)
                pair_mask = mask[:, 1:] & mask[:, :-1]
                reference = (x0[:, 1:] - x0[:, :-1]).norm(dim=-1)
                bond_total += float(
                    (((separations - reference).abs() * pair_mask).sum() / pair_mask.sum())
                )

                for row in range(x0.shape[0]):
                    valid = batch["residue_mask"][row]
                    ca = batch["ca_atom_index"][row].long()[valid]
                    source_contacts = contact_map(x0[row, ca])
                    contact_total += jaccard(contact_map(aligned[row, ca]), source_contacts)
                    if gain == GAINS[0]:
                        md_contact_total += jaccard(contact_map(x1[row, ca]), source_contacts)
                        md_count += 1
                    count += 1

            moved = (moved_energy / truth_energy) ** 0.5
            rigid = 1.0 - moved_energy / raw_energy
            print(f"{gain:>6.2f}{moved:>9.3f}{100 * rigid:>8.1f}%{contact_total / count:>10.3f}"
                  f"{md_contact_total / max(md_count, 1):>12.3f}{bond_total / len(batches):>9.4f}")

    print("\nmoved rising with contact Jaccard holding -> a scale deficit, correctable.")
    print("moved rising while contact Jaccard falls -> a direction deficit; the gain is")
    print("amplifying error and the field itself has to improve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
