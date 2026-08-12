"""Is the over-dispersion an integration error or the velocity field itself?

The sampled ensemble spreads further than MD does (spread 2-3 against a
target of 1.41) and loses contacts. Two very different causes look the same
from the outside:

  integration error -- the field is fine but a 20-step Heun rollout accumulates
                       error, which is uncorrelated across atoms and so
                       destroys collectivity. More steps would fix it.
  field error       -- the learned velocity is genuinely wrong. More steps
                       converge to a wrong answer, so the curve is flat.

Sweeping num_steps separates them, and the answer decides whether the next
move is cheaper sampling or a different architecture.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protein_flow.config import load_config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.shard_manifest import ShardManifest
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.inference import generate, load_model_for_inference
from protein_flow.train_rotating import _build_dataset

STEPS = [1, 2, 5, 10, 20, 50, 100]

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/mdcath_backbone_rotate_displacement.yaml")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--smoothing-rounds", type=int, default=None)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--draws", type=int, default=8)
parser.add_argument("--memory-fraction", type=float, default=0.25)
args = parser.parse_args()

device = torch.device(args.device)
torch.cuda.set_device(device)
if args.memory_fraction:
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)

config = load_config(args.config)
config.data.batch_size = 8
config.data.num_workers = 2
if args.smoothing_rounds is not None:
    config.flow.noise_smoothing_rounds = args.smoothing_rounds

manifest = ShardManifest.load(Path(config.data.rotation.manifest_path))
data_dir = Path(config.data.mdcath_dir)
files = [p for p in (data_dir / Path(e.path).name for e in manifest.val_domains) if p.exists()][:4]
dataset = _build_dataset(config, data_dir, files, seed=0, is_validation=True)
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2,
                    collate_fn=collate_protein_batch)
model = load_model_for_inference(args.checkpoint, config, device)

CONTACT_CUTOFF, CONTACT_SEPARATION = 8.0, 3


def contact_map(frame):
    distances = torch.cdist(frame, frame)
    position = torch.arange(frame.shape[0], device=frame.device)
    separation = (position.unsqueeze(0) - position.unsqueeze(1)).abs()
    return (distances < CONTACT_CUTOFF) & (separation >= CONTACT_SEPARATION)


raw = next(iter(loader))
batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
mask = batch["atom_mask"]
x0 = batch["source_coords"]
x1 = masked_kabsch_align(x0, batch["target_coords"], mask).aligned_target
moved_truth = float((((x1 - x0).pow(2).sum(-1) * mask).sum() / mask.sum()).sqrt())

# One source, repeated, so spread is measured over independent noise draws.
single = {k: (v[:1].expand(args.draws, *v.shape[1:]).contiguous() if torch.is_tensor(v) else v)
          for k, v in batch.items()}
extra = {k: single[k] for k in ("atom_mask", "atom_residue_index", "atom_element", "ca_atom_index")}
if "esm_input_ids" in single:
    extra["esm_input_ids"] = single["esm_input_ids"]
    extra["esm_attention_mask"] = single["esm_attention_mask"]
source = single["source_coords"]
draw_mask = single["atom_mask"]
valid = single["residue_mask"][0]
ca = single["ca_atom_index"][0].long()[valid]
source_contacts = contact_map(source[0, ca])

print(f"checkpoint      : {args.checkpoint}")
print(f"smoothing_rounds: {config.flow.noise_smoothing_rounds}")
print(f"GT RMS |delta|  : {moved_truth:.3f} A\n")
print(f"{'ODE steps':>10}{'moved':>9}{'spread':>9}{'contactJ':>10}")
print("-" * 38)

with torch.no_grad():
    for num_steps in STEPS:
        draws, _ = generate(
            model, source, single["sequence_embedding"], single["residue_types"],
            single["residue_mask"], single["temperature"], single["physical_delta_t"],
            num_steps=num_steps, solver="heun",
            generator=torch.Generator().manual_seed(0), **extra,
        )
        aligned = masked_kabsch_align(source, draws, draw_mask).aligned_target
        moved = float((((aligned - source).pow(2).sum(-1) * draw_mask).sum() / draw_mask.sum()).sqrt())
        pairs = [
            float((((aligned[i] - aligned[j]).pow(2).sum(-1) * draw_mask[i]).sum()
                   / draw_mask[i].sum()).sqrt())
            for i in range(args.draws) for j in range(i + 1, args.draws)
        ]
        spread = sum(pairs) / len(pairs)
        generated_contacts = contact_map(aligned[0, ca])
        union = (generated_contacts | source_contacts).sum()
        jaccard = float((generated_contacts & source_contacts).sum() / union) if union > 0 else 1.0
        print(f"{num_steps:>10}{moved/moved_truth:>9.3f}{spread/moved_truth:>9.3f}{jaccard:>10.3f}")

print("\nfalling spread with more steps -> integration error; flat -> the field itself.")
