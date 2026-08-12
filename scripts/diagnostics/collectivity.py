"""Is the generated motion collective, or is it noise of the right size?

The gate reports the right displacement magnitude (moved ~ 1.0) alongside a
contact Jaccard well below what MD scores over the same gap. Those two facts
together have one obvious explanation: real protein motion is *correlated* --
neighbouring atoms move together, so a 5 A displacement slides whole
substructures and preserves contacts -- while an uncorrelated displacement of
the same RMS scrambles them.

This measures that directly. For atom pairs binned by their separation in the
source structure, it reports

    C(r) = < d_i . d_j > / < |d_i| |d_j| >

for ground-truth MD displacements and, if a checkpoint is given, for
generated ones. Real MD should show C(r) near 1 at short range decaying
slowly; uncorrelated noise sits at 0 everywhere. The gap between the two
curves is the thing to fix, and its shape says whether the fix needs to be
local (a few angstroms) or global (a collective-mode module).
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
from protein_flow.train_rotating import _build_dataset

BIN_EDGES = torch.tensor([0.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 1e9])

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/mdcath_backbone_rotate_displacement.yaml")
parser.add_argument("--checkpoint", default=None, help="add a generated-motion curve")
parser.add_argument("--base-noise", action="store_true",
                    help="add a curve for the base distribution the config actually draws, "
                         "which is what flow.noise_smoothing_rounds is tuned against")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--batches", type=int, default=8)
parser.add_argument("--num-steps", type=int, default=20)
args = parser.parse_args()

device = torch.device(args.device)
torch.cuda.set_device(device)
config = load_config(args.config)
config.data.batch_size = 8
config.data.num_workers = 2

manifest = ShardManifest.load(Path(config.data.rotation.manifest_path))
data_dir = Path(config.data.mdcath_dir)
files = [p for p in (data_dir / Path(e.path).name for e in manifest.val_domains) if p.exists()]
dataset = _build_dataset(config, data_dir, files, seed=config.data.seed + 1, is_validation=True)
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2,
                    collate_fn=collate_protein_batch)

model = None
if args.checkpoint:
    from protein_flow.inference import generate, load_model_for_inference

    model = load_model_for_inference(args.checkpoint, config, device)


def correlation_by_distance(displacement: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor):
    """Accumulate sum(d_i . d_j) and sum(|d_i||d_j|) per distance bin."""
    dot_sums = torch.zeros(len(BIN_EDGES) - 1, device=displacement.device)
    norm_sums = torch.zeros(len(BIN_EDGES) - 1, device=displacement.device)
    for i in range(displacement.shape[0]):
        valid = mask[i]
        d = displacement[i][valid]
        x = coords[i][valid]
        separation = torch.cdist(x, x)
        dot = d @ d.T
        norms = d.norm(dim=-1)
        outer = norms.unsqueeze(0) * norms.unsqueeze(1)
        upper = torch.triu(torch.ones_like(separation, dtype=torch.bool), diagonal=1)
        bucket = torch.bucketize(separation[upper], BIN_EDGES.to(separation.device)) - 1
        dot_sums.index_add_(0, bucket, dot[upper])
        norm_sums.index_add_(0, bucket, outer[upper])
    return dot_sums, norm_sums


truth_dot = torch.zeros(len(BIN_EDGES) - 1, device=device)
truth_norm = torch.zeros(len(BIN_EDGES) - 1, device=device)
generated_dot = torch.zeros(len(BIN_EDGES) - 1, device=device)
generated_norm = torch.zeros(len(BIN_EDGES) - 1, device=device)
noise_dot = torch.zeros(len(BIN_EDGES) - 1, device=device)
noise_norm = torch.zeros(len(BIN_EDGES) - 1, device=device)

with torch.no_grad():
    for index, raw in enumerate(loader):
        if index >= args.batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
        mask = batch["atom_mask"]
        x0 = batch["source_coords"]
        x1 = masked_kabsch_align(x0, batch["target_coords"], mask).aligned_target
        d_dot, d_norm = correlation_by_distance(x1 - x0, x0, mask)
        truth_dot += d_dot
        truth_norm += d_norm

        if args.base_noise:
            from protein_flow.flow.paths import sample_zero_com_noise

            eps = sample_zero_com_noise(
                x0.shape, mask, config.flow.noise_scale, device, x0.dtype,
                torch.Generator().manual_seed(index),
                coords=x0, smoothing_rounds=config.flow.noise_smoothing_rounds,
                knn_k=config.model.graph.knn_k,
            )
            n_dot, n_norm = correlation_by_distance(eps, x0, mask)
            noise_dot += n_dot
            noise_norm += n_norm

        if model is not None:
            extra = {k: batch[k] for k in
                     ("atom_mask", "atom_residue_index", "atom_element", "ca_atom_index")}
            if "esm_input_ids" in batch:
                extra["esm_input_ids"] = batch["esm_input_ids"]
                extra["esm_attention_mask"] = batch["esm_attention_mask"]
            coords, _ = generate(
                model, x0, batch["sequence_embedding"], batch["residue_types"],
                batch["residue_mask"], batch["temperature"], batch["physical_delta_t"],
                num_steps=args.num_steps, solver="heun",
                generator=torch.Generator().manual_seed(index), **extra,
            )
            aligned = masked_kabsch_align(x0, coords, mask).aligned_target
            g_dot, g_norm = correlation_by_distance(aligned - x0, x0, mask)
            generated_dot += g_dot
            generated_norm += g_norm

labels = [f"{float(BIN_EDGES[i]):.0f}-{float(BIN_EDGES[i+1]):.0f}" for i in range(len(BIN_EDGES) - 2)]
labels.append(f">{float(BIN_EDGES[-2]):.0f}")

print("displacement direction correlation C(r) = <d_i.d_j> / <|d_i||d_j|>\n")
header = f"{'separation (A)':>16}{'MD':>10}"
if args.base_noise:
    header += f"{'base noise':>13}"
if model is not None:
    header += f"{'generated':>12}{'gap':>8}"
print(header)
print("-" * len(header))
noise_error = 0.0
for i, label in enumerate(labels):
    truth = float(truth_dot[i] / truth_norm[i].clamp(min=1e-12))
    line = f"{label:>16}{truth:>10.3f}"
    if args.base_noise:
        value = float(noise_dot[i] / noise_norm[i].clamp(min=1e-12))
        noise_error += (value - truth) ** 2
        line += f"{value:>13.3f}"
    if model is not None:
        gen = float(generated_dot[i] / generated_norm[i].clamp(min=1e-12))
        line += f"{gen:>12.3f}{gen - truth:>+8.3f}"
    print(line)
if args.base_noise:
    print(f"\nbase-noise RMSE vs MD: {(noise_error / len(labels)) ** 0.5:.3f} "
          f"(flow.noise_smoothing_rounds = {config.flow.noise_smoothing_rounds})")
print("\n1.0 = atoms move together; 0.0 = independent noise.")
