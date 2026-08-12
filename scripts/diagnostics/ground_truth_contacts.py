"""What contact Jaccard do real MD frame pairs score at the trained gap?

The gate compares a sampled structure's contact map against its source's and
I asserted the target should be > 0.9, on the reasoning that a 5-frame
displacement barely changes the fold. That reasoning is checkable and the
measured RMS|delta| of 4.6 A suggests it is wrong, so this measures the
ground truth directly: contact_jaccard(x1_aligned, x0) over real pairs.

Whatever number this prints is the ceiling the model should be judged
against -- scoring higher than the ground truth would mean the model moves
*less* than MD does, which is the previous formulation's failure, not a win.
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protein_flow.config import load_config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.shard_manifest import ShardManifest
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.train_rotating import _build_dataset
from torch.utils.data import DataLoader

CONTACT_CUTOFF = 8.0
CONTACT_SEPARATION = 3


def contact_map(frame: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(frame, frame)
    position = torch.arange(frame.shape[0])
    separation = (position.unsqueeze(0) - position.unsqueeze(1)).abs()
    return (distances < CONTACT_CUTOFF) & (separation >= CONTACT_SEPARATION)


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union > 0 else 1.0


def run():
    config = load_config("configs/mdcath_backbone_rotate.yaml")
    config.data.batch_size = 8
    config.data.num_workers = 2

    manifest = ShardManifest.load(Path(config.data.rotation.manifest_path))
    data_dir = Path(config.data.mdcath_dir)
    files = [p for p in (data_dir / Path(e.path).name for e in manifest.val_domains) if p.exists()]
    dataset = _build_dataset(config, data_dir, files, seed=config.data.seed + 1, is_validation=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=collate_protein_batch)

    by_temperature = defaultdict(lambda: [0.0, 0.0, 0])
    for index, batch in enumerate(loader):
        if index >= 40:
            break
        mask = batch["atom_mask"]
        x0 = batch["source_coords"]
        x1 = masked_kabsch_align(x0, batch["target_coords"], mask).aligned_target
        for i in range(x0.shape[0]):
            valid = batch["residue_mask"][i]
            ca = batch["ca_atom_index"][i].long()[valid]
            key = float(batch["temperature"][i].item())
            slot = by_temperature[key]
            slot[0] += jaccard(contact_map(x0[i, ca]), contact_map(x1[i, ca]))
            particle = mask[i]
            slot[1] += float(
                (((x1[i] - x0[i]).pow(2).sum(-1) * particle).sum() / particle.sum()).sqrt()
            )
            slot[2] += 1

    print(f"ground-truth pairs at frame_gap={config.data.mdcath_frame_gap}\n")
    print(f"{'temperature':>12}{'contact J':>12}{'RMS |delta|':>14}{'n':>6}")
    print("-" * 44)
    total_j, total_d, total_n = 0.0, 0.0, 0
    for temperature in sorted(by_temperature):
        j, d, n = by_temperature[temperature]
        print(f"{temperature:>9.0f} K {j/n:>12.4f}{d/n:>14.3f}{n:>6d}")
        total_j, total_d, total_n = total_j + j, total_d + d, total_n + n
    print("-" * 44)
    print(f"{'all':>12}{total_j/total_n:>12.4f}{total_d/total_n:>14.3f}{total_n:>6d}")
    print("\nThis is the ceiling: a model matching MD should land here, not above it.")


if __name__ == "__main__":
    run()
