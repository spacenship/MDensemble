#!/usr/bin/env python3
"""CLI entry point:

    python sample.py --checkpoint checkpoints/best.pt --num-steps 20 --solver heun

Draws a demo batch from the synthetic dataset (using the same DataConfig the
checkpoint was trained with) as the initial ``source_coords``, then
integrates the learned flow ODE from tau=0 to tau=1. Real usage would
instead supply source_coords / sequence_embedding / residue_types /
residue_mask / temperature / physical_delta_t from an actual structure
following the tensor schema in protein_flow/data/dataset.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protein_flow.config import load_config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.synthetic import SyntheticProteinTrajectoryDataset
from protein_flow.inference import generate, load_model_for_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample structures from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Defaults to config.yaml next to the checkpoint.")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=str, default="samples/generated.pt")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else checkpoint_path.parent / "config.yaml"
    config = load_config(config_path)

    device = torch.device(config.train.device)
    model = load_model_for_inference(checkpoint_path, config, device)

    demo_dataset = SyntheticProteinTrajectoryDataset(
        config.data, size=args.batch_size, seed=config.data.seed + 999
    )
    batch = collate_protein_batch([demo_dataset[i] for i in range(args.batch_size)])
    batch = {key: value.to(device) for key, value in batch.items()}

    generated_coords, trajectory = generate(
        model,
        batch["source_coords"],
        batch["sequence_embedding"],
        batch["residue_types"],
        batch["residue_mask"],
        batch["temperature"],
        batch["physical_delta_t"],
        num_steps=args.num_steps,
        solver=args.solver,
        return_trajectory=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generated_coords": generated_coords,
            "trajectory": trajectory,
            "residue_mask": batch["residue_mask"],
        },
        output_path,
    )
    print(f"generated_coords shape: {tuple(generated_coords.shape)}")
    print(f"trajectory shape: {tuple(trajectory.shape)}")
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
