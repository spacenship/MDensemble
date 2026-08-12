#!/usr/bin/env python3
"""CLI entry point.

Single GPU:
    python train.py --config configs/default.yaml

Multiple GPUs (DistributedDataParallel, one process per GPU):
    torchrun --nproc_per_node=2 train.py --config configs/mdcath_full.yaml

The same script serves both: `protein_flow.distributed.setup_distributed`
detects the environment variables torchrun sets and initialises the process
group only when they are present.

Configs with `data.rotation.enabled` train on more shards than fit on disk
by downloading, training on and deleting one chunk at a time
(`protein_flow.train_rotating`); everything else runs the ordinary
single-pass loop. Same entry point either way.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from protein_flow.config import load_config
from protein_flow.distributed import get_rank, is_torchrun_launch
from protein_flow.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the protein dual-graph flow-matching model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a YAML config file.")
    args = parser.parse_args()

    # Tag every line with its rank under torchrun, and keep non-zero ranks
    # quiet so the log stays readable (they would otherwise duplicate
    # warnings verbatim).
    if is_torchrun_launch():
        rank = get_rank()
        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.WARNING,
            format=f"%(asctime)s | %(levelname)s | [rank {rank}] %(message)s",
        )
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = load_config(args.config)
    ckpt_dir = Path(config.train.ckpt_dir)
    if config.data.rotation.enabled:
        from protein_flow.train_rotating import train_rotating

        train_rotating(config, config_save_path=ckpt_dir / "config.yaml")
    else:
        train(config, config_save_path=ckpt_dir / "config.yaml")


if __name__ == "__main__":
    main()
