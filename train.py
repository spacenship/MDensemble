#!/usr/bin/env python3
"""CLI entry point: python train.py --config configs/default.yaml"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from protein_flow.config import load_config
from protein_flow.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the protein dual-graph flow-matching model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a YAML config file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = load_config(args.config)
    ckpt_dir = Path(config.train.ckpt_dir)
    train(config, config_save_path=ckpt_dir / "config.yaml")


if __name__ == "__main__":
    main()
