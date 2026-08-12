#!/usr/bin/env python3
"""Freezes the rotation plan for a full-dataset mdCATH run.

    # The real thing: all 5,398 domains, reusing shards already on disk.
    python scripts/build_mdcath_manifest.py \
        --output configs/manifests/mdcath_all.json \
        --prefer-local /home/mipstu/wjYang/MolecularDynamics/mdCATH_sample100/data

    # No network: build a plan over shards already downloaded (smoke tests).
    python scripts/build_mdcath_manifest.py \
        --output /tmp/local.json --from-local-dir mdCATH_sample100/data \
        --chunk-size 50 --num-val-domains 10

The manifest fixes the validation holdout and the chunk composition once, so
the run is reproducible and can resume mid-rotation. Point
``data.rotation.manifest_path`` at the output file.
"""
from __future__ import annotations

import argparse
import logging

from protein_flow.data.shard_manifest import (
    DEFAULT_REPO_ID,
    build_manifest,
    list_local_entries,
    list_repo_entries,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an mdCATH shard-rotation manifest.")
    parser.add_argument("--output", required=True, help="Where to write the manifest JSON.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--seed", type=int, default=0, help="Fixes the holdout and chunk composition.")
    parser.add_argument("--chunk-size", type=int, default=200, help="Domains per chunk.")
    parser.add_argument(
        "--chunk-max-gb", type=float, default=160.0,
        help="Byte ceiling per chunk; shard sizes vary 3.7x so a count-only split can overshoot.",
    )
    parser.add_argument("--num-val-domains", type=int, default=25, help="Domains held out permanently.")
    parser.add_argument(
        "--num-domains", type=int, default=None,
        help="Train on a subset of the dataset instead of all of it (holdout comes from within it).",
    )
    parser.add_argument(
        "--prefer-local", default=None,
        help="Directory of already-downloaded shards; those domains go into the earliest chunks.",
    )
    parser.add_argument(
        "--from-local-dir", default=None,
        help="Build from shards on disk instead of querying the Hub (no network).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.from_local_dir:
        entries = list_local_entries(args.from_local_dir)
        print(f"Found {len(entries)} local shard(s) under {args.from_local_dir}")
    else:
        entries = list_repo_entries(args.repo_id)
        total_tb = sum(entry.size for entry in entries) / 1e12
        print(f"Found {len(entries)} shard(s) in {args.repo_id} ({total_tb:.2f} TB)")

    manifest = build_manifest(
        entries,
        repo_id=args.repo_id,
        seed=args.seed,
        chunk_size=args.chunk_size,
        chunk_max_gb=args.chunk_max_gb,
        num_val_domains=args.num_val_domains,
        num_domains=args.num_domains,
        prefer_local_dir=args.prefer_local,
    )
    manifest.save(args.output)

    print(f"\n{manifest.summary()}")
    print(f"wrote {args.output}")
    print("\nper-chunk size (GB):")
    for index in range(manifest.num_chunks):
        print(f"  chunk {index:3d}: {len(manifest.chunks[index]):4d} domains, {manifest.chunk_bytes(index) / 1e9:6.1f} GB")


if __name__ == "__main__":
    main()
