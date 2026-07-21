#!/usr/bin/env python3
"""CLI entry point:

    python precompute_esm_embeddings.py \
        --mdcath-dir mdCATH_sample100/data \
        --output-dir esm_cache \
        --model facebook/esm2_t6_8M_UR50D

Computes one real ESM2 embedding tensor per mdCATH domain (the sequence is
fixed per domain -- it does not change across temperature/replica/frame)
and saves it as ``{output_dir}/{domain}.pt``. Point
``data.mdcath_embedding_cache_dir`` at ``output_dir`` in a config to make
``MdCathDataset`` load these instead of its random placeholder embedding.

This script is the only place in the repository that actually runs ESM;
it is never invoked by train.py, sample.py, or the test suite.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch

from protein_flow.data.esm_adapter import DEFAULT_MODEL_NAME, compute_esm_embeddings
from protein_flow.data.mdcath import parse_ca_indices_and_resnames
from protein_flow.data.residue_vocab import resname_to_one_letter


def domain_sequence(h5_path: Path) -> tuple[str, str]:
    with h5py.File(h5_path, "r") as f:
        domain = next(iter(f.keys()))
        pdb_bytes = f[domain]["pdbProteinAtoms"][()]
    _, residue_names = parse_ca_indices_and_resnames(pdb_bytes)
    sequence = "".join(resname_to_one_letter(r) for r in residue_names)
    return domain, sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute real ESM2 embeddings for mdCATH domains.")
    parser.add_argument("--mdcath-dir", type=str, required=True, help="Directory of mdCATH .h5 shards.")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write {domain}.pt files.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    h5_files = sorted(Path(args.mdcath_dir).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 shards found under {args.mdcath_dir}")

    domains, sequences = [], []
    for path in h5_files:
        domain, sequence = domain_sequence(path)
        domains.append(domain)
        sequences.append(sequence)
    print(f"Found {len(domains)} domains under {args.mdcath_dir}")

    embeddings = compute_esm_embeddings(
        sequences, model_name=args.model, device=args.device, batch_size=args.batch_size
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for domain, sequence, embedding in zip(domains, sequences, embeddings):
        assert embedding.shape[0] == len(sequence), (domain, embedding.shape, len(sequence))
        torch.save(embedding, output_dir / f"{domain}.pt")
        print(f"  {domain}: sequence length {len(sequence)}, embedding shape {tuple(embedding.shape)}")

    print(f"Saved {len(domains)} embeddings to {output_dir}")


if __name__ == "__main__":
    main()
