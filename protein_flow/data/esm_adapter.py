"""Optional adapter for computing real ESM2 residue embeddings.

This module is never imported by the default training path
(``protein_flow/train.py``) and never required by the test suite -- ESM
must not run automatically inside training or tests (network + a model
download would be required on first use). Use the top-level
``precompute_esm_embeddings.py`` script to generate cached ``{domain}.pt``
embedding tensors once, offline from training; ``MdCathDataset`` picks
these up automatically via its ``embedding_cache_dir`` argument.
"""
from __future__ import annotations

from typing import List

import torch

DEFAULT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # hidden_size=320, matches DataConfig.plm_dim default


@torch.no_grad()
def compute_esm_embeddings(
    sequences: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    batch_size: int = 8,
) -> List[torch.Tensor]:
    """Computes per-residue ESM2 embeddings for a list of amino-acid sequences.

    Requires the optional ``transformers`` dependency, and a network
    connection the first time ``model_name`` is downloaded.

    Args:
        sequences: one-letter amino-acid sequences (e.g. from
            :func:`protein_flow.data.residue_vocab.resname_to_one_letter`).
        model_name: HuggingFace model id.
        device: torch device to run inference on.
        batch_size: number of sequences per forward pass.

    Returns:
        One ``[L_i, D]`` tensor per input sequence, with the tokenizer's
        special tokens and any batch padding already stripped, so
        ``embeddings[i].shape[0] == len(sequences[i])``.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings: List[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        encoded = tokenizer(batch_sequences, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded)
        hidden_states = output.last_hidden_state  # [batch, 1 + L + 1 (+ pad), D]

        for i, sequence in enumerate(batch_sequences):
            length = len(sequence)
            # The ESM tokenizer prepends <cls> (position 0), so residue j of
            # the original sequence sits at token position j + 1.
            per_residue = hidden_states[i, 1 : 1 + length]
            embeddings.append(per_residue.detach().cpu())

    return embeddings
