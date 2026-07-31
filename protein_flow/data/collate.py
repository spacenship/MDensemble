"""Pads a list of variable-length protein-trajectory samples into a batch.

Two particle representations share one batch layout (see
``DataConfig.representation``):

* ``ca``: the flowing particles are residues, so the atom axis and the
  residue axis coincide (``N == L``). The atom-level bookkeeping fields are
  synthesized as identities, which lets every downstream module treat the
  C-alpha MVP as a special case of the all-atom path rather than a separate
  code path.
* ``heavy_atom``: the flowing particles are non-hydrogen atoms
  (``N ~= 7.9 * L``), each mapped back to its residue by
  ``atom_residue_index``, with real covalent ``bond_index``/``angle_index``
  taken from the shard's PSF.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


def _pad_index_pairs(samples: List[Dict[str, Any]], key: str, arity: int):
    """Pads per-sample ``[E_i, arity]`` topology index tensors into
    ``[B, E_max, arity]`` plus a ``[B, E_max]`` validity mask."""
    counts = [int(s[key].shape[0]) for s in samples]
    max_count = max(max(counts), 1)
    batch_size = len(samples)
    padded = torch.zeros(batch_size, max_count, arity, dtype=torch.long)
    mask = torch.zeros(batch_size, max_count, dtype=torch.bool)
    for i, sample in enumerate(samples):
        count = counts[i]
        if count:
            padded[i, :count] = sample[key].to(torch.long)
            mask[i, :count] = True
    return padded, mask


def collate_protein_batch(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate a list of per-sample dicts (see :mod:`protein_flow.data.dataset`)
    into a padded batch.

    Returns a dict with keys:
        sequence_embedding: [B, L_max, D_plm]
        source_coords:      [B, N_max, 3]   (N_max == L_max in "ca" mode)
        target_coords:      [B, N_max, 3]
        residue_types:      [B, L_max] long
        residue_mask:       [B, L_max] bool
        atom_mask:          [B, N_max] bool
        atom_residue_index: [B, N_max] long -- residue each atom belongs to
        atom_element:       [B, N_max] long -- element id (0 = padding)
        ca_atom_index:      [B, L_max] long -- atom index of each residue's CA
        bond_index:         [B, E_b, 2] long, with bond_mask [B, E_b]
        angle_index:        [B, E_a, 3] long, with angle_mask [B, E_a]
        temperature:        [B, 1]
        physical_delta_t:   [B, 1]
    """
    batch_size = len(samples)
    residue_lengths = torch.tensor([s["residue_types"].shape[0] for s in samples], dtype=torch.long)
    max_residues = int(residue_lengths.max().item())

    sequence_embedding = pad_sequence([s["sequence_embedding"] for s in samples], batch_first=True)
    source_coords = pad_sequence([s["source_coords"] for s in samples], batch_first=True)
    target_coords = pad_sequence([s["target_coords"] for s in samples], batch_first=True)
    residue_types = pad_sequence([s["residue_types"] for s in samples], batch_first=True, padding_value=0)

    residue_mask = torch.arange(max_residues).unsqueeze(0).expand(batch_size, max_residues) < residue_lengths.unsqueeze(1)

    atom_counts = torch.tensor([s["source_coords"].shape[0] for s in samples], dtype=torch.long)
    max_atoms = int(atom_counts.max().item())
    atom_mask = torch.arange(max_atoms).unsqueeze(0).expand(batch_size, max_atoms) < atom_counts.unsqueeze(1)

    is_all_atom = "atom_residue_index" in samples[0]
    if is_all_atom:
        atom_residue_index = pad_sequence(
            [s["atom_residue_index"] for s in samples], batch_first=True, padding_value=0
        )
        atom_element = pad_sequence([s["atom_element"] for s in samples], batch_first=True, padding_value=0)
        ca_atom_index = pad_sequence([s["ca_atom_index"] for s in samples], batch_first=True, padding_value=0)
        bond_index, bond_mask = _pad_index_pairs(samples, "bond_index", 2)
        angle_index, angle_mask = _pad_index_pairs(samples, "angle_index", 3)
    else:
        # "ca" mode: atoms and residues are the same particles, so the
        # mapping is the identity and covalent topology is exactly the
        # peptide chain (i, i+1) / (i-1, i, i+1).
        positions = torch.arange(max_atoms).unsqueeze(0).expand(batch_size, max_atoms)
        atom_residue_index = positions.clone()
        atom_element = atom_mask.long()  # single pseudo-element for valid particles
        ca_atom_index = torch.arange(max_residues).unsqueeze(0).expand(batch_size, max_residues).clone()
        bond_index, bond_mask = _identity_chain_bonds(residue_lengths, max_residues)
        angle_index, angle_mask = _identity_chain_angles(residue_lengths, max_residues)

    temperature = torch.stack([s["temperature"].reshape(1) for s in samples], dim=0)
    physical_delta_t = torch.stack([s["physical_delta_t"].reshape(1) for s in samples], dim=0)

    esm_fields = {}
    if "esm_input_ids" in samples[0]:
        # ESM pad id is 1 for the esm2_* tokenizers; the attention mask is
        # what actually suppresses those positions, so the exact fill value
        # only needs to be a valid vocabulary index.
        token_lengths = [int(s["esm_input_ids"].shape[0]) for s in samples]
        max_tokens = max(token_lengths)
        input_ids = torch.full((batch_size, max_tokens), 1, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.long)
        for i, sample in enumerate(samples):
            count = token_lengths[i]
            input_ids[i, :count] = sample["esm_input_ids"]
            attention_mask[i, :count] = 1
        esm_fields = {"esm_input_ids": input_ids, "esm_attention_mask": attention_mask}

    return {
        **esm_fields,
        "sequence_embedding": sequence_embedding,
        "source_coords": source_coords,
        "target_coords": target_coords,
        "residue_types": residue_types,
        "residue_mask": residue_mask,
        "atom_mask": atom_mask,
        "atom_residue_index": atom_residue_index,
        "atom_element": atom_element,
        "ca_atom_index": ca_atom_index,
        "bond_index": bond_index,
        "bond_mask": bond_mask,
        "angle_index": angle_index,
        "angle_mask": angle_mask,
        "temperature": temperature,
        "physical_delta_t": physical_delta_t,
    }


def _identity_chain_bonds(lengths: torch.Tensor, max_residues: int):
    """Consecutive-residue bonds (i, i+1) for the C-alpha representation."""
    batch_size = lengths.shape[0]
    max_bonds = max(max_residues - 1, 1)
    index = torch.zeros(batch_size, max_bonds, 2, dtype=torch.long)
    mask = torch.zeros(batch_size, max_bonds, dtype=torch.bool)
    positions = torch.arange(max_bonds, dtype=torch.long)
    for i in range(batch_size):
        count = max(int(lengths[i].item()) - 1, 0)
        if count:
            index[i, :count, 0] = positions[:count]
            index[i, :count, 1] = positions[:count] + 1
            mask[i, :count] = True
    return index, mask


def _identity_chain_angles(lengths: torch.Tensor, max_residues: int):
    """Consecutive-residue angles (i, i+1, i+2) for the C-alpha representation."""
    batch_size = lengths.shape[0]
    max_angles = max(max_residues - 2, 1)
    index = torch.zeros(batch_size, max_angles, 3, dtype=torch.long)
    mask = torch.zeros(batch_size, max_angles, dtype=torch.bool)
    positions = torch.arange(max_angles, dtype=torch.long)
    for i in range(batch_size):
        count = max(int(lengths[i].item()) - 2, 0)
        if count:
            index[i, :count, 0] = positions[:count]
            index[i, :count, 1] = positions[:count] + 1
            index[i, :count, 2] = positions[:count] + 2
            mask[i, :count] = True
    return index, mask
