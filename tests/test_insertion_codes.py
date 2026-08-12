"""Residues that share a number but differ by PDB insertion code.

mdCATH's per-atom ``resid`` array stores only the residue *number*, so a
chain numbered 120, 120A, 120B, 120C, 120D appears in it as five identical
values. Grouping atoms by that array builds one 20-atom "residue" and breaks
the 4-atoms-per-residue invariant that the backbone representation, the
physics losses and the atom->residue broadcast all depend on.

Eight of mdCATH's 5,398 domains hit this (1a6tB01, 1bwvS00, 1hpgA02,
1iakA01, 1ruyH02, 1stfI00, 3h33A00, 3vhoA00) -- a small share, but a biased
one, since insertion codes come from antibody/immune numbering schemes.
"""
from __future__ import annotations

import numpy as np
import pytest

from protein_flow.data.mdcath import parse_residue_labels
from protein_flow.data.topology import parse_backbone_topology

# Three residues' worth of backbone atoms. The first two share residue
# number 10 and are separated only by an insertion code.
LABELS = np.asarray(
    ["A:10"] * 4 + ["A:10A"] * 4 + ["A:11"] * 4, dtype=object
)
RAW_RESID = np.asarray([10] * 8 + [11] * 4)  # what mdCATH actually stores
ELEMENTS = np.asarray(["N", "C", "C", "O"] * 3)
RESNAMES = np.asarray(["LEU"] * 4 + ["TYR"] * 4 + ["ASN"] * 4)


def _psf(num_atoms: int) -> bytes:
    """A minimal CHARMM PSF: !NATOM lines plus an empty bond/angle section."""
    lines = ["PSF", "", f"{num_atoms:8d} !NATOM"]
    for index in range(num_atoms):
        name = ("N", "CA", "C", "O")[index % 4]
        lines.append(f"{index + 1:8d} PROA {index // 4 + 10:<4d} ALA  {name:<5s} NH3 0.0 14.0 0")
    lines += ["", "       0 !NBOND: bonds", "", "       0 !NTHETA: angles"]
    return "\n".join(lines).encode()


def test_insertion_coded_residues_are_kept_separate():
    topology = parse_backbone_topology(_psf(12), ELEMENTS, LABELS, RESNAMES, 12)
    assert topology.num_residues == 3
    assert topology.num_particles == 12
    counts = np.bincount(topology.atom_residue_index)
    assert counts.tolist() == [4, 4, 4]


def test_raw_resid_would_have_merged_them():
    """Documents the defect: the array mdCATH ships is not enough on its own."""
    topology = parse_backbone_topology(_psf(12), ELEMENTS, RAW_RESID, RESNAMES, 12)
    assert topology.num_residues == 2  # 10 and 10A collapsed
    assert np.bincount(topology.atom_residue_index).tolist() == [8, 4]


def test_a_label_recurring_later_starts_a_new_residue():
    """Runs, not distinct values: a repeated label must not merge across a gap."""
    labels = np.asarray(["A:1"] * 4 + ["A:2"] * 4 + ["A:1"] * 4, dtype=object)
    topology = parse_backbone_topology(_psf(12), ELEMENTS, labels, RESNAMES, 12)
    assert topology.num_residues == 3


def test_parse_residue_labels_reads_insertion_codes_and_chain():
    pdb = (
        "ATOM      1  N   LEU 0 120      0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      2  CA  LEU 0 120      1.000   0.000   0.000  1.00  0.00\n"
        "ATOM      3  N   TYR 0 120A     2.000   0.000   0.000  1.00  0.00\n"
        "ATOM      4  CA  TYR 0 120A     3.000   0.000   0.000  1.00  0.00\n"
    ).encode()
    labels = parse_residue_labels(pdb)
    assert labels.tolist() == ["0:120", "0:120", "0:120A", "0:120A"]


def test_parse_residue_labels_aligns_one_entry_per_atom_record():
    pdb = (
        "REMARK ignored\n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000\n"
        "TER\n"
        "HETATM    2  CA  CAL A 900      0.000   0.000   0.000\n"
        "END\n"
    ).encode()
    assert parse_residue_labels(pdb).shape[0] == 2
