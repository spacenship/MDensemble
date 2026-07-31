"""Covalent topology extraction from mdCATH's embedded CHARMM PSF text.

Verified against local sample shards (see ``tests/test_topology.py``):
  - ``psf`` is a full-system CHARMM PSF (protein + solvent), whose ``!NATOM``
    section lists the protein atoms **first**, in exactly the same order as
    the file's per-atom ``resname``/``resid`` arrays and its ``coords``
    dataset. Solvent (``TIP3`` water, ions) follows and is absent from
    ``coords``, so any bond/angle touching an index >= ``numProteinAtoms``
    is dropped.
  - ``!NBOND``/``!NTHETA`` give real force-field covalent bonds and angles
    (1-based indices), not distance heuristics. On the inspected domain the
    resulting heavy-atom bond lengths span 1.17-1.66 A (mean 1.43) and bond
    angles 100-136 deg (mean 116), i.e. physically correct.

Using this real topology matters for the all-atom physics losses: unlike
the C-alpha MVP (where "bonded" simply meant "consecutive in sequence"),
an all-atom structure has branched side chains whose connectivity cannot
be inferred from atom ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class AtomTopology:
    """Heavy-atom topology of one domain, re-indexed to heavy atoms only.

    All index arrays address the *heavy-atom* ordering (0 .. num_heavy-1),
    i.e. after hydrogens have been dropped, so they can be used directly
    against the coordinate tensors produced by
    :class:`protein_flow.data.mdcath.MdCathDataset` in ``heavy_atom`` mode.

    Attributes:
        heavy_indices: [num_heavy] indices into the original all-atom arrays.
        bond_index: [num_bonds, 2] covalent bonds between heavy atoms.
        angle_index: [num_angles, 3] covalent angles (i, centre, k).
        atom_residue_index: [num_heavy] residue index (0-based, contiguous).
        atom_element: [num_heavy] element type id (see ELEMENT_TO_INDEX).
        ca_atom_index: [num_residues] heavy-atom index of each residue's CA.
        residue_names: 3-letter code per residue.
    """

    heavy_indices: np.ndarray
    bond_index: np.ndarray
    angle_index: np.ndarray
    atom_residue_index: np.ndarray
    atom_element: np.ndarray
    ca_atom_index: np.ndarray
    residue_names: List[str]

    @property
    def num_heavy(self) -> int:
        return int(self.heavy_indices.shape[0])

    @property
    def num_residues(self) -> int:
        return int(self.ca_atom_index.shape[0])


# Element vocabulary for protein heavy atoms. Index 0 is reserved for
# padding so an embedding table can use padding_idx=0.
ELEMENT_TO_INDEX = {"C": 1, "N": 2, "O": 3, "S": 4}
NUM_ELEMENT_TYPES = 6  # 0=pad, 1..4 known, 5=other


def element_to_index(element: str) -> int:
    return ELEMENT_TO_INDEX.get(element.strip().upper(), 5)


def _parse_psf_section(psf_lines: List[str], tag: str, arity: int) -> np.ndarray:
    """Parses a PSF index section (e.g. ``!NBOND``) into a [n, arity] 0-based array."""
    header = next((i for i, line in enumerate(psf_lines) if tag in line), None)
    if header is None:
        return np.zeros((0, arity), dtype=np.int64)
    count = int(psf_lines[header].split()[0])
    if count == 0:
        return np.zeros((0, arity), dtype=np.int64)

    values: List[int] = []
    cursor = header + 1
    needed = count * arity
    while len(values) < needed and cursor < len(psf_lines):
        values.extend(int(token) for token in psf_lines[cursor].split())
        cursor += 1
    if len(values) < needed:
        raise ValueError(f"PSF section {tag} truncated: got {len(values)} of {needed} indices")
    return np.asarray(values[:needed], dtype=np.int64).reshape(count, arity) - 1  # PSF is 1-based


def parse_heavy_atom_topology(
    psf_text_bytes: bytes,
    element: np.ndarray,
    resid: np.ndarray,
    resname: np.ndarray,
    num_protein_atoms: int,
) -> AtomTopology:
    """Builds a heavy-atom :class:`AtomTopology` from a domain's PSF text.

    Args:
        psf_text_bytes: raw ``psf`` dataset contents.
        element: [num_protein_atoms] element symbols (from the h5 ``element`` array).
        resid: [num_protein_atoms] residue numbers (from the h5 ``resid`` array).
        resname: [num_protein_atoms] residue 3-letter codes.
        num_protein_atoms: value of the domain's ``numProteinAtoms`` attribute.

    Returns:
        An :class:`AtomTopology` indexed over heavy atoms only.
    """
    psf_lines = psf_text_bytes.decode("utf-8", errors="replace").splitlines()

    atom_header = next((i for i, line in enumerate(psf_lines) if "!NATOM" in line), None)
    if atom_header is None:
        raise ValueError("PSF has no !NATOM section")
    atom_names = [psf_lines[atom_header + 1 + i].split()[4] for i in range(num_protein_atoms)]

    is_heavy = np.asarray([e.strip().upper() != "H" for e in element], dtype=bool)
    heavy_indices = np.where(is_heavy)[0]

    # Map original all-atom index -> heavy-atom index (-1 for dropped hydrogens).
    to_heavy = np.full(num_protein_atoms, -1, dtype=np.int64)
    to_heavy[heavy_indices] = np.arange(heavy_indices.shape[0])

    def keep_protein_heavy(index_array: np.ndarray) -> np.ndarray:
        if index_array.shape[0] == 0:
            return index_array
        within_protein = (index_array >= 0).all(axis=1) & (index_array < num_protein_atoms).all(axis=1)
        index_array = index_array[within_protein]
        if index_array.shape[0] == 0:
            return index_array
        heavy_only = is_heavy[index_array].all(axis=1)
        return to_heavy[index_array[heavy_only]]

    bond_index = keep_protein_heavy(_parse_psf_section(psf_lines, "!NBOND", 2))
    angle_index = keep_protein_heavy(_parse_psf_section(psf_lines, "!NTHETA", 3))

    # Contiguous 0-based residue indexing, preserving order of first appearance.
    heavy_resid = np.asarray(resid)[heavy_indices]
    _, first_positions = np.unique(heavy_resid, return_index=True)
    ordered_resids = heavy_resid[np.sort(first_positions)]
    resid_to_residue_index = {int(r): i for i, r in enumerate(ordered_resids)}
    atom_residue_index = np.asarray([resid_to_residue_index[int(r)] for r in heavy_resid], dtype=np.int64)

    atom_element = np.asarray(
        [element_to_index(element[i]) for i in heavy_indices], dtype=np.int64
    )

    ca_atom_index = np.full(len(ordered_resids), -1, dtype=np.int64)
    for heavy_position, original_index in enumerate(heavy_indices):
        if atom_names[original_index] == "CA":
            ca_atom_index[atom_residue_index[heavy_position]] = heavy_position
    if (ca_atom_index < 0).any():
        raise ValueError("PSF topology is missing a CA atom for at least one residue")

    resname_array = np.asarray(resname)
    residue_names = [
        str(resname_array[heavy_indices[np.where(atom_residue_index == r)[0][0]]])
        for r in range(len(ordered_resids))
    ]

    return AtomTopology(
        heavy_indices=heavy_indices,
        bond_index=bond_index,
        angle_index=angle_index,
        atom_residue_index=atom_residue_index,
        atom_element=atom_element,
        ca_atom_index=ca_atom_index,
        residue_names=residue_names,
    )
