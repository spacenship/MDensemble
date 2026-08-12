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

Two atom selections share this machinery (see ``DataConfig.representation``):
``heavy_atom`` keeps every non-hydrogen protein atom, and ``backbone`` keeps
only N/CA/C/O. Selection is the *only* difference: the bond/angle filter
already drops any interaction touching an unselected atom, so a backbone
selection yields exactly the backbone covalent graph (N-CA, CA-C, C=O and
the peptide C-N) with no special-casing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

# PDB/CHARMM atom names of the protein backbone. Verified against the local
# sample shards: every residue carries exactly these four, and the CHARMM
# terminal patches (CAY/CY/OY at the N-terminus, NT/CAT at the C-terminus)
# are named differently *and* share the resid of the first/last residue, so
# excluding them by name drops no residue and adds no ragged edge case.
BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")


@dataclass
class AtomTopology:
    """Atom-level topology of one domain, re-indexed to the selected atoms.

    All index arrays address the *selected-atom* ordering
    (0 .. num_particles-1), i.e. after hydrogens (and, in backbone mode,
    side chains) have been dropped, so they can be used directly against
    the coordinate tensors produced by
    :class:`protein_flow.data.mdcath.MdCathDataset` in ``heavy_atom`` or
    ``backbone`` mode.

    Attributes:
        particle_indices: [num_particles] indices into the original all-atom arrays.
        bond_index: [num_bonds, 2] covalent bonds between selected atoms.
        angle_index: [num_angles, 3] covalent angles (i, centre, k).
        atom_residue_index: [num_particles] residue index (0-based, contiguous).
        atom_element: [num_particles] element type id (see ELEMENT_TO_INDEX).
        ca_atom_index: [num_residues] selected-atom index of each residue's CA.
        residue_names: 3-letter code per residue.
    """

    particle_indices: np.ndarray
    bond_index: np.ndarray
    angle_index: np.ndarray
    atom_residue_index: np.ndarray
    atom_element: np.ndarray
    ca_atom_index: np.ndarray
    residue_names: List[str]

    @property
    def num_particles(self) -> int:
        return int(self.particle_indices.shape[0])

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


def _parse_atom_topology(
    psf_text_bytes: bytes,
    element: np.ndarray,
    residue_labels: np.ndarray,
    resname: np.ndarray,
    num_protein_atoms: int,
    select: Callable[[np.ndarray, List[str]], np.ndarray],
) -> AtomTopology:
    """Builds an :class:`AtomTopology` over whatever atoms ``select`` keeps.

    Args:
        psf_text_bytes: raw ``psf`` dataset contents.
        element: [num_protein_atoms] element symbols (from the h5 ``element`` array).
        residue_labels: [num_protein_atoms] per-atom residue identity. Must
            distinguish residues that share a number but differ by PDB
            insertion code; mdCATH's raw ``resid`` array does not, so
            callers build these from the embedded PDB text instead.
        resname: [num_protein_atoms] residue 3-letter codes.
        num_protein_atoms: value of the domain's ``numProteinAtoms`` attribute.
        select: given the per-atom element array and the PSF atom names,
            returns a [num_protein_atoms] boolean keep-mask.

    Returns:
        An :class:`AtomTopology` indexed over the selected atoms only.
    """
    psf_lines = psf_text_bytes.decode("utf-8", errors="replace").splitlines()

    atom_header = next((i for i, line in enumerate(psf_lines) if "!NATOM" in line), None)
    if atom_header is None:
        raise ValueError("PSF has no !NATOM section")
    atom_names = [psf_lines[atom_header + 1 + i].split()[4] for i in range(num_protein_atoms)]

    is_selected = np.asarray(select(np.asarray(element), atom_names), dtype=bool)
    particle_indices = np.where(is_selected)[0]

    # Map original all-atom index -> particle index (-1 for dropped atoms).
    to_particle = np.full(num_protein_atoms, -1, dtype=np.int64)
    to_particle[particle_indices] = np.arange(particle_indices.shape[0])

    def keep_selected(index_array: np.ndarray) -> np.ndarray:
        """Drops any bond/angle touching solvent or an unselected atom.

        This is what makes the backbone selection work without extra logic:
        every interaction involving a hydrogen or a side-chain atom simply
        falls out here, leaving the pure backbone covalent graph.
        """
        if index_array.shape[0] == 0:
            return index_array
        within_protein = (index_array >= 0).all(axis=1) & (index_array < num_protein_atoms).all(axis=1)
        index_array = index_array[within_protein]
        if index_array.shape[0] == 0:
            return index_array
        selected_only = is_selected[index_array].all(axis=1)
        return to_particle[index_array[selected_only]]

    bond_index = keep_selected(_parse_psf_section(psf_lines, "!NBOND", 2))
    angle_index = keep_selected(_parse_psf_section(psf_lines, "!NTHETA", 3))

    # Contiguous 0-based residue indexing: a new residue starts wherever the
    # label changes as the atom list is walked.
    #
    # Grouping by *distinct label value* instead would silently merge
    # residues that share one, which is what PDB insertion codes produce:
    # 1hpgA02 numbers five consecutive residues 120, 120A, 120B, 120C and
    # 120D, and mdCATH's per-atom ``resid`` array stores the bare number for
    # all five. Merging them would hand a single "residue" 20 backbone atoms
    # and break the 4-per-residue invariant everything downstream relies on.
    # Callers must therefore pass labels that already separate them (see
    # ``parse_residue_labels`` in protein_flow/data/mdcath.py); walking runs
    # keeps that separation even when a label recurs later in the chain.
    particle_labels = np.asarray(residue_labels)[particle_indices]
    if particle_labels.shape[0] == 0:
        atom_residue_index = np.zeros(0, dtype=np.int64)
        num_residues_found = 0
    else:
        starts_new_residue = np.empty(particle_labels.shape[0], dtype=bool)
        starts_new_residue[0] = True
        starts_new_residue[1:] = particle_labels[1:] != particle_labels[:-1]
        atom_residue_index = np.cumsum(starts_new_residue) - 1
        num_residues_found = int(atom_residue_index[-1]) + 1

    atom_element = np.asarray(
        [element_to_index(element[i]) for i in particle_indices], dtype=np.int64
    )

    ca_atom_index = np.full(num_residues_found, -1, dtype=np.int64)
    for particle_position, original_index in enumerate(particle_indices):
        if atom_names[original_index] == "CA":
            ca_atom_index[atom_residue_index[particle_position]] = particle_position
    if (ca_atom_index < 0).any():
        raise ValueError("PSF topology is missing a CA atom for at least one residue")

    resname_array = np.asarray(resname)
    residue_names = [
        str(resname_array[particle_indices[np.where(atom_residue_index == r)[0][0]]])
        for r in range(num_residues_found)
    ]

    return AtomTopology(
        particle_indices=particle_indices,
        bond_index=bond_index,
        angle_index=angle_index,
        atom_residue_index=atom_residue_index,
        atom_element=atom_element,
        ca_atom_index=ca_atom_index,
        residue_names=residue_names,
    )


def parse_heavy_atom_topology(
    psf_text_bytes: bytes,
    element: np.ndarray,
    residue_labels: np.ndarray,
    resname: np.ndarray,
    num_protein_atoms: int,
) -> AtomTopology:
    """Topology over every non-hydrogen protein atom (~7.9 per residue)."""
    return _parse_atom_topology(
        psf_text_bytes, element, residue_labels, resname, num_protein_atoms,
        select=lambda elements, names: np.asarray(
            [e.strip().upper() != "H" for e in elements], dtype=bool
        ),
    )


def parse_backbone_topology(
    psf_text_bytes: bytes,
    element: np.ndarray,
    residue_labels: np.ndarray,
    resname: np.ndarray,
    num_protein_atoms: int,
) -> AtomTopology:
    """Topology over the N/CA/C/O backbone only (exactly 4 atoms per residue).

    Selection is by PSF atom *name*, not element, so the CHARMM terminal
    patch atoms (CAY/CY/OY, NT/CAT) are excluded even though they are heavy
    -- they carry the same resid as the first/last residue, so no residue is
    lost and the per-residue count stays exactly 4.
    """
    backbone_names = set(BACKBONE_ATOM_NAMES)
    return _parse_atom_topology(
        psf_text_bytes, element, residue_labels, resname, num_protein_atoms,
        select=lambda elements, names: np.asarray(
            [
                name in backbone_names and elements[i].strip().upper() != "H"
                for i, name in enumerate(names)
            ],
            dtype=bool,
        ),
    )
