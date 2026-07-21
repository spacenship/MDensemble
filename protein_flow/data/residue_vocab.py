"""Amino-acid 3-letter-code -> integer type index mapping.

Covers the 20 standard residues plus common CHARMM/AMBER protonation-state
and disulfide-state aliases seen in real MD topologies (e.g. mdCATH, which
uses CHARMM residue naming). Anything not recognized maps to ``UNKNOWN_INDEX``
rather than raising, since real structures can contain modified residues.
"""
from __future__ import annotations

_STANDARD_20 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]

# CHARMM/AMBER protonation-state and disulfide-state aliases for the same
# canonical residue (e.g. HSD/HSE/HSP are different HIS protonation states
# in CHARMM force fields; CYX is a disulfide-bonded CYS in AMBER).
_ALIASES = {
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "CYX": "CYS", "CYM": "CYS",
    "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
}

UNKNOWN_INDEX = len(_STANDARD_20)  # 20

RESNAME_TO_INDEX = {name: i for i, name in enumerate(_STANDARD_20)}
for alias, canonical in _ALIASES.items():
    RESNAME_TO_INDEX[alias] = RESNAME_TO_INDEX[canonical]

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
UNKNOWN_ONE_LETTER = "X"  # ESM's tokenizer treats "X" as an unknown-residue token


def resname_to_index(resname: str) -> int:
    """Maps a 3-letter residue code to an integer index; unknown -> UNKNOWN_INDEX."""
    return RESNAME_TO_INDEX.get(resname.strip().upper(), UNKNOWN_INDEX)


def resname_to_one_letter(resname: str) -> str:
    """Maps a 3-letter residue code (including CHARMM/AMBER aliases) to its
    standard 1-letter code; unknown/modified residues -> "X"."""
    resname = resname.strip().upper()
    canonical = _ALIASES.get(resname, resname)
    return _THREE_TO_ONE.get(canonical, UNKNOWN_ONE_LETTER)
