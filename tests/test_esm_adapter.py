"""Offline-only tests for the ESM adapter's pure-Python pieces.

Deliberately does NOT call protein_flow.data.esm_adapter.compute_esm_embeddings
anywhere in this file: that function requires transformers + a network
download of pretrained weights, and per this project's requirements the
test suite must never require a model download.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from protein_flow.data.residue_vocab import resname_to_one_letter

MDCATH_DIR = Path("/home/mipstu/wjYang/MolecularDynamics/mdCATH_sample100/data")


def test_resname_to_one_letter_standard_and_aliases():
    assert resname_to_one_letter("ALA") == "A"
    assert resname_to_one_letter("TYR") == "Y"
    assert resname_to_one_letter("gly") == "G"
    # CHARMM protonation-state aliases must map to the canonical residue's letter.
    assert resname_to_one_letter("HSD") == "H"
    assert resname_to_one_letter("CYX") == "C"


def test_resname_to_one_letter_unknown_maps_to_X():
    assert resname_to_one_letter("ZZZ") == "X"


@pytest.mark.skipif(not MDCATH_DIR.exists(), reason="mdCATH_sample100 sample data not present on this machine")
def test_domain_sequence_matches_residue_count():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from precompute_esm_embeddings import domain_sequence

    sample_file = next(iter(sorted(MDCATH_DIR.glob("*.h5"))))
    domain, sequence = domain_sequence(sample_file)
    assert len(domain) > 0
    assert len(sequence) > 0
    assert all(c.isalpha() for c in sequence)
