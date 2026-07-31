"""Tests for the real mdCATH HDF5 adapter.

These run against the actual sample shards under mdCATH_sample100/ (outside
the repo, not something this project ships or generates), so they are
skipped automatically if that directory isn't present -- e.g. on a fresh
clone of this repo, or in CI, where only the synthetic-dataset tests apply.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")

from protein_flow.config import DataConfig
from protein_flow.data.mdcath import MdCathDataset, parse_ca_indices_and_resnames
from protein_flow.data.residue_vocab import UNKNOWN_INDEX
from protein_flow.geometry.kabsch import masked_kabsch_align

MDCATH_DIR = Path("/home/mipstu/wjYang/MolecularDynamics/mdCATH_sample100/data")

pytestmark = pytest.mark.skipif(
    not MDCATH_DIR.exists(), reason="mdCATH_sample100 sample data not present on this machine"
)


def _make_dataset(**kwargs) -> MdCathDataset:
    data_config = DataConfig(plm_dim=8, num_amino_acid_types=22)
    return MdCathDataset(MDCATH_DIR, data_config, seed=0, **kwargs)


def test_indexes_all_shards_and_trajectories():
    dataset = _make_dataset(frame_gap=5)
    num_shards = len(list(MDCATH_DIR.glob("*.h5")))
    # 5 temperatures x 5 replicas per domain, assuming every trajectory has > frame_gap frames.
    assert len(dataset) <= num_shards * 25
    assert len(dataset) > 0


def test_sample_shapes_and_dtypes():
    dataset = _make_dataset(frame_gap=5)
    sample = dataset[0]
    length = sample["residue_types"].shape[0]
    assert sample["sequence_embedding"].shape == (length, 8)
    assert sample["source_coords"].shape == (length, 3)
    assert sample["target_coords"].shape == (length, 3)
    assert sample["residue_types"].dtype == torch.long
    assert sample["temperature"].shape == (1,)
    assert sample["physical_delta_t"].shape == (1,)


def test_temperature_matches_known_mdcath_replica_temperatures():
    dataset = _make_dataset(frame_gap=5)
    known_temperatures = {320.0, 348.0, 379.0, 413.0, 450.0}
    seen = set()
    for i in range(0, len(dataset), 137):
        seen.add(float(dataset[i]["temperature"].item()))
    assert seen.issubset(known_temperatures)


def test_ca_consecutive_distances_are_physically_plausible():
    """If CA extraction were wrong (e.g. picked the wrong atom), consecutive
    distances would not cluster around the ~3.8 A virtual C-alpha bond."""
    dataset = _make_dataset(frame_gap=5)
    sample = dataset[0]
    coords = sample["source_coords"]
    distances = (coords[1:] - coords[:-1]).norm(dim=-1)
    assert distances.mean().item() == pytest.approx(3.8, abs=0.3)
    assert distances.min().item() > 2.5
    assert distances.max().item() < 5.0


def test_residue_types_are_mostly_known():
    dataset = _make_dataset(frame_gap=5)
    unknown = 0
    total = 0
    for i in range(0, len(dataset), 25):  # one trajectory per domain
        types = dataset[i]["residue_types"]
        unknown += (types == UNKNOWN_INDEX).sum().item()
        total += types.numel()
    assert unknown / total < 0.01


def test_deterministic_given_seed_and_index():
    dataset_a = _make_dataset(frame_gap=5)
    dataset_b = _make_dataset(frame_gap=5)
    sample_a = dataset_a[10]
    sample_b = dataset_b[10]
    torch.testing.assert_close(sample_a["source_coords"], sample_b["source_coords"])
    torch.testing.assert_close(sample_a["target_coords"], sample_b["target_coords"])


def test_pairs_per_trajectory_expands_dataset():
    single = _make_dataset(frame_gap=5, pairs_per_trajectory=1)
    multiple = _make_dataset(frame_gap=5, pairs_per_trajectory=4)
    assert len(multiple) == 4 * len(single)


def test_training_frame_pairs_change_with_epoch():
    dataset = _make_dataset(frame_gap=5, resample_each_epoch=True)
    dataset.set_epoch(0)
    source_epoch_zero = dataset[10]["source_coords"]
    sources = []
    for epoch in range(1, 5):
        dataset.set_epoch(epoch)
        sources.append(dataset[10]["source_coords"])
    assert any(not torch.equal(source_epoch_zero, source) for source in sources)


def test_validation_frame_pairs_ignore_epoch():
    dataset = _make_dataset(frame_gap=5, pairs_per_trajectory=3, resample_each_epoch=False)
    source_epoch_zero = dataset[11]["source_coords"]
    dataset.set_epoch(99)
    torch.testing.assert_close(source_epoch_zero, dataset[11]["source_coords"])


def test_sampling_max_gap_keeps_sources_paired_across_gap_ablations():
    gap_one = _make_dataset(frame_gap=1, sampling_max_frame_gap=5)
    gap_five = _make_dataset(frame_gap=5, sampling_max_frame_gap=5)
    for index in (0, 10, 100):
        torch.testing.assert_close(gap_one[index]["source_coords"], gap_five[index]["source_coords"])
        assert not torch.equal(gap_one[index]["target_coords"], gap_five[index]["target_coords"])


def test_ps_per_frame_scales_physical_delta_t():
    dataset = _make_dataset(frame_gap=4, ps_per_frame=10.0)
    sample = dataset[0]
    assert sample["physical_delta_t"].item() == pytest.approx(40.0)


def test_kabsch_alignment_works_on_real_frame_pairs():
    dataset = _make_dataset(frame_gap=20)
    sample = dataset[0]
    source = sample["source_coords"].unsqueeze(0)
    target = sample["target_coords"].unsqueeze(0)
    mask = torch.ones(1, source.shape[1], dtype=torch.bool)

    result = masked_kabsch_align(source, target, mask)
    assert torch.isfinite(result.post_rmsd).all()
    assert result.post_rmsd.item() <= result.pre_rmsd.item() + 1e-4


def test_parse_ca_indices_and_resnames_matches_domain_metadata():
    import h5py

    sample_file = next(iter(sorted(MDCATH_DIR.glob("*.h5"))))
    with h5py.File(sample_file, "r") as f:
        domain = next(iter(f.keys()))
        group = f[domain]
        num_residues = int(group.attrs["numResidues"])
        pdb_bytes = group["pdbProteinAtoms"][()]

    ca_indices, residue_names = parse_ca_indices_and_resnames(pdb_bytes)
    assert len(ca_indices) == num_residues
    assert len(residue_names) == num_residues
    assert all(len(name) == 3 for name in residue_names)
