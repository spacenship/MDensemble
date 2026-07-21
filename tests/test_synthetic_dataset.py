import torch

from protein_flow.config import DataConfig
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.synthetic import SyntheticProteinTrajectoryDataset
from protein_flow.geometry.kabsch import masked_kabsch_align


def _make_dataset(min_length=8, max_length=16, size=20, seed=0):
    config = DataConfig(plm_dim=12, num_amino_acid_types=22, min_length=min_length, max_length=max_length)
    return SyntheticProteinTrajectoryDataset(config, size=size, seed=seed)


def test_sample_shapes_and_length_range():
    dataset = _make_dataset()
    for i in range(len(dataset)):
        sample = dataset[i]
        length = sample["residue_types"].shape[0]
        assert 8 <= length <= 16
        assert sample["sequence_embedding"].shape == (length, 12)
        assert sample["source_coords"].shape == (length, 3)
        assert sample["target_coords"].shape == (length, 3)
        assert sample["temperature"].shape == (1,)
        assert sample["physical_delta_t"].shape == (1,)
        assert torch.all(sample["physical_delta_t"] > 0)


def test_deterministic_given_seed_and_index():
    dataset_a = _make_dataset(seed=42)
    dataset_b = _make_dataset(seed=42)
    sample_a = dataset_a[3]
    sample_b = dataset_b[3]
    torch.testing.assert_close(sample_a["source_coords"], sample_b["source_coords"])
    torch.testing.assert_close(sample_a["target_coords"], sample_b["target_coords"])


def test_source_backbone_has_consistent_bond_length():
    dataset = _make_dataset(min_length=20, max_length=20, size=5)
    sample = dataset[0]
    coords = sample["source_coords"]
    bond_lengths = (coords[1:] - coords[:-1]).norm(dim=-1)
    torch.testing.assert_close(bond_lengths, torch.full_like(bond_lengths, 3.8), atol=1e-4, rtol=1e-4)


def test_kabsch_removes_rigid_part_but_leaves_deformation():
    dataset = _make_dataset(min_length=24, max_length=24, size=5)
    sample = dataset[1]
    source = sample["source_coords"].unsqueeze(0)
    target = sample["target_coords"].unsqueeze(0)
    mask = torch.ones(1, source.shape[1], dtype=torch.bool)

    result = masked_kabsch_align(source, target, mask)
    # Rigid transform must be substantially removed...
    assert result.post_rmsd.item() < result.pre_rmsd.item()
    # ...but real internal deformation should remain (not exactly zero).
    assert result.post_rmsd.item() > 1e-2


def test_collate_batch_shapes_and_mask():
    dataset = _make_dataset(min_length=6, max_length=14, size=8)
    samples = [dataset[i] for i in range(8)]
    batch = collate_protein_batch(samples)

    lengths = [s["residue_types"].shape[0] for s in samples]
    max_length = max(lengths)

    assert batch["sequence_embedding"].shape == (8, max_length, 12)
    assert batch["source_coords"].shape == (8, max_length, 3)
    assert batch["target_coords"].shape == (8, max_length, 3)
    assert batch["residue_types"].shape == (8, max_length)
    assert batch["residue_mask"].shape == (8, max_length)
    assert batch["temperature"].shape == (8, 1)
    assert batch["physical_delta_t"].shape == (8, 1)

    for i, length in enumerate(lengths):
        assert batch["residue_mask"][i, :length].all()
        if length < max_length:
            assert not batch["residue_mask"][i, length:].any()
