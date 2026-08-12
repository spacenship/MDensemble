"""Tests for the backbone (N/CA/C/O) representation.

The topology tests run against real mdCATH shards, because the point of this
representation is that it matches real CHARMM atom naming -- a synthetic
fixture would only prove the code agrees with itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("h5py")

from protein_flow.config import Config, DataConfig, PhysicsLossConfig, is_atom_level
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.mdcath import MdCathDataset, parse_ca_indices_and_resnames
from protein_flow.data.topology import parse_backbone_topology, parse_heavy_atom_topology
from protein_flow.models.dual_graph_flow import DualGraphFlowModel
from protein_flow.train import compute_losses

MDCATH_DIR = Path("/home/mipstu/wjYang/MolecularDynamics/mdCATH_sample100/data")

pytestmark = pytest.mark.skipif(
    not MDCATH_DIR.exists(), reason="mdCATH_sample100 sample data not present on this machine"
)


def _topology_inputs(group):
    """Mirror production exactly: residue identity comes from the embedded
    PDB (which carries insertion codes), never from the raw ``resid`` array."""
    from protein_flow.data.mdcath import parse_residue_labels

    num_protein_atoms = int(group.attrs["numProteinAtoms"])
    return (
        group["psf"][()],
        np.asarray([e.decode() for e in group["element"][:]]),
        parse_residue_labels(group["pdbProteinAtoms"][()])[:num_protein_atoms],
        np.asarray([r.decode() for r in group["resname"][:]]),
        num_protein_atoms,
    )


def test_is_atom_level_covers_both_atom_representations():
    assert is_atom_level("backbone")
    assert is_atom_level("heavy_atom")
    assert not is_atom_level("ca")
    with pytest.raises(ValueError):
        is_atom_level("all_atom")


def test_backbone_topology_is_exactly_four_atoms_per_residue():
    """The selection must be regular: 4 particles per residue, no ragged
    termini. CHARMM's cap atoms (CAY/CY/OY, NT/CAT) share the first/last
    residue's resid, so excluding them by name must not drop a residue."""
    import h5py

    for shard in sorted(MDCATH_DIR.glob("*.h5"))[:3]:
        with h5py.File(shard, "r") as f:
            domain = next(iter(f.keys()))
            group = f[domain]
            topology = parse_backbone_topology(*_topology_inputs(group))
            heavy = parse_heavy_atom_topology(*_topology_inputs(group))
            ca_indices, residue_names = parse_ca_indices_and_resnames(group["pdbProteinAtoms"][()])
            first_frame = group["320"]["0"]["coords"][0]

        assert topology.num_particles == 4 * topology.num_residues
        assert topology.num_residues == len(ca_indices)
        assert topology.residue_names == residue_names
        # Fewer particles than the all-atom selection, by roughly half.
        assert topology.num_particles < heavy.num_particles

        # The CA found via PSF atom names must coincide with the independent
        # PDB-text path, exactly as in heavy-atom mode.
        backbone_coords = first_frame[topology.particle_indices]
        assert np.allclose(backbone_coords[topology.ca_atom_index], first_frame[ca_indices])


def test_backbone_covalent_graph_is_the_real_peptide_chain():
    """3 intra-residue bonds (N-CA, CA-C, C=O) plus one peptide C-N per
    junction, i.e. 4N-1 -- and physical lengths/angles."""
    import h5py

    shard = sorted(MDCATH_DIR.glob("*.h5"))[0]
    with h5py.File(shard, "r") as f:
        group = f[next(iter(f.keys()))]
        topology = parse_backbone_topology(*_topology_inputs(group))
        coords = group["320"]["0"]["coords"][0][topology.particle_indices]

    assert topology.bond_index.shape[0] == 4 * topology.num_residues - 1

    lengths = np.linalg.norm(
        coords[topology.bond_index[:, 0]] - coords[topology.bond_index[:, 1]], axis=1
    )
    assert lengths.min() > 1.0
    assert lengths.max() < 1.8  # no side chains, so no 2.05 A disulfide here

    first = coords[topology.angle_index[:, 0]] - coords[topology.angle_index[:, 1]]
    second = coords[topology.angle_index[:, 2]] - coords[topology.angle_index[:, 1]]
    cosine = (first * second).sum(-1) / (
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    assert angles.min() > 95.0
    assert angles.max() < 140.0


def test_clash_threshold_stays_below_real_backbone_contacts():
    """The configured threshold must not penalize real structures.

    Measured over 45 sample domains x 6 frames x 2 temperatures, the closest
    non-bonded backbone pair the k-NN clash term sees is 2.278 A; this checks
    the shipped default still clears the same bar on a couple of shards.
    """
    import h5py

    from protein_flow.geometry.graph import build_geometric_graph
    from protein_flow.losses.physics import _covalent_exclusion_keys

    threshold = PhysicsLossConfig().clash_threshold_for("backbone")
    for shard in sorted(MDCATH_DIR.glob("*.h5"))[:2]:
        with h5py.File(shard, "r") as f:
            group = f[next(iter(f.keys()))]
            topology = parse_backbone_topology(*_topology_inputs(group))
            dataset = group["320"]["0"]["coords"]
            frames = np.stack(
                [
                    np.asarray(dataset[index][topology.particle_indices], dtype=np.float32)
                    for index in (0, dataset.shape[0] // 2, dataset.shape[0] - 1)
                ]
            )

        coords = torch.from_numpy(frames)
        num_particles = coords.shape[1]
        mask = torch.ones(coords.shape[:2], dtype=torch.bool)
        bond = torch.from_numpy(topology.bond_index)[None].expand(coords.shape[0], -1, -1)
        angle = torch.from_numpy(topology.angle_index)[None].expand(coords.shape[0], -1, -1)
        graph = build_geometric_graph(coords, mask, k=16, use_radius_cutoff=False, radius_cutoff=12.0)
        src, dst = graph.edge_index
        local_src, local_dst = src % num_particles, dst % num_particles
        keys = (
            (src // num_particles) * num_particles * num_particles
            + torch.minimum(local_src, local_dst) * num_particles
            + torch.maximum(local_src, local_dst)
        )
        excluded = _covalent_exclusion_keys(
            bond, torch.ones(bond.shape[:2], dtype=torch.bool),
            angle, torch.ones(angle.shape[:2], dtype=torch.bool), num_particles,
        )
        closest = graph.distances[~torch.isin(keys, excluded)].min().item()
        assert closest > threshold, f"{shard.name}: real contact at {closest:.3f} A <= threshold {threshold}"


def test_max_residues_skips_long_domains():
    shards = sorted(MDCATH_DIR.glob("*.h5"))[:4]
    unfiltered = MdCathDataset(MDCATH_DIR, DataConfig(plm_dim=8), h5_files=shards, frame_gap=5, seed=0)
    lengths = sorted(meta[2] for meta in unfiltered._domain_meta.values())
    cap = lengths[len(lengths) // 2]

    filtered = MdCathDataset(
        MDCATH_DIR, DataConfig(plm_dim=8), h5_files=shards, frame_gap=5, seed=0, max_residues=cap
    )
    assert filtered.skipped_too_long, "expected at least one domain above the cap to be skipped"
    assert all(length > cap for _, length in filtered.skipped_too_long)
    assert all(meta[2] <= cap for meta in filtered._domain_meta.values())
    assert len(filtered) < len(unfiltered)


def test_backbone_mode_end_to_end():
    """Dataset -> collate -> model -> losses, with atom-level batch fields."""
    config = Config()
    config.data.representation = "backbone"
    config.data.plm_dim = 8
    for module in (
        config.model.sequence_encoder, config.model.geometric_encoder,
        config.model.fusion, config.model.decoder,
    ):
        module.hidden_dim = 16
    config.model.fusion.condition_dim = 8

    dataset = MdCathDataset(
        MDCATH_DIR, config.data, h5_files=sorted(MDCATH_DIR.glob("*.h5"))[:2],
        frame_gap=5, seed=0, representation="backbone",
    )
    batch = collate_protein_batch([dataset[0], dataset[1]])
    num_residues = batch["residue_mask"].sum(dim=1)
    num_particles = batch["atom_mask"].sum(dim=1)
    assert torch.equal(num_particles, 4 * num_residues)
    assert batch["bond_index"].shape[-1] == 2
    assert batch["angle_index"].shape[-1] == 3

    model = DualGraphFlowModel(config)
    assert model.element_embedding is not None, "backbone mode must distinguish N/C/O by element"
    losses = compute_losses(model, batch, config)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(p.grad is not None for p in model.parameters())
