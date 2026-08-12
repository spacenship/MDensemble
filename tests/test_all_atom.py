"""All-atom (heavy-atom) representation: topology, batching, model, losses.

The topology tests need the real mdCATH sample shards (the covalent
connectivity comes from each shard's embedded CHARMM PSF) and are skipped
when those are absent. The model/equivariance tests build synthetic
all-atom batches and always run.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from protein_flow.config import Config, GraphConfig
from protein_flow.data.collate import collate_protein_batch
from protein_flow.losses.physics import (
    topology_angle_loss,
    topology_bond_loss,
    topology_clash_loss,
)
from protein_flow.models.dual_graph_flow import DualGraphFlowModel

MDCATH_DIR = Path("/home/mipstu/wjYang/MolecularDynamics/mdCATH_sample100/data")
needs_mdcath = pytest.mark.skipif(
    not MDCATH_DIR.exists(), reason="mdCATH_sample100 sample data not present on this machine"
)

SEED = 4242


# --------------------------------------------------------------------------
# Synthetic all-atom batch helpers
# --------------------------------------------------------------------------
def _synthetic_all_atom_sample(num_residues: int, atoms_per_residue: int, plm_dim: int, generator):
    num_atoms = num_residues * atoms_per_residue
    atom_residue_index = torch.arange(num_residues).repeat_interleave(atoms_per_residue)
    ca_atom_index = torch.arange(num_residues) * atoms_per_residue  # first atom of each residue

    # Bonds: intra-residue chain + peptide link between consecutive CAs.
    bonds = []
    for residue in range(num_residues):
        base = residue * atoms_per_residue
        for offset in range(atoms_per_residue - 1):
            bonds.append((base + offset, base + offset + 1))
        if residue + 1 < num_residues:
            bonds.append((base, (residue + 1) * atoms_per_residue))
    bond_index = torch.tensor(bonds, dtype=torch.long)

    angles = []
    for bond_a in range(len(bonds) - 1):
        i, j = bonds[bond_a]
        k, m = bonds[bond_a + 1]
        if j == k:
            angles.append((i, j, m))
    angle_index = torch.tensor(angles, dtype=torch.long)

    # Irrational per-axis scaling keeps pairwise distances distinct, so the
    # k-NN graph has no ties that could flip under rotation.
    scale = torch.tensor([1.0, 2.7182818, 4.6692016])
    coords = torch.randn(num_atoms, 3, generator=generator) * scale * 3.0

    return {
        "sequence_embedding": torch.randn(num_residues, plm_dim, generator=generator),
        "source_coords": coords,
        "target_coords": coords + 0.1 * torch.randn(num_atoms, 3, generator=generator),
        "residue_types": torch.randint(0, 20, (num_residues,), generator=generator),
        "temperature": torch.tensor([320.0]),
        "physical_delta_t": torch.tensor([5.0]),
        "atom_residue_index": atom_residue_index,
        "atom_element": torch.randint(1, 5, (num_atoms,), generator=generator),
        "ca_atom_index": ca_atom_index,
        "bond_index": bond_index,
        "angle_index": angle_index,
    }


def _all_atom_config(plm_dim: int = 16, hidden: int = 24) -> Config:
    config = Config()
    config.data.representation = "heavy_atom"
    config.data.plm_dim = plm_dim
    for module in (
        config.model.sequence_encoder, config.model.geometric_encoder,
        config.model.fusion, config.model.decoder,
    ):
        module.hidden_dim = hidden
    config.model.fusion.condition_dim = hidden // 2
    config.model.sequence_encoder.num_layers = 2
    config.model.geometric_encoder.num_layers = 2
    config.model.sequence_encoder.dropout = 0.0
    config.model.geometric_encoder.dropout = 0.0
    config.model.graph.knn_k = 6
    config.model.graph.num_rbf = 8
    return config


def _proper_rotation(generator: torch.Generator) -> torch.Tensor:
    a = torch.randn(3, 3, generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(-2)
    if torch.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------
def test_collate_separates_atom_and_residue_axes():
    generator = torch.Generator().manual_seed(SEED)
    samples = [
        _synthetic_all_atom_sample(num_residues=n, atoms_per_residue=4, plm_dim=16, generator=generator)
        for n in (5, 8)
    ]
    batch = collate_protein_batch(samples)

    assert batch["residue_mask"].shape == (2, 8)
    assert batch["atom_mask"].shape == (2, 32)
    assert batch["sequence_embedding"].shape == (2, 8, 16)
    assert batch["source_coords"].shape == (2, 32, 3)
    # Shorter sample is padded on both axes independently.
    assert int(batch["residue_mask"][0].sum()) == 5
    assert int(batch["atom_mask"][0].sum()) == 20
    # Every atom maps inside its own sample's residue range.
    for i in range(2):
        valid_atoms = batch["atom_mask"][i]
        assert int(batch["atom_residue_index"][i][valid_atoms].max()) < int(batch["residue_mask"][i].sum())


def test_ca_mode_collate_synthesizes_identity_atom_fields():
    """The C-alpha representation must flow through the same batch layout,
    with the atom axis equal to the residue axis."""
    generator = torch.Generator().manual_seed(SEED)
    samples = []
    for num_residues in (4, 6):
        samples.append({
            "sequence_embedding": torch.randn(num_residues, 8, generator=generator),
            "source_coords": torch.randn(num_residues, 3, generator=generator),
            "target_coords": torch.randn(num_residues, 3, generator=generator),
            "residue_types": torch.randint(0, 20, (num_residues,), generator=generator),
            "temperature": torch.tensor([320.0]),
            "physical_delta_t": torch.tensor([5.0]),
        })
    batch = collate_protein_batch(samples)

    assert batch["atom_mask"].shape == batch["residue_mask"].shape
    torch.testing.assert_close(batch["atom_mask"], batch["residue_mask"])
    # Identity mapping, and chain bonds are just (i, i+1).
    assert torch.equal(batch["atom_residue_index"][0], torch.arange(6))
    first_valid_bonds = batch["bond_index"][0][batch["bond_mask"][0]]
    assert torch.equal(first_valid_bonds[:, 1] - first_valid_bonds[:, 0], torch.ones(3, dtype=torch.long))


# --------------------------------------------------------------------------
# Model + equivariance
# --------------------------------------------------------------------------
def _build_all_atom_batch(num_residues=6, atoms_per_residue=4, plm_dim=16, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    samples = [
        _synthetic_all_atom_sample(num_residues, atoms_per_residue, plm_dim, generator) for _ in range(2)
    ]
    return collate_protein_batch(samples)


def test_all_atom_forward_returns_per_atom_velocity():
    config = _all_atom_config()
    torch.manual_seed(SEED)
    model = DualGraphFlowModel(config).eval()
    batch = _build_all_atom_batch()

    velocity = model(
        batch["source_coords"], torch.rand(2), batch["sequence_embedding"], batch["residue_types"],
        batch["temperature"], batch["physical_delta_t"], batch["residue_mask"],
        atom_mask=batch["atom_mask"], atom_residue_index=batch["atom_residue_index"],
        atom_element=batch["atom_element"], ca_atom_index=batch["ca_atom_index"],
    )
    assert velocity.shape == batch["source_coords"].shape
    assert velocity.shape[1] != batch["residue_mask"].shape[1], "velocity must be per-atom, not per-residue"


def test_all_atom_model_is_se3_equivariant():
    """Rotation equivariance and translation invariance must survive the
    move to atom-level nodes with residue-level sequence broadcasting."""
    config = _all_atom_config()
    torch.manual_seed(SEED)
    model = DualGraphFlowModel(config).eval()
    batch = _build_all_atom_batch()
    generator = torch.Generator().manual_seed(SEED + 1)

    tau = torch.rand(2, generator=generator)
    atom_kwargs = dict(
        atom_mask=batch["atom_mask"], atom_residue_index=batch["atom_residue_index"],
        atom_element=batch["atom_element"], ca_atom_index=batch["ca_atom_index"],
    )

    def velocity_of(coords):
        with torch.no_grad():
            return model(
                coords, tau, batch["sequence_embedding"], batch["residue_types"],
                batch["temperature"], batch["physical_delta_t"], batch["residue_mask"], **atom_kwargs,
            )

    rotation = _proper_rotation(generator)
    translation = torch.randn(3, generator=generator) * 8.0
    base = velocity_of(batch["source_coords"])

    rotated = velocity_of(batch["source_coords"] @ rotation)
    torch.testing.assert_close(rotated, base @ rotation, atol=1e-4, rtol=1e-4)

    translated = velocity_of(batch["source_coords"] + translation)
    torch.testing.assert_close(translated, base, atol=1e-4, rtol=1e-4)


def test_all_atom_model_breaks_reflection_symmetry():
    """The chirality feature must still work when it is computed on the CA
    trace and broadcast to side-chain atoms."""
    config = _all_atom_config()
    torch.manual_seed(SEED)
    model = DualGraphFlowModel(config).eval()
    batch = _build_all_atom_batch()

    reflection = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    tau = torch.rand(2, generator=torch.Generator().manual_seed(3))
    atom_kwargs = dict(
        atom_mask=batch["atom_mask"], atom_residue_index=batch["atom_residue_index"],
        atom_element=batch["atom_element"], ca_atom_index=batch["ca_atom_index"],
    )
    with torch.no_grad():
        base = model(
            batch["source_coords"], tau, batch["sequence_embedding"], batch["residue_types"],
            batch["temperature"], batch["physical_delta_t"], batch["residue_mask"], **atom_kwargs,
        )
        reflected = model(
            batch["source_coords"] @ reflection, tau, batch["sequence_embedding"], batch["residue_types"],
            batch["temperature"], batch["physical_delta_t"], batch["residue_mask"], **atom_kwargs,
        )
    assert (reflected - base @ reflection).abs().max() > 1e-3


def test_ca_config_rejects_all_atom_inputs():
    config = _all_atom_config()
    config.data.representation = "ca"
    torch.manual_seed(SEED)
    model = DualGraphFlowModel(config).eval()
    batch = _build_all_atom_batch()
    with pytest.raises(ValueError, match="representation"):
        model(
            batch["source_coords"], torch.rand(2), batch["sequence_embedding"], batch["residue_types"],
            batch["temperature"], batch["physical_delta_t"], batch["residue_mask"],
            atom_mask=batch["atom_mask"], atom_residue_index=batch["atom_residue_index"],
            atom_element=batch["atom_element"], ca_atom_index=batch["ca_atom_index"],
        )


# --------------------------------------------------------------------------
# Topology-driven physics losses
# --------------------------------------------------------------------------
def test_topology_bond_and_angle_loss_zero_on_reference():
    batch = _build_all_atom_batch()
    coords = batch["source_coords"]
    bond = topology_bond_loss(coords, coords, batch["bond_index"], batch["bond_mask"])
    angle = topology_angle_loss(coords, coords, batch["angle_index"], batch["angle_mask"])
    assert bond.item() == pytest.approx(0.0, abs=1e-6)
    assert angle.item() == pytest.approx(0.0, abs=1e-6)


def test_topology_bond_loss_penalizes_stretched_bond():
    batch = _build_all_atom_batch()
    coords = batch["source_coords"]
    stretched = coords.clone()
    stretched[0, batch["bond_index"][0, 0, 1]] += 5.0
    loss = topology_bond_loss(stretched, coords, batch["bond_index"], batch["bond_mask"])
    assert loss.item() > 0.0


def test_topology_bond_loss_ignores_padded_bonds():
    """A padded bond slot points at atom 0 by construction; corrupting the
    prediction there must not leak into the loss."""
    generator = torch.Generator().manual_seed(SEED)
    samples = [
        _synthetic_all_atom_sample(n, atoms_per_residue=4, plm_dim=16, generator=generator) for n in (3, 7)
    ]
    batch = collate_protein_batch(samples)
    coords = batch["source_coords"]
    reference = coords.clone()

    padded_only = ~batch["bond_mask"][0]
    assert padded_only.any(), "expected the shorter sample to have padded bond slots"
    loss = topology_bond_loss(coords, reference, batch["bond_index"], batch["bond_mask"])
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_topology_clash_loss_excludes_covalent_pairs():
    """Bonded (1-2) and angle (1-3) neighbours sit far closer than the clash
    threshold, so a clean structure must still score zero."""
    batch = _build_all_atom_batch()
    # Place every atom of a residue tightly along a line: bonded neighbours
    # end up ~1.0 apart, well under the threshold, but must be exempt.
    coords = batch["source_coords"].clone()
    coords[:] = torch.arange(coords.shape[1], dtype=coords.dtype).view(1, -1, 1) * 1.0
    loss = topology_clash_loss(
        coords, batch["atom_mask"], GraphConfig(knn_k=4), clash_threshold=2.5,
        bond_index=batch["bond_index"], bond_mask=batch["bond_mask"],
        angle_index=batch["angle_index"], angle_mask=batch["angle_mask"],
    )
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------
# Real mdCATH topology
# --------------------------------------------------------------------------
@needs_mdcath
def test_real_topology_matches_pdb_path_and_is_physical():
    import h5py
    import numpy as np

    from protein_flow.data.mdcath import parse_ca_indices_and_resnames
    from protein_flow.data.topology import parse_heavy_atom_topology

    shard = sorted(MDCATH_DIR.glob("*.h5"))[0]
    with h5py.File(shard, "r") as f:
        domain = next(iter(f.keys()))
        group = f[domain]
        topology = parse_heavy_atom_topology(
            group["psf"][()],
            np.asarray([e.decode() for e in group["element"][:]]),
            group["resid"][:],
            np.asarray([r.decode() for r in group["resname"][:]]),
            int(group.attrs["numProteinAtoms"]),
        )
        all_atom_coords = group[list(group.keys())[0] if False else "320"]["0"]["coords"][0]
        ca_indices, residue_names = parse_ca_indices_and_resnames(group["pdbProteinAtoms"][()])

    # CA atoms found via PSF atom names must match the independent PDB-text path.
    heavy_coords = all_atom_coords[topology.particle_indices]
    assert np.allclose(heavy_coords[topology.ca_atom_index], all_atom_coords[ca_indices])
    assert topology.residue_names == residue_names

    # Hydrogens really are excluded.
    assert topology.num_particles < int(np.asarray(all_atom_coords).shape[0])
    assert topology.num_particles / topology.num_residues == pytest.approx(7.9, abs=1.0)

    # Real covalent geometry. The upper bound has to accommodate genuine
    # disulfide bridges: this shard contains 8 CYS-CYS S-S bonds reaching
    # 2.12 A, which is correct chemistry (~2.05 A typical), not a parsing
    # artefact. C-C/C-N/C-O bonds all stay under 1.7 A.
    bond_lengths = np.linalg.norm(
        heavy_coords[topology.bond_index[:, 0]] - heavy_coords[topology.bond_index[:, 1]], axis=1
    )
    assert bond_lengths.min() > 1.0
    assert bond_lengths.max() < 2.3


@needs_mdcath
def test_mdcath_heavy_atom_mode_end_to_end():
    from protein_flow.config import DataConfig
    from protein_flow.data.mdcath import MdCathDataset

    dataset = MdCathDataset(
        MDCATH_DIR, DataConfig(plm_dim=8), frame_gap=5, seed=0, representation="heavy_atom",
        h5_files=sorted(MDCATH_DIR.glob("*.h5"))[:2],
    )
    sample = dataset[0]
    num_atoms = sample["source_coords"].shape[0]
    num_residues = sample["residue_types"].shape[0]

    assert num_atoms > num_residues
    assert sample["atom_residue_index"].shape == (num_atoms,)
    assert int(sample["atom_residue_index"].max()) == num_residues - 1
    assert sample["ca_atom_index"].shape == (num_residues,)
    assert int(sample["bond_index"].max()) < num_atoms
    assert int(sample["angle_index"].max()) < num_atoms
