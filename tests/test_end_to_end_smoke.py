"""End-to-end CPU smoke test: dataset -> Kabsch -> model -> losses -> backward
-> optimizer step -> ODE sampling, all in one pass, on a tiny synthetic
configuration. This complements the per-module unit tests by checking the
full pipeline wiring rather than any single component in isolation.
"""
from __future__ import annotations

import torch

from protein_flow.config import Config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.synthetic import SyntheticProteinTrajectoryDataset
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.models.dual_graph_flow import DualGraphFlowModel
from protein_flow.train import compute_losses


def _tiny_config() -> Config:
    config = Config()
    config.data.plm_dim = 16
    config.data.num_amino_acid_types = 22
    config.data.min_length = 8
    config.data.max_length = 12
    config.model.sequence_encoder.hidden_dim = 12
    config.model.geometric_encoder.hidden_dim = 12
    config.model.fusion.hidden_dim = 12
    config.model.fusion.condition_dim = 6
    config.model.graph.knn_k = 4
    config.model.graph.num_rbf = 6
    return config


def test_full_pipeline_runs_on_cpu_and_all_shapes_are_consistent():
    torch.manual_seed(0)
    config = _tiny_config()

    dataset = SyntheticProteinTrajectoryDataset(config.data, size=6, seed=0)
    batch = collate_protein_batch([dataset[i] for i in range(6)])

    batch_size, length = batch["residue_mask"].shape
    assert batch["sequence_embedding"].shape == (batch_size, length, config.data.plm_dim)

    kabsch_result = masked_kabsch_align(batch["source_coords"], batch["target_coords"], batch["residue_mask"])
    assert kabsch_result.aligned_target.shape == (batch_size, length, 3)
    assert torch.all(torch.det(kabsch_result.rotation) > 0)

    model = DualGraphFlowModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses_over_steps = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(model, batch, config)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses_over_steps.append(losses["total"].item())

    assert all(torch.isfinite(torch.tensor(v)) for v in losses_over_steps)

    model.eval()
    generated_coords, trajectory = model.sample(
        batch["source_coords"],
        batch["sequence_embedding"],
        batch["residue_types"],
        batch["residue_mask"],
        batch["temperature"],
        batch["physical_delta_t"],
        num_steps=5,
        solver="heun",
        return_trajectory=True,
    )
    assert generated_coords.shape == (batch_size, length, 3)
    assert trajectory.shape == (6, batch_size, length, 3)
    assert torch.all(torch.isfinite(generated_coords))
    # padding residues must remain exactly at their (zero-padded) initial position
    assert torch.all(generated_coords[~batch["residue_mask"]] == 0.0)
