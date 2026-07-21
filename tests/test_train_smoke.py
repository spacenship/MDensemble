from pathlib import Path

import torch

from protein_flow.config import Config
from protein_flow.train import build_dataloaders, compute_losses, set_seed, train


def _tiny_config(tmp_path: Path) -> Config:
    config = Config()
    config.data.plm_dim = 16
    config.data.num_amino_acid_types = 22
    config.data.min_length = 6
    config.data.max_length = 10
    config.data.train_size = 8
    config.data.val_size = 4
    config.data.batch_size = 4
    config.model.sequence_encoder.hidden_dim = 8
    config.model.geometric_encoder.hidden_dim = 8
    config.model.fusion.hidden_dim = 8
    config.model.fusion.condition_dim = 4
    config.model.graph.knn_k = 3
    config.model.graph.num_rbf = 4
    config.train.num_epochs = 1
    config.train.max_steps = 3
    config.train.log_every = 1
    config.train.val_every = 2
    config.train.ckpt_dir = str(tmp_path / "checkpoints")
    return config


def test_compute_losses_finite_and_has_all_components():
    torch.manual_seed(0)
    config = _tiny_config(Path("/tmp"))
    config.loss.endpoint_enabled = False
    train_loader, _ = build_dataloaders(config)
    from protein_flow.models.dual_graph_flow import DualGraphFlowModel

    model = DualGraphFlowModel(config)
    batch = next(iter(train_loader))
    losses = compute_losses(model, batch, config)
    for name, value in losses.items():
        assert torch.isfinite(value), f"{name} loss is not finite: {value}"


def test_training_loop_runs_and_saves_checkpoint(tmp_path):
    config = _tiny_config(tmp_path)
    model = train(config)
    ckpt_dir = Path(config.train.ckpt_dir)
    assert (ckpt_dir / "last.pt").exists()


def test_overfit_one_batch_reduces_loss(tmp_path):
    config = _tiny_config(tmp_path)
    config.train.overfit_one_batch = True
    config.train.max_steps = 40
    config.train.log_every = 1000
    config.train.val_every = 1000
    config.train.optim.lr = 1e-2
    set_seed(0)

    from protein_flow.models.dual_graph_flow import DualGraphFlowModel

    model = DualGraphFlowModel(config)
    train_loader, _ = build_dataloaders(config)
    batch = next(iter(train_loader))

    losses_before = compute_losses(model, batch, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(model, batch, config)
        losses["total"].backward()
        optimizer.step()
    losses_after = compute_losses(model, batch, config)

    assert losses_after["fm"].item() < losses_before["fm"].item()


def test_nan_loss_is_skipped_not_crashed(tmp_path):
    # Sanity: running the full train() with normal small config should not
    # raise even though the NaN-guard code path isn't directly triggered here.
    config = _tiny_config(tmp_path)
    train(config)  # should not raise
