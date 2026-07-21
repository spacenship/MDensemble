"""Training loop for the dual-graph conditional-flow-matching model."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from protein_flow.config import Config, save_config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.synthetic import SyntheticProteinTrajectoryDataset
from protein_flow.flow.paths import build_flow_path, sample_tau
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.losses.flow_matching import flow_matching_loss
from protein_flow.losses.physics import compute_physics_losses, endpoint_rmsd_loss
from protein_flow.models.dual_graph_flow import DualGraphFlowModel

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    train_dataset = SyntheticProteinTrajectoryDataset(config.data, size=config.data.train_size, seed=config.data.seed)
    val_dataset = SyntheticProteinTrajectoryDataset(
        config.data, size=config.data.val_size, seed=config.data.seed + 1
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_protein_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_protein_batch,
    )
    return train_loader, val_loader


def _move_batch_to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _differentiable_euler_rollout(
    model: nn.Module, x0: Tensor, seq_emb: Tensor, residue_types: Tensor, mask: Tensor,
    temperature: Tensor, physical_delta_t: Tensor, num_steps: int,
) -> Tensor:
    """Gradient-carrying Euler rollout, used only for the optional endpoint
    RMSD loss. Unlike :func:`protein_flow.flow.solver.integrate_ode` (which
    runs under no_grad for pure sampling) this must stay differentiable, at
    the cost of backpropagating through ``num_steps`` full model calls."""
    device, dtype = x0.device, x0.dtype
    batch_size = x0.shape[0]
    dtau = 1.0 / num_steps
    x = x0
    for step in range(num_steps):
        tau_batch = torch.full((batch_size,), step * dtau, device=device, dtype=dtype)
        velocity = model(x, tau_batch, seq_emb, residue_types, temperature, physical_delta_t, mask)
        x = x + dtau * velocity
    return x


def compute_losses(model: nn.Module, batch: Dict[str, Tensor], config: Config) -> Dict[str, Tensor]:
    """Runs one forward pass and returns a dict of loss components (including 'total')."""
    source_coords = batch["source_coords"]
    target_coords = batch["target_coords"]
    residue_mask = batch["residue_mask"]

    kabsch_result = masked_kabsch_align(source_coords, target_coords, residue_mask)
    x0 = source_coords
    x1 = kabsch_result.aligned_target

    flow_path = build_flow_path(config.flow.path_type, config.flow.sigma_min)
    tau = sample_tau(x0.shape[0], device=x0.device, dtype=x0.dtype)
    x_tau, target_velocity = flow_path.sample(x0, x1, tau)

    predicted_velocity = model(
        x_tau, tau, batch["sequence_embedding"], batch["residue_types"],
        batch["temperature"], batch["physical_delta_t"], residue_mask,
    )

    loss_fm = flow_matching_loss(predicted_velocity, target_velocity, residue_mask)
    losses = {"fm": loss_fm}
    total = config.loss.lambda_fm * loss_fm

    physics_cfg = config.loss.physics
    if physics_cfg.enabled:
        reference_coords = x0 if physics_cfg.d_ref_source == "source" else x1
        if physics_cfg.apply_to == "x_tau":
            pred_coords_for_physics = x_tau
        else:
            pred_coords_for_physics = x_tau + physics_cfg.step_scale * predicted_velocity

        physics_out = compute_physics_losses(
            pred_coords_for_physics, reference_coords, residue_mask, config.model.graph,
            physics_cfg.clash_threshold, physics_cfg.clash_seq_sep,
        )
        losses["bond"] = physics_out.bond
        losses["angle"] = physics_out.angle
        losses["clash"] = physics_out.clash
        total = (
            total
            + config.loss.lambda_bond * physics_out.bond
            + config.loss.lambda_angle * physics_out.angle
            + config.loss.lambda_clash * physics_out.clash
        )

    if config.loss.endpoint_enabled:
        rollout_final = _differentiable_euler_rollout(
            model, x0, batch["sequence_embedding"], batch["residue_types"], residue_mask,
            batch["temperature"], batch["physical_delta_t"], num_steps=config.sampling.num_steps,
        )
        loss_endpoint = endpoint_rmsd_loss(rollout_final, x1, residue_mask)
        losses["endpoint"] = loss_endpoint
        total = total + config.loss.lambda_endpoint * loss_endpoint

    losses["total"] = total
    return losses


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, config: Config, step: int, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, config: Config, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in val_loader:
        batch = _move_batch_to_device(batch, device)
        losses = compute_losses(model, batch, config)
        total_loss += losses["total"].item()
        num_batches += 1
    model.train()
    return total_loss / max(num_batches, 1)


def train(config: Config, config_save_path: Optional[Path] = None) -> DualGraphFlowModel:
    set_seed(config.train.seed)
    device = torch.device(config.train.device)

    model = DualGraphFlowModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.optim.lr, weight_decay=config.train.optim.weight_decay
    )
    scaler = torch.amp.GradScaler(enabled=config.train.amp)

    train_loader, val_loader = build_dataloaders(config)

    ckpt_dir = Path(config.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if config_save_path is not None:
        save_config(config, config_save_path)

    best_val_loss = float("inf")
    global_step = 0
    overfit_batch: Optional[Dict[str, Tensor]] = None

    model.train()
    for epoch in range(config.train.num_epochs):
        for batch in train_loader:
            if config.train.overfit_one_batch:
                if overfit_batch is None:
                    overfit_batch = _move_batch_to_device(batch, device)
                batch = overfit_batch
            else:
                batch = _move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.train.amp):
                losses = compute_losses(model, batch, config)
                total_loss = losses["total"]

            if not torch.isfinite(total_loss):
                logger.warning("Non-finite loss at step %d, skipping update: %s", global_step, total_loss.item())
                global_step += 1
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.optim.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            if global_step % config.train.log_every == 0:
                component_str = ", ".join(f"{k}={v.item():.6f}" for k, v in losses.items())
                logger.info("step %d epoch %d | %s | grad_norm=%.4f", global_step, epoch, component_str, grad_norm.item())

            if global_step % config.train.val_every == 0 and global_step > 0 and not config.train.overfit_one_batch:
                val_loss = evaluate(model, val_loader, config, device)
                logger.info("step %d | val_loss=%.6f", global_step, val_loss)
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, config, global_step, best_val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(ckpt_dir / "best.pt", model, optimizer, config, global_step, best_val_loss)

            global_step += 1
            if config.train.max_steps is not None and global_step >= config.train.max_steps:
                break
        if config.train.max_steps is not None and global_step >= config.train.max_steps:
            break

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, config, global_step, best_val_loss)
    return model
