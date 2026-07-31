"""Training loop for the dual-graph conditional-flow-matching model."""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler

from protein_flow.config import Config, EndpointRolloutConfig, save_config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.synthetic import SyntheticProteinTrajectoryDataset
from protein_flow.distributed import (
    barrier,
    broadcast_object,
    cleanup_distributed,
    get_rank,
    is_distributed,
    is_main_process,
    setup_distributed,
    unwrap_model,
)
from protein_flow.flow.paths import build_flow_path, sample_tau
from protein_flow.geometry.kabsch import masked_kabsch_align
from protein_flow.losses.flow_matching import flow_matching_loss
from protein_flow.losses.physics import (
    compute_all_atom_physics_losses,
    compute_physics_losses,
    endpoint_rmsd_loss,
)
from protein_flow.models.dual_graph_flow import DualGraphFlowModel

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_train_loader(dataset, config: Config) -> DataLoader:
    """Training loader, sharded across ranks when running under DDP.

    ``drop_last=True`` is required for correctness under DDP, not just
    tidiness: every rank must execute the same number of backward passes or
    the gradient all-reduce deadlocks on the rank that runs out of batches
    first.
    """
    sampler = (
        DistributedSampler(dataset, shuffle=True, drop_last=True) if is_distributed() else None
    )
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=is_distributed(),
        num_workers=config.data.num_workers,
        collate_fn=collate_protein_batch,
    )


def _make_val_loader(dataset, config: Config) -> DataLoader:
    """Validation loader. Deliberately *not* sharded: validation runs on rank
    0 only (over the whole set) so the reported metrics are identical to a
    single-GPU run and no cross-rank metric reduction is needed."""
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_protein_batch,
    )


def _build_synthetic_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    train_dataset = SyntheticProteinTrajectoryDataset(config.data, size=config.data.train_size, seed=config.data.seed)
    val_dataset = SyntheticProteinTrajectoryDataset(
        config.data, size=config.data.val_size, seed=config.data.seed + 1
    )
    return _make_train_loader(train_dataset, config), _make_val_loader(val_dataset, config)


def _build_mdcath_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    from protein_flow.data.mdcath import MdCathDataset  # local import: h5py is an optional dependency

    data_cfg = config.data
    if not data_cfg.mdcath_dir:
        raise ValueError("data.mdcath_dir must be set when data.source == 'mdcath'")

    all_files = sorted(Path(data_cfg.mdcath_dir).glob("*.h5"))
    if not all_files:
        raise FileNotFoundError(f"No .h5 shards found under {data_cfg.mdcath_dir}")

    # Split by domain (file), not by trajectory, so the same domain never
    # leaks between train and val.
    rng = random.Random(data_cfg.seed)
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    num_val = max(1, int(len(shuffled) * data_cfg.mdcath_val_fraction))
    val_files, train_files = shuffled[:num_val], shuffled[num_val:]

    common_kwargs = dict(
        frame_gap=data_cfg.mdcath_frame_gap,
        sampling_max_frame_gap=data_cfg.mdcath_sampling_max_frame_gap,
        ps_per_frame=data_cfg.mdcath_ps_per_frame,
        embedding_cache_dir=data_cfg.mdcath_embedding_cache_dir,
        representation=data_cfg.representation,
        # With in-graph ESM the dataset emits token ids instead of relying on
        # the precomputed embedding cache.
        esm_tokenizer_name=config.model.esm.model_name if config.model.esm.enabled else None,
    )
    train_dataset = MdCathDataset(
        data_cfg.mdcath_dir, data_cfg, h5_files=train_files, seed=data_cfg.seed,
        pairs_per_trajectory=data_cfg.mdcath_train_pairs_per_trajectory,
        resample_each_epoch=data_cfg.mdcath_resample_train_each_epoch,
        **common_kwargs,
    )
    val_dataset = MdCathDataset(
        data_cfg.mdcath_dir, data_cfg, h5_files=val_files, seed=data_cfg.seed + 1,
        pairs_per_trajectory=data_cfg.mdcath_val_pairs_per_trajectory,
        resample_each_epoch=False,
        **common_kwargs,
    )

    return _make_train_loader(train_dataset, config), _make_val_loader(val_dataset, config)


def build_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    if config.data.source == "mdcath":
        return _build_mdcath_dataloaders(config)
    if config.data.source == "synthetic":
        return _build_synthetic_dataloaders(config)
    raise ValueError(f"Unknown data.source: {config.data.source!r}")


def _build_parameter_groups(model: nn.Module, config: Config):
    """Give the pretrained PLM its own (much smaller) learning rate.

    Training a pretrained ESM at the same rate as the randomly-initialised
    flow network typically destroys the PLM's representations within a few
    hundred steps, so the two get separate parameter groups whenever
    in-graph fine-tuning is enabled.
    """
    esm_parameters, other_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (esm_parameters if name.startswith("esm_encoder.") else other_parameters).append(parameter)

    groups = [{"params": other_parameters, "lr": config.train.optim.lr}]
    if esm_parameters:
        groups.append({"params": esm_parameters, "lr": config.model.esm.learning_rate})
    return groups


def _move_batch_to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _differentiable_rollout(
    model: nn.Module,
    x0: Tensor,
    seq_emb: Tensor,
    residue_types: Tensor,
    mask: Tensor,
    temperature: Tensor,
    physical_delta_t: Tensor,
    rollout_config: EndpointRolloutConfig,
    **atom_inputs: Tensor,
) -> Tensor:
    """Gradient-carrying ODE rollout for the endpoint loss.

    Unlike :func:`protein_flow.flow.solver.integrate_ode` (which runs under
    ``no_grad`` for pure sampling) this stays differentiable, so it is by far
    the most expensive term in the objective: every step is a full model
    evaluation that must be backpropagated through. See
    :class:`~protein_flow.config.EndpointRolloutConfig` for the knobs that
    keep it tractable -- a small step count, per-step gradient checkpointing,
    and optional truncated backpropagation.
    """
    if rollout_config.solver not in ("euler", "heun"):
        raise ValueError(f"Unknown rollout solver: {rollout_config.solver!r}")
    num_steps = rollout_config.num_steps
    if num_steps < 1:
        raise ValueError("endpoint_rollout.num_steps must be >= 1")

    device, dtype = x0.device, x0.dtype
    batch_size = x0.shape[0]
    dtau = 1.0 / num_steps

    first_grad_step = 0
    if rollout_config.backprop_last_steps is not None:
        first_grad_step = max(num_steps - rollout_config.backprop_last_steps, 0)

    def take_step(x: Tensor, tau_start_value: Tensor) -> Tensor:
        tau_batch = tau_start_value.expand(batch_size)
        velocity = model(
            x, tau_batch, seq_emb, residue_types, temperature, physical_delta_t, mask, **atom_inputs
        )
        if rollout_config.solver == "euler":
            return x + dtau * velocity
        x_euler = x + dtau * velocity
        tau_end = (tau_start_value + dtau).expand(batch_size)
        velocity_end = model(
            x_euler, tau_end, seq_emb, residue_types, temperature, physical_delta_t, mask, **atom_inputs
        )
        return x + dtau * 0.5 * (velocity + velocity_end)

    x = x0
    for step in range(num_steps):
        if step == first_grad_step and step > 0:
            # Truncated BPTT: everything before this point contributed to the
            # trajectory but is cut out of the backward graph.
            x = x.detach()
        tau_value = torch.tensor(step * dtau, device=device, dtype=dtype)

        if step < first_grad_step:
            with torch.no_grad():
                x = take_step(x, tau_value)
        elif rollout_config.gradient_checkpointing and torch.is_grad_enabled():
            x = torch.utils.checkpoint.checkpoint(take_step, x, tau_value, use_reentrant=False)
        else:
            x = take_step(x, tau_value)
    return x


def compute_losses(
    model: nn.Module,
    batch: Dict[str, Tensor],
    config: Config,
    tau: Optional[Tensor] = None,
    diagnostics: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    """Runs one forward pass and returns a dict of loss components (including 'total')."""
    source_coords = batch["source_coords"]
    target_coords = batch["target_coords"]
    residue_mask = batch["residue_mask"]

    is_all_atom = config.data.representation == "heavy_atom"
    # In all-atom mode the flowing particles are atoms, so every
    # particle-level operation (Kabsch, flow matching, physics) must use the
    # atom mask; residue_mask stays for the residue-level sequence encoder.
    particle_mask = batch["atom_mask"] if is_all_atom else residue_mask
    atom_inputs = (
        {
            "atom_mask": batch["atom_mask"],
            "atom_residue_index": batch["atom_residue_index"],
            "atom_element": batch["atom_element"],
            "ca_atom_index": batch["ca_atom_index"],
        }
        if is_all_atom
        else {}
    )
    if "esm_input_ids" in batch:
        atom_inputs["esm_input_ids"] = batch["esm_input_ids"]
        atom_inputs["esm_attention_mask"] = batch["esm_attention_mask"]

    kabsch_result = masked_kabsch_align(source_coords, target_coords, particle_mask)
    x0 = source_coords
    x1 = kabsch_result.aligned_target

    flow_path = build_flow_path(config.flow.path_type, config.flow.sigma_min)
    if tau is None:
        tau = sample_tau(x0.shape[0], device=x0.device, dtype=x0.dtype)
    elif tau.shape != (x0.shape[0],):
        raise ValueError(f"tau must have shape ({x0.shape[0]},), got {tuple(tau.shape)}")
    x_tau, target_velocity = flow_path.sample(x0, x1, tau)

    predicted_velocity = model(
        x_tau, tau, batch["sequence_embedding"], batch["residue_types"],
        batch["temperature"], batch["physical_delta_t"], residue_mask, **atom_inputs,
    )

    loss_fm = flow_matching_loss(predicted_velocity, target_velocity, particle_mask)
    if diagnostics is not None:
        mask = particle_mask.to(dtype=predicted_velocity.dtype)
        residue_count = mask.sum(dim=1).clamp(min=1.0)
        diagnostics["fm_per_sample"] = (
            (predicted_velocity - target_velocity).pow(2).sum(dim=-1) * mask
        ).sum(dim=1) / residue_count
        diagnostics["zero_fm"] = flow_matching_loss(
            torch.zeros_like(target_velocity), target_velocity, particle_mask
        )
    losses = {"fm": loss_fm}
    total = config.loss.lambda_fm * loss_fm

    physics_cfg = config.loss.physics
    if physics_cfg.enabled:
        reference_coords = x0 if physics_cfg.d_ref_source == "source" else x1
        if physics_cfg.apply_to == "x_tau":
            pred_coords_for_physics = x_tau
        else:
            pred_coords_for_physics = x_tau + physics_cfg.step_scale * predicted_velocity

        if is_all_atom:
            physics_out = compute_all_atom_physics_losses(
                pred_coords_for_physics, reference_coords, particle_mask,
                batch["bond_index"], batch["bond_mask"],
                batch["angle_index"], batch["angle_mask"],
                config.model.graph, physics_cfg.clash_threshold_heavy_atom,
            )
        else:
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
        rollout_final = _differentiable_rollout(
            model, x0, batch["sequence_embedding"], batch["residue_types"], residue_mask,
            batch["temperature"], batch["physical_delta_t"], config.loss.endpoint_rollout,
            **atom_inputs,
        )
        loss_endpoint = endpoint_rmsd_loss(rollout_final, x1, particle_mask)
        losses["endpoint"] = loss_endpoint
        total = total + config.loss.lambda_endpoint * loss_endpoint

        if config.loss.endpoint_physics_enabled:
            # Straight-line interpolation can pass through non-physical
            # geometry; penalizing the rollout endpoint is what actually
            # constrains the generated structure rather than the interpolant.
            if is_all_atom:
                endpoint_physics = compute_all_atom_physics_losses(
                    rollout_final, x0, particle_mask,
                    batch["bond_index"], batch["bond_mask"],
                    batch["angle_index"], batch["angle_mask"],
                    config.model.graph, physics_cfg.clash_threshold_heavy_atom,
                )
            else:
                endpoint_physics = compute_physics_losses(
                    rollout_final, x0, residue_mask, config.model.graph,
                    physics_cfg.clash_threshold, physics_cfg.clash_seq_sep,
                )
            endpoint_physics_total = (
                config.loss.lambda_bond * endpoint_physics.bond
                + config.loss.lambda_angle * endpoint_physics.angle
                + config.loss.lambda_clash * endpoint_physics.clash
            )
            losses["endpoint_bond"] = endpoint_physics.bond
            losses["endpoint_angle"] = endpoint_physics.angle
            losses["endpoint_clash"] = endpoint_physics.clash
            total = total + config.loss.lambda_endpoint_physics * endpoint_physics_total

    losses["total"] = total
    return losses


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    step: int,
    best_val_loss: float,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    best_endpoint_rmsd: float = float("inf"),
    validation_metrics: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "best_endpoint_rmsd": best_endpoint_rmsd,
            "validation_metrics": validation_metrics,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
) -> Dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


@torch.no_grad()
def evaluate_detailed(
    model: nn.Module, val_loader: DataLoader, config: Config, device: torch.device
) -> Dict[str, object]:
    """Deterministically evaluate a model over a fixed flow-time grid.

    Scalar loss components preserve the historical batch-mean aggregation.
    Temperature-stratified FM values are equal-weighted per trajectory pair,
    preventing long proteins from silently dominating those diagnostics.
    """
    tau_values = config.train.val_tau_values
    if not tau_values:
        raise ValueError("train.val_tau_values must contain at least one value")
    if any(value < 0.0 or value > 1.0 for value in tau_values):
        raise ValueError("train.val_tau_values must all lie in [0, 1]")

    model.eval()
    component_sums: Dict[str, float] = defaultdict(float)
    temperature_fm_sums: Dict[float, float] = defaultdict(float)
    temperature_counts: Dict[float, int] = defaultdict(int)
    zero_fm_sum = 0.0
    endpoint_source_rmsd_sum = 0.0
    endpoint_generated_rmsd_sum = 0.0
    endpoint_wins = 0
    endpoint_sample_count = 0
    endpoint_component_sums: Dict[str, float] = defaultdict(float)
    endpoint_temperature_source: Dict[float, float] = defaultdict(float)
    endpoint_temperature_generated: Dict[float, float] = defaultdict(float)
    endpoint_temperature_counts: Dict[float, int] = defaultdict(int)
    endpoint_batch_count = 0
    endpoint_batch_indices: set[int] = set()
    if config.train.val_endpoint_enabled:
        if config.train.val_endpoint_num_steps < 1:
            raise ValueError("train.val_endpoint_num_steps must be >= 1")
        if config.train.val_endpoint_max_batches < 1:
            raise ValueError("train.val_endpoint_max_batches must be >= 1")
        endpoint_rng = random.Random(config.train.seed + 91_273)
        endpoint_batch_indices = set(
            endpoint_rng.sample(
                range(len(val_loader)), min(config.train.val_endpoint_max_batches, len(val_loader))
            )
        )

    max_batches = config.train.val_max_batches
    if max_batches is not None and max_batches < 1:
        raise ValueError("train.val_max_batches must be >= 1 when set")

    num_batches = 0
    for batch_index, batch in enumerate(val_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move_batch_to_device(batch, device)
        for tau_index, tau_value in enumerate(tau_values):
            tau = torch.full(
                (batch["source_coords"].shape[0],), float(tau_value),
                device=device, dtype=batch["source_coords"].dtype,
            )
            diagnostics: Dict[str, Tensor] = {}
            losses = compute_losses(model, batch, config, tau=tau, diagnostics=diagnostics)
            for name, value in losses.items():
                component_sums[name] += value.item()

            if config.data.source == "mdcath":
                temperatures = batch["temperature"].squeeze(-1)
                for temperature in torch.unique(temperatures):
                    selection = temperatures == temperature
                    key = float(temperature.item())
                    temperature_fm_sums[key] += diagnostics["fm_per_sample"][selection].sum().item()
                    temperature_counts[key] += int(selection.sum().item())

            if tau_index == 0:
                zero_fm_sum += diagnostics["zero_fm"].item()

        if batch_index in endpoint_batch_indices:
            is_all_atom = config.data.representation == "heavy_atom"
            particle_mask = batch["atom_mask"] if is_all_atom else batch["residue_mask"]
            atom_inputs = (
                {
                    "atom_mask": batch["atom_mask"],
                    "atom_residue_index": batch["atom_residue_index"],
                    "atom_element": batch["atom_element"],
                    "ca_atom_index": batch["ca_atom_index"],
                }
                if is_all_atom
                else {}
            )
            if "esm_input_ids" in batch:
                atom_inputs["esm_input_ids"] = batch["esm_input_ids"]
                atom_inputs["esm_attention_mask"] = batch["esm_attention_mask"]
            kabsch_result = masked_kabsch_align(
                batch["source_coords"], batch["target_coords"], particle_mask
            )
            generated, _ = model.sample(
                batch["source_coords"], batch["sequence_embedding"], batch["residue_types"],
                batch["residue_mask"], batch["temperature"], batch["physical_delta_t"],
                num_steps=config.train.val_endpoint_num_steps,
                solver=config.train.val_endpoint_solver,
                return_trajectory=False,
                **atom_inputs,
            )
            mask = particle_mask.to(generated.dtype)
            residue_count = mask.sum(dim=1).clamp(min=1.0)
            generated_squared_distance = (
                (generated - kabsch_result.aligned_target).pow(2).sum(dim=-1) * mask
            )
            generated_rmsd = torch.sqrt(
                generated_squared_distance.sum(dim=1) / residue_count + 1e-8
            )
            source_rmsd = kabsch_result.post_rmsd
            endpoint_source_rmsd_sum += source_rmsd.sum().item()
            endpoint_generated_rmsd_sum += generated_rmsd.sum().item()
            endpoint_wins += int((generated_rmsd < source_rmsd).sum().item())
            endpoint_sample_count += int(generated_rmsd.numel())

            if is_all_atom:
                endpoint_physics = compute_all_atom_physics_losses(
                    generated, batch["source_coords"], particle_mask,
                    batch["bond_index"], batch["bond_mask"],
                    batch["angle_index"], batch["angle_mask"],
                    config.model.graph, config.loss.physics.clash_threshold_heavy_atom,
                )
            else:
                endpoint_physics = compute_physics_losses(
                    generated, batch["source_coords"], batch["residue_mask"], config.model.graph,
                    config.loss.physics.clash_threshold, config.loss.physics.clash_seq_sep,
                )
            endpoint_component_sums["bond"] += endpoint_physics.bond.item()
            endpoint_component_sums["angle"] += endpoint_physics.angle.item()
            endpoint_component_sums["clash"] += endpoint_physics.clash.item()
            endpoint_batch_count += 1

            if config.data.source == "mdcath":
                temperatures = batch["temperature"].squeeze(-1)
                for temperature in torch.unique(temperatures):
                    selection = temperatures == temperature
                    key = float(temperature.item())
                    endpoint_temperature_source[key] += source_rmsd[selection].sum().item()
                    endpoint_temperature_generated[key] += generated_rmsd[selection].sum().item()
                    endpoint_temperature_counts[key] += int(selection.sum().item())
        num_batches += 1
    model.train()

    denominator = max(num_batches * len(tau_values), 1)
    metrics: Dict[str, object] = {
        name: value / denominator for name, value in component_sums.items()
    }
    metrics["zero_fm"] = zero_fm_sum / max(num_batches, 1)
    zero_fm = float(metrics["zero_fm"])
    fm = float(metrics.get("fm", 0.0))
    metrics["fm_improvement_pct"] = 100.0 * (zero_fm - fm) / max(zero_fm, 1e-12)
    metrics["fm_by_temperature"] = {
        temperature: temperature_fm_sums[temperature] / temperature_counts[temperature]
        for temperature in sorted(temperature_fm_sums)
    }
    if endpoint_sample_count:
        source_rmsd = endpoint_source_rmsd_sum / endpoint_sample_count
        generated_rmsd = endpoint_generated_rmsd_sum / endpoint_sample_count
        metrics["endpoint_source_rmsd"] = source_rmsd
        metrics["endpoint_generated_rmsd"] = generated_rmsd
        metrics["endpoint_improvement_pct"] = 100.0 * (source_rmsd - generated_rmsd) / max(source_rmsd, 1e-12)
        metrics["endpoint_win_rate_pct"] = 100.0 * endpoint_wins / endpoint_sample_count
        for name, value in endpoint_component_sums.items():
            metrics[f"endpoint_{name}"] = value / endpoint_batch_count
        metrics["endpoint_by_temperature"] = {
            temperature: {
                "source_rmsd": endpoint_temperature_source[temperature] / endpoint_temperature_counts[temperature],
                "generated_rmsd": endpoint_temperature_generated[temperature] / endpoint_temperature_counts[temperature],
            }
            for temperature in sorted(endpoint_temperature_counts)
        }
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, config: Config, device: torch.device) -> float:
    """Backward-compatible scalar validation API."""
    return float(evaluate_detailed(model, val_loader, config, device)["total"])


def train(config: Config, config_save_path: Optional[Path] = None) -> DualGraphFlowModel:
    device = setup_distributed(torch.device(config.train.device).type)
    # Different seed per rank so the frame-pair sampling in MdCathDataset and
    # the tau draws are decorrelated across ranks; model init is broadcast by
    # DDP anyway, so this does not desynchronise the weights.
    set_seed(config.train.seed + get_rank())

    model = DualGraphFlowModel(config).to(device)
    if is_distributed():
        # static_graph lets DDP coexist with activation checkpointing: without
        # it, recomputation during backward marks a parameter ready twice and
        # DDP raises. The graph really is static here -- the same modules run
        # the same number of times every step.
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            static_graph=True,
        )
    optimizer = torch.optim.AdamW(
        _build_parameter_groups(model, config), weight_decay=config.train.optim.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.train.optim.plateau_factor,
        patience=config.train.optim.plateau_patience,
        min_lr=config.train.optim.min_lr,
    )
    scaler = torch.amp.GradScaler(enabled=config.train.amp)

    train_loader, val_loader = build_dataloaders(config)

    ckpt_dir = Path(config.train.ckpt_dir)
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if config_save_path is not None:
            save_config(config, config_save_path)

    best_val_loss = float("inf")
    best_endpoint_rmsd = float("inf")
    latest_val_metrics: Optional[Dict[str, object]] = None
    global_step = 0
    overfit_batch: Optional[Dict[str, Tensor]] = None

    model.train()
    for epoch in range(config.train.num_epochs):
        # Two distinct notions of "epoch" that both need setting:
        #   - the sampler's, which reshuffles the shard assignment per rank;
        #   - the dataset's, which draws fresh frame pairs per trajectory.
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        set_epoch = getattr(train_loader.dataset, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)
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

            if global_step % config.train.log_every == 0 and is_main_process():
                component_str = ", ".join(f"{k}={v.item():.6f}" for k, v in losses.items())
                logger.info("step %d epoch %d | %s | grad_norm=%.4f", global_step, epoch, component_str, grad_norm.item())

            if global_step % config.train.val_every == 0 and global_step > 0 and not config.train.overfit_one_batch:
                # Rank 0 evaluates the whole validation set alone, using the
                # unwrapped module: a DDP-wrapped forward would hang here,
                # since the other ranks are waiting at the barrier below and
                # would never join the gradient sync. Metrics are then
                # broadcast so every rank makes the same scheduler and
                # best-checkpoint decisions.
                if is_main_process():
                    val_metrics = evaluate_detailed(unwrap_model(model), val_loader, config, device)
                else:
                    val_metrics = None
                val_metrics = broadcast_object(val_metrics, source_rank=0)
                barrier()
                latest_val_metrics = val_metrics
                val_loss = float(val_metrics["total"])
                scheduler.step(val_loss)
                component_str = ", ".join(
                    f"{name}={float(val_metrics[name]):.6f}"
                    for name in ("fm", "bond", "angle", "clash")
                    if name in val_metrics
                )
                temperature_str = ", ".join(
                    f"{temperature:g}K={loss:.6f}"
                    for temperature, loss in val_metrics["fm_by_temperature"].items()
                )
                if is_main_process():
                    logger.info(
                        "step %d | val_loss=%.6f | %s | zero_fm=%.6f, fm_improvement=%.2f%% | "
                        "lr=%.3e | fm_by_temp: %s",
                        global_step, val_loss, component_str, float(val_metrics["zero_fm"]),
                        float(val_metrics["fm_improvement_pct"]), optimizer.param_groups[0]["lr"],
                        temperature_str,
                    )
                    if "endpoint_generated_rmsd" in val_metrics:
                        logger.info(
                            "step %d | endpoint_rmsd: source=%.6f, generated=%.6f, improvement=%.2f%%, "
                            "win_rate=%.2f%% | endpoint_physics: bond=%.6f, angle=%.6f, clash=%.6f",
                            global_step, float(val_metrics["endpoint_source_rmsd"]),
                            float(val_metrics["endpoint_generated_rmsd"]),
                            float(val_metrics["endpoint_improvement_pct"]),
                            float(val_metrics["endpoint_win_rate_pct"]),
                            float(val_metrics["endpoint_bond"]), float(val_metrics["endpoint_angle"]),
                            float(val_metrics["endpoint_clash"]),
                        )
                endpoint_rmsd = val_metrics.get("endpoint_generated_rmsd")
                endpoint_improved = endpoint_rmsd is not None and float(endpoint_rmsd) < best_endpoint_rmsd
                val_improved = val_loss < best_val_loss
                if endpoint_improved:
                    best_endpoint_rmsd = float(endpoint_rmsd)
                if val_improved:
                    best_val_loss = val_loss
                # Only rank 0 writes, and it writes the *unwrapped* model so
                # the checkpoint loads cleanly in a single-process run.
                if is_main_process():
                    if endpoint_improved:
                        save_checkpoint(
                            ckpt_dir / "best_endpoint.pt", unwrap_model(model), optimizer, config,
                            global_step, best_val_loss, scheduler, best_endpoint_rmsd, val_metrics,
                        )
                    if val_improved:
                        save_checkpoint(
                            ckpt_dir / "best.pt", unwrap_model(model), optimizer, config, global_step,
                            best_val_loss, scheduler, best_endpoint_rmsd, val_metrics,
                        )
                    save_checkpoint(
                        ckpt_dir / "last.pt", unwrap_model(model), optimizer, config, global_step,
                        best_val_loss, scheduler, best_endpoint_rmsd, val_metrics,
                    )

            global_step += 1
            if config.train.max_steps is not None and global_step >= config.train.max_steps:
                break
        if config.train.max_steps is not None and global_step >= config.train.max_steps:
            break

    if is_main_process():
        save_checkpoint(
            ckpt_dir / "last.pt", unwrap_model(model), optimizer, config, global_step,
            best_val_loss, scheduler, best_endpoint_rmsd, latest_val_metrics,
        )
    barrier()
    cleanup_distributed()
    return unwrap_model(model)
