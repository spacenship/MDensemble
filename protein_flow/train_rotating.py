"""Training over more mdCATH shards than fit on disk, a chunk at a time.

The ordinary loop in :mod:`protein_flow.train` assumes the dataset is sitting
in a directory. The full mdCATH dataset is 5,398 domains / 3.61 TB, so this
loop instead walks a frozen rotation plan
(:mod:`protein_flow.data.shard_manifest`): download a chunk, train
``steps_per_chunk`` optimizer steps on it, delete it, repeat -- with the next
chunk downloading in the background throughout.

Everything except the outer loop is shared with the single-pass trainer: the
same ``training_step``, ``validate_and_checkpoint``, loader builders and
dataset options, so the two cannot drift apart.

Three things are deliberate:

* **The validation set never rotates.** It is a fixed holdout, downloaded
  once and kept resident, because ReduceLROnPlateau and best-checkpoint
  selection compare validation loss across chunks -- meaningless if the
  measuring stick changes with the data.
* **Training loss jumps at chunk boundaries.** That is the data changing, not
  the model regressing; judge the run by the fixed validation set.
* **The run is resumable.** Each checkpoint records the rotation cursor, so
  an interrupted run continues at the chunk it reached instead of
  re-downloading terabytes. Resumption restarts the *current* chunk from its
  first step; at most ``steps_per_chunk`` steps are ever repeated.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from protein_flow.config import Config, save_config
from protein_flow.data.shard_manifest import ShardManifest
from protein_flow.data.shard_rotation import ShardPool, free_gb
from protein_flow.distributed import (
    barrier,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    enable_hang_diagnostics,
    setup_distributed,
    unwrap_model,
)
from protein_flow.models.dual_graph_flow import DualGraphFlowModel
from protein_flow.train import (
    needs_grad_scaler,
    TrainingProgress,
    _build_parameter_groups,
    _make_train_loader,
    _make_val_loader,
    _move_batch_to_device,
    load_checkpoint,
    mdcath_dataset_kwargs,
    save_checkpoint,
    set_seed,
    training_step,
    validate_and_checkpoint,
)

logger = logging.getLogger(__name__)


def _resolve_resume_path(config: Config) -> Optional[Path]:
    """``auto`` means {ckpt_dir}/last.pt when it exists; a path means that file."""
    resume = config.train.resume
    if not resume:
        return None
    if resume == "auto":
        candidate = Path(config.train.ckpt_dir) / "last.pt"
        return candidate if candidate.exists() else None
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"train.resume points at a missing checkpoint: {path}")
    return path


def chunk_step_budget(rotation_cfg, batches_per_pass: int) -> int:
    """How many optimizer steps this chunk gets.

    ``passes_per_chunk`` is expressed in whole passes over the resident data,
    so the budget lands exactly on a pass boundary no matter what the pass
    length turns out to be. ``steps_per_chunk`` is the raw alternative: it
    pins wall-clock (useful for staying ahead of the prefetch) but truncates
    the final pass whenever it is not a multiple of the pass length.
    """
    if rotation_cfg.passes_per_chunk is not None:
        if rotation_cfg.passes_per_chunk < 1:
            raise ValueError("data.rotation.passes_per_chunk must be >= 1 when set")
        return max(rotation_cfg.passes_per_chunk * batches_per_pass, 1)
    return rotation_cfg.steps_per_chunk


def _build_dataset(
    config: Config, data_dir: Path, h5_files: List[Path], *, seed: int, is_validation: bool
):
    from protein_flow.data.mdcath import MdCathDataset  # local import: h5py is optional

    data_cfg = config.data
    return MdCathDataset(
        data_dir,
        data_cfg,
        h5_files=h5_files,
        seed=seed,
        pairs_per_trajectory=(
            data_cfg.mdcath_val_pairs_per_trajectory
            if is_validation
            else data_cfg.mdcath_train_pairs_per_trajectory
        ),
        resample_each_epoch=(False if is_validation else data_cfg.mdcath_resample_train_each_epoch),
        **mdcath_dataset_kwargs(config),
    )


def train_rotating(config: Config, config_save_path: Optional[Path] = None) -> DualGraphFlowModel:
    rotation_cfg = config.data.rotation
    if not rotation_cfg.manifest_path:
        raise ValueError(
            "data.rotation.manifest_path must be set. Build one with:\n"
            "  python scripts/build_mdcath_manifest.py --output configs/manifests/mdcath_all.json"
        )
    if rotation_cfg.steps_per_chunk < 1:
        raise ValueError("data.rotation.steps_per_chunk must be >= 1")
    if rotation_cfg.num_cycles < 1:
        raise ValueError("data.rotation.num_cycles must be >= 1")

    if config.data.source != "mdcath":
        raise ValueError("data.source must be 'mdcath' when data.rotation.enabled is true")

    manifest = ShardManifest.load(rotation_cfg.manifest_path)
    if rotation_cfg.local_dir:
        local_dir = Path(rotation_cfg.local_dir)
    elif config.data.mdcath_dir:
        # Shards land in {local_dir}/data/, mirroring the repo layout, so
        # mdcath_dir is that subdirectory.
        local_dir = Path(config.data.mdcath_dir).parent
    else:
        raise ValueError("Set data.rotation.local_dir (or data.mdcath_dir) for a rotating run")
    data_dir = local_dir / "data"

    enable_hang_diagnostics()
    device = setup_distributed(
        torch.device(config.train.device).type,
        timeout_minutes=config.train.dist_timeout_minutes,
    )
    # Different seed per rank so frame-pair sampling and tau draws are
    # decorrelated across ranks; DDP broadcasts the initial weights anyway.
    set_seed(config.train.seed + get_rank())

    model = DualGraphFlowModel(config).to(device)
    if is_distributed():
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
    scaler = torch.amp.GradScaler(enabled=needs_grad_scaler(config))

    ckpt_dir = Path(config.train.ckpt_dir)
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if config_save_path is not None:
            save_config(config, config_save_path)

    progress = TrainingProgress()
    start_cursor = 0
    resume_path = _resolve_resume_path(config)
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, unwrap_model(model), optimizer, scheduler, scaler)
        progress.global_step = int(checkpoint.get("step", 0))
        progress.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        progress.best_endpoint_rmsd = float(checkpoint.get("best_endpoint_rmsd", float("inf")))
        start_cursor = int((checkpoint.get("extra") or {}).get("rotation_cursor", 0))
        logger.info(
            "Resumed from %s at cursor %d (step %d, best_val_loss %.6f)",
            resume_path, start_cursor, progress.global_step, progress.best_val_loss,
        )

    pool = ShardPool(
        manifest,
        local_dir,
        min_free_gb=rotation_cfg.min_free_gb,
        retries=rotation_cfg.download_retries,
        verify=rotation_cfg.verify_downloads,
        download_timeout_seconds=(
            rotation_cfg.download_timeout_minutes * 60.0
            if rotation_cfg.download_timeout_minutes > 0
            else None
        ),
        rank=get_rank(),
        world_size=get_world_size(),
    )
    total_cursors = rotation_cfg.num_cycles * manifest.num_chunks
    if is_main_process():
        logger.info("Rotation plan: %s", manifest.summary())
        # Only steps_per_chunk is known here. Under passes_per_chunk -- which
        # takes precedence, see chunk_step_budget -- the budget depends on the
        # pass length, which is not known until the chunk's dataset is built,
        # so reporting steps_per_chunk in that case quotes a number the run
        # will never use (2,000 against an actual 2,496, a 25% undercount that
        # is easy to plan against by mistake). Say what is actually known.
        if rotation_cfg.passes_per_chunk is not None:
            budget = f"{rotation_cfg.passes_per_chunk} pass(es)/chunk (steps resolved per chunk)"
            total = "total steps depend on pass length"
        else:
            budget = f"{rotation_cfg.steps_per_chunk} steps/chunk"
            total = f"{total_cursors * rotation_cfg.steps_per_chunk} steps total"
        logger.info(
            "%d cycle(s) x %d chunk(s) x %s = %s; %.0f GB free at %s",
            rotation_cfg.num_cycles, manifest.num_chunks, budget, total,
            free_gb(local_dir), local_dir,
        )

    # The validation holdout is fetched once and never released.
    pool.ensure_val()
    barrier()
    val_files = pool.val_files()
    if not val_files:
        raise RuntimeError(
            f"No validation shards available under {local_dir}; check network access and disk space."
        )
    val_loader = _make_val_loader(
        _build_dataset(config, data_dir, val_files, seed=config.data.seed + 1, is_validation=True),
        config,
    )
    logger.info("Validation set: %d shard(s), %d trajectory pair(s)", len(val_files), len(val_loader.dataset))

    epoch = 0  # a monotonic pass counter: reseeds frame offsets and reshuffles
    model.train()
    reached_max_steps = False
    # Where a resumed run should pick up. Only ever advanced past a chunk
    # that actually finished its step budget, so a run stopped part-way
    # through a chunk redoes that chunk instead of silently skipping it.
    next_cursor = start_cursor
    try:
        for cursor in range(start_cursor, total_cursors):
            cycle, chunk_index = divmod(cursor, manifest.num_chunks)
            pool.ensure(chunk_index)
            barrier()

            # Every rank derives this from the shared filesystem, so all ranks
            # agree on the dataset even though each downloaded only a slice.
            chunk_files = pool.resident_files(chunk_index)
            if not chunk_files:
                # Almost always a transient network or disk problem. The
                # cursor is not advanced past it, so a later resume retries
                # this chunk rather than silently dropping 200 domains.
                logger.error(
                    "Chunk %d has no usable shards; skipping it (a resumed run will retry it)",
                    chunk_index,
                )
                continue

            for ahead in range(1, rotation_cfg.prefetch_chunks + 1):
                if cursor + ahead < total_cursors:
                    pool.prefetch((cursor + ahead) % manifest.num_chunks)

            train_dataset = _build_dataset(
                config, data_dir, chunk_files, seed=config.data.seed, is_validation=False
            )
            train_loader = _make_train_loader(train_dataset, config)
            if is_main_process():
                logger.info(
                    "cycle %d/%d, chunk %d/%d (cursor %d): %d shard(s), %d trajectory pair(s), "
                    "%.0f GB free",
                    cycle + 1, rotation_cfg.num_cycles, chunk_index + 1, manifest.num_chunks,
                    cursor, len(chunk_files), len(train_dataset), free_gb(local_dir),
                )

            if len(train_loader) == 0:
                # Possible under DDP, where drop_last=True empties a loader
                # holding fewer than world_size * batch_size samples. Without
                # this guard the step-budget loop below would spin forever on
                # a pass that can never produce a step.
                logger.error(
                    "Chunk %d yields no batches (%d pair(s), batch_size %d, world size %d); skipping it",
                    chunk_index, len(train_dataset), config.data.batch_size, get_world_size(),
                )
                continue

            step_budget = chunk_step_budget(rotation_cfg, len(train_loader))
            if is_main_process():
                logger.info(
                    "chunk %d budget: %d step(s) = %.2f pass(es) of %d",
                    chunk_index, step_budget, step_budget / len(train_loader), len(train_loader),
                )

            steps_in_chunk = 0
            while steps_in_chunk < step_budget and not reached_max_steps:
                # Two notions of epoch, both needed: the sampler's (reshuffles
                # the per-rank assignment) and the dataset's (draws fresh
                # frame pairs), so repeated passes over a resident chunk are
                # not repeats of the same pairs.
                if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
                set_epoch = getattr(train_loader.dataset, "set_epoch", None)
                if set_epoch is not None:
                    set_epoch(epoch)
                epoch += 1

                for batch in train_loader:
                    batch = _move_batch_to_device(batch, device)
                    # The return value (whether the step was skipped for a
                    # non-finite loss) deliberately does not gate validation:
                    # a skip depends on that rank's data, and validation is a
                    # collective. Keying it on global_step alone keeps every
                    # rank entering it together.
                    training_step(model, batch, config, optimizer, scaler, device, progress, epoch)

                    if progress.global_step % config.train.val_every == 0 and progress.global_step > 0:
                        validate_and_checkpoint(
                            model, val_loader, config, device, optimizer, scheduler, progress,
                            ckpt_dir, epoch, extra={"rotation_cursor": cursor}, scaler=scaler,
                        )

                    progress.global_step += 1
                    steps_in_chunk += 1
                    if config.train.max_steps is not None and progress.global_step >= config.train.max_steps:
                        reached_max_steps = True
                        break
                    if steps_in_chunk >= step_budget:
                        break

            chunk_completed = steps_in_chunk >= step_budget
            next_cursor = cursor + 1 if chunk_completed else cursor

            # Free the chunk only once every rank has stopped reading it.
            barrier()
            del train_loader, train_dataset
            if (
                chunk_completed
                and rotation_cfg.delete_after_use
                and chunk_index not in rotation_cfg.keep_resident
            ):
                pool.release(chunk_index)
            elif not chunk_completed:
                # Keep it: this is exactly the chunk a resumed run needs first,
                # and re-downloading 100+ GB to redo it would be wasteful.
                logger.info("Keeping chunk %d resident: it did not finish its step budget", chunk_index)
            if is_main_process():
                save_checkpoint(
                    ckpt_dir / "last.pt", unwrap_model(model), optimizer, config, progress.global_step,
                    progress.best_val_loss, scheduler, progress.best_endpoint_rmsd,
                    progress.latest_val_metrics, extra={"rotation_cursor": next_cursor},
                    scaler=scaler,
                )
            if reached_max_steps:
                logger.info("Reached train.max_steps=%s; stopping rotation", config.train.max_steps)
                break
    finally:
        pool.stop_prefetch()

    if is_main_process():
        save_checkpoint(
            ckpt_dir / "last.pt", unwrap_model(model), optimizer, config, progress.global_step,
            progress.best_val_loss, scheduler, progress.best_endpoint_rmsd, progress.latest_val_metrics,
            extra={"rotation_cursor": next_cursor}, scaler=scaler,
        )
    barrier()
    cleanup_distributed()
    return unwrap_model(model)
