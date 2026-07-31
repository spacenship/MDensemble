"""Single-GPU / multi-GPU (DDP) helpers.

Every function here is safe to call when distributed training is *not*
active: they degrade to the obvious single-process answers (rank 0, world
size 1, no-op barriers). That keeps the training loop free of
``if distributed:`` branches.

Launch multi-GPU runs with ``torchrun``, which sets the ``RANK``,
``LOCAL_RANK`` and ``WORLD_SIZE`` environment variables this module reads:

    torchrun --nproc_per_node=2 train.py --config configs/mdcath_full.yaml
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


def is_torchrun_launch() -> bool:
    """True when the process was started by ``torchrun`` (or equivalent)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def is_distributed() -> bool:
    """True when a process group is initialised and spans more than one rank."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    """This process's global rank.

    Falls back to the ``RANK`` environment variable so it is still correct
    *before* :func:`setup_distributed` has initialised the process group --
    logging is configured at that point, and without the fallback every rank
    would report itself as rank 0.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main_process() -> bool:
    """Only this process should log, save checkpoints, or write files."""
    return get_rank() == 0


def setup_distributed(device_type: str = "cuda") -> torch.device:
    """Initialise the process group if launched under torchrun.

    Returns the device this rank should use. Safe (and a no-op beyond
    picking a device) for ordinary single-process runs.
    """
    if not is_torchrun_launch():
        return torch.device(device_type)

    backend = "nccl" if device_type == "cuda" and dist.is_nccl_available() else "gloo"

    if device_type == "cuda":
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_type)

    if not dist.is_initialized():
        # Binding the device up front lets collectives (notably barrier) pick
        # the right one instead of guessing from the ambient context.
        if backend == "nccl":
            dist.init_process_group(backend=backend, device_id=device)
        else:
            dist.init_process_group(backend=backend)
    return device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronise all ranks; no-op when not distributed."""
    if is_distributed():
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    """The underlying module, unwrapping DistributedDataParallel if present.

    Checkpoints must store the unwrapped ``state_dict`` so they can be
    loaded by a single-process run, and evaluation must call the unwrapped
    module so that a rank-0-only forward pass does not hang waiting for
    gradient synchronisation from the other ranks.
    """
    return getattr(model, "module", model)


def all_reduce_mean(value: float, device: Optional[torch.device] = None) -> float:
    """Average a scalar across ranks (identity when not distributed)."""
    if not is_distributed():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / get_world_size())


def broadcast_object(obj, source_rank: int = 0):
    """Share a Python object from ``source_rank`` to every rank.

    Used so that non-zero ranks learn validation results computed on rank 0
    and stay in step with learning-rate scheduling and early-stopping
    decisions.
    """
    if not is_distributed():
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=source_rank)
    return holder[0]
