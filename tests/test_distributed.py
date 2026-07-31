"""Multi-GPU (DDP) support.

Two layers of testing:

* Helper behaviour in the ordinary single-process case, which must degrade
  gracefully so the training loop needs no ``if distributed:`` branches.
* A real two-process run over the ``gloo`` CPU backend, which is where the
  actual DDP contract is checked: that gradients are all-reduced (parameters
  stay identical across ranks) and that the dataset is genuinely sharded.
  This spawns subprocesses, so it is skipped on platforms without ``fork``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from protein_flow.config import Config
from protein_flow.distributed import (
    all_reduce_mean,
    barrier,
    broadcast_object,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    is_torchrun_launch,
    unwrap_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Single-process degradation
# --------------------------------------------------------------------------
def test_helpers_degrade_to_single_process_defaults(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert is_torchrun_launch() is False
    assert is_distributed() is False
    assert get_rank() == 0
    assert get_local_rank() == 0
    assert get_world_size() == 1
    assert is_main_process() is True
    barrier()  # must be a no-op, not an error
    assert all_reduce_mean(3.5) == 3.5
    assert broadcast_object({"a": 1}) == {"a": 1}


def test_get_rank_reads_env_before_process_group_exists(monkeypatch):
    """Logging is configured before the process group is initialised, so the
    rank must come from the environment at that point -- otherwise every rank
    would label itself rank 0."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert get_rank() == 3
    assert is_torchrun_launch() is True
    assert is_main_process() is False


def test_unwrap_model_returns_inner_module():
    inner = torch.nn.Linear(2, 2)
    assert unwrap_model(inner) is inner

    class FakeDDP(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    assert unwrap_model(FakeDDP(inner)) is inner


def test_single_process_dataloader_has_no_distributed_sampler():
    from torch.utils.data import DistributedSampler

    from protein_flow.train import build_dataloaders

    config = Config()
    config.data.train_size = 8
    config.data.val_size = 4
    config.data.batch_size = 2
    train_loader, val_loader = build_dataloaders(config)
    assert not isinstance(train_loader.sampler, DistributedSampler)
    assert not isinstance(val_loader.sampler, DistributedSampler)
    assert train_loader.drop_last is False


# --------------------------------------------------------------------------
# Real two-process DDP over gloo
# --------------------------------------------------------------------------
DDP_WORKER = textwrap.dedent(
    """
    import os, sys, torch, torch.distributed as dist
    sys.path.insert(0, {repo!r})

    from protein_flow.config import Config
    from protein_flow.distributed import setup_distributed, cleanup_distributed, get_rank, get_world_size, is_distributed
    from protein_flow.models.dual_graph_flow import DualGraphFlowModel
    from protein_flow.train import build_dataloaders, compute_losses

    device = setup_distributed("cpu")
    rank, world = get_rank(), get_world_size()

    config = Config()
    config.data.plm_dim = 8
    config.data.train_size = 16
    config.data.val_size = 4
    config.data.batch_size = 2
    config.data.min_length = 6
    config.data.max_length = 6
    for m in (config.model.sequence_encoder, config.model.geometric_encoder,
              config.model.fusion, config.model.decoder):
        m.hidden_dim = 8
    config.model.fusion.condition_dim = 4
    config.model.sequence_encoder.num_layers = 2
    config.model.geometric_encoder.num_layers = 2
    config.model.graph.knn_k = 3
    config.model.graph.num_rbf = 4

    torch.manual_seed(0)
    model = DualGraphFlowModel(config)
    ddp = torch.nn.parallel.DistributedDataParallel(model, static_graph=True)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-2)

    train_loader, _ = build_dataloaders(config)
    from torch.utils.data import DistributedSampler
    assert isinstance(train_loader.sampler, DistributedSampler), "train loader must be sharded"
    assert train_loader.drop_last is True, "drop_last is required so ranks take equal steps"

    batch = next(iter(train_loader))
    opt.zero_grad(set_to_none=True)
    compute_losses(ddp, batch, config)["total"].backward()
    opt.step()

    probe = next(p for n, p in ddp.module.named_parameters() if p.dim() > 1).detach().clone()
    gathered = [torch.zeros_like(probe) for _ in range(world)]
    dist.all_gather(gathered, probe)
    max_diff = max((g - gathered[0]).abs().max().item() for g in gathered)

    if rank == 0:
        print("WORLD", world)
        print("DISTRIBUTED", is_distributed())
        print("MAXDIFF", max_diff)
        print("BATCHES_PER_RANK", len(train_loader))
    cleanup_distributed()
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX process launcher")
def test_two_process_ddp_synchronises_gradients(tmp_path):
    """The defining property of DDP: after one optimizer step every rank
    holds bit-identical parameters, because gradients were all-reduced.

    Uses the gloo CPU backend so this runs anywhere, no GPU required.
    """
    worker = tmp_path / "ddp_worker.py"
    worker.write_text(DDP_WORKER.format(repo=str(REPO_ROOT)))

    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["OMP_NUM_THREADS"] = "1"

    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node=2", "--master_port=29577", str(worker),
        ],
        capture_output=True, text=True, timeout=600, env=env, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"DDP worker failed:\n{result.stdout}\n{result.stderr}"

    output = result.stdout
    assert "WORLD 2" in output, output
    assert "DISTRIBUTED True" in output, output
    max_diff = float(output.split("MAXDIFF")[1].split()[0])
    assert max_diff == 0.0, f"parameters diverged across ranks (max diff {max_diff})"
    # 16 samples / batch 2 / 2 ranks = 4 batches each.
    assert "BATCHES_PER_RANK 4" in output, output
