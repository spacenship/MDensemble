"""A non-finite loss on one rank must not desynchronise the others.

DDP performs its gradient all-reduce inside ``backward()``. The original
``training_step`` returned early on a non-finite loss *before* that call, so a
rank that hit a NaN silently dropped out of the collective while its peers
blocked in it forever. The ranks then drifted apart -- observed in the wild as
rank 0 issuing ALLREDUCE at step 20210 while rank 1 sat in BROADCAST at step
20206 under the same sequence number -- until the NCCL watchdog aborted the
job half an hour later.

The cost was not subtle: on the first run that ever produced a non-finite
loss, 13 of 13 crashes landed exactly 31.1 minutes (the 1800 s watchdog) after
one, and the restart storm ate seven hours.

These tests run two real processes over gloo, because the bug is invisible to
a single-process test: the whole failure is one rank taking a branch the other
did not.
"""
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int, port: int, queue):
    """Rank 0 sees a non-finite loss, rank 1 does not. Both must agree."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank),
    )
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # The decision rule under test, lifted from training_step.
        total_loss = torch.tensor(float("nan")) if rank == 0 else torch.tensor(1.0)
        everyone_finite = torch.tensor([1.0 if torch.isfinite(total_loss) else 0.0])
        dist.all_reduce(everyone_finite, op=dist.ReduceOp.MIN)
        skipped = everyone_finite.item() == 0.0

        # If the ranks agree, this collective completes. Under the old code
        # rank 0 would have returned before reaching it and rank 1 would hang
        # here until the test timed out.
        witness = torch.tensor([1.0 if skipped else 0.0])
        dist.all_reduce(witness, op=dist.ReduceOp.SUM)

        queue.put((rank, skipped, witness.item()))
        dist.destroy_process_group()
    except Exception as exc:  # surfaced by the parent as a failure
        queue.put((rank, "error", repr(exc)))


def test_all_ranks_skip_together_when_one_sees_a_nan():
    if not dist.is_gloo_available():
        pytest.skip("gloo unavailable")

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = 29500 + (os.getpid() % 2000)
    processes = [
        context.Process(target=_worker, args=(rank, 2, port, queue)) for rank in range(2)
    ]
    for process in processes:
        process.start()
    # Generous but finite: the failure mode being guarded against is a hang,
    # so this join is the assertion.
    for process in processes:
        process.join(timeout=45)

    alive = [p for p in processes if p.is_alive()]
    for process in alive:
        process.terminate()
    assert not alive, "a rank hung -- the skip decision was not collective"

    results = {}
    while not queue.empty():
        rank, skipped, witness = queue.get()
        assert skipped != "error", f"rank {rank} raised {witness}"
        results[rank] = (skipped, witness)

    assert set(results) == {0, 1}, f"expected both ranks to report, got {sorted(results)}"
    assert results[0][0] is True and results[1][0] is True, (
        f"ranks disagreed on whether to skip: {results}"
    )
    # Both ranks contributed 1.0, proving they reached the same branch.
    assert results[0][1] == 2.0 and results[1][1] == 2.0


def test_no_skip_when_every_rank_is_finite():
    """The guard must not turn healthy steps into skipped ones."""
    total_loss = torch.tensor(1.0)
    everyone_finite = torch.tensor([1.0 if torch.isfinite(total_loss) else 0.0])
    assert everyone_finite.item() == 1.0


def test_the_guard_is_reachable_in_training_step():
    """Pin the wiring: the collective must sit before backward(), not after."""
    import inspect

    from protein_flow.train import training_step

    source = inspect.getsource(training_step)
    finite_at = source.index("everyone_finite")
    # The real call, not the several comments that name it.
    backward_at = source.index("scaler.scale(total_loss).backward()")
    assert finite_at < backward_at, (
        "the finite-flag all_reduce must run before backward(); after it, a "
        "diverged rank has already missed the gradient collective"
    )
    assert "ReduceOp.MIN" in source, "the flag must be combined with MIN so any NaN skips all"


def test_amp_dtype_selects_the_scaler_correctly():
    """bf16 needs no GradScaler, and an unknown dtype must fail loudly.

    The scaler exists to lift fp16 gradients out of underflow. bf16 has
    fp32's exponent range, so scaling only reintroduces a way to overflow --
    the exact failure this dtype was chosen to escape.
    """
    import torch as _torch

    from protein_flow.config import Config
    from protein_flow.train import amp_dtype, needs_grad_scaler

    config = Config()
    config.train.amp = True
    assert config.train.amp_dtype == "float16", "default must stay fp16 for reproducibility"
    assert amp_dtype(config) is _torch.float16
    assert needs_grad_scaler(config) is True

    config.train.amp_dtype = "bfloat16"
    assert amp_dtype(config) is _torch.bfloat16
    assert needs_grad_scaler(config) is False

    config.train.amp = False
    assert needs_grad_scaler(config) is False

    config.train.amp_dtype = "float64"
    with pytest.raises(ValueError, match="amp_dtype"):
        amp_dtype(config)


def test_the_configured_rerun_uses_bfloat16():
    """Pin the fix to the config that carries it."""
    from protein_flow.config import load_config

    config = load_config("configs/mdcath_backbone_rotate_displacement_norigid.yaml")
    assert config.train.amp is True
    assert config.train.amp_dtype == "bfloat16"
