"""Tests for the shard-rotation plan and the disk pool that executes it.

No network: shard "downloads" are served by an injected function that copies
from a fake remote directory, so the retry, verification, disk-guard and
release paths are all exercised offline.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from protein_flow.data.shard_manifest import (
    ShardEntry,
    ShardManifest,
    build_manifest,
    domain_from_path,
    list_local_entries,
)
from protein_flow.data.shard_rotation import ShardPool


def _entries(count: int, size: int = 1000) -> list[ShardEntry]:
    return [
        ShardEntry(domain=f"dom{index:04d}", path=f"data/mdcath_dataset_dom{index:04d}.h5", size=size)
        for index in range(count)
    ]


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def test_domain_from_path():
    assert domain_from_path("data/mdcath_dataset_1aocA00.h5") == "1aocA00"
    with pytest.raises(ValueError):
        domain_from_path("data/something_else.txt")


def test_same_seed_gives_the_same_plan_and_different_seeds_do_not():
    entries = _entries(100)
    first = build_manifest(entries, seed=7, chunk_size=20, chunk_max_gb=None, num_val_domains=10)
    second = build_manifest(entries, seed=7, chunk_size=20, chunk_max_gb=None, num_val_domains=10)
    other = build_manifest(entries, seed=8, chunk_size=20, chunk_max_gb=None, num_val_domains=10)

    assert [e.domain for e in first.val_domains] == [e.domain for e in second.val_domains]
    assert first.chunks == second.chunks
    assert [e.domain for e in first.val_domains] != [e.domain for e in other.val_domains]


def test_input_order_does_not_change_the_plan():
    """Hub listing order is not guaranteed stable; only the seed may matter."""
    entries = _entries(50)
    forward = build_manifest(entries, seed=3, chunk_size=10, chunk_max_gb=None, num_val_domains=5)
    reversed_ = build_manifest(entries[::-1], seed=3, chunk_size=10, chunk_max_gb=None, num_val_domains=5)
    assert forward.chunks == reversed_.chunks
    assert forward.val_domains == reversed_.val_domains


def test_validation_domains_never_appear_in_any_chunk():
    manifest = build_manifest(_entries(100), seed=0, chunk_size=15, chunk_max_gb=None, num_val_domains=12)
    val = {entry.domain for entry in manifest.val_domains}
    chunked = [entry.domain for chunk in manifest.chunks for entry in chunk]
    assert not val.intersection(chunked)
    assert len(chunked) == len(set(chunked)) == 88
    assert manifest.num_domains == 100


def test_chunks_respect_both_the_count_and_the_byte_ceiling():
    # 2 GB each: the byte ceiling (5 GB) binds before the count cap (10).
    entries = _entries(30, size=2_000_000_000)
    manifest = build_manifest(entries, seed=0, chunk_size=10, chunk_max_gb=5.0, num_val_domains=2)
    assert all(len(chunk) <= 10 for chunk in manifest.chunks)
    assert all(manifest.chunk_bytes(i) <= 5_000_000_000 for i in range(manifest.num_chunks))
    assert max(len(chunk) for chunk in manifest.chunks) == 2


def test_a_single_oversized_shard_still_gets_a_chunk():
    entries = _entries(3, size=1000) + [ShardEntry("huge", "data/mdcath_dataset_huge.h5", 9_000_000_000)]
    manifest = build_manifest(entries, seed=0, chunk_size=5, chunk_max_gb=1.0, num_val_domains=1)
    chunked = [entry.domain for chunk in manifest.chunks for entry in chunk]
    assert "huge" in chunked or "huge" in {e.domain for e in manifest.val_domains}


def test_prefer_local_puts_downloaded_domains_first(tmp_path):
    entries = _entries(40)
    local = tmp_path / "local"
    local.mkdir()
    wanted = {"dom0031", "dom0032", "dom0033"}
    for domain in wanted:
        (local / f"mdcath_dataset_{domain}.h5").write_bytes(b"x")

    manifest = build_manifest(
        entries, seed=1, chunk_size=5, chunk_max_gb=None, num_val_domains=4, prefer_local_dir=local
    )
    train_order = [entry.domain for chunk in manifest.chunks for entry in chunk]
    present = wanted - {entry.domain for entry in manifest.val_domains}
    assert set(train_order[: len(present)]) == present


def test_num_domains_subsets_the_dataset():
    manifest = build_manifest(
        _entries(500), seed=0, chunk_size=30, chunk_max_gb=None, num_val_domains=10, num_domains=120
    )
    assert manifest.num_domains == 120
    assert manifest.num_train_domains == 110


def test_manifest_round_trips_through_json(tmp_path):
    manifest = build_manifest(_entries(40), seed=5, chunk_size=8, chunk_max_gb=None, num_val_domains=4)
    path = tmp_path / "nested" / "manifest.json"
    manifest.save(path)
    loaded = ShardManifest.load(path)
    assert loaded.chunks == manifest.chunks
    assert loaded.val_domains == manifest.val_domains
    assert loaded.seed == manifest.seed
    assert loaded.summary() == manifest.summary()


def test_build_manifest_rejects_impossible_splits():
    with pytest.raises(ValueError):
        build_manifest(_entries(5), num_val_domains=5)
    with pytest.raises(ValueError):
        build_manifest(_entries(50), num_val_domains=10, num_domains=10)
    with pytest.raises(ValueError):
        build_manifest(_entries(50), chunk_size=0)


def test_list_local_entries_reads_real_sizes(tmp_path):
    (tmp_path / "mdcath_dataset_abcA00.h5").write_bytes(b"0123456789")
    (tmp_path / "notes.txt").write_text("ignored")
    entries = list_local_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0].domain == "abcA00" and entries[0].size == 10
    with pytest.raises(FileNotFoundError):
        list_local_entries(tmp_path / "empty")


# --------------------------------------------------------------------------
# Pool
# --------------------------------------------------------------------------
class FakeRemote:
    """Stands in for the Hub: copies from a directory, and can be told to fail."""

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[str] = []
        self.fail_times: dict[str, int] = {}
        self.corrupt: set[str] = set()

    def make(self, domain: str) -> ShardEntry:
        """Writes a minimal but genuinely valid shard, so ``verify=True``
        accepts it, and returns the entry with its real byte size."""
        import h5py

        repo_path = f"data/mdcath_dataset_{domain}.h5"
        path = self.root / repo_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            group = handle.create_group(domain)
            group.create_dataset("pdbProteinAtoms", data=b"ATOM")
        return ShardEntry(domain=domain, path=repo_path, size=path.stat().st_size)

    def __call__(self, repo_id: str, repo_path: str, local_dir: Path) -> Path:
        self.calls.append(repo_path)
        remaining = self.fail_times.get(repo_path, 0)
        if remaining:
            self.fail_times[repo_path] = remaining - 1
            raise RuntimeError("simulated network failure")
        destination = Path(local_dir) / repo_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if repo_path in self.corrupt:
            destination.write_bytes(b"garbage")
        else:
            shutil.copy(self.root / repo_path, destination)
        return destination


def _pool(tmp_path, manifest, remote, **kwargs) -> ShardPool:
    kwargs.setdefault("retry_backoff_seconds", 0.0)  # no sleeping in unit tests
    return ShardPool(
        manifest, tmp_path / "local", download_fn=remote, verify=False, min_free_gb=0.0, **kwargs
    )


def _manifest_with_files(tmp_path, count=8, chunk_size=2, num_val=2):
    remote = FakeRemote(tmp_path / "remote")
    entries = [remote.make(f"dom{index:04d}") for index in range(count)]
    manifest = build_manifest(
        entries, seed=0, chunk_size=chunk_size, chunk_max_gb=None, num_val_domains=num_val
    )
    return manifest, remote


def test_ensure_downloads_then_release_deletes(tmp_path):
    manifest, remote = _manifest_with_files(tmp_path)
    pool = _pool(tmp_path, manifest, remote)

    pool.ensure(0)
    resident = pool.resident_files(0)
    assert len(resident) == len(manifest.chunks[0])
    assert all(path.exists() for path in resident)

    freed = pool.release(0)
    assert freed > 0
    assert pool.resident_files(0) == []
    assert not any(path.exists() for path in resident)


def test_release_also_removes_the_hub_metadata_file(tmp_path):
    """A stale .metadata makes the cache claim a deleted shard is present."""
    manifest, remote = _manifest_with_files(tmp_path)
    pool = _pool(tmp_path, manifest, remote)
    pool.ensure(0)

    entry = manifest.chunks[0][0]
    metadata = pool._metadata_path(entry)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("stale")

    pool.release(0)
    assert not metadata.exists()


def test_already_complete_shards_are_not_redownloaded(tmp_path):
    manifest, remote = _manifest_with_files(tmp_path)
    pool = _pool(tmp_path, manifest, remote)
    pool.ensure(0)
    calls_after_first = len(remote.calls)

    pool.ensure(0)
    assert len(remote.calls) == calls_after_first


def test_download_is_retried_then_gives_up(tmp_path):
    manifest, remote = _manifest_with_files(tmp_path)
    entry = manifest.chunks[0][0]
    remote.fail_times[entry.path] = 2  # succeeds on the third attempt
    pool = _pool(tmp_path, manifest, remote, retries=3)
    pool.ensure(0)
    assert pool.local_path(entry).exists()

    other = manifest.chunks[1][0]
    remote.fail_times[other.path] = 99
    pool.ensure(1)
    assert not pool.local_path(other).exists()
    assert other.path not in {p.name for p in pool.resident_files(1)}


def test_corrupt_download_is_deleted_rather_than_left_behind(tmp_path):
    """resident_files() keys off existence, so a bad file must not survive --
    otherwise every rank would try to train on it."""
    manifest, remote = _manifest_with_files(tmp_path)
    entry = manifest.chunks[0][0]
    remote.corrupt.add(entry.path)
    pool = ShardPool(
        manifest, tmp_path / "local", download_fn=remote, verify=True, min_free_gb=0.0, retries=1,
        retry_backoff_seconds=0.0,
    )
    pool.ensure(0)
    assert not pool.local_path(entry).exists()
    assert len(pool.resident_files(0)) == len(manifest.chunks[0]) - 1


def test_disk_guard_blocks_downloads_that_would_breach_the_reserve(tmp_path):
    manifest, remote = _manifest_with_files(tmp_path)
    # A reserve far larger than the volume: nothing may be written.
    pool = ShardPool(
        manifest, tmp_path / "local", download_fn=remote, verify=False, min_free_gb=1e9
    )
    pool.ensure(0)
    assert pool.resident_files(0) == []
    assert remote.calls == []


def test_ranks_download_disjoint_slices_that_cover_the_chunk(tmp_path):
    manifest, remote = _manifest_with_files(tmp_path, count=20, chunk_size=6, num_val=2)
    pools = [
        ShardPool(
            manifest, tmp_path / "local", download_fn=remote, verify=False, min_free_gb=0.0,
            rank=rank, world_size=3,
        )
        for rank in range(3)
    ]
    for pool in pools:
        pool.ensure(0)

    # Disjoint work, complete coverage, and -- because ranks share the
    # filesystem -- an identical view afterwards, which is what keeps
    # DistributedSampler in step across ranks.
    assert len(remote.calls) == len(set(remote.calls)) == len(manifest.chunks[0])
    views = [pool.resident_files(0) for pool in pools]
    assert views[0] == views[1] == views[2]
    assert len(views[0]) == len(manifest.chunks[0])


def test_release_is_also_rank_sliced(tmp_path):
    manifest, remote = _manifest_with_files(tmp_path, count=20, chunk_size=6, num_val=2)
    pools = [
        ShardPool(
            manifest, tmp_path / "local", download_fn=remote, verify=False, min_free_gb=0.0,
            rank=rank, world_size=2,
        )
        for rank in range(2)
    ]
    pools[0].ensure(0)
    pools[1].ensure(0)
    assert len(pools[0].resident_files(0)) == len(manifest.chunks[0])

    pools[0].release(0)
    assert len(pools[0].resident_files(0)) == len(manifest.chunks[0]) - len(manifest.chunks[0][0::2])
    pools[1].release(0)
    assert pools[0].resident_files(0) == []


def test_prefetch_runs_in_the_background_and_ensure_joins_it(tmp_path):
    manifest, remote = _manifest_with_files(tmp_path, count=12, chunk_size=3, num_val=2)
    pool = _pool(tmp_path, manifest, remote)
    pool.prefetch(1)
    pool.ensure(1)  # must join the thread, not start a second download
    assert len(pool.resident_files(1)) == len(manifest.chunks[1])
    assert len(remote.calls) == len(set(remote.calls))


def test_prefetch_out_of_range_is_a_no_op(tmp_path):
    manifest, remote = _manifest_with_files(tmp_path)
    pool = _pool(tmp_path, manifest, remote)
    pool.prefetch(manifest.num_chunks + 5)
    pool.stop_prefetch()
    assert remote.calls == []


def test_val_files_are_fetched_and_survive_chunk_releases(tmp_path):
    manifest, remote = _manifest_with_files(tmp_path)
    pool = _pool(tmp_path, manifest, remote)
    pool.ensure_val()
    pool.ensure(0)
    pool.release(0)
    assert len(pool.val_files()) == len(manifest.val_domains)


# --------------------------------------------------------------------------
# Rotation loop plumbing
# --------------------------------------------------------------------------
def test_resume_path_resolution(tmp_path):
    from protein_flow.config import Config
    from protein_flow.train_rotating import _resolve_resume_path

    config = Config()
    config.train.ckpt_dir = str(tmp_path)

    config.train.resume = None
    assert _resolve_resume_path(config) is None

    # "auto" is tolerant of a first run, where there is nothing to resume.
    config.train.resume = "auto"
    assert _resolve_resume_path(config) is None
    (tmp_path / "last.pt").write_text("checkpoint")
    assert _resolve_resume_path(config) == tmp_path / "last.pt"

    # An explicit path that does not exist is a typo, not a fresh start.
    config.train.resume = str(tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError):
        _resolve_resume_path(config)


def test_save_checkpoint_round_trips_the_rotation_cursor(tmp_path):
    import torch

    from protein_flow.config import Config
    from protein_flow.train import load_checkpoint, save_checkpoint

    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(
        tmp_path / "last.pt", model, optimizer, Config(), step=17, best_val_loss=1.5,
        extra={"rotation_cursor": 4},
    )
    checkpoint = load_checkpoint(tmp_path / "last.pt", model, optimizer)
    assert checkpoint["extra"]["rotation_cursor"] == 4
    assert checkpoint["step"] == 17


def test_checkpoints_written_without_extra_still_load():
    """Older checkpoints predate the rotation cursor; resume must treat a
    missing one as 'start at the beginning' rather than crashing."""
    import torch

    from protein_flow.config import Config
    from protein_flow.train import save_checkpoint

    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "last.pt"
        save_checkpoint(path, model, optimizer, Config(), step=3, best_val_loss=2.0)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert (checkpoint.get("extra") or {}).get("rotation_cursor", 0) == 0


def test_chunk_step_budget_lands_on_pass_boundaries():
    """passes_per_chunk must never truncate a pass, whatever the pass length
    turns out to be -- that length moves with batch size, world size, and how
    many trajectories a chunk actually yields."""
    from protein_flow.config import RotationConfig
    from protein_flow.train_rotating import chunk_step_budget

    cfg = RotationConfig(passes_per_chunk=3, steps_per_chunk=2000)
    for batches_per_pass in (625, 312, 417, 1):
        budget = chunk_step_budget(cfg, batches_per_pass)
        assert budget == 3 * batches_per_pass
        assert budget % batches_per_pass == 0, "budget must end on a pass boundary"

    # steps_per_chunk stays available and is used verbatim when passes is unset.
    raw = RotationConfig(passes_per_chunk=None, steps_per_chunk=2000)
    assert chunk_step_budget(raw, 625) == 2000
    assert chunk_step_budget(raw, 625) % 625 != 0, "the raw knob is what truncates"

    with pytest.raises(ValueError):
        chunk_step_budget(RotationConfig(passes_per_chunk=0), 625)


# --------------------------------------------------------------------------
# Download isolation
#
# The downloader runs in a child interpreter so that nothing it pulls in --
# retry loops, connection pools, h5py, hf_xet's Rust thread pool -- can wedge
# the training process. Three runs hung before this was introduced.
# --------------------------------------------------------------------------
def test_injected_downloader_disables_isolation_and_the_real_one_enables_it(tmp_path):
    """Tests inject a downloader, which cannot survive a process boundary."""
    manifest, remote = _manifest_with_files(tmp_path)
    assert _pool(tmp_path, manifest, remote).isolate_downloads is False
    assert ShardPool(manifest, tmp_path / "solo", min_free_gb=0.0).isolate_downloads is True


def test_isolated_downloader_runs_in_a_child_process_and_reports_back(tmp_path):
    """End-to-end through the real subprocess, no network.

    Every shard is already on disk at its manifest size, so the child takes
    the ``_is_complete`` path and never reaches ``_hf_download`` -- which
    exercises the payload, the spawn, and the summary hand-back for real.
    """
    remote = FakeRemote(tmp_path / "remote")
    entries = [remote.make(f"dom{index:04d}") for index in range(4)]
    manifest = ShardManifest(
        repo_id="fake/repo", seed=0, created="", val_domains=[], chunks=[entries]
    )
    local_dir = tmp_path / "local"
    for entry in entries:  # pre-stage, so no download is needed
        destination = local_dir / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(remote.root / entry.path, destination)

    pool = ShardPool(manifest, local_dir, min_free_gb=0.0, retry_backoff_seconds=0.0)
    assert pool.isolate_downloads is True
    pool.ensure(0)

    assert sorted(p.name for p in pool.resident_files(0)) == sorted(
        Path(entry.path).name for entry in entries
    )


def test_stop_prefetch_leaves_no_orphan_downloader(tmp_path):
    remote = FakeRemote(tmp_path / "remote")
    entries = [remote.make(f"dom{index:04d}") for index in range(2)]
    manifest = ShardManifest(
        repo_id="fake/repo", seed=0, created="", val_domains=[], chunks=[entries]
    )
    local_dir = tmp_path / "local"
    for entry in entries:
        destination = local_dir / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(remote.root / entry.path, destination)

    pool = ShardPool(manifest, local_dir, min_free_gb=0.0, retry_backoff_seconds=0.0)
    pool.prefetch(0)
    job = pool._prefetch_jobs.get(0)
    pool.stop_prefetch()
    assert pool._prefetch_jobs == {}
    if job is not None:
        assert job.process.poll() is not None  # reaped, not left running


def test_a_hung_downloader_is_killed_instead_of_blocking_the_run(tmp_path):
    """A download child that never exits must not take the run with it.

    This is the failure this timeout exists for, reproduced exactly: a child
    that hangs rather than crashes left both training ranks blocked in
    ``subprocess.wait()`` for 2.5 days, holding their CUDA contexts with the
    GPUs at 0%. Neither guard covered it -- the NCCL watchdog only watches
    collectives, and the launcher only restarts a process that *exits*.
    """
    import json
    import subprocess
    import sys
    import time

    from protein_flow.data.shard_rotation import _DownloadJob

    payload = tmp_path / "payload.json"
    result = tmp_path / "payload.result.json"
    payload.write_text("{}")
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    job = _DownloadJob(process, payload, result, "chunk 12")

    started = time.monotonic()
    summary = job.wait(timeout=1.0)
    elapsed = time.monotonic() - started

    assert elapsed < 30, f"wait() blocked for {elapsed:.0f}s instead of giving up at its timeout"
    assert summary["timed_out"] is True
    assert process.poll() is not None, "the hung child was left running"
    # The run carries on with whatever reached disk rather than raising.
    assert summary["succeeded"] == 0 and summary["total"] == 0
    assert not payload.exists() and not result.exists()


def test_a_downloader_that_finishes_in_time_is_not_disturbed(tmp_path):
    """The timeout must not truncate a healthy download."""
    import json
    import subprocess
    import sys

    from protein_flow.data.shard_rotation import _DownloadJob

    payload = tmp_path / "payload.json"
    result = tmp_path / "payload.result.json"
    payload.write_text("{}")
    result.write_text(json.dumps({"succeeded": 3, "total": 3, "downloaded_bytes": 99, "elapsed": 0.5}))
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    summary = _DownloadJob(process, payload, result, "chunk 12").wait(timeout=60.0)

    assert summary["timed_out"] is False
    assert summary["succeeded"] == 3 and summary["downloaded_bytes"] == 99
