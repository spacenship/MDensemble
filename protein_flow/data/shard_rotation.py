"""Streams mdCATH chunks onto local disk during training, and off it again.

:class:`ShardPool` is the disk side of a rotating run (the plan itself lives
in :mod:`protein_flow.data.shard_manifest`): it downloads a chunk, verifies
every shard actually opens, hands the trainer the resident file list, and
deletes the chunk when the trainer is done with it -- while a background
thread is already fetching the next one so the GPU is not waiting on the
network.

Three properties matter for correctness rather than tidiness:

**Every rank must end up with the same file list.** Under DDP,
``DistributedSampler`` assumes all ranks share one dataset; if rank 0 saw 200
shards and rank 1 saw 199, the two would disagree on the number of batches
and the run would deadlock at the next all-reduce. So the resident list is
never "what I downloaded" -- it is recomputed by every rank from what exists
on disk (:meth:`resident_files`), after a barrier. A shard that failed
verification is deleted, so it is missing for everyone alike.

**Download work is split across ranks.** Each rank fetches
``entries[rank::world_size]``. This is N times faster than one rank doing it
all, and it removes the case where non-zero ranks sit at a collective for a
whole 134 GB download -- long enough to trip the NCCL watchdog. It assumes
ranks share a filesystem, which is true for single-node multi-GPU (what this
project runs) and for any shared mount; a multi-node run without one must set
``world_size=1`` here so every rank fetches everything.

**The disk budget is enforced before writing, not after.** Prefetching stops
(loudly) rather than filling the volume the training checkpoints also live on.

**Downloads run in a separate process, not a thread.** Three consecutive runs
hung while a prefetch was in flight, each in a different place, and the only
constant was that the downloader shared an address space with training. A
background downloader drags a great deal into that address space:
``huggingface_hub``'s retry loops, urllib3 connection pools, ``h5py`` for
verification, and -- with ``hf_xet`` installed -- a 32-thread Rust runtime,
all of it interleaving with DataLoader worker startup and NCCL collectives.
The third hang left one rank with 32 idle ``hf-xet`` threads, no sockets, and
a training thread that never advanced again.

Rather than keep guessing which lock it was, the download runs in a child
interpreter (see ``_DownloadJob`` and the ``__main__`` block at the bottom of
this file). Nothing about it is visible to training except the files it
leaves on disk, which is all :meth:`resident_files` ever consulted anyway.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from protein_flow.data.shard_manifest import ShardEntry, ShardManifest

logger = logging.getLogger(__name__)

# protein_flow/data/shard_rotation.py -> repo root, so the child interpreter
# can import protein_flow no matter where the trainer was launched from.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Signature of the injectable downloader: (repo_id, repo_relative_path,
# local_dir) -> path of the downloaded file. Real runs use
# huggingface_hub.hf_hub_download; tests substitute a local copy.
DownloadFn = Callable[[str, str, Path], Path]


# huggingface_hub defaults to a 10s read timeout on the etag HEAD that
# precedes every download. Shard fetches are the one thing here that genuinely
# needs the network, so they cannot go cache-only like the ESM assets in
# protein_flow/hf_cache.py; give the HEAD room to survive a slow moment
# instead, since burning a retry costs far more than waiting.
_ETAG_TIMEOUT_SECONDS = 60.0


def _die_with_parent() -> None:
    """Ask the kernel to SIGKILL this child when its parent dies.

    Runs in the child between fork and exec. Without it a downloader outlives
    a training process that was aborted rather than shut down -- and the
    NCCL watchdog aborts with SIGABRT, so that is the common case, not the
    rare one. Two such orphans were found still downloading hours later, and
    they do more than waste bandwidth: they inherit the launcher's stdout, so
    the `torchrun | tee` pipeline never sees EOF, `tee` never exits, and the
    restart loop never reaches the code that would relaunch the run. Both
    observed crashes left the GPUs idle for that reason.

    Best-effort: a platform without prctl(2) just keeps the old behaviour,
    which is what terminate() and the timeout in wait() already cover.
    """
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:  # noqa: BLE001 - never let this stop a download from starting
        pass


def _hf_download(repo_id: str, repo_path: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=repo_path,
            local_dir=str(local_dir),
            etag_timeout=_ETAG_TIMEOUT_SECONDS,
        )
    )


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


class _DownloadJob:
    """One batch of shards being fetched by a child interpreter.

    Holds no lock, no socket and no thread on the training side -- only a
    pid and the path the child will write its summary to. If the child dies
    for any reason the parent notices at :meth:`wait` and carries on with
    whatever landed on disk, because a chunk with a few shards missing is a
    slightly smaller chunk, not a broken run.

    A child that *hangs* rather than dies needs the same treatment, and that
    is what ``timeout`` is for. Without it a stalled network read inside the
    downloader takes the whole run with it: measured once, a child sat in a
    socket read for 2.5 days while both training ranks blocked in
    ``subprocess.wait()`` holding their CUDA contexts. Nothing else catches
    this -- the NCCL watchdog only monitors collectives, and the launcher's
    restart loop only fires when a process *exits*.
    """

    def __init__(self, process: subprocess.Popen, payload_path: Path, result_path: Path, label: str):
        self.process = process
        self.payload_path = payload_path
        self.result_path = result_path
        self.label = label
        self.started = time.monotonic()

    def wait(self, timeout: Optional[float] = None) -> dict:
        timed_out = False
        try:
            returncode = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            logger.error(
                "%s: downloader still running after %.0fs -- treating it as hung and killing it. "
                "The run continues with whatever shards reached disk; raise "
                "data.rotation.download_timeout_minutes if this chunk legitimately needs longer.",
                self.label, timeout,
            )
            self.process.kill()
            returncode = self.process.wait()
        # Wall time from spawn to reaping, which for a prefetch is mostly the
        # training that ran alongside it. The child reports its own working
        # time in the summary; only fall back to this when it never wrote one.
        waited = time.monotonic() - self.started
        summary = {"succeeded": 0, "total": 0, "downloaded_bytes": 0}
        try:
            summary.update(json.loads(self.result_path.read_text()))
        except Exception:  # noqa: BLE001 - a crashed child leaves no result file
            if returncode != 0 and not timed_out:
                logger.error(
                    "%s: downloader exited with code %d and wrote no summary; "
                    "continuing with whatever reached disk",
                    self.label, returncode,
                )
        finally:
            self.payload_path.unlink(missing_ok=True)
            self.result_path.unlink(missing_ok=True)
        summary.setdefault("elapsed", waited)
        summary["timed_out"] = timed_out
        return summary

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.payload_path.unlink(missing_ok=True)
        self.result_path.unlink(missing_ok=True)


class ShardPool:
    """Keeps a rotating subset of the dataset's shards on local disk."""

    def __init__(
        self,
        manifest: ShardManifest,
        local_dir: str | Path,
        *,
        min_free_gb: float = 150.0,
        retries: int = 3,
        verify: bool = True,
        rank: int = 0,
        world_size: int = 1,
        download_fn: Optional[DownloadFn] = None,
        retry_backoff_seconds: float = 2.0,
        download_timeout_seconds: Optional[float] = 10800.0,
    ):
        self.manifest = manifest
        self.local_dir = Path(local_dir)
        self.data_dir = self.local_dir / "data"
        self.min_free_gb = min_free_gb
        self.retries = max(1, retries)
        self.verify = verify
        self.rank = rank
        self.world_size = max(1, world_size)
        self._download_fn = download_fn or _hf_download
        self.retry_backoff_seconds = retry_backoff_seconds
        # How long an isolated downloader may run before it is presumed hung
        # and killed. A 146 GB chunk at the measured ~50 MB/s takes ~50 min,
        # so the 3 h default is ~3.6x headroom; anything past it is a stalled
        # socket, not a slow link. None disables the guard, which is what the
        # 2.5-day hang this exists to prevent looked like.
        self.download_timeout_seconds = download_timeout_seconds
        # An injected downloader is by definition in-process (tests pass a
        # local copy), so isolation only applies to the real one.
        self.isolate_downloads = download_fn is None
        self._prefetch_jobs: Dict[int, _DownloadJob] = {}
        self._wait_seconds = 0.0

        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def local_path(self, entry: ShardEntry) -> Path:
        return self.local_dir / entry.path

    def _metadata_path(self, entry: ShardEntry) -> Path:
        """Where huggingface_hub records this file's download metadata.

        Deleted alongside the shard: leaving it behind makes the cache claim
        a file is present that the rotation has already removed.
        """
        return self.local_dir / ".cache" / "huggingface" / "download" / f"{entry.path}.metadata"

    def _own_slice(self, entries: Sequence[ShardEntry]) -> List[ShardEntry]:
        return list(entries[self.rank :: self.world_size])

    # ------------------------------------------------------------------
    # Acquiring
    # ------------------------------------------------------------------
    def _is_complete(self, entry: ShardEntry) -> bool:
        path = self.local_path(entry)
        return path.exists() and path.stat().st_size == entry.size

    def _verify(self, entry: ShardEntry) -> bool:
        """Opens the shard to catch a truncated or corrupt download here,
        rather than as a baffling dataset-indexing error much later."""
        if not self.verify:
            return True
        try:
            import h5py

            with h5py.File(self.local_path(entry), "r") as handle:
                domain = next(iter(handle.keys()))
                return "pdbProteinAtoms" in handle[domain]
        except Exception as error:  # noqa: BLE001 - any failure means "unusable"
            logger.warning("Shard %s failed verification: %s", entry.domain, error)
            return False

    def _download_entry(self, entry: ShardEntry) -> bool:
        if self._is_complete(entry) and self._verify(entry):
            return True

        needed_gb = entry.size / 1e9
        if free_gb(self.local_dir) - needed_gb < self.min_free_gb:
            logger.warning(
                "Skipping %s (%.1f GB): only %.0f GB free, min_free_gb=%.0f",
                entry.domain, needed_gb, free_gb(self.local_dir), self.min_free_gb,
            )
            return False

        for attempt in range(1, self.retries + 1):
            try:
                self._download_fn(self.manifest.repo_id, entry.path, self.local_dir)
                if self._verify(entry):
                    return True
                # A corrupt file must not survive: resident_files() keys off
                # existence, so leaving it would poison every rank's view.
                self.local_path(entry).unlink(missing_ok=True)
                self._metadata_path(entry).unlink(missing_ok=True)
            except Exception as error:  # noqa: BLE001 - network errors are expected and retried
                logger.warning(
                    "Download failed for %s (attempt %d/%d): %s", entry.domain, attempt, self.retries, error
                )
                if self.retry_backoff_seconds:
                    time.sleep(min(30.0, self.retry_backoff_seconds**attempt))
        logger.error("Giving up on %s after %d attempts", entry.domain, self.retries)
        return False

    def _run_downloads(self, own: Sequence[ShardEntry]) -> dict:
        """Fetch this rank's slice in the current process (child, or a test).

        Times itself, because the parent cannot: it only reaps a prefetch when
        it needs the chunk, so measuring there would report the training that
        happened alongside the download and made throughput look ~0 MB/s.
        """
        started = time.monotonic()
        downloaded_bytes = sum(entry.size for entry in own if not self._is_complete(entry))
        succeeded = sum(int(self._download_entry(entry)) for entry in own)
        return {
            "succeeded": succeeded,
            "total": len(own),
            "downloaded_bytes": downloaded_bytes,
            "elapsed": time.monotonic() - started,
        }

    def _log_summary(self, label: str, summary: dict) -> None:
        elapsed = summary.get("elapsed", 0.0)
        downloaded_bytes = summary.get("downloaded_bytes", 0)
        throughput = downloaded_bytes / 1e6 / elapsed if elapsed > 0 and downloaded_bytes else 0.0
        fetched = (
            f"{downloaded_bytes / 1e9:.0f} GB in {elapsed:.0f}s ({throughput:.0f} MB/s)"
            if downloaded_bytes
            else f"nothing to fetch, verified in {elapsed:.0f}s"
        )
        logger.info(
            "%s: %d/%d shard(s) ready on rank %d -- %s, %.0f GB free",
            label, summary.get("succeeded", 0), summary.get("total", 0),
            self.rank, fetched, free_gb(self.local_dir),
        )

    def _start_job(self, entries: Sequence[ShardEntry], label: str) -> Optional[_DownloadJob]:
        """Launch the isolated downloader for this rank's slice of ``entries``."""
        own = self._own_slice(entries)
        if not own:
            return None
        handle, payload_name = tempfile.mkstemp(prefix="shardpool-", suffix=".json")
        os.close(handle)
        payload_path = Path(payload_name)
        result_path = payload_path.with_suffix(".result.json")
        payload_path.write_text(json.dumps({
            "repo_id": self.manifest.repo_id,
            "local_dir": str(self.local_dir),
            "entries": [asdict(entry) for entry in own],
            "retries": self.retries,
            "verify": self.verify,
            "min_free_gb": self.min_free_gb,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "result_path": str(result_path),
        }))
        process = subprocess.Popen(
            [sys.executable, "-m", "protein_flow.data.shard_rotation", str(payload_path)],
            cwd=str(_REPO_ROOT),
            preexec_fn=_die_with_parent,  # noqa: PLW1509 - see _die_with_parent
        )
        return _DownloadJob(process, payload_path, result_path, label)

    def _download_entries(self, entries: Sequence[ShardEntry], label: str) -> None:
        """Fetch this rank's slice and block until it is done."""
        own = self._own_slice(entries)
        if not own:
            return
        if not self.isolate_downloads:
            started = time.monotonic()
            summary = self._run_downloads(own)
            summary["elapsed"] = time.monotonic() - started
            self._log_summary(label, summary)
            return
        job = self._start_job(entries, label)
        if job is not None:
            self._log_summary(label, job.wait(timeout=self.download_timeout_seconds))

    def ensure_val(self) -> None:
        """Fetches the validation holdout, which then stays resident forever.

        A rotating run compares val loss across chunks, so the val set must
        not rotate with them.
        """
        self._download_entries(self.manifest.val_domains, "validation holdout")

    def ensure(self, chunk_index: int) -> None:
        """Blocks until this rank's slice of the chunk is on disk.

        Returns immediately if a prefetch already did the work; the wait it
        does report is the real cost of the rotation and is logged so
        ``steps_per_chunk`` can be tuned against it.
        """
        job = self._prefetch_jobs.pop(chunk_index, None)
        started = time.monotonic()
        if job is not None:
            self._log_summary(job.label, job.wait(timeout=self.download_timeout_seconds))
        else:
            self._download_entries(self.manifest.chunks[chunk_index], f"chunk {chunk_index}")
        waited = time.monotonic() - started
        self._wait_seconds += waited
        if waited > 1.0:
            logger.info("Waited %.0fs for chunk %d (cumulative download stall: %.0fs)", waited, chunk_index, self._wait_seconds)

    def prefetch(self, chunk_index: int) -> None:
        """Starts fetching a future chunk in the background (idempotent).

        With isolation on this returns as soon as the child is spawned, so
        the training process goes straight back to stepping. Without it
        (tests) the fetch happens inline, which keeps the ordering the same
        without needing a thread nobody would observe.
        """
        if chunk_index >= self.manifest.num_chunks or chunk_index in self._prefetch_jobs:
            return
        if free_gb(self.local_dir) < self.min_free_gb:
            logger.warning(
                "Not prefetching chunk %d: %.0f GB free is below min_free_gb=%.0f",
                chunk_index, free_gb(self.local_dir), self.min_free_gb,
            )
            return
        label = f"chunk {chunk_index} (prefetch)"
        if not self.isolate_downloads:
            self._download_entries(self.manifest.chunks[chunk_index], label)
            return
        job = self._start_job(self.manifest.chunks[chunk_index], label)
        if job is not None:
            self._prefetch_jobs[chunk_index] = job

    # ------------------------------------------------------------------
    # Using and releasing
    # ------------------------------------------------------------------
    def resident_files(self, chunk_index: int) -> List[Path]:
        """The chunk's shards that are actually on disk, in manifest order.

        Computed from the filesystem rather than from what this rank
        downloaded, so every rank derives an identical list from the shared
        volume -- which is what keeps DDP's samplers in step.
        """
        return [
            self.local_path(entry)
            for entry in self.manifest.chunks[chunk_index]
            if self.local_path(entry).exists()
        ]

    def val_files(self) -> List[Path]:
        return [self.local_path(entry) for entry in self.manifest.val_domains if self.local_path(entry).exists()]

    def release(self, chunk_index: int) -> float:
        """Deletes this rank's slice of a chunk. Returns the GB freed."""
        freed = 0
        for entry in self._own_slice(self.manifest.chunks[chunk_index]):
            path = self.local_path(entry)
            if path.exists():
                freed += path.stat().st_size
                path.unlink()
            self._metadata_path(entry).unlink(missing_ok=True)
        freed_gb = freed / 1e9
        if freed_gb:
            logger.info(
                "Released chunk %d on rank %d: freed %.0f GB (%.0f GB free)",
                chunk_index, self.rank, freed_gb, free_gb(self.local_dir),
            )
        return freed_gb

    def stop_prefetch(self) -> None:
        """Stops in-flight prefetches so a shutdown leaves no orphan process.

        Terminated rather than joined: a shutdown should not wait out a
        134 GB download, and a partially fetched chunk is harmless because
        :meth:`resident_files` keys off completed files.
        """
        for chunk_index, job in list(self._prefetch_jobs.items()):
            job.terminate()
            self._prefetch_jobs.pop(chunk_index, None)


def _main(payload_path: str) -> int:
    """Entry point of the isolated downloader.

    Runs as ``python -m protein_flow.data.shard_rotation <payload.json>``.
    It rebuilds just enough of a :class:`ShardPool` to reuse the retry,
    verification and disk-guard logic, then writes a JSON summary the parent
    turns into the usual log line. Everything the download drags in --
    connection pools, h5py, hf_xet's thread pool -- dies with this process.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | [downloader] %(message)s"
    )
    payload = json.loads(Path(payload_path).read_text())
    entries = [ShardEntry(**entry) for entry in payload["entries"]]
    # world_size=1: the parent already narrowed this to one rank's slice.
    pool = ShardPool(
        ShardManifest(
            repo_id=payload["repo_id"], seed=0, created="", val_domains=[], chunks=[entries]
        ),
        payload["local_dir"],
        min_free_gb=payload["min_free_gb"],
        retries=payload["retries"],
        verify=payload["verify"],
        rank=0,
        world_size=1,
        retry_backoff_seconds=payload["retry_backoff_seconds"],
    )
    summary = pool._run_downloads(entries)
    Path(payload["result_path"]).write_text(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1]))
