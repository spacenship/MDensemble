#!/usr/bin/env python3
"""How much of a rotating run's data is on disk right now.

    python scripts/rotation_status.py                  # one-shot report
    python scripts/rotation_status.py --watch 15       # refresh every 15s
    python scripts/rotation_status.py --no-rate        # skip throughput sampling

A rotating run is silent while it downloads (tens of GB per chunk, which can
be the better part of an hour), so "no log output and no GPU usage" is the
normal state at a chunk boundary, not a hang. This reads the *filesystem*
rather than the log and answers the actual question: how far along is the
download, and when does training start.

Read-only and safe to run against a live run: it only stats files.

Deliberately imports no torch -- it should start instantly even while the
GPUs are busy.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from protein_flow.config import load_config  # noqa: E402
from protein_flow.data.shard_manifest import ShardEntry, ShardManifest  # noqa: E402


def complete(root: Path, entry: ShardEntry) -> bool:
    """Same test the pool uses: present and the exact expected size."""
    path = root / entry.path
    return path.exists() and path.stat().st_size == entry.size


def progress(root: Path, entries: List[ShardEntry]) -> Tuple[int, int, int, int]:
    """(done, total, done_bytes, missing_bytes)"""
    done = [entry for entry in entries if complete(root, entry)]
    done_bytes = sum(entry.size for entry in done)
    total_bytes = sum(entry.size for entry in entries)
    return len(done), len(entries), done_bytes, total_bytes - done_bytes


def bar(done: int, total: int, width: int = 24) -> str:
    filled = 0 if total == 0 else round(width * done / total)
    return "#" * filled + "." * (width - filled)


def in_flight(root: Path) -> Tuple[int, int]:
    """Partially-downloaded files huggingface_hub is still writing."""
    files = list((root / ".cache").rglob("*.incomplete")) if (root / ".cache").exists() else []
    return len(files), sum(f.stat().st_size for f in files if f.exists())


def bytes_on_disk(root: Path) -> int:
    total = sum(f.stat().st_size for f in (root / "data").glob("*.h5") if f.exists())
    return total + in_flight(root)[1]


def current_cursor(log_path: Optional[Path]) -> Optional[int]:
    """The chunk the run last announced, read from its log."""
    if not log_path or not log_path.exists():
        return None
    cursors = re.findall(r"\(cursor (\d+)\)", log_path.read_text(errors="replace"))
    return int(cursors[-1]) if cursors else None


def humanize(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def report(args) -> None:
    config = load_config(args.config)
    rotation = config.data.rotation
    manifest = ShardManifest.load(args.manifest or rotation.manifest_path)
    root = Path(args.local_dir or rotation.local_dir or Path(config.data.mdcath_dir).parent)

    cursor = args.chunk if args.chunk is not None else current_cursor(args.log)
    chunk_index = 0 if cursor is None else cursor % manifest.num_chunks

    print(f"manifest   : {args.manifest or rotation.manifest_path}")
    print(f"             {manifest.num_domains} domains, {manifest.num_chunks} chunks, "
          f"{manifest.total_bytes / 1e12:.2f} TB total")
    free = shutil.disk_usage(root).free / 1e9
    print(f"local_dir  : {root}")
    print(f"             {free:.0f} GB free (min_free_gb {rotation.min_free_gb:.0f})"
          + ("   <-- BELOW RESERVE, prefetch paused" if free < rotation.min_free_gb else ""))
    print()

    val_done, val_total, _, val_missing = progress(root, manifest.val_domains)
    print(f"validation holdout  [{bar(val_done, val_total)}] {val_done:>4}/{val_total}   "
          f"{val_missing / 1e9:6.1f} GB left")

    chunk_done, chunk_total, _, chunk_missing = progress(root, manifest.chunks[chunk_index])
    label = f"chunk {chunk_index}" + ("" if args.chunk is not None or cursor is not None else " (assumed)")
    print(f"{label:<19} [{bar(chunk_done, chunk_total)}] {chunk_done:>4}/{chunk_total}   "
          f"{chunk_missing / 1e9:6.1f} GB left")

    if rotation.prefetch_chunks > 0:
        ahead = (chunk_index + 1) % manifest.num_chunks
        next_done, next_total, _, next_missing = progress(root, manifest.chunks[ahead])
        print(f"chunk {ahead} (prefetch)   [{bar(next_done, next_total)}] {next_done:>4}/{next_total}   "
              f"{next_missing / 1e9:6.1f} GB left")

    files, flight_bytes = in_flight(root)
    print(f"\nin flight  : {files} file(s), {flight_bytes / 1e9:.2f} GB partial")

    if args.sample_seconds > 0:
        before = bytes_on_disk(root)
        time.sleep(args.sample_seconds)
        rate = (bytes_on_disk(root) - before) / args.sample_seconds / 1e6
        if rate > 0.5:
            print(f"throughput : {rate:.0f} MB/s (sampled {args.sample_seconds}s)")
            remaining = val_missing + chunk_missing
            if remaining > 0:
                print(f"ETA        : {humanize(remaining / 1e6 / rate)} until this chunk is ready"
                      + (" (first training step)" if cursor is None else ""))
            else:
                print("ETA        : this chunk is complete -- training, or prefetching ahead")
        else:
            print(f"throughput : idle (<0.5 MB/s over {args.sample_seconds}s)")
            if val_missing + chunk_missing > 0:
                print("             nothing arriving but data is missing -- check the log for errors")
            else:
                print("             expected: this chunk is fully downloaded, so training should be running")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report download progress of a rotating mdCATH run.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/mdcath_backbone_rotate.yaml"))
    parser.add_argument("--manifest", default=None, help="Override the config's manifest path.")
    parser.add_argument("--local-dir", default=None, help="Override the config's rotation.local_dir.")
    parser.add_argument(
        "--log", type=Path, default=REPO_ROOT / "logs/mdcath_backbone_rotate.log",
        help="Run log, used to detect which chunk is current.",
    )
    parser.add_argument("--chunk", type=int, default=None, help="Report a specific chunk instead.")
    parser.add_argument(
        "--sample-seconds", type=float, default=15.0,
        help="Seconds to sample for a throughput estimate (0 to skip).",
    )
    parser.add_argument("--no-rate", action="store_true", help="Same as --sample-seconds 0.")
    parser.add_argument("--watch", type=float, default=0, help="Repeat every N seconds.")
    args = parser.parse_args()
    if args.no_rate:
        args.sample_seconds = 0

    while True:
        if args.watch:
            print("\033[2J\033[H", end="")  # clear, so successive reports do not scroll away
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
        report(args)
        if not args.watch:
            return
        time.sleep(max(args.watch - args.sample_seconds, 1))


if __name__ == "__main__":
    main()
