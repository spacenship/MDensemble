"""Which mdCATH shards a rotating run trains on, and in what order.

The full mdCATH dataset is 5,398 domains / 3.61 TB (measured via
``HfApi.list_repo_tree``), so a run cannot hold it on disk. Training
therefore rotates: download a chunk of domains, train on it, delete it, move
on. This module decides the *plan* -- the validation holdout and the chunk
composition -- and freezes it into a JSON file.

Freezing matters for three reasons:

* **Reproducibility.** The chunk a domain lands in is fixed once, not
  re-derived from a Hub listing that may change between runs.
* **Resumption.** A run interrupted after 20 chunks resumes by reading the
  same file; it never has to re-query the Hub to know what comes next.
* **Exact disk arithmetic.** Every entry carries its byte size, so chunks
  can be capped by real bytes rather than by a guessed average. Shard sizes
  vary 3.7x (median 564 MB, max 2.51 GB), so a count-only split would
  produce chunks that overshoot the disk budget.

The validation domains are held out *once*, globally, and are never part of
any chunk: a rotating run must compare validation loss across chunks (for
ReduceLROnPlateau and best-checkpoint selection), which only means something
if the validation set never changes.
"""
from __future__ import annotations

import datetime as _datetime
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "compsciencelab/mdCATH"
_FILENAME_PREFIX = "mdcath_dataset_"
_FILENAME_SUFFIX = ".h5"


def domain_from_path(repo_path: str) -> str:
    """``data/mdcath_dataset_1aocA00.h5`` -> ``1aocA00``."""
    name = Path(repo_path).name
    if not (name.startswith(_FILENAME_PREFIX) and name.endswith(_FILENAME_SUFFIX)):
        raise ValueError(f"Not an mdCATH shard filename: {repo_path!r}")
    return name[len(_FILENAME_PREFIX) : -len(_FILENAME_SUFFIX)]


@dataclass(frozen=True)
class ShardEntry:
    """One mdCATH domain: its CATH id, repo-relative path, and byte size."""

    domain: str
    path: str
    size: int


@dataclass
class ShardManifest:
    """A frozen rotation plan: the validation holdout plus ordered chunks."""

    repo_id: str
    seed: int
    created: str
    val_domains: List[ShardEntry]
    chunks: List[List[ShardEntry]]

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def num_train_domains(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)

    @property
    def num_domains(self) -> int:
        return self.num_train_domains + len(self.val_domains)

    def chunk_bytes(self, chunk_index: int) -> int:
        return sum(entry.size for entry in self.chunks[chunk_index])

    @property
    def val_bytes(self) -> int:
        return sum(entry.size for entry in self.val_domains)

    @property
    def total_bytes(self) -> int:
        return self.val_bytes + sum(self.chunk_bytes(i) for i in range(self.num_chunks))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "repo_id": self.repo_id,
            "seed": self.seed,
            "created": self.created,
            "val_domains": [asdict(entry) for entry in self.val_domains],
            "chunks": [[asdict(entry) for entry in chunk] for chunk in self.chunks],
        }
        with path.open("w") as fh:
            json.dump(payload, fh, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> "ShardManifest":
        with Path(path).open("r") as fh:
            payload = json.load(fh)
        return cls(
            repo_id=payload["repo_id"],
            seed=payload["seed"],
            created=payload["created"],
            val_domains=[ShardEntry(**entry) for entry in payload["val_domains"]],
            chunks=[[ShardEntry(**entry) for entry in chunk] for chunk in payload["chunks"]],
        )

    def summary(self) -> str:
        sizes = [self.chunk_bytes(i) / 1e9 for i in range(self.num_chunks)]
        return (
            f"{self.num_domains} domains ({self.total_bytes / 1e12:.2f} TB): "
            f"{len(self.val_domains)} held out for validation ({self.val_bytes / 1e9:.0f} GB), "
            f"{self.num_train_domains} in {self.num_chunks} chunks "
            f"({min(sizes):.0f}-{max(sizes):.0f} GB each)"
        )


def list_repo_entries(repo_id: str = DEFAULT_REPO_ID, token: Optional[str] = None) -> List[ShardEntry]:
    """Every ``data/*.h5`` shard in the Hub dataset repo, with byte sizes."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    entries = []
    for item in api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo="data", recursive=False):
        path = item.path
        size = getattr(item, "size", None)
        if not path.endswith(_FILENAME_SUFFIX) or size is None:
            continue  # folders, and the repo-root metadata files
        entries.append(ShardEntry(domain=domain_from_path(path), path=path, size=int(size)))
    if not entries:
        raise RuntimeError(f"No .h5 shards found in {repo_id}:data/")
    return entries


def list_local_entries(local_data_dir: str | Path) -> List[ShardEntry]:
    """Shards already on disk, sized by ``stat`` -- lets a manifest be built
    (and the rotation exercised) with no network at all."""
    entries = []
    for path in sorted(Path(local_data_dir).glob(f"*{_FILENAME_SUFFIX}")):
        entries.append(
            ShardEntry(domain=domain_from_path(path.name), path=f"data/{path.name}", size=path.stat().st_size)
        )
    if not entries:
        raise FileNotFoundError(f"No .h5 shards found under {local_data_dir}")
    return entries


def _split_into_chunks(
    entries: List[ShardEntry], chunk_size: int, chunk_max_bytes: Optional[int]
) -> List[List[ShardEntry]]:
    """Greedy split honouring both a domain count and a byte ceiling.

    The byte ceiling is what keeps a chunk within the disk budget when it
    happens to draw several of the 2 GB+ shards; a single oversized shard
    still forms a chunk of its own rather than being dropped.
    """
    chunks: List[List[ShardEntry]] = []
    current: List[ShardEntry] = []
    current_bytes = 0
    for entry in entries:
        exceeds_count = len(current) >= chunk_size
        exceeds_bytes = chunk_max_bytes is not None and current_bytes + entry.size > chunk_max_bytes
        if current and (exceeds_count or exceeds_bytes):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(entry)
        current_bytes += entry.size
    if current:
        chunks.append(current)
    return chunks


def build_manifest(
    entries: Iterable[ShardEntry],
    *,
    repo_id: str = DEFAULT_REPO_ID,
    seed: int = 0,
    chunk_size: int = 200,
    chunk_max_gb: Optional[float] = 160.0,
    num_val_domains: int = 25,
    num_domains: Optional[int] = None,
    prefer_local_dir: Optional[str | Path] = None,
) -> ShardManifest:
    """Freezes a rotation plan: validation holdout first, then ordered chunks.

    Args:
        entries: all candidate shards (from :func:`list_repo_entries` or
            :func:`list_local_entries`).
        seed: fixes the shuffle, hence the whole plan.
        chunk_size / chunk_max_gb: per-chunk domain count and byte ceilings.
        num_val_domains: domains held out globally and never rotated.
        num_domains: optionally train on a subset of the dataset rather than
            all 5,398 (the holdout is taken from within it).
        prefer_local_dir: directory of already-downloaded shards. Domains
            found there are ordered into the earliest chunks, so a run that
            already has shards on disk starts training without waiting on a
            download. The resulting order is saved in the manifest, so the
            plan stays reproducible even if that directory later changes.
    """
    # Sort before shuffling: Hub listing order is not guaranteed stable, and
    # the seed must be the only thing that determines the plan.
    ordered = sorted(entries, key=lambda entry: entry.path)
    if num_val_domains < 1:
        raise ValueError("num_val_domains must be >= 1")
    if len(ordered) <= num_val_domains:
        raise ValueError(
            f"Need more than num_val_domains={num_val_domains} shards to build a manifest, got {len(ordered)}"
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    rng = random.Random(seed)
    rng.shuffle(ordered)
    if num_domains is not None:
        if num_domains <= num_val_domains:
            raise ValueError("num_domains must exceed num_val_domains")
        ordered = ordered[:num_domains]

    val_domains = ordered[:num_val_domains]
    train_entries = ordered[num_val_domains:]

    if prefer_local_dir is not None:
        local_dir = Path(prefer_local_dir)
        present = {path.name for path in local_dir.glob(f"*{_FILENAME_SUFFIX}")}
        # Stable partition: keeps the seeded order within each group.
        already_here = [entry for entry in train_entries if Path(entry.path).name in present]
        remaining = [entry for entry in train_entries if Path(entry.path).name not in present]
        logger.info("Ordering %d already-downloaded domain(s) into the first chunks", len(already_here))
        train_entries = already_here + remaining

    chunk_max_bytes = int(chunk_max_gb * 1e9) if chunk_max_gb else None
    chunks = _split_into_chunks(train_entries, chunk_size, chunk_max_bytes)

    return ShardManifest(
        repo_id=repo_id,
        seed=seed,
        created=_datetime.datetime.now().isoformat(timespec="seconds"),
        val_domains=val_domains,
        chunks=chunks,
    )


def manifest_domain_counts(manifest: ShardManifest) -> Dict[str, int]:
    """Sanity counts used by tests and by the run's startup log."""
    return {
        "val": len(manifest.val_domains),
        "train": manifest.num_train_domains,
        "chunks": manifest.num_chunks,
    }
