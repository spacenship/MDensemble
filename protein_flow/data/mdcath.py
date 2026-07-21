"""Real mdCATH HDF5 dataset adapter.

This reads actual mdCATH shard files (one CATH domain per ``.h5`` file,
verified against local sample files under ``mdCATH_sample100/data/``) and
exposes them through the same generic schema as
:class:`protein_flow.data.synthetic.SyntheticProteinTrajectoryDataset` (see
:mod:`protein_flow.data.dataset`).

Verified (by directly inspecting sample files), not assumed:
  - Each file has one top-level group keyed by the CATH domain id, with
    attrs ``numChains``/``numProteinAtoms``/``numResidues``.
  - Per-atom arrays ``chain``, ``element``, ``resid``, ``resname``, ``z``
    (length ``numProteinAtoms``) describe the topology directly.
  - ``pdbProteinAtoms`` is an embedded PDB-format text blob whose ATOM
    records are in the exact same order as those per-atom arrays (checked:
    resname/resid parsed from this PDB text match the ``resname``/``resid``
    arrays exactly for every sample file inspected). This lets us find the
    true C-alpha atom (PDB atom name == "CA", using fixed-column PDB
    parsing so e.g. CHARMM's "CAY" cap atom is never confused with "CA").
  - Below the domain group are subgroups keyed by simulation temperature
    in Kelvin (e.g. "320", "348", ...), each containing subgroups keyed by
    replica index ("0".."4"), each with a ``coords`` dataset of shape
    ``[num_frames, num_atoms, 3]`` (Angstrom) and a ``numFrames`` attr.

NOT verified / explicitly not assumed here: the physical time gap between
consecutive saved frames. We found no per-frame timestamp in the files
themselves, and secondary sources disagreed with each other and with the
frame counts actually observed in these sample files. Rather than silently
hard-coding a possibly-wrong number, ``ps_per_frame`` must be supplied
explicitly by the caller if physical units are needed; otherwise
``physical_delta_t`` is reported in raw frame-count units (see below).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from protein_flow.config import DataConfig
from protein_flow.data.dataset import ProteinTrajectoryDataset
from protein_flow.data.residue_vocab import resname_to_index

logger = logging.getLogger(__name__)

try:
    import h5py

    _HAS_H5PY = True
except ImportError:  # pragma: no cover - exercised only when h5py is absent
    _HAS_H5PY = False

_NON_TRAJECTORY_KEYS = {"chain", "element", "pdb", "pdbProteinAtoms", "psf", "resid", "resname", "z"}


def parse_ca_indices_and_resnames(pdb_text_bytes: bytes) -> Tuple[np.ndarray, List[str]]:
    """Parses an embedded PDB text blob (ATOM records only, one line per
    atom, same order as the file's per-atom arrays) into (CA atom indices,
    CA residue 3-letter codes), using fixed PDB columns so short names like
    "CA" are never confused with longer ones like "CAY"/"CAT"."""
    text = pdb_text_bytes.decode("utf-8", errors="replace")
    ca_indices: List[int] = []
    residue_names: List[str] = []
    atom_index = 0
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        if atom_name == "CA":
            ca_indices.append(atom_index)
            residue_names.append(resname)
        atom_index += 1
    return np.asarray(ca_indices, dtype=np.int64), residue_names


class MdCathDataset(ProteinTrajectoryDataset):
    """Yields real (source, target) C-alpha frame pairs from mdCATH shards.

    Each dataset item is one randomly-offset (seeded, so deterministic per
    index) frame pair drawn from one (domain, temperature, replica)
    trajectory. ``sequence_embedding`` is a real PLM embedding if a cached
    ``{domain}.pt`` tensor is found under ``embedding_cache_dir``,
    otherwise a deterministic per-domain placeholder (NOT a real PLM
    embedding -- ESM itself is never run here, consistent with the rest of
    this project).
    """

    def __init__(
        self,
        data_dir: str | Path,
        data_config: DataConfig,
        h5_files: Optional[List[Path]] = None,
        frame_gap: int = 1,
        ps_per_frame: Optional[float] = None,
        embedding_cache_dir: Optional[str | Path] = None,
        seed: int = 0,
    ):
        if not _HAS_H5PY:
            raise ImportError("MdCathDataset requires h5py: pip install h5py")
        if frame_gap < 1:
            raise ValueError("frame_gap must be >= 1")

        self.data_config = data_config
        self.frame_gap = frame_gap
        self.ps_per_frame = ps_per_frame
        self.seed = seed
        self.embedding_cache_dir = Path(embedding_cache_dir) if embedding_cache_dir else None

        if h5_files is None:
            h5_files = sorted(Path(data_dir).glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found under {data_dir}")

        self._domain_meta: Dict[str, Tuple[np.ndarray, torch.Tensor, int]] = {}
        self.trajectory_index: List[Dict[str, Any]] = []

        for path in h5_files:
            try:
                self._index_file(path)
            except Exception:  # noqa: BLE001 - one malformed shard shouldn't kill the whole dataset
                logger.exception("Skipping unreadable mdCATH shard: %s", path)

        if not self.trajectory_index:
            raise RuntimeError(f"No usable (domain, temperature, replica) trajectories found under {data_dir}")

    def _index_file(self, path: Path) -> None:
        with h5py.File(path, "r") as f:
            domain = next(iter(f.keys()))
            group = f[domain]

            if domain not in self._domain_meta:
                pdb_bytes = group["pdbProteinAtoms"][()]
                ca_indices, residue_names = parse_ca_indices_and_resnames(pdb_bytes)
                residue_types = torch.tensor([resname_to_index(r) for r in residue_names], dtype=torch.long)
                self._domain_meta[domain] = (ca_indices, residue_types, len(ca_indices))

            for key in group.keys():
                if key in _NON_TRAJECTORY_KEYS:
                    continue
                temperature_group = group[key]
                for replica_key in temperature_group.keys():
                    num_frames = int(temperature_group[replica_key].attrs["numFrames"])
                    if num_frames > self.frame_gap:
                        self.trajectory_index.append(
                            {
                                "path": path,
                                "domain": domain,
                                "temperature": key,
                                "replica": replica_key,
                                "num_frames": num_frames,
                            }
                        )

    def __len__(self) -> int:
        return len(self.trajectory_index)

    def _load_or_make_embedding(self, domain: str, length: int) -> torch.Tensor:
        if self.embedding_cache_dir is not None:
            cached_path = self.embedding_cache_dir / f"{domain}.pt"
            if cached_path.exists():
                embedding = torch.load(cached_path, map_location="cpu")
                if tuple(embedding.shape) == (length, self.data_config.plm_dim):
                    return embedding
                logger.warning(
                    "Cached embedding for %s has shape %s, expected (%d, %d); falling back to placeholder.",
                    domain, tuple(embedding.shape), length, self.data_config.plm_dim,
                )
        domain_seed = self.seed + sum(ord(c) for c in domain)
        local_generator = torch.Generator().manual_seed(domain_seed)
        return torch.randn(length, self.data_config.plm_dim, generator=local_generator)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.trajectory_index[index]
        ca_indices, residue_types, length = self._domain_meta[entry["domain"]]

        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        max_start = entry["num_frames"] - self.frame_gap - 1
        source_frame = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
        target_frame = source_frame + self.frame_gap

        with h5py.File(entry["path"], "r") as f:
            coords_dataset = f[entry["domain"]][entry["temperature"]][entry["replica"]]["coords"]
            source_full = coords_dataset[source_frame]
            target_full = coords_dataset[target_frame]

        source_coords = torch.from_numpy(np.asarray(source_full[ca_indices], dtype=np.float32))
        target_coords = torch.from_numpy(np.asarray(target_full[ca_indices], dtype=np.float32))

        temperature_value = float(entry["temperature"])  # verified: this key IS the simulation temperature in Kelvin
        if self.ps_per_frame is not None:
            delta_t_value = self.frame_gap * self.ps_per_frame
        else:
            delta_t_value = float(self.frame_gap)  # raw frame-count units; see module docstring

        return {
            "sequence_embedding": self._load_or_make_embedding(entry["domain"], length),
            "source_coords": source_coords,
            "target_coords": target_coords,
            "residue_types": residue_types,
            "temperature": torch.tensor([temperature_value]),
            "physical_delta_t": torch.tensor([delta_t_value]),
        }
