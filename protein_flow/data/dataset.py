"""Generic protein-trajectory dataset interface.

This intentionally makes no assumption about any specific on-disk storage
format (e.g. mdCATH's actual HDF5 layout is not assumed here). Any dataset
implementation just needs to yield per-sample dicts with the following
keys, each for a single (unpadded, variable-length) protein of length L:

    sequence_embedding: FloatTensor [L, D_plm]  -- precomputed PLM residue embeddings.
    source_coords:      FloatTensor [L, 3]      -- C-alpha coordinates at time t.
    target_coords:      FloatTensor [L, 3]      -- C-alpha coordinates at time t + delta_t.
    residue_types:      LongTensor  [L]         -- amino-acid type index per residue.
    temperature:        FloatTensor [1].
    physical_delta_t:   FloatTensor [1]         -- physical MD time gap (not flow time tau).

:mod:`protein_flow.data.collate` turns a list of such samples into a padded
batch (adding ``residue_mask``). Real dataset adapters (mdCATH, ATLAS, ...)
should subclass :class:`ProteinTrajectoryDataset` and implement
``__getitem__``/``__len__`` to return dicts of this shape -- everything
downstream (Kabsch alignment, graph construction, flow matching, physics
losses) only depends on this schema, not on the storage format.

The residue-level schema here is a stepping stone: a future backbone
extension would instead yield ``[L, 4, 3]`` (N, CA, C, O) or local-frame
(rotation + translation per residue) tensors; the Dataset/collate
contract is designed so that swapping the coordinate representation does
not require changing anything in ``geometry/`` or ``flow/``.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, List

from torch.utils.data import Dataset


class ProteinTrajectoryDataset(Dataset, abc.ABC):
    """Abstract base: yields per-sample dicts following the schema above."""

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError


class ListProteinTrajectoryDataset(ProteinTrajectoryDataset):
    """Trivial wrapper around an in-memory list of pre-built sample dicts."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]
