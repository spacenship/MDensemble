"""Generic utilities: scalar embeddings and scatter-aggregation fallbacks.

``torch_scatter`` is an optional dependency. When it is not installed, every
scatter operation here falls back to a pure-PyTorch implementation based on
``index_add_`` / ``scatter_add_`` so the rest of the codebase never needs to
branch on availability.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

try:
    import torch_scatter  # type: ignore

    _HAS_TORCH_SCATTER = True
except ImportError:  # pragma: no cover - exercised whenever torch_scatter is absent
    _HAS_TORCH_SCATTER = False


def scatter_sum(src: Tensor, index: Tensor, dim_size: int, dim: int = 0) -> Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index``.

    Args:
        src: [N, ...] source values.
        index: [N] destination bucket for each row along ``dim``.
        dim_size: number of output buckets.
        dim: dimension along which to scatter (rows of src / entries of index).

    Returns:
        Tensor of shape ``(dim_size, *src.shape[1:])``.
    """
    if _HAS_TORCH_SCATTER:
        return torch_scatter.scatter_add(src, index, dim=dim, dim_size=dim_size)
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = src.new_zeros(out_shape)
    index_expanded = index
    for _ in range(src.dim() - 1):
        index_expanded = index_expanded.unsqueeze(-1)
    index_expanded = index_expanded.expand_as(src)
    out.scatter_add_(dim, index_expanded, src)
    return out


def scatter_mean(src: Tensor, index: Tensor, dim_size: int, dim: int = 0) -> Tensor:
    """Mean-aggregate ``src`` rows into ``dim_size`` buckets, 0 for empty buckets."""
    summed = scatter_sum(src, index, dim_size=dim_size, dim=dim)
    ones = src.new_ones(index.shape[0])
    counts = scatter_sum(ones, index, dim_size=dim_size, dim=0).clamp(min=1.0)
    count_shape = [dim_size] + [1] * (src.dim() - 1)
    return summed / counts.view(count_shape)


def sinusoidal_embedding(x: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Fourier/sinusoidal embedding of scalar values.

    Args:
        x: [...] arbitrary-shaped tensor of scalar values.
        dim: output embedding dimension (must be >= 2).
        max_period: controls the lowest frequency.

    Returns:
        Tensor of shape [..., dim].
    """
    if dim < 2:
        raise ValueError("sinusoidal_embedding requires dim >= 2")
    half = dim // 2
    freq_index = torch.arange(half, device=x.device, dtype=x.dtype)
    freqs = torch.exp(-math.log(max_period) * freq_index / half)
    args = x.unsqueeze(-1) * freqs
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    return embedding


def masked_mean(values: Tensor, mask: Tensor, dim: Optional[int] = None, eps: float = 1e-8) -> Tensor:
    """Mean of ``values`` over positions where ``mask`` is True.

    ``mask`` is broadcast against ``values``; reduction happens over ``dim``
    (all dims if ``dim`` is None).
    """
    mask = mask.to(dtype=values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    if dim is None:
        return (values * mask).sum() / (mask.sum() + eps)
    return (values * mask).sum(dim=dim) / (mask.sum(dim=dim) + eps)
