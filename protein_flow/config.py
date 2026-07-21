"""Dataclass-based configuration for protein_flow.

This project intentionally avoids Hydra. Configuration is a plain nested
dataclass tree that can be constructed from a YAML file via
:func:`load_config` and serialized back via :func:`save_config`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, get_type_hints

import yaml


@dataclass
class DataConfig:
    """Describes the tensor schema and synthetic-data generation knobs."""

    plm_dim: int = 320
    num_amino_acid_types: int = 22
    min_length: int = 16
    max_length: int = 48
    train_size: int = 256
    val_size: int = 32
    batch_size: int = 8
    num_workers: int = 0
    seed: int = 0


@dataclass
class GraphConfig:
    """Dynamic geometric graph construction."""

    knn_k: int = 16
    use_radius_cutoff: bool = False
    radius_cutoff: float = 12.0
    num_rbf: int = 16
    rbf_min_dist: float = 0.0
    rbf_max_dist: float = 20.0


@dataclass
class SequenceEncoderConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    use_position_encoding: bool = True


@dataclass
class GeometricEncoderConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    update_coordinates: bool = False  # default: only update hidden features


@dataclass
class FusionConfig:
    hidden_dim: int = 128
    condition_dim: int = 64


@dataclass
class DecoderConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    remove_com_velocity: bool = True


@dataclass
class ModelConfig:
    sequence_encoder: SequenceEncoderConfig = field(default_factory=SequenceEncoderConfig)
    geometric_encoder: GeometricEncoderConfig = field(default_factory=GeometricEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


@dataclass
class FlowConfig:
    path_type: str = "linear"  # "linear" | "gaussian_bridge"
    sigma_min: float = 0.0  # only used by gaussian_bridge path


@dataclass
class PhysicsLossConfig:
    """Config for physics-informed regularization terms."""

    enabled: bool = True
    apply_to: str = "euler_step"  # "x_tau" | "euler_step"
    step_scale: float = 1.0
    d_ref_source: str = "source"  # "source" | "aligned_target"
    clash_threshold: float = 3.5
    clash_seq_sep: int = 2  # residues within this sequence separation are exempt from clash loss


@dataclass
class LossConfig:
    lambda_fm: float = 1.0
    lambda_bond: float = 1.0
    lambda_angle: float = 1.0
    lambda_clash: float = 1.0
    lambda_endpoint: float = 0.0
    endpoint_enabled: bool = False
    physics: PhysicsLossConfig = field(default_factory=PhysicsLossConfig)


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0


@dataclass
class TrainConfig:
    num_epochs: int = 2
    max_steps: Optional[int] = None
    log_every: int = 10
    val_every: int = 50
    ckpt_dir: str = "checkpoints"
    amp: bool = False
    device: str = "cpu"
    seed: int = 0
    overfit_one_batch: bool = False
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class SamplingConfig:
    num_steps: int = 50
    solver: str = "heun"  # "euler" | "heun"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


def _dict_to_dataclass(cls, data: Dict[str, Any]):
    if not is_dataclass(cls):
        return data
    kwargs = {}
    type_hints = get_type_hints(cls)
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = type_hints[f.name]
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a :class:`Config` from a YAML file, falling back to defaults for
    any field not present in the file."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return _dict_to_dataclass(Config, raw)


def save_config(config: Config, path: str | Path) -> None:
    """Serialize a :class:`Config` to YAML."""
    with open(path, "w") as fh:
        yaml.safe_dump(dataclasses.asdict(config), fh, sort_keys=False)
