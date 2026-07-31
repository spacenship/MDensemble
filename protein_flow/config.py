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
    """Describes the tensor schema and synthetic-data generation knobs.

    ``source`` selects between the synthetic dataset (default, always
    runnable with no external files) and the real mdCATH HDF5 adapter
    (:class:`protein_flow.data.mdcath.MdCathDataset`), configured via the
    ``mdcath_*`` fields below.
    """

    plm_dim: int = 320
    num_amino_acid_types: int = 22
    min_length: int = 16
    max_length: int = 48
    train_size: int = 256
    val_size: int = 32
    batch_size: int = 8
    num_workers: int = 0
    seed: int = 0

    source: str = "synthetic"  # "synthetic" | "mdcath"
    # Which particles carry the flow. "ca" is the original residue-level
    # C-alpha MVP; "heavy_atom" flows every non-hydrogen protein atom
    # (~7.9 per residue) using the real CHARMM covalent topology from the
    # shard's PSF (see protein_flow/data/topology.py). Hydrogens are
    # excluded: they are force-field-added, their positions are largely
    # slaved to the heavy atoms, and including them would double the graph.
    representation: str = "ca"  # "ca" | "heavy_atom"
    mdcath_dir: Optional[str] = None
    mdcath_frame_gap: int = 1
    # When comparing several gaps, reserve enough trailing frames for the
    # largest gap so every run samples the same source-frame pool.
    mdcath_sampling_max_frame_gap: Optional[int] = None
    mdcath_ps_per_frame: Optional[float] = None
    mdcath_val_fraction: float = 0.15
    mdcath_embedding_cache_dir: Optional[str] = None
    # Training draws new frame offsets whenever the dataset epoch changes;
    # validation keeps several fixed offsets per trajectory for reproducible,
    # broader coverage of each trajectory.
    mdcath_train_pairs_per_trajectory: int = 1
    mdcath_val_pairs_per_trajectory: int = 2
    mdcath_resample_train_each_epoch: bool = True


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
    # Adds the signed backbone-dihedral pseudo-scalar (protein_flow/geometry/chirality.py)
    # to the initial node features. This makes the model SE(3)-equivariant
    # (proper rotation + translation only) instead of E(3)-equivariant
    # (which would also treat mirror-image structures as equivalent, wrong
    # for real chiral proteins). Disable only for E(3) ablation experiments.
    use_chirality_features: bool = True


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
class EsmConfig:
    """In-graph ESM2 fine-tuning (see protein_flow/models/esm_encoder.py).

    When ``enabled``, the PLM runs inside the model and its weights are
    trained end-to-end with the flow-matching objective, replacing the
    precomputed ``mdcath_embedding_cache_dir`` embeddings.
    ``DataConfig.plm_dim`` must match the checkpoint's hidden size
    (640 for esm2_t30_150M); this is validated at model build time.
    """

    enabled: bool = False
    model_name: str = "facebook/esm2_t30_150M_UR50D"
    trainable: bool = True
    gradient_checkpointing: bool = True
    num_frozen_layers: int = 0
    # ESM is pretrained; the rest of the network is not. Training both at one
    # learning rate tends to wreck the PLM, so its parameter group gets its
    # own (typically much smaller) rate.
    learning_rate: float = 1e-5


@dataclass
class ModelConfig:
    sequence_encoder: SequenceEncoderConfig = field(default_factory=SequenceEncoderConfig)
    geometric_encoder: GeometricEncoderConfig = field(default_factory=GeometricEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    esm: EsmConfig = field(default_factory=EsmConfig)


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
    # Separate threshold for the all-atom representation. Measured on real
    # mdCATH frames: no non-bonded heavy-atom pair (excluding 1-2 and 1-3
    # covalent neighbours) comes closer than 2.51 A, so 2.5 penalizes only
    # genuinely non-physical overlap. Reusing the 3.5 A C-alpha value here
    # would instead penalize a large fraction of perfectly normal contacts.
    clash_threshold_heavy_atom: float = 2.5


@dataclass
class EndpointRolloutConfig:
    """Differentiable ODE rollout used by the endpoint loss.

    This is the expensive term: it unrolls the flow ODE inside the training
    step and backpropagates through every model evaluation. Three knobs keep
    it affordable.

    ``num_steps`` is deliberately separate from ``SamplingConfig.num_steps``
    (which governs inference-time quality): backpropagating through 50
    evaluations of a 200M+ parameter model is not practical, while a coarse
    5-10 step rollout still gives a useful endpoint signal.

    ``gradient_checkpointing`` recomputes each step's activations during the
    backward pass instead of storing all of them, turning rollout activation
    memory from O(num_steps) into roughly O(1) for ~2x forward compute. This
    is what makes many-step unrolling fit at all.

    ``backprop_last_steps`` implements truncated backpropagation through
    time: earlier steps still run (so the trajectory is correct) but are
    detached, so gradient only flows through the final K evaluations.
    ``None`` backpropagates through the whole rollout.
    """

    num_steps: int = 8
    solver: str = "euler"  # "euler" | "heun"
    gradient_checkpointing: bool = True
    backprop_last_steps: Optional[int] = None


@dataclass
class LossConfig:
    lambda_fm: float = 1.0
    lambda_bond: float = 1.0
    lambda_angle: float = 1.0
    lambda_clash: float = 1.0
    lambda_endpoint: float = 0.0
    endpoint_enabled: bool = False
    # Also apply the bond/angle/clash terms to the rollout endpoint, which is
    # where non-physical geometry produced by straight-line interpolation
    # actually shows up.
    endpoint_physics_enabled: bool = False
    lambda_endpoint_physics: float = 1.0
    physics: PhysicsLossConfig = field(default_factory=PhysicsLossConfig)
    endpoint_rollout: EndpointRolloutConfig = field(default_factory=EndpointRolloutConfig)


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    min_lr: float = 1e-6


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
    # A fixed grid removes checkpoint-selection noise caused by randomly
    # sampling flow time during validation.
    val_tau_values: list[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    # Cap the number of validation batches per evaluation. The full mdCATH
    # validation split is ~375 batches, and every batch is evaluated at each
    # tau in val_tau_values, so an uncapped pass costs ~1000 forwards. Under
    # DDP that whole cost lands on rank 0 while the other ranks wait, so a
    # cap keeps the training loop from stalling. None = evaluate everything.
    val_max_batches: Optional[int] = None
    # Endpoint rollout is an evaluation metric only. Limiting it to a fixed,
    # seeded batch subset keeps validation cost bounded and reproducible.
    val_endpoint_enabled: bool = False
    val_endpoint_num_steps: int = 10
    val_endpoint_max_batches: int = 10
    val_endpoint_solver: str = "heun"
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


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path, stack: tuple[Path, ...] = ()) -> Dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"Circular base_config chain: {chain}")
    with resolved.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {resolved}")

    base_config = raw.pop("base_config", None)
    if base_config is None:
        return raw
    if not isinstance(base_config, str):
        raise ValueError(f"base_config must be a path string: {resolved}")
    base_raw = _load_raw_config((resolved.parent / base_config).resolve(), (*stack, resolved))
    return _deep_merge(base_raw, raw)


def load_config(path: str | Path) -> Config:
    """Load a :class:`Config` from a YAML file, falling back to defaults for
    any field not present in the file. A config may set ``base_config`` to a
    YAML path relative to itself; nested mappings are recursively merged."""
    raw = _load_raw_config(Path(path))
    return _dict_to_dataclass(Config, raw)


def save_config(config: Config, path: str | Path) -> None:
    """Serialize a :class:`Config` to YAML."""
    with open(path, "w") as fh:
        yaml.safe_dump(dataclasses.asdict(config), fh, sort_keys=False)
