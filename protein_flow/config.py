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
class RotationConfig:
    """Trains over more mdCATH shards than fit on disk, a chunk at a time.

    The full mdCATH dataset is 5,398 domains / 3.61 TB, so it cannot be
    staged locally. With rotation enabled the run downloads one chunk of
    domains, trains ``steps_per_chunk`` optimizer steps on it, deletes it,
    and moves to the next -- prefetching the next chunk in the background so
    the GPU does not idle waiting on the network.

    Which domains land in which chunk is fixed once, in a manifest JSON
    (:mod:`protein_flow.data.shard_manifest`), so a run is reproducible and
    can resume mid-rotation without re-querying the Hub.

    Disk arithmetic (measured: mean shard 670 MB, max 2.51 GB): with
    ``chunk_size: 200`` a chunk is ~134 GB, so the resident set is
    val (~17 GB) + current + prefetched ~= 285 GB. Raising ``chunk_size``
    to 500 makes a chunk ~335 GB, which needs ``prefetch_chunks: 0`` to
    stay within a ~1 TB disk.
    """

    enabled: bool = False
    repo_id: str = "compsciencelab/mdCATH"
    manifest_path: Optional[str] = None
    # Where chunks are downloaded to. The shards themselves land in
    # ``{local_dir}/data/`` (mirroring the repo layout), which is what
    # ``DataConfig.mdcath_dir`` should point at.
    local_dir: Optional[str] = None
    steps_per_chunk: int = 2000
    # Preferred alternative to steps_per_chunk: budget the chunk in whole
    # passes over its data. A step budget that is not a multiple of the pass
    # length truncates the final pass (2000 steps over a 625-step pass runs
    # 625+625+625+125), and the pass length is not something the config can
    # pin down -- it moves with batch_size, world size, and how many
    # trajectories a given chunk actually yields. Counting passes ends on a
    # boundary by construction. Takes precedence when set.
    passes_per_chunk: Optional[int] = None
    # How many times to loop over every chunk. One cycle = one pass over the
    # whole dataset = the full download volume (3.61 TB for all of mdCATH).
    num_cycles: int = 1
    prefetch_chunks: int = 1
    delete_after_use: bool = True
    # Chunk indices never deleted (e.g. [0] when chunk 0 holds shards that
    # were already on disk before the run and should stay).
    keep_resident: list[int] = field(default_factory=list)
    # Prefetching pauses (with a warning) rather than filling the disk.
    min_free_gb: float = 150.0
    download_retries: int = 3
    # How long an isolated downloader may run before it is presumed hung and
    # killed, so the run continues with whatever shards reached disk instead
    # of blocking forever. Measured once at 0: a stalled socket inside the
    # downloader held both training ranks in subprocess.wait() for 2.5 days
    # with their CUDA contexts allocated and the GPUs at 0%. Nothing else
    # catches this -- the NCCL watchdog only sees collectives, and the
    # launcher's restart loop only fires when a process exits. A 146 GB chunk
    # at the measured ~50 MB/s takes ~50 min, so 180 is ~3.6x headroom.
    download_timeout_minutes: int = 180
    # Open each downloaded shard with h5py before use. A truncated file would
    # otherwise surface as a confusing dataset-indexing failure much later.
    verify_downloads: bool = True


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
    # "backbone" is the middle ground: exactly the N/CA/C/O of every residue
    # (4 particles, verified exact on real shards), which keeps real peptide
    # geometry -- bond, angle and carbonyl orientation -- at roughly half
    # the particle count of "heavy_atom".
    representation: str = "ca"  # "ca" | "backbone" | "heavy_atom"
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
    # Skip domains longer than this at dataset-indexing time. The 100-shard
    # sample tops out at 479 residues, but the full 5,398-domain dataset has
    # a long tail; without a cap a single huge domain can OOM a run hours in.
    mdcath_max_residues: Optional[int] = None
    rotation: RotationConfig = field(default_factory=RotationConfig)


def is_atom_level(representation: str) -> bool:
    """True when the flowing particles are atoms rather than whole residues.

    Both ``backbone`` and ``heavy_atom`` use the atom-level batch layout
    (atom_mask / atom_residue_index / bond_index / ...); they differ only in
    which atoms are selected. ``ca`` is the residue-level special case.
    """
    if representation not in ("ca", "backbone", "heavy_atom"):
        raise ValueError(
            f"representation must be 'ca', 'backbone' or 'heavy_atom', got {representation!r}"
        )
    return representation != "ca"


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
    # Recompute each EGNN layer's activations during the backward pass rather
    # than storing them. At atom resolution these dominate the model's memory:
    # measured at 512 residues (2,048 backbone particles, k=16) and batch 16,
    # dropping from 6 layers to 3 freed 16.6 GiB, i.e. ~5.5 GiB per layer.
    # Costs roughly one extra forward over the encoder. Off by default so
    # existing C-alpha runs are unaffected; turn it on to trade time for batch.
    gradient_checkpointing: bool = False


@dataclass
class FusionConfig:
    hidden_dim: int = 128
    condition_dim: int = 64
    # Ranges used to normalise the conditioning scalars onto [0, embedding_scale]
    # before they are sinusoidally embedded. The defaults span mdCATH's five
    # simulation temperatures. These are not cosmetic: sinusoidal_embedding
    # only resolves inputs spanning hundreds of units, so feeding it raw values
    # (tau in [0, 1], or Kelvin) yields a condition vector that carries no
    # information at all -- see ConditionEncoder's docstring for the measured
    # collapse this caused.
    temperature_min: float = 320.0
    temperature_max: float = 450.0
    delta_t_max: float = 1000.0
    embedding_scale: float = 1000.0
    # FiLM modulation of the fused representation by the condition. The gate
    # alone is one scalar in (0, 1) per particle and so cannot change the
    # scale -- let alone the sign -- of the predicted velocity, which is
    # exactly what tau-dependent flow matching needs. Disable only to
    # reproduce the pre-FiLM behaviour for an ablation.
    film_conditioning: bool = True


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
    # "linear" / "gaussian_bridge" flow structure -> structure, so the base
    # distribution is a point mass at the conditioning and the sampler is
    # deterministic. "displacement" flows noise -> (x1 - x0) with x0 as a
    # side input, which is what makes the model generative (and ensemble
    # metrics definable). See protein_flow/flow/paths.py.
    path_type: str = "linear"  # "linear" | "gaussian_bridge" | "displacement"
    sigma_min: float = 0.0  # only used by gaussian_bridge path
    # Per-axis standard deviation of the displacement path's base
    # distribution, in angstroms. Default matches the measured backbone
    # displacement over the 1,250-pair holdout: RMS|delta| = 4.620 A, i.e.
    # 4.620 / sqrt(3) per axis. Deliberately one number for all temperatures
    # -- the 3.9x spread from 320 K to 450 K is what the conditioning has to
    # learn, and matching sigma per temperature would hand it over for free.
    noise_scale: float = 2.667  # only used by displacement path
    # How many k-NN hops to smooth the base noise over before using it, which
    # gives it the spatial correlation real protein displacements have (0.96
    # within 4 A, 0.60 at 6-8 A, negative beyond 16 A; white noise has none).
    # Fitting the base distribution's C(r) to MD's takes 8 rounds.
    #
    # Default 0 anyway: measured head-to-head, 8 rounds trained worse on every
    # distributional metric *and* on physics. It conflicts with the
    # ||s_j - s_i|| edge feature, which measures what smoothing removes. See
    # configs/mdcath_backbone_rotate_displacement.yaml for the numbers.
    noise_smoothing_rounds: int = 0
    # Keep the whole flow inside the subspace the target actually lives in:
    # project both the base noise and the predicted velocity onto zero linear
    # *and* angular momentum about the source structure's centroid.
    #
    # ``delta`` is defined by Kabsch-aligning x1 onto x0, so it has no
    # rigid-body component at all -- re-aligning it removes 0.0% of its energy,
    # against 74.3% before alignment. The decoder already projects out the
    # three translations; nothing projected out the three rotations, and the
    # first full run spent 17.0% of its output energy on them, which the
    # evaluation's own Kabsch step then discards. The base noise is not the
    # source: isotropic noise carries only 3/(3N-3), measured 0.4%. See
    # scripts/diagnostics/rigid_share.py and README section 19.
    #
    # One flag drives both ends on purpose. Projecting the velocity while
    # leaving the noise unprojected would strand the noise's own rotational
    # component in the state forever, since nothing else can remove it.
    #
    # Default off so the completed run stays reproducible; on in
    # configs/mdcath_backbone_rotate_displacement_norigid.yaml.
    remove_rigid_motion: bool = False


@dataclass
class PhysicsLossConfig:
    """Config for physics-informed regularization terms."""

    enabled: bool = True
    apply_to: str = "euler_step"  # "x_tau" | "euler_step"
    step_scale: float = 1.0
    # How far along the predicted velocity the physics terms are evaluated,
    # when apply_to == "euler_step".
    #
    #   "remaining" -- x_tau + step_scale * (1 - tau) * v   (default)
    #   "constant"  -- x_tau + step_scale * v
    #
    # "remaining" is the only choice consistent with the flow-matching target.
    # With the linear path, v = x1 - x0 is the correct answer for every tau,
    # and x_tau + (1 - tau) * v lands exactly on x1 -- a real frame, whose
    # geometry the physics terms should find unobjectionable. "constant"
    # instead lands on x1 + tau * (x1 - x0), an overshoot past x1 that grows
    # with tau, so it charges the *correct* velocity an ever-larger penalty:
    # measured on real backbone frames, the bond term against the ground-truth
    # velocity rose from 0.0049 at tau=0.05 to 1.1302 at tau=0.95, versus a
    # flat 0.0023 under "remaining". It is kept only to reproduce old runs.
    step_scale_mode: str = "remaining"  # "remaining" | "constant"
    d_ref_source: str = "source"  # "source" | "aligned_target"
    clash_threshold: float = 3.5
    clash_seq_sep: int = 2  # residues within this sequence separation are exempt from clash loss
    # Separate threshold for the all-atom representation. Measured on real
    # mdCATH frames: no non-bonded heavy-atom pair (excluding 1-2 and 1-3
    # covalent neighbours) comes closer than 2.51 A, so 2.5 penalizes only
    # genuinely non-physical overlap. Reusing the 3.5 A C-alpha value here
    # would instead penalize a large fraction of perfectly normal contacts.
    clash_threshold_heavy_atom: float = 2.5
    # Same idea for the backbone representation, measured the same way over
    # 45 sample domains x 6 frames x 2 temperatures: the closest non-bonded
    # backbone pair the k-NN clash term actually sees is 2.278 A, so 2.2
    # leaves real geometry unpenalized. Note this is *lower* than the
    # heavy-atom value: with 4 particles per residue the k=16 neighbourhood
    # reaches further along the chain and picks up 1-4 pairs (e.g. O(i) to
    # CA(i+1)) that the denser all-atom graph never includes as edges.
    clash_threshold_backbone: float = 2.2

    def clash_threshold_for(self, representation: str) -> float:
        """The threshold matching the particles actually being flowed.

        Picking this by hand at each call site is how a run silently ends up
        penalizing every normal contact (C-alpha's 3.5 A applied to atoms) or
        none at all (2.2 A applied to C-alphas).
        """
        if representation == "backbone":
            return self.clash_threshold_backbone
        if representation == "heavy_atom":
            return self.clash_threshold_heavy_atom
        return self.clash_threshold


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
    # Weight on the scale-free cosine term. MSE alone lets the model buy a
    # lower loss by shrinking toward zero whenever the target is dominated by
    # unpredictable thermal noise; this term keeps paying for direction.
    # 0.0 keeps the historical objective exactly.
    lambda_direction: float = 0.0
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
    # Autocast dtype. "float16" tops out at 65504, and the forward pass really
    # does reach it: on a run whose weights had grown to max|param| 26.7, 23 of
    # 60 training batches produced a non-finite loss under fp16 and 0 of 60
    # under bf16, with identical finite losses (29.82) where both survived --
    # the arithmetic is the same, only the range differs. Validation was
    # unaffected throughout because it runs outside autocast, which is exactly
    # what makes this failure look like bad data rather than a dtype ceiling.
    #
    # Default stays float16 so completed runs reproduce; new runs should set
    # bfloat16, which has fp32's exponent range and is native on this hardware.
    amp_dtype: str = "float16"  # "float16" | "bfloat16"
    device: str = "cpu"
    seed: int = 0
    # Where to resume from: "auto" picks up {ckpt_dir}/last.pt when it
    # exists, a path resumes from that file, null always starts fresh.
    # A rotating full-dataset run downloads terabytes, so being able to
    # continue an interrupted one is not optional.
    resume: Optional[str] = None
    # How long a rank waits at a collective before the NCCL watchdog aborts
    # the run. This is a *deadlock* detector, so it only has to exceed the
    # longest legitimate gap between two ranks reaching the same collective.
    #
    # The binding constraint is the chunk barrier in train_rotating, which
    # each rank reaches only after ShardPool.ensure() has its slice on disk.
    # An earlier note here assumed that gap was "minutes, because the ranks
    # download equal-sized slices in parallel" -- wrong, and it cost a run:
    # at the chunk 11->12 rotation rank 0's slice was already resident and
    # cleared ensure() in 0 s while rank 1 still had a real download, so rank
    # 0 sat at the barrier alone and the watchdog aborted at exactly 1800 s.
    # The skew between ranks is a whole download, not a scheduling jitter.
    #
    # So this has to stay above data.rotation.download_timeout_minutes, which
    # is the real bound on how long ensure() can block -- validate_config
    # enforces it. That ordering also puts the two watchdogs in the right
    # order: the downloader's fires first and degrades gracefully (kill the
    # child, keep the shards that landed, carry on), and this one is left to
    # catch genuine deadlocks, which is what it is for.
    dist_timeout_minutes: int = 210
    overfit_one_batch: bool = False
    # A fixed grid removes checkpoint-selection noise caused by randomly
    # sampling flow time during validation.
    val_tau_values: list[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    # Cap the number of validation batches per evaluation. The full mdCATH
    # validation split is ~375 batches, and every batch is evaluated at each
    # tau in val_tau_values, so an uncapped pass costs ~1000 forwards plus
    # the endpoint rollouts. This is a *global* budget: under DDP it is
    # divided across ranks, so the cap means the same amount of validation
    # work no matter how many GPUs are used. None = evaluate everything.
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
    config = _dict_to_dataclass(Config, raw)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Reject combinations that would silently misbehave rather than fail."""
    valid_paths = ("linear", "gaussian_bridge", "displacement")
    if config.flow.path_type not in valid_paths:
        raise ValueError(f"flow.path_type must be one of {valid_paths}, got {config.flow.path_type!r}")

    if config.flow.path_type == "displacement":
        if config.flow.noise_scale <= 0.0:
            raise ValueError(
                f"flow.noise_scale must be positive for the displacement path, "
                f"got {config.flow.noise_scale}"
            )
        if config.flow.noise_smoothing_rounds < 0:
            raise ValueError(
                f"flow.noise_smoothing_rounds must be >= 0, got "
                f"{config.flow.noise_smoothing_rounds}"
            )
        if config.loss.endpoint_enabled or config.train.val_endpoint_enabled:
            # _differentiable_rollout and the validation rollout both integrate
            # in coordinate space from x0 and never build a flow state, so they
            # would call the model with the argument it was sized for missing.
            raise ValueError(
                "flow.path_type='displacement' is incompatible with the endpoint rollout "
                "(loss.endpoint_enabled / train.val_endpoint_enabled): that rollout integrates "
                "in coordinate space. Set both to false; use scripts/evaluate_ensemble.py for "
                "sampled-structure metrics instead."
            )

    if config.data.rotation.enabled:
        # ShardPool.ensure() blocks a rank for up to download_timeout_minutes
        # while the other rank waits at the chunk barrier, so a shorter NCCL
        # timeout guarantees the barrier aborts the run before the downloader
        # watchdog can fire and recover. See TrainConfig.dist_timeout_minutes.
        if config.train.dist_timeout_minutes <= config.data.rotation.download_timeout_minutes:
            raise ValueError(
                f"train.dist_timeout_minutes ({config.train.dist_timeout_minutes}) must exceed "
                f"data.rotation.download_timeout_minutes "
                f"({config.data.rotation.download_timeout_minutes}): a rank can sit in "
                f"ShardPool.ensure() for the whole download timeout, and the other rank is "
                f"waiting at the chunk barrier that entire time."
            )


def save_config(config: Config, path: str | Path) -> None:
    """Serialize a :class:`Config` to YAML."""
    with open(path, "w") as fh:
        yaml.safe_dump(dataclasses.asdict(config), fh, sort_keys=False)
