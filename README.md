# protein_dual_flow

A research prototype for modeling protein conformational dynamics with
**conditional flow matching** over a **dual-graph** (sequence + geometric)
representation of a protein, at C-alpha, backbone or all-atom resolution.

This is built to be run, not just read: `python train.py --config
configs/default.yaml` trains on a synthetic dataset in seconds on CPU,
`configs/mdcath_full.yaml` trains a 227M-parameter all-atom model on real
mdCATH trajectories, and `pytest` (168 tests) exercises every component,
including automated SE(3)-equivariance checks at both resolutions.

## 0. Repository map

```
protein_dual_flow/
├── configs/                     # dataclass-mirroring YAML (no Hydra)
│   ├── default.yaml             # synthetic data, CPU, seconds to run
│   ├── mdcath.yaml              # real mdCATH, C-alpha, small model
│   ├── mdcath_large.yaml        # + ~80M-parameter flow network
│   ├── mdcath_all_atom.yaml     # + all heavy atoms (real PSF topology)
│   ├── mdcath_esm.yaml          # + in-graph ESM2-150M fine-tuning
│   ├── mdcath_full.yaml         # + differentiable ODE unrolling  (everything)
│   ├── mdcath_backbone_rotate.yaml  # all of mdCATH, backbone atoms, rotated (§17)
│   ├── manifests/               # frozen rotation plans (generated)
│   └── ablations/               # frame-gap ablation configs
├── protein_flow/
│   ├── config.py                # the whole config tree as nested dataclasses
│   ├── data/
│   │   ├── dataset.py           # storage-agnostic Dataset interface
│   │   ├── synthetic.py         # runnable-anywhere synthetic trajectories
│   │   ├── mdcath.py            # real mdCATH HDF5 adapter
│   │   ├── topology.py          # covalent bonds/angles from CHARMM PSF
│   │   ├── shard_manifest.py    # frozen rotation plan: val holdout + chunks
│   │   ├── shard_rotation.py    # downloads/verifies/releases chunks on disk
│   │   ├── residue_vocab.py     # 3-letter <-> index <-> 1-letter
│   │   ├── esm_adapter.py       # offline ESM2 embedding precomputation
│   │   └── collate.py           # padding for residue, atom, and topology axes
│   ├── geometry/
│   │   ├── kabsch.py            # masked batched rigid-body alignment
│   │   ├── graph.py             # sequence graph + dynamic k-NN geometric graph
│   │   ├── chirality.py         # signed-dihedral pseudo-scalar (SE(3) vs E(3))
│   │   └── features.py          # RBF distance encoding
│   ├── models/
│   │   ├── sequence_encoder.py  # bidirectional typed sequence graph
│   │   ├── geometric_encoder.py # EGNN over the dynamic graph
│   │   ├── esm_encoder.py       # in-graph, fine-tunable ESM2
│   │   ├── fusion.py            # gated fusion + condition embedding
│   │   ├── vector_field.py      # equivariant velocity decoder
│   │   └── dual_graph_flow.py   # the full model
│   ├── flow/{paths,solver}.py   # conditional paths; Euler/Heun ODE solver
│   ├── losses/{flow_matching,physics}.py
│   ├── distributed.py           # DDP helpers (rank, barrier, unwrap)
│   ├── train.py                 # training loop, losses, validation metrics
│   ├── train_rotating.py        # outer loop for datasets too big for disk
│   └── inference.py
├── scripts/
│   ├── run_training_tmux.sh     # detached background training (see §11)
│   ├── build_mdcath_manifest.py # freeze a rotation plan (see §17)
│   ├── watch_rotation.sh        # summarize a rotating run's log (see §17)
│   └── run_mdcath_gap_ablation.sh
├── tests/                       # 168 tests
├── train.py / sample.py / precompute_esm_embeddings.py
└── README.md
```

## 1. What this model does

Given two structures of the *same* protein sampled at different points in a
trajectory (`source_coords` at time `t`, `target_coords` at time
`t + physical_delta_t`), the model learns a velocity field that
continuously deforms one structure into the other. At inference time,
integrating that velocity field from `t=0` to `t=1` (an ODE, not a single
forward pass) generates a plausible intermediate/next conformation starting
from `source_coords`.

The model conditions on:
- a **PLM residue embedding** per residue -- precomputed, or produced by an
  in-graph ESM2 that is fine-tuned end-to-end (§14),
- the **amino-acid type** per residue (and **chemical element** per atom in
  all-atom mode),
- the **temperature** of the simulation,
- the **physical time gap** between source and target frames.

The flowing particles are one C-alpha per residue, the N/CA/C/O backbone,
or every non-hydrogen atom, selected by `data.representation` (§13).

## 2. Dual-graph architecture

Two different graphs are encoded separately and then fused per particle.
The **sequence side always runs at residue level**; the **geometric side
runs over whatever particles carry the flow** (C-alpha atoms, or all heavy
atoms).

1. **Bidirectional typed sequence graph** (`protein_flow/models/sequence_encoder.py`).
   Nodes are residues; edges are the fixed peptide connectivity `i -> i+1`
   (forward) and `i+1 -> i` (backward), each treated as a distinct edge
   type. **This graph contains edges in both directions and is therefore
   not a DAG** -- we call it a bidirectional typed sequence graph
   throughout the code and docs. A handful of typed message-passing layers
   propagate PLM + amino-acid-type + positional information along the
   chain in O(L) time.

2. **Dynamic geometric graph** (`protein_flow/geometry/graph.py`,
   `protein_flow/models/geometric_encoder.py`). At every flow-time step,
   a k-nearest-neighbor graph is rebuilt *from the current 3D coordinates*
   `x_tau` (padding-aware, no cross-sample edges, safe when `k` exceeds the
   number of valid particles). An EGNN-style layer stack produces
   rotation/translation-**invariant** hidden features from this graph.
   Edge features are always scalars: distance, RBF-encoded distance,
   normalized **residue** separation, peptide-neighbour and same-residue
   indicators.

3. **Gated fusion** (`protein_flow/models/fusion.py`) combines the two
   representations with a learned, sigmoid gate conditioned on a sinusoidal
   embedding of flow time `tau`, temperature, and `physical_delta_t`, then
   applies **FiLM** modulation from the same condition. In all-atom mode the
   residue-level sequence representation is first broadcast down to each
   atom of its residue, so a residue's language-model context reaches all of
   its atoms.

   The gate alone is not enough, and this is not a refinement. It produces
   one scalar in `(0, 1)` per particle, so it can choose *which*
   representation to read but cannot change the scale -- let alone the sign
   -- of the velocity the decoder emits. The flow-matching target needs
   exactly that: measured on a collapsed checkpoint, the scalar `alpha`
   minimising `||alpha*pred - target||` ran from **-86 at `tau=0.05` to +86
   at `tau=0.95`**. With no way to flip sign with `tau`, gradients from the
   two halves of the `tau` range cancel and the predicted velocity collapses
   toward zero. FiLM supplies the missing degree of freedom; its projection
   is zero-initialised, so a fresh model starts at exactly the old gated
   behaviour. Disable with `model.fusion.film_conditioning: false` for an
   ablation.

4. **Equivariant vector-field decoder** (`protein_flow/models/vector_field.py`)
   turns the fused invariant features into a `[B, N, 3]` velocity by
   weighting the geometric graph's *relative position vectors*
   `x_j - x_i` with learned invariant scalar coefficients -- never by
   emitting raw xyz from an MLP. See Section 6 for why this guarantees
   equivariance.

## 3. Role of Kabsch alignment

Two structures of the same protein can differ by (a) an arbitrary rigid
rotation/translation of the whole frame, which carries no information
about conformational change, and (b) genuine internal deformation, which is
exactly what we want the model to learn to predict.

`protein_flow/geometry/kabsch.py` implements masked, batched Kabsch
alignment to remove (a): it finds the rotation (`det = +1`, no
reflections) and translation that best superpose `target_coords` onto
`source_coords`, ignoring padding residues, and reports pre/post-alignment
RMSD. **It removes rigid-body pose only.** The aligned target still
differs from the source by whatever internal backbone motion actually
occurred between frames -- Kabsch alignment does not and cannot remove
that, and `tests/test_kabsch.py` explicitly checks that a known internal
deformation survives alignment while a known rigid transform is undone.
Alignment runs under `torch.no_grad()` as a fixed preprocessing step, not
as part of the differentiable training graph.

## 4. Flow time `tau` vs. physical time `physical_delta_t`

These are two different, deliberately separate quantities:

- **`tau in [0, 1]`** is a purely mathematical interpolation variable used
  by conditional flow matching to define a path between the source
  structure (`tau=0`) and the Kabsch-aligned target structure (`tau=1`).
  It has no physical units and is *sampled uniformly per training step* --
  it does not correspond to any particular simulated time.
- **`physical_delta_t`** is the actual MD time elapsed between the source
  and target frames (e.g. in ns), passed to the model purely as a
  conditioning signal (log-scaled, sinusoidally embedded) so the model can
  learn different dynamics for different frame spacings.

Conflating the two would be a modeling error: `tau` never appears as a
physical rate, and `physical_delta_t` is never integrated over during
sampling (the ODE solver always integrates `tau` from 0 to 1).

### Scaling conditioning scalars before embedding them (learned the hard way)

`tau`, `temperature` and `physical_delta_t` all reach the model through
`ConditionEncoder` (`protein_flow/models/fusion.py`), which embeds each one
with `sinusoidal_embedding`. That function lays its frequencies out
geometrically from `1.0` down to `1 / max_period`, so it only resolves
inputs that **span hundreds of units** -- the diffusion convention of
feeding a timestep in `[0, 1000]`.

Passing raw values breaks it in both directions, silently:

- `tau in [0, 1]` leaves nearly every frequency band at `sin(x) ~ 0`,
  `cos(x) ~ 1`, so the embedding is almost the same constant vector for
  every `tau`.
- raw Kelvin (320-450) pushes the high-frequency bands to hundreds of
  radians, where they alias into noise rather than a smooth ordering.

Summing the three embeddings compounded it: each is dominated by that same
near-constant "all cosines ~ 1" direction, so addition mostly accumulated
the shared constant.

A 34,500-step run was lost to this. The trained encoder had collapsed to

```
cosine(condition@tau=0, condition@tau=1) = 0.999977
cosine(condition@320K,  condition@450K ) = 0.998356
```

so the model could see neither its own flow time nor the simulation
temperature, and the predicted velocity fell to **0.24%** of the target's
magnitude while `val_loss` moved 0.01% over the whole run. Nothing in the
loss curve distinguished that from ordinary slow progress.

Each scalar is therefore normalised onto `[0, model.fusion.embedding_scale]`
(default 1000) before embedding, and the three embeddings are
**concatenated, not summed**. `tests/test_conditioning.py` asserts the
resulting vectors stay distinguishable, which is the regression that
matters here.

## 5. Conditional flow-matching objective

With `x0 = source_coords` and `x1 = ` Kabsch-aligned `target_coords`, the
default path (`protein_flow/flow/paths.py::LinearPath`) is

```
x_tau = (1 - tau) * x0 + tau * x1
target_velocity = x1 - x0
```

and the model is trained to regress

```
predicted_velocity = v_theta(x_tau, tau, sequence_embedding, residue_types,
                              temperature, physical_delta_t, residue_mask)
```

against `target_velocity` with a masked MSE
(`protein_flow/losses/flow_matching.py`). The path abstraction also
includes a `GaussianBridgePath` stub for future noisy-path experiments
(off by default; see the docstring for its simplifying assumptions).

**Limitation:** straight Cartesian interpolation between two folded
structures can pass through non-physical intermediate conformations (bond
stretching, clashes, broken secondary structure) -- it is only guaranteed
to match the two endpoints, not to be physically valid along the way. The
physics-informed losses (Section 7) are a mitigation, not a fix; a more
principled fix would use an internal-coordinate or torsion-space path
(see Section 9).

## 6. How SE(3)-equivariance is guaranteed (and why not plain E(3))

The geometric encoder and vector-field decoder are built so that rotation
equivariance and translation invariance hold **by construction**, not by
hoping the network learns them:

- All *hidden* scalar features are computed only from rotation/translation
  invariant quantities: pairwise distances, RBF-encoded distances,
  normalized sequence separation, and peptide-neighbor indicators. Raw
  coordinate components never enter an MLP.
- The only vector-valued quantities are the geometric graph's relative
  position vectors `x_j - x_i`, which are equivariant under rotation and
  invariant under translation by construction.
- The decoder's output is `sum_j a_ij * (x_j - x_i) / (||x_j - x_i|| + eps)`
  where `a_ij` is an invariant scalar coefficient -- a rotation of all
  input coordinates rotates every `x_j - x_i` by the same matrix, so the
  weighted sum rotates identically; a translation cancels exactly inside
  every difference `x_j - x_i`.
- Optionally (`decoder.remove_com_velocity`, default `true`) the masked
  center-of-mass of the predicted velocity is subtracted, which removes
  spurious net translation of the whole structure and is itself
  translation-invariant.

**A subtlety we deliberately correct for:** every feature listed above
(distances, RBF, sequence separation, peptide indicator) is a *true*
scalar -- identical for a structure and its mirror image. A network built
only from such features is not just SE(3)-equivariant, it is
**E(3)-equivariant**: it cannot distinguish a structure from its
reflection at all. Real proteins are chiral (built entirely from
L-amino acids), so treating a protein and its mirror image as
interchangeable is physically wrong. `protein_flow/geometry/chirality.py`
adds one pseudo-scalar feature -- the signed dihedral angle of each
consecutive C-alpha quadruple `(i, i+1, i+2, i+3)` -- which is invariant
under proper rotation and translation but **flips sign under reflection**.
`GeometricEncoder` folds this into the initial node features by default
(`model.geometric_encoder.use_chirality_features: true`), which breaks the
reflection symmetry while leaving rotation/translation equivariance
untouched: the model is SE(3)-equivariant, not E(3)-equivariant. Setting
that flag to `false` restores the old, fully-achiral E(3) behavior as an
ablation option.

This is verified, not just asserted:
- `tests/test_equivariance.py` runs the full model on tie-free (no k-NN
  distance ties), fixed-seed coordinates and checks
  `v(x @ R + b) ~= v(x) @ R` for a proper rotation `R` (atol/rtol = 1e-4)
  plus a translation-only invariance check, including with padding
  residues present.
- `tests/test_chirality.py` checks the dihedral feature itself is
  invariant under proper rotation + translation and flips sign under
  reflection.
- `tests/test_se3_vs_e3.py` is the direct SE(3)-vs-E(3) check: with the
  chirality feature on, proper-rotation equivariance still holds but the
  naive reflection identity `v(x @ O) == v(x) @ O` for an improper `O`
  (det = -1) now measurably **fails**; with the feature turned off, that
  same reflection identity holds again, confirming the toggle does what it
  claims in both directions.
- `tests/test_geometric_encoder.py` checks proper-rotation equivariance at
  the sub-module level.

## 7. Physics-informed losses

`protein_flow/losses/physics.py` implements each term in two variants,
dispatched on `data.representation`:

| Term | C-alpha | All-atom |
|---|---|---|
| **Bond distance** | MSE on consecutive C-alpha distances | MSE on real covalent bond lengths (`bond_index` from the PSF) |
| **Bond angle** | cosine MSE on consecutive triplets | cosine MSE on real covalent angles (`angle_index`) |
| **Steric clash** | `relu(threshold - d)^2` for pairs beyond `clash_seq_sep` in sequence | same, but excluding true 1-2 and 1-3 covalent neighbours |

Both variants use cosine-based angles (numerically stable near 0/180
degrees) and compute clashes over the sparse k-NN graph, never an `N x N`
pair tensor. The all-atom exclusion set is built from the real topology
because "close in sequence" no longer implies "covalently bonded" once side
chains branch.

- **Optional endpoint RMSD loss** (`loss.endpoint_enabled`, default
  `false`): masked RMSD between an ODE-rollout endpoint and the aligned
  target. **It unrolls the ODE inside every training step with gradients
  enabled, which is far more expensive than the other, single-shot
  losses** -- see §15 for the step count, gradient-checkpointing, and
  truncated-BPTT knobs that make it affordable.

`loss.physics.apply_to` controls whether bond/angle/clash losses are
applied directly to `x_tau` or to a one-step Euler prediction (the
default). `loss.physics.step_scale_mode` then controls how far that step
goes:

| Mode | Predicted coordinates | Where the ground-truth velocity lands |
|---|---|---|
| `remaining` (default) | `x_tau + step_scale * (1 - tau) * v` | exactly `x1` |
| `constant` | `x_tau + step_scale * v` | `x1 + tau * (x1 - x0)` |

Only `remaining` is consistent with the objective. With the linear path,
`v = x1 - x0` is the correct answer at *every* `tau`, and `(1 - tau) * v`
lands on `x1` -- a real frame, whose geometry the physics terms have no
reason to object to. `constant` overshoots past `x1` by an amount that
grows with `tau`, so it charges the **correct** velocity an ever-larger
penalty: measured on real backbone frames, the bond term against the
ground-truth velocity rose from `0.0049` at `tau=0.05` to `1.1302` at
`tau=0.95`, versus a flat `0.0023` under `remaining`. That gradient points
directly against the model producing a full-size velocity. `constant` is
retained only to reproduce runs made before this was found.

`loss.physics.d_ref_source` controls whether reference bond
distances/angles come from `source_coords` or the aligned target.
`loss.endpoint_physics_enabled` additionally applies them to the rollout
endpoint.

Total loss:
```
L_total = lambda_fm * L_fm + lambda_bond * L_bond + lambda_angle * L_angle
        + lambda_clash * L_clash + lambda_endpoint * L_endpoint
        + lambda_endpoint_physics * (endpoint bond/angle/clash)
```

## 8. ODE sampling

`protein_flow/flow/solver.py` integrates `dx/dtau = v_theta(x, tau, ...)`
from `tau=0` to `tau=1` with either `euler` or `heun` (improved Euler). The
geometric k-NN graph is rebuilt from scratch at every step (and at both
sub-evaluations of a Heun step), since it depends on the current, moving
coordinates. Sampling always runs under `torch.no_grad()`.

```python
generated_coords, trajectory = model.sample(
    source_coords=x0,
    sequence_embedding=embedding,
    residue_types=residue_types,
    residue_mask=mask,
    temperature=temperature,
    physical_delta_t=physical_delta_t,
    num_steps=50,
    solver="heun",
    return_trajectory=True,
)
# generated_coords: [B, L, 3]
# trajectory:       [num_steps + 1, B, L, 3]
```

## 9. Beyond C-alpha: what is implemented and what is next

**Implemented (§13):** the all-atom heavy-atom representation. Rather than
the fixed `[B, L, 4, 3]` backbone tensor originally sketched here, the
particle axis is a flat `[B, N, 3]` with an `atom_residue_index` mapping,
because heavy atoms per residue are genuinely variable (4-15) and a padded
per-residue tensor would waste roughly half its slots. Selecting only the
N/CA/C/O backbone atoms is a strict subset of this machinery and needs no
new code path -- just a different atom filter in
`protein_flow/data/topology.py`.

**Still open.** The `Dataset`/`collate` contract
(`protein_flow/data/dataset.py`) and the `FlowPath` abstraction
(`protein_flow/flow/paths.py`) remain deliberately decoupled from "the
coordinates are Cartesian", so these extensions do not require touching
graph construction or flow matching:

- **Local-frame representation** (per-residue rotation + translation,
  a.k.a. rigid-body frames as in AlphaFold/FrameDiff-style models): the
  flow path would interpolate translations linearly and rotations via a
  geodesic on `SO(3)` (e.g. via matrix logarithm/exponential or
  quaternion slerp) rather than linear Cartesian interpolation, directly
  addressing the Section 5 limitation.
- **Torsion/internal-coordinate flow**: replace Cartesian `x_tau`
  interpolation with interpolation of backbone dihedral angles
  (phi/psi/omega), reconstructing Cartesian coordinates via NeRF only when
  the geometric graph or physics losses need them. `FlowPath` is already
  an abstract interface for exactly this kind of substitution.

## 10. Tensor schema for connecting real data (mdCATH / ATLAS)

This repository does not assume any on-disk format up front -- it defines
the tensor schema below, a synthetic dataset that satisfies it, and (now)
a real adapter for mdCATH, `protein_flow.data.mdcath.MdCathDataset`,
verified against 100 real shards (`mdCATH_sample100/`, ~2,500
domain/temperature/replica trajectories, lengths 53-479 residues).

What was actually verified by opening these files (not assumed):
- Each shard is one CATH domain, with per-atom arrays `chain`, `element`,
  `resid`, `resname`, `z` (length `numProteinAtoms`) describing the full
  topology directly -- no separate PDB/PSF file is needed.
- Each shard also embeds a full PDB-format text blob
  (`pdbProteinAtoms`) whose ATOM records are in *exactly* the same
  order as those per-atom arrays (checked: parsed resname/resid from this
  text matched the arrays exactly on every file inspected). Fixed-column
  PDB parsing (`parse_ca_indices_and_resnames`) extracts the true
  C-alpha atom index per residue this way -- using the atom *name* field
  ("CA"), not just element ("C"), so e.g. CHARMM's "CAY" N-terminal cap
  atom is never confused with the alpha carbon. Sanity check: resulting
  consecutive CA-CA distances average 3.83 A across sampled domains,
  matching the known C-alpha virtual bond length.
- Below the domain group, subgroups are keyed by simulation temperature
  in Kelvin (`"320"`, `"348"`, `"379"`, `"413"`, `"450"` -- these are
  literal group names in the file, not an assumption) each containing
  replica subgroups (`"0"`..`"4"`) with a `coords` dataset of shape
  `[num_frames, num_atoms, 3]`.

What was **not** verified: the physical time gap between consecutive saved
frames. There is no per-frame timestamp in these files, and external
sources we checked disagreed with each other and with the frame counts we
actually observed here. `MdCathDataset` therefore does not silently
hard-code a nanosecond/picosecond value -- by default `physical_delta_t`
is reported in raw frame-count units; pass `ps_per_frame=` (or
`data.mdcath_ps_per_frame` in the YAML config) only once you've confirmed
the correct value for your own mdCATH download.

Run it with:
```bash
python train.py --config configs/mdcath.yaml
```
`configs/mdcath.yaml` points `data.mdcath_dir` at `mdCATH_sample100/data`
(splits train/val **by domain** so no domain leaks across the split) and
sets `data.source: mdcath`. `sequence_embedding` for real mdCATH samples is
a deterministic per-domain placeholder unless `data.mdcath_embedding_cache_dir`
points at cached `{domain}.pt` real PLM embedding tensors (`[L, plm_dim]`)
you've precomputed yourself -- ESM is still never run automatically.

Frame-pair sampling deliberately differs by split. The training dataset
selects a fresh, deterministic random offset for every trajectory whenever
`train()` advances the dataset epoch, so frames stored in a trajectory are
not reduced to one permanently memorized pair. Validation uses
`mdcath_val_pairs_per_trajectory` fixed, evenly-spaced offsets instead. This
makes repeated checkpoint comparisons stable while covering more than a
single arbitrary location in each validation trajectory. Cached PLM tensors
are also retained in memory after their first load. CPU evaluation is exactly
repeatable; CUDA scatter kernels may retain only floating-point-scale
non-determinism (observed around `1e-7`, rather than the former `1e-1` random
flow-time variation).

Validation does not randomly sample flow time. It averages the fixed
`train.val_tau_values` grid and logs the individual FM/physics components,
the zero-velocity FM baseline, relative FM improvement, and mdCATH FM split
by temperature. The scalar `evaluate(...)` API remains available and returns
the deterministic total; `evaluate_detailed(...)` returns the complete metric
dictionary. The default mdCATH configuration uses two frame pairs and three
flow times as a runtime-conscious training-time validation. A denser grid can
be used for final model comparison.

Validation also performs an ODE rollout on a fixed, seeded subset when
`train.val_endpoint_enabled` is true. This metric is not backpropagated and
does not enable the expensive endpoint training loss. It reports source-to-
target RMSD, generated-to-target RMSD, relative improvement, the fraction of
samples improved over the unchanged source, and endpoint bond/angle/clash
statistics. `val_endpoint_num_steps` and `val_endpoint_max_batches` bound its
cost; the default mdCATH run uses 10 Heun steps on 10 fixed batches.

The optimizer is paired with `ReduceLROnPlateau`: after
`train.optim.plateau_patience` unimproved validation checks it multiplies the
learning rate by `plateau_factor`, down to `min_lr`. Training itself stops
only at `num_epochs` or `max_steps`; early stopping is intentionally deferred
until later experiments establish an appropriate stopping criterion. Both
`last.pt` and `best.pt` store scheduler state, so checkpoint metadata reflects
the learning-rate decisions that led to that model.
When endpoint validation is enabled, `best_endpoint.pt` independently preserves
the lowest generated endpoint RMSD and its full validation metrics. This avoids
losing a rollout-optimal model when the flow-matching validation minimum occurs
at a different step; it does not stop training.

### Frame-gap ablation

Small configs can inherit another YAML through a relative `base_config` path;
nested mappings are deep-merged and the fully resolved config is still saved
next to each checkpoint. The provided horizon comparison is:

```bash
python train.py --config configs/ablations/mdcath_gap1.yaml
python train.py --config configs/ablations/mdcath_gap2.yaml
python train.py --config configs/ablations/mdcath_gap5.yaml
```

Or run the same comparison on two GPUs and summarize both the best validation
checkpoint and the best endpoint checkpoint:

```bash
./scripts/run_mdcath_gap_ablation.sh
./scripts/summarize_gap_ablation.py --log-dir ablation_logs \
  --json ablation_logs/summary.json --markdown ablation_logs/summary.md
```

All three inherit the same model, split, seed, and validation settings, while
writing separate checkpoint directories. They also inherit
`mdcath_sampling_max_frame_gap: 5`, so source frames are sampled from the same
valid pool and only the target horizon changes. This makes zero-baseline,
flow-matching improvement, and endpoint-RMSD improvement directly comparable.

ATLAS has its own, separately-documented layout and was not inspected at
all for this project; connecting it would mean writing another adapter of
the same shape as `MdCathDataset`.

Any adapter -- real or synthetic -- just needs to subclass
`protein_flow.data.dataset.ProteinTrajectoryDataset` and, per
`__getitem__`, return a dict matching this schema:

```python
{
    "sequence_embedding": FloatTensor[L, D_plm],   # precomputed PLM embedding
    "source_coords":      FloatTensor[L, 3],       # CA coords at time t
    "target_coords":      FloatTensor[L, 3],       # CA coords at time t + physical_delta_t
    "residue_types":      LongTensor[L],
    "temperature":        FloatTensor[1],
    "physical_delta_t":   FloatTensor[1],
}
```

`protein_flow/data/collate.py` pads a list of these into a batch and adds
`residue_mask: BoolTensor[B, L]`. Everything downstream (Kabsch alignment,
both graphs, flow matching, physics losses, sampling) only depends on this
schema, not on any file format.

### ESM2 embedding adapter (implemented, opt-in)

`protein_flow/data/esm_adapter.py::compute_esm_embeddings` runs real
`facebook/esm2_t6_8M_UR50D` (hidden size 320, matching `DataConfig.plm_dim`'s
default) via `transformers` and returns one `[L, 320]` tensor per input
sequence, special tokens and batch padding already stripped. **It is never
imported by `train.py`, `sample.py`, or the test suite** -- ESM only runs
when you explicitly invoke:

```bash
python precompute_esm_embeddings.py \
    --mdcath-dir mdCATH_sample100/data \
    --output-dir esm_cache
```

This parses each domain's embedded PDB text once (same code path as
`MdCathDataset`) to get its one-letter sequence, runs ESM2 once per domain
(not per frame -- the sequence is fixed across temperature/replica/frame),
and writes `esm_cache/{domain}.pt`. Verified end-to-end on all 100
`mdCATH_sample100` domains: 100 embeddings computed in ~7s on CPU (model
weights cached locally after the first download), each with shape
`[num_residues, 320]`. Point `data.mdcath_embedding_cache_dir` at that
directory (already the default in `configs/mdcath.yaml`) and
`MdCathDataset` loads the real embedding for every domain that has one,
falling back to the placeholder only for domains missing from the cache.

## 11. Running it

```bash
pip install -r requirements.txt

# Train on the synthetic dataset (CPU, a few seconds):
python train.py --config configs/default.yaml

# Train on real mdCATH shards instead (see Section 10):
python train.py --config configs/mdcath.yaml

# Everything on: ~80M flow net + all-atom + ESM2 fine-tuning + unrolling:
python train.py --config configs/mdcath_full.yaml

# Sample from a trained checkpoint:
python sample.py --checkpoint checkpoints/best.pt --num-steps 20 --solver heun
```

### Which config to use

Each config adds one capability on top of the previous one, so they double
as an ablation ladder.

| Config | Data | Particles | Flow net | PLM | Endpoint rollout | Batch | AMP |
|---|---|---|---|---|---|---|---|
| `default` | synthetic | C-alpha | 0.85M | random placeholder | off | 8 | off |
| `mdcath` | real | C-alpha | 0.85M | precomputed cache | off | 4 | off |
| `mdcath_large` | real | C-alpha | **79.4M** | precomputed cache | off | 32 | on |
| `mdcath_all_atom` | real | **heavy atoms** | 79.4M | precomputed cache | off | 4 | on |
| `mdcath_esm` | real | heavy atoms | 79.4M | **ESM2-150M, fine-tuned** | off | 2 | on |
| `mdcath_full` | real | heavy atoms | 79.4M | ESM2-150M, fine-tuned | **on (8 steps)** | 2 | on |
| `mdcath_backbone_rotate` | **all of mdCATH** | **backbone (N/CA/C/O)** | 79.4M | ESM2-150M, fine-tuned | on (8 steps) | 4 | on |

`default` is the only one that needs no external data and runs on CPU in
seconds; the middle rows expect mdCATH shards already sitting at
`data.mdcath_dir`. `mdcath_backbone_rotate` is the only one that does not:
it streams the whole 3.61 TB dataset through local disk a chunk at a time
(§17).

### Long runs in the background (tmux)

Real-data runs take hours, so `scripts/run_training_tmux.sh` starts one in a
**detached tmux session**: it survives SSH disconnects and leaves your shell
free.

```bash
# Launch (session name defaults to the config's basename):
scripts/run_training_tmux.sh --config configs/mdcath_full.yaml --gpu 0

# Two GPUs (DistributedDataParallel):
scripts/run_training_tmux.sh --config configs/mdcath_full.yaml --gpu 0,1 --nproc 2

# Watch it:
tmux attach -t mdcath_full      # detach again with Ctrl-b then d
tail -f logs/mdcath_full.log    # or just follow the log

tmux ls                                # list running sessions
tmux kill-session -t mdcath_full       # stop the run
```

Options: `--config` (required), `--session`, `--gpu`, `--log`, `--force`.
Python output is unbuffered so `tail -f` shows progress live, and the pane
stays open after the run ends so its exit status remains visible.

The script runs under the **af3 conda environment** by default (override
with `RUN_PYTHON=...`). A bare `python3` would not be a safe default: in a
fresh login shell it resolves to the base conda env, which has torch but no
`h5py`, so an mdCATH run would fail minutes in. The script also preflights
the interpreter against the config -- checking `h5py` for
`data.source: mdcath` and `transformers` for `model.esm.enabled` -- and
refuses to launch with a clear message rather than dying partway through.

The script **refuses to start** if the tmux session name, the log file, or
the config's `train.ckpt_dir` already exists, so re-running a command from
your shell history can never silently clobber a run in progress; `--force`
overrides. It also rejects `--nproc` larger than the device count in
`--gpu`, which would otherwise silently train on the wrong number of GPUs
or deadlock.

### Multi-GPU (DistributedDataParallel)

`--nproc N` launches the run under `torchrun` with one process per GPU. The
same `train.py` serves both cases: `setup_distributed` initialises a process
group only when torchrun's environment variables are present.

```bash
# Equivalent to the tmux launch above, run directly:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --config configs/mdcath_full.yaml
```

What DDP changes:

- **Data is sharded.** The training loader gets a `DistributedSampler`;
  each rank sees a disjoint slice, so an epoch takes `1/N` as many steps.
  Note the **effective batch size is `data.batch_size * N`** -- with
  `batch_size: 2` on two GPUs each optimizer step consumes 4 samples.
- **`drop_last=True` on the training loader**, which is a correctness
  requirement rather than tidiness: every rank must run the same number of
  backward passes, or the gradient all-reduce deadlocks on whichever rank
  runs out of batches first.
- **Rank 0 alone logs and writes checkpoints**, and it saves the
  *unwrapped* model so checkpoints load fine in a later single-process run.
- **Validation is sharded too**, and every rank participates. Each rank
  evaluates its own slice with the *unwrapped* module (no gradients are
  needed there), and the raw accumulators are merged afterwards. Because
  every metric is a sum over a count, summing the sums and the counts
  reproduces the single-process result; averaging per-rank means would not.

  > An earlier version ran validation on rank 0 only. That was a **liveness
  > bug, not just an inefficiency**: the other ranks blocked at the next
  > collective for the entire pass, and once a pass exceeded NCCL's
  > 10-minute watchdog timeout they were aborted and the whole run died with
  > `Watchdog caught collective operation timeout`. Sharding removes the
  > asymmetry; the process-group timeout is also raised to 2 hours as
  > defence in depth, since long collective-free stretches are legitimate
  > here.

  One caveat worth knowing: the scalar loss components are **batch means**,
  so regrouping samples into different batches shifts them slightly. Going
  from 1 to 2 GPUs changes that grouping, and the resulting shift is exactly
  the same size as changing `batch_size` on a single GPU (measured:
  `fm` 29.0209 at 1 GPU x batch 4, 28.9459 at 2 GPUs x batch 4, and
  28.9459 at 1 GPU x batch 8 -- the DDP number reproduces the single-GPU
  regrouped number to 7 significant figures). Per-sample metrics such as
  `fm_by_temperature` are invariant, matching to ~7 decimal places.

- **`train.val_max_batches` and `train.val_endpoint_max_batches` are global
  budgets**, divided across ranks, so a cap means the same amount of
  validation work regardless of GPU count. Worth setting: the uncapped
  mdCATH split is ~375 batches x 3 tau values plus endpoint rollouts.
- **`static_graph=True`** on the DDP wrapper. This is what lets DDP coexist
  with activation checkpointing: without it, recomputation during the
  backward pass marks a parameter ready twice and DDP raises
  `"Expected to mark a variable ready only once"`. The graph genuinely is
  static here -- the same modules run the same number of times each step.
  ESM's checkpointing is likewise requested with `use_reentrant=False` for
  the same reason.
- **Per-rank seeds** (`train.seed + rank`) so frame-pair sampling and `tau`
  draws are decorrelated; DDP broadcasts the initial weights regardless, so
  the models still start identical.
- **Parameter groups are built from the *unwrapped* module.** DDP prefixes
  every parameter name with `module.`, so `_build_parameter_groups` matching
  on `esm_encoder.` against the wrapped model matched **nothing**: the two
  groups silently collapsed into one and the pretrained PLM was fine-tuned at
  the flow network's `3e-4` instead of its own `1e-5` -- 30x too high, in
  exactly the situation that function exists to prevent.

  > This is worth calling out because of how it hid. Nothing crashed, nothing
  > warned, and single-GPU runs were unaffected; the only visible trace was a
  > checkpoint whose `optimizer_state_dict` had one param group instead of
  > two. It was found by reading a checkpoint before deleting it, after ~8,600
  > steps of a multi-GPU run had already been trained that way.
  > `tests/test_distributed.py::test_esm_parameter_group_survives_ddp_wrapping`
  > now pins the behaviour.

Verified on 2x RTX PRO 6000: both GPUs at 94-100% utilisation, and after one
optimizer step the parameters are **bit-identical across ranks**
(`max |param(rank_i) - param(rank_0)| = 0.0`), which is the actual proof
that gradients are being all-reduced rather than two independent jobs
running side by side. `tests/test_distributed.py` asserts the same property
in a real two-process `gloo` run, so it is checked without needing a GPU.

### Expected tensor shapes

Default (C-alpha) config, `plm_dim=320`, `batch_size=8`, `L` = padded
residue count:

| Tensor | Shape |
|---|---|
| `sequence_embedding` | `[B, L, 320]` |
| `source_coords`, `target_coords`, `aligned_target` | `[B, L, 3]` |
| `residue_mask` | `[B, L]` bool |
| `predicted_velocity` | `[B, L, 3]` |
| `model.sample(...)` trajectory | `[num_steps + 1, B, L, 3]` |

All-atom config, additionally, with `N ~= 7.9 * L` heavy atoms:

| Tensor | Shape |
|---|---|
| `source_coords`, `target_coords`, `predicted_velocity` | `[B, N, 3]` |
| `atom_mask` | `[B, N]` bool |
| `atom_residue_index`, `atom_element` | `[B, N]` long |
| `ca_atom_index` | `[B, L]` long |
| `bond_index` / `angle_index` | `[B, E_b, 2]` / `[B, E_a, 3]` long |
| `esm_input_ids`, `esm_attention_mask` | `[B, L + 2]` long |

Note the two independent axes: the **residue axis `L`** carries the
sequence/PLM side, the **atom axis `N`** carries the geometry and the flow
itself, and `atom_residue_index` maps the second onto the first.

## 12. Tests

```bash
pytest -q
```

137 tests cover: masked batched Kabsch alignment (including that internal
deformation survives alignment while rigid transforms are undone, and that
it survives AMP -- see §16), sparse sequence/geometric graph construction
(padding, no cross-batch edges, safe `k > valid_residues`, radius cutoff),
the flow path and flow-matching loss, the sequence encoder, the geometric
encoder + equivariant decoder, gated fusion + the full model, all physics
losses, the Euler/Heun ODE solver (checked against analytic constant- and
linear-velocity fields), the synthetic dataset + collate, a full
training-loop smoke test (including an overfit-one-batch check that loss
actually decreases), and one end-to-end CPU smoke test.

Equivariance is verified rather than asserted, in three files:
`tests/test_equivariance.py` (full model, C-alpha),
`tests/test_chirality.py` (the pseudo-scalar itself),
`tests/test_se3_vs_e3.py` (reflection symmetry is broken by default and
restored when the feature is disabled), plus an all-atom equivariance test
in `tests/test_all_atom.py`.

The newer subsystems have their own suites: `tests/test_all_atom.py`
(topology, dual-axis batching, all-atom SE(3) equivariance, topology-driven
physics losses), `tests/test_esm_in_graph.py` (token alignment, gradient
flow into the PLM, separate learning-rate groups), and
`tests/test_endpoint_rollout.py` (checkpointed unrolling matches the plain
rollout in both value and gradient, truncated BPTT, solver options), and
`tests/test_distributed.py` (single-process degradation of every helper,
plus a real two-process `gloo` run asserting gradients are all-reduced and
the dataset is sharded, and regressions for the validation-budget bugs
described in section 11).

Tests that would need external data or downloads skip cleanly: the real
mdCATH tests skip when `mdCATH_sample100/` is absent, and the ESM tests
build a tiny randomly-initialised ESM locally rather than downloading the
150M checkpoint.

## 13. All-atom and backbone representations

`data.representation` selects what the flow acts on:

| | `ca` (default) | `backbone` | `heavy_atom` |
|---|---|---|---|
| particles | 1 per residue | N, CA, C, O | every non-hydrogen atom |
| count | `L` | `4 * L` exactly | `~7.9 * L` |
| covalent topology | "adjacent in sequence" | real CHARMM bonds/angles from the shard's PSF | same |
| clash threshold | 3.5 A | 2.2 A | 2.5 A |

`backbone` and `heavy_atom` share one code path -- the atom-level batch
layout, the PSF-derived topology, the same losses -- and differ only in
which atoms are selected (`is_atom_level()` in `protein_flow/config.py` is
the single predicate that distinguishes them from `ca`).

Hydrogens are excluded deliberately: they are force-field-added, their
positions are largely slaved to the heavy atoms, and including them would
double the graph for little structural information (they are 50.3% of
mdCATH's atoms).

Three things make this more than a change of `N`:

1. **Covalent connectivity cannot be inferred from atom ordering** once side
   chains branch. `protein_flow/data/topology.py` parses each shard's
   embedded CHARMM PSF for real bonds and angles. Verified against the
   sample shards: the CA atoms it finds are identical to those from the
   independent PDB-text path, heavy-atom bond lengths span 1.17-1.66 A, and
   the outliers at 2.02-2.12 A are genuine CYS-CYS disulfide bridges.
2. **Sequence separation must stay a residue-level quantity.** Two atoms in
   the same residue are separated by 0 residues, not by their atom-slot
   distance, so `build_geometric_graph` takes a `separation_index`. All-atom
   mode also gains a `same_residue_indicator` edge feature (identically zero
   in C-alpha mode).
3. **The clash threshold is representation-specific.** Measured on real
   mdCATH frames, no non-bonded heavy-atom pair (excluding 1-2 and 1-3
   covalent neighbours) comes closer than **2.507 A**, and none is below
   2.5 A at all. Reusing the C-alpha value of 3.5 A would penalize a large
   fraction of perfectly normal contacts, so
   `loss.physics.clash_threshold_heavy_atom` defaults to 2.5.

The sequence side stays residue-level throughout; its representation is
broadcast to each atom of its residue before fusion. The chirality
pseudo-scalar (§6) is likewise computed on the C-alpha backbone -- the only
place backbone handedness is defined -- and then broadcast to that
residue's atoms.

### Why `backbone` is a clean middle ground

Selecting N/CA/C/O by PSF atom name gives **exactly 4 particles per
residue** on real shards -- verified, not assumed. The CHARMM terminal
patches (`CAY`/`CY`/`OY` at the N-terminus, `NT`/`CAT` at the C-terminus)
are heavy atoms, but they are named differently *and* carry the same
`resid` as the first/last residue, so excluding them by name drops no
residue and leaves nothing ragged. The PSF bond/angle filter then needs no
special-casing at all: every interaction touching a hydrogen or a side-chain
atom already falls out, leaving precisely the peptide graph --
`4L - 1` bonds (N-CA, CA-C, C=O per residue, plus one peptide C-N per
junction), with lengths 1.16-1.64 A and angles 100-136 A on real frames.

The clash threshold was measured the same way as the heavy-atom one, over 45
sample domains x 6 frames x 2 temperatures: the closest non-bonded backbone
pair the k-NN clash term actually sees is **2.278 A**, so
`loss.physics.clash_threshold_backbone` defaults to 2.2. Note this is
*lower* than the heavy-atom 2.5: with 4 particles per residue the k=16
neighbourhood reaches further along the chain and includes 1-4 pairs (e.g.
O(i) to CA(i+1)) that the denser all-atom graph never has as edges.

## 14. In-graph ESM2 fine-tuning

There are two distinct ways to use a PLM here, and they are separate code
paths on purpose:

- **Precomputed (default).** `precompute_esm_embeddings.py` runs ESM2 once
  per domain, offline, into `esm_cache/{domain}.pt`. Training never imports
  ESM. This is `protein_flow/data/esm_adapter.py`.
- **In-graph, fine-tuned** (`model.esm.enabled: true`). ESM2 lives inside
  the model, consumes token ids, and its weights receive gradient from the
  flow-matching objective. This is `protein_flow/models/esm_encoder.py`.

With `configs/mdcath_full.yaml` the parameter budget is:

| Component | Parameters |
|---|---|
| Flow network (§15) | 79.7M |
| ESM2-150M (fine-tuned) | 147.7M |
| **Total** | **227.5M** |

Details that matter:
- **Token alignment.** `esm2_*` tokenizers emit `[<cls>, r_1..r_L, <eos>]`,
  so residue `j` is at token `j + 1`; the encoder slices that offset off
  rather than assuming alignment. `data.plm_dim` must equal the checkpoint's
  hidden size (640 for `esm2_t30_150M`) and this is validated at build time.
- **Separate learning rate.** ESM is pretrained and the flow network is not.
  Training both at one rate tends to destroy the PLM's representations, so
  `_build_parameter_groups` puts ESM in its own group at
  `model.esm.learning_rate` (default 1e-5) against `train.optim.lr` (3e-4).
- **Memory.** `model.esm.gradient_checkpointing` (default true) keeps the
  full all-atom + 227M-parameter step at ~10 GB.
- `model.esm.num_frozen_layers` freezes the embedding table plus the lowest
  N transformer layers; `trainable: false` freezes the PLM entirely (which
  is equivalent to the precomputed path, without needing a cache on disk).
- ESM2's auxiliary contact-prediction head is never called by this encoder,
  so it is explicitly frozen rather than left in the optimizer receiving no
  gradient.

## 15. Model scale and the endpoint rollout

### Scaling the flow network

`configs/mdcath_large.yaml` and everything built on it use
`hidden_dim=1024` with 4 sequence / 6 geometric / 3 decoder layers, giving
**79.4M** parameters (sequence 30.1M, geometric 38.0M, fusion 5.2M, decoder
6.4M). Measured peak VRAM for a training step at `L~250`:

| batch | AMP off | AMP on |
|---|---|---|
| 16 | 6.7 GB | 4.2 GB |
| 32 | 13.0 GB | 7.9 GB |
| 64 | 25.6 GB | 15.2 GB |

### Differentiable ODE unrolling

`loss.endpoint_enabled` turns on the endpoint RMSD term, which unrolls the
flow ODE **inside the training step** and backpropagates through every
model evaluation. This is by far the most expensive part of the objective,
and `loss.endpoint_rollout` exists to keep it tractable:

- `num_steps` (default 8) is deliberately **separate from
  `sampling.num_steps`** (50). Inference wants a fine discretization;
  backpropagating through 50 evaluations of a 227M-parameter model is not
  practical.
- `gradient_checkpointing` (default true) recomputes each step's activations
  during the backward pass. Measured on the all-atom config: rollout memory
  becomes essentially **flat in `num_steps`**, and without it the run does
  not fit at all.

  | rollout steps | checkpointing off | on |
  |---|---|---|
  | 2 | 39.7 GB | 27.0 GB |
  | 8 | **OOM** | 27.0 GB |
  | 16 | — | 27.0 GB |

- `backprop_last_steps` implements truncated BPTT: earlier steps still run
  (so the trajectory is correct) but are detached from the backward graph.
- `loss.endpoint_physics_enabled` additionally applies the bond/angle/clash
  terms to the rollout endpoint, which is where straight-line Cartesian
  interpolation (§5) actually produces non-physical geometry.

**`configs/mdcath_backbone_rotate.yaml` turns this term off.** Over a full
34,500-step run it never helped once -- `endpoint_improvement` sat at
-0.06% and `win_rate` at 0.00% -- while its gradient contribution measured
~5% of the flow-matching term's and it consumed 82% of the step (batch 8:
2.21 s/step with the rollout, 0.39 s/step without). `train.val_endpoint_*`
stays on, so the metric is still reported at every validation; if it starts
improving, that is the signal to bring the term back into the objective.

### Checkpointing the geometric encoder

`model.geometric_encoder.gradient_checkpointing` (default `false`) does the
same trick for the EGNN stack, which holds the largest activations in the
model at atom resolution: with `k=16` neighbours over 4 particles per
residue, a 512-residue sample carries 32,768 edges *per layer*. Measured at
512 residues, batch 16, endpoint rollout off, dropping from 6 layers to 3
freed 16.6 GiB -- about 5.5 GiB per layer.

Turning it on trades ~23% step time for 58% of peak memory (54.8 -> 22.9
GiB at batch 16), which is what lets the rotation config run batch 32 at
the `mdcath_max_residues` cap:

| batch | peak | s/step | s/sample |
|---|---|---|---|
| 16 | 22.9 GiB | 0.997 | 0.0623 |
| 32 | 42.8 GiB | 1.879 | 0.0587 |
| 48 | 62.6 GiB | 2.818 | 0.0587 |
| 64 | 82.5 GiB | 3.790 | 0.0592 |

Per-sample throughput is already flat at 32, so larger batches buy only
fewer optimizer steps. `tests/test_geometric_encoder.py` asserts the
checkpointed forward and backward match the plain ones exactly -- it is a
memory trade, not a numerical one.

## 16. Optional dependencies

`torch_scatter` is never required: `protein_flow/utils.py` provides a
pure-PyTorch fallback (`index_add_`-based) for every scatter-aggregation
used in the sequence encoder, geometric encoder, and vector-field decoder,
and uses `torch_scatter` automatically if it happens to be installed.

`h5py` is only required for `data.source: mdcath`, `transformers` only for
ESM (either precomputation or in-graph fine-tuning), and `huggingface_hub`
only for shard rotation (§17). The default synthetic pipeline needs none of
them, and the test suite skips or stubs all three.

### A note on AMP

`torch.linalg.svd` has no half-precision CUDA kernel, so enabling
`train.amp` used to crash inside Kabsch alignment with
`"svd_cuda_gesvdjBatched" not implemented for 'Half'`. Alignment now forces
full precision internally and casts back to the caller's dtype -- it is
cheap `no_grad` preprocessing, and an SVD driving a rigid-body fit is
exactly the kind of ill-conditioned operation that should not run in fp16.
`tests/test_kabsch.py` covers this. AMP cuts training memory by roughly 40%
and is enabled in all the scaled configs.

## 17. Training on all of mdCATH by shard rotation

Everything above trains on the 100 shards staged at `data.mdcath_dir`. The
full dataset is **5,398 domains / 3.61 TB** (measured with
`HfApi.list_repo_tree`: mean shard 670 MB, median 564 MB, max 2.51 GB), so
it cannot sit on disk next to the checkpoints. Setting
`data.rotation.enabled` switches `train.py` to a loop that **downloads a
chunk of domains, trains on it, deletes it, and moves to the next**, with
the following chunk downloading in the background throughout
(`protein_flow/train_rotating.py`).

```bash
# 1. Freeze the rotation plan (once). --prefer-local orders shards you
#    already have into the earliest chunks, so training starts immediately.
python scripts/build_mdcath_manifest.py \
    --output configs/manifests/mdcath_all.json \
    --prefer-local /home/mipstu/wjYang/MolecularDynamics/mdCATH_sample100/data

# 2. Make those shards visible to the rotation without copying 64 GB, and
#    without the rotation ever being able to delete the originals: hard
#    links share the inode, so releasing a chunk only drops the extra name.
mkdir -p ../mdCATH_rotating/data
ln -f ../mdCATH_sample100/data/*.h5 ../mdCATH_rotating/data/

# 3. Train (single GPU, or two with torchrun / the tmux launcher).
python train.py --config configs/mdcath_backbone_rotate.yaml
scripts/run_training_tmux.sh -c configs/mdcath_backbone_rotate.yaml -g 0,1 -n 2

# Interrupted? Just run it again: train.resume: auto picks up last.pt and
# continues at the chunk it had reached.
```

With the shipped defaults this yields 25 validation domains (15 GB) plus 27
chunks of 115-146 GB, of which chunk 0 already has 99 of its 200 shards on
disk. Peak residency is ~306 GB.

### Watching a run

`run_training_tmux.sh` tees everything to `logs/<session>.log`, so the run is
readable from any shell without attaching to tmux. Since a rotating run logs
one line per 100 steps for days, `scripts/watch_rotation.sh` pulls out the
lines that actually say how it is going -- current chunk, fixed-holdout
validation, download stalls, and any errors:

```bash
scripts/watch_rotation.sh              # summary of the default log
scripts/watch_rotation.sh -n 20        # more history per section
scripts/watch_rotation.sh -f           # stream it live (same as tail -f)
scripts/watch_rotation.sh -l logs/other.log
```

The `cumulative download stall` figure it reports is the one to act on: if it
grows steadily, the GPU is waiting on the network and `steps_per_chunk`
should go up.

### The manifest is the plan, and it is frozen

`scripts/build_mdcath_manifest.py` writes a JSON file recording, for every
domain, its repo path and **byte size**, split into a validation holdout and
an ordered list of chunks. Freezing it buys three things: the plan is
reproducible from a seed rather than from whatever the Hub listing returns
that day; a run interrupted after 20 chunks resumes without re-querying
anything; and chunk sizes can be capped in **real bytes**, which matters
because shard sizes vary 3.7x -- a count-only split occasionally draws
several 2 GB shards and overshoots the disk budget.

### The validation set does not rotate

`num_val_domains` (default 25, ~17 GB) are held out globally, downloaded
once, and never deleted or trained on. This is a correctness requirement,
not thrift: `ReduceLROnPlateau` and best-checkpoint selection compare
validation loss *across* chunks, which is meaningless if the measuring stick
changes with the data. Expect **training loss to jump at chunk boundaries** --
that is the data changing, not the model regressing. Judge the run by the
fixed validation set.

### Disk and network budget

| `chunk_size` | chunk | chunks | resident peak (val + current + prefetch) |
|---|---|---|---|
| 100 | ~67 GB | 54 | ~150 GB |
| **200 (default)** | **~134 GB** | **27** | **~285 GB** |
| 500 | ~335 GB | 11 | ~690 GB -- needs `prefetch_chunks: 0` |

`min_free_gb` (default 150) is a hard floor: prefetching stops with a
warning rather than filling the volume the checkpoints live on.

One full cycle downloads the whole 3.61 TB, so **`steps_per_chunk` has to
outlast the prefetch or the GPU stalls at every boundary**. Measured on this
machine: ~1.4 s per optimizer step at `batch_size: 4`, so the default 2000
steps is ~46 min of training, against ~32 min to fetch a 134 GB chunk at the
~70 MB/s two ranks reach downloading in parallel (~35 MB/s per stream, so
~64 min on a single GPU -- raise `steps_per_chunk` to ~3000 there). Don't
guess: the pool logs `Waited Ns for chunk k` plus the achieved MB/s, which is
exactly the number to tune against. `HF_HUB_ENABLE_HF_TRANSFER=1` speeds
downloads up if that package is installed.

### What makes it safe under DDP

- **Ranks download disjoint slices** (`entries[rank::world_size]`), so a
  134 GB chunk arrives N times faster and no rank sits at a collective for
  the whole transfer -- long enough to trip the NCCL watchdog. This assumes
  ranks share a filesystem (true for single-node multi-GPU); a multi-node
  run without one must have every rank fetch everything.
- **The resident file list is read back off disk**, never taken to be "what
  I downloaded". `DistributedSampler` assumes every rank holds the same
  dataset; if one rank saw 200 shards and another 199 they would disagree on
  the batch count and deadlock. A shard that fails verification is deleted,
  so it is missing for everyone alike.
- **Every downloaded shard is opened with h5py before use.** A truncated
  transfer otherwise surfaces much later as a baffling indexing error.
- **Chunks are released after a barrier**, once no rank is still reading.
- **Validation is triggered purely by `global_step`**, never by whether a
  rank happened to skip a non-finite update -- a skip is per-rank data, and
  validation is collective.

### Resuming

Every checkpoint carries the rotation cursor in its `extra` field.
`train.resume: auto` (or a path) restores model, optimizer, scheduler,
`global_step` and best-so-far metrics, and continues at the recorded chunk.
Resumption restarts the *current* chunk from its first step, so at most
`steps_per_chunk` steps are ever repeated.

### Where the time actually goes

Measured on one RTX PRO 6000 (95.0 GiB) against the longest resident domain
(446 residues), backbone representation with in-graph ESM2-150M:

| | peak | s/step | s/sample |
|---|---|---|---|
| batch 4 | 28.9 GiB | 2.43 | 0.615 |
| **batch 8** | **53.8 GiB** | **4.01** | **0.505** |
| batch 12 | 78.7 GiB | 5.80 | 0.483 |

and, at batch 8, by objective term:

| | peak | s/step |
|---|---|---|
| endpoint rollout, 8 steps | 53.8 GiB | 4.01 |
| endpoint rollout, 4 steps | 53.8 GiB | 2.21 |
| endpoint rollout, 2 steps | 53.7 GiB | 1.33 |
| endpoint loss off (FM + physics only) | 29.0 GiB | 0.39 |

Two things follow, and both are counter-intuitive enough to be worth writing
down:

- **The endpoint rollout is ~90% of training time** (3.6 s of 4.0 s), for a
  term whose measured gradient contribution is ~5% of the flow-matching
  term's. `loss.endpoint_rollout.num_steps` is therefore the single largest
  throughput knob in the config -- far bigger than batch size or workers.
- **`model.esm.gradient_checkpointing` is a no-op here.** Toggling it changed
  neither memory nor time (28.9 GiB / 2.43 s either way), because the
  rollout's own checkpointing already governs 8 of the 9 ESM invocations per
  step. `loss.endpoint_rollout.gradient_checkpointing`, by contrast, is
  load-bearing: turning it off OOMs even at batch 4.

Data loading is not a bottleneck at any of these settings: measured at
`num_workers: 2`, the loader supplies 236 samples/s against the ~7/s a batch-8
step consumes.

---

## 18. Two flow formulations, and why the default changed

There are two `flow.path_type` families in this repository and they solve
different problems. The distinction is not a tuning choice -- one of them is
structurally incapable of the task, and it took a complete 67,250-step run to
establish that, so the evidence is recorded here.

### The coordinate-space path (`linear`, `gaussian_bridge`)

```python
x_tau = (1 - tau) * x0 + tau * x1        # protein_flow/flow/paths.py
v     = x1 - x0
```

The flow runs from the source structure to the target structure. `x0` is both
the start of the flow and the model's conditioning, so **the base
distribution, given the conditioning, is a point mass**. The marginal velocity
field is `E[x1 - x0 | x_tau, x0]`, and at `tau=0` the state `x_tau = x0` adds
nothing to what is already conditioned on. What remains is `E[x1|x0] - x0`,
and for two frames drawn from the same equilibrium ensemble a few frames
apart, that expectation is x0 itself.

So the objective is a regression to the conditional mean, wearing the clothes
of a flow. Two consequences follow, both of which were measured:

- **The optimal map is near-identity.** A run that trained cleanly by every
  training-time metric (`fm_improvement` 31.2%, physics healthy) produced
  structures that moved 4.7% of the required distance. A sweep over constant
  rescalings of its rollout found no gain that beat the do-nothing baseline
  (k=1: -0.05%, k=10: -4.12%, k=100: -180%), so it was not an under-scaled
  field -- the direction at `tau=0` carried no usable information.
- **Sampling is deterministic.** Ensemble metrics (RMSF correlation,
  diversity) are not merely bad, they are undefined: every draw is identical.
  Measured diversity, over 8 draws from one source: exactly 0.000.

`evaluate_checkpoint.py`'s "improve %" column is the trap here. For any
stochastic model the conditional mean wins that comparison by construction,
so optimising it teaches the model to sit still. It is retained as a
diagnostic; it is not a success metric.

### The displacement path (`displacement`, the default for new runs)

```python
eps   = zero_com_gaussian(noise_scale)   # independent of the conditioning
x_t   = (1 - t) * eps + t * delta        # delta = x1 - x0
v     = delta - eps
```

The flow now runs in **displacement space**, from an isotropic Gaussian to the
displacement. `x0` is demoted to a side input: it builds the geometric graph
and conditions the network, but is never an endpoint. The base distribution is
therefore independent of the conditioning, the field must transport the
Gaussian onto the *whole* of `p(delta | x0, T)` rather than onto its mean, and
different noise draws give different structures.

Three implementation details are load-bearing:

**The decoder needs a wider output basis.** `vector_field.py` builds its
output as `sum_j a_ij * unit(x_j - x_i)` with invariant coefficients. At `t=0`
the correct velocity is `delta - eps == -eps == -flow_state`, an isotropic
vector that is in general orthogonal to everything that span can build: no
assignment of invariant coefficients can produce it. The decoder therefore
gains two zero-initialised terms, `b_i * s_i` and `mean_j c_ij * s_j`, which
is what makes its own target representable at all.
`tests/test_displacement_flow.py::test_flow_state_basis_is_what_makes_the_tau_zero_target_learnable`
fits `v = -flow_state` with and without them and is the regression test for
this.

**The noise must be zero-centre-of-mass.** Kabsch alignment sets
`centroid(x1_aligned) == centroid(x0)`, so `delta` has exactly zero COM
(measured: max |COM| = 5.2e-05 A over the 1,250-pair holdout). The decoder
removes the mean velocity, so any COM the noise starts with can never be
integrated away and would translate the generated structure.

**The flow state enters only through invariants and equivariant basis
vectors** -- a per-node magnitude embedding, three edge scalars
(`s_i . u_ij`, `s_j . u_ij`, `||s_j - s_i||`), and the two decoder terms
above. This is the only thing keeping the model SE(3)-equivariant now that a
raw 3-vector reaches the network.

`flow.noise_scale` is the per-axis standard deviation and defaults to 2.667 A,
matching the measured backbone displacement (RMS|delta| = 4.620 A globally,
1.989 A at 320 K rising to 7.797 A at 450 K). It is deliberately **one value
for all temperatures**: the 3.9x spread is what the temperature conditioning
has to learn, and matching sigma per temperature would supply it for free and
make the result unmeasurable.

### Cost

Carrying the flow state widens every EGNN layer's edge MLP and adds two
decoder heads. Measured at the 512-residue cap on a 95.0 GiB card:

| batch | coordinate-space | displacement | |
|---|---|---|---|
| 16 | 22.9 GiB, 0.997 s/step | 27.6 GiB, 1.270 s/step | +21% mem, +27% time |
| 32 | 42.8 GiB, 1.879 s/step | 53.7 GiB, 2.391 s/step | +25% mem, +27% time |
| 48 | 62.6 GiB, 2.818 s/step | 79.9 GiB, 3.568 s/step | +28% mem, +27% time |

Batch 32 is retained; 48 would leave 15 GiB of headroom, which does not
survive a multi-day DDP run.

### How to tell whether it worked

`velocity_ratio` and `velocity_cosine` are logged every validation and are the
collapse detectors -- the coordinate-space run sat at `velocity_ratio` 0.0024
for 34,500 steps while its MSE looked fine. End-to-end quality is
`scripts/evaluate_ensemble.py`, which compares the *distribution* of sampled
displacements against the MD one at the trained frame gap:

| metric | coordinate-space baseline | untrained displacement | target |
|---|---|---|---|
| moved (RMS abs delta gen / gt) | 0.040 | 0.61 | ~1.0 |
| spread (pairwise RMSD / gt) | 0.000 | 0.86 | ~1.41 |
| profile r | +0.306 | -- | > 0.5 |
| contact Jaccard | 0.888 | 0.27 | **match MD, see below** |

Two traps in that table, both of which cost a wrong conclusion before being
measured.

**`moved` and `spread` are nearly free at initialisation.** `noise_scale` was
chosen to match the data, and an untrained network barely perturbs the noise
it starts from, so `x0 + eps` already has the right scale -- while being a
destroyed protein (bond loss 9.93 at init, versus 0.43 after 250 steps). They
are necessary, not sufficient.

**Contact Jaccard must be compared against MD, not against 1.0.** The
intuition that "a 5-frame displacement barely changes the fold" is wrong, and
measuring it is what shows that. Real mdCATH pairs at `frame_gap: 5` score:

| | 320 K | 348 K | 379 K | 413 K | 450 K | all |
|---|---|---|---|---|---|---|
| GT contact Jaccard | 0.786 | 0.726 | 0.592 | 0.464 | 0.304 | **0.586** |
| GT RMS abs delta (A) | 2.32 | 3.51 | 5.04 | 7.00 | 10.30 | 5.46 |

Five frames is ~5.5 ns and at 450 K the fold genuinely reorganises. A model
scoring **above** these numbers is moving less than MD does -- which is the
previous formulation's failure, not a success. The coordinate-space
baseline's 0.888 is exactly that: it looks like the best number in the table
and it is the symptom of a model that does not move.

### What MD displacements actually look like

Getting the magnitude right is not the same as getting the motion right, and
contact Jaccard is the metric that separates them. The reason is that real
protein motion is **collective**. Measured over the holdout, the direction
correlation between two atoms' displacements as a function of their
separation in the source structure, `C(r) = <d_i.d_j> / <|d_i||d_j|>`:

| separation (A) | 0-4 | 4-6 | 6-8 | 8-12 | 12-16 | 16-24 | 24-32 | 32-48 | >48 |
|---|---|---|---|---|---|---|---|---|---|
| C(r) | 0.959 | 0.807 | 0.598 | 0.275 | 0.027 | -0.156 | -0.243 | -0.292 | -0.162 |

Two features matter. The correlation length is ~12 A -- whole substructures
slide together, which is why a 5 A displacement can leave most contacts
intact. And beyond 16 A the correlation goes **negative**: with the
rigid-body component removed by Kabsch, one part of the protein moving one
way forces another part the other way. That is the signature of collective
normal modes, and uncorrelated noise has neither feature (C(r) = 0
everywhere).

This is why `moved ~ 1.0` alongside a low contact Jaccard is a coherent
failure rather than a contradiction: an uncorrelated displacement of exactly
the right RMS scrambles far more contacts than a collective one of the same
size. `scripts/diagnostics/collectivity.py` prints this table for generated motion
next to the MD one; the shape of the gap says whether a fix has to be local
or global.

Measured on a trained displacement model, the generated curve tracks MD
exactly as far as the decoder's one-hop k-NN reach and then stops:

| separation (A) | 0-4 | 4-6 | 6-8 | 8-12 | 24-32 |
|---|---|---|---|---|---|
| MD | 0.955 | 0.786 | 0.556 | 0.222 | -0.232 |
| generated | 0.916 | 0.649 | 0.281 | -0.032 | -0.059 |

k=16 neighbours is a radius of 6-8 A at backbone resolution, and that is
precisely where the agreement ends. The decoder builds its output from
one-hop terms only, so this looked like an architectural ceiling rather than a
training shortfall.

**That conclusion was premature** -- it was measured mid-training. By step
65,000 the same curve reads 0.499 at 6-8 A and 0.153 at 8-12 A, so the decoder
does reach past its k-NN radius given enough training. Section 19 has the
comparison; the residual gap is real but roughly half what this table shows.

### The correlated-noise experiment (tried, measured, rejected)

Flow matching does not require an isotropic base -- only one that can be
sampled and is not a point mass. So the base can be *made* collective:
smooth the white noise over the source structure's own k-NN graph before
using it (`flow.noise_smoothing_rounds`, implemented and tested), and the
network no longer has to manufacture the correlation. Fitting the base
distribution's C(r) against MD's takes 8 rounds and works outright -- RMSE
falls 0.495 (0 rounds) -> 0.146 (4) -> 0.071 (8), negative tail included.

It still loses. Head-to-head at 1,500 steps on the same 4 domains and seed,
evaluated per temperature against MD on 2 held-out domains:

| | 0 rounds | 8 rounds | MD |
|---|---|---|---|
| moved | **1.261** | 1.610 | 1.0 |
| spread | **1.746** | 2.134 | 1.414 |
| profile r | **+0.533** | +0.429 | > 0.5 |
| contact Jaccard | **0.431** | 0.407 | 0.585 |
| bond | **1.330** | 2.097 | -- |
| velocity cosine | **0.759** | 0.555 | -- |
| fm improvement | **69.7%** | 45.4% | -- |

Smoothing led early on the physics terms -- bond 0.093 against 9.99 at step
0, since collective motion does not stretch bonds -- but the trained model
was worse everywhere, physics included.

The cause is a conflict with this repo's own feature design, not with the
idea. The flow state reaches the network through three invariant edge
scalars, one of which is `||s_j - s_i||` -- exactly what smoothing removes.
Its mean falls 6.00 -> 2.65 (4 rounds) -> 1.43 (8) while the other two hold
at ~2.64, starving a third of the flow-state signal
(`scripts/diagnostics/flow_state_features.py`). Retrying this is worthwhile
once that feature is made scale-free, e.g.
`||s_j - s_i|| / (||s_i|| + ||s_j||)`. Until then the default is 0.

### Two measurement traps this cost

Both were wrong assumptions that measurement caught, and both are easy to
repeat:

- **Contact Jaccard is temperature-dependent and needs averaging.** A single
  sample from a single mixed-temperature batch scored 0.24 for a checkpoint
  that scores 0.43 when properly averaged per temperature -- and 0.74 at
  320 K alone. Always compare against the `contact MD` column at the same
  temperature, never against a global number.
- **The ODE is converged by 20 steps.** `moved` and `spread` are flat from 20
  to 100 steps (`scripts/diagnostics/ode_step_sweep.py`), so over-dispersion
  is the field, not integration error. Sampling with 50 steps buys nothing
  over 20.

Note that script compares against the **5-frame displacement distribution**,
not the pooled full-trajectory ensemble that AlphaFlow-style evaluations use.
Those are different objects: a full 450 K mdCATH trajectory unfolds (RMSF 12.3
A about its own mean) and no single-shot 5-frame model can or should reproduce
that spread. Reaching it needs an autoregressive rollout this model is not
trained for -- and that is precisely where the sibling project's naive flow
matching came apart (contact Jaccard 0.115, i.e. diverse garbage).

## 19. The full mdCATH run, and where the remaining amplitude goes

### Conditions

| | |
|---|---|
| config | `configs/mdcath_backbone_rotate_displacement.yaml` |
| data | all of mdCATH by rotation -- 27 chunks x 200 shards, 5,398 domains, 3.61 TB |
| holdout | 25 domains x 5 temperatures, frozen, never rotated (1,250 pairs) |
| representation | `backbone` (N/CA/C/O), `mdcath_frame_gap: 5` (~5.5 ns) |
| flow | `displacement`, `noise_scale: 2.667` per axis, `noise_smoothing_rounds: 0` |
| loss | `lambda_direction: 0.1`, physics on `x0 + delta_hat`, `endpoint_enabled: false` |
| hardware | 2 x RTX PRO 6000 Blackwell (95.0 GiB), DDP, batch 32 per rank |
| schedule | `passes_per_chunk: 32`, lr 3e-4 |

Ran to completion: 27/27 chunks, **67,040 steps**, `exit 0`, zero restarts, no
`OOM` and no watchdog trip. It took two legs -- the first hung at chunk 12
(step 29,952) on a stalled download, which is what
`rotation.download_timeout_minutes` exists to break; the second leg resumed
from `last.pt` and ran the remaining 15 chunks in 12.5 h without incident.

Best checkpoint is step 65,000 (`val_loss` 7.0237). Validation was flat from
~58,000 onward and the last 2,000 steps did not beat it, so this is the
converged point for this capacity and this data.

| | coordinate-space run | this run |
|---|---|---|
| `fm_improvement` | 31.2% | **85.9%** |
| `velocity_ratio` | 0.0024 | **0.936** |
| `velocity_cosine` | -- | **0.877** |

### Result

`scripts/evaluate_ensemble.py`, 5 domains x 5 temperatures, 100 draws each,
50-step Heun, against 500 pooled MD displacement pairs per unit:

| | 320 K | 348 K | 379 K | 413 K | 450 K | all |
|---|---|---|---|---|---|---|
| moved | 0.803 | 0.637 | 0.683 | 0.743 | 0.822 | **0.738** |
| spread | 1.038 | 0.841 | 0.889 | 0.956 | 1.032 | **0.951** |
| profile r | +0.909 | +0.836 | +0.897 | +0.795 | +0.752 | **+0.838** |
| JS | 0.036 | 0.042 | 0.084 | 0.028 | 0.028 | **0.044** |
| contact J | 0.904 | 0.900 | 0.856 | 0.629 | 0.373 | **0.732** |
| contact MD | 0.765 | 0.735 | 0.665 | 0.481 | 0.363 | 0.602 |

`bond` 0.0088, `angle` 0.0167, `clash` 0.0051. Pairwise view
(`evaluate_checkpoint.py`, all 25 holdout domains): `moved` 0.812, RMSD
improve **-18.1%**, win rate 15.5%.

Three things are settled by this table. The deterministic collapse is gone --
`spread` and `profile r` are *defined* for the first time, and the previous
formulation's `moved` 0.047 is now 0.74-0.81. The temperature response works:
`moved` is flat across a 3.9x range of true displacement, so the model is
supplying the scale rather than inheriting it. And the protein survives --
24 of 25 units sit at or above the MD contact baseline, nowhere near the
sibling project's 0.115 failure. The one exception is `1sxjE02` at 450 K
(0.054 against MD's 0.271), a single unit at the hardest condition.

**The -18.1% RMSD improve is the expected sign, not a regression.** Which
thermal fluctuation actually occurred is not a function of the starting
structure (measured direction cosine between two MD displacements of the same
domain and temperature: **0.000**, `scripts/diagnostics/predictability.py`),
so that metric is minimised by not moving. The old run's 0.00% was the correct
answer to the wrong question. Any model that genuinely moves scores negative
here.

### What `moved = 0.74` decomposes into

The end-point number hides two independent deficits, and
`scripts/diagnostics/path_amplitude.py` separates them. Because
`x_t = (1-t)eps + t*delta` with `eps` independent of `delta`, the marginal
amplitude a correct sampler must reproduce at *every* tau is known in closed
form, `RMS|x_t| = sqrt((1-t)^2 a + t^2 b)` with `a = E|eps|^2`, `b = E|delta|^2`.
Measured against it (16 x 8 holdout pairs, 50-step Heun):

| tau | 0.00 | 0.20 | 0.30 | 0.50 | 0.70 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|
| RMS state | 4.619 | 3.919 | 3.774 | 3.945 | 4.626 | 5.625 | 6.191 |
| analytic | 4.619 | 3.979 | 3.918 | 4.351 | 5.345 | 6.653 | 7.374 |
| ratio | 1.000 | 0.985 | 0.963 | 0.907 | 0.866 | 0.845 | **0.840** |

and then, on the same samples, split by whether Kabsch alignment keeps it:

| | 320 K | 348 K | 379 K | 413 K | 450 K | all |
|---|---|---|---|---|---|---|
| generated, raw | 3.052 | 4.213 | 6.427 | 6.892 | 9.943 | 6.185 |
| generated, aligned | 2.770 | 3.869 | 5.688 | 6.329 | 9.152 | 5.633 |
| MD | 3.171 | 5.714 | 7.478 | 9.295 | 10.877 | 7.364 |
| rigid share of energy | 17.6% | 15.7% | 21.7% | 15.7% | 15.3% | **17.0%** |

The two factors multiply out exactly: `0.840 * sqrt(1 - 0.170) = 0.765`, which
is the aligned `moved` on this sample and consistent with the 0.738 the
ensemble script reports on its own.

**Deficit 1 -- the flow under-expands, and not at the end.** The natural
reading of a low `moved` is "the trajectory stops short of the structure". It
does not. The ratio is 1.000 at tau=0 and still 0.985 at tau=0.2; it departs
from tau~0.25 and degrades *monotonically* to 0.840. Note the analytic curve
is **not monotonic** -- with `a ~ b` the state must first contract to about
`sqrt(b/2)` near tau=0.3 and only then grow. The model tracks the contraction
(0.963) and then fails to deliver the expansion, losing a roughly constant
fraction per step over the whole second half. So it is not a truncated
trajectory; it is a velocity field that is systematically a little too small
wherever it has to *create* displacement structure.

`scripts/diagnostics/fm_by_tau.py` on the same checkpoint says the same thing
from the loss side -- `velocity_cosine` is 0.90-0.94 across tau 0.5-0.9, so
about a tenth of the predicted direction is wrong. Direction error that is
random would *add* variance; error that is systematic shrinkage toward the
conditional mean removes it, and an MSE-fitted field has exactly that bias
wherever it is under-resolved.

**This one is not a scale deficit, so no gain fixes it.** Sweeping a constant
multiplier on the velocity at inference
(`scripts/diagnostics/velocity_scale.py`):

| gain | 1.00 | 1.10 | 1.20 | 1.35 | 1.50 |
|---|---|---|---|---|---|
| moved | 0.741 | 0.911 | 1.112 | 1.515 | 2.106 |
| rigid share | 15.1% | 18.3% | 23.2% | 30.2% | 34.2% |
| contact J | **0.481** | 0.464 | 0.442 | 0.413 | 0.377 |

Contact Jaccard falls monotonically, and the extra amplitude goes
*disproportionately into rotation* (15% to 34%), i.e. the gain is amplifying
error rather than lengthening a correct vector. Note also where gain 1.0 sits:
contact J 0.481 against MD's 0.482 on this sample. **The model already
disturbs the fold exactly as much as MD does while moving only 74% as far.**

That reframes the deficit. Motion that is less collective breaks more contacts
per angstrom moved, and the collectivity table below shows the model is
measurably less collective than MD at every scale past one hop. The amplitude
shortfall is not an independent problem to be dialled out -- it is what the
collectivity shortfall looks like when the model is pushed to keep the fold
intact. Fixing it means making the motion more collective, not larger.

**Deficit 2 -- a sixth of the generated motion is a global rotation that
cannot score.** `delta` is defined by Kabsch-aligning `x1` to `x0`, so it has
**zero rigid-body content by construction**, and 17.0% of the generated
displacement energy is removed by alignment at evaluation time. That
amplitude is spent on a mode the target has no component along.

This is the same class of bug as the zero-COM invariant already documented
above, one derivative up. `remove_com_velocity` projects out *translation*;
nothing projected out *rotation*.

The obvious explanation for the 17% -- "the isotropic base noise carries a
rotation and the flow fails to cancel it" -- is **wrong, and measuring it is
what shows that**. An isotropic field on N particles has only `3/(3N-3)` of
its energy in the three rotation modes, which at N~270 is 0.4%, not 17%.
Measured with one estimator across four fields
(`scripts/diagnostics/rigid_share.py`):

| field | RMS (A) | RMS aligned | rigid share |
|---|---|---|---|
| base noise | 4.619 | 4.610 | **0.4%** |
| MD raw (before alignment) | 15.611 | 7.918 | 74.3% |
| MD aligned (the training target) | 7.918 | 7.918 | **0.0%** |
| generated | 6.754 | 6.154 | **17.0%** |

The `MD aligned` row is the control: the estimator reports 0.0% when there is
nothing to remove, so the 17.0% is real. And since the base contributes 0.4%,
**the rotation is manufactured by the network**, which means projecting the
base distribution would have fixed nothing. The velocity is where it has to be
removed.

Why the model manufactures it: a global rotation is the longest-wavelength
collective mode there is, and the decoder is weakest exactly there. The
supporting evidence sits where it should -- `fm_by_tau` measures
`velocity_cosine` **0.659 at tau=0**, its worst value anywhere on the path,
and tau=0 is where cancelling `eps` is the entire job.

**Fixed** by `flow.remove_rigid_motion` (`protein_flow/geometry/rigid.py`),
which projects both the base noise and the predicted velocity onto zero linear
*and* angular momentum about the source centroid. It is an orthogonal
projection onto a subspace that provably contains the target, so it can only
lower the flow-matching error. Applied to the existing checkpoint at inference
it takes the generated rigid share from 17.0% to 0.0%; the training-time gain
needs the run in
`configs/mdcath_backbone_rotate_displacement_norigid.yaml`. One flag drives
both ends on purpose -- projecting the velocity while leaving the noise alone
would strand the noise's own 0.4% in the state permanently, since with the
velocity constrained nothing could remove it afterwards.

### Stochastic versus deterministic

`moved` and `spread` together separate the mean displacement field from the
fluctuation about it. Writing the generated displacement as `m + n` with `n`
independent across draws, `moved^2 = (|m|^2 + E|n|^2)/b` and
`spread^2 ~ 2 E|n|^2 / b`, so from 0.738 and 0.951:

| | model | MD |
|---|---|---|
| fluctuation amplitude | 0.672 | 1.0 |
| mean-field amplitude | ~0.305 | ~0 (direction cos 0.000) |

So the honest statement is sharper than "moves 26% too little": **the thermal
fluctuation itself is about a third too small**, and a deterministic component
MD does not obviously have makes up part of the difference in the `moved`
number. Two caveats keep this an estimate rather than a measurement. `spread`
is a mean of pairwise RMSDs rather than a root-mean-square, which biases
`E|n|^2` slightly low and `|m|` correspondingly high; and separating a
spurious bias from genuine deterministic relaxation would need several MD
replicas launched from *one* structure, which mdCATH does not provide -- its
five replicas per (domain, temperature) start up to 25 A apart.

### What was ruled out

- **Integration error.** `scripts/diagnostics/ode_step_sweep.py` on this
  checkpoint: `moved` and `spread` are identical at 20, 50 and 100 steps
  (1.287/1.613, 1.305/1.634, 1.306/1.635). The field is what it is; more steps
  buy nothing. This also re-confirms the earlier finding at convergence.
- **A per-temperature noise-scale mismatch.** `noise_scale` is one global
  value, so the base sits at 1.46x the true displacement at 320 K and 0.43x at
  450 K -- a 3.4x range the flow has to supply itself. If that were the binding
  constraint, `moved` would fall monotonically with temperature. It does not:
  0.803, 0.637, 0.683, 0.743, 0.822, i.e. *best* at both ends and worst in the
  middle. The temperature conditioning is doing its job.
- **Domain size.** The five evaluated domains span 61 to 144 residues with no
  ordering in `moved` (the 144-residue domain is mid-pack).

### The collectivity ceiling moved

The one-hop limit recorded in section 18 was measured mid-training and is no
longer where it was. At step 65,000:

| separation (A) | 0-4 | 4-6 | 6-8 | 8-12 | 12-16 | 24-32 | 32-48 |
|---|---|---|---|---|---|---|---|
| MD | 0.958 | 0.810 | 0.601 | 0.273 | 0.019 | -0.259 | -0.260 |
| generated (step 65k) | 0.939 | 0.749 | 0.499 | 0.153 | -0.085 | -0.171 | -0.150 |
| generated (mid-training) | 0.916 | 0.649 | 0.281 | -0.032 | -- | -0.059 | -- |

At 6-8 A the model went 0.281 -> 0.499 against MD's 0.601, and at 8-12 A
-0.032 -> 0.153 against 0.273. The decoder does reach past its k-NN radius
given enough training, so "architectural ceiling" was too strong a claim. What
remains is a consistent under-correlation of about -0.10 through the 6-16 A
band and a long-range negative tail at roughly 60% of MD's depth -- the model
is still measurably *less* collective than MD at every scale beyond one hop.

### What to try next, in order

1. **Done: `flow.remove_rigid_motion`.** Addresses deficit 2, a measured 17.0%
   of the output energy, and is a strict reduction of the flow-matching error
   rather than a trade. Run
   `configs/mdcath_backbone_rotate_displacement_norigid.yaml` to collect it.
2. **Make `||s_j - s_i||` scale-free** (e.g. `/ (||s_i|| + ||s_j||)`) and retry
   `noise_smoothing_rounds`. Correlated noise was rejected for starving that
   feature, and the residual under-collectivity is exactly what it was meant
   to fix. This is the cheapest attack on deficit 1.
3. Widen the decoder's reach -- larger `knn_k`, more geometric layers, or a
   genuine collective-mode term. The step sweep says the field is the limit
   and the gain sweep says the field's *direction* is the limit, so this is
   where the remaining 0.840 lives. It costs a retrain and a memory decision,
   which is why it is last rather than first.

A note on what not to do: **do not chase `moved` with a velocity gain.** It
works on the metric and breaks the protein, and the run that most needs
watching for this is the one that follows a genuine fix, because the fix will
move `moved` in the same direction that the mistake does.
