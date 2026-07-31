# protein_dual_flow

A research prototype for modeling protein conformational dynamics with
**conditional flow matching** over a **dual-graph** (sequence + geometric)
representation of a protein, at either C-alpha or all-atom resolution.

This is built to be run, not just read: `python train.py --config
configs/default.yaml` trains on a synthetic dataset in seconds on CPU,
`configs/mdcath_full.yaml` trains a 227M-parameter all-atom model on real
mdCATH trajectories, and `pytest` (135 tests) exercises every component,
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
│   └── ablations/               # frame-gap ablation configs
├── protein_flow/
│   ├── config.py                # the whole config tree as nested dataclasses
│   ├── data/
│   │   ├── dataset.py           # storage-agnostic Dataset interface
│   │   ├── synthetic.py         # runnable-anywhere synthetic trajectories
│   │   ├── mdcath.py            # real mdCATH HDF5 adapter
│   │   ├── topology.py          # covalent bonds/angles from CHARMM PSF
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
│   └── inference.py
├── scripts/
│   ├── run_training_tmux.sh     # detached background training (see §11)
│   └── run_mdcath_gap_ablation.sh
├── tests/                       # 135 tests
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

The flowing particles are either one C-alpha per residue or every
non-hydrogen atom, selected by `data.representation` (§13).

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
   embedding of flow time `tau`, temperature, and `physical_delta_t`. In
   all-atom mode the residue-level sequence representation is first
   broadcast down to each atom of its residue, so a residue's language-model
   context reaches all of its atoms.

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
applied directly to `x_tau` or to a one-step Euler prediction
`x_pred = x_tau + step_scale * predicted_velocity` (the default).
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

`default` is the only one that needs no external data and runs on CPU in
seconds; everything else expects mdCATH shards at `data.mdcath_dir`.

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
- **Validation runs on rank 0 only**, over the full validation set, using
  the unwrapped module. A DDP-wrapped forward would hang there, since the
  other ranks are waiting at a barrier and would never join the gradient
  sync. The resulting metrics are broadcast so every rank makes the same
  learning-rate-scheduler and best-checkpoint decisions. Because the other
  ranks idle during this, consider capping it with `train.val_max_batches`
  (the uncapped mdCATH split is ~375 batches x 3 tau values ~= 1000
  forwards, which took ~5 minutes on the full config).
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

135 tests cover: masked batched Kabsch alignment (including that internal
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
the dataset is sharded).

Tests that would need external data or downloads skip cleanly: the real
mdCATH tests skip when `mdCATH_sample100/` is absent, and the ESM tests
build a tiny randomly-initialised ESM locally rather than downloading the
150M checkpoint.

## 13. All-atom representation

`data.representation` selects what the flow acts on:

| | `ca` (default) | `heavy_atom` |
|---|---|---|
| particles | 1 per residue | every non-hydrogen atom |
| count | `L` | `~7.9 * L` |
| covalent topology | "adjacent in sequence" | real CHARMM bonds/angles from the shard's PSF |
| clash threshold | 3.5 A | 2.5 A |

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

## 16. Optional dependencies

`torch_scatter` is never required: `protein_flow/utils.py` provides a
pure-PyTorch fallback (`index_add_`-based) for every scatter-aggregation
used in the sequence encoder, geometric encoder, and vector-field decoder,
and uses `torch_scatter` automatically if it happens to be installed.

`h5py` is only required for `data.source: mdcath`, and `transformers` only
for ESM (either precomputation or in-graph fine-tuning). The default
synthetic pipeline needs neither, and the test suite skips or stubs both.

### A note on AMP

`torch.linalg.svd` has no half-precision CUDA kernel, so enabling
`train.amp` used to crash inside Kabsch alignment with
`"svd_cuda_gesvdjBatched" not implemented for 'Half'`. Alignment now forces
full precision internally and casts back to the caller's dtype -- it is
cheap `no_grad` preprocessing, and an SVD driving a rigid-body fit is
exactly the kind of ill-conditioned operation that should not run in fp16.
`tests/test_kabsch.py` covers this. AMP cuts training memory by roughly 40%
and is enabled in all the scaled configs.
