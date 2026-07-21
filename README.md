# protein_dual_flow

A research prototype for modeling protein conformational dynamics with
**conditional flow matching** over a **dual-graph** (sequence + geometric)
representation of the C-alpha backbone.

This is an MVP built to be run, not just read: `python train.py --config
configs/default.yaml` trains on a synthetic dataset in seconds on CPU, and
`pytest` exercises every component, including an automated E(3)-equivariance
check.

## 1. What this model does

Given two C-alpha structures of the *same* protein sampled at different
points in a trajectory (`source_coords` at time `t`, `target_coords` at time
`t + physical_delta_t`), the model learns a velocity field that
continuously deforms one structure into the other. At inference time,
integrating that velocity field from `t=0` to `t=1` (an ODE, not a single
forward pass) generates a plausible intermediate/next conformation starting
from `source_coords`.

The model conditions on:
- a **precomputed PLM (e.g. ESM) residue embedding** per residue,
- the **amino-acid type** per residue,
- the **temperature** of the simulation,
- the **physical time gap** between source and target frames.

## 2. Dual-graph architecture

Two different graphs are built over the same set of residues, encoded
separately, and then fused per-residue:

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
   number of valid residues). An EGNN-style layer stack produces
   rotation/translation-**invariant** hidden features from this graph.

3. **Gated fusion** (`protein_flow/models/fusion.py`) combines the two
   per-residue representations with a learned, sigmoid gate conditioned on
   a sinusoidal embedding of flow time `tau`, temperature, and
   `physical_delta_t`.

4. **Equivariant vector-field decoder** (`protein_flow/models/vector_field.py`)
   turns the fused invariant features into a `[B, L, 3]` velocity by
   weighting the geometric graph's *relative position vectors*
   `x_j - x_i` with learned invariant scalar coefficients -- never by
   emitting raw xyz from an MLP. See Section 5 for why this guarantees
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

## 6. How E(3)-equivariance is guaranteed

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

This is verified, not just asserted: `tests/test_equivariance.py` runs the
full model on tie-free (no k-NN distance ties), fixed-seed coordinates and
checks `v(x @ R + b) ~= v(x) @ R` (atol/rtol = 1e-4) plus a
translation-only invariance check, including with padding residues present.
`tests/test_geometric_encoder.py` checks the same property at the
sub-module level.

## 7. Physics-informed losses

`protein_flow/losses/physics.py` implements:

- **Bond-distance loss**: MSE between predicted and reference consecutive
  C-alpha distances.
- **Bond-angle loss**: cosine-based MSE over consecutive C-alpha triplets
  (numerically stable near 0/180 degrees).
- **Steric-clash loss**: `relu(clash_threshold - distance)^2` over
  non-covalently-adjacent pairs, computed over the same sparse k-NN
  geometric graph (no `L x L` pair tensor).
- **Optional endpoint RMSD loss** (`loss.endpoint_enabled`, default
  `false`): masked RMSD between an ODE-rollout endpoint and the aligned
  target. **This is disabled by default because it requires unrolling the
  full ODE integrator (`sampling.num_steps` model calls) inside every
  training step with gradients enabled, which is far more expensive per
  step than the other, single-shot losses.** Enable it only if you
  understand this cost.

`loss.physics.apply_to` controls whether bond/angle/clash losses are
applied directly to `x_tau` or to a one-step Euler prediction
`x_pred = x_tau + step_scale * predicted_velocity` (the default).
`loss.physics.d_ref_source` controls whether reference bond
distances/angles come from `source_coords` or the aligned target.

Total loss:
```
L_total = lambda_fm * L_fm + lambda_bond * L_bond + lambda_angle * L_angle
        + lambda_clash * L_clash + lambda_endpoint * L_endpoint
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

## 9. From C-alpha MVP to backbone / local-frame / torsion flows

The current MVP flows raw C-alpha xyz coordinates. The `Dataset`/`collate`
contract (`protein_flow/data/dataset.py`) and the `FlowPath` abstraction
(`protein_flow/flow/paths.py`) are deliberately decoupled from "coordinates
are `[B, L, 3]`" so this can be extended without touching graph
construction or flow matching:

- **Full backbone** (`[B, L, 4, 3]` for N/CA/C/O): swap the coordinate
  tensor's last two dims; the geometric graph would key off CA (or a
  configurable atom) for k-NN while carrying the other atoms as additional
  per-node vector channels through the same equivariant machinery
  (relative vectors between same-type atoms remain equivariant).
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

This repository does not assume mdCATH's on-disk format -- it only defines
the tensor schema below and a synthetic dataset that satisfies it. We did
verify (against a local sample file, `mdcath_dataset_1a87A01.h5`) that
mdCATH HDF5 files store **all-atom** coordinates keyed by
`<domain>/<temperature>/<replica>/coords` with shape
`[num_frames, num_atoms, 3]` (e.g. 1607 atoms for a 97-residue domain, per
`dssp`/`rmsf` array lengths) alongside `forces`, `rmsd`, `gyrationRadius`,
and `dssp`. Extracting C-alpha-only coordinates from this requires a
topology (atom -> residue/element mapping, typically a companion PDB/PSF
file) that we did not inspect here -- do not assume a fixed atom-index
formula for CA without checking that topology for the domains you use.
ATLAS has its own, separately-documented layout and was not inspected at
all for this project.

A real adapter should subclass `protein_flow.data.dataset.ProteinTrajectoryDataset`
and, per `__getitem__`, return a dict matching exactly what
`SyntheticProteinTrajectoryDataset` returns:

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

### Optional ESM embedding adapter

ESM itself is never run inside training or the test suite (no network/model
download required). `protein_flow/data/synthetic.py` shows the expected
shape; a real adapter would look like:

```python
# not part of the default training path -- illustrative only
def compute_esm_embeddings(sequences: list[str]) -> list[torch.Tensor]:
    import esm
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    ...  # tokenize, forward pass, extract per-residue representations
```

## 11. Running it

```bash
pip install -r requirements.txt

# Train on the synthetic dataset (CPU, a few seconds):
python train.py --config configs/default.yaml

# Sample from a trained checkpoint:
python sample.py --checkpoint checkpoints/best.pt --num-steps 20 --solver heun
```

Expected shapes for the default config (`plm_dim=320`, `batch_size=8`,
residue length variable in `[16, 48]`, padded to the batch max `L`):

| Tensor | Shape |
|---|---|
| `sequence_embedding` | `[B, L, 320]` |
| `source_coords`, `target_coords`, `aligned_target` | `[B, L, 3]` |
| `residue_mask` | `[B, L]` bool |
| `predicted_velocity` | `[B, L, 3]` |
| `model.sample(...)` trajectory | `[num_steps + 1, B, L, 3]` |

## 12. Tests

```bash
pytest -q
```

64 tests cover: masked batched Kabsch alignment (including that internal
deformation survives alignment while rigid transforms are undone), sparse
sequence/geometric graph construction (padding, no cross-batch edges, safe
`k > valid_residues`, radius cutoff), the flow path and flow-matching loss,
the sequence encoder, the geometric encoder + equivariant decoder
(including a direct rotation/translation equivariance check), gated
fusion + the full model, all physics losses, the Euler/Heun ODE solver
(checked against analytic constant- and linear-velocity fields), the
synthetic dataset + collate, a full training-loop smoke test (including an
overfit-one-batch check that loss actually decreases), the dedicated
full-model equivariance suite (`tests/test_equivariance.py`), and one
end-to-end CPU smoke test wiring every stage together.

## 13. Optional dependencies

`torch_scatter` is never required: `protein_flow/utils.py` provides a
pure-PyTorch fallback (`index_add_`-based) for every scatter-aggregation
used in the sequence encoder, geometric encoder, and vector-field decoder,
and uses `torch_scatter` automatically if it happens to be installed.
