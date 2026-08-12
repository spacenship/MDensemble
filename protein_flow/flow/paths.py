"""Conditional flow-matching paths.

The flow time ``tau`` lives in [0, 1] and is a purely mathematical
interpolation variable -- it must never be interpreted as physical MD time.
Physical time (``physical_delta_t``) is a separate conditioning input to the
model (see :mod:`protein_flow.models.fusion`).

Two families live here, and they differ in *what the flow transports*:

``LinearPath`` / ``GaussianBridgePath`` flow **structure to structure**:
the state is a coordinate tensor moving from x0 to x1. Because x0 is also
the model's conditioning, the base "distribution" is a point mass, so the
marginal velocity field at tau=0 reduces to ``E[x1 | x0] - x0``. For two
frames drawn from the same equilibrium ensemble that expectation is
essentially x0 itself, which makes the optimal field ~0 and the sampler
deterministic. Measured on a fully trained checkpoint: the rollout moved
4.7% of the required distance, and no constant rescaling of its output beat
the do-nothing baseline.

``DisplacementPath`` flows **noise to displacement**: the state lives in
displacement space, the base distribution is an isotropic Gaussian that is
independent of the conditioning, and the source structure is handed to the
model as a side input instead of being the start of the flow. That makes it
a genuine conditional generative model -- different noise draws give
different structures -- which is what the ensemble metrics (RMSF
correlation, diversity ratio) require in order to be defined at all.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from protein_flow.geometry.rigid import remove_rigid_motion as project_out_rigid_motion
from protein_flow.utils import masked_mean


def sample_tau(batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Sample one flow time tau ~ Uniform(0, 1) per batch element."""
    return torch.rand(batch_size, device=device, dtype=dtype)


def smooth_over_graph(noise: Tensor, coords: Tensor, particle_mask: Tensor, rounds: int, knn_k: int) -> Tensor:
    """Average each particle's noise with its k-NN neighbours, ``rounds`` times.

    Turns white noise into noise that looks like protein motion. Real
    displacements are collective: measured over the holdout, the direction
    correlation between two atoms' displacements is 0.96 within 4 A, 0.60 at
    6-8 A, and goes *negative* beyond 16 A (with the rigid-body component
    removed, one part of the protein moving one way forces another the other
    way -- the signature of collective normal modes). White noise has none of
    that, so a model starting from it has to manufacture the correlation from
    nothing, and measurably fails to past its decoder's one-hop reach.

    Each round is one hop over the same k-NN graph the encoder builds, so the
    correlation length grows with ``rounds``. Fitted against the MD curve, the
    RMSE falls 0.495 (white) -> 0.146 (4 rounds) -> 0.071 (8 rounds), and 8
    rounds reproduces the negative long-range tail as well.

    Note the graph comes from ``coords`` -- the source structure -- so the
    smoothing follows spatial proximity, not sequence order: residues far
    apart in sequence but touching in space move together, which is what
    "collective" means.
    """
    if rounds < 1:
        return noise
    from protein_flow.geometry.graph import build_geometric_graph
    from protein_flow.utils import scatter_mean

    batch_size, length, _ = noise.shape
    graph = build_geometric_graph(coords, particle_mask, k=knn_k)
    if graph.edge_index.shape[1] == 0:
        return noise
    src, dst = graph.edge_index
    flat = noise.reshape(batch_size * length, 3)
    for _ in range(rounds):
        flat = 0.5 * flat + 0.5 * scatter_mean(flat[src], dst, dim_size=flat.shape[0])
    return flat.reshape(batch_size, length, 3)


def sample_zero_com_noise(
    shape: Tuple[int, int, int],
    particle_mask: Tensor,
    noise_scale: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    coords: Optional[Tensor] = None,
    smoothing_rounds: int = 0,
    knn_k: int = 16,
    remove_rigid_motion: bool = False,
) -> Tensor:
    """Gaussian noise projected onto the zero-centre-of-mass subspace.

    With ``smoothing_rounds > 0`` (and ``coords`` supplied) the draw is first
    smoothed over the source structure's k-NN graph, which gives it the
    spatial correlation real protein motion has -- see
    :func:`smooth_over_graph`. The result is rescaled so ``noise_scale``
    still means the same thing: smoothing removes variance, and without the
    rescale the base distribution would silently shrink as rounds increase.

    Kabsch alignment sets ``centroid(x1_aligned) == centroid(x0)``, so the
    displacement ``x1 - x0`` has *exactly* zero centre of mass (measured:
    max |COM| = 5.2e-05 A over the 1,250-pair holdout). The base distribution
    has to live in that same subspace: the decoder removes the mean velocity
    (``decoder.remove_com_velocity``), so any centre-of-mass component the
    noise starts with can never be integrated away and would translate the
    whole generated structure.

    With ``remove_rigid_motion`` the same argument is extended to the three
    *rotation* modes, which Kabsch removes from ``delta`` just as exactly as
    it removes translation. The share of an isotropic draw that this takes
    away is small by construction -- ``3/(3N-3)``, measured 0.4% at N~270 --
    so it is not where the trained model's 17% rotational output came from.
    It is done anyway because nothing downstream can remove it: with the
    velocity projected, whatever rotation the noise starts with survives to
    the end of the trajectory untouched.

    Padding positions are returned as exactly zero and take no part in the
    mean, so a batch's translation does not depend on how much padding it
    happens to carry.
    """
    # torch.randn refuses a generator whose device differs from the output's,
    # and the natural thing for a caller to hold is a CPU generator (seeding a
    # CUDA one per draw is awkward and makes results device-dependent). Draw on
    # the generator's own device and move, which also makes a given seed
    # reproduce the same structure on CPU and GPU.
    draw_device = generator.device if generator is not None else device
    noise = torch.randn(shape, device=draw_device, dtype=dtype, generator=generator)
    noise = noise.to(device)
    mask = particle_mask.unsqueeze(-1).to(dtype)
    noise = noise * mask

    if smoothing_rounds > 0:
        if coords is None:
            raise ValueError(
                "smoothing_rounds > 0 needs coords: the correlation follows the source "
                "structure's k-NN graph, which cannot be built without it."
            )
        noise = smooth_over_graph(noise, coords, particle_mask, smoothing_rounds, knn_k) * mask

    centre = masked_mean(noise, particle_mask, dim=1)  # [B, 3]
    noise = (noise - centre.unsqueeze(1)) * mask

    if remove_rigid_motion:
        if coords is None:
            raise ValueError(
                "remove_rigid_motion needs coords: the rotation is taken about the source "
                "structure's centroid, which cannot be located without it."
            )
        noise = project_out_rigid_motion(noise, coords, particle_mask) * mask

    # Rescale to the requested per-axis scale. Both the smoothing and the
    # centre-of-mass projection shrink the variance, so this is what keeps
    # noise_scale meaning "per-axis standard deviation" independently of them.
    per_axis = (noise.pow(2).sum() / (mask.sum() * 3).clamp(min=1.0)).sqrt()
    return noise * (noise_scale / per_axis.clamp(min=1e-8))


def clean_displacement(state: Tensor, velocity: Tensor, tau: Tensor) -> Tensor:
    """Reconstruct the clean displacement from a point on a displacement path.

    Along ``x_tau = (1 - tau) * eps + tau * delta`` with velocity
    ``delta - eps``, ``x_tau + (1 - tau) * velocity == delta`` exactly, for
    every tau. This is what the physics terms are evaluated on -- adding it
    to x0 gives a real candidate structure rather than an interpolant.
    """
    return state + (1.0 - tau).view(-1, 1, 1) * velocity


class FlowPath(abc.ABC):
    """Interface for a conditional probability path between x0 and x1."""

    #: When True, ``sample`` returns a state in *displacement* space and the
    #: model must be given x0 separately (as the coordinates that build the
    #: geometric graph) plus the state as ``flow_state``. When False, the
    #: returned state is itself the coordinate tensor to encode.
    flows_in_displacement_space: bool = False

    @abc.abstractmethod
    def sample(
        self,
        x0: Tensor,
        x1: Tensor,
        tau: Tensor,
        particle_mask: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return (state, target_velocity) for the given endpoints and tau.

        Args:
            x0: [B, L, 3] source coordinates.
            x1: [B, L, 3] Kabsch-aligned target coordinates.
            tau: [B] flow time in [0, 1].
            particle_mask: [B, L] bool particle validity. Required by paths
                that draw their own noise; ignored by the coordinate-space
                paths, which are fully determined by their endpoints.
            generator: optional RNG for reproducible noise draws.

        Returns:
            state: [B, L, 3] the point on the path (coordinates, or a
                displacement -- see ``flows_in_displacement_space``).
            target_velocity: [B, L, 3] velocity the model should regress to.
        """
        raise NotImplementedError


class LinearPath(FlowPath):
    """Deterministic straight-line conditional path (default, fully supported).

    x_tau = (1 - tau) * x0 + tau * x1
    target_velocity = x1 - x0
    """

    def sample(
        self,
        x0: Tensor,
        x1: Tensor,
        tau: Tensor,
        particle_mask: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        tau_b = tau.view(-1, 1, 1)
        x_tau = (1.0 - tau_b) * x0 + tau_b * x1
        target_velocity = x1 - x0
        return x_tau, target_velocity


class GaussianBridgePath(FlowPath):
    """Noisy bridge path around the linear mean (future-extension stub).

    x_tau = (1 - tau) * x0 + tau * x1 + sigma_min * tau * (1 - tau) * eps

    The added noise vanishes at both endpoints (tau=0 and tau=1) so x0/x1
    are recovered exactly, but the conditional target velocity here still
    uses the linear-mean derivative (x1 - x0) and ignores the noise-term
    derivative. This is a simplified approximation intended as scaffolding
    for future work, not a rigorously derived stochastic bridge -- it is
    not used by the default config (``flow.path_type: linear``).
    """

    def __init__(self, sigma_min: float = 0.0):
        self.sigma_min = sigma_min

    def sample(
        self,
        x0: Tensor,
        x1: Tensor,
        tau: Tensor,
        particle_mask: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        tau_b = tau.view(-1, 1, 1)
        mean = (1.0 - tau_b) * x0 + tau_b * x1
        target_velocity = x1 - x0
        if self.sigma_min > 0.0:
            noise_scale = self.sigma_min * tau_b * (1.0 - tau_b)
            x_tau = mean + noise_scale * torch.randn_like(mean)
        else:
            x_tau = mean
        return x_tau, target_velocity


class DisplacementPath(FlowPath):
    """Straight-line path from isotropic noise to the displacement x1 - x0.

        x_t   = (1 - t) * eps + t * delta,   delta = x1 - x0, eps ~ N(0, s^2 I)
        v     = delta - eps

    The source structure x0 does not appear here at all: it is passed to the
    model as conditioning (it builds the geometric graph), not as an endpoint
    of the flow. That independence is the whole point -- the base
    distribution no longer collapses to a point mass given the conditioning,
    so the marginal field transports the Gaussian onto the *full*
    p(delta | x0, T), not onto its mean.

    ``noise_scale`` is the per-axis standard deviation and should match the
    scale of the data being transported. Measured over the 1,250-pair
    holdout at backbone resolution, RMS|delta| = 4.620 A globally (1.989 A at
    320 K rising to 7.797 A at 450 K), which is a per-axis sigma of 2.667 A.
    Deliberately *not* matched per temperature: the 3.9x spread across
    temperatures is exactly what the temperature conditioning has to learn,
    and handing it over through the noise scale would make that result
    unmeasurable.
    """

    flows_in_displacement_space = True

    def __init__(self, noise_scale: float = 2.667, smoothing_rounds: int = 0, knn_k: int = 16,
                 remove_rigid_motion: bool = False):
        if noise_scale <= 0.0:
            raise ValueError(f"DisplacementPath needs a positive noise_scale, got {noise_scale}")
        if smoothing_rounds < 0:
            raise ValueError(f"smoothing_rounds must be >= 0, got {smoothing_rounds}")
        self.noise_scale = noise_scale
        self.smoothing_rounds = smoothing_rounds
        self.knn_k = knn_k
        self.remove_rigid_motion = remove_rigid_motion

    def sample(
        self,
        x0: Tensor,
        x1: Tensor,
        tau: Tensor,
        particle_mask: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        if particle_mask is None:
            raise ValueError(
                "DisplacementPath.sample needs particle_mask: the noise is projected onto the "
                "zero-centre-of-mass subspace over valid particles only."
            )
        delta = (x1 - x0) * particle_mask.unsqueeze(-1).to(x0.dtype)
        eps = sample_zero_com_noise(
            delta.shape, particle_mask, self.noise_scale, delta.device, delta.dtype, generator,
            coords=x0, smoothing_rounds=self.smoothing_rounds, knn_k=self.knn_k,
            remove_rigid_motion=self.remove_rigid_motion,
        )
        tau_b = tau.view(-1, 1, 1)
        state = (1.0 - tau_b) * eps + tau_b * delta
        return state, delta - eps


def build_flow_path(
    path_type: str,
    sigma_min: float = 0.0,
    noise_scale: float = 2.667,
    smoothing_rounds: int = 0,
    knn_k: int = 16,
    remove_rigid_motion: bool = False,
) -> FlowPath:
    """Factory selecting a :class:`FlowPath` implementation by name."""
    if path_type == "linear":
        return LinearPath()
    if path_type == "gaussian_bridge":
        return GaussianBridgePath(sigma_min=sigma_min)
    if path_type == "displacement":
        return DisplacementPath(
            noise_scale=noise_scale, smoothing_rounds=smoothing_rounds, knn_k=knn_k,
            remove_rigid_motion=remove_rigid_motion,
        )
    raise ValueError(f"Unknown flow path_type: {path_type!r}")
