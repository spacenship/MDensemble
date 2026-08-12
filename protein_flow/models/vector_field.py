"""E(3)-equivariant vector-field decoder.

The decoder never predicts raw xyz vectors from an MLP. Instead, for every
geometric-graph edge it predicts *invariant* scalar coefficients from
invariant inputs (fused hidden states, distance, RBF features), and builds
the output velocity as a weighted sum of the *equivariant* relative
position vectors ``x_j - x_i``:

    a_ij = phi_x(h_i, h_j, edge_scalars_ij)
    v_i  = sum_j a_ij * (x_j - x_i) / (||x_j - x_i|| + eps)

This guarantees rotation-equivariance and translation-invariance of the
output by construction, as long as ``h`` is itself invariant.

When the model carries a flow state (the displacement path), that span is
not enough. At tau=0 the correct velocity is ``delta - eps == -eps ==
-flow_state``, an isotropically-distributed vector that is in general
*orthogonal* to everything the graph's relative vectors can build: no
assignment of invariant coefficients a_ij can produce it, because the
coefficients cannot know the noise direction. The decoder therefore takes
two further equivariant terms, each weighted by its own invariant scalar:

    v_i += b_i * s_i                       (the node's own flow state)
    v_i += mean_j c_ij * s_j               (its neighbours' flow states)

Both coefficient heads are zero-initialised, so a freshly built decoder is
numerically identical to one without them and training moves b_i toward -1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from protein_flow.geometry.graph import GeometricGraph
# Aliased: the decoder carries a boolean attribute of the same name.
from protein_flow.geometry.rigid import remove_rigid_motion as project_out_rigid_motion
from protein_flow.models.geometric_encoder import build_edge_scalar_features, edge_scalar_dim
from protein_flow.utils import masked_mean, scatter_mean


class VectorFieldDecoder(nn.Module):
    """Decodes fused invariant node features into an equivariant [B, L, 3] velocity.

    Combines two equivariant terms built from the same geometric-graph
    edges: a local geometric term and a second ("residual") term with
    independently-learned coefficients. Optionally removes the masked
    center-of-mass velocity from the combined output.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_rbf: int = 16,
        rbf_min_dist: float = 0.0,
        rbf_max_dist: float = 20.0,
        remove_com_velocity: bool = True,
        use_flow_state: bool = False,
        remove_rigid_motion: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.rbf_min_dist = rbf_min_dist
        self.rbf_max_dist = rbf_max_dist
        self.remove_com_velocity = remove_com_velocity
        self.use_flow_state = use_flow_state
        self.remove_rigid_motion = remove_rigid_motion
        self.eps = eps

        feature_dim = 2 * hidden_dim + edge_scalar_dim(num_rbf, with_flow_state=use_flow_state)

        def make_coefficient_mlp(input_dim: int = feature_dim) -> nn.Module:
            layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
            for _ in range(max(num_layers - 2, 0)):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*layers)

        self.local_coeff_mlp = make_coefficient_mlp()
        self.residual_coeff_mlp = make_coefficient_mlp()

        self.state_coeff_mlp = None
        self.state_neighbour_mlp = None
        if use_flow_state:
            self.state_coeff_mlp = make_coefficient_mlp(hidden_dim)
            self.state_neighbour_mlp = make_coefficient_mlp()
            # Zero-init the readout so the decoder starts out identical to the
            # flow-state-free one; the gradient then drives b_i toward -1,
            # which is what cancels the noise at the start of the trajectory.
            for head in (self.state_coeff_mlp, self.state_neighbour_mlp):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        h_fused: Tensor,
        graph: GeometricGraph,
        residue_mask: Tensor,
        flow_state: Tensor | None = None,
        coords: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            h_fused: [B, L, hidden_dim] invariant fused node features.
            graph: :class:`GeometricGraph` built from the encoded coordinates.
            residue_mask: [B, L] bool.
            flow_state: [B, L, 3] point on the displacement path; required
                when the decoder was built with ``use_flow_state=True``.
            coords: [B, L, 3] the coordinates the graph was built from.
                Required only when ``remove_rigid_motion`` is set, which needs
                them to build the three rotation generators.

        Returns:
            [B, L, 3] predicted velocity field.
        """
        batch_size, length, hidden_dim = h_fused.shape
        num_nodes = batch_size * length
        device, dtype = h_fused.device, h_fused.dtype

        if self.use_flow_state and flow_state is None:
            raise ValueError(
                "VectorFieldDecoder was built with use_flow_state=True but no flow_state was "
                "passed; without it the decoder cannot represent the velocity at tau=0."
            )
        if flow_state is not None and not self.use_flow_state:
            raise ValueError(
                "flow_state was passed to a VectorFieldDecoder built with use_flow_state=False; "
                "its coefficient MLPs are sized for the smaller edge-feature vector."
            )

        if graph.edge_index.shape[1] == 0:
            return torch.zeros(batch_size, length, 3, device=device, dtype=dtype)

        src, dst = graph.edge_index
        h_flat = h_fused.reshape(num_nodes, hidden_dim)
        flow_state_flat = flow_state.reshape(num_nodes, 3) if flow_state is not None else None
        edge_scalars = build_edge_scalar_features(
            graph, self.num_rbf, self.rbf_min_dist, self.rbf_max_dist, flow_state_flat, self.eps
        )
        edge_input = torch.cat([h_flat[src], h_flat[dst], edge_scalars], dim=-1)

        local_coeff = self.local_coeff_mlp(edge_input)  # [E, 1], invariant
        residual_coeff = self.residual_coeff_mlp(edge_input)  # [E, 1], invariant

        unit_vectors = graph.relative_vectors / (graph.distances.unsqueeze(-1) + self.eps)  # equivariant

        combined_coeff = local_coeff + residual_coeff
        weighted = combined_coeff * unit_vectors  # [E, 3], equivariant
        velocity_flat = scatter_mean(weighted, dst, dim_size=num_nodes)  # equivariant

        if flow_state_flat is not None:
            neighbour_coeff = self.state_neighbour_mlp(edge_input)  # [E, 1], invariant
            velocity_flat = velocity_flat + scatter_mean(
                neighbour_coeff * flow_state_flat[src], dst, dim_size=num_nodes
            )
            velocity_flat = velocity_flat + self.state_coeff_mlp(h_flat) * flow_state_flat

        velocity = velocity_flat.reshape(batch_size, length, 3)
        mask = residue_mask.unsqueeze(-1).to(dtype)
        velocity = velocity * mask

        # Applied after the flow-state terms on purpose: the displacement the
        # flow transports is exactly zero-centre-of-mass (Kabsch guarantees
        # it), so the output has to be projected back into that subspace after
        # every contribution, not just the graph one.
        if self.remove_com_velocity:
            com = masked_mean(velocity, residue_mask, dim=1, eps=self.eps)  # [B, 3], invariant under translation
            velocity = (velocity - com.unsqueeze(1)) * mask

        # The same argument, one derivative up. Kabsch removes rotation from
        # the target as exactly as it removes translation, so the three
        # rotation modes about the centroid are as unavailable to the target
        # as the three translations -- and projecting onto a subspace that
        # contains the target can only lower the error. Measured on the run
        # that lacked this, 17.0% of the output energy sat in those modes.
        if self.remove_rigid_motion:
            if coords is None:
                raise ValueError(
                    "VectorFieldDecoder was built with remove_rigid_motion=True but no coords "
                    "were passed; the rotation generators e_k x (x_i - centroid) cannot be "
                    "built without them."
                )
            velocity = project_out_rigid_motion(velocity, coords, residue_mask, eps=self.eps)

        return velocity
