import torch

from protein_flow.models.geometric_encoder import GeometricEncoder
from protein_flow.models.vector_field import VectorFieldDecoder


def _random_rotation(generator: torch.Generator) -> torch.Tensor:
    a = torch.randn(3, 3, generator=generator)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diagonal(r))
    q = q * d.unsqueeze(-2)
    if torch.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


def _make_encoder(hidden_dim=16, num_layers=2, update_coordinates=False):
    return GeometricEncoder(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
        update_coordinates=update_coordinates,
        knn_k=4,
        num_rbf=8,
        rbf_min_dist=0.0,
        rbf_max_dist=20.0,
    )


def test_output_shapes_and_padding():
    torch.manual_seed(0)
    encoder = _make_encoder()
    batch, length, hidden = 3, 10, 16
    coords = torch.randn(batch, length, 3)
    node_features = torch.randn(batch, length, hidden)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[:, :6] = True

    h_out, graph = encoder(coords, node_features, mask)
    assert h_out.shape == (batch, length, hidden)
    assert torch.all(h_out[:, 6:] == 0.0)


def test_hidden_features_are_rotation_translation_invariant():
    torch.manual_seed(1)
    gen = torch.Generator().manual_seed(1)
    encoder = _make_encoder()
    batch, length, hidden = 2, 9, 16
    # Well-separated coordinates to avoid kNN ties under rotation.
    coords = torch.randn(batch, length, 3, generator=gen) * 3.0
    node_features = torch.randn(batch, length, hidden, generator=gen)
    mask = torch.ones(batch, length, dtype=torch.bool)

    rotation = _random_rotation(gen)
    translation = torch.randn(3, generator=gen)
    coords_transformed = coords @ rotation + translation

    h_out, _ = encoder(coords, node_features, mask)
    h_out_t, _ = encoder(coords_transformed, node_features, mask)

    torch.testing.assert_close(h_out, h_out_t, atol=1e-4, rtol=1e-4)


def test_vector_field_decoder_is_rotation_equivariant_and_translation_invariant():
    torch.manual_seed(2)
    gen = torch.Generator().manual_seed(2)
    hidden = 16
    encoder = _make_encoder(hidden_dim=hidden)
    decoder = VectorFieldDecoder(hidden_dim=hidden, num_rbf=8, remove_com_velocity=True)

    batch, length = 2, 9
    coords = torch.randn(batch, length, 3, generator=gen) * 3.0
    node_features = torch.randn(batch, length, hidden, generator=gen)
    mask = torch.ones(batch, length, dtype=torch.bool)

    rotation = _random_rotation(gen)
    translation = torch.randn(3, generator=gen)
    coords_rotated = coords @ rotation
    coords_translated = coords + translation

    h_out, graph = encoder(coords, node_features, mask)
    velocity = decoder(h_out, graph, mask)

    h_out_r, graph_r = encoder(coords_rotated, node_features, mask)
    velocity_r = decoder(h_out_r, graph_r, mask)

    h_out_t, graph_t = encoder(coords_translated, node_features, mask)
    velocity_t = decoder(h_out_t, graph_t, mask)

    expected_rotated = velocity @ rotation
    torch.testing.assert_close(velocity_r, expected_rotated, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(velocity_t, velocity, atol=1e-3, rtol=1e-3)


def test_com_removal_zeroes_masked_mean_velocity():
    torch.manual_seed(3)
    hidden = 16
    encoder = _make_encoder(hidden_dim=hidden)
    decoder = VectorFieldDecoder(hidden_dim=hidden, num_rbf=8, remove_com_velocity=True)

    batch, length = 2, 8
    coords = torch.randn(batch, length, 3) * 3.0
    node_features = torch.randn(batch, length, hidden)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[:, :5] = True

    h_out, graph = encoder(coords, node_features, mask)
    velocity = decoder(h_out, graph, mask)

    com = (velocity * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    torch.testing.assert_close(com, torch.zeros_like(com), atol=1e-5, rtol=1e-5)


def test_update_coordinates_option_runs_and_stays_invariant():
    torch.manual_seed(4)
    gen = torch.Generator().manual_seed(4)
    encoder = _make_encoder(update_coordinates=True)
    batch, length, hidden = 2, 8, 16
    coords = torch.randn(batch, length, 3, generator=gen) * 3.0
    node_features = torch.randn(batch, length, hidden, generator=gen)
    mask = torch.ones(batch, length, dtype=torch.bool)

    rotation = _random_rotation(gen)
    h_out, _ = encoder(coords, node_features, mask)
    h_out_r, _ = encoder(coords @ rotation, node_features, mask)
    torch.testing.assert_close(h_out, h_out_r, atol=1e-4, rtol=1e-4)


def _checkpointing_pair(update_coordinates=False):
    """Two encoders that differ only in whether they recompute activations."""
    torch.manual_seed(7)
    plain = _make_encoder(update_coordinates=update_coordinates)
    torch.manual_seed(7)
    checkpointed = _make_encoder(update_coordinates=update_coordinates)
    checkpointed.gradient_checkpointing = True
    checkpointed.load_state_dict(plain.state_dict())
    return plain.train(), checkpointed.train()


def test_gradient_checkpointing_matches_the_plain_forward_and_backward():
    """Recomputing activations must be a memory trade, not a numerical one."""
    gen = torch.Generator().manual_seed(7)
    batch, length, hidden = 2, 9, 16
    coords = torch.randn(batch, length, 3, generator=gen) * 3.0
    node_features = torch.randn(batch, length, hidden, generator=gen)
    mask = torch.ones(batch, length, dtype=torch.bool)

    plain, checkpointed = _checkpointing_pair()
    outputs = []
    for encoder in (plain, checkpointed):
        features = node_features.clone().requires_grad_(True)
        h_out, _ = encoder(coords, features, mask)
        h_out.square().sum().backward()
        outputs.append((h_out.detach(), features.grad.clone()))

    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=1e-5, rtol=1e-5)


def test_gradient_checkpointing_is_inert_in_eval_mode():
    """Inference has no backward pass to recompute for, so it must not engage."""
    gen = torch.Generator().manual_seed(8)
    coords = torch.randn(2, 7, 3, generator=gen) * 3.0
    node_features = torch.randn(2, 7, 16, generator=gen)
    mask = torch.ones(2, 7, dtype=torch.bool)

    plain, checkpointed = _checkpointing_pair()
    with torch.no_grad():
        expected, _ = plain.eval()(coords, node_features, mask)
        actual, _ = checkpointed.eval()(coords, node_features, mask)
    torch.testing.assert_close(expected, actual)
