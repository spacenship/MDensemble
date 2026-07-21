import torch

from protein_flow.config import Config
from protein_flow.models.dual_graph_flow import DualGraphFlowModel


def _small_config() -> Config:
    config = Config()
    config.data.plm_dim = 24
    config.data.num_amino_acid_types = 22
    config.model.sequence_encoder.hidden_dim = 16
    config.model.geometric_encoder.hidden_dim = 16
    config.model.fusion.hidden_dim = 16
    config.model.fusion.condition_dim = 8
    config.model.graph.knn_k = 4
    config.model.graph.num_rbf = 8
    return config


def _random_batch(config: Config, batch=3, length=10, generator=None):
    seq_emb = torch.randn(batch, length, config.data.plm_dim, generator=generator)
    residue_types = torch.randint(0, config.data.num_amino_acid_types, (batch, length), generator=generator)
    coords = torch.randn(batch, length, 3, generator=generator) * 3.0
    mask = torch.ones(batch, length, dtype=torch.bool)
    temperature = torch.rand(batch, 1, generator=generator)
    delta_t = torch.rand(batch, 1, generator=generator)
    tau = torch.rand(batch, generator=generator)
    return seq_emb, residue_types, coords, mask, temperature, delta_t, tau


def test_forward_shape():
    torch.manual_seed(0)
    config = _small_config()
    model = DualGraphFlowModel(config)
    seq_emb, residue_types, coords, mask, temperature, delta_t, tau = _random_batch(config)

    velocity = model(coords, tau, seq_emb, residue_types, temperature, delta_t, mask)
    assert velocity.shape == coords.shape


def test_padding_produces_zero_velocity():
    torch.manual_seed(1)
    config = _small_config()
    model = DualGraphFlowModel(config)
    batch, length = 2, 12
    seq_emb, residue_types, coords, mask, temperature, delta_t, tau = _random_batch(config, batch, length)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[:, :7] = True

    velocity = model(coords, tau, seq_emb, residue_types, temperature, delta_t, mask)
    assert torch.all(velocity[:, 7:] == 0.0)


def test_gradients_flow_through_all_submodules():
    torch.manual_seed(2)
    config = _small_config()
    model = DualGraphFlowModel(config)
    seq_emb, residue_types, coords, mask, temperature, delta_t, tau = _random_batch(config)
    coords.requires_grad_(True)

    velocity = model(coords, tau, seq_emb, residue_types, temperature, delta_t, mask)
    # NOTE: plain .sum() would be identically zero here because
    # remove_com_velocity forces the per-batch sum to vanish by
    # construction, which trivially zeroes its own gradient. Use a
    # loss that isn't structurally constant instead.
    velocity.pow(2).sum().backward()

    assert coords.grad is not None and torch.any(coords.grad != 0.0)
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"


def test_different_tau_gives_different_velocity():
    torch.manual_seed(3)
    config = _small_config()
    model = DualGraphFlowModel(config)
    model.eval()
    seq_emb, residue_types, coords, mask, temperature, delta_t, _ = _random_batch(config)

    with torch.no_grad():
        v0 = model(coords, torch.zeros(3), seq_emb, residue_types, temperature, delta_t, mask)
        v1 = model(coords, torch.ones(3), seq_emb, residue_types, temperature, delta_t, mask)
    assert not torch.allclose(v0, v1)
