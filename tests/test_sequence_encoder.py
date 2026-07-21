import torch

from protein_flow.models.sequence_encoder import SequenceEncoder


def _make_encoder(plm_dim=32, hidden_dim=16, num_layers=3):
    return SequenceEncoder(
        plm_dim=plm_dim,
        num_amino_acid_types=22,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
        use_position_encoding=True,
    )


def test_output_shape():
    torch.manual_seed(0)
    encoder = _make_encoder()
    batch, length, plm_dim = 4, 12, 32
    seq_emb = torch.randn(batch, length, plm_dim)
    residue_types = torch.randint(0, 22, (batch, length))
    mask = torch.ones(batch, length, dtype=torch.bool)

    out = encoder(seq_emb, residue_types, mask)
    assert out.shape == (batch, length, 16)


def test_padding_positions_are_zeroed():
    torch.manual_seed(1)
    encoder = _make_encoder()
    batch, length, plm_dim = 2, 10, 32
    seq_emb = torch.randn(batch, length, plm_dim)
    residue_types = torch.randint(0, 22, (batch, length))
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[:, :6] = True

    out = encoder(seq_emb, residue_types, mask)
    assert torch.all(out[:, 6:] == 0.0)
    assert not torch.all(out[:, :6] == 0.0)


def test_gradients_flow_to_input_projection():
    torch.manual_seed(2)
    encoder = _make_encoder()
    batch, length, plm_dim = 2, 8, 32
    seq_emb = torch.randn(batch, length, plm_dim, requires_grad=True)
    residue_types = torch.randint(0, 22, (batch, length))
    mask = torch.ones(batch, length, dtype=torch.bool)

    out = encoder(seq_emb, residue_types, mask)
    out.sum().backward()
    assert seq_emb.grad is not None
    assert torch.any(seq_emb.grad != 0.0)
    for param in encoder.parameters():
        assert param.grad is not None


def test_message_passing_propagates_information_along_chain():
    """Perturbing residue 0's embedding should influence downstream
    residues' representations after >=2 message-passing layers, since the
    forward peptide edge chains information along the sequence."""
    torch.manual_seed(3)
    encoder = _make_encoder(num_layers=3)
    batch, length, plm_dim = 1, 6, 32
    seq_emb = torch.randn(batch, length, plm_dim)
    residue_types = torch.randint(0, 22, (batch, length))
    mask = torch.ones(batch, length, dtype=torch.bool)

    out_base = encoder(seq_emb, residue_types, mask)

    seq_emb_perturbed = seq_emb.clone()
    seq_emb_perturbed[0, 0] += 10.0
    out_perturbed = encoder(seq_emb_perturbed, residue_types, mask)

    # residue 2 is 2 hops away via forward edges; with 3 layers info should reach it.
    diff = (out_perturbed[0, 2] - out_base[0, 2]).abs().sum()
    assert diff > 1e-4


def test_rejects_out_of_range_num_layers():
    import pytest

    with pytest.raises(ValueError):
        SequenceEncoder(plm_dim=32, num_amino_acid_types=22, num_layers=1)
    with pytest.raises(ValueError):
        SequenceEncoder(plm_dim=32, num_amino_acid_types=22, num_layers=5)
