"""In-graph ESM fine-tuning wiring.

The real ESM2-150M checkpoint is a ~600MB download, so these tests use a
tiny randomly-initialised ESM config built locally: the point is to verify
*wiring* (token alignment, gradient flow into the PLM, separate learning
rates, batching), not the pretrained weights. Tests needing `transformers`
skip cleanly when it is absent.
"""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from protein_flow.config import Config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.models.dual_graph_flow import DualGraphFlowModel
from protein_flow.models.esm_encoder import EsmSequenceEncoder
from protein_flow.train import _build_parameter_groups

TINY_HIDDEN = 32


@pytest.fixture(scope="module")
def tiny_esm_path(tmp_path_factory):
    """A tiny, randomly-initialised ESM2 saved to disk -- no network needed."""
    from transformers import EsmConfig, EsmModel

    # Token ids mirror the real esm2_* tokenizers: <cls>=0, <pad>=1, <eos>=2.
    # pad_token_id must be set or EsmEmbeddings' padding_idx stays None.
    config = transformers.EsmConfig(
        vocab_size=33, hidden_size=TINY_HIDDEN, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=1026, position_embedding_type="rotary",
        pad_token_id=1, bos_token_id=0, eos_token_id=2, mask_token_id=32,
    )
    model = EsmModel(config, add_pooling_layer=False)
    path = tmp_path_factory.mktemp("tiny_esm")
    model.save_pretrained(path)
    from transformers import AutoTokenizer

    return str(path)


def test_esm_encoder_strips_cls_and_returns_residue_aligned_output(tiny_esm_path):
    encoder = EsmSequenceEncoder(tiny_esm_path, trainable=True, gradient_checkpointing=False)
    batch_size, num_residues = 2, 7
    # [<cls>, r_1..r_L, <eos>]
    input_ids = torch.randint(4, 30, (batch_size, num_residues + 2))
    input_ids[:, 0] = 0
    input_ids[:, -1] = 2
    attention_mask = torch.ones_like(input_ids)

    out = encoder(input_ids, attention_mask, num_residues=num_residues)
    assert out.shape == (batch_size, num_residues, TINY_HIDDEN)


def test_esm_contact_head_is_frozen(tiny_esm_path):
    encoder = EsmSequenceEncoder(tiny_esm_path, trainable=True, gradient_checkpointing=False)
    if hasattr(encoder.esm, "contact_head"):
        assert all(not p.requires_grad for p in encoder.esm.contact_head.parameters())


def test_frozen_esm_has_no_trainable_parameters(tiny_esm_path):
    encoder = EsmSequenceEncoder(tiny_esm_path, trainable=False, gradient_checkpointing=False)
    assert all(not p.requires_grad for p in encoder.parameters())


def test_num_frozen_layers_freezes_only_the_lower_stack(tiny_esm_path):
    encoder = EsmSequenceEncoder(
        tiny_esm_path, trainable=True, gradient_checkpointing=False, num_frozen_layers=1
    )
    assert all(not p.requires_grad for p in encoder.esm.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in encoder.esm.encoder.layer[1].parameters())
    assert all(not p.requires_grad for p in encoder.esm.embeddings.parameters())


def _esm_config(tiny_esm_path) -> Config:
    config = Config()
    config.data.plm_dim = TINY_HIDDEN
    config.model.esm.enabled = True
    config.model.esm.model_name = tiny_esm_path
    config.model.esm.gradient_checkpointing = False
    config.model.esm.learning_rate = 1e-5
    for module in (
        config.model.sequence_encoder, config.model.geometric_encoder,
        config.model.fusion, config.model.decoder,
    ):
        module.hidden_dim = 16
    config.model.fusion.condition_dim = 8
    config.model.sequence_encoder.num_layers = 2
    config.model.geometric_encoder.num_layers = 2
    config.model.graph.knn_k = 4
    config.model.graph.num_rbf = 4
    config.train.optim.lr = 3e-4
    return config


def _ca_batch_with_tokens(num_residues=6, batch_size=2, plm_dim=TINY_HIDDEN):
    generator = torch.Generator().manual_seed(0)
    samples = []
    for _ in range(batch_size):
        token_ids = torch.randint(4, 30, (num_residues + 2,))
        token_ids[0], token_ids[-1] = 0, 2
        samples.append({
            "sequence_embedding": torch.randn(num_residues, plm_dim, generator=generator),
            "source_coords": torch.randn(num_residues, 3, generator=generator) * 3.0,
            "target_coords": torch.randn(num_residues, 3, generator=generator) * 3.0,
            "residue_types": torch.randint(0, 20, (num_residues,), generator=generator),
            "temperature": torch.tensor([320.0]),
            "physical_delta_t": torch.tensor([5.0]),
            "esm_input_ids": token_ids,
        })
    return collate_protein_batch(samples)


def test_collate_pads_tokens_and_builds_attention_mask():
    generator = torch.Generator().manual_seed(1)
    samples = []
    for num_residues in (4, 7):
        token_ids = torch.randint(4, 30, (num_residues + 2,))
        token_ids[0], token_ids[-1] = 0, 2
        samples.append({
            "sequence_embedding": torch.randn(num_residues, TINY_HIDDEN, generator=generator),
            "source_coords": torch.randn(num_residues, 3, generator=generator),
            "target_coords": torch.randn(num_residues, 3, generator=generator),
            "residue_types": torch.randint(0, 20, (num_residues,), generator=generator),
            "temperature": torch.tensor([320.0]),
            "physical_delta_t": torch.tensor([5.0]),
            "esm_input_ids": token_ids,
        })
    batch = collate_protein_batch(samples)

    assert batch["esm_input_ids"].shape == (2, 9)  # 7 residues + cls + eos
    assert batch["esm_attention_mask"][0].sum() == 6  # 4 residues + cls + eos
    assert batch["esm_attention_mask"][1].sum() == 9
    # Token axis is L+2, distinct from the residue axis.
    assert batch["esm_input_ids"].shape[1] == batch["residue_mask"].shape[1] + 2


def test_gradient_reaches_esm_weights(tiny_esm_path):
    config = _esm_config(tiny_esm_path)
    torch.manual_seed(0)
    model = DualGraphFlowModel(config)
    batch = _ca_batch_with_tokens()

    velocity = model(
        batch["source_coords"], torch.rand(2), batch["sequence_embedding"], batch["residue_types"],
        batch["temperature"], batch["physical_delta_t"], batch["residue_mask"],
        esm_input_ids=batch["esm_input_ids"], esm_attention_mask=batch["esm_attention_mask"],
    )
    velocity.pow(2).sum().backward()

    esm_grads = [
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if n.startswith("esm_encoder.") and p.requires_grad and p.grad is not None
    ]
    assert esm_grads, "no ESM parameter received gradient"
    assert sum(esm_grads) > 0.0


def test_precomputed_embedding_is_ignored_when_esm_is_in_graph(tiny_esm_path):
    """With in-graph ESM the model must derive embeddings from token ids, so
    corrupting the precomputed `sequence_embedding` must change nothing."""
    config = _esm_config(tiny_esm_path)
    torch.manual_seed(0)
    model = DualGraphFlowModel(config).eval()
    batch = _ca_batch_with_tokens()
    tau = torch.rand(2, generator=torch.Generator().manual_seed(5))

    def run(sequence_embedding):
        with torch.no_grad():
            return model(
                batch["source_coords"], tau, sequence_embedding, batch["residue_types"],
                batch["temperature"], batch["physical_delta_t"], batch["residue_mask"],
                esm_input_ids=batch["esm_input_ids"], esm_attention_mask=batch["esm_attention_mask"],
            )

    base = run(batch["sequence_embedding"])
    corrupted = run(torch.randn_like(batch["sequence_embedding"]) * 100.0)
    torch.testing.assert_close(base, corrupted)


def test_missing_tokens_raise_a_clear_error(tiny_esm_path):
    config = _esm_config(tiny_esm_path)
    torch.manual_seed(0)
    model = DualGraphFlowModel(config)
    batch = _ca_batch_with_tokens()
    with pytest.raises(ValueError, match="esm_input_ids"):
        model(
            batch["source_coords"], torch.rand(2), batch["sequence_embedding"], batch["residue_types"],
            batch["temperature"], batch["physical_delta_t"], batch["residue_mask"],
        )


def test_plm_dim_mismatch_is_rejected(tiny_esm_path):
    config = _esm_config(tiny_esm_path)
    config.data.plm_dim = TINY_HIDDEN + 1
    with pytest.raises(ValueError, match="plm_dim"):
        DualGraphFlowModel(config)


def test_parameter_groups_give_esm_its_own_learning_rate(tiny_esm_path):
    config = _esm_config(tiny_esm_path)
    torch.manual_seed(0)
    model = DualGraphFlowModel(config)
    groups = _build_parameter_groups(model, config)

    assert len(groups) == 2
    learning_rates = sorted(group["lr"] for group in groups)
    assert learning_rates == [config.model.esm.learning_rate, config.train.optim.lr]
    # Every trainable parameter lands in exactly one group.
    grouped = sum(len(group["params"]) for group in groups)
    assert grouped == sum(1 for p in model.parameters() if p.requires_grad)


def test_single_parameter_group_without_esm():
    config = Config()
    config.data.plm_dim = 16
    for module in (
        config.model.sequence_encoder, config.model.geometric_encoder,
        config.model.fusion, config.model.decoder,
    ):
        module.hidden_dim = 16
    config.model.fusion.condition_dim = 8
    model = DualGraphFlowModel(config)
    assert len(_build_parameter_groups(model, config)) == 1
