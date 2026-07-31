"""In-graph ESM2 encoder for end-to-end fine-tuning.

This is deliberately separate from
:mod:`protein_flow.data.esm_adapter`, which only *precomputes* frozen
embeddings offline. Here ESM2 lives inside the model, so its weights
receive gradient from the flow-matching objective and are updated together
with the rest of the network.

Verified token layout for ``facebook/esm2_*`` tokenizers: a sequence of L
residues becomes ``[<cls>, r_1, ..., r_L, <eos>]``, so residue ``j`` sits at
token index ``j + 1``. The encoder slices that offset off rather than
assuming the hidden states are already residue-aligned.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class EsmSequenceEncoder(nn.Module):
    """Wraps a HuggingFace ESM2 model to emit residue-aligned embeddings.

    Args:
        model_name: HuggingFace model id (e.g. ``facebook/esm2_t30_150M_UR50D``).
        trainable: if False, parameters are frozen and the forward pass runs
            under ``torch.no_grad()`` (equivalent to precomputed embeddings,
            but without needing a cache on disk).
        gradient_checkpointing: trade compute for activation memory inside
            the transformer stack. Worth enabling for the larger checkpoints.
        num_frozen_layers: freeze the embedding table plus this many of the
            lowest transformer layers. Fine-tuning only the upper layers is a
            common way to keep a large PLM stable on a small downstream set.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t30_150M_UR50D",
        trainable: bool = True,
        gradient_checkpointing: bool = False,
        num_frozen_layers: int = 0,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "In-graph ESM fine-tuning requires `transformers`: pip install transformers"
            ) from error

        self.esm = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.trainable = trainable
        self.output_dim = int(self.esm.config.hidden_size)

        if gradient_checkpointing:
            # use_reentrant=False is required for DDP: the reentrant
            # implementation re-triggers autograd hooks during recomputation,
            # which makes DDP raise "Expected to mark a variable ready only
            # once". Older transformers releases ignore the kwarg, so fall
            # back rather than failing outright.
            try:
                self.esm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:  # pragma: no cover - depends on transformers version
                self.esm.gradient_checkpointing_enable()

        # ESM2 ships an auxiliary contact-prediction head that this encoder
        # never calls. Left trainable it would sit in the optimizer's
        # parameter group forever receiving no gradient, so freeze it.
        if hasattr(self.esm, "contact_head"):
            for parameter in self.esm.contact_head.parameters():
                parameter.requires_grad_(False)

        if not trainable:
            for parameter in self.esm.parameters():
                parameter.requires_grad_(False)
        elif num_frozen_layers > 0:
            for parameter in self.esm.embeddings.parameters():
                parameter.requires_grad_(False)
            for layer in self.esm.encoder.layer[:num_frozen_layers]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, num_residues: int) -> Tensor:
        """
        Args:
            input_ids: [B, T] token ids including <cls>/<eos>/padding.
            attention_mask: [B, T] 1 for real tokens.
            num_residues: L, the residue-axis length the caller expects.

        Returns:
            [B, L, hidden_size] residue-aligned embeddings.
        """
        if self.trainable:
            hidden_states = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            with torch.no_grad():
                hidden_states = self.esm(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
        # Drop the leading <cls>; positions past a sample's own length are
        # <eos>/padding and get masked out downstream.
        return hidden_states[:, 1 : 1 + num_residues]
