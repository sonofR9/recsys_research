from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Literal

import mup
import torch
from torch import nn

from dcn.models import CrossAttentionRetrievalModel
from dcn.models.history_tokens import EndQuerySlots
from dcn.nn.transformer import CrossAttentionBlock, TransformerBlock
from dcn.training import register_stable_optimizer_groups

from .generation import (
    GenerationExperiment,
    _initialize_mup_parameters,
    _initialize_standard_parameters,
)
from .networks import (
    build_causal_transformer,
    build_transformer_decoder,
    build_transformer_encoder,
)
from .settings import TRANSFORMER, TransformerConfig, transformer_metadata

logger = logging.getLogger(__name__)

QueryArchitecture = Literal["encoder_decoder", "decoder_decoder"]


@dataclass
class CrossAttentionGenerationExperiment(GenerationExperiment):
    window: Literal["next_item", "bounded_prefix"] = "next_item"
    query_architecture: QueryArchitecture = "encoder_decoder"
    retrieval_decoder: TransformerConfig = field(
        default_factory=lambda: replace(
            TRANSFORMER,
            num_layers=1,
            ffn="swiglu",
            ffn_intermediate_dim=128,
            learned_positions="forward",
            attention_window=None,
        )
    )
    query_slots_shared: bool = False
    include_history_memory: bool = False
    num_query_slots: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.query_architecture not in {"encoder_decoder", "decoder_decoder"}:
            raise ValueError(f"unknown query architecture {self.query_architecture!r}")
        if self.transformer.dim != self.retrieval_decoder.dim:
            raise ValueError("history and retrieval decoder widths must match")
        if self.retrieval_decoder.num_layers != 1:
            raise ValueError("retrieval decoder must have exactly one layer")
        if self.retrieval_decoder.ffn != "swiglu":
            raise ValueError("retrieval decoder must use SwiGLU")
        if self.retrieval_decoder.ffn_intermediate_dim % 32:
            raise ValueError("retrieval decoder SwiGLU width must be divisible by 32")
        if self.query_architecture == "encoder_decoder" and (
            self.query_slots_shared or self.include_history_memory
        ):
            raise ValueError("query-slot settings apply only to decoder-decoder")
        if self.num_query_slots != 4:
            raise ValueError("the approved decoder-decoder architecture has four slots")
        if self.bos or self.effective_cls_token_mode != "none":
            raise ValueError("cross-attention architectures own their query tokens")
        if self.effective_per_layer_item_features != "none":
            raise ValueError(
                "cross-attention architectures do not use layer item features"
            )
        if self.window not in {"next_item", "bounded_prefix"}:
            raise ValueError(
                "cross-attention training needs one final target per example"
            )

    def _create_model(self) -> CrossAttentionRetrievalModel:
        tokenizer = self.create_tokenizer()
        if self.query_architecture == "encoder_decoder":
            memory_encoder = build_transformer_encoder(
                self.transformer,
                max_seq_len=self.max_seq_len,
                is_causal=False,
            )
            query_slots = None
        else:
            memory_encoder = build_causal_transformer(
                self.transformer,
                max_seq_len=self.max_seq_len + self.num_query_slots,
            )
            query_slots = EndQuerySlots(
                self.model_dim,
                num_slots=self.num_query_slots,
                shared=self.query_slots_shared,
            )
        model = CrossAttentionRetrievalModel(
            tokenizer=tokenizer,
            memory_encoder=memory_encoder,
            decoder=build_transformer_decoder(
                self.retrieval_decoder,
                max_seq_len=1,
            ),
            item_embedding=self.item_embedding,
            item_id_column=self.item_id_column,
            query_projection=self.create_query_projection(),
            query_slots=query_slots,
            include_history_memory=self.include_history_memory,
        )
        if self.initializer_std is not None:
            _initialize_standard_parameters(model, self.initializer_std)
        return model

    def _training_counts(self) -> tuple[int, int]:
        dataset = self.sequence_train_loader.dataset
        examples = len(dataset)
        first_stage_tokens = dataset.event_count - examples
        if self.query_architecture == "decoder_decoder":
            first_stage_tokens += examples * self.num_query_slots
        return examples, first_stage_tokens + examples

    def generation_architecture_metadata(self) -> dict[str, object]:
        dataset = self.sequence_train_loader.dataset
        return {
            "query_architecture": self.query_architecture,
            "prefix_length_rule": self.prefix_length_rule,
            "prefix_cap": self.prefix_cap,
            "query_slots_shared": self.query_slots_shared,
            "include_history_memory": self.include_history_memory,
            "num_query_slots": self.num_query_slots,
            "retrieval_decoder": transformer_metadata(self.retrieval_decoder),
            "original_users_per_epoch": dataset.original_user_count,
            "expanded_examples_per_epoch": len(dataset),
            "candidate_targets_per_epoch": self.training_targets_per_epoch,
            "ntp_targets_per_epoch": 0,
            "input_tokens_per_epoch": self.training_tokens_per_epoch,
        }

    def training_count_architecture_invariants(self) -> tuple[object, ...]:
        return (
            self.query_architecture,
            self.num_query_slots,
            self.include_history_memory,
        )


@dataclass
class MuTransferCrossAttentionGenerationExperiment(CrossAttentionGenerationExperiment):
    mup_base_dim: int = 16
    mup_delta_dim: int = 32
    mup_base_ffn_dim: int | None = None
    mup_delta_ffn_dim: int | None = None
    item_embedding_dim: int | None = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        for dim in (self.mup_base_dim, self.mup_delta_dim, self.transformer.dim):
            if dim % self.transformer.nhead:
                raise ValueError(
                    f"μP width {dim} must be divisible by {self.transformer.nhead} heads"
                )
        if (self.mup_base_ffn_dim is None) != (self.mup_delta_ffn_dim is None):
            raise ValueError("μP base and delta FFN widths must be set together")

    @staticmethod
    def _scaled_ffn(config: TransformerConfig, dim: int) -> int:
        return round(config.ffn_intermediate_dim / config.dim * dim)

    def _model_at_width(
        self, dim: int, history_ffn_dim: int | None = None
    ) -> CrossAttentionRetrievalModel:
        narrower = replace(
            self,
            initializer_std=None,
            transformer=replace(
                self.transformer,
                dim=dim,
                ffn_intermediate_dim=(
                    history_ffn_dim
                    if history_ffn_dim is not None
                    else self._scaled_ffn(self.transformer, dim)
                ),
            ),
            retrieval_decoder=replace(
                self.retrieval_decoder,
                dim=dim,
                ffn_intermediate_dim=self._scaled_ffn(self.retrieval_decoder, dim),
            ),
        )
        narrower.__dict__["artifacts"] = self.artifacts
        narrower.__dict__["item_embeddings"] = self.item_embeddings
        return narrower._create_model()

    def create_input_projection(self) -> nn.Linear:
        return nn.Linear(self.item_embedding.embedding_dim, self.model_dim, bias=False)

    def create_query_projection(self) -> nn.Linear:
        return mup.MuReadout(
            self.model_dim,
            self.item_embedding.embedding_dim,
            bias=False,
            readout_zero_init=True,
        )

    @cached_property
    def base_model(self) -> CrossAttentionRetrievalModel:
        model = self._model_at_width(
            self.transformer.dim, self.transformer.ffn_intermediate_dim
        ).to(self.runner_build_device)
        base = self._model_at_width(self.mup_base_dim, self.mup_base_ffn_dim)
        delta = self._model_at_width(self.mup_delta_dim, self.mup_delta_ffn_dim)
        mup.set_base_shapes(model, base, delta=delta)
        if self.initializer_std is not None:
            _initialize_mup_parameters(model, self.initializer_std)
        head_dim = self.transformer.dim // self.transformer.nhead
        base_head_dim = self.mup_base_dim // self.transformer.nhead
        softmax_scale = base_head_dim**0.5 / head_dim
        for module in model.modules():
            if isinstance(module, (TransformerBlock, CrossAttentionBlock)):
                module.softmax_scale = softmax_scale
        logger.info("Model architecture:\n%s", model)
        logger.info(
            "Total parameters: %s",
            f"{sum(parameter.numel() for parameter in model.parameters()):,}",
        )
        return model

    def create_optimizers(self) -> torch.optim.Optimizer:
        embedding_params, deep_params = self.split_parameters(
            self.base_model, self.embedding_types
        )
        groups = [
            {
                "params": embedding_params,
                "lr": self.embedding_learning_rate,
                "schedule_group": "embedding",
            },
            {
                "params": deep_params,
                "lr": self.deep_learning_rate,
                "schedule_group": "deep",
            },
        ]
        optimizer = mup.MuAdam(
            [group for group in groups if group["params"]],
            weight_decay=self.weight_decay,
            fused=self.runner_build_device.type == "cuda",
        )
        return register_stable_optimizer_groups(optimizer)
