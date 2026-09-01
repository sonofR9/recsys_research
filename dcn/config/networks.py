"""Architecture construction more than one experiment wants, kept out of the
experiments so a variant reads as configuration rather than as a builder."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch
import torch.nn as nn

from dcn.config.settings import (
    EmbeddingConfig,
    FFNKind,
    PositionOrder,
    TransformerConfig,
)
from dcn.models import MultiHeadNetwork
from dcn.nn import (
    CrossHeadDescription,
    CrossNetwork,
    DcnV2,
    DenseNet,
    GEGLU,
    MultiTaskEmbeddingLayer,
    ReGLU,
    RegularMLP,
    ResNet1D,
    SwiGLU,
)
from dcn.nn.history_encoder import HistoryEncoder
from dcn.nn.types import FFNFactory, ModuleWithDim
from dcn.nn.transformer import (
    BoundedReverseAdditivePositionInput,
    BoundedReverseConcatenatedPositionInput,
    BoundedReverseDenseCorrection,
    BoundedReverseEmbeddingCorrection,
    ConcatenatedLearnedPositionInput,
    CrossAttentionBlock,
    ForwardPositionInput,
    IndexPositions,
    LearnedPositionInput,
    LogValuePositions,
    ReverseRelativePositionInput,
    Rope,
    TransformerBlock,
    TransformerDecoder,
    TransformerEncoder,
    ValuePositions,
    make_norm,
)


def build_dcn_v2_trunk(
    input_dim: int,
    *,
    compression_dim: int = 256,
    output_dim: int = 128,
    dropout: float = 0.1,
) -> DcnV2:
    cross = CrossNetwork(
        input_dim=compression_dim,
        width=128,
        heads_descriptions=[
            CrossHeadDescription(
                num_layers=3,
                compression_activation=nn.GELU,
                decompression_activation=nn.GELU,
                compressed_norm=nn.BatchNorm1d,
                decompressed_norm=None,
            ),
            CrossHeadDescription(
                num_layers=2,
                compression_activation=nn.GELU,
                decompression_activation=None,
                compressed_norm=nn.BatchNorm1d,
                decompressed_norm=None,
            ),
        ],
        global_compressed_norm=None,
        global_decompressed_norm=None,
    )
    return DcnV2(
        cross=cross,
        deep_parallel=ResNet1D(
            input_dim=compression_dim,
            hidden_dims=[compression_dim, output_dim],
            activation_factory=nn.ReLU,
            norm_factory=nn.BatchNorm1d,
            norm_type="pre",
            dropout=dropout,
        ),
        mode="parallel",
        output_dim=output_dim,
        compression=nn.Linear(input_dim, compression_dim),
        deep_sequential=None,
    )


def build_dcn_v2_head(input_dim: int, *, dropout: float = 0.1) -> DcnV2:
    cross = CrossNetwork(
        input_dim=input_dim,
        width=64,
        heads_descriptions=[
            CrossHeadDescription(
                num_layers=2,
                compression_activation=nn.GELU,
                decompression_activation=None,
                compressed_norm=nn.BatchNorm1d,
                decompressed_norm=None,
            )
        ],
        global_compressed_norm=None,
        global_decompressed_norm=None,
    )
    deep_parallel = ResNet1D(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        activation_factory=nn.ReLU,
        norm_factory=nn.BatchNorm1d,
        norm_type="pre",
        dropout=dropout,
    )
    return DcnV2(
        cross=cross,
        deep_parallel=deep_parallel,
        mode="combined",
        output_dim=None,
        compression=None,
        deep_sequential=ResNet1D(
            input_dim=cross.out_dim + deep_parallel.out_dim,
            hidden_dims=[64, 32, 1],
            activation_factory=nn.ReLU,
            norm_factory=nn.BatchNorm1d,
            norm_type="pre",
            dropout=dropout,
        ),
    )


def build_multi_head_dcn(
    *,
    categorical_features: Sequence[str],
    embedding: EmbeddingConfig,
    split_ratios: dict[str, float],
    num_hashes: int,
    task_names: Sequence[str],
    feature_encoders: Sequence[tuple[str, ModuleWithDim]],
    dense_feature_names: Sequence[str],
    num_counters: int,
    dense_encoder: ModuleWithDim | None = None,
    history_encoder_factory: Callable[[int], HistoryEncoder | None] = lambda _: None,
) -> MultiHeadNetwork:
    """Hashed embeddings and side features into a DCNv2 trunk, then a head per task."""
    multi_task_embedding = MultiTaskEmbeddingLayer(
        feature_configs={name: num_hashes for name in categorical_features},
        num_embeddings=embedding.num_embeddings,
        embedding_dim=embedding.dim,
        split_ratios=dict(split_ratios),
        sparse=embedding.sparse,
        mode="sum",
    )
    embedding_dims = multi_task_embedding.out_dim

    token_dim = (
        embedding_dims.shared
        + sum(encoder.out_dim for _, encoder in feature_encoders)
        + (num_counters if dense_encoder is None else dense_encoder.out_dim)
    )
    history_encoder = history_encoder_factory(token_dim)
    trunk = build_dcn_v2_trunk(
        input_dim=token_dim
        + (0 if history_encoder is None else history_encoder.out_dim)
    )
    return MultiHeadNetwork(
        multi_task_embedding=multi_task_embedding,
        shared_network=trunk,
        task_networks={
            name: build_dcn_v2_head(trunk.out_dim + embedding_dims.dims[name])
            for name in task_names
        },
        feature_encoders=list(feature_encoders),
        dense_feature_names=list(dense_feature_names),
        dense_encoder=dense_encoder,
        history_encoder=history_encoder,
    )


_REGULAR_FFNS = {"relu", "gelu", "silu"}
_GATED_FFNS: dict[FFNKind, type[ModuleWithDim]] = {
    "reglu": ReGLU,
    "geglu": GEGLU,
    "swiglu": SwiGLU,
}

_POSITION_INPUTS: dict[PositionOrder, type[LearnedPositionInput]] = {
    "reverse": ReverseRelativePositionInput,
    "forward": ForwardPositionInput,
}

_ROPE_POSITIONS = {
    "reverse": IndexPositions(reverse=True),
    "forward": IndexPositions(),
    "timestamp": ValuePositions(scale=1 / 86_400),
    "timestamp_reverse": ValuePositions(scale=1 / 86_400, reverse=True),
    "timestamp_log": LogValuePositions(),
    "timestamp_log_reverse": LogValuePositions(reverse=True),
}

_Module = TypeVar("_Module", bound=nn.Module)


def _without_advancing_initializer_rng(factory: Callable[[], _Module]) -> _Module:
    with torch.random.fork_rng(devices=[]):
        return factory()


def _ffn_factory(transformer: TransformerConfig) -> FFNFactory:
    if transformer.ffn in _REGULAR_FFNS:
        return lambda dim: RegularMLP(
            dim,
            transformer.ffn_intermediate_dim,
            transformer.ffn_dropout,
            transformer.ffn,
        )
    ffn = _GATED_FFNS[transformer.ffn]
    dropout = transformer.ffn_dropout if transformer.gated_ffn_dropout else 0.0
    return lambda dim: ffn(dim, transformer.ffn_intermediate_dim, dropout)


def _transformer_blocks(
    transformer: TransformerConfig, *, is_causal: bool = True
) -> list[TransformerBlock]:
    ffn_factory = _ffn_factory(transformer)
    head_dim = transformer.dim // transformer.nhead
    return [
        TransformerBlock(
            dim=transformer.dim,
            nhead=transformer.nhead,
            num_kv_heads=transformer.num_kv_heads,
            ffn_factory=ffn_factory,
            dropout=transformer.dropout,
            use_alibi=transformer.alibi,
            rope=(
                None
                if transformer.rope is None
                else Rope(
                    head_dim,
                    _ROPE_POSITIONS[transformer.rope],
                    base=transformer.rope_base,
                )
            ),
            norm=transformer.norm,
            norm_place=transformer.norm_place,
            attention_window=transformer.attention_window,
            is_causal=is_causal,
        )
        for _ in range(transformer.num_layers)
    ]


def build_causal_transformer(
    transformer: TransformerConfig, *, max_seq_len: int
) -> TransformerEncoder:
    """SASRec-shaped stack: causal, and positioned however the config says."""
    return build_transformer_encoder(
        transformer, max_seq_len=max_seq_len, is_causal=True
    )


def build_transformer_encoder(
    transformer: TransformerConfig,
    *,
    max_seq_len: int,
    is_causal: bool = True,
) -> TransformerEncoder:
    dim = transformer.dim
    learned = transformer.learned_positions
    learned_orders = (
        () if learned is None else learned if isinstance(learned, tuple) else (learned,)
    )
    if transformer.learned_position_reverse_correction == "bounded_tanh":
        forward_input = ForwardPositionInput(dim, max_seq_len)
        if transformer.learned_position_fusion == "add":
            forward_control: ForwardPositionInput | ConcatenatedLearnedPositionInput = (
                forward_input
            )
        else:
            forward_control = ConcatenatedLearnedPositionInput(
                [forward_input],
                DenseNet(dim * 2, dim, preserve_input_rms=True),
                rezero=True,
            )
        if transformer.learned_position_fusion == "add":
            reverse_correction = _without_advancing_initializer_rng(
                lambda: BoundedReverseEmbeddingCorrection(
                    ReverseRelativePositionInput(dim, max_seq_len),
                    transformer.learned_position_reverse_max_scale,
                    transformer.learned_position_reverse_initializer_rng_nonadvancing,
                )
            )
            position_inputs = [
                BoundedReverseAdditivePositionInput(
                    forward_control,
                    reverse_correction,
                )
            ]
        else:
            reverse_correction = _without_advancing_initializer_rng(
                lambda: BoundedReverseDenseCorrection(
                    ReverseRelativePositionInput(dim, max_seq_len),
                    DenseNet(dim * 3, dim, preserve_input_rms=True),
                    transformer.learned_position_reverse_max_scale,
                    transformer.learned_position_reverse_initializer_rng_nonadvancing,
                )
            )
            position_inputs = [
                BoundedReverseConcatenatedPositionInput(
                    forward_control,
                    reverse_correction,
                )
            ]
    else:
        learned_inputs = [
            _POSITION_INPUTS[order](
                dim,
                max_seq_len,
                zero_init=(
                    transformer.learned_position_initialization == "zero_reverse"
                    and order == "reverse"
                ),
            )
            for order in learned_orders
        ]
        position_inputs = learned_inputs
    if (
        transformer.learned_position_reverse_correction is None
        and transformer.learned_position_fusion == "concat"
    ):
        position_inputs = [
            ConcatenatedLearnedPositionInput(
                learned_inputs,
                DenseNet(
                    dim * (1 + len(learned_inputs)),
                    dim,
                    preserve_input_rms=(
                        transformer.learned_position_fusion_normalization == "input_rms"
                        or transformer.learned_position_fusion_residual == "rezero"
                    ),
                ),
                rezero=transformer.learned_position_fusion_residual == "rezero",
            )
        ]
    return TransformerEncoder(
        blocks=_transformer_blocks(transformer, is_causal=is_causal),
        final_norm=make_norm(dim, transformer.final_norm),
        input_norm=make_norm(dim, transformer.input_norm),
        position_inputs=position_inputs,
        position_dropout=transformer.input_dropout,
        max_seqlen=max_seq_len,
    )


def build_transformer_decoder(
    transformer: TransformerConfig, *, max_seq_len: int
) -> TransformerDecoder:
    """The same stack with a cross-attention layer to a memory after each block."""
    dim = transformer.dim
    return TransformerDecoder(
        self_attention_blocks=_transformer_blocks(transformer),
        cross_attention_blocks=[
            CrossAttentionBlock(
                dim=dim,
                nhead=transformer.nhead,
                num_kv_heads=transformer.num_kv_heads,
                ffn_factory=_ffn_factory(transformer),
                dropout=transformer.dropout,
                norm=transformer.norm,
                norm_place=transformer.norm_place,
            )
            for _ in range(transformer.num_layers)
        ],
        final_norm=make_norm(dim, transformer.final_norm),
        input_norm=make_norm(dim, transformer.input_norm),
        position_inputs=[ForwardPositionInput(dim, max_seq_len)],
    )
