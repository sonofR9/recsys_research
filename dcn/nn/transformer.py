import math
import warnings
from abc import abstractmethod
from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from dcn.data.packed import ragged_positions
from utils.global_config import config

from .types import FFNFactory, ModuleWithDim
from .layer_item_features import LayerItemFeatureFusion

if torch.cuda.is_available():
    from flash_attn import flash_attn_varlen_func

_cpu_attention_warned = False
_FLASH_DTYPES = (torch.float16, torch.bfloat16)


def _assert_flash_dtype(dtype: torch.dtype) -> None:
    assert dtype in _FLASH_DTYPES, (
        f"flash attention supports {_FLASH_DTYPES}, got {dtype}: set"
        " runtime.dtype to bfloat16 so AutoCast covers the transformer,"
        " or enable cpu attention"
    )


def _warn_cpu_attention_once() -> None:
    global _cpu_attention_warned
    if not _cpu_attention_warned:
        warnings.warn(
            "Using CPU attention fallback: ignores ALiBi slopes and dropout."
            " Intended for integration tests only.",
            stacklevel=3,
        )
        _cpu_attention_warned = True


def _repeat_kv_heads(
    k: torch.Tensor, v: torch.Tensor, nhead: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_kv_heads = k.shape[1]
    if num_kv_heads == nhead:
        return k, v
    repeat = nhead // num_kv_heads
    return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)


def _max_seqlen(cumulative_lens: torch.Tensor) -> int:
    """Reads a device tensor, so it drains the CUDA queue. The encoder resolves
    it once and hands it to every block rather than paying that per layer."""
    return int(cumulative_lens.diff().max())


def _cpu_varlen_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_lens_q: torch.Tensor,
    cumulative_lens_kv: torch.Tensor,
    is_causal: bool,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    _warn_cpu_attention_once()
    k, v = _repeat_kv_heads(k, v, q.shape[1])

    output = torch.empty_like(q)
    for sequence_index in range(cumulative_lens_q.shape[0] - 1):
        q_start = int(cumulative_lens_q[sequence_index].item())
        q_end = int(cumulative_lens_q[sequence_index + 1].item())
        kv_start = int(cumulative_lens_kv[sequence_index].item())
        kv_end = int(cumulative_lens_kv[sequence_index + 1].item())
        sequence_q = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        sequence_k = k[kv_start:kv_end].transpose(0, 1).unsqueeze(0)
        sequence_v = v[kv_start:kv_end].transpose(0, 1).unsqueeze(0)
        attention = F.scaled_dot_product_attention(
            sequence_q,
            sequence_k,
            sequence_v,
            is_causal=is_causal,
            scale=softmax_scale,
        )
        output[q_start:q_end] = attention.squeeze(0).transpose(0, 1)
    return output


def _cpu_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_lens: torch.Tensor,
    attention_window: int | None = None,
    softmax_scale: float | None = None,
    is_causal: bool = True,
) -> torch.Tensor:
    if attention_window is not None:
        _warn_cpu_attention_once()
        k, v = _repeat_kv_heads(k, v, q.shape[1])
        output = torch.empty_like(q)
        for sequence_index in range(cumulative_lens.shape[0] - 1):
            start = int(cumulative_lens[sequence_index].item())
            end = int(cumulative_lens[sequence_index + 1].item())
            sequence_q = q[start:end].transpose(0, 1).unsqueeze(0)
            sequence_k = k[start:end].transpose(0, 1).unsqueeze(0)
            sequence_v = v[start:end].transpose(0, 1).unsqueeze(0)
            positions = torch.arange(end - start, device=q.device)
            distance = positions[:, None] - positions[None, :]
            allowed = (
                (distance >= 0) & (distance < attention_window)
                if is_causal
                else distance.abs() < attention_window
            )
            attention = F.scaled_dot_product_attention(
                sequence_q,
                sequence_k,
                sequence_v,
                attn_mask=allowed,
                scale=softmax_scale,
            )
            output[start:end] = attention.squeeze(0).transpose(0, 1)
        return output
    return _cpu_varlen_cross_attention(
        q,
        k,
        v,
        cumulative_lens,
        cumulative_lens,
        is_causal=is_causal,
        softmax_scale=softmax_scale,
    )


class RopePositions(nn.Module):
    """Per-token positions fed to :class:`Rope`. Positions need not be integers."""

    @abstractmethod
    def forward(
        self,
        total_tokens: int,
        cumulative_lens: torch.Tensor,
        values: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class IndexPositions(RopePositions):
    def __init__(self, reverse: bool = False):
        super().__init__()
        self.reverse = reverse

    def forward(
        self,
        total_tokens: int,
        cumulative_lens: torch.Tensor,
        values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lengths = cumulative_lens.diff()
        sequences, ranks = ragged_positions(lengths, total_tokens)
        if self.reverse:
            return (lengths[sequences] - 1 - ranks).float()
        return ranks.float()


class ValuePositions(RopePositions):
    """Continuous positions derived from a per-token value, e.g. a timestamp."""

    def __init__(self, scale: float = 1.0, reverse: bool = False):
        super().__init__()
        self.scale = scale
        self.reverse = reverse

    def transform_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        return offsets

    def forward(
        self,
        total_tokens: int,
        cumulative_lens: torch.Tensor,
        values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert values is not None, "ValuePositions needs a per-token value tensor"
        assert (
            values.shape[0] == total_tokens
        ), "value tensor must be one entry per token"

        lengths = cumulative_lens.diff()
        anchor_index = cumulative_lens[1:] - 1 if self.reverse else cumulative_lens[:-1]
        anchors = values[anchor_index.long()].repeat_interleave(lengths)
        offsets = (anchors - values if self.reverse else values - anchors).float()

        return self.transform_offsets(offsets) * self.scale


class LogValuePositions(ValuePositions):
    def transform_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        return torch.log1p(offsets.clamp(min=0.0))


class Rope(nn.Module):
    def __init__(
        self,
        head_dim: int,
        positions: RopePositions | None = None,
        base: float = 10000.0,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Rope rotates dimension pairs, head_dim must be even"
        if not math.isfinite(base) or base <= 0:
            raise ValueError("RoPE base must be positive finite")
        self.base = float(base)
        self.positions = positions if positions is not None else IndexPositions()
        self.inv_freq = nn.Buffer(
            1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        cumulative_lens: torch.Tensor,
        position_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positions = self.positions(x.shape[0], cumulative_lens, position_values)
        angles = torch.outer(positions, self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)

        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)

        return x * cos + self._rotate_half(x) * sin


NormKind = Literal["rms", "layer", "batch"]
NormPlace = Literal["pre", "post"]

_NORMS: dict[NormKind, type[nn.Module]] = {
    "rms": nn.RMSNorm,
    "layer": nn.LayerNorm,
    "batch": nn.BatchNorm1d,
}


def make_norm(dim: int, kind: NormKind | None) -> nn.Module:
    return nn.Identity() if kind is None else _NORMS[kind](dim)


class ResidualNorm(nn.Module):
    """One normalization, applied on the way into a residual sublayer or on the
    way out of it. Pre-norm leaves the residual stream untouched; post-norm is
    the original Transformer's, and SASRec's."""

    def __init__(self, dim: int, kind: NormKind, place: NormPlace):
        super().__init__()
        self.norm = _NORMS[kind](dim)
        self.pre = place == "pre"

    def before(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) if self.pre else x

    def after(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.pre else self.norm(x)


class _AttentionBlock(ModuleWithDim):
    """Attention with a residual FFN. Where the keys come from, and under which
    mask, is left to the subclass."""

    def __init__(
        self,
        dim: int,
        nhead: int,
        num_kv_heads: int,
        ffn_factory: FFNFactory,
        dropout: float,
        norm: NormKind = "rms",
        norm_place: NormPlace = "pre",
    ) -> None:
        super().__init__()
        assert dim % nhead == 0, "dim must be divisible by nhead"
        self.nhead = nhead
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // nhead
        self._dim = dim

        self.attention_norm = ResidualNorm(dim, norm, norm_place)
        self.q_proj = nn.Linear(dim, nhead * self.head_dim)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(nhead * self.head_dim, dim)
        self.ffn_norm = ResidualNorm(dim, norm, norm_place)
        self.ffn = ffn_factory(dim)
        self.dropout = nn.Dropout(dropout, inplace=True)

    @property
    def out_dim(self) -> int:
        return self._dim

    @property
    def _dropout_p(self) -> float:
        return self.dropout.p if self.training else 0.0

    def _query(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x).view(-1, self.nhead, self.head_dim)

    def _keys_values(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.k_proj(source).view(-1, self.num_kv_heads, self.head_dim),
            self.v_proj(source).view(-1, self.num_kv_heads, self.head_dim),
        )

    def _combine(
        self, residual: torch.Tensor, attention_out: torch.Tensor
    ) -> torch.Tensor:
        projected = self.out_proj(attention_out.reshape(-1, self.nhead * self.head_dim))
        x = self.attention_norm.after(residual + self.dropout(projected))
        return self.ffn_norm.after(x + self.dropout(self.ffn(self.ffn_norm.before(x))))

    def init_weights(self, initializer_range: float = 0.02):
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.trunc_normal_(
                layer.weight,
                mean=0.0,
                std=initializer_range,
                a=-2 * initializer_range,
                b=2 * initializer_range,
            )
            nn.init.zeros_(layer.bias)
        self.ffn.init_weights(initializer_range, initializer_range)


class TransformerBlock(_AttentionBlock):
    def __init__(
        self,
        dim: int,
        nhead: int,
        num_kv_heads: int,
        ffn_factory: FFNFactory,
        dropout: float,
        use_alibi: bool = True,
        rope: Rope | None = None,
        norm: NormKind = "rms",
        norm_place: NormPlace = "pre",
        attention_window: int | None = None,
        softmax_scale: float | None = None,
        is_causal: bool = True,
    ) -> None:
        super().__init__(
            dim, nhead, num_kv_heads, ffn_factory, dropout, norm, norm_place
        )
        self.use_alibi = use_alibi
        self.rope = rope
        self.attention_window = attention_window
        self.softmax_scale = softmax_scale
        self.is_causal = is_causal
        self.alibi_slopes = (
            nn.Buffer(self._get_alibi_slopes(n_heads=nhead)) if use_alibi else None
        )

    @staticmethod
    def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
        assert (
            n_heads > 0 and n_heads & (n_heads - 1) == 0
        ), "Press et al. ALiBi slopes are defined for power-of-two head counts"
        start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
        ratio = start
        return torch.tensor(
            [start * ratio**i for i in range(n_heads)], dtype=torch.float32
        )

    def forward(
        self,
        x: torch.Tensor,
        cumulative_lens: torch.Tensor,
        position_values: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.attention_norm.before(x)
        q = self._query(x)
        k, v = self._keys_values(x)

        if self.rope is not None:
            q = self.rope(q, cumulative_lens, position_values)
            k = self.rope(k, cumulative_lens, position_values)

        if config.cpu_attention:
            attention_out = _cpu_varlen_attention(
                q,
                k,
                v,
                cumulative_lens,
                self.attention_window,
                self.softmax_scale,
                self.is_causal,
            )
        else:
            _assert_flash_dtype(q.dtype)
            cu_seqlens = cumulative_lens.to(torch.int32)
            if max_seqlen is None:
                max_seqlen = _max_seqlen(cumulative_lens)
            attention_out = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self._dropout_p,
                causal=self.is_causal,
                window_size=(
                    (-1, -1)
                    if self.attention_window is None
                    else (
                        self.attention_window - 1,
                        0 if self.is_causal else self.attention_window - 1,
                    )
                ),
                alibi_slopes=self.alibi_slopes,
                softmax_scale=self.softmax_scale,
            )
        return self._combine(residual, attention_out)


class CrossAttentionBlock(_AttentionBlock):
    """Attends from a packed query sequence to a separate packed memory sequence.

    No ALiBi, unlike its self-attention sibling: query and key positions live in
    different sequences, so a relative distance between them means nothing.
    """

    def __init__(
        self,
        dim: int,
        nhead: int,
        num_kv_heads: int,
        ffn_factory: FFNFactory,
        dropout: float,
        norm: NormKind = "rms",
        norm_place: NormPlace = "pre",
        softmax_scale: float | None = None,
    ) -> None:
        super().__init__(
            dim, nhead, num_kv_heads, ffn_factory, dropout, norm, norm_place
        )
        self.memory_norm = _NORMS[norm](dim)
        self.softmax_scale = softmax_scale

    def forward(
        self,
        x: torch.Tensor,
        cumulative_lens_q: torch.Tensor,
        memory: torch.Tensor,
        cumulative_lens_kv: torch.Tensor,
    ) -> torch.Tensor:
        assert (
            cumulative_lens_q.shape[0] == cumulative_lens_kv.shape[0]
        ), "query and memory must describe the same batch of sequences"
        residual = x
        q = self._query(self.attention_norm.before(x))
        k, v = self._keys_values(self.memory_norm(memory))

        if config.cpu_attention:
            attention_out = _cpu_varlen_cross_attention(
                q,
                k,
                v,
                cumulative_lens_q,
                cumulative_lens_kv,
                is_causal=False,
                softmax_scale=self.softmax_scale,
            )
        else:
            _assert_flash_dtype(q.dtype)
            attention_out = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cumulative_lens_q.to(torch.int32),
                cu_seqlens_k=cumulative_lens_kv.to(torch.int32),
                max_seqlen_q=_max_seqlen(cumulative_lens_q),
                max_seqlen_k=_max_seqlen(cumulative_lens_kv),
                dropout_p=self._dropout_p,
                causal=False,
                softmax_scale=self.softmax_scale,
            )
        return self._combine(residual, attention_out)


class LearnedPositionInput(ModuleWithDim):
    """A learned table added to every token, indexed by where the token sits."""

    def __init__(self, dim: int, max_seq_len: int, zero_init: bool = False):
        super().__init__()
        self._dim = dim
        self.position_embeddings = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=dim,
            _weight=torch.zeros(max_seq_len, dim) if zero_init else None,
        )
        self.preserve_declared_initialization = zero_init
        if zero_init:
            nn.init.zeros_(self.position_embeddings.weight)
        else:
            nn.init.trunc_normal_(self.position_embeddings.weight, std=0.02)

    @property
    def out_dim(self) -> int:
        return self._dim

    @abstractmethod
    def _positions(
        self,
        cumulative_lens: torch.Tensor,
        total: int,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor: ...

    def embeddings(
        self,
        cumulative_lens: torch.Tensor,
        total: int,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor:
        return self.position_embeddings(
            self._positions(cumulative_lens, total, reverse_position_offset)
        )

    def forward(
        self,
        x: torch.Tensor,
        cumulative_lens: torch.Tensor,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor:
        return x + self.embeddings(cumulative_lens, x.shape[0], reverse_position_offset)


class ReverseRelativePositionInput(LearnedPositionInput):
    """Position 0 is the newest token, so recency means the same at every length."""

    def _positions(
        self,
        cumulative_lens: torch.Tensor,
        total: int,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor:
        lengths = cumulative_lens.diff()
        sequences, ranks = ragged_positions(lengths, total)
        return (lengths[sequences] - 1 - reverse_position_offset - ranks).clamp_min(0)


class ForwardPositionInput(LearnedPositionInput):
    """Position 0 is the oldest token, so appending one renumbers nothing."""

    def _positions(
        self,
        cumulative_lens: torch.Tensor,
        total: int,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor:
        _, ranks = ragged_positions(cumulative_lens.diff(), total)
        return ranks


class ConcatenatedLearnedPositionInput(ModuleWithDim):
    def __init__(
        self,
        position_inputs: Sequence[LearnedPositionInput],
        encoder: ModuleWithDim,
        rezero: bool = False,
    ) -> None:
        super().__init__()
        if not position_inputs:
            raise ValueError("at least one learned position input is required")
        self.position_inputs = nn.ModuleList(position_inputs)
        self.encoder = encoder
        self.gate = nn.Parameter(torch.zeros(())) if rezero else None

    @property
    def out_dim(self) -> int:
        return self.encoder.out_dim

    def forward(
        self,
        x: torch.Tensor,
        cumulative_lens: torch.Tensor,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor:
        position_embeddings = [
            position.embeddings(cumulative_lens, x.shape[0], reverse_position_offset)
            for position in self.position_inputs
        ]
        encoded = self.encoder(torch.cat([x, *position_embeddings], dim=-1))
        return encoded if self.gate is None else x + self.gate * encoded


class BoundedReverseEmbeddingCorrection(ModuleWithDim):
    def __init__(
        self,
        reverse_input: ReverseRelativePositionInput,
        max_scale: float,
        initializer_rng_nonadvancing: bool = False,
    ) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(()))
        self.reverse_input = reverse_input
        self.max_scale = max_scale
        self.initializer_rng_nonadvancing = initializer_rng_nonadvancing

    @property
    def out_dim(self) -> int:
        return self.reverse_input.out_dim

    @property
    def scale(self) -> torch.Tensor:
        return self.max_scale * self.gate.tanh()

    def forward(
        self,
        cumulative_lens: torch.Tensor,
        total: int,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor:
        return self.scale * self.reverse_input.embeddings(
            cumulative_lens, total, reverse_position_offset
        )


class BoundedReverseAdditivePositionInput(ModuleWithDim):
    def __init__(
        self,
        forward_control: ForwardPositionInput,
        reverse_correction: BoundedReverseEmbeddingCorrection,
    ) -> None:
        super().__init__()
        self.forward_control = forward_control
        self.reverse_correction = reverse_correction

    @property
    def out_dim(self) -> int:
        return self.forward_control.out_dim

    def forward(
        self,
        x: torch.Tensor,
        cumulative_lens: torch.Tensor,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor:
        return self.forward_control(x, cumulative_lens) + self.reverse_correction(
            cumulative_lens, x.shape[0], reverse_position_offset
        )


class BoundedReverseDenseCorrection(ModuleWithDim):
    def __init__(
        self,
        reverse_input: ReverseRelativePositionInput,
        encoder: ModuleWithDim,
        max_scale: float,
        initializer_rng_nonadvancing: bool = False,
    ) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(()))
        self.reverse_input = reverse_input
        self.encoder = encoder
        self.max_scale = max_scale
        self.initializer_rng_nonadvancing = initializer_rng_nonadvancing

    @property
    def out_dim(self) -> int:
        return self.encoder.out_dim

    @property
    def scale(self) -> torch.Tensor:
        return self.max_scale * self.gate.tanh()

    def forward(
        self,
        item: torch.Tensor,
        forward_embeddings: torch.Tensor,
        cumulative_lens: torch.Tensor,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor:
        reverse_embeddings = self.reverse_input.embeddings(
            cumulative_lens, item.shape[0], reverse_position_offset
        )
        encoded = self.encoder(
            torch.cat([item, forward_embeddings, reverse_embeddings], dim=-1)
        )
        return self.scale * encoded


class BoundedReverseConcatenatedPositionInput(ModuleWithDim):
    def __init__(
        self,
        forward_control: ConcatenatedLearnedPositionInput,
        reverse_correction: BoundedReverseDenseCorrection,
    ) -> None:
        super().__init__()
        if len(forward_control.position_inputs) != 1 or not isinstance(
            forward_control.position_inputs[0], ForwardPositionInput
        ):
            raise ValueError("bounded reverse concat requires a forward-concat control")
        self.forward_control = forward_control
        self.reverse_correction = reverse_correction

    @property
    def out_dim(self) -> int:
        return self.forward_control.out_dim

    def forward(
        self,
        x: torch.Tensor,
        cumulative_lens: torch.Tensor,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor:
        forward_embeddings = self.forward_control.position_inputs[0].embeddings(
            cumulative_lens, x.shape[0]
        )
        return self.forward_control(x, cumulative_lens) + self.reverse_correction(
            x, forward_embeddings, cumulative_lens, reverse_position_offset
        )


class _TransformerStack(ModuleWithDim):
    """Shell shared by the encoder and the decoder: dim, position inputs, final norm."""

    def __init__(
        self,
        dim: int,
        final_norm: nn.Module | None,
        position_inputs: list[nn.Module] | None,
        position_dropout: float,
        input_norm: nn.Module,
    ):
        super().__init__()
        self._dim = dim
        self.final_norm = final_norm if final_norm is not None else nn.Identity()
        self.position_inputs = nn.ModuleList(position_inputs or [])
        # Not conditional on there being position inputs: dropping the table
        # would then also drop the input norm, and a run comparing position
        # encodings would be comparing two things.
        self.input_norm = input_norm
        self.position_dropout: nn.Module = nn.Dropout(position_dropout)

    @property
    def out_dim(self) -> int:
        return self._dim

    def _apply_position_inputs(
        self,
        x: torch.Tensor,
        cumulative_lens: torch.Tensor,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor:
        for position_input in self.position_inputs:
            x = position_input(x, cumulative_lens, reverse_position_offset)
        return self.position_dropout(self.input_norm(x))


def _shared_dim(blocks: Sequence[ModuleWithDim]) -> int:
    assert blocks, "at least one transformer block required"
    dim = blocks[0].out_dim
    for block in blocks:
        assert block.out_dim == dim, "all transformer blocks must share the same dim"
    return dim


class TransformerEncoder(_TransformerStack):
    def __init__(
        self,
        blocks: list[TransformerBlock],
        final_norm: nn.Module | None = None,
        position_inputs: list[nn.Module] | None = None,
        position_dropout: float = 0.0,
        input_norm: nn.Module | None = None,
        max_seqlen: int | None = None,
    ):
        input_norm = input_norm or nn.LayerNorm(_shared_dim(blocks), eps=1e-9)
        super().__init__(
            _shared_dim(blocks),
            final_norm,
            position_inputs,
            position_dropout,
            input_norm,
        )
        self.layers = nn.ModuleList(blocks)
        self.max_seqlen = max_seqlen

    def init_weights(self, initializer_range: float = 0.02):
        for layer in self.layers:
            layer.init_weights(initializer_range)

    def forward(
        self,
        x: torch.Tensor,
        cumulative_lens: torch.Tensor,
        position_values: torch.Tensor | None = None,
        layer_inputs: Sequence[torch.Tensor] | None = None,
        layer_item_features: Sequence[torch.Tensor] | None = None,
        layer_item_feature_fusions: Sequence[LayerItemFeatureFusion] | None = None,
        layer_item_feature_mask: torch.Tensor | None = None,
        reverse_position_offset: int = 0,
    ) -> torch.Tensor:
        if layer_inputs is not None and len(layer_inputs) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} layer inputs, got {len(layer_inputs)}"
            )
        if (layer_item_features is None) != (layer_item_feature_fusions is None):
            raise ValueError(
                "layer item features and fusions must be provided together"
            )
        if layer_item_features is not None and (
            len(layer_item_features) != len(self.layers)
            or len(layer_item_feature_fusions) != len(self.layers)
        ):
            raise ValueError(
                f"expected {len(self.layers)} layer item features and fusions"
            )
        if layer_inputs is not None and layer_item_features is not None:
            raise ValueError("legacy layer inputs cannot be combined with item fusions")
        original_input = x
        x = self._apply_position_inputs(x, cumulative_lens, reverse_position_offset)
        max_seqlen = self.max_seqlen
        if not config.cpu_attention and max_seqlen is None:
            max_seqlen = _max_seqlen(cumulative_lens)
        for index, layer in enumerate(self.layers):
            if layer_inputs is not None:
                x = x + layer_inputs[index]
            fusion = (
                None
                if layer_item_feature_fusions is None
                else layer_item_feature_fusions[index]
            )
            if fusion is not None and fusion.placement == "before":
                x = fusion(
                    x,
                    original_input,
                    layer_item_features[index],
                    layer_item_feature_mask,
                )
            x = layer(x, cumulative_lens, position_values, max_seqlen)
            if fusion is not None and fusion.placement == "after":
                x = fusion(
                    x,
                    original_input,
                    layer_item_features[index],
                    layer_item_feature_mask,
                )
        return self.final_norm(x)


class TransformerDecoder(_TransformerStack):
    """Alternates causal self-attention over the query sequence with cross-attention to a memory."""

    def __init__(
        self,
        self_attention_blocks: list[TransformerBlock],
        cross_attention_blocks: list[CrossAttentionBlock],
        final_norm: nn.Module | None = None,
        position_inputs: list[nn.Module] | None = None,
        position_dropout: float = 0.0,
        input_norm: nn.Module | None = None,
    ):
        assert len(self_attention_blocks) == len(
            cross_attention_blocks
        ), "every self-attention block must be paired with a cross-attention block"
        dim = _shared_dim([*self_attention_blocks, *cross_attention_blocks])
        super().__init__(
            dim,
            final_norm,
            position_inputs,
            position_dropout,
            input_norm or nn.LayerNorm(dim, eps=1e-9),
        )
        self.self_attention_layers = nn.ModuleList(self_attention_blocks)
        self.cross_attention_layers = nn.ModuleList(cross_attention_blocks)

    def init_weights(self, initializer_range: float = 0.02):
        for layer in [*self.self_attention_layers, *self.cross_attention_layers]:
            layer.init_weights(initializer_range)

    def forward(
        self,
        x: torch.Tensor,
        cumulative_lens: torch.Tensor,
        memory: torch.Tensor,
        memory_cumulative_lens: torch.Tensor,
    ) -> torch.Tensor:
        x = self._apply_position_inputs(x, cumulative_lens)
        for self_attention, cross_attention in zip(
            self.self_attention_layers, self.cross_attention_layers
        ):
            x = self_attention(x, cumulative_lens)
            x = cross_attention(x, cumulative_lens, memory, memory_cumulative_lens)
        return self.final_norm(x)
