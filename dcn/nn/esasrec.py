import torch
from torch import nn

import dcn.nn.transformer as transformer_module
from utils.global_config import config

from .ffn import RegularMLP, SwiGLU
from .types import ModuleWithDim


class _PackedCausalBlock(ModuleWithDim):
    def __init__(self, dim: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if dim % nhead:
            raise ValueError("dim must be divisible by nhead")
        self._dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.attention_dropout = dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    @property
    def out_dim(self) -> int:
        return self._dim

    def _project(self, source: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        return projection(source).view(-1, self.nhead, self.head_dim)

    def _attention(
        self,
        query_source: torch.Tensor,
        key_value_source: torch.Tensor,
        cumulative_lens: torch.Tensor,
        max_seqlen: int | None,
    ) -> torch.Tensor:
        query = self._project(query_source, self.q_proj)
        key = self._project(key_value_source, self.k_proj)
        value = self._project(key_value_source, self.v_proj)
        if config.cpu_attention:
            attention = transformer_module._cpu_varlen_attention(
                query, key, value, cumulative_lens
            )
        else:
            transformer_module._assert_flash_dtype(query.dtype)
            if max_seqlen is None:
                max_seqlen = transformer_module._max_seqlen(cumulative_lens)
            cumulative_lens = cumulative_lens.to(torch.int32)
            attention = transformer_module.flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cumulative_lens,
                cu_seqlens_k=cumulative_lens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
            )
        return self.out_proj(attention.reshape(-1, self._dim))

    def _initialize_attention(self, initializer_std: float) -> None:
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.trunc_normal_(
                projection.weight,
                std=initializer_std,
                a=-2 * initializer_std,
                b=2 * initializer_std,
            )
            nn.init.zeros_(projection.bias)


class SASRecBlock(_PackedCausalBlock):
    def __init__(
        self,
        dim: int,
        nhead: int,
        intermediate_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(dim, nhead, dropout)
        self.attention_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = RegularMLP(
            dim,
            intermediate_dim,
            dropout=dropout,
            activation="relu",
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cumulative_lens: torch.Tensor,
        position_values: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        normalized_query = self.attention_norm(x)
        x = normalized_query + self._attention(
            normalized_query, x, cumulative_lens, max_seqlen
        )
        normalized_ffn = self.ffn_norm(x)
        return normalized_ffn + self.dropout(self.ffn(normalized_ffn))

    def init_weights(self, initializer_std: float = 0.02) -> None:
        self._initialize_attention(initializer_std)
        self.ffn.init_weights(initializer_std, initializer_std)


class LiGRBlock(_PackedCausalBlock):
    def __init__(
        self,
        dim: int,
        nhead: int,
        intermediate_dim: int,
        dropout: float,
    ) -> None:
        if intermediate_dim % 32:
            raise ValueError("LiGR SwiGLU width must be divisible by 32")
        super().__init__(dim, nhead, dropout)
        self.attention_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = SwiGLU(
            dim,
            intermediate_dim,
            dropout=dropout,
            bias=False,
        )
        self.attention_gate = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cumulative_lens: torch.Tensor,
        position_values: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(x)
        attention = self._attention(normalized, normalized, cumulative_lens, max_seqlen)
        x = torch.addcmul(x, self.attention_gate(x).sigmoid(), self.dropout(attention))
        normalized = self.ffn_norm(x)
        return torch.addcmul(
            x,
            self.ffn_gate(x).sigmoid(),
            self.dropout(self.ffn(normalized)),
        )

    def init_weights(self, initializer_std: float = 0.02) -> None:
        self._initialize_attention(initializer_std)
        self.ffn.init_weights(initializer_std, initializer_std)
        for gate in (self.attention_gate, self.ffn_gate):
            nn.init.trunc_normal_(
                gate.weight,
                std=initializer_std,
                a=-2 * initializer_std,
                b=2 * initializer_std,
            )
            nn.init.zeros_(gate.bias)
