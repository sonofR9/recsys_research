"""Transformer encoder using flash_attn package. Same interface as nn/transformer.TransformerEncoder."""

import torch
import torch.nn as nn
from flash_attn import flash_attn_func


class FlashAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ff_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        self.softmax_scale = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ff_dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def _self_attn(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.nhead, self.head_dim)
        k = self.k_proj(x).view(B, S, self.nhead, self.head_dim)
        v = self.v_proj(x).view(B, S, self.nhead, self.head_dim)

        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )  # (B, S, H, D)
        out = out.reshape(B, S, -1)
        return self.out_proj(out)

    def forward(self, src: torch.Tensor, causal: bool) -> torch.Tensor:
        x = src + self.attn_dropout(self._self_attn(self.self_attn_norm(src), causal))
        x = x + self.ff_dropout(
            self.linear2(self.activation(self.linear1(self.ff_norm(x))))
        )
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.causal = causal

        self.layers = nn.ModuleList(
            [
                FlashAttentionEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            src: Input embeddings (batch_size, seq_len, d_model).

        Returns:
            Encoded sequence (batch_size, seq_len, d_model).
        """
        x = src
        for layer in self.layers:
            x = layer(x, causal=self.causal)
        return x
