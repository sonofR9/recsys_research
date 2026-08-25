"""Basic transformer encoder model based on torch.nn.TransformerEncoder."""

import torch
import torch.nn as nn


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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=True,
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
        mask = None
        if self.causal:
            seq_len = src.size(1)
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=src.device),
                diagonal=1,
            )
        out = self.transformer_encoder(src, mask=mask)
        return out
