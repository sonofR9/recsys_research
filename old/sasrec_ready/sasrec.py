import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F

import kagglehub
from kagglehub import KaggleDatasetAdapter

from transformers import get_cosine_schedule_with_warmup

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import shutil

MAX_SEQ_LEN = 265


class SASRec(nn.Module):
    def __init__(self, item2ind, SPECIAL_TOKENS):
        super(SASRec, self).__init__()

        d_model = 64
        n_heads = 2
        dropout = 0.0
        n_layers = 5

        self.item_emb = nn.Embedding(
            len(item2ind) + len(SPECIAL_TOKENS), d_model, padding_idx=0
        )
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Linear(
            d_model, len(item2ind) + len(SPECIAL_TOKENS), bias=False
        )
        self.head.weight = self.item_emb.weight

        self._init_weights(0.02)

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(self, X):
        """
        item_seq: LongTensor of shape (B, T) with padding = pad_idx
        """
        X = self.get_hidden(X)

        logits = self.head(X)
        return logits

    def get_hidden(self, X):
        B, T = X.shape
        device = X.device

        key_padding_mask = X == 0

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        X = self.item_emb(X) + self.pos_emb(pos)

        causal_mask = self.causal_mask[:T, :T].to(
            device=device, dtype=torch.bool
        )

        X = self.encoder(
            X,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=True,
        )

        return X

    @torch.no_grad()
    def top_k(
        self,
        history: torch.LongTensor,
        k: int,
    ) -> torch.LongTensor:
        device = history.device
        B = history.size(0)

        history = torch.cat(
            [torch.ones(B, 1, device=device, dtype=torch.long), history], dim=1
        )
        lengths = (history != 0).sum(dim=1)
        h = self.get_hidden(history)

        last_idx = lengths - 1
        last_h = h[torch.arange(B, device=device), last_idx]

        logits = self.head(last_h)
        logits[:, 0:3] = -torch.inf

        vals, idxs = torch.topk(logits, k=k, dim=-1)

        return idxs

    @torch.no_grad()
    def _init_weights(self, initializer_range: float) -> None:
        """
        Initialize all model parameters (weights and biases) in-place.
        For each parameter in the model:
            - If the parameter name contains 'weight':
                - If it also contains 'norm' (e.g., for normalization layers), initialize with ones.
                - Otherwise, initialize with a truncated normal distribution (mean=0, std=initializer_range)
                and values clipped to the range [-2 * initializer_range, 2 * initializer_range].
            - If the parameter name contains 'bias', initialize with zeros.
            - If the parameter name does not match either case, raise a ValueError.
        Args:
            initializer_range (float): Standard deviation for the truncated normal distribution
                used to initialize non-normalization weights.
        Note:
            This method should be called during model initialization to ensure all weights and biases
            are properly set. It runs in a no-grad context and does not track gradients.
        """
        for key, value in self.named_parameters():
            if "weight" in key:
                if "norm" in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range,
                    )
            else:
                assert "bias" in key
                nn.init.zeros_(value.data)
