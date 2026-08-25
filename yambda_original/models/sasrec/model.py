from enum import Enum
from typing import Dict

import torch
import torch.nn as nn

from models.sasrec.transformer_copied import TransformerUserEncoder


class OutputProjection(str, Enum):
    NONE = "none"
    EMBEDDING_DIM = "embedding_dim"
    PRETRAINED_DIM = "pretrained_dim"


class SASRecEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_bos_tokens: bool = True,
        use_alibi: bool = False,
        use_positional_embedding: bool = True,
        use_swiglu: bool = False,
        num_kv_heads: int | None = None,
        pretrained_embeddings: torch.Tensor | None = None,
        output_projection: str = "none",
    ) -> None:
        super().__init__()
        self._num_items = num_items
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim
        self._output_projection = OutputProjection(output_projection)

        self._pretrained_dim = (
            pretrained_embeddings.size(1)
            if pretrained_embeddings is not None
            else 0
        )

        if pretrained_embeddings is not None and use_bos_tokens:
            bos_row = torch.zeros(1, self._pretrained_dim)
            pretrained_embeddings = torch.cat([pretrained_embeddings, bos_row], dim=0)

        total_dim = embedding_dim + self._pretrained_dim

        self._transformer = TransformerUserEncoder(
            num_items=num_items,
            embedding_dim=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads or num_heads,
            max_seq_len=max_sequence_length,
            pretrained_embeddings=pretrained_embeddings,
            sparse=False,
            use_swiglu=use_swiglu,
            use_bos_tokens=use_bos_tokens,
            use_alibi=use_alibi,
            use_positional_embedding=use_positional_embedding,
        )

        self._item_embeddings = self._transformer.item_embeddings

        if self._output_projection == OutputProjection.NONE:
            self._output_dim = total_dim
            self._output_proj = nn.Identity()
        elif self._output_projection == OutputProjection.EMBEDDING_DIM:
            self._output_dim = embedding_dim
            self._output_proj = nn.Linear(total_dim, embedding_dim)
        else:
            self._output_dim = self._pretrained_dim
            self._output_proj = nn.Linear(total_dim, self._pretrained_dim)

        self._init_weights(initializer_range)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def item_embeddings(self) -> nn.Module:
        return self._item_embeddings

    @property
    def num_items(self) -> int:
        return self._num_items

    @torch.no_grad()
    def _init_weights(self, initializer_range: float) -> None:
        self._transformer.init_weights()

    def get_all_item_embeddings(self) -> torch.Tensor:
        n = self._item_embeddings.num_real_items
        if self._output_projection == OutputProjection.NONE:
            return self._item_embeddings(torch.arange(n, device=self._item_embeddings.embedding.weight.device)).detach()
        elif self._output_projection == OutputProjection.EMBEDDING_DIM:
            return self._item_embeddings.embedding.weight[:n].detach()
        else:
            return self._item_embeddings.pretrained[:n].detach()

    def _get_item_embeddings_for_scoring(self, item_ids: torch.Tensor) -> torch.Tensor:
        if self._output_projection == OutputProjection.NONE:
            return self._item_embeddings(item_ids)
        elif self._output_projection == OutputProjection.EMBEDDING_DIM:
            return self._item_embeddings.embedding(item_ids)
        else:
            return self._item_embeddings.pretrained[item_ids]

    @staticmethod
    def _get_last_embedding(
        embeddings: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        cumulative_lengths = torch.cumsum(lengths, dim=0)
        last_indices = cumulative_lengths - 1
        return embeddings[last_indices]

    def forward(self, inputs: Dict) -> torch.Tensor:
        all_sample_events = inputs["item.ids"]
        all_sample_lengths = inputs["item.length"]

        transformer_inputs = {
            "item.ids": all_sample_events,
            "item.length": all_sample_lengths,
        }
        embeddings = self._output_proj(self._transformer(transformer_inputs))

        if self.training:
            positive_embeddings = self._get_item_embeddings_for_scoring(inputs["positive.ids"])
            positive_scores = torch.bmm(
                embeddings.unsqueeze(1),  # shape: (b, 1, d)
                positive_embeddings.unsqueeze(2)  # shape: (b, d, 1)
            ).squeeze()  # shape: (b,)

            negative_embeddings = self._get_item_embeddings_for_scoring(inputs["negative.ids"])
            negative_scores = torch.bmm(
                embeddings.unsqueeze(1),  # shape: (b, 1, d)
                negative_embeddings.unsqueeze(2)  # shape: (b, d, 1)
            ).squeeze()  # shape: (b,)

            loss = nn.functional.binary_cross_entropy_with_logits(
                torch.cat([positive_scores, negative_scores], dim=0),
                torch.cat(
                    [
                        torch.ones_like(positive_scores),
                        torch.zeros_like(negative_scores),
                    ]
                ),
            )

            return loss
        else:
            return self._get_last_embedding(embeddings, all_sample_lengths)
