from collections.abc import Callable, Sequence

import torch
from torch import nn

from dcn.data.features import FeatureValues
from dcn.nn.multi_task_embedding import MultiTaskEmbeddingLayer
from dcn.nn.precomputed_embeddings import segment_sum
from dcn.nn.sampled_softmax import InBatchSampledSoftmaxLoss
from dcn.nn.types import ModuleWithDim
from neuralrec.utils import LOSS_DENOMINATOR

from .sequence_targets import NextItemTargets


def _sum_counter_bags(
    counters: list[FeatureValues], reference: torch.Tensor
) -> torch.Tensor:
    # A counter column is ragged: one bag per token, one value per window.
    if not counters:
        return reference.new_zeros(reference.shape[0], 0)
    pooled = [
        segment_sum(counter.values.unsqueeze(-1), counter.offsets)
        for counter in counters
    ]
    return torch.cat(pooled, dim=1)


class TowerInputEncoder(ModuleWithDim):
    def __init__(
        self,
        *,
        num_embeddings: int,
        embedding_dim: int,
        categorical_columns: Sequence[str],
        body_factory: Callable[[int], ModuleWithDim] | None = None,
        counter_encoder: ModuleWithDim | None = None,
        num_hashes: int = 1,
    ):
        """The body is built here, from the width this encoder actually produces.

        A caller computing that width itself re-derives the embedding layout,
        and mis-shapes the body the moment a column or a hash is added.
        """
        super().__init__()
        self.embedding = MultiTaskEmbeddingLayer(
            feature_configs={name: num_hashes for name in categorical_columns},
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            split_ratios={"shared": 1.0},
            sparse=False,
            mode="concat",
        )
        self.counter_encoder = counter_encoder
        counter_dim = 0 if counter_encoder is None else counter_encoder.out_dim
        self._input_dim = self.embedding.out_dim.shared + counter_dim
        self.body = None if body_factory is None else body_factory(self._input_dim)

    @property
    def out_dim(self) -> int:
        return self._input_dim if self.body is None else self.body.out_dim

    def forward(
        self, categorical: dict[str, FeatureValues], counters: list[FeatureValues]
    ) -> torch.Tensor:
        token_embedding = self.embedding(categorical)["shared"]
        counter_features = _sum_counter_bags(counters, token_embedding)
        if counters:
            assert self.counter_encoder is not None, "counters need a counter_encoder"
            counter_features = self.counter_encoder(counter_features)
        encoded = torch.cat([token_embedding, counter_features], dim=1)
        return encoded if self.body is None else self.body(encoded)


class Tower(ModuleWithDim):
    """One side of a two-tower model: the columns it reads and what it does
    with them. A tower with a sequence model summarises the packed history;
    one without scores each row on its own."""

    def __init__(
        self,
        encoder: ModuleWithDim,
        *,
        categorical_columns: Sequence[str],
        counter_columns: Sequence[str] = (),
        sequence_model: ModuleWithDim | None = None,
    ):
        super().__init__()
        assert sequence_model is None or encoder.out_dim == sequence_model.out_dim, (
            "encoder/sequence model dim mismatch"
        )
        self.encoder = encoder
        self.sequence_model = sequence_model
        self.categorical_columns = list(categorical_columns)
        self.counter_columns = list(counter_columns)

    @property
    def out_dim(self) -> int:
        return (
            self.encoder.out_dim
            if self.sequence_model is None
            else self.sequence_model.out_dim
        )

    def forward(self, batch: dict) -> torch.Tensor:
        encoded = self.encoder(
            {name: batch["int_columns"][name] for name in self.categorical_columns},
            [batch["float_columns"][name] for name in self.counter_columns],
        )
        if self.sequence_model is None:
            return encoded
        return self.sequence_model(encoded, batch["cumulative_lens"])


class TwoTowerModel(nn.Module):
    def __init__(
        self, query_tower: Tower, item_tower: Tower, *, item_id_column: str
    ) -> None:
        super().__init__()
        assert query_tower.out_dim == item_tower.out_dim, "tower dim mismatch"
        self.query_tower = query_tower
        self.item_tower = item_tower
        self.item_id_column = item_id_column

    def encode_queries(self, batch: dict) -> torch.Tensor:
        return self.query_tower(batch)

    def encode_items(self, batch: dict) -> torch.Tensor:
        return self.item_tower(batch)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        return {
            "query_repr": self.encode_queries(batch),
            "item_repr": self.encode_items(batch),
            "item_ids": batch["int_columns"][self.item_id_column].dense(),
            "lengths": batch["cumulative_lens"].diff(),
        }


class TwoTowerLoss(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        loss: InBatchSampledSoftmaxLoss,
        targets: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        self.targets = targets if targets is not None else NextItemTargets()

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        out = self.model(batch)
        pairs = self.targets(out)
        if pairs.query_repr.shape[0] == 0:
            # graph-connected so the trainer's unconditional backward() still works
            zero = (out["query_repr"].sum() + out["item_repr"].sum()) * 0.0
            return {
                "loss": zero,
                "hit_rate": zero.detach(),
                LOSS_DENOMINATOR: 0,
            }

        # loss and hit rate share one pass, so both see the same negatives
        logits = self.loss.logits(
            pairs.query_repr, pairs.positive_repr, pairs.positive_ids, pairs.group_sizes
        )
        return {
            "loss": self.loss.loss_from_logits(logits),
            "hit_rate": (logits.detach().argmax(dim=1) == 0).float().mean(),
            LOSS_DENOMINATOR: pairs.query_repr.shape[0],
        }
