from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn

from dcn.config.networks import build_causal_transformer
from dcn.config.retrieval import SampledSoftmaxExperiment
from dcn.config.settings import (
    TRANSFORMER,
    DataloaderConfig,
    DayRangeConfig,
    EmbeddingConfig,
    RuntimeConfig,
)
from dcn.config.yambda_base import NUM_HASHES, YambdaSourceExperiment, counted_field
from dcn.data import EmaCounter
from dcn.eval import TrueMetricCallback, build_interaction_sets, build_item_snapshot
from dcn.models import Tower, TowerInputEncoder, TwoTowerModel
from dcn.nn import ResNet1D

_CATEGORICAL_FEATURES = ("compact_item_id", "artist_id")

_RESNET_HIDDEN_DIMS = [128]


@dataclass
class TwoTowerRetrievalExperiment(YambdaSourceExperiment, SampledSoftmaxExperiment):
    """A query tower and an item tower, scored against each other by dot product."""

    enable_true_metric: bool = False

    embedding: EmbeddingConfig = field(
        default_factory=lambda: EmbeddingConfig(num_embeddings=2**17, dim=64)
    )

    def settings_defaults(self) -> dict[str, Any]:
        # A short slice: these exercise the two-tower stack rather than being
        # compared against the ranking and generation runs.
        return {
            **super().settings_defaults(),
            "day_range": DayRangeConfig(start_day=0, end_day=40),
            "dataloader": DataloaderConfig(
                batch_size=128, val_batch_size=128, num_workers=8, prefetch_factor=4
            ),
            "runtime": RuntimeConfig(dtype=torch.bfloat16),
        }

    @cached_property
    def _counter_groups(self) -> dict[str, list[EmaCounter]]:
        return {"history": [], "item": []}

    def create_counters(self) -> list[EmaCounter]:
        return [*self._counter_groups["history"], *self._counter_groups["item"]]

    @cached_property
    def history_counter_columns(self) -> list[str]:
        return [
            column
            for counter in self._counter_groups["history"]
            for column in counter.get_output_columns()
        ]

    @cached_property
    def item_counter_columns(self) -> list[str]:
        return [
            column
            for counter in self._counter_groups["item"]
            for column in counter.get_output_columns()
        ]

    @property
    def emit_user_column(self) -> bool:
        return self.enable_true_metric

    @property
    @abstractmethod
    def item_snapshot_columns(self) -> list[str]: ...

    def tower(
        self,
        categorical_columns: Sequence[str],
        *,
        counter_columns: Sequence[str] = (),
        counter_encoder: nn.Module | None = None,
        body_factory: Callable[[int], Any] | None = None,
        sequence_model: Any | None = None,
    ) -> Tower:
        return Tower(
            TowerInputEncoder(
                num_embeddings=self.embedding.num_embeddings,
                embedding_dim=self.embedding.dim,
                categorical_columns=categorical_columns,
                body_factory=body_factory,
                counter_encoder=counter_encoder,
                num_hashes=NUM_HASHES,
            ),
            categorical_columns=categorical_columns,
            counter_columns=counter_columns,
            sequence_model=sequence_model,
        )

    def extra_callbacks(self, train_days: list[int], val_days: list[int]) -> list[Any]:
        if not self.enable_true_metric:
            return []

        day_to_path = self.dataset_manager.day_to_path
        train_files = [day_to_path[day] for day in train_days]
        val_files = [day_to_path[day] for day in val_days]

        return [
            TrueMetricCallback(
                model=self.base_model,
                item_batch=build_item_snapshot(
                    train_files,
                    item_id_column=self.item_id_column,
                    columns=self.item_snapshot_columns,
                    timestamp_column=self.artifacts.timestamp_column,
                ),
                query_loader=self.make_cutoff_query_loader(train_days),
                relevance=build_interaction_sets(
                    val_files,
                    user_column=self.user_column,
                    item_id_column=self.item_id_column,
                ),
                train_seen=build_interaction_sets(
                    train_files,
                    user_column=self.user_column,
                    item_id_column=self.item_id_column,
                ),
                user_column=self.user_column,
                item_id_column=self.item_id_column,
            )
        ]


@dataclass
class SasRecExperiment(TwoTowerRetrievalExperiment):
    """SASRec-style two-tower retrieval over hashed features and counters."""

    run_name: str = "sasrec"

    @cached_property
    def _counter_groups(self) -> dict[str, list[EmaCounter]]:
        return {
            "history": [
                self.counter(keys=["uid"], fields=[counted_field("like", [7])]),
                self.counter(
                    keys=["uid", "item_id"], fields=[counted_field("like", [7])]
                ),
                self.counter(
                    keys=["uid", "artist_id"],
                    fields=[counted_field("like", [7])],
                    aggregations=("mean",),
                ),
            ],
            "item": [
                self.counter(
                    keys=["item_id"], fields=[counted_field("like", [7, 30, 90])]
                ),
                self.counter(keys=["item_id"], fields=[counted_field("dislike", [7])]),
            ],
        }

    @property
    def sequence_columns(self) -> list[str]:
        return [
            *_CATEGORICAL_FEATURES,
            *self.history_counter_columns,
            *self.item_counter_columns,
        ]

    @property
    def item_snapshot_columns(self) -> list[str]:
        return [*_CATEGORICAL_FEATURES, *self.item_counter_columns]

    def _tower(self, counter_columns: Sequence[str], **kwargs: Any) -> Tower:
        return self.tower(
            _CATEGORICAL_FEATURES,
            counter_columns=counter_columns,
            counter_encoder=self.fit_counter_encoder(counter_columns),
            body_factory=lambda input_dim: ResNet1D(
                input_dim=input_dim,
                hidden_dims=[*_RESNET_HIDDEN_DIMS, TRANSFORMER.dim],
                norm_factory=nn.LayerNorm,
                dropout=TRANSFORMER.dropout,
            ),
            **kwargs,
        )

    def _create_model(self) -> TwoTowerModel:
        return TwoTowerModel(
            self._tower(
                self.history_counter_columns,
                sequence_model=build_causal_transformer(
                    TRANSFORMER, max_seq_len=self.max_seq_len
                ),
            ),
            self._tower(self.item_counter_columns),
            item_id_column=self.item_id_column,
        )


@dataclass
class SimpleTwoTowerExperiment(TwoTowerRetrievalExperiment):
    """Plain user-id x item-id two-tower retrieval (embedding-only)."""

    run_name: str = "simple_two_tower"
    enable_true_metric: bool = True

    @property
    def item_snapshot_columns(self) -> list[str]:
        return [self.item_id_column]

    @property
    def sequence_columns(self) -> list[str]:
        return [self.item_id_column]

    @property
    def emit_user_column(self) -> bool:
        return True

    def _create_model(self) -> TwoTowerModel:
        return TwoTowerModel(
            self.tower([self.user_column]),
            self.tower([self.item_id_column]),
            item_id_column=self.item_id_column,
        )
