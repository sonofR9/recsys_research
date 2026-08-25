"""What every yambda run shares regardless of what it trains."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch.nn as nn

from dcn.config.experiment import Experiment
from dcn.data import DecayConfig, FieldConfig
from dcn.datasets.base import DatasetSource
from dcn.datasets.yambda import UserSample, YambdaDatasetSource, YambdaSize
from dcn.eval import PairwiseTarget
from dcn.models import CriterionSpec, MultiCriterion, TargetExtractionWrapper
from neuralrec.nn.metrics import RMSE, LogLikelihoodOfPrediction, R2Score

# The raw yambda frame the counters read, before events are remapped to ids.
EVENT_TYPE_COLUMN = "event_type"

TASK_NAMES = ("like", "listen")
MULTI_TASK_SPLIT_RATIOS = {"shared": 0.4, "like": 0.3, "listen": 0.3}
NUM_HASHES = 2

extract_like = partial(
    TargetExtractionWrapper, prediction_column="like", target_column="target_like"
)
extract_listen = partial(
    TargetExtractionWrapper,
    prediction_column="listen",
    target_column="target_listen",
    mask_column="listen_mask",
)
extract_full_play = partial(
    TargetExtractionWrapper,
    prediction_column="full_play",
    target_column="target_full_play",
)


def counted_field(event_type: str, half_life_days: list[float]) -> FieldConfig:
    return FieldConfig.matching(
        EVENT_TYPE_COLUMN,
        event_type,
        [DecayConfig(half_life_days=days) for days in half_life_days],
    )


like_field = partial(counted_field, "like")


@dataclass
class YambdaSourceExperiment(Experiment):
    """An experiment whose events come from the yambda dataset."""

    size: YambdaSize = "50m"
    user_sample: UserSample | None = field(
        default_factory=lambda: UserSample(max_users=10_000)
    )
    listen_sample_fraction: float = 1.0
    event_type_filter: str | None = None
    min_item_interactions_per_item: int = 0
    drop_unmapped_items: bool = False

    dataset_source_class = YambdaDatasetSource

    def create_dataset_source(self) -> DatasetSource:
        return self.dataset_source_class(
            data_path=Path(self.base_path) / "yambda_data",
            size=self.size,
            user_sample=self.user_sample,
            listen_sample_fraction=self.listen_sample_fraction,
            event_type_filter=self.event_type_filter,
            min_item_interactions_per_item=self.min_item_interactions_per_item,
            drop_unmapped_items=self.drop_unmapped_items,
            invalidate_cache=self.invalidate_cache,
        )


class LikeListenTargets:
    """Predict like/not-like and, for listens only, the played ratio."""

    task_names = TASK_NAMES
    head_split_ratios = MULTI_TASK_SPLIT_RATIOS
    target_columns = ["target_like", "target_listen", "listen_mask"]

    def create_criterion(self) -> nn.Module:
        return MultiCriterion(
            [
                CriterionSpec(
                    name="like",
                    criterion=extract_like(nn.BCEWithLogitsLoss()),
                    weight=1.0,
                ),
                CriterionSpec(
                    name="listen", criterion=extract_listen(nn.MSELoss()), weight=0.5
                ),
            ]
        )

    def create_metrics(self) -> list[TargetExtractionWrapper]:
        return [
            extract_like(LogLikelihoodOfPrediction()),
            extract_listen(RMSE()),
            extract_listen(R2Score()),
        ]

    def pairwise_targets(self) -> list[PairwiseTarget]:
        """What the pairwise ordering metric compares, per head.

        The listen head is scored against the played ratio itself, so a pair
        counts whenever one of the two was played longer.
        """
        return [
            PairwiseTarget(
                name="like", prediction_column="like", target_column="target_like"
            ),
            PairwiseTarget(
                name="listen",
                prediction_column="listen",
                target_column="target_listen",
                mask_column="listen_mask",
            ),
        ]


class LikeFullPlayTargets:
    """The ranking homework's two binary targets: was the track liked, and was
    it played to the end."""

    task_names = ("like", "full_play")
    head_split_ratios = {"shared": 0.4, "like": 0.3, "full_play": 0.3}
    target_columns = ["target_like", "target_full_play"]

    def create_criterion(self) -> nn.Module:
        return MultiCriterion(
            [
                CriterionSpec(
                    name=name,
                    criterion=extract(nn.BCEWithLogitsLoss()),
                    weight=1.0,
                )
                for name, extract in (
                    ("like", extract_like),
                    ("full_play", extract_full_play),
                )
            ]
        )

    def create_metrics(self) -> list[TargetExtractionWrapper]:
        return [
            extract_like(LogLikelihoodOfPrediction()),
            extract_full_play(LogLikelihoodOfPrediction()),
        ]

    def pairwise_targets(self) -> list[PairwiseTarget]:
        return [
            PairwiseTarget(
                name="like", prediction_column="like", target_column="target_like"
            ),
            PairwiseTarget(
                name="full_play",
                prediction_column="full_play",
                target_column="target_full_play",
            ),
        ]
