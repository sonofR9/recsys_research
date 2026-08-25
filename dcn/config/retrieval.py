from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch.nn as nn

from dcn.config.sequence import SequenceExperiment
from dcn.models import NextItemTargets, SequenceTargets, TwoTowerLoss
from dcn.nn import StreamingInBatchSoftmax


@dataclass
class RetrievalExperiment(SequenceExperiment):
    """A run whose criterion owns the model rather than wrapping its output."""

    @property
    def num_items(self) -> int:
        """Highest compact item id in the catalog; ids run ``1..num_items``."""
        return self.item_embeddings.num_known_ids

    @property
    def catalog_size(self) -> int:
        """One row wider than the catalog: id 0 is the unknown item."""
        return self.num_items + 1

    @property
    def embedding_types(self) -> Sequence[type[nn.Module]]:
        return [nn.Embedding, *super().embedding_types]

    def create_training_model(self) -> nn.Module:
        return self.apply_runtime_wrappers(
            self.create_criterion().to(self.runner_build_device)
        )


@dataclass
class SampledSoftmaxExperiment(RetrievalExperiment):
    """Trained by in-batch sampled softmax over the item catalog."""

    num_in_batch_negatives: int = 512
    logq_alpha: float = 0.01

    def create_targets(self) -> SequenceTargets:
        return NextItemTargets()

    def create_criterion(self) -> nn.Module:
        return TwoTowerLoss(
            self.base_model,
            StreamingInBatchSoftmax(
                hash_size=self.catalog_size,
                num_in_batch_negatives=self.num_in_batch_negatives,
                alpha=self.logq_alpha,
            ),
            targets=self.create_targets(),
        )
