"""Pairwise accuracy: does the model order adjacent events the way the user did?"""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from dcn.data.packed import ragged_positions, to_cumulative_lens
from dcn.eval.base import ScoredBatchesCallback

SESSION_GAP_SECONDS = 15 * 60


def pairwise_ordering_scores(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    timestamps: torch.Tensor,
    cumulative_lens: torch.Tensor,
    session_gap_seconds: int,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """1, 0.5 or 0 for every adjacent pair of events worth comparing."""
    if valid is not None:
        kept = valid.nonzero().flatten()
        cumulative_lens = _kept_lens(kept, cumulative_lens, valid.shape[0])
        predictions, targets, timestamps = (
            predictions[kept],
            targets[kept],
            timestamps[kept],
        )
    if predictions.shape[0] < 2:
        return predictions.new_empty(0)

    sequences, _ = ragged_positions(cumulative_lens.diff(), targets.shape[0])
    target_steps = targets.float().diff()
    prediction_steps = predictions.float().diff()

    compared = (
        (sequences.diff() == 0)
        & (timestamps.diff() <= session_gap_seconds)
        & (target_steps != 0)
    )
    scores = (target_steps * prediction_steps > 0).float()
    return scores.masked_fill(prediction_steps == 0, 0.5)[compared]


@dataclass(frozen=True)
class PairwiseTarget:
    name: str
    prediction_column: str
    target_column: str
    mask_column: str | None = None


class PairwiseAccuracyCallback(ScoredBatchesCallback):
    def __init__(
        self,
        *,
        model: nn.Module,
        loader: Iterable[dict],
        targets: list[PairwiseTarget],
        session_gap_seconds: int = SESSION_GAP_SECONDS,
        dtype: torch.dtype = torch.float32,
        every_n_epochs: int = 1,
        prefix: str = "epoch/val_pairwise",
    ) -> None:
        super().__init__(
            model=model,
            loader=loader,
            prefix=prefix,
            dtype=dtype,
            every_n_epochs=every_n_epochs,
        )
        self.targets = targets
        self.session_gap_seconds = session_gap_seconds

    def _batch_scores(self, batch: dict) -> dict[str, torch.Tensor]:
        predictions = self.model(batch)
        return {
            target.name: pairwise_ordering_scores(
                predictions=predictions[target.prediction_column].values.squeeze(-1),
                targets=batch["float_columns"][target.target_column].dense(),
                timestamps=batch["timestamp"],
                cumulative_lens=batch["cumulative_lens"],
                session_gap_seconds=self.session_gap_seconds,
                valid=None
                if target.mask_column is None
                else batch["int_columns"][target.mask_column].dense().bool(),
            )
            for target in self.targets
        }


def _kept_lens(
    kept: torch.Tensor, cumulative_lens: torch.Tensor, total: int
) -> torch.Tensor:
    lengths = cumulative_lens.diff()
    sequences, _ = ragged_positions(lengths, total)
    return to_cumulative_lens(
        torch.zeros_like(lengths).scatter_add_(
            0, sequences[kept], torch.ones_like(kept)
        )
    )
