import pytest
import torch
from torch import nn

from dcn.data.features import FeatureValues
from dcn.data.packed import to_cumulative_lens
from dcn.eval.pairwise import (
    PairwiseAccuracyCallback,
    PairwiseTarget,
    pairwise_ordering_scores,
)
from neuralrec.utils import EXTRA_METRICS

TARGET_COLUMN = "target_like"
MASK_COLUMN = "listen_mask"
PREDICTION_COLUMN = "like"
GAP = 100


def _scores(
    predictions: list[float],
    targets: list[float],
    timestamps: list[int],
    lengths: list[int],
    valid: list[bool] | None = None,
) -> list[float]:
    return pairwise_ordering_scores(
        predictions=torch.tensor(predictions),
        targets=torch.tensor(targets),
        timestamps=torch.tensor(timestamps),
        cumulative_lens=to_cumulative_lens(torch.tensor(lengths)),
        session_gap_seconds=GAP,
        valid=None if valid is None else torch.tensor(valid),
    ).tolist()


@pytest.mark.parametrize(
    "predictions, expected", [([0.1, 0.9], 1.0), ([0.9, 0.1], 0.0), ([0.5, 0.5], 0.5)]
)
def test_a_pair_is_scored_by_how_the_model_ordered_it(
    predictions: list[float], expected: float
) -> None:
    assert _scores(predictions, [0.0, 1.0], [0, 1], [2]) == [expected]


def test_events_of_two_users_are_never_compared() -> None:
    assert _scores([0.9, 0.1], [0.0, 1.0], [0, 1], [1, 1]) == []


def test_a_pair_further_apart_than_the_session_gap_is_not_compared() -> None:
    assert _scores([0.9, 0.1], [0.0, 1.0], [0, GAP + 1], [2]) == []


def test_a_pair_the_target_ranks_equal_is_not_compared() -> None:
    assert _scores([0.9, 0.1], [1.0, 1.0], [0, 1], [2]) == []


def test_only_adjacent_events_are_compared() -> None:
    assert _scores([0.9, 0.1, 0.5], [0.0, 0.0, 1.0], [0, 1, 2], [3]) == [1.0]


def test_an_event_the_metric_cannot_score_leaves_its_neighbours_adjacent() -> None:
    scores = _scores(
        [0.1, 9.9, 0.9],
        [0.0, 0.0, 1.0],
        [0, 1, 2],
        [3],
        valid=[True, False, True],
    )

    assert scores == [1.0]


def test_a_batch_with_nothing_comparable_yields_no_pairs() -> None:
    assert _scores([0.9], [1.0], [0], [1]) == []


class _ScoreStub(nn.Module):
    def __init__(self, scores: list[list[float]]) -> None:
        super().__init__()
        self.scores = scores
        self.unused = nn.Parameter(torch.zeros(1))
        self._batch_index = 0

    def forward(self, batch: dict) -> dict[str, FeatureValues]:
        scores = torch.tensor(self.scores[self._batch_index]).unsqueeze(-1)
        self._batch_index += 1
        return {PREDICTION_COLUMN: FeatureValues(scores, batch["cumulative_lens"])}


def _batch(targets: list[float], timestamps: list[int], lengths: list[int]) -> dict:
    offsets = torch.arange(len(targets) + 1)
    return {
        "int_columns": {
            MASK_COLUMN: FeatureValues(
                torch.ones(len(targets), dtype=torch.int64), offsets
            )
        },
        "float_columns": {TARGET_COLUMN: FeatureValues(torch.tensor(targets), offsets)},
        "timestamp": torch.tensor(timestamps),
        "cumulative_lens": to_cumulative_lens(torch.tensor(lengths)),
    }


def _callback(model: nn.Module, batches: list[dict]) -> PairwiseAccuracyCallback:
    return PairwiseAccuracyCallback(
        model=model,
        loader=batches,
        targets=[
            PairwiseTarget(
                name="like",
                prediction_column=PREDICTION_COLUMN,
                target_column=TARGET_COLUMN,
                mask_column=MASK_COLUMN,
            )
        ],
        session_gap_seconds=GAP,
    )


def test_it_averages_over_pairs_rather_than_over_batches() -> None:
    # Three comparable pairs in the first batch, one in the second: 2/3 and 0
    # per batch, but 2 correct pairs out of 4 overall.
    batches = [
        _batch([0.0, 1.0, 0.0, 1.0], [0, 1, 2, 3], [4]),
        _batch([0.0, 1.0], [0, 1], [2]),
    ]
    model = _ScoreStub([[0.1, 0.9, 0.1, 0.0], [0.9, 0.1]])
    state: dict = {}

    _callback(model, batches).on_epoch_end(state)

    assert state[EXTRA_METRICS]["epoch/val_pairwise"]["like"] == pytest.approx(0.5)


def test_a_target_with_no_comparable_pair_is_not_reported() -> None:
    model = _ScoreStub([[0.9, 0.1]])
    state: dict = {}

    _callback(model, [_batch([1.0, 1.0], [0, 1], [2])]).on_epoch_end(state)

    assert EXTRA_METRICS not in state
