from functools import partial
from typing import Any

import torch
from torch import nn

from dcn.data.features import FeatureValues
from dcn.models.criterions import (
    CriterionSpec,
    MultiCriterion,
    TargetExtractionWrapper,
)
from dcn.models.loss_wrapper import LossWrapper
from dcn.tests.helpers import scalar_feature
from neuralrec.utils import LOSS_DENOMINATOR

_extract_like = partial(
    TargetExtractionWrapper, prediction_column="like", target_column="target_like"
)
_extract_listen = partial(
    TargetExtractionWrapper,
    prediction_column="listen",
    target_column="target_listen",
    mask_column="listen_mask",
)


def _yambda_criterion(
    like_weight: float = 1.0, listen_weight: float = 1.0
) -> MultiCriterion:
    return MultiCriterion(
        [
            CriterionSpec("like", _extract_like(nn.BCEWithLogitsLoss()), like_weight),
            CriterionSpec("listen", _extract_listen(nn.MSELoss()), listen_weight),
        ]
    )


class _StubModel(nn.Module):
    def __init__(self, predictions: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._marker = nn.Parameter(torch.zeros(1))
        self._predictions = predictions

    def forward(self, batch: dict[str, Any]) -> dict[str, FeatureValues]:
        return {name: scalar_feature(pred) for name, pred in self._predictions.items()}


def _make_yambda_batch(
    listen_target: torch.Tensor,
    listen_mask: torch.Tensor,
) -> dict[str, Any]:
    batch_size = listen_target.shape[0]
    return {
        "int_columns": {"listen_mask": scalar_feature(listen_mask)},
        "float_columns": {
            "target_like": scalar_feature(torch.zeros(batch_size)),
            "target_listen": scalar_feature(listen_target),
        },
    }


def test_listen_loss_ignores_masked_out_targets() -> None:
    batch_size = 8
    predictions = {
        "like": torch.zeros(batch_size, 1),
        "listen": torch.zeros(batch_size, 1),
    }
    wrapper = LossWrapper(_StubModel(predictions), criterion=_yambda_criterion())

    listen_mask = torch.tensor([True, False, True, False, True, False, True, False])
    listen_target_a = torch.tensor([0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.4, 0.5])
    listen_target_b = listen_target_a.clone()
    listen_target_b[~listen_mask] = -42.0

    output_a = wrapper(_make_yambda_batch(listen_target_a, listen_mask))
    output_b = wrapper(_make_yambda_batch(listen_target_b, listen_mask))

    assert torch.equal(output_a["listen_loss"], output_b["listen_loss"])
    assert torch.equal(output_a["loss"], output_b["loss"])


def test_loss_keys_per_task() -> None:
    batch_size = 4
    predictions = {
        "like": torch.zeros(batch_size, 1),
        "listen": torch.zeros(batch_size, 1),
    }
    wrapper = LossWrapper(_StubModel(predictions), criterion=_yambda_criterion())

    output = wrapper(
        _make_yambda_batch(
            torch.zeros(batch_size), torch.ones(batch_size, dtype=torch.bool)
        )
    )

    assert {"loss", "like_loss", "listen_loss", "like_pred", "listen_pred"}.issubset(
        output.keys()
    )
    assert LOSS_DENOMINATOR not in output


def test_metrics_are_reported_under_their_name() -> None:
    batch_size = 4
    predictions = {
        "like": torch.zeros(batch_size, 1),
        "listen": torch.zeros(batch_size, 1),
    }
    metric = _extract_like(nn.L1Loss())
    wrapper = LossWrapper(
        _StubModel(predictions), criterion=_yambda_criterion(), metrics=[metric]
    )

    output = wrapper(
        _make_yambda_batch(
            torch.zeros(batch_size), torch.ones(batch_size, dtype=torch.bool)
        )
    )

    assert metric.name in output


def test_weighted_total_loss() -> None:
    batch_size = 4
    predictions = {
        "like": torch.zeros(batch_size, 1),
        "listen": torch.ones(batch_size, 1),
    }
    criterion = MultiCriterion(
        [
            CriterionSpec(
                "like",
                TargetExtractionWrapper(
                    nn.BCEWithLogitsLoss(),
                    prediction_column="like",
                    target_column="target_like",
                ),
                2.0,
            ),
            CriterionSpec(
                "listen",
                TargetExtractionWrapper(
                    nn.MSELoss(),
                    prediction_column="listen",
                    target_column="target_listen",
                ),
                3.0,
            ),
        ]
    )
    wrapper = LossWrapper(_StubModel(predictions), criterion=criterion)

    batch: dict[str, Any] = {
        "int_columns": {},
        "float_columns": {
            "target_like": scalar_feature(torch.zeros(batch_size)),
            "target_listen": scalar_feature(torch.zeros(batch_size)),
        },
    }

    output = wrapper(batch)

    expected = 2.0 * output["like_loss"] + 3.0 * output["listen_loss"]
    assert torch.allclose(output["loss"], expected)


def test_single_task() -> None:
    batch_size = 4
    predictions = {"rating": torch.zeros(batch_size, 1)}
    criterion = MultiCriterion(
        [
            CriterionSpec(
                "rating",
                TargetExtractionWrapper(
                    nn.MSELoss(),
                    prediction_column="rating",
                    target_column="target_rating",
                ),
                1.0,
            )
        ]
    )
    wrapper = LossWrapper(_StubModel(predictions), criterion=criterion)

    batch: dict[str, Any] = {
        "int_columns": {},
        "float_columns": {"target_rating": scalar_feature(torch.zeros(batch_size))},
    }

    output = wrapper(batch)

    assert "loss" in output
    assert "rating_loss" in output
    assert "rating_pred" in output
    assert torch.equal(output["loss"], output["rating_loss"])
