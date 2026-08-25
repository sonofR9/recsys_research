import math

import torch

from neuralrec.nn.metrics import LogLikelihoodOfPrediction


def _logit(probability: float) -> float:
    return math.log(probability / (1 - probability))


class TestLogLikelihoodOfPrediction:
    def test_weights_are_optional(self) -> None:
        metric = LogLikelihoodOfPrediction()
        predictions = torch.tensor([2.0, -1.0, 0.5, -3.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

        assert metric(predictions, targets) == metric(
            predictions, targets, torch.ones_like(targets)
        )

    def test_the_constant_baseline_scores_zero(self) -> None:
        # The baseline is the best constant prediction, so predicting exactly
        # the positive rate must gain nothing over it.
        metric = LogLikelihoodOfPrediction()
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        predictions = torch.full_like(targets, _logit(1 / 3))

        assert metric(predictions, targets).abs() < 1e-6

    def test_a_better_than_baseline_prediction_is_negative(self) -> None:
        metric = LogLikelihoodOfPrediction()
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        confident = torch.tensor([5.0, 5.0, -5.0, -5.0])

        assert metric(confident, targets) < 0

    def test_no_positives_scores_zero(self) -> None:
        metric = LogLikelihoodOfPrediction()
        targets = torch.zeros(4)

        assert metric(torch.tensor([1.0, 2.0, 3.0, 4.0]), targets) == 0.0
