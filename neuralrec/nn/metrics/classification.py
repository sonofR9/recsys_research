import torch
import torch.nn.functional as F

from .base import Metric


class LogLikelihoodOfPrediction(Metric):
    """CatBoost's LogLikelihoodOfPrediction, in loss orientation."""

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if weights is None:
            weights = torch.ones_like(targets)

        positives = (targets * weights).sum()
        if positives == 0:
            return torch.zeros((), device=predictions.device)

        base_rate = (positives / weights.sum()).clamp(min=1e-7, max=1 - 1e-7)
        baseline_logit = torch.logit(base_rate).expand_as(predictions)
        baseline_nll = F.binary_cross_entropy_with_logits(
            baseline_logit, targets, weights, reduction="sum"
        )
        nll = F.binary_cross_entropy_with_logits(
            predictions, targets, weights, reduction="sum"
        )
        return (nll - baseline_nll) / positives

    @property
    def name(self) -> str:
        return "llp"
