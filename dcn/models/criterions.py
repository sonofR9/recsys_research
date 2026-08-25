from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from neuralrec.nn.metrics import Metric


class TargetExtractionWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module | Metric,
        prediction_column: str,
        target_column: str,
        mask_column: str | None = None,
    ):
        super().__init__()
        self.module = module
        self.prediction_column = prediction_column
        self.target_column = target_column
        self.mask_column = mask_column

    @property
    def name(self) -> str:
        inner = getattr(self.module, "name", None) or type(self.module).__name__
        return f"{self.prediction_column}_{inner}"

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        pred = batch["predictions"][self.prediction_column].values.squeeze(-1)
        target = batch["float_columns"][self.target_column].dense()
        if self.mask_column is not None:
            mask = batch["int_columns"][self.mask_column].dense().bool()
            pred = pred[mask]
            target = target[mask]
        return self.module(pred, target)

    def mean_denominator(self, batch: dict[str, Any]) -> int:
        reduction = getattr(self.module, "reduction", None)
        if reduction != "mean":
            raise ValueError(
                f"gradient accumulation requires mean reduction, got {reduction!r}"
            )
        target = batch["float_columns"][self.target_column].dense()
        if self.mask_column is not None:
            mask = batch["int_columns"][self.mask_column].dense().bool()
            target = target[mask]
        return target.numel()


@dataclass
class CriterionSpec:
    name: str
    criterion: nn.Module
    weight: float


class MultiCriterion(nn.Module):
    """Weighted sum of named sub-criterions over the whole batch."""

    def __init__(self, components: list[CriterionSpec]):
        super().__init__()
        assert components, "MultiCriterion requires at least one criterion"

        self.names = [spec.name for spec in components]
        self.criterions = nn.ModuleDict(
            {spec.name: spec.criterion for spec in components}
        )
        self.weights = {spec.name: float(spec.weight) for spec in components}

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        # FIXME(sonofr): you also need to support softmax losses etc. in this interface. Take it into account.
        components: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name in self.names:
            value = self.criterions[name](batch)
            components[name] = value
            term = self.weights[name] * value
            total = term if total is None else total + term
        assert total is not None
        return {"loss": total, **components}

    def accumulation_spec(self, batch: dict[str, Any]) -> dict[str, tuple[float, int]]:
        spec = {}
        for name in self.names:
            criterion = self.criterions[name]
            if not isinstance(criterion, TargetExtractionWrapper):
                raise ValueError(
                    "gradient accumulation requires TargetExtractionWrapper criteria"
                )
            spec[f"{name}_loss"] = (
                self.weights[name],
                criterion.mean_denominator(batch),
            )
        return spec
