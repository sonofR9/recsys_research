from typing import Any

import torch
import torch.nn as nn

from dcn.models.criterions import TargetExtractionWrapper


class LossWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        metrics: list[TargetExtractionWrapper] | None = None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.metrics = nn.ModuleList(metrics or [])

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        predictions = self.model(batch)
        extraction_batch = {**batch, "predictions": predictions}

        criterion_out = self.criterion(extraction_batch)

        result: dict[str, torch.Tensor] = {"loss": criterion_out["loss"]}
        for name, value in criterion_out.items():
            if name != "loss":
                result[f"{name}_loss"] = value

        for head_name, pred in predictions.items():
            result[f"{head_name}_pred"] = pred

        for metric in self.metrics:
            result[metric.name] = metric(extraction_batch)

        return result

    def accumulation_spec(self, batch: dict[str, Any]) -> dict[str, tuple[float, int]]:
        if not hasattr(self.criterion, "accumulation_spec"):
            raise ValueError("gradient accumulation is unsupported for this criterion")
        return self.criterion.accumulation_spec(batch)
