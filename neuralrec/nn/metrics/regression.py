import torch

from .base import Metric


class RMSE(Metric):
    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        mse = torch.mean((predictions - targets) ** 2)
        return torch.sqrt(mse)

    @property
    def name(self) -> str:
        return "rmse"


class R2Score(Metric):
    """R² (coefficient of determination): 1 - SS_res/SS_tot"""

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)

        if ss_tot == 0:
            return torch.tensor(0.0, device=predictions.device)

        return 1 - (ss_res / ss_tot)

    @property
    def name(self) -> str:
        return "r2"
