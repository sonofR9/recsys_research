from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    @abstractmethod
    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name for logging."""
        pass
