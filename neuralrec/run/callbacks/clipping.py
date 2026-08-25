from __future__ import annotations

from typing import Any

import torch
from neuralrec.run.callbacks.base import Callback


class GradientNormClippingCallback(Callback):
    def __init__(self, max_norm: float) -> None:
        self._max_norm = max_norm

    def on_before_optimizer_step(self, state: dict[str, Any]) -> None:
        assert "model" in state
        model = state["model"]

        torch.nn.utils.clip_grad_norm_(model.parameters(), self._max_norm)
