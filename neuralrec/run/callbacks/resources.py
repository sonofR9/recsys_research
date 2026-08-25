from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from neuralrec.run.callbacks.base import Callback
from neuralrec.utils import add_metrics

_BYTES_IN_GB = 1024**3


class ResourceUsageCallback(Callback):
    """What a run costs to train, next to what it scores."""

    def __init__(
        self,
        *,
        model: nn.Module,
        embedding_parameters: Sequence[nn.Parameter] = (),
        prefix: str = "resources",
    ) -> None:
        self.model = model
        self.embedding_ids = {id(parameter) for parameter in embedding_parameters}
        self.prefix = prefix

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        parameters = list(self.model.parameters())
        embedding = [p for p in parameters if id(p) in self.embedding_ids]
        usage = {
            "params_total": float(sum(p.numel() for p in parameters)),
            "params_trainable": float(
                sum(p.numel() for p in parameters if p.requires_grad)
            ),
            "params_embedding": float(sum(p.numel() for p in embedding)),
            "params_deep": float(
                sum(p.numel() for p in parameters if id(p) not in self.embedding_ids)
            ),
        }
        if torch.cuda.is_available():
            usage["peak_memory_gb"] = torch.cuda.max_memory_allocated() / _BYTES_IN_GB
        add_metrics(state, self.prefix, usage)
