from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


AggregationType = Literal["mean", "sum", "none"]


class RecallAtK(nn.Module):
    def __init__(
        self,
        k: int = 100,
        aggregation: AggregationType = "mean",
    ) -> None:
        super().__init__()
        self.k = k
        self.aggregation = aggregation

    def forward(
        self,
        user_ids: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        unique_users = user_ids.unique()
        recalls = []

        for user_id in unique_users:
            mask = user_ids == user_id
            user_scores = scores[mask]
            user_labels = labels[mask]

            n_positives = user_labels.sum().item()
            if n_positives == 0:
                continue

            top_k_indices = user_scores.topk(min(self.k, len(user_scores))).indices
            hits = user_labels[top_k_indices].sum().item()
            recalls.append(hits / n_positives)

        if not recalls:
            return torch.tensor(0.0, device=scores.device)

        recalls_tensor = torch.tensor(recalls, device=scores.device)

        if self.aggregation == "mean":
            return recalls_tensor.mean()
        if self.aggregation == "sum":
            return recalls_tensor.sum()
        if self.aggregation == "none":
            return recalls_tensor
        assert False, "unreachable"
