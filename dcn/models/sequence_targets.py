from abc import abstractmethod
from typing import Any, NamedTuple

import torch

from dcn.data.packed import ragged_positions
from torch import nn


class TargetPairs(NamedTuple):
    query_repr: torch.Tensor
    positive_repr: torch.Tensor
    positive_ids: torch.Tensor
    group_sizes: torch.Tensor


class _TokenLayout(NamedTuple):
    sequence_of_token: torch.Tensor
    sequence_end: torch.Tensor
    positions: torch.Tensor
    next_rank: torch.Tensor

    @property
    def last_rank(self) -> int:
        return self.positions.shape[0] - 1

    def at_rank(self, ranks: torch.Tensor) -> torch.Tensor:
        return self.positions[ranks.clamp(max=self.last_rank)]


class SequenceTargets(nn.Module):
    """Picks the (query, positive) pairs a packed batch is trained on."""

    @abstractmethod
    def forward(self, output: dict[str, Any]) -> TargetPairs: ...

    @staticmethod
    def _layout(output: dict[str, Any]) -> _TokenLayout:
        """Where each token sits, and which positions may answer it."""
        lengths = output["lengths"]
        device = lengths.device
        total = int(output["item_ids"].shape[0])
        token_indices = torch.arange(total, device=device)
        sequence_of_token, _ = ragged_positions(lengths, total)

        is_target = output.get("is_target")
        positions = (
            token_indices if is_target is None else is_target.nonzero(as_tuple=True)[0]
        )
        return _TokenLayout(
            sequence_of_token=sequence_of_token,
            sequence_end=torch.cumsum(lengths, dim=0)[sequence_of_token],
            positions=positions,
            next_rank=torch.searchsorted(positions, token_indices, right=True),
        )

    @staticmethod
    def _empty(output: dict[str, Any]) -> TargetPairs:
        item_ids = output["item_ids"]
        return TargetPairs(
            output["query_repr"][:0],
            output["item_repr"][:0],
            item_ids[:0],
            item_ids.new_zeros(0),
        )

    @staticmethod
    def _pairs(
        output: dict[str, Any],
        layout: _TokenLayout,
        queries: torch.Tensor,
        positives: torch.Tensor,
    ) -> TargetPairs:
        group_sizes = torch.bincount(
            layout.sequence_of_token[queries], minlength=output["lengths"].shape[0]
        )
        return TargetPairs(
            output["query_repr"][queries],
            output["item_repr"][positives],
            output["item_ids"][positives],
            group_sizes[group_sizes > 0],
        )


class NextItemTargets(SequenceTargets):
    """Pairs every token of a packed batch with the next target of its sequence."""

    def forward(self, output: dict[str, Any]) -> TargetPairs:
        layout = self._layout(output)
        if layout.positions.numel() == 0:
            return self._empty(output)

        next_position = layout.at_rank(layout.next_rank)
        has_next = (layout.next_rank <= layout.last_rank) & (
            next_position < layout.sequence_end
        )
        if output.get("is_query") is not None:
            has_next &= output["is_query"]

        queries = has_next.nonzero(as_tuple=True)[0]
        return self._pairs(output, layout, queries, next_position[queries])


class TimeWindowTargets(SequenceTargets):
    """Pairs a token with one target drawn from the window that follows it.

    Anything liked within ``window_seconds`` counts, one candidate sampled per
    token. ``lookahead`` counts candidates, not tokens, so an event costing
    several tokens keeps the window's reach.
    """

    def __init__(self, window_seconds: float, lookahead: int = 32):
        super().__init__()
        self.window_seconds = window_seconds
        self.lookahead = lookahead

    def forward(self, output: dict[str, Any]) -> TargetPairs:
        layout = self._layout(output)
        if layout.positions.numel() == 0:
            return self._empty(output)

        timestamps = output["timestamps"]
        ranks = layout.next_rank.unsqueeze(1) + torch.arange(
            self.lookahead, device=timestamps.device
        )
        candidates = layout.at_rank(ranks)
        inside = (
            (ranks <= layout.last_rank)
            & (candidates < layout.sequence_end.unsqueeze(1))
            & (timestamps[candidates] <= timestamps.unsqueeze(1) + self.window_seconds)
        )

        queries = inside.any(dim=1).nonzero(as_tuple=True)[0]
        if queries.numel() == 0:
            return self._empty(output)
        chosen = torch.multinomial(inside[queries].float(), num_samples=1).squeeze(1)
        return self._pairs(output, layout, queries, candidates[queries, chosen])
