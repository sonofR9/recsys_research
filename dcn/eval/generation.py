"""Recall of what a generative recommender actually generates."""

from typing import Iterable, Sequence

import torch
from torch import nn

from dcn.data.packed import ragged_positions, split_last_events
from dcn.eval.base import ScoredBatchesCallback
from dcn.models import SemanticIdConstraint

_BEAM_BATCHES = 16


class GenerationRecallCallback(ScoredBatchesCallback):
    """Recall@k of beam search over held-out histories.

    Each validation sequence gives up its last event; that event is the answer.
    ``sid_recall`` counts the code tuple, ``recall`` the items carrying it.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        loader: Iterable[dict],
        constraint: SemanticIdConstraint,
        item_id_column: str,
        num_levels: int | None = None,
        beam_width: int = 10,
        ks: Sequence[int] | None = None,
        dtype: torch.dtype = torch.float32,
        max_batches: int = _BEAM_BATCHES,
        every_n_epochs: int = 1,
        prefix: str = "epoch/val_beam",
    ) -> None:
        super().__init__(
            model=model,
            loader=loader,
            prefix=prefix,
            dtype=dtype,
            # Beam search cannot afford a whole validation day; the loader
            # shuffles, so these are different users every epoch.
            max_batches=max_batches,
            every_n_epochs=every_n_epochs,
        )
        self.ks = tuple(ks if ks is not None else (1, beam_width))
        assert beam_width >= max(self.ks), "cannot report recall past the last beam"
        self.constraint = constraint
        self.item_id_column = item_id_column
        self.num_levels = num_levels or constraint.tokens_per_event
        self.beam_width = beam_width

    def _match_positions(
        self, prefixes: torch.Tensor, answer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Where the answer turns up: in which beam's bucket, and at which rank
        of the beam-ordered items. Both are ``max(ks)`` when it never does."""
        num_sequences, beam_width = prefixes.shape[:2]
        items, cumulative_lens = self.constraint.items_under(prefixes.flatten(0, 1))
        lengths = cumulative_lens.diff()

        rows, _ = ragged_positions(lengths, items.shape[0])
        sequences, ranks = ragged_positions(
            lengths.view(num_sequences, beam_width).sum(dim=1), items.shape[0]
        )
        matched = items == answer[sequences]

        never = max(self.ks)
        return (
            _least_per_sequence(
                torch.where(matched, rows % beam_width, never),
                sequences,
                num_sequences,
                never,
            ),
            _least_per_sequence(
                torch.where(matched, ranks, never), sequences, num_sequences, never
            ),
        )

    def _batch_scores(self, batch: dict) -> dict[str, torch.Tensor]:
        self.constraint.to(self._device)
        history, tail = split_last_events(batch)
        if history["cumulative_lens"].shape[0] <= 1:
            return {}

        generated, _ = self.model.generate(
            history, beam_width=self.beam_width, num_slots=self.num_levels
        )
        answer = tail["int_columns"][self.item_id_column].dense()

        beam, rank = self._match_positions(generated, answer)
        hits = {}
        for k in self.ks:
            hits[f"recall@{k}"] = (rank < k).float()
            hits[f"sid_recall@{k}"] = (beam < k).float()
        for level in range(self.num_levels - 1):
            beam, _ = self._match_positions(generated[..., : level + 1], answer)
            for k in self.ks:
                hits[f"level{level}_sid_recall@{k}"] = (beam < k).float()
        return hits


def _least_per_sequence(
    values: torch.Tensor, sequences: torch.Tensor, num_sequences: int, fill: int
) -> torch.Tensor:
    least = torch.full((num_sequences,), fill, dtype=values.dtype, device=values.device)
    return least.scatter_reduce_(0, sequences, values, reduce="amin")
