"""Models that write an event out as a fixed number of tokens."""

from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcn.data.packed import (
    append_to_sequences,
    ragged_positions,
    tail_positions,
    repeat_sequences,
    to_cumulative_lens,
)
from dcn.nn.transformer import TransformerDecoder
from dcn.nn.types import ModuleWithDim
from neuralrec.utils import LOSS_DENOMINATOR

from .history_tokens import EventTokenizer


class PackedContext(NamedTuple):
    """What a generation step conditions on, one packed sequence per beam."""

    values: torch.Tensor
    cumulative_lens: torch.Tensor

    @property
    def num_sequences(self) -> int:
        return self.cumulative_lens.shape[0] - 1

    def repeat(self, repeats: int) -> PackedContext:
        return PackedContext(*repeat_sequences(*self, repeats))


class TokenConstraint(nn.Module):
    """Which tokens may fill each slot of an event."""

    @property
    @abstractmethod
    def vocabulary_size(self) -> int: ...

    @property
    @abstractmethod
    def tokens_per_event(self) -> int: ...

    @property
    @abstractmethod
    def beginning_token(self) -> int:
        """What a decoder reads before it has written anything."""

    @abstractmethod
    def slot_mask(self, slots: torch.Tensor) -> torch.Tensor:
        """``[slots, vocabulary]``: tokens each slot may hold, whatever precedes."""

    @abstractmethod
    def next_mask(self, slot: int, prefix: torch.Tensor) -> torch.Tensor:
        """``[prefixes, vocabulary]``: tokens that may follow each prefix."""


class TokenDecoder(nn.Module):
    """A model whose output is one token per slot, written left to right."""

    def __init__(
        self,
        tokenizer: EventTokenizer,
        constraint: TokenConstraint,
        hidden_dim: int,
    ):
        super().__init__()
        assert tokenizer.tokens_per_event == constraint.tokens_per_event, (
            f"tokenizer emits {tokenizer.tokens_per_event} tokens per event but the"
            f" constraint describes {constraint.tokens_per_event}"
        )
        self.tokenizer = tokenizer
        self.constraint = constraint
        self.head = nn.Linear(hidden_dim, constraint.vocabulary_size)

    @property
    def tokens_per_event(self) -> int:
        return self.constraint.tokens_per_event

    @abstractmethod
    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """``logits``, the ``targets`` they are scored on, and each target's ``slots``."""

    @abstractmethod
    def _context(self, batch: dict) -> PackedContext: ...

    @abstractmethod
    def _step_hidden(
        self, context: PackedContext, prefix: torch.Tensor
    ) -> torch.Tensor:
        """State that predicts slot ``prefix.shape[1]``, one row per beam."""

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _restrict(self, hidden: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        return self.head(hidden).masked_fill(
            ~self.constraint.slot_mask(slots), -torch.inf
        )

    def _empty_prediction(
        self, hidden_dim_like: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits = self.head(hidden_dim_like.new_zeros(0, self.head.in_features))
        empty = logits.new_zeros(0, dtype=torch.int64)
        return {"logits": logits, "targets": empty, "slots": empty}

    @torch.no_grad()
    def generate(
        self, batch: dict, *, beam_width: int, num_slots: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Beam search over the token sequences the constraint allows."""
        num_slots = num_slots or self.tokens_per_event
        assert 1 <= num_slots <= self.tokens_per_event, (
            f"cannot decode {num_slots} of {self.tokens_per_event} slots"
        )
        vocabulary_size = self.constraint.vocabulary_size
        context = self._context(batch)
        num_sequences = context.num_sequences
        context = context.repeat(beam_width)

        tokens = torch.zeros(
            num_sequences * beam_width, 0, dtype=torch.int64, device=self._device
        )
        scores = torch.full(
            (num_sequences, beam_width), -torch.inf, device=self._device
        )
        scores[:, 0] = 0.0

        for slot in range(num_slots):
            hidden = self._step_hidden(context, tokens)
            slots = torch.full((hidden.shape[0],), slot, device=self._device)
            # Normalised before the mask: an all-forbidden softmax gives a dead beam
            # NaN, which outranks the -inf topk has to skip.
            log_probs = self._restrict(hidden, slots).float().log_softmax(dim=-1)
            log_probs = log_probs.masked_fill(
                ~self.constraint.next_mask(slot, tokens), -torch.inf
            )

            candidates = (scores.view(-1, 1) + log_probs).view(num_sequences, -1)
            scores, chosen = candidates.topk(beam_width, dim=1)

            beams = chosen // vocabulary_size
            tokens = torch.cat(
                [
                    tokens.view(num_sequences, beam_width, -1).gather(
                        1, beams.unsqueeze(-1).expand(-1, -1, tokens.shape[1])
                    ),
                    (chosen % vocabulary_size).unsqueeze(-1),
                ],
                dim=-1,
            ).view(num_sequences * beam_width, -1)

        tokens = tokens.view(num_sequences, beam_width, num_slots)
        return tokens.masked_fill(~scores.isfinite().unsqueeze(-1), -1), scores


class CausalTokenDecoder(TokenDecoder):
    """Causal language model over the tokens of a user's history."""

    def __init__(
        self,
        tokenizer: EventTokenizer,
        sequence_model: ModuleWithDim,
        constraint: TokenConstraint,
    ):
        super().__init__(tokenizer, constraint, sequence_model.out_dim)
        assert tokenizer.out_dim == sequence_model.out_dim, (
            f"tokenizer emits {tokenizer.out_dim}-wide tokens but the sequence"
            f" model reads {sequence_model.out_dim}"
        )
        self.sequence_model = sequence_model

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(batch)
        cumulative_lens = tokens.cumulative_lens
        hidden = self.sequence_model(tokens.embeddings, cumulative_lens)

        device = hidden.device
        has_successor = torch.ones(hidden.shape[0], dtype=torch.bool, device=device)
        has_successor[cumulative_lens[1:] - 1] = False
        if not bool(has_successor.any()):
            return self._empty_prediction(hidden)

        queries = has_successor.nonzero().flatten()
        slots = self._slots(cumulative_lens, tokens.token_ids.shape[0])[queries + 1]
        return {
            "logits": self._restrict(hidden[queries], slots),
            "targets": tokens.token_ids[queries + 1],
            "slots": slots,
        }

    def _slots(self, cumulative_lens: torch.Tensor, total: int) -> torch.Tensor:
        _, within_sequence = ragged_positions(cumulative_lens.diff(), total)
        return within_sequence % self.tokens_per_event

    def _context(self, batch: dict) -> PackedContext:
        tokens = self.tokenizer(batch)
        return PackedContext(tokens.embeddings, tokens.cumulative_lens)

    def _step_hidden(
        self, context: PackedContext, prefix: torch.Tensor
    ) -> torch.Tensor:
        grown, grown_lens = append_to_sequences(
            *context, self.tokenizer.embed_tokens(prefix)
        )
        return self.sequence_model(grown, grown_lens)[grown_lens[1:] - 1]


class Seq2SeqTokenDecoder(TokenDecoder):
    """TIGER: encode the history, then decode the next event's tokens."""

    def __init__(
        self,
        tokenizer: EventTokenizer,
        encoder: ModuleWithDim,
        decoder: TransformerDecoder,
        constraint: TokenConstraint,
    ):
        super().__init__(tokenizer, constraint, decoder.out_dim)
        assert tokenizer.out_dim == encoder.out_dim == decoder.out_dim, (
            "tokens, memory and decoder states all share one width"
        )
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(batch)
        width = self.tokens_per_event
        lengths = tokens.cumulative_lens.diff()

        has_history = lengths > width
        if not bool(has_history.any()):
            return self._empty_prediction(tokens.embeddings)

        used, is_target = tail_positions(lengths, width, tokens.token_ids.shape[0])
        history_lens = to_cumulative_lens((lengths - width)[has_history])
        memory = self.encoder(tokens.embeddings[used & ~is_target], history_lens)
        target_tokens = tokens.token_ids[used & is_target].view(-1, width)
        hidden = self._decode(memory, history_lens, self._teacher_forced(target_tokens))
        slots = torch.arange(width, device=lengths.device).repeat(
            target_tokens.shape[0]
        )
        return {
            "logits": self._restrict(hidden, slots),
            "targets": target_tokens.flatten(),
            "slots": slots,
        }

    def _teacher_forced(self, target_tokens: torch.Tensor) -> torch.Tensor:
        start = self._start(target_tokens.shape[0])
        return torch.cat([start, target_tokens[:, :-1]], dim=1)

    def _start(self, num_rows: int) -> torch.Tensor:
        return torch.full(
            (num_rows, 1), self.constraint.beginning_token, device=self._device
        )

    def _decode(
        self,
        memory: torch.Tensor,
        memory_cumulative_lens: torch.Tensor,
        input_tokens: torch.Tensor,
    ) -> torch.Tensor:
        num_sequences, width = input_tokens.shape
        embeddings = self.tokenizer.embed_tokens(input_tokens).flatten(0, 1)
        cumulative_lens = (
            torch.arange(num_sequences + 1, device=input_tokens.device) * width
        )
        return self.decoder(embeddings, cumulative_lens, memory, memory_cumulative_lens)

    def _context(self, batch: dict) -> PackedContext:
        tokens = self.tokenizer(batch)
        memory = self.encoder(tokens.embeddings, tokens.cumulative_lens)
        return PackedContext(memory, tokens.cumulative_lens)

    def _step_hidden(
        self, context: PackedContext, prefix: torch.Tensor
    ) -> torch.Tensor:
        input_tokens = torch.cat([self._start(prefix.shape[0]), prefix], dim=1)
        hidden = self._decode(*context, input_tokens)
        return hidden.view(prefix.shape[0], input_tokens.shape[1], -1)[:, -1]


class TokenPredictionLoss(nn.Module):
    """Cross entropy over whichever tokens the model chose to predict."""

    def __init__(self, model: TokenDecoder):
        super().__init__()
        self.model = model

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        out = self.model(batch)
        logits, targets, slots = out["logits"].float(), out["targets"], out["slots"]
        # Graph-connected, so backward() still works on an empty batch.
        zero = logits.sum() * 0.0
        if targets.numel() == 0:
            losses = correct = logits.new_zeros(0)
        else:
            losses = F.cross_entropy(logits, targets, reduction="none")
            correct = (logits.argmax(dim=-1) == targets).float()

        result = {
            "loss": _mean_or(losses, zero),
            "accuracy": _mean_or(correct, zero),
            LOSS_DENOMINATOR: targets.numel(),
        }
        for slot in range(self.model.tokens_per_event):
            at_slot = slots == slot
            result[f"slot{slot}_loss"] = _mean_or(losses[at_slot], zero)
            result[f"slot{slot}_accuracy"] = _mean_or(correct[at_slot], zero)
        return result


def _mean_or(values: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    return values.mean() if values.numel() else fallback
