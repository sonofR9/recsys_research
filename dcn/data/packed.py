"""Index arithmetic for variable-length sequences stored back to back."""

from __future__ import annotations

import torch


def to_cumulative_lens(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])


def ragged_positions(
    lengths: torch.Tensor, total: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row index and within-row rank of every element of a packed batch.

    ``total`` is the packed length, which every caller already holds as a
    tensor shape. Deriving it here instead -- what ``repeat_interleave`` over a
    tensor of repeats does -- reads a device tensor from the host, and that
    drains the CUDA queue on every call.

    The search runs over the row ends rather than over a slice of
    ``to_cumulative_lens``: inductor loses a sliced tensor's offset here and
    silently searches the wrong window (torch 2.11, CUDA only).
    """
    ends = lengths.cumsum(0)
    starts = ends - lengths
    index = torch.arange(total, device=lengths.device)
    rows = torch.searchsorted(ends, index, right=True).clamp_max(lengths.shape[0] - 1)
    return rows, index - starts[rows]


def repeat_sequences(
    values: torch.Tensor, cumulative_lens: torch.Tensor, repeats: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequence-major: a sequence's ``repeats`` copies land next to each other.

    Beam search reshapes on that layout to map a beam back to its sequence.
    """
    lengths = cumulative_lens.diff()
    repeated_lengths = lengths.repeat_interleave(repeats)
    rows, ranks = ragged_positions(repeated_lengths, values.shape[0] * repeats)
    starts = cumulative_lens[:-1].repeat_interleave(repeats)
    return values[starts[rows] + ranks], to_cumulative_lens(repeated_lengths)


def tail_positions(
    lengths: torch.Tensor, count: int, total: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per packed position: is it in a sequence longer than ``count``, and is it
    one of that sequence's last ``count``."""
    of_event, within_sequence = ragged_positions(lengths, total)
    return (
        (lengths > count)[of_event],
        within_sequence >= lengths[of_event] - count,
    )


def split_last_events(batch: dict, count: int = 1) -> tuple[dict, dict]:
    """Split every sequence into all but its last ``count`` events, and those."""
    lengths = batch["cumulative_lens"].diff()
    used, is_tail = tail_positions(lengths, count, batch["timestamp"].shape[0])
    kept_lengths = lengths[lengths > count]
    return (
        _select_events(batch, used & ~is_tail, kept_lengths - count),
        _select_events(batch, used & is_tail, torch.full_like(kept_lengths, count)),
    )


def _select_events(batch: dict, events: torch.Tensor, lengths: torch.Tensor) -> dict:
    return {
        **batch,
        "int_columns": {
            name: column.select(events) for name, column in batch["int_columns"].items()
        },
        "float_columns": {
            name: column.select(events)
            for name, column in batch["float_columns"].items()
        },
        "cumulative_lens": to_cumulative_lens(lengths),
        "timestamp": batch["timestamp"][events],
    }


def append_to_sequences(
    values: torch.Tensor, cumulative_lens: torch.Tensor, extra: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``extra`` is ``[sequences, k]``: every sequence grows by the same ``k``."""
    lengths = cumulative_lens.diff()
    grown_lengths = lengths + extra.shape[1]
    total = values.shape[0] + extra.shape[0] * extra.shape[1]
    rows, ranks = ragged_positions(grown_lengths, total)

    is_extra = ranks >= lengths[rows]
    grown = values.new_empty((total, *values.shape[1:]))
    grown[~is_extra] = values
    grown[is_extra] = extra.flatten(0, 1).to(values.dtype)
    return grown, to_cumulative_lens(grown_lengths)
