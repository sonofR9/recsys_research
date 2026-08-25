from typing import Any, NamedTuple

import numpy as np
import torch

from .packed import ragged_positions, to_cumulative_lens


class FeatureValues(NamedTuple):
    """One categorical feature for N rows: values back to back, offsets per row."""

    values: torch.Tensor
    offsets: torch.Tensor

    def num_rows(self) -> int:
        return self.offsets.shape[0] - 1

    def dense(self) -> torch.Tensor:
        assert self.values.shape[0] == self.num_rows(), (
            "dense() requires exactly one value per row"
        )
        return self.values

    def matrix(self) -> torch.Tensor:
        """``[N, width]`` view for a column whose rows all carry the same count."""
        num_rows = self.num_rows()
        total = self.values.shape[0]
        assert num_rows > 0 and total % num_rows == 0, (
            f"matrix() needs a constant row width, got {total} values in {num_rows} rows"
        )
        # reshape, not view: select() returns non-contiguous values.
        return self.values.reshape(num_rows, total // num_rows)

    def select(self, rows: torch.Tensor) -> "FeatureValues":
        counts = self.offsets.diff()
        kept_counts = counts[rows]
        starts = self.offsets[:-1][rows]
        row_of_value, rank = ragged_positions(kept_counts, int(kept_counts.sum()))
        return FeatureValues(
            values=self.values[starts[row_of_value] + rank],
            offsets=to_cumulative_lens(kept_counts),
        )


def as_sequence(cell: Any) -> list:
    if isinstance(cell, (list, tuple, np.ndarray)):
        return list(cell)
    return [cell]


def ragged_from_rows(per_row: list[list], dtype: torch.dtype) -> FeatureValues:
    offsets = [0]
    flat: list = []
    for cell in per_row:
        flat.extend(cell)
        offsets.append(len(flat))
    return FeatureValues(
        values=torch.tensor(flat, dtype=dtype),
        offsets=torch.tensor(offsets, dtype=torch.int64),
    )
