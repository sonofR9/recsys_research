from __future__ import annotations

from math import prod

import torch
import torch.nn as nn

from dcn.data.packed import ragged_positions, to_cumulative_lens

from .codes import SemanticCodes


class CodeTrie(nn.Module):
    """Which codes may follow a prefix, and which items a prefix names."""

    def __init__(self, codes: SemanticCodes):
        super().__init__()
        self.codes_per_level = codes.codes_per_level
        self.num_levels = codes.num_levels

        radix = torch.tensor(codes.codes_per_level, dtype=torch.int64)
        self.register_buffer("radix", radix)

        order = self._sort_lexicographically(codes.codes)
        sorted_codes = codes.codes[order]
        self.register_buffer("item_id_by_row", codes.item_ids[order])
        self.register_buffer("full_keys", self._keys(sorted_codes, self.num_levels))

        for level in range(self.num_levels):
            parent_keys = self._keys(sorted_codes, level)
            edges = torch.stack([parent_keys, sorted_codes[:, level]], dim=1)
            unique_edges = torch.unique(edges, dim=0)
            parents, counts = torch.unique_consecutive(
                unique_edges[:, 0], return_counts=True
            )
            self.register_buffer(f"parents_{level}", parents)
            self.register_buffer(
                f"child_offsets_{level}",
                torch.cat([counts.new_zeros(1), counts.cumsum(0)]),
            )
            self.register_buffer(f"children_{level}", unique_edges[:, 1])

    @staticmethod
    def _sort_lexicographically(codes: torch.Tensor) -> torch.Tensor:
        order = torch.arange(len(codes))
        for level in reversed(range(codes.shape[1])):
            order = order[torch.argsort(codes[order, level], stable=True)]
        return order

    def _keys(self, codes: torch.Tensor, length: int) -> torch.Tensor:
        """Positional encoding of a prefix, so a prefix is one sortable integer."""
        key = torch.zeros(len(codes), dtype=torch.int64, device=codes.device)
        for level in range(length):
            key = key * self.radix[level] + codes[:, level]
        return key

    def allowed_mask(self, level: int, prefix_codes: torch.Tensor) -> torch.Tensor:
        parents: torch.Tensor = getattr(self, f"parents_{level}")
        offsets: torch.Tensor = getattr(self, f"child_offsets_{level}")
        children: torch.Tensor = getattr(self, f"children_{level}")

        keys = self._keys(prefix_codes, level)
        rows_in_trie = torch.searchsorted(parents, keys).clamp(max=len(parents) - 1)
        known = parents[rows_in_trie] == keys
        starts = offsets[rows_in_trie]
        counts = torch.where(known, offsets[rows_in_trie + 1] - starts, 0)

        rows, ranks = ragged_positions(counts, int(counts.sum()))
        mask = torch.zeros(
            len(prefix_codes),
            self.codes_per_level[level],
            dtype=torch.bool,
            device=prefix_codes.device,
        )
        mask[rows, children[starts[rows] + ranks]] = True
        return mask

    def items_under(
        self, prefix_codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every item whose codes start with each prefix, as a packed batch."""
        depth = prefix_codes.shape[1]
        prefix_keys = self.full_keys // prod(self.codes_per_level[depth:])
        keys = self._keys(prefix_codes, depth)

        first = torch.searchsorted(prefix_keys, keys)
        counts = torch.searchsorted(prefix_keys, keys, right=True) - first
        rows, ranks = ragged_positions(counts, int(counts.sum()))
        return self.item_id_by_row[first[rows] + ranks], to_cumulative_lens(counts)
