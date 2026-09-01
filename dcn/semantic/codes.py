from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

PADDING_TOKEN = 0
BEGINNING_TOKEN = 1
_NUM_SPECIAL_TOKENS = 2


@dataclass(frozen=True)
class SemanticVocabulary:
    """Flat token ids for ``(level, code)`` pairs."""

    codes_per_level: tuple[int, ...]

    padding_token: int = PADDING_TOKEN
    beginning_token: int = BEGINNING_TOKEN
    num_special_tokens: int = _NUM_SPECIAL_TOKENS

    @property
    def num_levels(self) -> int:
        return len(self.codes_per_level)

    @property
    def size(self) -> int:
        return self.num_special_tokens + sum(self.codes_per_level)

    def level_range(self, level: int) -> tuple[int, int]:
        """``[first, last)`` token ids belonging to ``level``."""
        first = self.num_special_tokens + sum(self.codes_per_level[:level])
        return first, first + self.codes_per_level[level]

    @property
    def level_offsets(self) -> torch.Tensor:
        return torch.tensor(
            [self.level_range(level)[0] for level in range(self.num_levels)]
        )

    def tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """``[..., levels]`` codes to their token ids."""
        assert codes.shape[-1] == self.num_levels, (
            f"expected {self.num_levels} levels, got {codes.shape[-1]}"
        )
        return codes + self.level_offsets.to(codes.device)


@dataclass(frozen=True)
class SemanticCodes:
    """The code tuple assigned to each item."""

    item_ids: torch.Tensor
    codes: torch.Tensor
    codes_per_level: tuple[int, ...]

    def __post_init__(self) -> None:
        assert self.codes.ndim == 2 and len(self.codes) == len(self.item_ids), (
            f"expected one code tuple per item, got {tuple(self.codes.shape)}"
            f" for {len(self.item_ids)} items"
        )
        assert self.codes.shape[1] == len(self.codes_per_level), (
            f"{self.codes.shape[1]} levels but {len(self.codes_per_level)} widths"
        )
        widths = torch.tensor(self.codes_per_level)
        assert len(self.codes) == 0 or bool(
            ((self.codes >= 0) & (self.codes < widths)).all()
        ), (
            "a code outside its level's width silently becomes another level's"
            f" token: widths {self.codes_per_level}, maxima"
            f" {self.codes.max(dim=0).values.tolist()}"
        )

    @classmethod
    def with_collision_suffix(
        cls, item_ids: torch.Tensor, codes: torch.Tensor, num_codes: int
    ) -> SemanticCodes:
        """Append a level that separates items sharing a code tuple.

        Suffixes start at 1: the all-zero tuple is row 0 of ``lookup_table``,
        the unknown item.
        """
        assert len(codes) > 0, "cannot assign semantic ids to an empty catalog"
        _, inverse = torch.unique(codes, dim=0, return_inverse=True)
        order = torch.argsort(inverse, stable=True)
        ranks = torch.empty(len(codes), dtype=torch.int64, device=codes.device)
        group_starts = torch.searchsorted(inverse[order], inverse[order])
        ranks[order] = torch.arange(len(codes), device=codes.device) - group_starts + 1
        return cls(
            item_ids=item_ids,
            codes=torch.cat([codes, ranks.unsqueeze(1)], dim=1),
            codes_per_level=(*[num_codes] * codes.shape[1], int(ranks.max()) + 1),
        )

    @classmethod
    def without_collision_suffix(
        cls, item_ids: torch.Tensor, codes: torch.Tensor, num_codes: int
    ) -> SemanticCodes:
        return cls(
            item_ids=item_ids,
            codes=codes,
            codes_per_level=tuple([num_codes] * codes.shape[1]),
        )

    @property
    def num_levels(self) -> int:
        return self.codes.shape[1]

    @property
    def vocabulary(self) -> SemanticVocabulary:
        return SemanticVocabulary(self.codes_per_level)

    def lookup_table(self, num_items: int) -> torch.Tensor:
        """``[num_items + 1, levels]`` rows indexed by item id."""
        table = torch.zeros(num_items + 1, self.num_levels, dtype=torch.int64)
        table[self.item_ids] = self.codes
        return table

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "item_ids": self.item_ids.cpu(),
                "codes": self.codes.cpu(),
                "codes_per_level": list(self.codes_per_level),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> SemanticCodes:
        stored = torch.load(path, weights_only=True)
        return cls(
            item_ids=stored["item_ids"],
            codes=stored["codes"],
            codes_per_level=tuple(stored["codes_per_level"]),
        )
