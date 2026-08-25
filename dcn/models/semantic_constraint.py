"""What makes a token decoder a semantic-id decoder, and nothing else."""

from __future__ import annotations

import torch

from dcn.semantic import CodeTrie, SemanticCodes, SemanticVocabulary

from .token_generation import TokenConstraint


class SemanticIdConstraint(TokenConstraint):
    def __init__(self, codes: SemanticCodes):
        super().__init__()
        self.vocabulary = codes.vocabulary
        self.trie = CodeTrie(codes)
        self.register_buffer("level_mask", _level_masks(self.vocabulary))
        self.register_buffer("level_offsets", self.vocabulary.level_offsets)

    @property
    def vocabulary_size(self) -> int:
        return self.vocabulary.size

    @property
    def tokens_per_event(self) -> int:
        return self.vocabulary.num_levels

    @property
    def beginning_token(self) -> int:
        return self.vocabulary.beginning_token

    def slot_mask(self, slots: torch.Tensor) -> torch.Tensor:
        return self.level_mask[slots]

    def next_mask(self, slot: int, prefix: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(
            prefix.shape[0],
            self.vocabulary_size,
            dtype=torch.bool,
            device=prefix.device,
        )
        first, last = self.vocabulary.level_range(slot)
        mask[:, first:last] = self.trie.allowed_mask(slot, self.codes(prefix))
        return mask

    def codes(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens - self.level_offsets[: tokens.shape[1]]

    def items_under(self, prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.trie.items_under(self.codes(prefix))


def _level_masks(vocabulary: SemanticVocabulary) -> torch.Tensor:
    mask = torch.zeros(vocabulary.num_levels, vocabulary.size, dtype=torch.bool)
    for level in range(vocabulary.num_levels):
        first, last = vocabulary.level_range(level)
        mask[level, first:last] = True
    return mask
