from contextlib import contextmanager
from dataclasses import dataclass, replace
from collections.abc import Iterator

import torch
from torch import nn

from dcn.data.packed import to_cumulative_lens
from dcn.nn.types import ModuleWithDim

from .history_tokens import EndQuerySlots, EventTokenizer, TokenizedHistory


@dataclass(frozen=True)
class InferenceMemoryLesion:
    remove_history: bool = False
    dropped_query_slot: int | None = None


class CrossAttentionRetrievalModel(nn.Module):
    def __init__(
        self,
        tokenizer: EventTokenizer,
        memory_encoder: ModuleWithDim,
        decoder: ModuleWithDim,
        item_embedding: nn.Embedding,
        item_id_column: str,
        query_projection: nn.Linear | None = None,
        query_slots: EndQuerySlots | None = None,
        include_history_memory: bool = False,
        query_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if tokenizer.out_dim != memory_encoder.out_dim:
            raise ValueError("tokenizer and memory encoder widths must match")
        if memory_encoder.out_dim != decoder.out_dim:
            raise ValueError("memory encoder and decoder widths must match")
        if query_projection is None:
            if decoder.out_dim != item_embedding.embedding_dim:
                raise ValueError("unprojected queries and items must share a width")
        elif (
            query_projection.in_features != decoder.out_dim
            or query_projection.out_features != item_embedding.embedding_dim
        ):
            raise ValueError("query projection must map decoder width to item width")
        self.tokenizer = tokenizer
        self.memory_encoder = memory_encoder
        self.decoder = decoder
        self.item_embedding = item_embedding
        self.item_id_column = item_id_column
        self.query_projection = query_projection
        self.query_slots = query_slots
        self.include_history_memory = include_history_memory
        self.query_multiplier = query_multiplier
        if include_history_memory and query_slots is None:
            raise ValueError("history-plus-slots memory requires query slots")
        self.decoder_query = nn.Parameter(torch.zeros(decoder.out_dim))
        self._inference_memory_lesion: InferenceMemoryLesion | None = None
        nn.init.trunc_normal_(self.decoder_query, std=0.02)

    @contextmanager
    def inference_memory_lesion(
        self, *, remove_history: bool = False, dropped_query_slot: int | None = None
    ) -> Iterator[None]:
        if self.query_slots is None:
            raise ValueError("memory lesions require query slots")
        if not remove_history and dropped_query_slot is None:
            raise ValueError("memory lesion must remove history or one query slot")
        if remove_history and not self.include_history_memory:
            raise ValueError("history lesion requires history-plus-slots memory")
        if dropped_query_slot is not None and not (
            0 <= dropped_query_slot < self.query_slots.num_slots
        ):
            raise ValueError("dropped query slot is outside the model slot range")
        if self._inference_memory_lesion is not None:
            raise RuntimeError("memory lesions cannot be nested")
        self._inference_memory_lesion = InferenceMemoryLesion(
            remove_history=remove_history,
            dropped_query_slot=dropped_query_slot,
        )
        try:
            yield
        finally:
            self._inference_memory_lesion = None

    def _encode_memory(
        self, tokens: TokenizedHistory
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_tokens = (
            tokens if self.query_slots is None else self.query_slots(tokens)
        )
        hidden = self.memory_encoder(
            encoded_tokens.embeddings,
            encoded_tokens.cumulative_lens,
            encoded_tokens.timestamps,
        )
        if self.query_slots is None:
            return hidden, encoded_tokens.cumulative_lens
        lesion = self._inference_memory_lesion
        if lesion is not None and self.training:
            raise RuntimeError(
                "memory lesions are inference-only and cannot run during training"
            )
        retained_slots = None
        include_history = self.include_history_memory
        if lesion is not None:
            include_history &= not lesion.remove_history
            if lesion.dropped_query_slot is not None:
                retained_slots = tuple(
                    slot
                    for slot in range(self.query_slots.num_slots)
                    if slot != lesion.dropped_query_slot
                )
        memory = self.query_slots.extract_memory(
            hidden,
            encoded_tokens,
            include_history=include_history,
            retained_query_slots=retained_slots,
        )
        return memory.embeddings, memory.cumulative_lens

    def _decode(self, tokens: TokenizedHistory) -> torch.Tensor:
        memory, memory_cumulative_lens = self._encode_memory(tokens)
        batch_size = tokens.cumulative_lens.shape[0] - 1
        query = self.decoder_query.unsqueeze(0).expand(batch_size, -1)
        query_cumulative_lens = torch.arange(
            batch_size + 1,
            dtype=tokens.cumulative_lens.dtype,
            device=tokens.cumulative_lens.device,
        )
        decoded = self.decoder(
            query,
            query_cumulative_lens,
            memory,
            memory_cumulative_lens,
        )
        decoded = self.query_multiplier * decoded
        return (
            decoded if self.query_projection is None else self.query_projection(decoded)
        )

    @staticmethod
    def _training_history(
        tokens: TokenizedHistory,
    ) -> tuple[TokenizedHistory, torch.Tensor, torch.Tensor]:
        lengths = tokens.cumulative_lens.diff()
        if bool((lengths < 2).any()):
            raise ValueError("cross-attention training needs history plus one target")
        target_positions = tokens.cumulative_lens[1:] - 1
        keep = torch.ones(
            tokens.item_ids.shape[0], dtype=torch.bool, device=tokens.item_ids.device
        )
        keep[target_positions] = False
        history = replace(
            tokens,
            embeddings=tokens.embeddings[keep],
            cumulative_lens=to_cumulative_lens(lengths - 1),
            is_target=tokens.is_target[keep],
            item_ids=tokens.item_ids[keep],
            timestamps=tokens.timestamps[keep],
            token_ids=(None if tokens.token_ids is None else tokens.token_ids[keep]),
            is_query=(None if tokens.is_query is None else tokens.is_query[keep]),
        )
        return (
            history,
            tokens.item_ids[target_positions],
            tokens.timestamps[target_positions],
        )

    def encode_queries(self, batch: dict) -> torch.Tensor:
        return self._decode(self.tokenizer(batch))

    def encode_cutoff_queries(self, batch: dict) -> torch.Tensor:
        return self.encode_queries(batch)

    def encode_items(self, batch: dict) -> torch.Tensor:
        return self.item_embedding(batch["int_columns"][self.item_id_column].dense())

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        history, target_ids, target_timestamps = self._training_history(
            self.tokenizer(batch)
        )
        queries = self._decode(history)
        batch_size = queries.shape[0]
        query_repr = torch.stack([queries, torch.zeros_like(queries)], dim=1).flatten(
            0, 1
        )
        item_ids = torch.stack(
            [torch.zeros_like(target_ids), target_ids], dim=1
        ).flatten()
        return {
            "query_repr": query_repr,
            "item_repr": self.item_embedding(item_ids),
            "item_ids": item_ids,
            "lengths": torch.full(
                (batch_size,), 2, dtype=torch.long, device=queries.device
            ),
            "is_target": torch.tensor(
                [False, True], dtype=torch.bool, device=queries.device
            ).repeat(batch_size),
            "is_query": torch.tensor(
                [True, False], dtype=torch.bool, device=queries.device
            ).repeat(batch_size),
            "timestamps": target_timestamps.repeat_interleave(2),
        }
