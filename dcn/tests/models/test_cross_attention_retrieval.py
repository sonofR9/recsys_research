import torch
import pytest
from torch import nn

from dcn.models.cross_attention_retrieval import CrossAttentionRetrievalModel
from dcn.models.history_tokens import EndQuerySlots, ItemTokenizer
from dcn.models.sequence_targets import NextItemTargets
from dcn.nn.types import ModuleWithDim
from dcn.tests.helpers import ITEM_COLUMN, packed_batch


DIM = 8
NUM_ITEMS = 32


class IdentityMemoryEncoder(ModuleWithDim):
    @property
    def out_dim(self) -> int:
        return DIM

    def forward(
        self,
        embeddings: torch.Tensor,
        cumulative_lens: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return embeddings


class MemoryMeanDecoder(ModuleWithDim):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(DIM))

    @property
    def out_dim(self) -> int:
        return DIM

    def forward(
        self,
        query: torch.Tensor,
        query_cumulative_lens: torch.Tensor,
        memory: torch.Tensor,
        memory_cumulative_lens: torch.Tensor,
    ) -> torch.Tensor:
        means = [
            memory[start:end].mean(dim=0)
            for start, end in zip(
                memory_cumulative_lens[:-1], memory_cumulative_lens[1:]
            )
        ]
        return query + torch.stack(means)


def _model(
    *, query_slots: EndQuerySlots | None = None, include_history_memory: bool = False
) -> CrossAttentionRetrievalModel:
    item_embedding = nn.Embedding(NUM_ITEMS, DIM)
    return CrossAttentionRetrievalModel(
        tokenizer=ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN),
        memory_encoder=IdentityMemoryEncoder(),
        decoder=MemoryMeanDecoder(),
        item_embedding=item_embedding,
        item_id_column=ITEM_COLUMN,
        query_slots=query_slots,
        include_history_memory=include_history_memory,
    )


class TestCrossAttentionRetrievalModel:
    def test_one_candidate_target_is_emitted_per_training_sequence(self) -> None:
        model = _model()

        output = model(packed_batch([1, 2, 3, 10, 11], [3, 2]))
        pairs = NextItemTargets()(output)

        assert output["lengths"].tolist() == [2, 2]
        assert pairs.positive_ids.tolist() == [3, 11]
        assert pairs.group_sizes.tolist() == [1, 1]

    def test_training_query_cannot_see_the_candidate_target(self) -> None:
        model = _model().eval()

        original = model(packed_batch([1, 2, 3, 10, 11], [3, 2]))
        changed = model(packed_batch([1, 2, 4, 10, 12], [3, 2]))

        torch.testing.assert_close(
            original["query_repr"][[0, 2]], changed["query_repr"][[0, 2]]
        )

    def test_cutoff_queries_use_every_observed_history_item(self) -> None:
        model = _model().eval()
        first = packed_batch([1, 2, 10], [2, 1])
        changed = packed_batch([1, 3, 10], [2, 1])

        original_queries = model.encode_cutoff_queries(first)
        changed_queries = model.encode_cutoff_queries(changed)

        assert original_queries.shape == (2, DIM)
        assert not torch.allclose(original_queries[0], changed_queries[0])
        torch.testing.assert_close(original_queries[1], changed_queries[1])

    def test_queries_and_catalog_items_share_a_scoring_width(self) -> None:
        model = _model().eval()
        query_batch = packed_batch([1, 2, 10], [2, 1])
        item_batch = packed_batch([3, 4, 5], [3])

        queries = model.encode_queries(query_batch)
        items = model.encode_items(item_batch)

        assert queries.shape == (2, DIM)
        assert items.shape == (3, DIM)
        assert (queries @ items.T).shape == (2, 3)

    def test_query_multiplier_scales_queries_without_changing_items(self) -> None:
        model = _model().eval()
        batch = packed_batch([1, 2], [2])
        regular = model.encode_queries(batch)
        items = model.encode_items(batch)

        model.query_multiplier = 0.25

        torch.testing.assert_close(model.encode_queries(batch), regular * 0.25)
        torch.testing.assert_close(model.encode_items(batch), items)

    def test_downstream_loss_reaches_the_memory_encoder_and_decoder_query(self) -> None:
        model = _model()
        output = model(packed_batch([1, 2, 3, 10, 11], [3, 2]))
        loss = NextItemTargets()(output).query_repr.square().mean()

        loss.backward()

        assert model.item_embedding.weight.grad is not None
        assert model.decoder_query.grad is not None
        assert model.decoder_query.grad.abs().sum() > 0

    def test_downstream_loss_reaches_every_distinct_first_decoder_slot(self) -> None:
        slots = EndQuerySlots(DIM, num_slots=4, shared=False)
        model = _model(query_slots=slots)
        output = model(packed_batch([1, 2, 3, 10, 11], [3, 2]))

        NextItemTargets()(output).query_repr.square().mean().backward()

        assert slots.embeddings.grad is not None
        assert slots.embeddings.grad.abs().sum(dim=1).gt(0).tolist() == [True] * 4

    def test_history_plus_slots_memory_keeps_history_states(self) -> None:
        torch.manual_seed(0)
        slots_only = _model(
            query_slots=EndQuerySlots(DIM, num_slots=4, shared=False)
        ).eval()
        torch.manual_seed(0)
        with_history = _model(
            query_slots=EndQuerySlots(DIM, num_slots=4, shared=False),
            include_history_memory=True,
        ).eval()
        with_history.load_state_dict(slots_only.state_dict())
        original = packed_batch([1, 2, 3], [3])
        changed = packed_batch([1, 2, 4], [3])

        slots_only_original = slots_only.encode_queries(original)
        slots_only_changed = slots_only.encode_queries(changed)
        with_history_original = with_history.encode_queries(original)
        with_history_changed = with_history.encode_queries(changed)

        torch.testing.assert_close(slots_only_original, slots_only_changed)
        assert not torch.allclose(with_history_original, with_history_changed)

    def test_inference_lesions_remove_only_the_requested_memory(self) -> None:
        model = _model(
            query_slots=EndQuerySlots(DIM, num_slots=4, shared=False),
            include_history_memory=True,
        ).eval()
        batch = packed_batch([1, 2, 3], [3])
        normal = model.encode_queries(batch)

        with model.inference_memory_lesion(remove_history=True):
            without_history = model.encode_queries(batch)
        with model.inference_memory_lesion(dropped_query_slot=2):
            without_slot_two = model.encode_queries(batch)

        assert not torch.allclose(normal, without_history)
        assert not torch.allclose(normal, without_slot_two)
        torch.testing.assert_close(model.encode_queries(batch), normal)

    def test_inference_lesions_fail_closed_for_training_and_invalid_axes(self) -> None:
        slots_only = _model(query_slots=EndQuerySlots(DIM, num_slots=4, shared=True))

        with pytest.raises(ValueError, match="history lesion"):
            with slots_only.inference_memory_lesion(remove_history=True):
                pass
        with pytest.raises(ValueError, match="query slot"):
            with slots_only.inference_memory_lesion(dropped_query_slot=4):
                pass
        with slots_only.inference_memory_lesion(dropped_query_slot=0):
            with pytest.raises(RuntimeError, match="training"):
                slots_only.encode_queries(packed_batch([1, 2], [2]))
