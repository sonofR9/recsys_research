import pytest
import torch
from torch.nn import functional as F

import dcn.nn.transformer as transformer_module
from dcn.models.history_tokens import ActionTokenizer, BosTokenizer, ItemTokenizer
from dcn.models.sequence_targets import NextItemTargets
from dcn.nn.sampled_softmax import RandomCatalogNegatives, StreamingInBatchSoftmax
from dcn.models.sequence_retrieval import SequenceRetrievalModel
from dcn.nn.transformer import (
    ReverseRelativePositionInput,
    Rope,
    TransformerBlock,
    TransformerEncoder,
    ValuePositions,
)
from dcn.models.two_tower import TwoTowerLoss
from dcn.tests.helpers import (
    ACTION_COLUMN,
    ITEM_COLUMN,
    TinyFFN,
    packed_batch,
    tiny_encoder,
)
from utils.global_config import config

pytestmark = pytest.mark.usefixtures("cpu_attention")

NUM_ITEMS = 16
DIM = 8


class _ProjectedItemEncoder(torch.nn.Module):
    def __init__(self, num_items: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_items, input_dim)
        self.projection = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.out_dim = output_dim

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(item_ids))


def _model(tokenizer_factory) -> SequenceRetrievalModel:
    item_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
    return SequenceRetrievalModel(
        tokenizer=tokenizer_factory(item_embedding),
        sequence_model=tiny_encoder(DIM),
        item_embedding=item_embedding,
        item_id_column=ITEM_COLUMN,
    )


def _item_model() -> SequenceRetrievalModel:
    return _model(
        lambda embedding: ItemTokenizer(embedding, item_id_column=ITEM_COLUMN)
    )


def _cls_model() -> SequenceRetrievalModel:
    item_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
    return SequenceRetrievalModel(
        tokenizer=ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN),
        sequence_model=tiny_encoder(DIM),
        item_embedding=item_embedding,
        item_id_column=ITEM_COLUMN,
        cls_token=True,
    )


def _interleaved_cls_model() -> SequenceRetrievalModel:
    item_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
    return SequenceRetrievalModel(
        tokenizer=ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN),
        sequence_model=tiny_encoder(DIM),
        item_embedding=item_embedding,
        item_id_column=ITEM_COLUMN,
        cls_token_mode="interleaved",
    )


def _cpu_flash_attention(**kwargs) -> torch.Tensor:
    assert kwargs["causal"] is True
    q, k, v = kwargs["q"], kwargs["k"], kwargs["v"]
    cumulative_lens = kwargs["cu_seqlens_q"]
    output = torch.empty_like(q)
    for start, end in zip(cumulative_lens[:-1], cumulative_lens[1:]):
        start, end = int(start), int(end)
        query = q[start:end].transpose(0, 1).unsqueeze(0).float()
        keys = k[start:end].transpose(0, 1).unsqueeze(0).float()
        values = v[start:end].transpose(0, 1).unsqueeze(0).float()
        attended = F.scaled_dot_product_attention(
            query,
            keys,
            values,
            is_causal=True,
            scale=kwargs["softmax_scale"],
        )
        output[start:end] = attended.squeeze(0).transpose(0, 1).to(q.dtype)
    return output


class TestSequenceRetrievalModel:
    def test_terminal_target_reverse_positions_match_cutoff_inference(self) -> None:
        item_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
        sequence_model = tiny_encoder(DIM)
        sequence_model.position_inputs.append(
            ReverseRelativePositionInput(DIM, max_seq_len=8)
        )
        model = SequenceRetrievalModel(
            tokenizer=ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN),
            sequence_model=sequence_model,
            item_embedding=item_embedding,
            item_id_column=ITEM_COLUMN,
            training_reverse_position_offset=1,
        ).eval()

        training = model(packed_batch([1, 2, 3, 4, 5, 6, 7], [4, 3]))
        inference = model.encode_cutoff_queries(
            packed_batch([1, 2, 3, 5, 6], [3, 2])
        )

        torch.testing.assert_close(training["query_repr"][[2, 5]], inference)

    def test_every_token_gets_a_query_and_an_item_representation(self) -> None:
        out = _item_model()(packed_batch([1, 2, 3, 4, 5], [3, 2]))

        assert out["query_repr"].shape == (5, DIM)
        assert out["item_repr"].shape == (5, DIM)
        assert out["item_ids"].tolist() == [1, 2, 3, 4, 5]
        assert out["lengths"].tolist() == [3, 2]
        assert out["is_target"].tolist() == [True] * 5

    def test_items_are_scored_against_the_table_the_history_is_read_from(self) -> None:
        model = _item_model()
        batch = packed_batch([1, 2], [2])

        out = model(batch)

        assert torch.equal(out["item_repr"], model.item_embedding.weight[[1, 2]])
        assert torch.equal(model.encode_item_ids(torch.tensor([1, 2])), out["item_repr"])
        assert torch.equal(model.encode_items(batch), out["item_repr"])
        assert model.tokenizer.item_embedding is model.item_embedding
        assert not any(
            name.startswith("catalog_item_encoder.") for name in model.state_dict()
        )

    def test_catalog_encoder_is_independent_from_history_input(self) -> None:
        history_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
        catalog_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
        model = SequenceRetrievalModel(
            tokenizer=ItemTokenizer(history_embedding, item_id_column=ITEM_COLUMN),
            sequence_model=tiny_encoder(DIM),
            item_embedding=history_embedding,
            catalog_item_encoder=catalog_embedding,
            item_id_column=ITEM_COLUMN,
        )
        batch = packed_batch([1, 2, 3], [3])

        history_before = model.tokenizer(batch).embeddings.detach().clone()
        output_before = model(batch)["item_repr"].detach().clone()
        with torch.no_grad():
            history_embedding.weight.add_(10)
        history_after = model.tokenizer(batch).embeddings.detach().clone()
        output_after = model(batch)["item_repr"].detach().clone()

        assert not torch.equal(history_after, history_before)
        assert torch.equal(output_after, output_before)
        assert torch.equal(output_after, catalog_embedding.weight[[1, 2, 3]])

    def test_untied_history_and_catalog_encoders_both_receive_gradients(self) -> None:
        history_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
        catalog_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
        model = SequenceRetrievalModel(
            tokenizer=ItemTokenizer(history_embedding, item_id_column=ITEM_COLUMN),
            sequence_model=tiny_encoder(DIM),
            item_embedding=history_embedding,
            catalog_item_encoder=catalog_embedding,
            item_id_column=ITEM_COLUMN,
        )
        criterion = TwoTowerLoss(
            model,
            StreamingInBatchSoftmax(hash_size=NUM_ITEMS, num_in_batch_negatives=4),
        )

        criterion(packed_batch([1, 2, 3, 4, 5], [3, 2]))["loss"].backward()

        assert history_embedding.weight.grad is not None
        assert catalog_embedding.weight.grad is not None

    def test_non_embedding_history_encoder_is_independent_from_catalog(self) -> None:
        history_encoder = _ProjectedItemEncoder(NUM_ITEMS, 6, DIM)
        catalog_encoder = _ProjectedItemEncoder(NUM_ITEMS, 7, 5)
        tokenizer = ItemTokenizer(history_encoder, item_id_column=ITEM_COLUMN)
        model = SequenceRetrievalModel(
            tokenizer=tokenizer,
            sequence_model=tiny_encoder(DIM),
            item_embedding=history_encoder,
            catalog_item_encoder=catalog_encoder,
            item_id_column=ITEM_COLUMN,
            query_projection=torch.nn.Linear(DIM, 5, bias=False),
        )
        batch = packed_batch([1, 2, 3], [3])

        output = model(batch)

        assert tokenizer.out_dim == DIM
        assert output["query_repr"].shape == (3, 5)
        assert torch.equal(output["item_repr"], catalog_encoder(torch.tensor([1, 2, 3])))

    def test_random_negatives_use_the_public_catalog_encoding_path(self) -> None:
        history_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
        catalog_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
        model = SequenceRetrievalModel(
            tokenizer=ItemTokenizer(history_embedding, item_id_column=ITEM_COLUMN),
            sequence_model=tiny_encoder(DIM),
            item_embedding=history_embedding,
            catalog_item_encoder=catalog_embedding,
            item_id_column=ITEM_COLUMN,
        )
        negatives = RandomCatalogNegatives(
            catalog_size=NUM_ITEMS,
            num_negatives=4,
            item_encoder=model.encode_item_ids,
            first_item_id=1,
            dense_scores=True,
        )

        torch.manual_seed(23)
        sampled, item_ids = negatives(num_examples=3, device=torch.device("cpu"))

        assert torch.equal(sampled, model.encode_item_ids(item_ids))

    def test_query_projection_decouples_transformer_and_item_widths(self) -> None:
        item_embedding = torch.nn.Embedding(NUM_ITEMS, 4)
        model = SequenceRetrievalModel(
            tokenizer=ItemTokenizer(
                item_embedding,
                item_id_column=ITEM_COLUMN,
                projection=torch.nn.Linear(4, DIM),
            ),
            sequence_model=tiny_encoder(DIM),
            item_embedding=item_embedding,
            item_id_column=ITEM_COLUMN,
            query_projection=torch.nn.Linear(DIM, 4),
        )

        out = model(packed_batch([1, 2, 3], [3]))

        assert out["query_repr"].shape == (3, 4)
        assert out["item_repr"].shape == (3, 4)

    def test_query_multiplier_does_not_change_item_representations(self) -> None:
        model = _item_model().eval()
        scaled = _item_model().eval()
        scaled.load_state_dict(model.state_dict())
        scaled.query_multiplier = 0.25
        batch = packed_batch([1, 2], [2])

        regular = model(batch)
        reduced = scaled(batch)

        assert torch.allclose(reduced["query_repr"], regular["query_repr"] * 0.25)
        assert torch.equal(reduced["item_repr"], regular["item_repr"])

    def test_a_query_cannot_see_the_future(self) -> None:
        model = _item_model().eval()

        prefix = model(packed_batch([1, 2], [2]))["query_repr"]
        extended = model(packed_batch([1, 2, 3], [3]))["query_repr"]

        assert torch.allclose(prefix, extended[:2], atol=1e-5)

    def test_cls_replaces_the_last_history_query_without_exposing_its_target(
        self,
    ) -> None:
        output = _cls_model()(packed_batch([1, 2, 3, 4], [4]))
        pairs = NextItemTargets()(output)

        assert output["item_ids"].tolist() == [1, 2, 3, 0, 4]
        assert output["is_target"].tolist() == [True, True, True, False, True]
        assert output["is_query"].tolist() == [True, True, False, True, True]
        assert pairs.positive_ids.tolist() == [2, 3, 4]
        assert torch.equal(pairs.query_repr, output["query_repr"][[0, 1, 3]])

    @pytest.mark.parametrize("attention_path", ["cpu", "flash-contract"])
    def test_cls_query_representation_cannot_see_its_target(
        self,
        attention_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _cls_model().eval()
        if attention_path == "flash-contract":
            monkeypatch.setattr(
                transformer_module,
                "flash_attn_varlen_func",
                _cpu_flash_attention,
                raising=False,
            )
            config.set_cpu_attention(False)
            model = model.to(torch.bfloat16)

        first = model(packed_batch([1, 2, 3, 4], [4]))
        changed_target = model(packed_batch([1, 2, 3, 5], [4]))
        first_query = first["query_repr"][3]
        changed_query = changed_target["query_repr"][3]

        assert torch.allclose(first_query, changed_query, atol=1e-5)
        assert torch.allclose(
            first_query @ model.item_embedding.weight.T,
            changed_query @ model.item_embedding.weight.T,
            atol=1e-5,
        )

    def test_cls_cutoff_query_is_appended_after_the_observed_history(self) -> None:
        model = _cls_model().eval()
        batch = packed_batch([1, 2, 3, 4, 5], [3, 2])
        tokens = model.end_of_history_query.for_inference(model.tokenizer(batch))
        encoded = model.encode_queries(batch)

        assert tokens.item_ids.tolist() == [1, 2, 3, 0, 4, 5, 0]
        assert tokens.cumulative_lens.tolist() == [0, 4, 7]
        assert torch.equal(
            model.encode_cutoff_queries(batch),
            encoded[tokens.cumulative_lens[1:] - 1],
        )

    def test_interleaved_cls_alone_predicts_each_following_item(self) -> None:
        output = _interleaved_cls_model()(packed_batch([1, 2, 3, 4, 5], [3, 2]))
        pairs = NextItemTargets()(output)

        assert output["item_ids"].tolist() == [1, 0, 2, 0, 3, 0, 4, 0, 5, 0]
        assert output["lengths"].tolist() == [6, 4]
        assert output["is_target"].tolist() == [True, False] * 5
        assert output["is_query"].tolist() == [False, True] * 5
        assert pairs.positive_ids.tolist() == [2, 3, 5]
        assert torch.equal(pairs.query_repr, output["query_repr"][[1, 3, 7]])

    def test_interleaved_cls_uses_the_same_layout_for_cutoff_queries(self) -> None:
        model = _interleaved_cls_model().eval()
        batch = packed_batch([1, 2, 3, 4, 5], [3, 2])
        output = model(batch)
        encoded = model.encode_queries(batch)

        assert encoded.shape == output["query_repr"].shape == (10, DIM)
        assert torch.equal(
            model.encode_cutoff_queries(batch),
            encoded[torch.tensor([5, 9])],
        )

    @pytest.mark.parametrize("attention_path", ["cpu", "flash-contract"])
    def test_interleaved_cls_query_cannot_see_the_following_item(
        self,
        attention_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _interleaved_cls_model().eval()
        if attention_path == "flash-contract":
            monkeypatch.setattr(
                transformer_module,
                "flash_attn_varlen_func",
                _cpu_flash_attention,
                raising=False,
            )
            config.set_cpu_attention(False)
            model = model.to(torch.bfloat16)

        first = model(packed_batch([1, 2, 3], [3]))
        changed = model(packed_batch([1, 4, 3], [3]))

        assert torch.allclose(first["query_repr"][1], changed["query_repr"][1], atol=1e-5)

    def test_cutoff_queries_follow_tokenizer_added_tokens(self) -> None:
        item_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
        model = SequenceRetrievalModel(
            tokenizer=BosTokenizer(
                ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN)
            ),
            sequence_model=tiny_encoder(DIM),
            item_embedding=item_embedding,
            item_id_column=ITEM_COLUMN,
        ).eval()
        batch = packed_batch([1, 2, 3, 4, 5], [3, 2])
        tokens = model.tokenizer(batch)

        expected = model.encode_queries(batch)[tokens.cumulative_lens[1:] - 1]

        assert torch.equal(model.encode_cutoff_queries(batch), expected)

    def test_timestamps_reach_value_position_rope(self) -> None:
        item_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
        encoder = TransformerEncoder(
            [
                TransformerBlock(
                    dim=DIM,
                    nhead=2,
                    num_kv_heads=1,
                    ffn_factory=TinyFFN,
                    dropout=0.0,
                    rope=Rope(DIM // 2, ValuePositions()),
                )
            ]
        ).eval()
        model = SequenceRetrievalModel(
            tokenizer=ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN),
            sequence_model=encoder,
            item_embedding=item_embedding,
            item_id_column=ITEM_COLUMN,
        ).eval()
        evenly_spaced = packed_batch([1, 2, 3], [3])
        irregular = packed_batch([1, 2, 3], [3])
        irregular["timestamp"] = torch.tensor([0, 1, 20])

        even_output = model.encode_queries(evenly_spaced)
        irregular_output = model.encode_queries(irregular)

        assert not torch.allclose(even_output, irregular_output)

    def test_per_layer_item_embeddings_receive_gradients(self) -> None:
        item_embedding = torch.nn.Embedding(NUM_ITEMS, DIM)
        layer_embeddings = [torch.nn.Embedding(NUM_ITEMS, DIM) for _ in range(2)]
        model = SequenceRetrievalModel(
            tokenizer=ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN),
            sequence_model=tiny_encoder(DIM, num_layers=2),
            item_embedding=item_embedding,
            item_id_column=ITEM_COLUMN,
            layer_item_embeddings=layer_embeddings,
        )
        criterion = TwoTowerLoss(
            model,
            StreamingInBatchSoftmax(hash_size=NUM_ITEMS, num_in_batch_negatives=4),
        )

        criterion(packed_batch([1, 2, 3, 4, 5], [3, 2]))["loss"].backward()

        assert all(embedding.weight.grad is not None for embedding in layer_embeddings)

    def test_action_tokens_are_context_but_never_targets(self) -> None:
        model = _model(
            lambda embedding: ActionTokenizer(
                embedding,
                action_embedding=torch.nn.Embedding(3, DIM),
                item_id_column=ITEM_COLUMN,
                action_column=ACTION_COLUMN,
            )
        )

        out = model(packed_batch([1, 2, 3], [2, 1], actions=[0, 1, 2]))

        assert out["query_repr"].shape == (6, DIM)
        assert out["is_target"].tolist() == [True, False] * 3
        assert out["lengths"].tolist() == [4, 2]

    def test_it_trains_through_the_in_batch_softmax(self) -> None:
        model = _item_model()
        criterion = TwoTowerLoss(
            model,
            StreamingInBatchSoftmax(hash_size=NUM_ITEMS, num_in_batch_negatives=4),
        )

        out = criterion(packed_batch([1, 2, 3, 4, 5], [3, 2]))
        out["loss"].backward()

        assert out["loss"].item() > 0
        assert model.item_embedding.weight.grad is not None
