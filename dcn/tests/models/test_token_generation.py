import pytest
import torch

from dcn.models.history_tokens import SemanticIdTokenizer
from dcn.models.semantic_constraint import SemanticIdConstraint
from dcn.models.token_generation import (
    CausalTokenDecoder,
    Seq2SeqTokenDecoder,
    TokenDecoder,
    TokenPredictionLoss,
)
from dcn.nn.semantic_embedding import SemanticIdEmbedding
from dcn.nn.transformer import (
    CrossAttentionBlock,
    TransformerDecoder,
)
from dcn.semantic import SemanticCodes
from dcn.tests.helpers import (
    CODES,
    ITEM_COLUMN,
    TinyFFN,
    packed_batch,
    tiny_encoder,
)

pytestmark = pytest.mark.usefixtures("cpu_attention")

DIM = 8
NUM_LEVELS = 2

# Two items sharing a first level, so the trie forbids most of the space.
SPARSE_CODES = SemanticCodes(
    item_ids=torch.tensor([1, 2]),
    codes=torch.tensor([[0, 0], [0, 1]]),
    codes_per_level=(2, 4),
)


def _tokenizer(codes: SemanticCodes = CODES) -> SemanticIdTokenizer:
    return SemanticIdTokenizer(
        SemanticIdEmbedding.learned(codes, num_items=8, embedding_dim=DIM),
        item_id_column=ITEM_COLUMN,
    )


def _decoder_only(codes: SemanticCodes = CODES) -> CausalTokenDecoder:
    return CausalTokenDecoder(
        tokenizer=_tokenizer(codes),
        sequence_model=tiny_encoder(DIM),
        constraint=SemanticIdConstraint(codes),
    )


def _seq2seq(codes: SemanticCodes = CODES) -> Seq2SeqTokenDecoder:
    return Seq2SeqTokenDecoder(
        tokenizer=_tokenizer(codes),
        encoder=tiny_encoder(DIM),
        decoder=TransformerDecoder(
            self_attention_blocks=list(tiny_encoder(DIM).layers),
            cross_attention_blocks=[
                CrossAttentionBlock(
                    dim=DIM,
                    nhead=2,
                    num_kv_heads=1,
                    ffn_factory=TinyFFN,
                    dropout=0.0,
                )
            ],
            final_norm=torch.nn.LayerNorm(DIM),
        ),
        constraint=SemanticIdConstraint(codes),
    )


def _items(model: TokenDecoder, generated: torch.Tensor) -> set[int]:
    items, _ = model.constraint.items_under(generated.flatten(0, 1))
    return set(items.tolist())


class TestCausalTokenDecoder:
    def test_every_token_but_the_last_of_a_sequence_is_predicted(self) -> None:
        model = _decoder_only()

        out = model(packed_batch([1, 2, 3, 4], [3, 1]))

        # 4 items x 2 levels = 8 tokens, minus one final token per sequence.
        assert out["logits"].shape == (6, CODES.vocabulary.size)
        assert out["targets"].shape == (6,)

    def test_the_target_of_a_token_is_the_next_token_of_its_own_sequence(
        self,
    ) -> None:
        model = _decoder_only()
        tokens = CODES.vocabulary.tokens(CODES.codes)

        out = model(packed_batch([1, 2, 3], [2, 1]))

        expected = [
            tokens[0][1],
            tokens[1][0],
            tokens[1][1],  # first sequence: items 1, 2
            tokens[2][1],  # second sequence: item 3
        ]
        assert out["targets"].tolist() == [int(token) for token in expected]

    def test_every_target_is_labelled_with_the_slot_it_fills(self) -> None:
        model = _decoder_only()

        out = model(packed_batch([1, 2, 3], [2, 1]))

        assert out["slots"].tolist() == [1, 0, 1, 1]

    def test_a_prediction_can_only_name_a_code_of_the_level_it_predicts(
        self,
    ) -> None:
        model = _decoder_only()
        vocabulary = CODES.vocabulary

        out = model(packed_batch([1, 2], [2]))

        finite = out["logits"].isfinite()
        # Positions alternate: token 0 predicts level 1, token 1 predicts level 0.
        for position, level in enumerate([1, 0, 1]):
            first, last = vocabulary.level_range(level)
            assert finite[position].nonzero().flatten().tolist() == list(
                range(first, last)
            )

    def test_generation_only_ever_names_items_of_the_catalog(self) -> None:
        model = _decoder_only(SPARSE_CODES).eval()

        generated, scores = model.generate(
            packed_batch([1, 2, 1], [2, 1]), beam_width=2
        )

        assert generated.shape == (2, 2, NUM_LEVELS)
        assert _items(model, generated) <= {1, 2}
        assert (scores[:, 0] >= scores[:, 1]).all()

    def test_it_can_stop_before_the_last_slot(self) -> None:
        model = _decoder_only().eval()

        generated, _ = model.generate(
            packed_batch([1, 2], [2]), beam_width=2, num_slots=1
        )

        assert generated.shape == (1, 2, 1)
        assert _items(model, generated) == {1, 2, 3, 4}

    def test_it_generates_the_continuation_it_was_trained_on(self) -> None:
        """Training predicts the next token and generation decodes one; if the
        two disagree about level order or code offsets, the model fits its
        training data and then names something else."""
        # Pinned: this is the only test here that trains, and a model this small
        # does not converge from every init.
        torch.manual_seed(1)
        model = _decoder_only()
        criterion = TokenPredictionLoss(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05, fused=True)
        # Item 1 is always followed by 2, item 3 by 4.
        batch = packed_batch([1, 2, 3, 4], [2, 2])

        for _ in range(200):
            optimizer.zero_grad()
            out = criterion(batch)
            out["loss"].backward()
            optimizer.step()
        assert out["accuracy"].item() == 1.0

        generated, _ = model.eval().generate(packed_batch([1, 3], [1, 1]), beam_width=2)

        assert _items(model, generated[:, :1]) == {2, 4}

    def test_a_beam_with_no_continuation_left_names_nothing(self) -> None:
        # Two items over a 2x4 space: a third beam has nowhere to go.
        model = _decoder_only(SPARSE_CODES).eval()

        generated, scores = model.generate(packed_batch([1, 2], [2]), beam_width=3)

        assert scores[0, 2] == -torch.inf
        assert (generated[0, 2] == -1).all()
        assert _items(model, generated) == {1, 2}

    def test_beams_come_back_best_first(self) -> None:
        model = _decoder_only().eval()

        _, scores = model.generate(packed_batch([1, 2], [2]), beam_width=3)

        assert (scores.diff(dim=1) <= 1e-6).all()


class TestSeq2SeqTokenDecoder:
    def test_it_predicts_one_token_per_level_of_the_last_item(self) -> None:
        model = _seq2seq()
        tokens = CODES.vocabulary.tokens(CODES.codes)

        out = model(packed_batch([1, 2, 3, 4], [3, 1]))

        # One sequence of 3 items and one of 1: only the first can be decoded.
        assert out["logits"].shape == (NUM_LEVELS, CODES.vocabulary.size)
        assert out["targets"].tolist() == [int(token) for token in tokens[2]]

    def test_every_target_is_labelled_with_the_slot_it_fills(self) -> None:
        model = _seq2seq()

        out = model(packed_batch([1, 2, 3, 4], [3, 1]))

        assert out["slots"].tolist() == [0, 1]

    def test_the_decoder_is_the_only_thing_that_sees_the_target(self) -> None:
        model = _seq2seq().eval()

        with_other_target = model(packed_batch([1, 2, 3], [3]))["logits"]
        changed_target = model(packed_batch([1, 2, 4], [3]))["logits"]

        assert torch.allclose(with_other_target[0], changed_target[0], atol=1e-5)

    def test_generation_only_ever_names_items_of_the_catalog(self) -> None:
        model = _seq2seq(SPARSE_CODES).eval()

        generated, _ = model.generate(packed_batch([1, 2, 1], [3]), beam_width=2)

        assert generated.shape == (1, 2, NUM_LEVELS)
        assert _items(model, generated) <= {1, 2}


class TestTokenPredictionLoss:
    def test_it_trains_the_model_it_wraps(self) -> None:
        model = _decoder_only()
        criterion = TokenPredictionLoss(model)

        out = criterion(packed_batch([1, 2, 3, 4], [4]))
        out["loss"].backward()

        assert out["loss"].item() > 0
        assert 0.0 <= out["accuracy"].item() <= 1.0
        assert any(p.grad is not None for p in model.parameters())

    def test_a_batch_with_nothing_to_predict_still_backpropagates(self) -> None:
        # Every sequence is one item long, so the seq2seq split leaves no history.
        criterion = TokenPredictionLoss(_seq2seq())

        out = criterion(packed_batch([1, 2], [1, 1]))
        out["loss"].backward()

        assert out["loss"].item() == 0.0

    def test_it_reports_the_same_keys_whatever_the_batch_holds(self) -> None:
        criterion = TokenPredictionLoss(_seq2seq())

        full = criterion(packed_batch([1, 2, 3, 4], [3, 1]))
        empty = criterion(packed_batch([1, 2], [1, 1]))

        assert full.keys() == empty.keys()
        assert "slot0_loss" in full and "slot1_accuracy" in full

    def test_the_reported_accuracy_is_the_per_slot_ones_put_together(
        self,
    ) -> None:
        criterion = TokenPredictionLoss(_decoder_only())

        # Three predicted tokens per sequence: two of slot 1, one of slot 0.
        out = criterion(packed_batch([1, 2, 3, 4], [2, 2]))

        per_slot = (2 * out["slot1_accuracy"] + 1 * out["slot0_accuracy"]) / 3
        assert out["accuracy"].item() == pytest.approx(per_slot.item())
