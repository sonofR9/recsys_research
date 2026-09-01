import pytest
import torch
from torch import nn

from dcn.models import (
    BosTokenizer,
    NextItemTargets,
    SemanticHistoryTokenizer,
    SequenceRetrievalModel,
    TimestampDeltaTokenizer,
)
from dcn.nn import ConcatenatedItemFeatureResidual, DenseNet
from dcn.nn.semantic_embedding import (
    CombinedSemanticIdEmbedding,
    SemanticIdEmbedding,
)
from dcn.nn.types import ModuleWithDim
from dcn.semantic import ResidualCodebooks, SemanticCodes
from dcn.tests.helpers import ITEM_COLUMN, packed_batch


MODEL_DIM = 4
SEMANTIC_DIM = 3
CODES = SemanticCodes.with_collision_suffix(
    item_ids=torch.tensor([1, 2, 3, 4]),
    codes=torch.tensor([[0, 0], [0, 0], [0, 1], [1, 0]]),
    num_codes=2,
)
CODEBOOKS = ResidualCodebooks(
    centroids=torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        ]
    )
)


def _item_embedding() -> nn.Embedding:
    return nn.Embedding(5, MODEL_DIM)


def _learned() -> SemanticIdEmbedding:
    return SemanticIdEmbedding.learned(
        CODES,
        num_items=4,
        embedding_dim=SEMANTIC_DIM,
    )


def _frozen() -> SemanticIdEmbedding:
    return SemanticIdEmbedding.from_codebooks(
        CODES,
        CODEBOOKS,
        num_items=4,
        train_collision_suffix=True,
    )


def _combined() -> CombinedSemanticIdEmbedding:
    return CombinedSemanticIdEmbedding([_learned(), _frozen()])


def _tokenizers() -> list[tuple[str, SemanticHistoryTokenizer, int]]:
    levels = CODES.num_levels
    return [
        (
            "learned SID event",
            SemanticHistoryTokenizer.learned_sid_event(
                _learned(),
                ITEM_COLUMN,
                model_dim=MODEL_DIM,
                encoder_hidden_dim=SEMANTIC_DIM,
            ),
            1,
        ),
        (
            "item and frozen SID event",
            SemanticHistoryTokenizer.item_frozen_sid_event(
                _item_embedding(),
                _frozen(),
                ITEM_COLUMN,
                model_dim=MODEL_DIM,
                encoder_hidden_dim=SEMANTIC_DIM,
            ),
            1,
        ),
        (
            "item, learned SID, and frozen SID event",
            SemanticHistoryTokenizer.item_learned_frozen_sid_event(
                _item_embedding(),
                _combined(),
                ITEM_COLUMN,
                model_dim=MODEL_DIM,
                encoder_hidden_dim=SEMANTIC_DIM,
            ),
            1,
        ),
        (
            "item and frozen SID event with learned SID residual",
            SemanticHistoryTokenizer.item_frozen_sid_learned_residual_event(
                _item_embedding(),
                _frozen(),
                _learned(),
                ITEM_COLUMN,
                model_dim=MODEL_DIM,
                encoder_hidden_dim=7,
            ),
            1,
        ),
        (
            "learned SID tokens",
            SemanticHistoryTokenizer.learned_sid_tokens(
                _learned(), ITEM_COLUMN, model_dim=MODEL_DIM
            ),
            levels,
        ),
        (
            "learned and frozen SID tokens",
            SemanticHistoryTokenizer.learned_frozen_sid_tokens(
                _combined(),
                ITEM_COLUMN,
                model_dim=MODEL_DIM,
                encoder_hidden_dim=SEMANTIC_DIM,
            ),
            levels,
        ),
        (
            "frozen SID tokens",
            SemanticHistoryTokenizer.frozen_sid_tokens(
                _frozen(),
                ITEM_COLUMN,
                model_dim=MODEL_DIM,
                encoder_hidden_dim=SEMANTIC_DIM,
            ),
            levels,
        ),
        (
            "interleaved item and SID tokens",
            SemanticHistoryTokenizer.interleaved_item_sid_tokens(
                _item_embedding(), _learned(), ITEM_COLUMN, model_dim=MODEL_DIM
            ),
            levels + 1,
        ),
    ]


@pytest.mark.parametrize(
    ("name", "tokenizer", "tokens_per_event"),
    _tokenizers(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_every_representation_preserves_items_and_uses_final_event_queries(
    name: str,
    tokenizer: SemanticHistoryTokenizer,
    tokens_per_event: int,
) -> None:
    del name

    tokens = tokenizer(packed_batch([1, 2, 3], [2, 1]))

    assert tokenizer.tokens_per_event == tokens_per_event
    assert tokenizer.out_dim == MODEL_DIM
    assert tokens.embeddings.shape == (3 * tokens_per_event, MODEL_DIM)
    assert tokens.cumulative_lens.tolist() == [
        0,
        2 * tokens_per_event,
        3 * tokens_per_event,
    ]
    assert tokens.item_ids.tolist() == [
        item_id for item_id in [1, 2, 3] for _ in range(tokens_per_event)
    ]
    expected_target = [True, *([False] * (tokens_per_event - 1))] * 3
    expected_query = [*([False] * (tokens_per_event - 1)), True] * 3
    assert tokens.is_target.tolist() == expected_target
    assert tokens.is_query is not None
    assert tokens.is_query.tolist() == expected_query


def test_every_dense_encoded_representation_uses_densenet() -> None:
    concatenated = [_tokenizers()[index][1] for index in [0, 1, 2, 5, 6]]

    assert all(
        any(isinstance(module, DenseNet) for module in tokenizer.modules())
        for tokenizer in concatenated
    )
    assert all(
        module.input_projection.out_features == SEMANTIC_DIM
        for tokenizer in concatenated
        for module in tokenizer.modules()
        if isinstance(module, DenseNet)
    )


def test_learned_sid_residual_starts_as_the_exact_frozen_event() -> None:
    item_embedding = _item_embedding()
    frozen_embedding = _frozen()
    control = SemanticHistoryTokenizer.item_frozen_sid_event(
        item_embedding,
        frozen_embedding,
        ITEM_COLUMN,
        model_dim=MODEL_DIM,
        encoder_hidden_dim=7,
    )
    treatment = SemanticHistoryTokenizer.item_frozen_sid_learned_residual_event(
        item_embedding,
        frozen_embedding,
        _learned(),
        ITEM_COLUMN,
        model_dim=MODEL_DIM,
        encoder_hidden_dim=7,
    )
    treatment.encoder.load_state_dict(control.encoder.state_dict())
    batch = packed_batch([1, 2, 3], [2, 1])

    assert isinstance(treatment.event_residual, ConcatenatedItemFeatureResidual)
    assert treatment.event_residual.residual_scale.item() == 0
    assert torch.equal(control(batch).embeddings, treatment(batch).embeddings)


def test_learned_sid_residual_accepts_a_bounded_local_gate() -> None:
    tokenizer = SemanticHistoryTokenizer.item_frozen_sid_learned_residual_event(
        _item_embedding(),
        _frozen(),
        _learned(),
        ITEM_COLUMN,
        model_dim=MODEL_DIM,
        encoder_hidden_dim=7,
        residual_max_scale=0.025,
    )

    assert tokenizer.event_residual.max_scale == 0.025
    assert tokenizer.event_residual.effective_residual_scale().item() == 0


def test_learned_sid_residual_keeps_base_width_independent_of_learned_width() -> None:
    learned = SemanticIdEmbedding.learned(CODES, num_items=4, embedding_dim=5)
    tokenizer = SemanticHistoryTokenizer.item_frozen_sid_learned_residual_event(
        _item_embedding(),
        _frozen(),
        learned,
        ITEM_COLUMN,
        model_dim=MODEL_DIM,
        encoder_hidden_dim=7,
    )

    assert isinstance(tokenizer.encoder, DenseNet)
    assert tokenizer.encoder.input_projection.out_features == 7
    assert tokenizer.residual_semantic_embedding is learned
    assert tokenizer.event_residual.feature_dim == CODES.num_levels * 5
    assert learned.initializer_rng_nonadvancing


def test_learned_sid_residual_construction_does_not_shift_control_rng() -> None:
    torch.manual_seed(19)
    control = SemanticHistoryTokenizer.item_frozen_sid_event(
        _item_embedding(),
        _frozen(),
        ITEM_COLUMN,
        model_dim=MODEL_DIM,
        encoder_hidden_dim=7,
    )
    control_after = nn.Linear(MODEL_DIM, MODEL_DIM)

    torch.manual_seed(19)
    item_embedding = _item_embedding()
    frozen_embedding = _frozen()
    with torch.random.fork_rng(devices=[]):
        learned_embedding = _learned()
    treatment = SemanticHistoryTokenizer.item_frozen_sid_learned_residual_event(
        item_embedding,
        frozen_embedding,
        learned_embedding,
        ITEM_COLUMN,
        model_dim=MODEL_DIM,
        encoder_hidden_dim=7,
    )
    treatment_after = nn.Linear(MODEL_DIM, MODEL_DIM)

    assert torch.equal(control.item_embedding.weight, treatment.item_embedding.weight)
    assert all(
        torch.equal(control_parameter, treatment_parameter)
        for control_parameter, treatment_parameter in zip(
            control.encoder.parameters(), treatment.encoder.parameters(), strict=True
        )
    )
    assert torch.equal(control_after.weight, treatment_after.weight)


def test_learned_sid_residual_gate_and_branch_receive_gradients_in_order() -> None:
    tokenizer = SemanticHistoryTokenizer.item_frozen_sid_learned_residual_event(
        _item_embedding(),
        _frozen(),
        _learned(),
        ITEM_COLUMN,
        model_dim=MODEL_DIM,
        encoder_hidden_dim=7,
    )
    batch = packed_batch([1, 2, 3], [2, 1])

    tokenizer(batch).embeddings.square().sum().backward()

    assert tokenizer.event_residual.residual_scale.grad is not None
    assert tokenizer.event_residual.residual_scale.grad.abs().item() > 0
    learned_parameters = list(tokenizer.residual_semantic_embedding.parameters())
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in learned_parameters
    )

    tokenizer.zero_grad(set_to_none=True)
    tokenizer.event_residual.residual_scale.data.fill_(1)
    tokenizer(batch).embeddings.square().sum().backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in learned_parameters
    )


def test_frozen_codebooks_keep_a_trainable_collision_suffix() -> None:
    frozen = _frozen()
    first, second = frozen.per_level(torch.tensor([1, 2]))

    assert torch.equal(first[:-1], second[:-1])
    assert not torch.equal(first[-1], second[-1])
    assert [parameter.requires_grad for parameter in frozen.parameters()] == [True]
    frozen_weights = next(
        buffer
        for buffer in frozen.buffers()
        if buffer.shape == (CODES.vocabulary.size, CODEBOOKS.dim)
    )
    for level in range(CODEBOOKS.num_levels):
        first_token, last_token = CODES.vocabulary.level_range(level)
        assert torch.equal(
            frozen_weights[first_token:last_token], CODEBOOKS.centroids[level]
        )


def test_expansion_happens_after_item_count_truncation() -> None:
    tokenizer = SemanticHistoryTokenizer.interleaved_item_sid_tokens(
        _item_embedding(), _learned(), ITEM_COLUMN, model_dim=MODEL_DIM
    )
    retained_items = list(range(1, 5))

    tokens = tokenizer(packed_batch(retained_items, [len(retained_items)]))

    assert tokens.cumulative_lens.tolist() == [
        0,
        len(retained_items) * tokenizer.tokens_per_event,
    ]
    assert tokens.item_ids[:: tokenizer.tokens_per_event].tolist() == retained_items


def test_time_features_repeat_one_event_delta_across_its_expanded_tokens() -> None:
    semantic = _learned()
    semantic.embedding.weight.data.zero_()
    tokenizer = TimestampDeltaTokenizer(
        SemanticHistoryTokenizer.learned_sid_tokens(
            semantic, ITEM_COLUMN, model_dim=MODEL_DIM
        ),
        kind="bins",
        num_bins=4,
    )
    time_encoder = tokenizer.time_encoder
    assert isinstance(time_encoder, nn.Embedding)
    time_encoder.weight.data.copy_(
        torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(-1, MODEL_DIM)
    )
    batch = packed_batch([1, 2], [2])
    batch["timestamp"] = torch.tensor([10, 1000])

    tokens = tokenizer(batch)

    per_event = tokens.embeddings.reshape(2, CODES.num_levels, MODEL_DIM)
    assert torch.equal(per_event[0, 0], per_event[0, -1])
    assert torch.equal(per_event[1, 0], per_event[1, -1])
    assert not torch.equal(per_event[0, 0], per_event[1, 0])


def test_time_features_reject_sequence_tokens_inside_the_event_wrapper() -> None:
    tokenizer = TimestampDeltaTokenizer(
        BosTokenizer(
            SemanticHistoryTokenizer.learned_sid_tokens(
                _learned(), ITEM_COLUMN, model_dim=MODEL_DIM
            )
        ),
        kind="bins",
    )

    with pytest.raises(ValueError, match="before sequence-level tokens"):
        tokenizer(packed_batch([1, 2, 3], [2, 1]))


class _IdentitySequence(ModuleWithDim):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self._out_dim = dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        embeddings: torch.Tensor,
        cumulative_lens: torch.Tensor,
        timestamps: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        del cumulative_lens, timestamps, kwargs
        return embeddings


def test_bos_and_end_query_keep_multi_token_targets_causal() -> None:
    item_embedding = _item_embedding()
    tokenizer = BosTokenizer(
        SemanticHistoryTokenizer.learned_sid_tokens(
            _learned(), ITEM_COLUMN, model_dim=MODEL_DIM
        )
    )
    model = SequenceRetrievalModel(
        tokenizer=tokenizer,
        sequence_model=_IdentitySequence(MODEL_DIM),
        item_embedding=item_embedding,
        item_id_column=ITEM_COLUMN,
        cls_token_mode="end_only",
    )

    pairs = NextItemTargets()(model(packed_batch([1, 2, 3], [3])))

    assert pairs.positive_ids.tolist() == [1, 2, 3]
    assert pairs.group_sizes.tolist() == [3]
    assert model.end_of_history_query is not None
    assert torch.equal(pairs.query_repr[-1], model.end_of_history_query.embedding)
    assert model.encode_cutoff_queries(packed_batch([1, 2, 3], [3])).shape == (
        1,
        MODEL_DIM,
    )
