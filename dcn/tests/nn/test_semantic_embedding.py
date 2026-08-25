import pytest
import torch

from dcn.data.features import FeatureValues
from dcn.nn.semantic_embedding import CombinedSemanticIdEmbedding, SemanticIdEmbedding
from dcn.semantic import ResidualCodebooks, SemanticCodes

CODES = SemanticCodes(
    item_ids=torch.tensor([1, 2, 3]),
    codes=torch.tensor([[0, 1], [0, 0], [1, 1]]),
    codes_per_level=(2, 2),
)


def _feature(item_ids: list[int]) -> FeatureValues:
    return FeatureValues(
        torch.tensor(item_ids), torch.arange(len(item_ids) + 1, dtype=torch.int64)
    )


class TestSemanticIdEmbedding:
    def test_concatenates_one_vector_per_level(self) -> None:
        embedding = SemanticIdEmbedding.learned(CODES, num_items=3, embedding_dim=4)

        output = embedding(_feature([1, 2, 3]))

        assert embedding.out_dim == 8
        assert output.shape == (3, 8)

    def test_items_sharing_a_code_share_that_part_of_the_vector(self) -> None:
        embedding = SemanticIdEmbedding.learned(CODES, num_items=3, embedding_dim=4)

        output = embedding(_feature([1, 2]))

        # Items 1 and 2 share level 0 (code 0) and differ at level 1.
        assert torch.equal(output[0, :4], output[1, :4])
        assert not torch.equal(output[0, 4:], output[1, 4:])

    def test_an_item_without_codes_gets_its_own_vector(self) -> None:
        embedding = SemanticIdEmbedding.learned(CODES, num_items=5, embedding_dim=4)

        output = embedding(_feature([4, 0]))

        assert torch.equal(output[0], output[1])

    def test_codebook_embeddings_reconstruct_the_quantized_vector(self) -> None:
        centroids = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[10.0, 0.0], [0.0, 10.0]],
            ]
        )
        codebooks = ResidualCodebooks(centroids)

        embedding = SemanticIdEmbedding.from_codebooks(CODES, codebooks, num_items=3)
        output = embedding(_feature([1, 3]))

        assert embedding.out_dim == 2
        assert torch.equal(output, codebooks.decode(CODES.codes[[0, 2]]))

    def test_codebook_embeddings_are_frozen(self) -> None:
        codebooks = ResidualCodebooks(torch.randn(2, 2, 3))

        embedding = SemanticIdEmbedding.from_codebooks(CODES, codebooks, num_items=3)

        assert not any(parameter.requires_grad for parameter in embedding.parameters())

    def test_the_collision_suffix_has_no_centroid_behind_it(self) -> None:
        """The suffix separates items the codebooks quantized identically, so
        there is nothing in them it could be built from."""
        codes = SemanticCodes.with_collision_suffix(
            torch.tensor([1, 2, 3]),
            torch.tensor([[0, 1], [0, 1], [1, 0]]),
            num_codes=2,
        )
        codebooks = ResidualCodebooks(torch.randn(2, 2, 3))

        embedding = SemanticIdEmbedding.from_codebooks(codes, codebooks, num_items=3)
        per_level = embedding.per_level(torch.tensor([1, 2, 3]))

        assert embedding.num_levels == 3
        assert torch.equal(per_level[:, -1], torch.zeros(3, 3))
        assert torch.equal(per_level[0], per_level[1])

    def test_codes_the_codebooks_cannot_explain_are_rejected(self) -> None:
        codes = SemanticCodes(
            item_ids=torch.tensor([1]),
            codes=torch.tensor([[0, 1, 0]]),
            codes_per_level=(2, 2, 2),
        )

        with pytest.raises(AssertionError):
            SemanticIdEmbedding.from_codebooks(
                codes, ResidualCodebooks(torch.randn(1, 2, 3)), num_items=1
            )

    def test_per_level_keeps_the_levels_apart(self) -> None:
        embedding = SemanticIdEmbedding.learned(CODES, num_items=3, embedding_dim=4)

        per_level = embedding.per_level(torch.tensor([1, 2, 3]))

        assert per_level.shape == (3, 2, 4)
        assert embedding.level_dim == 4
        assert torch.equal(per_level.flatten(1), embedding(_feature([1, 2, 3])))

    def test_tokens_name_the_row_each_level_reads(self) -> None:
        embedding = SemanticIdEmbedding.learned(CODES, num_items=3, embedding_dim=4)

        tokens = embedding.tokens(torch.tensor([1, 3]))

        vocabulary = CODES.vocabulary
        assert torch.equal(tokens, vocabulary.tokens(CODES.codes[[0, 2]]))


class TestCombinedSemanticIdEmbedding:
    def test_it_lays_both_views_of_a_level_side_by_side(self) -> None:
        learned = SemanticIdEmbedding.learned(CODES, num_items=3, embedding_dim=4)
        frozen = SemanticIdEmbedding.from_codebooks(
            CODES, ResidualCodebooks(torch.randn(2, 2, 3)), num_items=3
        )

        combined = CombinedSemanticIdEmbedding([learned, frozen])
        item_ids = torch.tensor([1, 2])
        per_level = combined.embed_tokens(combined.tokens(item_ids))

        assert combined.level_dim == 7
        assert combined.num_levels == 2
        assert torch.equal(per_level[:, :, :4], learned.per_level(torch.tensor([1, 2])))
        assert torch.equal(per_level[:, :, 4:], frozen.per_level(torch.tensor([1, 2])))

    def test_the_frozen_half_stays_frozen(self) -> None:
        learned = SemanticIdEmbedding.learned(CODES, num_items=3, embedding_dim=4)
        frozen = SemanticIdEmbedding.from_codebooks(
            CODES, ResidualCodebooks(torch.randn(2, 2, 3)), num_items=3
        )

        combined = CombinedSemanticIdEmbedding([learned, frozen])

        trainable = [p for p in combined.parameters() if p.requires_grad]
        assert len(trainable) == 1
        assert trainable[0] is learned.embedding.weight
