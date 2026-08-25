import pytest
import torch

from dcn.semantic import SemanticCodes, SemanticVocabulary


def _codes(rows: list[list[int]], codes_per_level: tuple[int, ...]) -> SemanticCodes:
    return SemanticCodes(
        item_ids=torch.arange(1, len(rows) + 1),
        codes=torch.tensor(rows),
        codes_per_level=codes_per_level,
    )


class TestSemanticCodes:
    def test_colliding_items_get_distinct_suffixes(self) -> None:
        codebook_codes = torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1]])

        codes = SemanticCodes.with_collision_suffix(
            item_ids=torch.tensor([5, 6, 7, 8]),
            codes=codebook_codes,
            num_codes=2,
        )

        assert codes.codes[:, :2].tolist() == codebook_codes.tolist()
        assert sorted(codes.codes[[0, 1, 3], 2].tolist()) == [1, 2, 3]
        assert codes.codes[2, 2].item() == 1
        assert codes.codes_per_level == (2, 2, 4)

    def test_no_item_carries_the_tuple_the_unknown_row_gets(self) -> None:
        codes = SemanticCodes.with_collision_suffix(
            item_ids=torch.tensor([1, 2]),
            codes=torch.tensor([[0, 0], [1, 1]]),
            num_codes=2,
        )

        table = codes.lookup_table(num_items=2)

        assert (table[1:] != table[0]).any(dim=1).all()

    def test_a_code_wider_than_its_level_is_rejected(self) -> None:
        with pytest.raises(AssertionError):
            _codes([[0, 2]], (2, 2))

    def test_lookup_table_maps_item_ids_to_their_codes(self) -> None:
        codes = _codes([[0, 1], [1, 0]], (2, 2))

        table = codes.lookup_table(num_items=4)

        assert table.shape == (5, 2)
        assert table[1].tolist() == [0, 1]
        assert table[2].tolist() == [1, 0]
        # 0 is the unknown item, and 3/4 carry no code of their own.
        assert table[0].tolist() == [0, 0]
        assert table[3].tolist() == [0, 0]

    def test_round_trip_through_a_file(self, tmp_path) -> None:
        codes = _codes([[0, 1, 2], [1, 0, 0]], (2, 2, 3))

        path = tmp_path / "codes.pt"
        codes.save(path)
        loaded = SemanticCodes.load(path)

        assert torch.equal(loaded.codes, codes.codes)
        assert torch.equal(loaded.item_ids, codes.item_ids)
        assert loaded.codes_per_level == codes.codes_per_level


class TestSemanticVocabulary:
    def test_every_level_and_code_gets_its_own_token(self) -> None:
        vocabulary = SemanticVocabulary(codes_per_level=(3, 2))

        tokens = vocabulary.tokens(torch.tensor([[0, 0], [2, 1]]))

        assert vocabulary.size == vocabulary.num_special_tokens + 5
        assert len(set(tokens.flatten().tolist())) == 4
        assert tokens.min() >= vocabulary.num_special_tokens

    def test_token_ids_of_a_level_are_contiguous(self) -> None:
        vocabulary = SemanticVocabulary(codes_per_level=(3, 2))

        first, last = vocabulary.level_range(1)

        assert last - first == 2
        assert vocabulary.tokens(torch.tensor([[0, 1]]))[0, 1] == first + 1

    def test_padding_and_beginning_of_sequence_are_reserved(self) -> None:
        vocabulary = SemanticVocabulary(codes_per_level=(4,))

        assert vocabulary.padding_token != vocabulary.beginning_token
        assert vocabulary.tokens(torch.tensor([[0]])).min() > max(
            vocabulary.padding_token, vocabulary.beginning_token
        )
