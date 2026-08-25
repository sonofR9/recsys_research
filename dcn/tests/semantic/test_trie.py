import torch

from dcn.semantic import CodeTrie, SemanticCodes

CODES = torch.tensor(
    [
        [0, 0, 1],
        [0, 2, 0],
        [1, 1, 0],
        [1, 1, 2],
    ]
)
CODES_PER_LEVEL = (3, 3, 3)


def _trie() -> CodeTrie:
    return CodeTrie(
        SemanticCodes(
            item_ids=torch.tensor([10, 11, 12, 13]),
            codes=CODES,
            codes_per_level=CODES_PER_LEVEL,
        )
    )


class TestCodeTrie:
    def test_first_level_allows_only_codes_in_use(self) -> None:
        mask = _trie().allowed_mask(0, torch.zeros(1, 0, dtype=torch.int64))

        assert mask.tolist() == [[True, True, False]]

    def test_children_depend_on_the_prefix(self) -> None:
        trie = _trie()

        mask = trie.allowed_mask(1, torch.tensor([[0], [1]]))

        assert mask[0].tolist() == [True, False, True]
        assert mask[1].tolist() == [False, True, False]

    def test_deeper_prefixes_narrow_further(self) -> None:
        trie = _trie()

        mask = trie.allowed_mask(2, torch.tensor([[1, 1], [0, 2]]))

        assert mask[0].tolist() == [True, False, True]
        assert mask[1].tolist() == [True, False, False]

    def test_a_full_tuple_names_exactly_its_own_item(self) -> None:
        trie = _trie()

        items, cumulative_lens = trie.items_under(torch.tensor([[1, 1, 2], [0, 0, 1]]))

        assert cumulative_lens.tolist() == [0, 1, 2]
        assert items.tolist() == [13, 10]

    def test_a_prefix_names_every_item_below_it(self) -> None:
        trie = _trie()

        items, cumulative_lens = trie.items_under(torch.tensor([[1, 1], [0, 0]]))

        assert cumulative_lens.tolist() == [0, 2, 3]
        assert sorted(items[:2].tolist()) == [12, 13]
        assert items[2:].tolist() == [10]

    def test_the_empty_prefix_names_the_whole_catalog(self) -> None:
        trie = _trie()

        items, cumulative_lens = trie.items_under(torch.zeros(1, 0, dtype=torch.int64))

        assert cumulative_lens.tolist() == [0, 4]
        assert sorted(items.tolist()) == [10, 11, 12, 13]

    def test_a_prefix_no_item_carries_names_nothing(self) -> None:
        trie = _trie()

        items, cumulative_lens = trie.items_under(torch.tensor([[2, 2], [-1, -1]]))

        assert cumulative_lens.tolist() == [0, 0, 0]
        assert items.tolist() == []

    def test_every_masked_walk_reaches_an_item(self) -> None:
        trie = _trie()
        prefixes = torch.zeros(1, 0, dtype=torch.int64)

        for level in range(len(CODES_PER_LEVEL)):
            mask = trie.allowed_mask(level, prefixes)
            rows, codes = mask.nonzero(as_tuple=True)
            prefixes = torch.cat([prefixes[rows], codes.unsqueeze(1)], dim=1)

        assert sorted(prefixes.tolist()) == sorted(CODES.tolist())
        _, cumulative_lens = trie.items_under(prefixes)
        assert cumulative_lens.diff().tolist() == [1] * len(CODES)
