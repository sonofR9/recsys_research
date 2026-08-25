import torch

from dcn.models.sequence_targets import NextItemTargets, TimeWindowTargets

DIM = 3


def _output(
    item_ids: list[int],
    lengths: list[int],
    timestamps: list[int] | None = None,
    is_target: list[bool] | None = None,
) -> dict:
    total = len(item_ids)
    return {
        "query_repr": torch.arange(total * DIM, dtype=torch.float32).view(total, DIM),
        "item_repr": torch.arange(total * DIM, dtype=torch.float32).view(total, DIM)
        * -1,
        "item_ids": torch.tensor(item_ids),
        "lengths": torch.tensor(lengths),
        "timestamps": torch.tensor(
            timestamps if timestamps is not None else list(range(total))
        ),
        "is_target": None if is_target is None else torch.tensor(is_target),
    }


def _query_tokens(output: dict, pairs) -> list[int]:
    return (pairs.query_repr[:, 0] / DIM).long().tolist()


class TestNextItemTargets:
    def test_every_token_but_the_last_predicts_its_successor(self) -> None:
        output = _output([10, 11, 12, 20, 21], [3, 2])

        pairs = NextItemTargets()(output)

        assert _query_tokens(output, pairs) == [0, 1, 3]
        assert pairs.positive_ids.tolist() == [11, 12, 21]
        assert pairs.group_sizes.tolist() == [2, 1]

    def test_non_target_tokens_are_skipped_over_not_predicted(self) -> None:
        output = _output(
            [10, 10, 11, 11, 12, 12],
            [6],
            is_target=[True, False, True, False, True, False],
        )

        pairs = NextItemTargets()(output)

        assert _query_tokens(output, pairs) == [0, 1, 2, 3]
        assert pairs.positive_ids.tolist() == [11, 11, 12, 12]

    def test_a_batch_of_singletons_has_nothing_to_predict(self) -> None:
        pairs = NextItemTargets()(_output([10, 20], [1, 1]))

        assert pairs.query_repr.shape[0] == 0
        assert pairs.group_sizes.tolist() == []


class TestTimeWindowTargets:
    def test_a_token_gets_one_positive_from_inside_its_window(self) -> None:
        output = _output([10, 11, 12, 13], [4], timestamps=[0, 10, 20, 30])

        pairs = TimeWindowTargets(window_seconds=25)(output)

        assert _query_tokens(output, pairs) == [0, 1, 2]
        assert pairs.group_sizes.tolist() == [3]

    def test_the_positive_is_always_within_the_window(self) -> None:
        output = _output([10, 11, 12, 13], [4], timestamps=[0, 1, 2, 1000])
        targets = TimeWindowTargets(window_seconds=5)

        chosen = {tuple(targets(output).positive_ids.tolist()) for _ in range(20)}

        # Token 0 may pick 11 or 12, token 1 only 12; token 2's window is empty.
        assert chosen <= {(11, 12), (12, 12)}

    def test_the_window_does_not_reach_into_the_next_sequence(self) -> None:
        output = _output([10, 11, 20, 21], [2, 2], timestamps=[0, 1, 2, 3])

        pairs = TimeWindowTargets(window_seconds=100)(output)

        assert _query_tokens(output, pairs) == [0, 2]
        assert pairs.positive_ids.tolist() == [11, 21]
        assert pairs.group_sizes.tolist() == [1, 1]

    def test_non_target_tokens_are_never_chosen_as_positives(self) -> None:
        output = _output(
            [10, 10, 11, 11],
            [4],
            timestamps=[0, 0, 1, 1],
            is_target=[True, False, True, False],
        )

        pairs = TimeWindowTargets(window_seconds=100)(output)

        assert pairs.positive_ids.tolist() == [11, 11]

    def test_an_empty_window_yields_no_pairs(self) -> None:
        output = _output([10, 11], [2], timestamps=[0, 1000])

        pairs = TimeWindowTargets(window_seconds=5)(output)

        assert pairs.query_repr.shape[0] == 0

    def test_candidates_further_than_the_lookahead_are_not_considered(self) -> None:
        output = _output([10, 11, 12, 13], [4], timestamps=[0, 1, 2, 3])

        pairs = TimeWindowTargets(window_seconds=100, lookahead=1)(output)

        assert pairs.positive_ids.tolist() == [11, 12, 13]

    def test_the_lookahead_counts_candidates_not_tokens(self) -> None:
        """An event costing two tokens must not halve how far ahead the window
        reaches: only one of the two can ever be a positive."""
        output = _output(
            [10, 10, 11, 11, 12, 12],
            [6],
            timestamps=[0, 0, 1, 1, 2, 2],
            is_target=[True, False, True, False, True, False],
        )

        pairs = TimeWindowTargets(window_seconds=100, lookahead=2)(output)
        reachable = {
            tuple(
                TimeWindowTargets(window_seconds=100, lookahead=2)(
                    output
                ).positive_ids.tolist()
            )
            for _ in range(20)
        }

        assert pairs.positive_ids.shape[0] == 4
        # Token 0 and 1 can both reach item 12, two candidate events ahead.
        assert any(chosen[0] == 12 for chosen in reachable)
