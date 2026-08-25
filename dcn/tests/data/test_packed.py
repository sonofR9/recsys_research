import pytest
import torch

from dcn.data.features import FeatureValues
from dcn.data.packed import (
    append_to_sequences,
    ragged_positions,
    repeat_sequences,
    split_last_events,
    to_cumulative_lens,
)


class TestRaggedPositions:
    def test_every_element_gets_its_row_and_its_rank_in_that_row(self) -> None:
        lengths = torch.tensor([2, 0, 3])

        rows, ranks = ragged_positions(lengths, 5)

        assert rows.tolist() == [0, 0, 2, 2, 2]
        assert ranks.tolist() == [0, 1, 0, 1, 2]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a gpu")
    def test_compiling_it_does_not_change_the_answer(self) -> None:
        lengths = torch.randint(
            0, 30, (64,), generator=torch.Generator().manual_seed(0)
        )
        lengths = lengths.cuda()
        total = int(lengths.sum())

        compiled = torch.compile(ragged_positions, dynamic=False)(lengths, total)

        for expected, actual in zip(ragged_positions(lengths, total), compiled):
            assert torch.equal(expected, actual)


def _packed(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor([value for sequence in sequences for value in sequence])
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    return values.unsqueeze(1).float(), to_cumulative_lens(lengths)


class TestRepeatSequences:
    def test_copies_of_a_sequence_stay_next_to_each_other(self) -> None:
        values, cumulative_lens = _packed([[1, 2, 3], [4]])

        repeated, lengths = repeat_sequences(values, cumulative_lens, 2)

        assert repeated.flatten().tolist() == [1, 2, 3, 1, 2, 3, 4, 4]
        assert lengths.tolist() == [0, 3, 6, 7, 8]


class TestAppendToSequences:
    def test_the_extra_elements_land_at_the_end_of_their_own_sequence(self) -> None:
        values, cumulative_lens = _packed([[1, 2], [3]])
        extra = torch.tensor([[[7.0], [8.0]], [[9.0], [10.0]]])

        grown, lengths = append_to_sequences(values, cumulative_lens, extra)

        assert grown.flatten().tolist() == [1, 2, 7, 8, 3, 9, 10]
        assert lengths.tolist() == [0, 4, 7]

    def test_appending_nothing_leaves_the_batch_alone(self) -> None:
        values, cumulative_lens = _packed([[1, 2], [3]])

        grown, lengths = append_to_sequences(
            values, cumulative_lens, torch.zeros(2, 0, 1)
        )

        assert torch.equal(grown, values)
        assert torch.equal(lengths, cumulative_lens)


def _sequence_batch(sequences: list[list[int]]) -> dict:
    items = [item for sequence in sequences for item in sequence]
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    return {
        "int_columns": {
            "item": FeatureValues(
                torch.tensor(items),
                to_cumulative_lens(torch.ones(len(items), dtype=torch.int64)),
            )
        },
        "float_columns": {},
        "cumulative_lens": to_cumulative_lens(lengths),
        "timestamp": torch.arange(len(items)),
    }


class TestSplitLastEvents:
    def test_the_tail_is_the_answer_to_the_history_before_it(self) -> None:
        batch = _sequence_batch([[1, 2, 3], [4, 5]])

        history, tail = split_last_events(batch)

        assert history["int_columns"]["item"].dense().tolist() == [1, 2, 4]
        assert history["cumulative_lens"].tolist() == [0, 2, 3]
        assert tail["int_columns"]["item"].dense().tolist() == [3, 5]
        assert tail["cumulative_lens"].tolist() == [0, 1, 2]

    def test_a_sequence_with_no_history_left_drops_out_of_both_halves(self) -> None:
        batch = _sequence_batch([[1, 2], [3], [4, 5, 6]])

        history, tail = split_last_events(batch)

        assert history["int_columns"]["item"].dense().tolist() == [1, 4, 5]
        assert tail["int_columns"]["item"].dense().tolist() == [2, 6]

    def test_timestamps_follow_the_events_they_belong_to(self) -> None:
        batch = _sequence_batch([[1, 2, 3]])

        history, tail = split_last_events(batch, count=2)

        assert history["timestamp"].tolist() == [0]
        assert tail["timestamp"].tolist() == [1, 2]
