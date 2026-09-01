from types import SimpleNamespace

import torch

from experiments.g4_future_items.native500m_targets import (
    Native500MFuturePositiveTwoTowerLoss,
    Native500MPeriodArtifactTargets,
)
from experiments.g4_future_items.protocol.materialization import MaterializationQuery
from dcn.tests.experiments.g4_future_items.test_targets import (
    _PeriodArtifact,
    _candidate_period,
    _output,
)


class _DenseColumn:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def dense(self) -> torch.Tensor:
        return self.values


def _packed_end_cls_fixture(
    raw_lengths: list[int],
) -> tuple[
    dict[str, object],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    lengths = torch.tensor(raw_lengths, dtype=torch.long)
    raw_cumulative = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    output_lengths = lengths + 2
    output_cumulative = torch.cat(
        (torch.zeros(1, dtype=torch.long), output_lengths.cumsum(0))
    )
    raw_count = int(raw_cumulative[-1])
    raw_item_ids = torch.arange(11, 11 + raw_count)
    raw_timestamps = torch.arange(101, 101 + raw_count)
    user_ids = torch.repeat_interleave(torch.arange(41, 41 + len(lengths)), lengths)
    occurrences = torch.cat([torch.arange(length) for length in raw_lengths])
    output_count = int(output_cumulative[-1])
    item_ids = torch.zeros(output_count, dtype=torch.long)
    timestamps = torch.zeros(output_count, dtype=torch.long)
    is_query = torch.zeros(output_count, dtype=torch.bool)
    is_target = torch.zeros(output_count, dtype=torch.bool)
    for sequence_index, raw_length in enumerate(raw_lengths):
        raw_start = int(raw_cumulative[sequence_index])
        raw_end = int(raw_cumulative[sequence_index + 1])
        output_start = int(output_cumulative[sequence_index])
        output_end = int(output_cumulative[sequence_index + 1])
        item_ids[output_start + 1 : output_start + raw_length] = raw_item_ids[
            raw_start : raw_end - 1
        ]
        timestamps[output_start + 1 : output_start + raw_length] = raw_timestamps[
            raw_start : raw_end - 1
        ]
        query_position = output_start + raw_length
        timestamps[query_position] = raw_timestamps[
            raw_end - 2 if raw_length > 1 else raw_start
        ]
        is_query[query_position] = True
        item_ids[output_end - 1] = raw_item_ids[raw_end - 1]
        timestamps[output_end - 1] = raw_timestamps[raw_end - 1]
        is_target[output_end - 1] = True
    batch: dict[str, object] = {
        "int_columns": {"item": _DenseColumn(raw_item_ids)},
        "timestamp": raw_timestamps,
        "cumulative_lens": raw_cumulative,
    }
    output = {
        "item_ids": item_ids,
        "timestamps": timestamps,
        "lengths": output_lengths,
        "is_query": is_query,
        "is_target": is_target,
    }
    return batch, output, user_ids, occurrences


def _scalar_reference(
    batch: dict[str, object],
    output: dict[str, torch.Tensor],
    user_ids: torch.Tensor,
    occurrence_positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    raw_item_ids = batch["int_columns"]["item"].dense()
    raw_timestamps = batch["timestamp"]
    raw_cumulative = batch["cumulative_lens"].tolist()
    output_cumulative = [0, *output["lengths"].cumsum(0).tolist()]
    inserted_cls = output["is_query"] & ~output["is_target"] & output["item_ids"].eq(0)
    expected = {
        "future_query_mask": torch.zeros_like(inserted_cls),
        "user_ids": torch.zeros_like(output["item_ids"]),
        "occurrence_positions": torch.zeros_like(output["item_ids"]),
        "prefix_item_ids": torch.zeros_like(output["item_ids"]),
        "prefix_timestamps": torch.zeros_like(output["timestamps"]),
        "is_query": output["is_query"].clone(),
    }
    for sequence_index in range(len(raw_cumulative) - 1):
        raw_start, raw_end = raw_cumulative[sequence_index : sequence_index + 2]
        output_start, output_end = output_cumulative[
            sequence_index : sequence_index + 2
        ]
        query_position = int(
            inserted_cls[output_start:output_end].nonzero(as_tuple=True)[0].item()
            + output_start
        )
        raw_length = raw_end - raw_start
        if raw_length == 1:
            continue
        cls_source_position = raw_end - 2
        for raw_position in range(raw_start, raw_end):
            relative_position = raw_position - raw_start
            output_position = (
                output_start + 1 + relative_position
                if raw_position < raw_end - 1
                else output_end - 1
            )
            expected["user_ids"][output_position] = user_ids[raw_position]
            expected["occurrence_positions"][output_position] = occurrence_positions[
                raw_position
            ]
            expected["prefix_item_ids"][output_position] = raw_item_ids[raw_position]
            expected["prefix_timestamps"][output_position] = raw_timestamps[
                raw_position
            ]
            if raw_position < raw_end - 2:
                expected["future_query_mask"][output_position] = True
        expected["user_ids"][query_position] = user_ids[cls_source_position]
        expected["occurrence_positions"][query_position] = occurrence_positions[
            cls_source_position
        ]
        expected["prefix_item_ids"][query_position] = raw_item_ids[cls_source_position]
        expected["prefix_timestamps"][query_position] = raw_timestamps[
            cls_source_position
        ]
        expected["future_query_mask"][query_position] = True
    return expected


def test_end_only_cls_metadata_matches_scalar_reference_for_mixed_lengths() -> None:
    batch, output, user_ids, occurrences = _packed_end_cls_fixture([1, 2, 3, 5, 8])
    expected = _scalar_reference(batch, output, user_ids, occurrences)
    criterion = Native500MFuturePositiveTwoTowerLoss.__new__(
        Native500MFuturePositiveTwoTowerLoss
    )
    criterion.model = SimpleNamespace(item_id_column="item")

    criterion._attach_end_only_cls_metadata(
        batch,
        output,
        user_ids=user_ids,
        occurrence_positions=occurrences,
    )

    for name, expected_values in expected.items():
        torch.testing.assert_close(output[name], expected_values, rtol=0, atol=0)


def test_rq3_cls_query_uses_real_prefix_and_actual_next_target() -> None:
    query = MaterializationQuery(17, 100, 9, 99)
    artifact = _PeriodArtifact(
        query,
        (_candidate_period(200, 1.0, (210, 20)),),
        occurrence_position=4,
    )
    targets = Native500MPeriodArtifactTargets(
        objective_id="rq3_learned_hard",
        training_seed=42,
        artifact=artifact,
        period_count=1,
    )
    output = _output(
        [0, 99],
        [2],
        [100, 101],
        [17, 0],
        occurrence_positions=[4, 0],
    )
    output["is_query"] = torch.tensor([True, False])
    output["future_query_mask"] = torch.tensor([True, False])
    output["prefix_item_ids"] = torch.tensor([9, 0])
    output["prefix_timestamps"] = torch.tensor([100, 0])

    pairs = targets(output)

    assert artifact.keys == [(17, 100, 9, 4)]
    assert pairs.positive_ids.item() == 20
