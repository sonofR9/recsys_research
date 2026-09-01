import hashlib
import json

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from dcn.data import SequenceDataset
from dcn.models.sequence_targets import NextItemTargets
from dcn.nn.sampled_softmax import OfflineInBatchSoftmax, RandomCatalogNegatives
from dcn.tests.helpers import scalar_feature
from dcn.training import EpochTrainer
from experiments.g4_future_items.configs.treatments import build_treatment
from experiments.g4_future_items.protocol.materialization import (
    CandidateOccurrence,
    CandidatePeriod,
    MaterializationQuery,
    PeriodArtifact,
    ScoredOccurrence,
    ScoredPeriod,
    ScoredQuery,
    materialize_target,
    write_period_artifact,
)
from experiments.g4_future_items.targets import (
    OCCURRENCE_POSITION_COLUMN,
    FutureEventIndex,
    FuturePositiveTwoTowerLoss,
    FutureWindowTargets,
    PeriodArtifactTargets,
)

DIM = 3


def _output(
    item_ids: list[int],
    lengths: list[int],
    timestamps: list[int],
    user_ids: list[int],
    occurrence_positions: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    total = len(item_ids)
    return {
        "query_repr": torch.arange(total * DIM, dtype=torch.float32).view(total, DIM),
        "item_repr": -torch.arange(total * DIM, dtype=torch.float32).view(total, DIM),
        "item_ids": torch.tensor(item_ids),
        "lengths": torch.tensor(lengths),
        "timestamps": torch.tensor(timestamps),
        "user_ids": torch.tensor(user_ids),
        "occurrence_positions": torch.tensor(
            occurrence_positions
            if occurrence_positions is not None
            else [position for length in lengths for position in range(length)]
        ),
        "is_target": torch.ones(total, dtype=torch.bool),
    }


def _index(output: dict[str, torch.Tensor]) -> FutureEventIndex:
    return FutureEventIndex.from_columns(
        user_ids=output["user_ids"].tolist(),
        timestamps=output["timestamps"].tolist(),
        item_ids=output["item_ids"].tolist(),
    )


def _query_positions(pairs) -> list[int]:
    return (pairs.query_repr[:, 0] / DIM).long().tolist()


def _acceptable_by_query(pairs) -> list[list[int]]:
    return [
        pairs.acceptable_positive_ids[start:end].tolist()
        for start, end in zip(
            pairs.acceptable_positive_offsets[:-1],
            pairs.acceptable_positive_offsets[1:],
        )
    ]


def _expected_item(
    *,
    candidates: list[tuple[int, int]],
    training_seed: int,
    epoch: int,
    objective_id: str,
    uid: int,
    prefix_timestamp: int,
    prefix_item_id: int,
) -> int:
    payload = [
        "g4-target-v1",
        training_seed,
        epoch,
        objective_id,
        uid,
        prefix_timestamp,
        prefix_item_id,
    ]
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).digest()
    generator = np.random.Generator(
        np.random.PCG64(int.from_bytes(digest[:8], byteorder="big", signed=False))
    )
    return candidates[int(generator.integers(len(candidates)))][1]


def test_next_item_target_output_stays_identical() -> None:
    output = _output(
        [10, 11, 12, 20, 21],
        [3, 2],
        [0, 1, 2, 0, 1],
        [7, 7, 7, 9, 9],
    )

    pairs = NextItemTargets()(output)

    assert type(pairs).__name__ == "TargetPairs"
    assert len(pairs) == 4
    assert _query_positions(pairs) == [0, 1, 3]
    assert pairs.positive_ids.tolist() == [11, 12, 21]
    assert pairs.group_sizes.tolist() == [2, 1]


def test_rq1_preserves_pairs_uses_strict_time_and_complete_masks() -> None:
    output = _output(
        [10, 11, 12, 13, 12, 14],
        [6],
        [0, 0, 5, 5, 10, 90_000],
        [101] * 6,
    )
    targets = FutureWindowTargets(
        objective_id="rq1_24h",
        training_seed=42,
        window_seconds=86_400,
        event_index=_index(output),
    )

    pairs = targets(output)

    assert _query_positions(pairs) == [0, 1, 2, 3, 4]
    assert pairs.group_sizes.tolist() == [5]
    assert _acceptable_by_query(pairs) == [[12, 13], [12, 13], [12], [12], [14]]
    expected_first = _expected_item(
        candidates=[(5, 12), (5, 13), (10, 12)],
        training_seed=42,
        epoch=0,
        objective_id="rq1_24h",
        uid=101,
        prefix_timestamp=0,
        prefix_item_id=10,
    )
    assert pairs.positive_ids[0].item() == expected_first
    assert pairs.positive_ids[-1].item() == 14


def test_rq2_samples_occurrences_sorted_by_timestamp_then_item() -> None:
    output = _output(
        [50, 30, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120],
        [12],
        [0, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [17] * 12,
    )
    targets = FutureWindowTargets(
        objective_id="rq2_next10",
        training_seed=9,
        event_lookahead=10,
        event_index=_index(output),
    )
    targets.set_epoch(3)

    pairs = targets(output)

    assert pairs.query_repr.shape[0] == NextItemTargets()(output).query_repr.shape[0]
    assert _acceptable_by_query(pairs)[0] == [20, 30, 40, 60, 70, 80, 90, 100, 110]
    expected = _expected_item(
        candidates=[
            (2, 20),
            (2, 30),
            (2, 30),
            (3, 40),
            (4, 60),
            (5, 70),
            (6, 80),
            (7, 90),
            (8, 100),
            (9, 110),
        ],
        training_seed=9,
        epoch=3,
        objective_id="rq2_next10",
        uid=17,
        prefix_timestamp=0,
        prefix_item_id=50,
    )
    assert pairs.positive_ids[0].item() == expected


def test_sampling_is_invariant_to_batch_traversal_order() -> None:
    first = _output(
        [10, 12, 11, 20, 23, 21],
        [3, 3],
        [0, 1, 1, 10, 11, 11],
        [7, 7, 7, 8, 8, 8],
    )
    swapped = _output(
        [20, 23, 21, 10, 12, 11],
        [3, 3],
        [10, 11, 11, 0, 1, 1],
        [8, 8, 8, 7, 7, 7],
    )
    targets = FutureWindowTargets(
        objective_id="rq2_next10",
        training_seed=42,
        event_lookahead=10,
        event_index=FutureEventIndex.from_columns(
            user_ids=first["user_ids"].tolist(),
            timestamps=first["timestamps"].tolist(),
            item_ids=first["item_ids"].tolist(),
        ),
    )

    first_pairs = targets(first)
    swapped_pairs = targets(swapped)

    first_results = {
        (int(first["user_ids"][query]), int(first["timestamps"][query])): int(item)
        for query, item in zip(_query_positions(first_pairs), first_pairs.positive_ids)
    }
    swapped_results = {
        (int(swapped["user_ids"][query]), int(swapped["timestamps"][query])): int(item)
        for query, item in zip(
            _query_positions(swapped_pairs), swapped_pairs.positive_ids
        )
    }
    assert swapped_results == first_results


def test_training_cutoff_excludes_crossing_candidates() -> None:
    output = _output([10, 11, 12], [3], [0, 9, 10], [5, 5, 5])
    targets = FutureWindowTargets(
        objective_id="rq1_24h",
        training_seed=42,
        window_seconds=86_400,
        training_cutoff_timestamp=10,
        event_index=_index(output),
    )

    pairs = targets(output)

    assert _query_positions(pairs) == [0]
    assert pairs.positive_ids.tolist() == [11]
    assert _acceptable_by_query(pairs) == [[11]]


def test_invalid_objective_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        FutureWindowTargets(
            objective_id="rq1_24h",
            training_seed=42,
            event_index=FutureEventIndex.from_columns([], [], []),
        )
    with pytest.raises(ValueError, match="event_lookahead"):
        FutureWindowTargets(
            objective_id="rq2_next10",
            training_seed=42,
            event_index=FutureEventIndex.from_columns([], [], []),
        )


def test_future_candidates_cross_local_sequence_boundary() -> None:
    full = FutureEventIndex.from_columns(
        user_ids=[4, 4, 4, 4],
        timestamps=[0, 1, 2, 3],
        item_ids=[10, 11, 12, 13],
    )
    output = _output(
        [10, 11],
        [2],
        [0, 1],
        [4, 4],
        occurrence_positions=[0, 1],
    )
    targets = FutureWindowTargets(
        objective_id="rq2_next10",
        training_seed=42,
        event_lookahead=10,
        event_index=full,
    )

    pairs = targets(output)

    assert _acceptable_by_query(pairs) == [[11, 12, 13]]


def test_sequence_dataset_emits_stable_occurrence_positions(tmp_path) -> None:
    source = tmp_path / "events.parquet"
    pl.DataFrame(
        {
            "uid": [4, 4, 4, 4],
            "timestamp": [3, 1, 2, 4],
            "item_id": [13, 11, 12, 14],
        }
    ).write_parquet(source)
    dataset = SequenceDataset(
        [source],
        ["item_id", OCCURRENCE_POSITION_COLUMN],
        tmp_path / "cache",
        user_column="uid",
        max_seq_len=2,
        min_seq_len=2,
        window="next_item",
        n_buckets=1,
    )

    assert [
        dataset[index]["int_columns"][OCCURRENCE_POSITION_COLUMN]
        for index in range(len(dataset))
    ] == [[0, 1, 2], [2, 3]]


def test_complete_acceptable_set_masks_row_specific_negatives() -> None:
    loss = OfflineInBatchSoftmax(
        q=torch.full((20,), 0.05),
        num_in_batch_negatives=2,
        correction="none",
        mask_false_negatives=False,
    ).eval()
    query_repr = torch.randn(2, 4)
    positive_repr = torch.randn(2, 4)
    positive_ids = torch.tensor([5, 6])
    negative_repr = torch.randn(2, 2, 4)
    negative_ids = torch.tensor([[7, 8], [7, 8]])

    logits = loss.logits(
        query_repr,
        positive_repr,
        positive_ids,
        torch.tensor([1, 1]),
        negatives=(negative_repr, negative_ids),
        acceptable_positive_ids=torch.tensor([5, 7, 6, 8]),
        acceptable_positive_offsets=torch.tensor([0, 2, 4]),
    )

    assert logits[0, 1] == -torch.inf
    assert torch.isfinite(logits[0, 2])
    assert torch.isfinite(logits[1, 1])
    assert logits[1, 2] == -torch.inf


def test_epoch_trainer_sets_target_epoch_before_each_training_epoch() -> None:
    class EpochTarget(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.epoch = -1

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

    class EpochLoss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.targets = EpochTarget()
            self.seen: list[int] = []

        def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            self.seen.append(self.targets.epoch)
            return {"loss": self.weight.square() * batch.mean()}

    model = EpochLoss()
    trainer = EpochTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        train_loader=DataLoader(torch.ones(1), batch_size=1),
        num_epochs=3,
    )

    trainer.train()

    assert model.seen == [0, 1, 2]


class _PeriodArtifact:
    def __init__(
        self,
        query: MaterializationQuery,
        periods: tuple[CandidatePeriod, ...],
        *,
        occurrence_position: int,
    ) -> None:
        self.query = query
        self.periods = periods
        self.occurrence_position = occurrence_position
        self.manifest = {"selector_kind": "learned"}
        self.keys: list[tuple[int, int, int, int]] = []

    def lookup(
        self,
        uid: int,
        prefix_timestamp: int,
        prefix_item_id: int,
        occurrence_position: int,
    ) -> tuple[MaterializationQuery, tuple[CandidatePeriod, ...]]:
        key = (uid, prefix_timestamp, prefix_item_id, occurrence_position)
        self.keys.append(key)
        expected = (
            self.query.uid,
            self.query.prefix_timestamp,
            self.query.prefix_item_id,
            self.occurrence_position,
        )
        if key != expected:
            raise KeyError(key)
        return self.query, self.periods


def _candidate_period(
    start: int,
    score: float,
    *occurrences: tuple[int, int],
) -> CandidatePeriod:
    return CandidatePeriod(
        start=start,
        end=start + 100,
        score=score,
        occurrences=tuple(
            CandidateOccurrence(timestamp, item_id)
            for timestamp, item_id in occurrences
        ),
    )


def test_rq3_hard_target_consumes_period_artifact_and_masks_selected_union() -> None:
    query = MaterializationQuery(17, 100, 9, 99)
    periods = (
        _candidate_period(300, 0.4, (310, 30)),
        _candidate_period(200, 0.9, (220, 20), (210, 21)),
        _candidate_period(400, 0.0, (410, 40)),
        _candidate_period(150, 0.9, (160, 10)),
    )
    artifact = _PeriodArtifact(query, periods, occurrence_position=4)
    targets = PeriodArtifactTargets(
        objective_id="rq3_learned_hard",
        training_seed=42,
        artifact=artifact,
        period_count=2,
    )
    output = _output(
        [9, 99],
        [2],
        [100, 101],
        [17, 17],
        occurrence_positions=[4, 5],
    )

    pairs = targets(output)
    expected = materialize_target(
        query,
        periods,
        objective_id="rq3_learned_hard",
        period_count=2,
        training_seed=42,
        epoch=0,
    )

    assert artifact.keys == [(17, 100, 9, 4)]
    assert pairs.group_sizes.tolist() == [1]
    assert pairs.positive_ids.tolist() == [expected.target_item_id]
    assert _acceptable_by_query(pairs) == [[10, 20, 21]]


def test_rq3_proportional_masks_all_positive_probability_periods() -> None:
    query = MaterializationQuery(17, 100, 9, 99)
    periods = (
        _candidate_period(200, 0.2, (210, 20)),
        _candidate_period(100, 0.9, (110, 10)),
        _candidate_period(300, 0.0, (310, 30)),
    )
    targets = PeriodArtifactTargets(
        objective_id="rq3_learned_proportional",
        training_seed=42,
        artifact=_PeriodArtifact(query, periods, occurrence_position=0),
        period_count=1,
    )

    pairs = targets(_output([9, 99], [2], [100, 101], [17, 17]))

    assert pairs.positive_ids.item() in {10, 20}
    assert _acceptable_by_query(pairs) == [[10, 20]]


def test_rq3_no_positive_period_falls_back_to_aligned_next_item() -> None:
    query = MaterializationQuery(17, 100, 9, 99)
    targets = PeriodArtifactTargets(
        objective_id="rq3_deterministic_hard",
        training_seed=42,
        artifact=_PeriodArtifact(
            query,
            (_candidate_period(100, 0.0, (110, 10)),),
            occurrence_position=0,
        ),
        period_count=4,
    )

    pairs = targets(_output([9, 99], [2], [100, 101], [17, 17]))

    assert pairs.positive_ids.tolist() == [99]
    assert _acceptable_by_query(pairs) == [[99]]


def test_rq3_rejects_artifact_with_a_different_next_occurrence() -> None:
    targets = PeriodArtifactTargets(
        objective_id="rq3_deterministic_hard",
        training_seed=42,
        artifact=_PeriodArtifact(
            MaterializationQuery(17, 100, 9, 98),
            (),
            occurrence_position=0,
        ),
        period_count=1,
    )

    with pytest.raises(ValueError, match="next item"):
        targets(_output([9, 99], [2], [100, 101], [17, 17]))


def test_rq3_lookup_uses_control_occurrence_order_for_equal_timestamps() -> None:
    query = MaterializationQuery(17, 100, 30, 20)
    artifact = _PeriodArtifact(query, (), occurrence_position=1)
    targets = PeriodArtifactTargets(
        objective_id="rq3_deterministic_hard",
        training_seed=42,
        artifact=artifact,
        period_count=1,
    )
    output = _output(
        [40, 30, 20],
        [3],
        [100, 100, 101],
        [17, 17, 17],
        occurrence_positions=[0, 1, 2],
    )
    output["is_query"] = torch.tensor([False, True, False])

    pairs = targets(output)

    assert artifact.keys == [(17, 100, 30, 1)]
    assert pairs.positive_ids.tolist() == [20]


@pytest.mark.parametrize(
    ("objective_id", "mask"),
    [
        ("rq3_deterministic_hard", "selected_period_union_unique"),
        ("rq3_learned_hard", "selected_period_union_unique"),
        (
            "rq3_learned_proportional",
            "all_positive_probability_periods_unique",
        ),
    ],
)
def test_rq3_treatment_config_builds_period_artifact_target(
    objective_id: str,
    mask: str,
) -> None:
    artifact = _PeriodArtifact(
        MaterializationQuery(17, 100, 9, 99),
        (),
        occurrence_position=0,
    )
    experiment = build_treatment(
        objective={
            "id": objective_id,
            "selector_artifact_sha256": "a" * 64,
            "period_count": 2,
        },
        valid_positive_mask_mode=mask,
        run_name=f"g4_{objective_id}_unit",
        batch_size=512,
        embedding_learning_rate=0.02,
        deep_learning_rate=0.01,
        lr_schedule_horizon_epochs=20,
        seed=42,
    )
    experiment.__dict__["period_artifact"] = artifact
    artifact.manifest["selector_kind"] = (
        "deterministic" if objective_id == "rq3_deterministic_hard" else "learned"
    )
    experiment.__dict__["validation_cutoff_timestamp"] = 1_700_000_000

    targets = experiment.create_targets()

    assert isinstance(targets, PeriodArtifactTargets)
    assert targets.objective_id == objective_id
    assert targets.period_count == 2
    assert targets.artifact is artifact
    assert targets.training_cutoff_timestamp == 1_700_000_000


def test_rq3_target_reads_digest_verified_mmap_period_artifact(tmp_path) -> None:
    identity = write_period_artifact(
        [
            ScoredQuery(
                uid=17,
                prefix_timestamp=100,
                prefix_item_id=9,
                occurrence_position=1,
                next_item=99,
                fold=2,
                periods=(
                    ScoredPeriod(
                        start=200,
                        end=300,
                        score=0.75,
                        occurrences=(
                            ScoredOccurrence(210, 20, 3),
                            ScoredOccurrence(210, 21, 2),
                        ),
                    ),
                ),
            ),
        ],
        selector_kind="learned",
        selected_configuration={"family": "learned"},
        provenance={"fixture": "target-consumer"},
        cost={"wall_seconds": 0.0},
        output_root=tmp_path,
    )
    artifact = PeriodArtifact.open(tmp_path, expected_sha256=identity.sha256)
    targets = PeriodArtifactTargets(
        objective_id="rq3_learned_proportional",
        training_seed=42,
        artifact=artifact,
        period_count=1,
    )
    output = _output(
        [30, 9, 99],
        [3],
        [100, 100, 101],
        [17, 17, 17],
        occurrence_positions=[0, 1, 2],
    )
    output["is_query"] = torch.tensor([False, True, False])

    pairs = targets(output)

    assert pairs.positive_ids.item() in {20, 21}
    assert _acceptable_by_query(pairs) == [[20, 21]]


def test_rq3_config_rejects_selector_artifact_from_the_wrong_family() -> None:
    experiment = build_treatment(
        objective={
            "id": "rq3_deterministic_hard",
            "selector_artifact_sha256": "a" * 64,
            "period_count": 1,
        },
        valid_positive_mask_mode="selected_period_union_unique",
        run_name="g4_wrong_selector_kind_unit",
        batch_size=512,
        embedding_learning_rate=0.02,
        deep_learning_rate=0.01,
        lr_schedule_horizon_epochs=20,
        seed=42,
    )
    experiment.__dict__["period_artifact"] = _PeriodArtifact(
        MaterializationQuery(17, 100, 9, 99),
        (),
        occurrence_position=0,
    )

    with pytest.raises(ValueError, match="artifact kind"):
        experiment.create_targets()


def test_rq3_loss_embeds_a_selected_item_outside_the_local_sequence() -> None:
    class RetrievalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.item_embedding = torch.nn.Embedding(128, 4)

        def forward(self, batch: dict) -> dict[str, torch.Tensor]:
            item_ids = batch["int_columns"]["item"].dense()
            representations = self.item_embedding(item_ids)
            return {
                "query_repr": representations,
                "item_repr": representations,
                "item_ids": item_ids,
                "lengths": batch["cumulative_lens"].diff(),
                "timestamps": batch["timestamp"],
                "is_target": torch.ones_like(item_ids, dtype=torch.bool),
            }

    model = RetrievalModel()
    query = MaterializationQuery(17, 100, 9, 99)
    targets = PeriodArtifactTargets(
        objective_id="rq3_learned_hard",
        training_seed=42,
        artifact=_PeriodArtifact(
            query,
            (_candidate_period(200, 1.0, (210, 10)),),
            occurrence_position=0,
        ),
        period_count=1,
    )
    random_negatives = RandomCatalogNegatives(
        catalog_size=128,
        num_negatives=1,
        item_encoder=model.item_embedding,
        probabilities=torch.nn.functional.one_hot(
            torch.tensor(12), num_classes=128
        ).float(),
    )
    criterion = FuturePositiveTwoTowerLoss(
        model,
        OfflineInBatchSoftmax(
            q=torch.full((128,), 1 / 128),
            num_in_batch_negatives=0,
            correction="none",
            random_negatives=random_negatives,
        ),
        targets=targets,
        user_id_column="uid",
    )
    batch = {
        "int_columns": {
            "item": scalar_feature(torch.tensor([9, 99])),
            "uid": scalar_feature(torch.tensor([17, 17])),
            OCCURRENCE_POSITION_COLUMN: scalar_feature(torch.tensor([0, 1])),
        },
        "float_columns": {},
        "cumulative_lens": torch.tensor([0, 2]),
        "timestamp": torch.tensor([100, 101]),
    }

    output = criterion(batch)
    output["loss"].backward()

    assert torch.isfinite(output["loss"])
    assert model.item_embedding.weight.grad[10].abs().sum() > 0

    criterion.eval()
    validation_batch = {
        **batch,
        "int_columns": {
            "item": scalar_feature(torch.tensor([11, 12])),
            "uid": scalar_feature(torch.tensor([999, 999])),
            OCCURRENCE_POSITION_COLUMN: scalar_feature(torch.tensor([0, 1])),
        },
        "timestamp": torch.tensor([500, 501]),
    }
    validation_output = criterion(validation_batch)

    assert torch.isfinite(validation_output["loss"])


def test_miniature_g4_training_updates_model() -> None:
    class RetrievalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.item_embedding = torch.nn.Embedding(32, 4)

        def forward(self, batch: dict) -> dict[str, torch.Tensor]:
            item_ids = batch["int_columns"]["item"].dense()
            representations = self.item_embedding(item_ids)
            return {
                "query_repr": representations,
                "item_repr": representations,
                "item_ids": item_ids,
                "lengths": batch["cumulative_lens"].diff(),
                "timestamps": batch["timestamp"],
                "is_target": torch.ones_like(item_ids, dtype=torch.bool),
            }

    model = RetrievalModel()
    targets = FutureWindowTargets(
        objective_id="rq2_next10",
        training_seed=42,
        event_lookahead=10,
        event_index=FutureEventIndex.from_columns(
            user_ids=[17, 17, 17],
            timestamps=[100, 101, 102],
            item_ids=[9, 10, 11],
        ),
    )
    criterion = FuturePositiveTwoTowerLoss(
        model,
        OfflineInBatchSoftmax(
            q=torch.full((32,), 1 / 32),
            num_in_batch_negatives=0,
            correction="none",
            random_negatives=RandomCatalogNegatives(
                catalog_size=32,
                num_negatives=1,
                item_encoder=model.item_embedding,
                probabilities=torch.nn.functional.one_hot(
                    torch.tensor(12), num_classes=32
                ).float(),
            ),
        ),
        targets=targets,
        user_id_column="uid",
    )
    batch = {
        "int_columns": {
            "item": scalar_feature(torch.tensor([9, 10, 11])),
            "uid": scalar_feature(torch.tensor([17, 17, 17])),
            OCCURRENCE_POSITION_COLUMN: scalar_feature(torch.tensor([0, 1, 2])),
        },
        "float_columns": {},
        "cumulative_lens": torch.tensor([0, 3]),
        "timestamp": torch.tensor([100, 101, 102]),
    }
    before = model.item_embedding.weight.detach().clone()
    trainer = EpochTrainer(
        model=criterion,
        optimizer=torch.optim.SGD(criterion.parameters(), lr=0.1),
        train_loader=[batch],
        num_epochs=2,
    )

    trainer.train()

    assert targets.epoch == 1
    assert not torch.equal(before, model.item_embedding.weight)
