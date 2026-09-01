from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol

import numpy as np
import torch

from dcn.models.sequence_targets import NextItemTargets, SequenceTargets
from dcn.models.two_tower import TwoTowerLoss
from experiments.g4_future_items.protocol.materialization import (
    CandidatePeriod,
    MaterializationQuery,
    MaterializedTarget,
    materialize_target,
)
from neuralrec.utils import LOSS_DENOMINATOR

FixedWindowObjectiveId = Literal["rq1_24h", "rq2_next10"]
PeriodObjectiveId = Literal[
    "rq3_deterministic_hard",
    "rq3_learned_hard",
    "rq3_learned_proportional",
]
ObjectiveId = FixedWindowObjectiveId | PeriodObjectiveId
TARGET_SEED_REVISION = "g4-target-v1"
OCCURRENCE_POSITION_COLUMN = "_g4_occurrence_position"


@dataclass(frozen=True)
class FutureEvent:
    timestamp: int
    item_id: int


class FutureEventIndex:
    def __init__(self, events_by_user: dict[int, tuple[FutureEvent, ...]]) -> None:
        self._events_by_user = events_by_user

    @classmethod
    def from_columns(
        cls,
        user_ids: Sequence[int],
        timestamps: Sequence[int],
        item_ids: Sequence[int],
    ) -> FutureEventIndex:
        if not (len(user_ids) == len(timestamps) == len(item_ids)):
            raise ValueError("future-event columns must have equal lengths")
        indexed: dict[int, list[tuple[int, FutureEvent]]] = defaultdict(list)
        for order, (uid, timestamp, item_id) in enumerate(
            zip(user_ids, timestamps, item_ids)
        ):
            indexed[int(uid)].append(
                (order, FutureEvent(timestamp=int(timestamp), item_id=int(item_id)))
            )
        return cls(
            {
                uid: tuple(
                    event
                    for _, event in sorted(
                        events,
                        key=lambda indexed_event: (
                            indexed_event[1].timestamp,
                            indexed_event[0],
                        ),
                    )
                )
                for uid, events in indexed.items()
            }
        )

    def candidates(
        self,
        *,
        uid: int,
        occurrence_position: int,
        prefix_timestamp: int,
        prefix_item_id: int,
        objective_id: FixedWindowObjectiveId,
        window_seconds: int | None,
        event_lookahead: int | None,
        training_cutoff_timestamp: int | None,
    ) -> tuple[FutureEvent, ...]:
        try:
            events = self._events_by_user[uid]
            prefix = events[occurrence_position]
        except (KeyError, IndexError) as error:
            raise ValueError(
                "training query is absent from the future-event index"
            ) from error
        if (prefix.timestamp, prefix.item_id) != (prefix_timestamp, prefix_item_id):
            raise ValueError("training query disagrees with the future-event index")
        if occurrence_position + 1 >= len(events):
            return ()
        if objective_id == "rq2_next10":
            assert event_lookahead is not None
            candidates = events[
                occurrence_position + 1 : occurrence_position + 1 + event_lookahead
            ]
        else:
            assert window_seconds is not None
            selected = []
            end_timestamp = prefix_timestamp + window_seconds
            for event in events[occurrence_position + 1 :]:
                if event.timestamp > end_timestamp:
                    break
                if event.timestamp > prefix_timestamp:
                    selected.append(event)
            candidates = tuple(selected)
        if training_cutoff_timestamp is not None:
            candidates = tuple(
                event
                for event in candidates
                if event.timestamp < training_cutoff_timestamp
            )
        if not candidates:
            fallback = events[occurrence_position + 1]
            if (
                training_cutoff_timestamp is not None
                and fallback.timestamp >= training_cutoff_timestamp
            ):
                return ()
            candidates = (fallback,)
        return tuple(
            sorted(candidates, key=lambda event: (event.timestamp, event.item_id))
        )


class FutureTargetPairs(NamedTuple):
    query_repr: torch.Tensor
    positive_repr: torch.Tensor
    positive_ids: torch.Tensor
    group_sizes: torch.Tensor
    acceptable_positive_ids: torch.Tensor
    acceptable_positive_offsets: torch.Tensor


class FutureWindowTargets(SequenceTargets):
    def __init__(
        self,
        *,
        objective_id: FixedWindowObjectiveId,
        training_seed: int,
        event_index: FutureEventIndex,
        window_seconds: int | None = None,
        event_lookahead: int | None = None,
        training_cutoff_timestamp: int | None = None,
    ) -> None:
        super().__init__()
        if objective_id == "rq1_24h":
            if (
                not isinstance(window_seconds, int)
                or isinstance(window_seconds, bool)
                or window_seconds <= 0
            ):
                raise ValueError("rq1_24h requires positive integer window_seconds")
            if event_lookahead is not None:
                raise ValueError("event_lookahead applies only to rq2_next10")
        elif objective_id == "rq2_next10":
            if (
                not isinstance(event_lookahead, int)
                or isinstance(event_lookahead, bool)
                or event_lookahead <= 0
            ):
                raise ValueError("rq2_next10 requires positive integer event_lookahead")
            if window_seconds is not None:
                raise ValueError("window_seconds applies only to rq1_24h")
        else:
            raise ValueError(f"unsupported G4 objective {objective_id!r}")
        if not isinstance(training_seed, int) or isinstance(training_seed, bool):
            raise TypeError("training_seed must be an integer")
        if training_cutoff_timestamp is not None and (
            not isinstance(training_cutoff_timestamp, int)
            or isinstance(training_cutoff_timestamp, bool)
        ):
            raise TypeError("training_cutoff_timestamp must be an integer or None")
        self.objective_id = objective_id
        self.training_seed = training_seed
        self.event_index = event_index
        self.window_seconds = window_seconds
        self.event_lookahead = event_lookahead
        self.training_cutoff_timestamp = training_cutoff_timestamp
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def forward(self, output: dict[str, Any]) -> FutureTargetPairs:
        layout = self._layout(output)
        if layout.positions.numel() == 0:
            return self._empty_pairs(output)

        next_positions = layout.at_rank(layout.next_rank)
        has_next = (layout.next_rank <= layout.last_rank) & (
            next_positions < layout.sequence_end
        )
        if self.training_cutoff_timestamp is not None:
            has_next &= (
                output["timestamps"][next_positions] < self.training_cutoff_timestamp
            )
        if output.get("is_query") is not None:
            has_next &= output["is_query"]
        queries = has_next.nonzero(as_tuple=True)[0]
        if queries.numel() == 0:
            return self._empty_pairs(output)

        candidate_rows = self._candidate_rows(output, queries)
        chosen_ids = self._chosen_ids(output, queries, candidate_rows)
        group_sizes = torch.bincount(
            layout.sequence_of_token[queries], minlength=output["lengths"].shape[0]
        )
        acceptable_ids, acceptable_offsets = self._acceptable_sets(
            output["item_ids"], candidate_rows
        )
        return FutureTargetPairs(
            output["query_repr"][queries],
            output["item_repr"][queries],
            chosen_ids,
            group_sizes[group_sizes > 0],
            acceptable_ids,
            acceptable_offsets,
        )

    def _candidate_rows(
        self, output: dict[str, Any], queries: torch.Tensor
    ) -> list[tuple[FutureEvent, ...]]:
        user_ids = output.get("user_ids")
        occurrence_positions = output.get("occurrence_positions")
        if user_ids is None or occurrence_positions is None:
            raise KeyError("G4 future targets require user and occurrence ids")
        query_indices = queries.detach()
        query_users = user_ids[query_indices].detach().cpu().tolist()
        query_positions = occurrence_positions[query_indices].detach().cpu().tolist()
        query_timestamps = output["timestamps"][query_indices].detach().cpu().tolist()
        query_items = output["item_ids"][query_indices].detach().cpu().tolist()
        return [
            self.event_index.candidates(
                uid=uid,
                occurrence_position=occurrence_position,
                prefix_timestamp=prefix_timestamp,
                prefix_item_id=prefix_item_id,
                objective_id=self.objective_id,
                window_seconds=self.window_seconds,
                event_lookahead=self.event_lookahead,
                training_cutoff_timestamp=self.training_cutoff_timestamp,
            )
            for uid, occurrence_position, prefix_timestamp, prefix_item_id in zip(
                query_users, query_positions, query_timestamps, query_items
            )
        ]

    def _chosen_ids(
        self,
        output: dict[str, Any],
        queries: torch.Tensor,
        candidates: list[tuple[FutureEvent, ...]],
    ) -> torch.Tensor:
        user_ids = output.get("user_ids")
        if user_ids is None:
            raise KeyError("G4 future targets require output['user_ids']")
        if user_ids.shape == output["item_ids"].shape:
            query_user_ids = user_ids[queries]
        elif user_ids.shape == output["lengths"].shape:
            layout = self._layout(output)
            query_user_ids = user_ids[layout.sequence_of_token[queries]]
        else:
            raise ValueError("user_ids must contain one id per token or sequence")

        query_values = zip(
            query_user_ids.detach().cpu().tolist(),
            output["timestamps"][queries].detach().cpu().tolist(),
            output["item_ids"][queries].detach().cpu().tolist(),
            candidates,
        )
        selected = [
            row[
                int(self._query_generator(uid, timestamp, item_id).integers(len(row)))
            ].item_id
            for uid, timestamp, item_id, row in query_values
        ]
        return torch.tensor(selected, device=queries.device, dtype=torch.long)

    def _query_generator(
        self, uid: int, prefix_timestamp: int, prefix_item_id: int
    ) -> np.random.Generator:
        payload = [
            TARGET_SEED_REVISION,
            self.training_seed,
            self.epoch,
            self.objective_id,
            uid,
            prefix_timestamp,
            prefix_item_id,
        ]
        canonical = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(canonical).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return np.random.Generator(np.random.PCG64(seed))

    @staticmethod
    def _acceptable_sets(
        item_ids: torch.Tensor,
        candidates: Iterable[tuple[FutureEvent, ...]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [sorted({event.item_id for event in row}) for row in candidates]
        offsets = torch.tensor(
            [0, *np.cumsum([len(row) for row in rows]).tolist()],
            device=item_ids.device,
            dtype=torch.long,
        )
        flattened = [item_id for row in rows for item_id in row]
        return item_ids.new_tensor(flattened), offsets

    @staticmethod
    def _empty_pairs(output: dict[str, Any]) -> FutureTargetPairs:
        base = SequenceTargets._empty(output)
        return FutureTargetPairs(
            base.query_repr,
            base.positive_repr,
            base.positive_ids,
            base.group_sizes,
            output["item_ids"][:0],
            output["item_ids"].new_zeros(1),
        )


class PeriodArtifactReader(Protocol):
    def lookup(
        self,
        uid: int,
        prefix_timestamp: int,
        prefix_item_id: int,
        occurrence_position: int,
    ) -> tuple[MaterializationQuery, tuple[CandidatePeriod, ...]]: ...


class PeriodArtifactTargets(SequenceTargets):
    def __init__(
        self,
        *,
        objective_id: PeriodObjectiveId,
        training_seed: int,
        artifact: PeriodArtifactReader,
        period_count: int,
        training_cutoff_timestamp: int | None = None,
    ) -> None:
        super().__init__()
        if objective_id not in {
            "rq3_deterministic_hard",
            "rq3_learned_hard",
            "rq3_learned_proportional",
        }:
            raise ValueError(f"unsupported RQ3 objective {objective_id!r}")
        if period_count not in {1, 2, 4}:
            raise ValueError("period_count must be 1, 2, or 4")
        if not isinstance(training_seed, int) or isinstance(training_seed, bool):
            raise TypeError("training_seed must be an integer")
        if training_cutoff_timestamp is not None and (
            not isinstance(training_cutoff_timestamp, int)
            or isinstance(training_cutoff_timestamp, bool)
        ):
            raise TypeError("training_cutoff_timestamp must be an integer or None")
        self.objective_id = objective_id
        self.training_seed = training_seed
        self.artifact = artifact
        self.period_count = period_count
        self.training_cutoff_timestamp = training_cutoff_timestamp
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def forward(self, output: dict[str, Any]) -> FutureTargetPairs:
        layout = self._layout(output)
        if layout.positions.numel() == 0:
            return FutureWindowTargets._empty_pairs(output)
        next_positions = layout.at_rank(layout.next_rank)
        has_next = (layout.next_rank <= layout.last_rank) & (
            next_positions < layout.sequence_end
        )
        if self.training_cutoff_timestamp is not None:
            has_next &= (
                output["timestamps"][next_positions] < self.training_cutoff_timestamp
            )
        if output.get("is_query") is not None:
            has_next &= output["is_query"]
        queries = has_next.nonzero(as_tuple=True)[0]
        if queries.numel() == 0:
            return FutureWindowTargets._empty_pairs(output)

        results = self._materialized_rows(output, queries, next_positions[queries])
        group_sizes = torch.bincount(
            layout.sequence_of_token[queries], minlength=output["lengths"].shape[0]
        )
        acceptable_rows = [sorted(result.acceptable_item_ids) for result in results]
        acceptable_offsets = output["item_ids"].new_tensor(
            [0, *np.cumsum([len(row) for row in acceptable_rows]).tolist()]
        )
        return FutureTargetPairs(
            output["query_repr"][queries],
            output["item_repr"][queries],
            output["item_ids"].new_tensor(
                [result.target_item_id for result in results]
            ),
            group_sizes[group_sizes > 0],
            output["item_ids"].new_tensor(
                [item_id for row in acceptable_rows for item_id in row]
            ),
            acceptable_offsets,
        )

    def _materialized_rows(
        self,
        output: dict[str, Any],
        queries: torch.Tensor,
        next_positions: torch.Tensor,
    ) -> list[MaterializedTarget]:
        user_ids = output.get("user_ids")
        occurrence_positions = output.get("occurrence_positions")
        if user_ids is None or occurrence_positions is None:
            raise KeyError("RQ3 targets require user and occurrence ids")
        if (
            user_ids.shape != output["item_ids"].shape
            or occurrence_positions.shape != output["item_ids"].shape
        ):
            raise ValueError("RQ3 user and occurrence ids must align with model tokens")
        query_values = zip(
            user_ids[queries].detach().cpu().tolist(),
            output["timestamps"][queries].detach().cpu().tolist(),
            output["item_ids"][queries].detach().cpu().tolist(),
            occurrence_positions[queries].detach().cpu().tolist(),
            output["item_ids"][next_positions].detach().cpu().tolist(),
        )
        results = []
        for uid, timestamp, item_id, occurrence_position, next_item in query_values:
            query, periods = self.artifact.lookup(
                uid,
                timestamp,
                item_id,
                occurrence_position,
            )
            expected_query = MaterializationQuery(uid, timestamp, item_id, next_item)
            if query != expected_query:
                raise ValueError("RQ3 artifact next item or query identity disagrees")
            results.append(
                materialize_target(
                    query,
                    periods,
                    objective_id=self.objective_id,
                    period_count=self.period_count,
                    training_seed=self.training_seed,
                    epoch=self.epoch,
                )
            )
        return results


class FuturePositiveTwoTowerLoss(TwoTowerLoss):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        *,
        targets: FutureWindowTargets | PeriodArtifactTargets,
        user_id_column: str,
    ) -> None:
        super().__init__(model, loss, targets=targets)
        self.user_id_column = user_id_column
        self.validation_targets = NextItemTargets()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | int]:
        output = self.model(batch)
        user_ids = batch["int_columns"][self.user_id_column].dense()
        occurrence_positions = batch["int_columns"][OCCURRENCE_POSITION_COLUMN].dense()
        if (
            user_ids.shape != output["item_ids"].shape
            or occurrence_positions.shape != output["item_ids"].shape
        ):
            raise ValueError(
                "G4 user and occurrence ids must align one-to-one with model tokens"
            )
        output["user_ids"] = user_ids
        output["occurrence_positions"] = occurrence_positions
        pairs = (
            self.targets(output) if self.training else self.validation_targets(output)
        )
        if pairs.query_repr.shape[0] == 0:
            zero = (output["query_repr"].sum() + output["item_repr"].sum()) * 0.0
            return {
                "loss": zero,
                "hit_rate": zero.detach(),
                LOSS_DENOMINATOR: 0,
            }
        if self.training:
            logits = self.loss.logits(
                pairs.query_repr,
                self.model.item_embedding(pairs.positive_ids),
                pairs.positive_ids,
                pairs.group_sizes,
                acceptable_positive_ids=pairs.acceptable_positive_ids,
                acceptable_positive_offsets=pairs.acceptable_positive_offsets,
            )
        else:
            logits = self.loss.logits(
                pairs.query_repr,
                pairs.positive_repr,
                pairs.positive_ids,
                pairs.group_sizes,
            )
        return {
            "loss": self.loss.loss_from_logits(logits),
            "hit_rate": (logits.detach().argmax(dim=1) == 0).float().mean(),
            LOSS_DENOMINATOR: pairs.query_repr.shape[0],
        }
