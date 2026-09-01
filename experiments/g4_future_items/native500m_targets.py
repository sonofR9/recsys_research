from __future__ import annotations

from typing import Any

import torch

from dcn.models.sequence_targets import NextItemTargets
from experiments.g4_future_items.protocol.materialization import (
    MaterializationQuery,
    MaterializedTarget,
    materialize_target,
)
from experiments.g4_future_items.targets import (
    OCCURRENCE_POSITION_COLUMN,
    FutureEvent,
    FuturePositiveTwoTowerLoss,
    FutureTargetPairs,
    FutureWindowTargets,
    PeriodArtifactTargets,
)
from neuralrec.utils import LOSS_DENOMINATOR


class Native500MFutureWindowTargets(FutureWindowTargets, NextItemTargets):
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

        candidate_rows = self._native_candidate_rows(
            output, queries, next_positions[queries]
        )
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

    def _native_candidate_rows(
        self,
        output: dict[str, Any],
        queries: torch.Tensor,
        next_positions: torch.Tensor,
    ) -> list[tuple[FutureEvent, ...]]:
        user_ids = output.get("user_ids")
        occurrence_positions = output.get("occurrence_positions")
        if user_ids is None or occurrence_positions is None:
            raise KeyError("G4 future targets require user and occurrence ids")
        query_indices = queries.detach()
        query_users = user_ids[query_indices].detach().cpu().tolist()
        query_positions = occurrence_positions[query_indices].detach().cpu().tolist()
        query_timestamps = (
            output["prefix_timestamps"][query_indices].detach().cpu().tolist()
        )
        query_items = output["prefix_item_ids"][query_indices].detach().cpu().tolist()
        is_future = output["future_query_mask"][query_indices].detach().cpu().tolist()
        next_timestamps = output["timestamps"][next_positions].detach().cpu().tolist()
        next_items = output["item_ids"][next_positions].detach().cpu().tolist()
        rows = []
        for (
            uid,
            occurrence_position,
            prefix_timestamp,
            prefix_item_id,
            future,
            next_timestamp,
            next_item,
        ) in zip(
            query_users,
            query_positions,
            query_timestamps,
            query_items,
            is_future,
            next_timestamps,
            next_items,
            strict=True,
        ):
            if not future:
                rows.append((FutureEvent(next_timestamp, next_item),))
                continue
            rows.append(
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
            )
        return rows

    def _chosen_ids(
        self,
        output: dict[str, Any],
        queries: torch.Tensor,
        candidates: list[tuple[FutureEvent, ...]],
    ) -> torch.Tensor:
        query_values = zip(
            output["user_ids"][queries].detach().cpu().tolist(),
            output["prefix_timestamps"][queries].detach().cpu().tolist(),
            output["prefix_item_ids"][queries].detach().cpu().tolist(),
            candidates,
        )
        selected = [
            (
                row[0]
                if len(row) == 1
                else row[
                    int(
                        self._query_generator(uid, timestamp, item_id).integers(
                            len(row)
                        )
                    )
                ]
            ).item_id
            for uid, timestamp, item_id, row in query_values
        ]
        return torch.tensor(selected, device=queries.device, dtype=torch.long)


class Native500MPeriodArtifactTargets(PeriodArtifactTargets, NextItemTargets):
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
            output["prefix_timestamps"][queries].detach().cpu().tolist(),
            output["prefix_item_ids"][queries].detach().cpu().tolist(),
            occurrence_positions[queries].detach().cpu().tolist(),
            output["item_ids"][next_positions].detach().cpu().tolist(),
            output["future_query_mask"][queries].detach().cpu().tolist(),
            strict=True,
        )
        results = []
        for (
            uid,
            timestamp,
            item_id,
            occurrence_position,
            next_item,
            future,
        ) in query_values:
            if not future:
                results.append(
                    MaterializedTarget(
                        target_item_id=next_item,
                        acceptable_item_ids=frozenset({next_item}),
                        selected_period_starts=(),
                        used_fallback=True,
                        rng_seed=0,
                    )
                )
                continue
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


class Native500MFuturePositiveTwoTowerLoss(FuturePositiveTwoTowerLoss):
    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | int]:
        output = self.model(batch)
        user_ids = batch["int_columns"][self.user_id_column].dense()
        occurrence_positions = batch["int_columns"][OCCURRENCE_POSITION_COLUMN].dense()
        self._attach_end_only_cls_metadata(
            batch,
            output,
            user_ids=user_ids,
            occurrence_positions=occurrence_positions,
        )
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

    def _attach_end_only_cls_metadata(
        self,
        batch: dict[str, Any],
        output: dict[str, torch.Tensor],
        *,
        user_ids: torch.Tensor,
        occurrence_positions: torch.Tensor,
    ) -> None:
        device = output["item_ids"].device
        raw_item_ids = (
            batch["int_columns"][self.model.item_id_column]
            .dense()
            .to(device=device, dtype=output["item_ids"].dtype)
        )
        raw_timestamps = batch["timestamp"].to(
            device=device, dtype=output["timestamps"].dtype
        )
        user_ids = user_ids.to(device=device, dtype=output["item_ids"].dtype)
        occurrence_positions = occurrence_positions.to(
            device=device, dtype=output["item_ids"].dtype
        )
        raw_cumulative = batch["cumulative_lens"].to(device=device, dtype=torch.long)
        output_lengths = output["lengths"].to(device=device, dtype=torch.long)
        if (
            not (
                raw_item_ids.shape
                == raw_timestamps.shape
                == user_ids.shape
                == occurrence_positions.shape
            )
            or raw_cumulative.numel() != output_lengths.numel() + 1
        ):
            raise ValueError("G4 raw event metadata does not match packed sequences")
        raw_lengths = raw_cumulative.diff()
        output_cumulative = torch.cat(
            (output_lengths.new_zeros(1), output_lengths.cumsum(0))
        )
        cls_positions = output_cumulative[1:] - 2
        inserted_cls = (
            output["is_query"] & ~output["is_target"] & output["item_ids"].eq(0)
        )
        multi_event_sequences = raw_lengths > 1
        cls_source_positions = raw_cumulative[1:][multi_event_sequences] - 2
        multi_event_cls_positions = cls_positions[multi_event_sequences]
        valid_layout = (
            (raw_lengths > 0).all()
            & output_lengths.eq(raw_lengths + 2).all()
            & raw_cumulative[0].eq(0)
            & raw_cumulative[-1].eq(raw_item_ids.shape[0])
            & output_cumulative[-1].eq(output["item_ids"].shape[0])
            & inserted_cls[cls_positions].all()
            & inserted_cls.sum().eq(raw_lengths.numel())
            & output["item_ids"][multi_event_cls_positions - 1]
            .eq(raw_item_ids[cls_source_positions])
            .all()
            & output["timestamps"][multi_event_cls_positions]
            .eq(raw_timestamps[cls_source_positions])
            .all()
        )
        if not bool(valid_layout):
            raise ValueError("BOS/end-only CLS packed layout differs from protocol")

        raw_positions = torch.arange(raw_item_ids.shape[0], device=device)
        sequence_indices = torch.zeros_like(raw_positions)
        sequence_indices[raw_cumulative[1:-1]] = 1
        sequence_indices.cumsum_(0)
        output_positions = raw_positions + 2 * sequence_indices + 1
        is_last_event = raw_positions.eq(raw_cumulative[1:][sequence_indices] - 1)
        output_positions += is_last_event

        future_query_mask = torch.zeros_like(inserted_cls)
        aligned_user_ids = torch.zeros_like(output["item_ids"])
        aligned_occurrences = torch.zeros_like(output["item_ids"])
        prefix_item_ids = torch.zeros_like(output["item_ids"])
        prefix_timestamps = torch.zeros_like(output["timestamps"])
        aligned_user_ids.index_copy_(0, output_positions, user_ids)
        aligned_occurrences.index_copy_(0, output_positions, occurrence_positions)
        prefix_item_ids.index_copy_(0, output_positions, raw_item_ids)
        prefix_timestamps.index_copy_(0, output_positions, raw_timestamps)

        singleton_sequences = ~multi_event_sequences
        singleton_raw_positions = raw_cumulative[:-1][singleton_sequences]
        singleton_output_positions = output_positions[singleton_raw_positions]
        aligned_user_ids.index_fill_(0, singleton_output_positions, 0)
        aligned_occurrences.index_fill_(0, singleton_output_positions, 0)
        prefix_item_ids.index_fill_(0, singleton_output_positions, 0)
        prefix_timestamps.index_fill_(0, singleton_output_positions, 0)

        aligned_user_ids.index_copy_(
            0, multi_event_cls_positions, user_ids[cls_source_positions]
        )
        aligned_occurrences.index_copy_(
            0,
            multi_event_cls_positions,
            occurrence_positions[cls_source_positions],
        )
        prefix_item_ids.index_copy_(
            0, multi_event_cls_positions, raw_item_ids[cls_source_positions]
        )
        prefix_timestamps.index_copy_(
            0, multi_event_cls_positions, raw_timestamps[cls_source_positions]
        )

        future_raw_events = raw_positions < raw_cumulative[1:][sequence_indices] - 2
        future_query_mask[output_positions[future_raw_events]] = True
        future_query_mask[multi_event_cls_positions] = True
        output["future_query_mask"] = future_query_mask
        output["user_ids"] = aligned_user_ids
        output["occurrence_positions"] = aligned_occurrences
        output["prefix_item_ids"] = prefix_item_ids
        output["prefix_timestamps"] = prefix_timestamps
