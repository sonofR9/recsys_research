from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import torch

from dcn.eval.ranking_evidence import RankingEvidence
from dcn.semantic import SemanticCodes


_CUTOFFS = (10, 50, 100)


def slice_comparison(
    control: RankingEvidence,
    semantic: RankingEvidence,
    *,
    semantic_codes: SemanticCodes,
    semantic_base_levels: int,
    control_run_name: str,
    semantic_run_name: str,
) -> dict[str, object]:
    _require_shared_context(control, semantic)
    if semantic_base_levels < 1 or semantic_base_levels > semantic_codes.num_levels:
        raise ValueError("semantic base levels do not select existing code levels")
    if min(control.max_k, semantic.max_k) < max(_CUTOFFS):
        raise ValueError("slice evidence must retain ranks through 100")

    frequencies = control.relevant_train_frequencies.tolist()
    ordered_frequencies = sorted(frequencies)
    low_boundary = ordered_frequencies[(len(ordered_frequencies) - 1) // 3]
    middle_boundary = ordered_frequencies[2 * (len(ordered_frequencies) - 1) // 3]
    target_masks = {
        "frequency_low": lambda value: value <= low_boundary,
        "frequency_middle": lambda value: low_boundary < value <= middle_boundary,
        "frequency_high": lambda value: value > middle_boundary,
    }
    collided_items = _collided_item_ids(semantic_codes, semantic_base_levels)
    collided_users = _history_collision_mask(control, collided_items)
    user_masks = {
        "history_has_collided_base_sid": collided_users,
        "history_has_no_collided_base_sid": ~collided_users,
    }

    slices: dict[str, object] = {}
    for name, include_target in target_masks.items():
        slices[name] = _comparison_for_targets(control, semantic, include_target)
    for name, include_user in user_masks.items():
        slices[name] = _comparison_for_users(control, semantic, include_user)

    return {
        "schema_version": 1,
        "dataset_size": "native-50m",
        "comparison": {
            "control_run_name": control_run_name,
            "semantic_run_name": semantic_run_name,
        },
        "frequency_terciles": {
            "population": "evaluable final-evaluation targets",
            "ordering": "target train-event frequency",
            "boundaries": [low_boundary, middle_boundary],
        },
        "collision_slice": {
            "base_levels": semantic_base_levels,
            "collided_item_count": len(collided_items),
            "definition": "query history contains an item in a non-singleton base-SID bucket",
        },
        "metrics": [
            f"{metric}@{cutoff}"
            for metric in ("recall", "ndcg", "mrr", "capped_recall")
            for cutoff in _CUTOFFS
        ],
        "slices": slices,
    }


def bucket_size_comparison(
    control: RankingEvidence,
    semantic: RankingEvidence,
    *,
    semantic_codes: SemanticCodes,
    semantic_base_levels: int,
) -> dict[str, object]:
    _require_shared_context(control, semantic)
    if semantic_base_levels < 1 or semantic_base_levels > semantic_codes.num_levels:
        raise ValueError("semantic base levels do not select existing code levels")
    if min(control.max_k, semantic.max_k) < max(_CUTOFFS):
        raise ValueError("slice evidence must retain ranks through 100")

    base_codes = semantic_codes.codes[:, :semantic_base_levels]
    _, inverse, counts = torch.unique(
        base_codes, dim=0, return_inverse=True, return_counts=True
    )
    bucket_size_by_item = {
        int(item_id): int(bucket_size)
        for item_id, bucket_size in zip(
            semantic_codes.item_ids.tolist(),
            counts[inverse].tolist(),
            strict=True,
        )
    }
    try:
        target_bucket_sizes = torch.tensor(
            [
                bucket_size_by_item[int(item_id)]
                for item_id in control.relevant_item_ids.tolist()
            ],
            dtype=torch.int64,
        )
    except KeyError as error:
        raise ValueError("slice target is outside the fitted semantic catalog") from error

    masks = {
        "bucket_size_1": target_bucket_sizes == 1,
        "bucket_size_2": target_bucket_sizes == 2,
        "bucket_size_3_to_4": (target_bucket_sizes >= 3)
        & (target_bucket_sizes <= 4),
        "bucket_size_5_plus": target_bucket_sizes >= 5,
    }
    return {
        "schema_version": 1,
        "base_levels": semantic_base_levels,
        "definition": "base-SID collision-bucket size of the relevant target item",
        "maximum_bucket_size": int(counts.max()),
        "slices": {
            name: _comparison(control, semantic, mask)
            for name, mask in masks.items()
        },
    }


def _require_shared_context(
    control: RankingEvidence, semantic: RankingEvidence
) -> None:
    for name in (
        "user_ids",
        "history_item_ids",
        "history_offsets",
        "relevant_item_ids",
        "relevance_offsets",
        "relevant_train_frequencies",
    ):
        if not torch.equal(getattr(control, name), getattr(semantic, name)):
            raise ValueError(f"selected runs have different {name}")


def _collided_item_ids(codes: SemanticCodes, base_levels: int) -> set[int]:
    base = codes.codes[:, :base_levels]
    _, inverse, counts = torch.unique(
        base, dim=0, return_inverse=True, return_counts=True
    )
    collided = counts[inverse] > 1
    return set(codes.item_ids[collided].tolist())


def _history_collision_mask(
    evidence: RankingEvidence, collided_items: set[int]
) -> torch.Tensor:
    flags = []
    for start, end in zip(
        evidence.history_offsets[:-1].tolist(),
        evidence.history_offsets[1:].tolist(),
        strict=True,
    ):
        flags.append(
            any(
                item_id in collided_items
                for item_id in evidence.history_item_ids[start:end].tolist()
            )
        )
    return torch.tensor(flags, dtype=torch.bool)


def _comparison_for_targets(
    control: RankingEvidence,
    semantic: RankingEvidence,
    include: Callable[[int], bool],
) -> dict[str, object]:
    selections = [
        include(value) for value in control.relevant_train_frequencies.tolist()
    ]
    return _comparison(control, semantic, torch.tensor(selections, dtype=torch.bool))


def _comparison_for_users(
    control: RankingEvidence,
    semantic: RankingEvidence,
    include_users: torch.Tensor,
) -> dict[str, object]:
    selections = torch.zeros_like(control.relevant_item_ids, dtype=torch.bool)
    for user, (start, end) in enumerate(
        zip(
            control.relevance_offsets[:-1].tolist(),
            control.relevance_offsets[1:].tolist(),
            strict=True,
        )
    ):
        if bool(include_users[user]):
            selections[start:end] = True
    return _comparison(control, semantic, selections)


def _comparison(
    control: RankingEvidence,
    semantic: RankingEvidence,
    selected_targets: torch.Tensor,
) -> dict[str, object]:
    selected_users = 0
    selected_target_count = int(selected_targets.sum())
    for start, end in zip(
        control.relevance_offsets[:-1].tolist(),
        control.relevance_offsets[1:].tolist(),
        strict=True,
    ):
        selected_users += int(bool(selected_targets[start:end].any()))
    return {
        "num_users": selected_users,
        "num_targets": selected_target_count,
        "control": _metrics(control, selected_targets),
        "semantic": _metrics(semantic, selected_targets),
    }


def _metrics(
    evidence: RankingEvidence, selected_targets: torch.Tensor
) -> dict[str, float]:
    sums = {
        f"{metric}@{cutoff}": 0.0
        for metric in ("recall", "ndcg", "mrr", "capped_recall")
        for cutoff in _CUTOFFS
    }
    users = 0
    for start, end in zip(
        evidence.relevance_offsets[:-1].tolist(),
        evidence.relevance_offsets[1:].tolist(),
        strict=True,
    ):
        mask = selected_targets[start:end]
        if not bool(mask.any()):
            continue
        users += 1
        ranks = evidence.relevant_ranks[start:end][mask]
        target_count = ranks.shape[0]
        for cutoff in _CUTOFFS:
            hits = ranks[(ranks > 0) & (ranks <= cutoff)]
            hit_count = hits.shape[0]
            ideal_length = min(target_count, cutoff)
            sums[f"recall@{cutoff}"] += hit_count / target_count
            sums[f"capped_recall@{cutoff}"] += hit_count / ideal_length
            sums[f"ndcg@{cutoff}"] += sum(
                1.0 / math.log2(int(rank) + 1) for rank in hits
            ) / sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_length + 1))
            sums[f"mrr@{cutoff}"] += (
                0.0 if hits.shape[0] == 0 else 1.0 / int(hits.min())
            )
    return {name: value / users if users else 0.0 for name, value in sums.items()}


def write_slice_comparison(path: Path, document: dict[str, object]) -> None:
    content = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"existing G6 RQ0 slice evidence differs: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)
