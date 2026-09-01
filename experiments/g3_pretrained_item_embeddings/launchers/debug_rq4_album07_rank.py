from __future__ import annotations

import json
import math
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from experiments.g3_pretrained_item_embeddings.configs.model import (
    G3Representation,
    build_g3_experiment,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import PROJECT_ROOT
from experiments.g3_pretrained_item_embeddings.launchers.rq4_initial import (
    verify_rq4_initial_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_initial_ledger import (
    RQ3_FINAL_EVIDENCE_LOGICAL_SHA256,
    RQ3_FINAL_SELECTED_ROW_ID,
    RQ4_INITIAL_LEDGER_LOGICAL_SHA256,
    RQ4_INITIAL_LEDGER_PATH,
    load_rq4_initial_ledger,
)


RUN_NAME = "g3_rq4_album_capacity_07_rank_diagnostic_v3_native50m"
SOURCE_ROW_ID = "rq4_album:07"
TARGET_USER_ID = 543400
TARGET_ITEM_ID = 14455


ledger_path = PROJECT_ROOT / RQ4_INITIAL_LEDGER_PATH
ledger = load_rq4_initial_ledger(
    ledger_path,
    root=PROJECT_ROOT,
    expected_ledger_sha256=RQ4_INITIAL_LEDGER_LOGICAL_SHA256,
    expected_rq3_sha256=RQ3_FINAL_EVIDENCE_LOGICAL_SHA256,
    expected_rq3_row_id=RQ3_FINAL_SELECTED_ROW_ID,
)
feature_data_path = verify_rq4_initial_inputs(PROJECT_ROOT, ledger)
row = next(row for row in ledger.rows if row.id == SOURCE_ROW_ID)
experiment = build_g3_experiment(
    run_name=RUN_NAME,
    dataset_size="native-50m",
    embedding_learning_rate=row.embedding_learning_rate,
    deep_learning_rate=row.deep_learning_rate,
    lr_schedule_horizon_epochs=row.horizon_epochs,
    seed=42,
    representation=G3Representation(
        history_representation="id_content",
        history_hidden_dim=row.history_hidden_dim,
        catalog_representation=row.catalog_representation,
        metadata=row.metadata,
        metadata_dim=row.metadata_dim,
    ),
    feature_data_path=feature_data_path,
)


original_report = experiment._report_final_metrics


def _aggregate_metrics(prepared: Any, positions: torch.Tensor) -> dict[str, float]:
    catalog_positions = {
        item_id: position for position, item_id in enumerate(prepared.item_id_list)
    }
    totals = {
        f"{name}@{cutoff}": 0.0
        for name in ("recall", "ndcg", "mrr")
        for cutoff in (10, 50, 100)
    }
    for user_position, user in enumerate(prepared.evaluable):
        ranked = {
            int(item_position): rank
            for rank, item_position in enumerate(
                positions[user_position].tolist(), start=1
            )
        }
        ranks = [
            ranked.get(catalog_positions[item_id], 0) for item_id in user.relevant
        ]
        for cutoff in (10, 50, 100):
            hits = [rank for rank in ranks if 0 < rank <= cutoff]
            totals[f"recall@{cutoff}"] += len(hits) / len(ranks)
            ideal_length = min(cutoff, len(ranks))
            totals[f"ndcg@{cutoff}"] += sum(
                1.0 / math.log2(rank + 1) for rank in hits
            ) / sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, ideal_length + 1)
            )
            totals[f"mrr@{cutoff}"] += 1.0 / min(hits) if hits else 0.0
    return {
        name: value / len(prepared.evaluable) for name, value in totals.items()
    }


def _report_with_rank_scores(self: Any, runner: Any) -> None:
    epoch_prepared = self.true_metric._prepared.get(self.eval_max_users)
    if epoch_prepared is None:
        raise RuntimeError("rank diagnostic has no epoch ranking context")
    epoch_user_position = next(
        index
        for index, user in enumerate(epoch_prepared.evaluable)
        if user.user_id == TARGET_USER_ID
    )
    chunk_count = len(epoch_prepared.query_rows_by_chunk)
    target_chunk = epoch_user_position // epoch_prepared.user_chunk
    target_chunk_row = epoch_user_position % epoch_prepared.user_chunk
    catalog_ids = self.true_metric.item_batch["int_columns"][
        self.item_id_column
    ].dense()
    catalog_width = int(catalog_ids.shape[0])
    target_catalog_position = int((catalog_ids == TARGET_ITEM_ID).nonzero()[0, 0])
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    original_topk = torch.topk

    def recording_topk(input: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        result = original_topk(input, *args, **kwargs)
        k = args[0] if args else kwargs.get("k")
        if input.ndim == 2 and input.shape[1] == catalog_width and k == 100:
            chunk = len(calls) % chunk_count
            full_rank = 0
            if chunk == target_chunk:
                full_order = original_topk(
                    input[target_chunk_row], input.shape[1], dim=0
                ).indices
                full_rank = int(
                    (full_order == target_catalog_position).nonzero()[0, 0]
                ) + 1
            calls.append(
                (
                    result.values.detach().float().cpu(),
                    result.indices.detach().cpu(),
                    input[:, target_catalog_position].detach().float().cpu(),
                    full_rank,
                )
            )
        return result

    torch.topk = recording_topk
    try:
        original_report(runner)
    finally:
        torch.topk = original_topk

    prepared = self.true_metric._prepared[None]
    if [user.user_id for user in prepared.evaluable] != [
        user.user_id for user in epoch_prepared.evaluable
    ]:
        raise RuntimeError("final ranking population differs from the epoch context")
    if len(calls) != 2 * chunk_count:
        raise RuntimeError(
            f"rank diagnostic observed {len(calls)} of {2 * chunk_count} "
            "expected final ranking chunks"
        )
    user_position = next(
        index
        for index, user in enumerate(prepared.evaluable)
        if user.user_id == TARGET_USER_ID
    )
    if prepared.item_id_list[target_catalog_position] != TARGET_ITEM_ID:
        raise RuntimeError("diagnostic catalog positions changed")
    document: dict[str, object] = {
        "schema_version": 1,
        "kind": "g3_rq4_rank_reencoding_diagnostic",
        "source_row_id": SOURCE_ROW_ID,
        "run_name": RUN_NAME,
        "target_user_id": TARGET_USER_ID,
        "target_item_id": TARGET_ITEM_ID,
        "reported_metrics": json.loads(
            (
                Path(self.base_path) / "logs" / RUN_NAME / "final_metrics.json"
            ).read_text()
        ),
        "passes": [],
    }
    for name, pass_calls in zip(
        ("metric_and_ranking_evidence", "top_item_rankings_snapshot"),
        (calls[:chunk_count], calls[chunk_count:]),
        strict=True,
    ):
        scores = torch.cat([values for values, _, _, _ in pass_calls])
        positions = torch.cat([indices for _, indices, _, _ in pass_calls])
        target_scores = torch.cat([values for _, _, values, _ in pass_calls])
        full_ranks = [rank for _, _, _, rank in pass_calls if rank]
        if len(full_ranks) != 1:
            raise RuntimeError("rank diagnostic did not capture one exact full rank")
        target_positions = (
            positions[user_position] == target_catalog_position
        ).nonzero()
        if target_positions.shape[0] > 1:
            raise RuntimeError("target item is duplicated in the top 100")
        target_top100_rank = (
            int(target_positions[0, 0]) + 1 if target_positions.shape[0] else 0
        )
        target_rank = full_ranks[0]
        if target_top100_rank and target_top100_rank != target_rank:
            raise RuntimeError("top-100 and full-catalog target ranks differ")
        neighborhood = []
        center = target_top100_rank if target_top100_rank else 100
        for rank in range(max(1, center - 2), min(100, center + 2) + 1):
            item_position = int(positions[user_position, rank - 1])
            neighborhood.append(
                {
                    "rank": rank,
                    "item_id": int(prepared.item_id_list[item_position]),
                    "score": float(scores[user_position, rank - 1]),
                }
            )
        document["passes"].append(
            {
                "name": name,
                "target_rank": target_rank,
                "target_top100_rank": target_top100_rank,
                "target_score": float(target_scores[user_position]),
                "neighborhood": neighborhood,
                "aggregate_metrics": _aggregate_metrics(prepared, positions),
            }
        )
    first, second = document["passes"]
    document["snapshot_minus_evidence"] = {
        name: second["aggregate_metrics"][name] - first["aggregate_metrics"][name]
        for name in first["aggregate_metrics"]
    }
    destination = Path(self.base_path) / "logs" / RUN_NAME / "rank_diagnostic.json"
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


experiment._report_final_metrics = MethodType(_report_with_rank_scores, experiment)
