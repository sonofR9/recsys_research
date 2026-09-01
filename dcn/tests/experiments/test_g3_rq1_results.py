from pathlib import Path
import hashlib
import json
import math

import pytest
import polars as pl
import torch

from dcn.eval.ranking_evidence import RankingEvidence, write_ranking_evidence
from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    _recompute_metrics,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq1_results import (
    APPROVED_RQ1_EVIDENCE_SHA256,
    assess_rq1_boundaries,
    build_rq1_report_documents,
    load_rq1_evidence,
    load_training_item_counts,
    _ranking_slices,
    select_rq1_winner,
    verify_rq1_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq1 import PROJECT_ROOT


def _run(
    row_id: str,
    recall: float,
    ndcg: float,
    seconds: float,
    *,
    embedding_learning_rate: float = 0.2,
    deep_learning_rate: float = 0.03,
    horizon_epochs: int = 25,
    best_epoch: int = 20,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "embedding_learning_rate": embedding_learning_rate,
        "deep_learning_rate": deep_learning_rate,
        "horizon_epochs": horizon_epochs,
        "best_epoch": best_epoch,
        "queue_wall_seconds": seconds,
        "efficiency": {"logged_training_seconds": seconds},
        "metrics": {"recall@100": recall, "ndcg@100": ndcg},
    }


def test_rq1_selection_uses_recall_ndcg_time_then_manifest_order() -> None:
    runs = (
        _run("rq1:01", 0.1, 0.04, 12.0),
        _run("rq1:02", 0.1, 0.05, 13.0),
        _run("rq1:03", 0.1, 0.05, 11.0),
        _run("rq1:04", 0.1, 0.05, 11.0),
    )

    assert select_rq1_winner(runs)["row_id"] == "rq1:03"


def test_rq1_boundary_decision_requires_only_the_approved_extensions() -> None:
    interior = assess_rq1_boundaries(
        _run(
            "rq1:02",
            0.1,
            0.05,
            11.0,
            embedding_learning_rate=0.2183583071089141,
            deep_learning_rate=0.021004505318001004,
            horizon_epochs=40,
            best_epoch=22,
        )
    )
    assert interior["extension_required"] is False
    assert interior["embedding_learning_rate"]["direction"] is None
    assert interior["deep_learning_rate"]["direction"] is None
    assert interior["horizon"]["extend_to_epochs"] is None

    boundary = assess_rq1_boundaries(
        _run(
            "rq1:boundary",
            0.1,
            0.05,
            11.0,
            embedding_learning_rate=0.04,
            deep_learning_rate=0.129,
            horizon_epochs=40,
            best_epoch=40,
        )
    )
    assert boundary["extension_required"] is True
    assert boundary["embedding_learning_rate"]["direction"] == "lower"
    assert boundary["deep_learning_rate"]["direction"] == "upper"
    assert boundary["horizon"]["extend_to_epochs"] == 60


def test_rq1_frequency_catalog_excludes_only_unknown_compact_id_zero(
    tmp_path: Path,
) -> None:
    path = tmp_path / "features.parquet"
    pl.DataFrame(
        {"compact_item_id": [0, 1, 2, 3], "training_count": [0, 4, 2, 9]}
    ).write_parquet(path)

    assert load_training_item_counts(path) == {1: 4, 2: 2, 3: 9}


def test_ranking_evidence_can_authoritatively_source_metrics_and_item_slices(
    tmp_path: Path,
) -> None:
    catalog = list(range(1, 102))
    rankings = {
        10: [*range(1, 42), 43, 42, *range(44, 101)],
        20: catalog[:100],
        30: catalog[:100],
    }
    evidence = RankingEvidence(
        user_ids=torch.tensor([10, 20, 30]),
        history_item_ids=torch.tensor([1, 1, 1]),
        history_offsets=torch.tensor([0, 1, 2, 3]),
        relevant_item_ids=torch.tensor([42, 1, 2]),
        relevance_offsets=torch.tensor([0, 1, 2, 3]),
        relevant_train_frequencies=torch.tensor([42, 1, 2]),
        relevant_ranks=torch.tensor([42, 1, 2]),
        max_k=100,
    )
    context_path = tmp_path / "context.pt"
    ranking_path = tmp_path / "ranking.pt"
    snapshot_path = tmp_path / "snapshot.json"
    write_ranking_evidence(
        evidence, context_path=context_path, ranking_path=ranking_path
    )
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "catalog_sha256": hashlib.sha256(
                    json.dumps(catalog, separators=(",", ":")).encode()
                ).hexdigest(),
                "catalog_size": len(catalog),
                "exclude_seen": False,
                "max_k": 100,
                "rankings": [
                    {"user_id": user_id, "item_ids": item_ids}
                    for user_id, item_ids in rankings.items()
                ],
            }
        )
    )

    metrics = _recompute_metrics(context_path, ranking_path, snapshot_path)
    slices = _ranking_slices(
        context_path=context_path,
        ranking_path=ranking_path,
        rankings_path=snapshot_path,
        item_counts={item_id: item_id for item_id in catalog},
        rank_source="ranking_evidence",
    )

    assert metrics["ndcg@50"] == pytest.approx(
        (1 / math.log2(43) + 1 + 1 / math.log2(3)) / 3
    )
    assert metrics["mrr@50"] == pytest.approx((1 / 42 + 1 + 1 / 2) / 3)
    assert metrics["recall@50"] == 1.0
    assert slices["mid"]["recall@50"] == 1.0
    assert metrics["coverage@100"] == len(
        set().union(*(set(items) for items in rankings.values()))
    ) / len(catalog)
    with pytest.raises(ValueError, match="ranks differ"):
        _ranking_slices(
            context_path=context_path,
            ranking_path=ranking_path,
            rankings_path=snapshot_path,
            item_counts={item_id: item_id for item_id in catalog},
        )


def test_rq1_reports_separate_tuning_overall_slices_and_efficiency() -> None:
    evidence = {
        "tuning_ledger": [
            _run("rq1:01", 0.09, 0.04, 12.0)
            | {"run_name": "machine-run-1", "epochs_trained": 25},
            _run("rq1:02", 0.10, 0.05, 11.0)
            | {"run_name": "machine-run-2", "epochs_trained": 25},
        ],
        "selected_treatment": {"row_id": "rq1:02"},
        "comparison": {
            "tied_original": {
                "metrics": {"recall@100": 0.11, "ndcg@100": 0.045},
                "slices": {
                    name: {
                        "num_users": 10,
                        "num_targets": 20,
                        "recall@100": value,
                        "item_membership_sha256": name,
                    }
                    for name, value in (("head", 0.15), ("mid", 0.03), ("tail", 0.006))
                },
                "efficiency": {
                    "best_epoch": 20,
                    "declared_horizon_epochs": 25,
                    "queue_wall_seconds": 80.0,
                    "targets_per_second": 1100.0,
                    "peak_gpu_memory_gb": 4.0,
                    "parameter_count": 120,
                    "full_catalog_observed_upper_bound_seconds": 1.0,
                    "full_catalog_encoding_scoring_seconds": None,
                    "full_catalog_timing_limitation": "not separately instrumented",
                },
            },
            "control": {
                "metrics": {"recall@100": 0.08, "ndcg@100": 0.03},
                "slices": {
                    name: {
                        "num_users": 10,
                        "num_targets": 20,
                        "recall@100": value,
                        "item_membership_sha256": name,
                    }
                    for name, value in (("head", 0.12), ("mid", 0.02), ("tail", 0.004))
                },
                "efficiency": {
                    "best_epoch": 20,
                    "declared_horizon_epochs": 25,
                    "queue_wall_seconds": 100.0,
                    "targets_per_second": 1000.0,
                    "peak_gpu_memory_gb": 4.0,
                    "parameter_count": 100,
                    "full_catalog_observed_upper_bound_seconds": 1.0,
                    "full_catalog_encoding_scoring_seconds": None,
                    "full_catalog_timing_limitation": "not separately instrumented",
                },
            },
            "treatment": {
                "metrics": {"recall@100": 0.10, "ndcg@100": 0.04},
                "slices": {
                    name: {
                        "num_users": 10,
                        "num_targets": 20,
                        "recall@100": value,
                        "item_membership_sha256": name,
                    }
                    for name, value in (("head", 0.14), ("mid", 0.04), ("tail", 0.012))
                },
                "efficiency": {
                    "best_epoch": 18,
                    "declared_horizon_epochs": 25,
                    "queue_wall_seconds": 90.0,
                    "targets_per_second": 900.0,
                    "peak_gpu_memory_gb": 4.5,
                    "parameter_count": 110,
                    "full_catalog_observed_upper_bound_seconds": 1.2,
                    "full_catalog_encoding_scoring_seconds": None,
                    "full_catalog_timing_limitation": "not separately instrumented",
                },
            },
        },
        "boundary_decision": {
            "embedding_learning_rate": {
                "selected": 0.2183583071089141,
                "bounds": [0.0368614745, 0.5897835914],
                "normalized_position": 0.32825026719222217,
                "direction": None,
            },
            "deep_learning_rate": {
                "selected": 0.021004505318001004,
                "bounds": [0.0081084848, 0.1297357573],
                "normalized_position": 0.10602902007854367,
                "direction": None,
            },
            "horizon": {
                "selected_epochs": 40,
                "restored_best_epoch": 22,
                "extend_to_epochs": None,
            },
            "extension_required": False,
        },
        "promotion_decision": {
            "promoted": True,
            "route": "tail_tradeoff",
            "comparators": {
                "aggregate_improvement": ["tied_original"],
                "aggregate_tail_tradeoff_band": ["tied_original"],
                "tail_improvement": ["untied_control"],
            },
            "reason": (
                "aggregate Recall@100 is within band and tail Recall@100 is higher"
            ),
        },
    }

    tuning, reader = build_rq1_report_documents(evidence)

    assert "**rq1:02**" in tuning
    assert "machine-run" not in tuning
    assert "## RQ1" in reader
    assert "### Overall" in reader
    assert "### Item-frequency slices" in reader
    assert "### Efficiency" in reader
    assert "tail" in reader
    assert "full-catalog evaluation upper bound" in reader
    assert "vs tied original" in reader
    assert "tied difference unresolved within tied band" in reader
    assert "### Frozen boundary and promotion decision" in reader
    assert "0.2183583071089141" in reader
    assert "tail_tradeoff" in reader
    assert "tied_original" in reader
    assert "untied_control" in reader
    assert "descriptive only; no slice-specific repeat calibration" in reader
    assert "tail-route evidence status" in reader
    assert "advisory-lock wait" in reader
    assert "checkpoint-only timing replay" in reader


def test_materialized_rq1_evidence_has_the_approved_identity() -> None:
    path = (
        PROJECT_ROOT
        / "experiments/g3_pretrained_item_embeddings/evidence/"
        "rq1_content_input.json"
    )
    evidence = verify_rq1_evidence(path, root=PROJECT_ROOT)

    assert evidence["sha256"] == APPROVED_RQ1_EVIDENCE_SHA256
    assert evidence["queue_batch"]["batch_id"] == (
        "05d24119ed134265897eaeabfd8b19a6"
    )
    assert len(evidence["tuning_ledger"]) == 9
    assert evidence["boundary_decision"]["extension_required"] is False
    assert evidence["promotion_decision"]["promoted"] is True
    assert evidence["promotion_decision"]["route"] == "tail_tradeoff"
    assert set(evidence["promotion_decision"]["absolute_bands"]) == {
        "tied_original",
        "untied_control",
    }
    assert evidence["promotion_decision"]["comparators"] == {
        "aggregate_improvement": ["tied_original"],
        "aggregate_tail_tradeoff_band": ["tied_original"],
        "tail_improvement": ["untied_control"],
    }
    comparisons = evidence["comparison"]["direct_comparisons"]
    assert comparisons["untied_control"]["recall_at_100_delta_percent"] == pytest.approx(
        23.752361275238165
    )
    assert comparisons["untied_control"]["recall_at_100_within_band"] is False
    assert comparisons["tied_original"]["recall_at_100_delta_percent"] == pytest.approx(
        -7.051461770940703
    )
    assert comparisons["tied_original"]["recall_at_100_within_band"] is True


def test_rq1_evidence_loader_rejects_hash_drift(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text('{"schema_version":1,"kind":"g3_rq1_content_input_evidence"}')

    with pytest.raises(ValueError, match="evidence"):
        load_rq1_evidence(path)
