from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from experiments.g3_pretrained_item_embeddings.analysis.rq3_post_boundary_results import (
    RQ3_POST_BOUNDARY_BATCH_ID,
    assess_rq3_family_boundaries,
    build_rq3_matched_coordinate_contrasts,
    build_rq3_paired_contrasts,
    build_rq3_post_boundary_evidence,
    resolve_rq3_downstream_selection,
    select_rq3_family_winners,
)
from experiments.g3_pretrained_item_embeddings.launchers.rq3_post_boundary import (
    RQ3_POST_BOUNDARY_LEDGER_PATH,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    RQ3_OUTPUT_FAMILY_IDS,
)


def _run(
    family_id: str,
    row_id: str,
    *,
    recall: float,
    ndcg: float,
    wall: float,
    embedding_learning_rate: float = 0.1,
    deep_learning_rate: float = 0.03,
    horizon_epochs: int = 25,
    best_epoch: int = 20,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "family_id": family_id,
        "embedding_learning_rate": embedding_learning_rate,
        "deep_learning_rate": deep_learning_rate,
        "horizon_epochs": horizon_epochs,
        "best_epoch": best_epoch,
        "queue_wall_seconds": wall,
        "metrics": {
            "recall@100": recall,
            "ndcg@100": ndcg,
        },
        "slices": {
            name: {
                "recall@100": recall + offset,
                "num_users": 10,
                "num_targets": 20,
                "item_membership_sha256": name * 8,
            }
            for name, offset in (("head", 0.01), ("mid", 0.0), ("tail", -0.01))
        },
        "efficiency": {
            "queue_wall_seconds": wall,
            "logged_training_seconds": wall / 2,
            "examples_per_second": 100.0 / wall,
            "targets_per_second": 100.0 / wall,
            "peak_gpu_memory_gb": 4.0,
            "parameter_count": 1000,
        },
    }


def test_family_selection_uses_recall_ndcg_wall_time_then_row_id() -> None:
    family = RQ3_OUTPUT_FAMILY_IDS[0]
    runs = [
        _run(family, "z", recall=0.2, ndcg=0.1, wall=3.0),
        _run(family, "y", recall=0.2, ndcg=0.2, wall=4.0),
        _run(family, "x", recall=0.2, ndcg=0.2, wall=2.0),
        _run(family, "w", recall=0.2, ndcg=0.2, wall=2.0),
        *[
            _run(family, f"low-{index}", recall=0.1, ndcg=0.1, wall=1.0)
            for index in range(5)
        ],
    ]

    selected = select_rq3_family_winners(runs, family_ids=(family,))

    assert selected[family]["row_id"] == "w"


def test_family_selection_requires_exactly_nine_unique_opportunities() -> None:
    family = RQ3_OUTPUT_FAMILY_IDS[0]
    runs = [
        _run(family, f"{family}:{index:02d}", recall=index / 100, ndcg=0.1, wall=1.0)
        for index in range(1, 9)
    ]

    with pytest.raises(ValueError, match="nine"):
        select_rq3_family_winners(runs, family_ids=(family,))


def test_boundary_assessment_reports_lr_and_horizon_extensions() -> None:
    family = RQ3_OUTPUT_FAMILY_IDS[0]
    winner = _run(
        family,
        f"{family}:09",
        recall=0.2,
        ndcg=0.1,
        wall=2.0,
        embedding_learning_rate=0.04,
        deep_learning_rate=0.12,
        horizon_epochs=40,
        best_epoch=40,
    )
    selected = {
        family: winner
    }
    surface = [winner] + [
        _run(
            family,
            f"{family}:{index:02d}",
            recall=0.1,
            ndcg=0.1,
            wall=3.0,
            embedding_learning_rate=0.04 + index * 0.01,
            deep_learning_rate=0.01 + index * 0.01,
        )
        for index in range(1, 9)
    ]

    boundary = assess_rq3_family_boundaries(selected, surface)

    assert boundary[family]["embedding_learning_rate"]["direction"] == "lower"
    assert boundary[family]["deep_learning_rate"]["direction"] == "upper"
    assert boundary[family]["horizon"]["extend_to_epochs"] == 60
    assert boundary[family]["extension_required"] is True


def test_boundary_assessment_flags_trial04_in_actual_surface_outer_ten_percent() -> None:
    family = RQ3_OUTPUT_FAMILY_IDS[3]
    deep_rates = [
        0.12582682908321982,
        0.032433939334700325,
        0.03993056713468058,
        0.005733564587228046,
        0.014506684820055783,
        0.0040542424,
        0.002866782293614023,
        0.015941768409277256,
        0.03772073950128722,
    ]
    surface = [
        _run(
            family,
            f"{family}:{index:02d}",
            recall=0.2 if index == 4 else 0.1,
            ndcg=0.1,
            wall=2.0,
            embedding_learning_rate=0.1 + index * 0.01,
            deep_learning_rate=rate,
            horizon_epochs=40 if index == 4 else 25,
            best_epoch=18 if index == 4 else 20,
        )
        for index, rate in enumerate(deep_rates, start=1)
    ]

    boundary = assess_rq3_family_boundaries({family: surface[3]}, surface)

    assert boundary[family]["deep_learning_rate"]["direction"] == "lower"
    assert boundary[family]["deep_learning_rate"]["has_tested_lower"] is True
    assert boundary[family]["deep_learning_rate"]["has_tested_higher"] is True
    assert boundary[family]["horizon"]["extend_to_epochs"] is None
    assert boundary[family]["extension_required"] is True


def test_paired_contrasts_cover_target_type_initialization_and_freezing() -> None:
    winners = {
        family_id: _run(
            family_id,
            f"{family_id}:01",
            recall=0.1 + index * 0.01,
            ndcg=0.05 + index * 0.005,
            wall=10.0 + index,
        )
        for index, family_id in enumerate(RQ3_OUTPUT_FAMILY_IDS)
    }

    contrasts = build_rq3_paired_contrasts(winners)

    assert set(contrasts) == {
        "target_type_frozen_content_vs_learned_id",
        "pretrained_initialization_trainable_content_vs_learned_id",
        "freezing_content_target",
        "learned_id_augmentation_of_frozen_content",
        "freezing_concatenated_target",
        "expected_variant_4_vs_learned_id",
    }
    assert contrasts["freezing_content_target"]["axis"] == "freezing"
    assert contrasts["freezing_content_target"]["overall_metric_deltas"][
        "recall@100"
    ] == pytest.approx(0.01)
    assert contrasts[
        "pretrained_initialization_trainable_content_vs_learned_id"
    ]["isolated_axis"] is False


def test_matched_contrasts_require_and_summarize_nine_shared_coordinates() -> None:
    runs = []
    for family_index, family_id in enumerate(RQ3_OUTPUT_FAMILY_IDS):
        for coordinate_index in range(9):
            runs.append(
                _run(
                    family_id,
                    f"{family_id}:{coordinate_index + 1:02d}",
                    recall=0.1 + family_index * 0.01 + coordinate_index * 0.001,
                    ndcg=0.05 + family_index * 0.005,
                    wall=10.0,
                    embedding_learning_rate=0.04 + coordinate_index * 0.01,
                    deep_learning_rate=0.01 + coordinate_index * 0.001,
                    horizon_epochs=15 + coordinate_index,
                )
            )

    contrasts = build_rq3_matched_coordinate_contrasts(runs)

    expected = contrasts["expected_variant_4_vs_learned_id"]
    assert len(expected["pairs"]) == 9
    assert expected["summary"]["pair_count"] == 9
    assert expected["summary"]["treatment_recall@100_win_count"] == 9
    assert expected["summary"]["mean_recall@100_delta"] == pytest.approx(0.03)


def test_downstream_selection_requires_band_and_resolved_boundaries() -> None:
    winners = {
        family_id: _run(
            family_id,
            f"{family_id}:01",
            recall=0.1 + (0.02 if index == 3 else 0.0),
            ndcg=0.05,
            wall=10.0,
        )
        for index, family_id in enumerate(RQ3_OUTPUT_FAMILY_IDS)
    }
    boundaries = {
        family_id: {"extension_required": family_id == RQ3_OUTPUT_FAMILY_IDS[-1]}
        for family_id in RQ3_OUTPUT_FAMILY_IDS
    }

    decision = resolve_rq3_downstream_selection(winners, boundaries)

    assert decision["rq4_scientific_selected"]["family_id"] == (
        "rq3_output_learned_frozen_content"
    )
    assert decision["aggregate_selected"]["family_id"] == (
        "rq3_output_learned_frozen_content"
    )
    assert decision["treatment_promoted"] is True
    assert decision["status"] == "boundary_extensions_required"
    assert decision["unresolved_boundary_families"] == [
        "rq3_output_learned_trainable_content"
    ]


def test_production_builder_rejects_incomplete_batch_before_collecting_results(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    ledger_source = project_root / RQ3_POST_BOUNDARY_LEDGER_PATH
    batch_source = (
        project_root
        / "generated/training-queue-service/batches"
        / f"{RQ3_POST_BOUNDARY_BATCH_ID}.json"
    )
    ledger_path = tmp_path / RQ3_POST_BOUNDARY_LEDGER_PATH
    batch_path = (
        tmp_path
        / "generated/training-queue-service/batches"
        / f"{RQ3_POST_BOUNDARY_BATCH_ID}.json"
    )
    ledger_path.parent.mkdir(parents=True)
    batch_path.parent.mkdir(parents=True)
    shutil.copyfile(ledger_source, ledger_path)
    shutil.copyfile(batch_source, batch_path)

    with pytest.raises(RuntimeError, match="not complete"):
        build_rq3_post_boundary_evidence(tmp_path)
