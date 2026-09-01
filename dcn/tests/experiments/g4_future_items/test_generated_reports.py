from experiments.g4_future_items.analysis.generate_reports import (
    build_compact_report,
    build_tuning_report,
    validate_gate_artifact,
    validate_selector_artifact,
)
from experiments.g4_future_items.configs.selectors import selector_trial_from_job
from experiments.g4_future_items.launchers.run_selectors import SelectorSearchResult
from experiments.g4_future_items.selectors import SelectorMetrics


def _candidate(
    role: str, trial: int, recall: float, *, embedding_rate: float
) -> dict[str, object]:
    return {
        "row_id": f"{role}:{trial:02d}",
        "run_name": f"{role}_{trial:02d}",
        "parameters": {
            "batch_size": 512,
            "embedding_learning_rate": embedding_rate,
            "deep_learning_rate": 0.01,
            "lr_schedule_horizon_epochs": 20,
        },
        "declared_horizon_epochs": 20,
        "restored_best_epoch": 12,
        "validation_recall_at_100": recall,
        "validation_loss": 5.0,
    }


def _evidence() -> dict[str, object]:
    control = _candidate("control", 1, 0.10, embedding_rate=0.02)
    control_other = _candidate("control", 2, 0.09, embedding_rate=0.03)
    rq1 = _candidate("rq1", 1, 0.11, embedding_rate=0.04)
    rq2 = _candidate("rq2", 1, 0.13, embedding_rate=0.05)
    rows = {
        "control_next_item": {"recall@100": 0.1, "ndcg@100": 0.04},
        "rq1_24h": {"recall@100": 0.11, "ndcg@100": 0.042},
        "rq2_next10": {"recall@100": 0.13, "ndcg@100": 0.05},
    }
    evidence = {
        "dataset_size": "native-50m",
        "selection_provenance": {
            "control_next_item": {
                "candidates": [control, control_other],
                "winner": control,
            },
            "rq1_24h": {"candidates": [rq1], "winner": rq1},
            "rq2_next10": {"candidates": [rq2], "winner": rq2},
        },
        "calibration": {
            "relative_dispersion": {"recall@100": 0.1, "ndcg@100": 0.1}
        },
        "overall": {"rows": rows},
        "slices": {
            name: {"rows": rows}
            for name in (
                "target_distance_0_6h",
                "target_distance_6_24h",
                "target_distance_1_3d",
                "target_distance_3_7d",
                "user_activity_q1",
                "user_activity_q2",
                "user_activity_q3",
                "user_activity_q4",
            )
        },
    }
    evidence["selected_runs"] = {
        "control_next_item": control,
        "rq1_24h": rq1,
        "rq2_next10": rq2,
    }
    return evidence


def _selector(family: str, trial_id: int, ndcg: float) -> dict[str, object]:
    return {
        "output_artifact_sha256": f"{trial_id:064x}",
        "trial": {
            "stage": "selector_search",
            "trial_id": trial_id,
            "boundary_round": None,
            "run_name": f"g4_selector_{family}_trial_{trial_id:02d}_native50m",
            "family": family,
            "period_width_seconds": 21600,
            "lookahead_seconds": 259200,
            "minimum_liked_events": 1,
            "time_tolerance_seconds": 3600 if family == "time" else None,
            "frequency_entity": "item" if family == "frequency" else None,
            "max_leaf_nodes": 15 if family == "learned" else None,
            "learning_rate": 0.05 if family == "learned" else None,
            "l2_regularization": 0.01 if family == "learned" else None,
            "seed": 42,
        },
        "validation_metrics": {
            "user_balanced_ndcg_at_10": ndcg,
            "auroc": 0.55,
            "query_count": 10,
            "user_count": 5,
            "pair_count": 20,
            "positive_count": 5,
            "negative_count": 15,
            "positive_rate": 0.25,
        },
    }


def _search_result(document: dict[str, object]) -> SelectorSearchResult:
    trial_document = document["trial"]
    job = {
        key: trial_document[key]
        for key in (
            "family",
            "trial_id",
            "boundary_round",
            "period_width_seconds",
            "lookahead_seconds",
            "minimum_liked_events",
            "time_tolerance_seconds",
            "frequency_entity",
            "max_leaf_nodes",
            "learning_rate",
            "l2_regularization",
            "seed",
        )
    }
    return SelectorSearchResult(
        trial=selector_trial_from_job(job),
        metrics=SelectorMetrics(**document["validation_metrics"]),
        relevance_threshold=0.5,
        artifact_sha256=document["output_artifact_sha256"],
        artifact_payload_sha256="a" * 64,
        artifact_path=None,
        prepared_sha256="b" * 64,
        prepared_semantics_sha256="c" * 64,
        wall_seconds=1.0,
    )


def test_generated_reports_include_every_trial_and_bold_each_method_winner() -> None:
    selectors = [
        _selector("time", 1, 0.2),
        _selector("time", 2, 0.3),
        _selector("content", 3, 0.31),
        _selector("frequency", 4, 0.29),
        _selector("learned", 5, 0.32),
    ]

    report = build_tuning_report(_evidence(), selectors)

    assert report.count("**01**") == 3
    assert "control_tuning" not in report
    assert "rq1_tuning" not in report
    assert "**B1-01**" not in report
    assert report.count("**trial 02**") == 1
    assert report.count("**trial 03**") == 1
    assert report.count("**trial 04**") == 1
    assert report.count("**trial 05**") == 1
    assert "selector_time_2" not in report
    assert "embedding lr" in report
    assert "validation ndcg@10" in report


def test_compact_report_has_only_rq_headings_and_three_decimal_result_tables() -> None:
    gate = {
        "deterministic": {
            "test_metrics": {
                "user_balanced_ndcg_at_10": 0.295811,
                "auroc": 0.551314,
            }
        },
        "learned": {
            "test_metrics": {
                "user_balanced_ndcg_at_10": 0.297088,
                "auroc": 0.557199,
            }
        },
    }

    report = build_compact_report(_evidence(), gate)

    assert "## RQ1: Does a 24-hour future window help?" in report
    assert "## RQ2: Does a next-10-liked-events window help?" in report
    assert "## RQ3: Can behavior-similar future periods define better positives?" in report
    assert "+30.0% (0.130)" in report
    assert "0.295811" not in report
    assert "0.296" in report
    assert "embedding_learning_rate" not in report
    assert "| variant | recall@100 | ndcg@100 | selected horizon | restored epoch |" in report
    assert "**next liked item**" in report
    assert "**uniform among next 10 liked events**" in report
    assert "**learned**" in report
    rq2_table = report.split(
        "## RQ2: Does a next-10-liked-events window help?", 1
    )[1]
    assert rq2_table.index("next 10 liked events") < rq2_table.index("next 24 hours")
    assert report.count("0–6 hours") == 2
    assert report.count("Q1, least active") == 2
    assert "0.1300" not in report


def test_selector_artifact_validation_binds_loaded_result_to_ledger_job() -> None:
    document = _selector("learned", 5, 0.32)
    result = _search_result(document)
    job = result.trial.to_dict() | {
        "output_artifact_sha256": result.artifact_sha256,
    }

    validate_selector_artifact(job, document, result)

    forged = dict(document)
    forged["validation_metrics"] = dict(document["validation_metrics"])
    forged["validation_metrics"]["user_balanced_ndcg_at_10"] = 0.99
    try:
        validate_selector_artifact(job, forged, result)
    except ValueError as error:
        assert "metrics" in str(error)
    else:
        raise AssertionError("forged selector metrics were accepted")


def test_gate_artifact_validation_binds_frozen_winners_and_payloads() -> None:
    deterministic = _search_result(_selector("content", 3, 0.31))
    learned = _search_result(_selector("learned", 5, 0.32))
    job = {
        "output_artifact_sha256": "d" * 64,
        "deterministic_artifact_sha256": deterministic.artifact_sha256,
        "deterministic_payload_sha256": deterministic.artifact_payload_sha256,
        "learned_artifact_sha256": learned.artifact_sha256,
        "learned_payload_sha256": learned.artifact_payload_sha256,
    }
    gate = {
        "output_artifact_sha256": "d" * 64,
        "passes": True,
        "deterministic": {
            "artifact_sha256": deterministic.artifact_sha256,
            "artifact_payload_sha256": deterministic.artifact_payload_sha256,
        },
        "learned": {
            "artifact_sha256": learned.artifact_sha256,
            "artifact_payload_sha256": learned.artifact_payload_sha256,
        },
    }

    validate_gate_artifact(job, gate, [deterministic, learned])

    forged = dict(gate)
    forged["learned"] = dict(gate["learned"])
    forged["learned"]["artifact_payload_sha256"] = "e" * 64
    try:
        validate_gate_artifact(job, forged, [deterministic, learned])
    except ValueError as error:
        assert "winner" in str(error)
    else:
        raise AssertionError("forged gate winner payload was accepted")
