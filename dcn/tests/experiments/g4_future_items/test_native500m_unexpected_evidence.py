import pytest

from experiments.g4_future_items.report.native500m_unexpected_evidence import (
    build_unexpected_result_diagnostics,
)


def test_unexpected_result_diagnostics_capture_breadth_and_activity_patterns() -> None:
    evaluation = {
        "kind": "g4_rq1_rq2_evaluation_native500m",
        "overall": {
            "rows": {
                "control_next_item": {"recall@100": 0.15},
                "rq1_24h": {"recall@100": 0.14},
                "rq2_next10": {"recall@100": 0.145},
            }
        },
        "slices": {
            f"user_activity_q{quartile}": {
                "rows": {
                    "control_next_item": {"recall@100": 0.1},
                    "rq1_24h": {"recall@100": 0.1 + rq1_delta},
                    "rq2_next10": {"recall@100": 0.1 + rq2_delta},
                }
            }
            for quartile, rq1_delta, rq2_delta in (
                (1, -0.008, -0.007),
                (2, -0.006, -0.004),
                (3, -0.004, 0.001),
                (4, -0.003, 0.002),
            )
        },
    }
    targets = {
        "kind": "g4_native500m_target_statistics",
        "objectives": {
            "control_next_item": {"candidate_occurrences": {"mean": 1.0}},
            "rq1_24h": {
                "candidate_occurrences": {"mean": 30.0},
                "eligibility_rate": 0.7,
                "fallback_rate": 0.3,
            },
            "rq2_next10": {"candidate_occurrences": {"mean": 9.5}},
        },
    }

    diagnostics = build_unexpected_result_diagnostics(evaluation, targets)

    assert diagnostics["status"] == "awaiting_user_validation"
    assert diagnostics["breadth_dose_response"]["candidate_means"] == {
        "control_next_item": 1.0,
        "rq2_next10": 9.5,
        "rq1_24h": 30.0,
    }
    assert diagnostics["breadth_dose_response"]["monotonic"] is True
    assert diagnostics["activity_moderation"]["monotonic"] == {
        "rq1_24h": True,
        "rq2_next10": True,
    }


def test_unexpected_result_conclusion_requires_exact_user_validation() -> None:
    validation = {
        "schema_version": 1,
        "kind": "g4_native500m_rq1_rq2_conclusion_validation",
        "dataset_size": "native-500m",
        "status": "validated",
        "scope": "RQ1/RQ2 inferior; control retained",
        "message_sha256": "da0fd6b46d4bb41f96bc9c01a4e975c3260eae4130f158d12e68f11ebe892f6a",
    }
    evaluation = {
        "kind": "g4_rq1_rq2_evaluation_native500m",
        "overall": {
            "rows": {
                "control_next_item": {"recall@100": 0.15},
                "rq1_24h": {"recall@100": 0.14},
                "rq2_next10": {"recall@100": 0.145},
            }
        },
        "slices": {
            f"user_activity_q{quartile}": {
                "rows": {
                    "control_next_item": {"recall@100": 0.1},
                    "rq1_24h": {"recall@100": 0.09 + quartile / 1000},
                    "rq2_next10": {"recall@100": 0.092 + quartile / 1000},
                }
            }
            for quartile in range(1, 5)
        },
    }
    targets = {
        "kind": "g4_native500m_target_statistics",
        "objectives": {
            "control_next_item": {"candidate_occurrences": {"mean": 1.0}},
            "rq1_24h": {
                "candidate_occurrences": {"mean": 30.0},
                "eligibility_rate": 0.7,
                "fallback_rate": 0.3,
            },
            "rq2_next10": {"candidate_occurrences": {"mean": 9.5}},
        },
    }

    diagnostics = build_unexpected_result_diagnostics(
        evaluation, targets, user_validation=validation
    )

    assert diagnostics["status"] == "conclusion_validated"
    assert diagnostics["mechanism_status"] == "tentative"
    assert diagnostics["user_validation"] == validation

    validation["message_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="validation"):
        build_unexpected_result_diagnostics(
            evaluation, targets, user_validation=validation
        )
