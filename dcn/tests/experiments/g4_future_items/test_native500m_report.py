from pathlib import Path

from experiments.g4_future_items.analysis.generate_native500m_report import (
    build_compact_report,
    build_tuning_report,
)


_ROLES = ("control_next_item", "rq1_24h", "rq2_next10")
_SLICES = (
    "target_distance_0_6h",
    "target_distance_6_24h",
    "target_distance_1_3d",
    "target_distance_3_7d",
    "target_event_rank_1",
    "target_event_rank_2_5",
    "target_event_rank_6_10",
    "target_event_rank_11_plus",
    "user_activity_q1",
    "user_activity_q2",
    "user_activity_q3",
    "user_activity_q4",
)


def _metrics(value: float) -> dict[str, float | int]:
    return {
        **{
            f"{metric}@{cutoff}": value
            for metric in ("recall", "capped_recall", "ndcg", "mrr", "coverage")
            for cutoff in (10, 50, 100)
        },
        "num_users": 4,
    }


def _evidence() -> dict:
    values = {"control_next_item": 0.10, "rq1_24h": 0.09, "rq2_next10": 0.11}
    candidates = {
        role: {
            "winner_row_id": f"{role}:01",
            "candidates": [
                {
                    "row_id": f"{role}:01",
                    "deep_learning_rate": 0.01,
                    "declared_horizon_epochs": 15,
                    "restored_best_epoch": 10,
                    "validation_recall_at_100": value,
                    "validation_loss": 6.0,
                }
            ],
        }
        for role, value in values.items()
    }
    return {
        "selection_provenance": candidates,
        "selected_runs": {
            role: {"declared_horizon_epochs": 15, "restored_best_epoch": 10}
            for role in _ROLES
        },
        "calibration": {
            "relative_dispersion": {
                f"{metric}@{cutoff}": 0.02
                for metric in ("recall", "capped_recall", "ndcg", "mrr", "coverage")
                for cutoff in (10, 50, 100)
            },
            "operational_bands_from_current_control": {
                "recall@100": 0.002,
                "ndcg@100": 0.002,
            },
        },
        "overall": {
            "rows": {role: _metrics(value) for role, value in values.items()},
            "qualification": {
                "rq1_24h": {
                    "recall_at_100_delta_points": -0.01,
                    "supported": False,
                },
                "rq2_next10": {
                    "recall_at_100_delta_points": 0.01,
                    "non_inferior": True,
                    "supported": True,
                },
            },
            "aggregate_role": "rq2_next10",
        },
        "slices": {
            name: {
                "rows": {
                    role: {"recall@100": value, "ndcg@100": value}
                    for role, value in values.items()
                }
            }
            for name in _SLICES
        },
    }


def test_native500m_compact_has_only_consecutive_rq_tables() -> None:
    report = build_compact_report(
        _evidence(),
        rq3_decision={
            "status": "preselector_stop",
            "audit_document": {
                "decision_basis": {
                    "reason": "The approved classifier requires a population-sized in-memory fit and has no external-memory fit path."
                }
            },
        },
    )

    assert report.count("## RQ1:") == 1
    assert report.count("## RQ2:") == 1
    assert report.count("## RQ3:") == 1
    assert (
        "| pre-selector feasibility | The approved classifier requires a "
        "population-sized in-memory fit and has no external-memory fit path. | "
        "**stopped** |"
    ) in report
    assert (
        report.count(
            "| variant | recall@100 | ndcg@100 | coverage@100 | horizon | restored epoch |"
        )
        == 2
    )
    assert (
        "| target distance | next liked item recall@100 | next liked item ndcg@100 "
        "| uniform among the next 10 liked events recall@100 | uniform among the "
        "next 10 liked events ndcg@100 |"
    ) in report
    assert "run_name" not in report
    assert "| deep lr |" not in report
    assert "is supported" not in report
    assert "not promoted" not in report
    assert "## Aggregated improvement" not in report
    assert all(line.startswith(("#", "|")) or not line for line in report.splitlines())


def test_native500m_tuning_report_exposes_only_deep_lr() -> None:
    report = build_tuning_report(_evidence())

    assert "| trial | deep lr | horizon |" in report
    assert "| embedding lr |" not in report.lower()
    assert "run_name" not in report


def test_reader_report_uses_consecutive_rq_headers_and_native500m_results() -> None:
    report = Path("experiments/g4_future_items/README.md").read_text()

    assert [line for line in report.splitlines() if line.startswith("## ")] == [
        "## RQ1: Does a 24-hour future window help?",
        "## RQ2: Does a next-10-liked-events window help?",
        "## RQ3: Can behavior-similar future periods define better positives?",
        "## Aggregated improvement",
    ]
    assert "native Yambda-500M" in report
    assert "native Yambda-50M" not in report
    assert "| Recall@100 | 0.153 | 0.153 | 0.000 |" in report
    assert "| Coverage@100 | 0.458 | 0.458 | 0.000 |" in report


def test_compact_rq_tables_match_reader_schema_and_order() -> None:
    reader = Path("experiments/g4_future_items/README.md").read_text()
    compact = Path(
        "experiments/g4_future_items/scratchpad/compact_native500m.md"
    ).read_text()
    compact_tables = _table_blocks(compact)

    assert _table_blocks(reader)[: len(compact_tables)] == compact_tables


def _table_blocks(report: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    for line in report.splitlines():
        if line.startswith("|"):
            if not blocks or blocks[-1][-1] == "":
                blocks.append([])
            blocks[-1].append(line)
        elif blocks and blocks[-1] and blocks[-1][-1] != "":
            blocks[-1].append("")
    return [[line for line in block if line] for block in blocks]
