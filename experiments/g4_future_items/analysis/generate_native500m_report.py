from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.g4_future_items.report.native500m_evidence import (
    write_native500m_evidence,
)
from experiments.g4_future_items.report.native500m_rq3_decision import (
    load_rq3_decision,
)
from experiments.g4_future_items.report.native500m_target_statistics import (
    write_native500m_target_statistics,
)
from experiments.g4_future_items.report.native500m_unexpected_evidence import (
    write_unexpected_result_diagnostics,
)


_ROLE_LABELS = {
    "control_next_item": "next liked item",
    "rq1_24h": "uniform liked event in the next 24 hours",
    "rq2_next10": "uniform among the next 10 liked events",
}
_DISTANCE_SLICES = (
    ("target_distance_0_6h", "0–6 hours"),
    ("target_distance_6_24h", "6–24 hours"),
    ("target_distance_1_3d", "1–3 days"),
    ("target_distance_3_7d", "3–7 days"),
)
_EVENT_SLICES = (
    ("target_event_rank_1", "first final-window event"),
    ("target_event_rank_2_5", "events 2–5"),
    ("target_event_rank_6_10", "events 6–10"),
    ("target_event_rank_11_plus", "events 11+"),
)
_ACTIVITY_SLICES = (
    ("user_activity_q1", "Q1, least active"),
    ("user_activity_q2", "Q2"),
    ("user_activity_q3", "Q3"),
    ("user_activity_q4", "Q4, most active"),
)


def build_tuning_report(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# G4 native-500M hyperparameter tuning",
        "",
        "Batch size 512, embedding LR 0.0468526465053628, and the 15-epoch "
        "one-cycle cosine horizon are fixed. Deep LR is the only tuned field.",
        "",
    ]
    sections = (
        ("Control: next liked item", "control_next_item"),
        ("RQ1: 24-hour future window", "rq1_24h"),
        ("RQ2: next 10 liked events", "rq2_next10"),
    )
    for title, role in sections:
        selection = evidence["selection_provenance"][role]
        lines.extend(
            (
                f"## {title}",
                "",
                "| trial | deep lr | horizon | best epoch | validation recall@100 | validation loss |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for candidate in selection["candidates"]:
            selected = candidate["row_id"] == selection["winner_row_id"]
            values = (
                _trial_label(candidate["row_id"]),
                f"{float(candidate['deep_learning_rate']):.8g}",
                str(candidate["declared_horizon_epochs"]),
                str(candidate["restored_best_epoch"]),
                f"{float(candidate['validation_recall_at_100']):.4f}",
                f"{float(candidate['validation_loss']):.4f}",
            )
            lines.append(_row(values, selected=selected))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_compact_report(
    evidence: Mapping[str, Any], *, rq3_decision: Mapping[str, Any]
) -> str:
    lines = [
        "# G4 native-500M compact results",
        "",
        "## RQ1: Does a 24-hour future window help?",
        "",
    ]
    lines.extend(_overall_table(evidence, ("control_next_item", "rq1_24h")))
    lines.append("")
    lines.extend(
        _slice_table(
            evidence,
            _DISTANCE_SLICES,
            ("control_next_item", "rq1_24h"),
            "target distance",
        )
    )
    lines.append("")
    lines.extend(
        _slice_table(
            evidence,
            _EVENT_SLICES,
            ("control_next_item", "rq1_24h"),
            "final-window event rank",
        )
    )
    lines.append("")
    lines.extend(
        _slice_table(
            evidence,
            _ACTIVITY_SLICES,
            ("control_next_item", "rq1_24h"),
            "user activity",
        )
    )
    lines.extend(
        (
            "",
            "## RQ2: Does a next-10-liked-events window help?",
            "",
        )
    )
    lines.extend(
        _overall_table(evidence, ("control_next_item", "rq2_next10", "rq1_24h"))
    )
    lines.append("")
    roles = ("control_next_item", "rq2_next10")
    lines.extend(_slice_table(evidence, _DISTANCE_SLICES, roles, "target distance"))
    lines.append("")
    lines.extend(
        _slice_table(evidence, _EVENT_SLICES, roles, "final-window event rank")
    )
    lines.append("")
    lines.extend(_slice_table(evidence, _ACTIVITY_SLICES, roles, "user activity"))
    lines.extend(
        (
            "",
            "## RQ3: Can behavior-similar future periods define better positives?",
            "",
        )
    )
    status = rq3_decision.get("status")
    audit = rq3_decision.get("audit_document")
    if status == "preselector_stop":
        if not isinstance(audit, Mapping):
            raise ValueError("RQ3 compact decision lacks its authenticated audit")
        detail = audit["decision_basis"]["reason"]
        lines.extend(
            (
                "| stage | authenticated static audit | outcome |",
                "| :--- | :--- | :---: |",
                f"| pre-selector feasibility | {detail} | **stopped** |",
            )
        )
    elif status == "pending_validation":
        lines.extend(
            (
                "| stage | outcome |",
                "| :--- | :---: |",
                "| pre-selector feasibility | awaiting validation |",
            )
        )
    else:
        raise ValueError("unsupported RQ3 compact decision")
    return "\n".join(lines).rstrip() + "\n"


def _overall_table(evidence: Mapping[str, Any], roles: Sequence[str]) -> list[str]:
    rows = evidence["overall"]["rows"]
    selected = evidence["overall"]["aggregate_role"]
    lines = [
        "| variant | recall@100 | ndcg@100 | coverage@100 | horizon | restored epoch |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for role in roles:
        run = evidence["selected_runs"][role]
        values = (
            _ROLE_LABELS[role],
            _metric_cell(evidence, role, "recall@100"),
            _metric_cell(evidence, role, "ndcg@100"),
            _metric_cell(evidence, role, "coverage@100"),
            str(run["declared_horizon_epochs"]),
            str(run["restored_best_epoch"]),
        )
        lines.append(_row(values, selected=role == selected))
    return lines


def _slice_table(
    evidence: Mapping[str, Any],
    slices: Sequence[tuple[str, str]],
    roles: Sequence[str],
    first_header: str,
) -> list[str]:
    headers = [first_header]
    for role in roles:
        headers.extend(
            (f"{_ROLE_LABELS[role]} recall@100", f"{_ROLE_LABELS[role]} ndcg@100")
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join(":---:" for _ in headers[1:]) + " |",
    ]
    relative = evidence["calibration"]["relative_dispersion"]
    for slice_id, label in slices:
        slice_rows = evidence["slices"][slice_id]["rows"]
        values = [label]
        for role in roles:
            for metric in ("recall@100", "ndcg@100"):
                value = float(slice_rows[role][metric])
                reference = float(slice_rows["control_next_item"][metric])
                values.append(
                    f"{value:.3f}"
                    if role == "control_next_item"
                    else _change_cell(reference, value, float(relative[metric]))
                )
        lines.append(_row(values))
    return lines


def _metric_cell(evidence: Mapping[str, Any], role: str, metric: str) -> str:
    rows = evidence["overall"]["rows"]
    value = float(rows[role][metric])
    if role == "control_next_item":
        return f"{value:.3f}"
    return _change_cell(
        float(rows["control_next_item"][metric]),
        value,
        float(evidence["calibration"]["relative_dispersion"][metric]),
    )


def _change_cell(reference: float, value: float, relative_dispersion: float) -> str:
    percent = 100 * (value / reference - 1)
    rendered = f"{percent:+.1f}% ({value:.3f})".replace("-", "−")
    threshold = reference * relative_dispersion
    if value - reference > threshold:
        return f'<span style="color: green">{rendered}</span>'
    if reference - value > threshold:
        return f'<span style="color: red">{rendered}</span>'
    return rendered


def _trial_label(row_id: str) -> str:
    parts = row_id.split(":")
    if "boundary" in parts[0]:
        return f"{parts[1]}-r{parts[2][1:]}-{parts[3]}"
    return f"base-{parts[-1]}"


def _row(values: Sequence[str], *, selected: bool = False) -> str:
    rendered = [f"**{value}**" if selected else value for value in values]
    return "| " + " | ".join(rendered) + " |"


def _default_ledgers(root: Path) -> dict[str, list[Path]]:
    ledger_root = root / "experiments/g4_future_items/protocol/native500m/ledgers"
    result = {
        "control_next_item": [
            ledger_root / "control_tuning_retry2.json",
            ledger_root / "control_tuning_boundary_upper_r1.json",
        ],
        "rq1_24h": [
            ledger_root / "rq1_tuning.json",
            ledger_root / "rq1_tuning_boundary_lower_r1.json",
        ],
        "rq2_next10": [
            ledger_root / "rq2_tuning.json",
            ledger_root / "rq2_tuning_boundary_lower_r1.json",
        ],
    }
    for role, prefix in (("rq1_24h", "rq1"), ("rq2_next10", "rq2")):
        round_two = ledger_root / f"{prefix}_tuning_boundary_lower_r2.json"
        if round_two.exists():
            result[role].append(round_two)
    return result


def generate(
    repo_root: Path, *, rq3_decision_path: Path
) -> tuple[Path, Path, Path, Path, Path]:
    root = repo_root.resolve(strict=True)
    rq3_decision = load_rq3_decision(root, rq3_decision_path)
    ledgers = _default_ledgers(root)
    evidence_path = (
        root / "experiments/g4_future_items/evidence/rq1_rq2_evaluation_native500m.json"
    )
    write_native500m_evidence(evidence_path, repo_root=root, role_ledgers=ledgers)
    evidence = json.loads(evidence_path.read_text())
    target_statistics_path = (
        root / "experiments/g4_future_items/evidence/target_statistics_native500m.json"
    )
    control_contract = Path(
        evidence["selected_runs"]["control_next_item"]["artifacts"]["g4_job.json"][
            "path"
        ]
    )
    write_native500m_target_statistics(
        target_statistics_path,
        repo_root=root,
        control_contract_path=control_contract,
    )
    unexpected_path = (
        root
        / "experiments/g4_future_items/evidence/unexpected_results_native500m_v2.json"
    )
    conclusion_validation_path = (
        root
        / "experiments/g4_future_items/protocol/native500m/evidence"
        / "rq1_rq2_conclusion_validation.json"
    )
    write_unexpected_result_diagnostics(
        unexpected_path,
        evaluation_path=evidence_path,
        target_statistics_path=target_statistics_path,
        user_validation_path=conclusion_validation_path,
    )
    tuning_path = root / "experiments/g4_future_items/scratchpad/tuning_native500m.md"
    compact_path = root / "experiments/g4_future_items/scratchpad/compact_native500m.md"
    tuning_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_path.write_text(build_tuning_report(evidence))
    compact_path.write_text(build_compact_report(evidence, rq3_decision=rq3_decision))
    return (
        evidence_path,
        target_statistics_path,
        unexpected_path,
        tuning_path,
        compact_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--rq3-decision", type=Path, required=True)
    arguments = parser.parse_args()
    for path in generate(
        arguments.repo_root,
        rq3_decision_path=arguments.rq3_decision.resolve(strict=True),
    ):
        print(path)


if __name__ == "__main__":
    main()
