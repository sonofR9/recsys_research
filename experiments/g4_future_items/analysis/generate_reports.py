from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.g4_future_items.configs.selectors import (
    SelectorTrialResult,
    select_family_winner,
    selector_trial_from_job,
)
from experiments.g4_future_items.launchers.run_selectors import (
    SelectorSearchResult,
    load_gate_result,
    load_search_result,
)
from experiments.g4_future_items.protocol.manifest import (
    canonical_sha256,
    load_ledger,
    load_strict_json,
    verify_ledger_semantics,
)
from experiments.g4_future_items.report.rq1_rq2_evidence import (
    verify_rq1_rq2_evidence,
)
from experiments.g4_future_items.selectors import SelectorMetrics


ROLE_TITLES = {
    "control_next_item": "Control: next liked item",
    "rq1_24h": "RQ1: next 24 hours",
    "rq2_next10": "RQ2: next 10 liked events",
}
RQ_TITLES = (
    "RQ1: Does a 24-hour future window help?",
    "RQ2: Does a next-10-liked-events window help?",
    "RQ3: Can behavior-similar future periods define better positives?",
)


def _row(values: Sequence[str], *, selected: bool = False) -> str:
    rendered = [f"**{value}**" if selected else value for value in values]
    return "| " + " | ".join(rendered) + " |"


def _rate(value: Any) -> str:
    return f"{float(value):.6g}"


def _optional(value: Any, *, scale: float = 1.0) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{float(value) / scale:.6g}"


def _recommender_table(
    candidates: Sequence[Mapping[str, Any]], winner_row_id: str
) -> list[str]:
    lines = [
        "| trial | embedding lr | deep lr | horizon | best epoch | "
        "validation recall@100 | validation loss |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in candidates:
        parameters = candidate["parameters"]
        selected = candidate["row_id"] == winner_row_id
        row_parts = str(candidate["row_id"]).split(":")
        trial_label = (
            f"B{row_parts[-2]}-{row_parts[-1]}"
            if "boundary" in row_parts[0]
            else row_parts[-1]
        )
        lines.append(
            _row(
                (
                    trial_label,
                    _rate(parameters["embedding_learning_rate"]),
                    _rate(parameters["deep_learning_rate"]),
                    str(candidate["declared_horizon_epochs"]),
                    str(candidate["restored_best_epoch"]),
                    f"{float(candidate['validation_recall_at_100']):.3f}",
                    f"{float(candidate['validation_loss']):.3f}",
                ),
                selected=selected,
            )
        )
    return lines


def _selector_result(document: Mapping[str, Any]) -> SelectorTrialResult:
    metrics = SelectorMetrics(**document["validation_metrics"])
    return SelectorTrialResult(
        trial=selector_trial_from_job(document["trial"]),
        metrics=metrics,
        artifact_sha256=str(document["output_artifact_sha256"]),
    )


def _selector_table(
    documents: Sequence[Mapping[str, Any]], winner_sha256: str
) -> list[str]:
    lines = [
        "| trial | width hours | lookahead days | minimum events | tolerance hours | "
        "frequency entity | leaves | learning rate | L2 | validation ndcg@10 | "
        "validation auroc |",
        "| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for document in documents:
        trial = document["trial"]
        metrics = document["validation_metrics"]
        lines.append(
            _row(
                (
                    (
                        f"boundary {trial['boundary_round']} trial {int(trial['trial_id']):02d}"
                        if trial["boundary_round"] is not None
                        else f"trial {int(trial['trial_id']):02d}"
                    ),
                    _optional(trial["period_width_seconds"], scale=3600),
                    _optional(trial["lookahead_seconds"], scale=86400),
                    str(trial["minimum_liked_events"]),
                    _optional(trial["time_tolerance_seconds"], scale=3600),
                    _optional(trial["frequency_entity"]),
                    _optional(trial["max_leaf_nodes"]),
                    _optional(trial["learning_rate"]),
                    _optional(trial["l2_regularization"]),
                    f"{float(metrics['user_balanced_ndcg_at_10']):.3f}",
                    (
                        "—"
                        if metrics["auroc"] is None
                        else f"{float(metrics['auroc']):.3f}"
                    ),
                ),
                selected=document["output_artifact_sha256"] == winner_sha256,
            )
        )
    return lines


def build_tuning_report(
    recommender_evidence: Mapping[str, Any],
    selector_artifacts: Sequence[Mapping[str, Any]],
) -> str:
    lines = ["# G4 native-50M hyperparameter tuning", ""]
    provenance = recommender_evidence["selection_provenance"]
    for role, title in ROLE_TITLES.items():
        selection = provenance[role]
        lines.extend((f"## {title}", ""))
        lines.extend(
            _recommender_table(
                selection["candidates"], str(selection["winner"]["row_id"])
            )
        )
        lines.append("")

    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for artifact in selector_artifacts:
        family = str(artifact["trial"]["family"])
        by_family.setdefault(family, []).append(artifact)
    for family in ("time", "content", "frequency", "learned"):
        documents = by_family.get(family)
        if not documents:
            continue
        results = [_selector_result(document) for document in documents]
        winner = select_family_winner(results)
        lines.extend((f"## RQ3 selector: {family}", ""))
        lines.extend(_selector_table(documents, winner.artifact_sha256))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _change_cell(
    reference: float, value: float, relative_dispersion: float | None
) -> str:
    percent = 100.0 * (value / reference - 1.0)
    rendered = f"{percent:+.1f}% ({value:.3f})".replace("-", "−")
    if relative_dispersion is not None:
        delta = value - reference
        threshold = relative_dispersion * reference
        if delta > threshold:
            return f'<span style="color: green">{rendered}</span>'
        if -delta > threshold:
            return f'<span style="color: red">{rendered}</span>'
    return rendered


def _compact_recommender_table(
    evidence: Mapping[str, Any], roles: Sequence[tuple[str, str]], selected_role: str
) -> list[str]:
    rows = evidence["overall"]["rows"]
    relative = evidence["calibration"]["relative_dispersion"]
    reference = rows["control_next_item"]
    lines = [
        "| variant | recall@100 | ndcg@100 | selected horizon | restored epoch |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for role, label in roles:
        metrics = rows[role]
        if role == "control_next_item":
            recall = f"{float(metrics['recall@100']):.3f}"
            ndcg = f"{float(metrics['ndcg@100']):.3f}"
        else:
            recall = _change_cell(
                float(reference["recall@100"]),
                float(metrics["recall@100"]),
                float(relative["recall@100"]),
            )
            ndcg = _change_cell(
                float(reference["ndcg@100"]),
                float(metrics["ndcg@100"]),
                float(relative["ndcg@100"]),
            )
        selected_run = evidence["selected_runs"][role]
        lines.append(
            _row(
                (
                    label,
                    recall,
                    ndcg,
                    str(selected_run["declared_horizon_epochs"]),
                    str(selected_run["restored_best_epoch"]),
                ),
                selected=role == selected_role,
            )
        )
    return lines


_DISTANCE_SLICES = (
    ("target_distance_0_6h", "0–6 hours"),
    ("target_distance_6_24h", "6–24 hours"),
    ("target_distance_1_3d", "1–3 days"),
    ("target_distance_3_7d", "3–7 days"),
)
_ACTIVITY_SLICES = (
    ("user_activity_q1", "Q1, least active"),
    ("user_activity_q2", "Q2"),
    ("user_activity_q3", "Q3"),
    ("user_activity_q4", "Q4, most active"),
)


def _compact_slice_table(
    evidence: Mapping[str, Any],
    slices: Sequence[tuple[str, str]],
    columns: Sequence[tuple[str, str, str]],
    first_header: str,
) -> list[str]:
    relative = evidence["calibration"]["relative_dispersion"]
    lines = [
        "| "
        + " | ".join((first_header, *(header for _, _, header in columns)))
        + " |",
        "| :--- | " + " | ".join(":---:" for _ in columns) + " |",
    ]
    for slice_id, label in slices:
        rows = evidence["slices"][slice_id]["rows"]
        rendered = [label]
        for role, metric, _ in columns:
            value = float(rows[role][metric])
            reference = float(rows["control_next_item"][metric])
            rendered.append(
                f"{value:.3f}"
                if role == "control_next_item"
                else _change_cell(reference, value, float(relative[metric]))
            )
        lines.append(_row(rendered))
    return lines


def build_compact_report(
    recommender_evidence: Mapping[str, Any], gate: Mapping[str, Any]
) -> str:
    lines = ["# G4 native-50M compact results", "", f"## {RQ_TITLES[0]}", ""]
    lines.extend(
        _compact_recommender_table(
            recommender_evidence,
            (
                ("control_next_item", "next liked item"),
                ("rq1_24h", "uniform liked event in the next 24 hours"),
            ),
            "control_next_item",
        )
    )
    lines.append("")
    lines.extend(
        _compact_slice_table(
            recommender_evidence,
            _DISTANCE_SLICES,
            (
                ("control_next_item", "recall@100", "next-item recall@100"),
                ("rq1_24h", "recall@100", "24-hour recall@100"),
                ("control_next_item", "ndcg@100", "next-item ndcg@100"),
                ("rq1_24h", "ndcg@100", "24-hour ndcg@100"),
            ),
            "target distance",
        )
    )
    lines.append("")
    lines.extend(
        _compact_slice_table(
            recommender_evidence,
            _ACTIVITY_SLICES,
            (
                ("control_next_item", "recall@100", "next-item recall@100"),
                ("rq1_24h", "recall@100", "24-hour recall@100"),
                ("control_next_item", "ndcg@100", "next-item ndcg@100"),
                ("rq1_24h", "ndcg@100", "24-hour ndcg@100"),
            ),
            "user activity",
        )
    )
    lines.extend(("", f"## {RQ_TITLES[1]}", ""))
    lines.extend(
        _compact_recommender_table(
            recommender_evidence,
            (
                ("control_next_item", "next liked item"),
                ("rq2_next10", "uniform among next 10 liked events"),
                ("rq1_24h", "uniform liked event in the next 24 hours"),
            ),
            "rq2_next10",
        )
    )
    rq2_slice_columns = (
        ("control_next_item", "recall@100", "next-item recall@100"),
        ("rq1_24h", "recall@100", "24-hour recall@100"),
        ("rq2_next10", "recall@100", "next-10 recall@100"),
        ("control_next_item", "ndcg@100", "next-item ndcg@100"),
        ("rq2_next10", "ndcg@100", "next-10 ndcg@100"),
    )
    lines.append("")
    lines.extend(
        _compact_slice_table(
            recommender_evidence,
            _DISTANCE_SLICES,
            rq2_slice_columns,
            "target distance",
        )
    )
    lines.append("")
    lines.extend(
        _compact_slice_table(
            recommender_evidence,
            _ACTIVITY_SLICES,
            rq2_slice_columns,
            "user activity",
        )
    )
    deterministic = gate["deterministic"]["test_metrics"]
    learned = gate["learned"]["test_metrics"]
    deterministic_ndcg = float(deterministic["user_balanced_ndcg_at_10"])
    deterministic_auroc = float(deterministic["auroc"])
    lines.extend(
        (
            "",
            f"## {RQ_TITLES[2]}",
            "",
            "| selector | test user-balanced ndcg@10 | test auroc |",
            "| :--- | :---: | :---: |",
            _row(
                (
                    "strongest deterministic",
                    f"{deterministic_ndcg:.3f}",
                    f"{deterministic_auroc:.3f}",
                )
            ),
            _row(
                (
                    "learned",
                    _change_cell(
                        deterministic_ndcg,
                        float(learned["user_balanced_ndcg_at_10"]),
                        None,
                    ),
                    _change_cell(
                        deterministic_auroc,
                        float(learned["auroc"]),
                        None,
                    ),
                ),
                selected=True,
            ),
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def validate_selector_artifact(
    ledger_job: Mapping[str, Any],
    document: Mapping[str, Any],
    result: SelectorSearchResult,
) -> None:
    expected_trial = selector_trial_from_job(ledger_job)
    if result.trial != expected_trial or document.get("trial") != expected_trial.to_dict():
        raise ValueError("selector trial differs from its ledger job")
    expected_sha256 = ledger_job["output_artifact_sha256"]
    if (
        result.artifact_sha256 != expected_sha256
        or document.get("output_artifact_sha256") != expected_sha256
    ):
        raise ValueError("selector artifact identity differs from its ledger job")
    if document.get("validation_metrics") != asdict(result.metrics):
        raise ValueError("selector artifact metrics differ from the validated result")


def validate_gate_artifact(
    ledger_job: Mapping[str, Any],
    gate: Mapping[str, Any],
    search_results: Sequence[SelectorSearchResult],
) -> None:
    if gate.get("output_artifact_sha256") != ledger_job["output_artifact_sha256"]:
        raise ValueError("selector gate identity differs from its ledger job")
    results_by_sha = {result.artifact_sha256: result for result in search_results}
    for name in ("deterministic", "learned"):
        expected_sha256 = ledger_job[f"{name}_artifact_sha256"]
        expected_payload_sha256 = ledger_job[f"{name}_payload_sha256"]
        selected = gate.get(name)
        result = results_by_sha.get(expected_sha256)
        if (
            not isinstance(selected, Mapping)
            or selected.get("artifact_sha256") != expected_sha256
            or selected.get("artifact_payload_sha256") != expected_payload_sha256
            or result is None
            or result.artifact_payload_sha256 != expected_payload_sha256
        ):
            raise ValueError(f"selector gate {name} winner differs from frozen evidence")


def _gate_payload_from_materialization_ledger(
    ledger: Mapping[str, Any], gate_sha256: str
) -> str:
    identities = {
        (row["job"]["input_artifact_sha256"], row["job"]["input_payload_sha256"])
        for row in ledger["rows"]
    }
    if len(identities) != 1:
        raise ValueError("materialization ledger does not freeze one selector gate")
    frozen_sha256, frozen_payload_sha256 = identities.pop()
    if frozen_sha256 != gate_sha256:
        raise ValueError("materialization ledger freezes a different selector gate")
    return str(frozen_payload_sha256)


def _load_historical_selector_ledger(
    path: Path, *, compatibility_path: Path, repo_root: Path
) -> dict[str, Any]:
    ledger = load_ledger(path)
    compatibility = load_strict_json(compatibility_path)
    if canonical_sha256(compatibility) != ledger["treatment_semantics_manifest_sha256"]:
        raise ValueError("selector ledger semantics differ from frozen compatibility")
    evidence_identity = compatibility["compatibility"]["evidence"]
    evidence_path = (repo_root / evidence_identity["path"]).resolve()
    if not evidence_path.is_relative_to(repo_root) or not evidence_path.is_file():
        raise ValueError("selector compatibility evidence is missing")
    evidence = load_strict_json(evidence_path)
    if (
        canonical_sha256(evidence) != evidence_identity["sha256"]
        or evidence.get("scope") != compatibility["compatibility"]["scope"]
        or evidence.get("predecessor_treatment_semantics_manifest_sha256")
        != compatibility["historical_lineage"]["treatment_semantics"]["sha256"]
        or evidence.get("source_changes") != compatibility["source_changes"]
    ):
        raise ValueError("selector compatibility evidence differs")
    return ledger


def generate_reports(repo_root: Path, output_root: Path) -> tuple[Path, Path]:
    experiment_root = repo_root / "experiments/g4_future_items"
    evidence = verify_rq1_rq2_evidence(
        experiment_root / "evidence/rq1_rq2_evaluation_native50m.json",
        repo_root=repo_root,
    )
    compatibility_v2 = experiment_root / "protocol/treatment_semantics_compatibility_v2.json"
    selector_ledger = _load_historical_selector_ledger(
        experiment_root / "protocol/ledgers/selector_search.json",
        compatibility_path=compatibility_v2,
        repo_root=repo_root,
    )
    selector_artifacts = []
    search_results = []
    search_root = repo_root / "generated/g4_selector_search"
    for row in selector_ledger["rows"]:
        artifact_sha256 = row["job"]["output_artifact_sha256"]
        result = load_search_result(search_root, artifact_sha256)
        artifact = load_strict_json(
            search_root / artifact_sha256 / "artifact.json"
        )
        validate_selector_artifact(row["job"], artifact, result)
        selector_artifacts.append(artifact)
        search_results.append(result)

    gate_ledger = _load_historical_selector_ledger(
        experiment_root / "protocol/ledgers/selector_gate.json",
        compatibility_path=compatibility_v2,
        repo_root=repo_root,
    )
    gate_job = gate_ledger["rows"][0]["job"]
    gate_sha256 = gate_job["output_artifact_sha256"]
    materialization_ledger = load_ledger(
        experiment_root
        / "protocol/ledgers/selector_materialization_final_v8.json"
    )
    verify_ledger_semantics(
        materialization_ledger,
        {
            "treatment_semantics_manifest_sha256": experiment_root
            / "protocol/treatment_semantics_compatibility_v8.json"
        },
    )
    gate = load_gate_result(
        repo_root / "generated/g4_selector_gate",
        gate_sha256,
        expected_payload_sha256=_gate_payload_from_materialization_ledger(
            materialization_ledger, gate_sha256
        ),
    )
    validate_gate_artifact(gate_job, gate, search_results)

    output_root.mkdir(parents=True, exist_ok=True)
    tuning_path = output_root / "tuning_native50m.md"
    compact_path = output_root / "compact_native50m.md"
    tuning_path.write_text(build_tuning_report(evidence, selector_artifacts))
    compact_path.write_text(build_compact_report(evidence, gate))
    return tuning_path, compact_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/g4_future_items/analysis/generated"),
    )
    arguments = parser.parse_args()
    tuning_path, compact_path = generate_reports(
        arguments.repo_root.resolve(), arguments.output_root
    )
    print(tuning_path)
    print(compact_path)


if __name__ == "__main__":
    main()
