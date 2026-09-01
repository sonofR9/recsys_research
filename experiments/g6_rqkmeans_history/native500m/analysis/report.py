from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from experiments.g6_rqkmeans_history.native500m.analysis.collect import (
    NATIVE500M_RELATIVE_DISPERSIONS,
    collect_stage_candidates,
)
from experiments.g6_rqkmeans_history.native500m.launchers.queue import (
    canonical_bytes,
    load_queue_manifest,
    persist_immutable_bytes,
)
from experiments.g6_rqkmeans_history.native500m.protocol.selection import (
    Candidate,
    MetricValues,
    SeedEvidence,
    decide_rq1_initialization,
    decide_rq23,
    promote_against_two_baselines,
    select_by_quality,
)


METRICS = ("recall@100", "ndcg@100", "mrr@100", "coverage@100")
SID_CUTOFFS = (10, 50, 100)
SLICE_NAMES = (
    "frequency_low",
    "frequency_middle",
    "frequency_high",
    "history_has_collided_base_sid",
    "history_has_no_collided_base_sid",
    "target_bucket_size_1",
    "target_bucket_size_2",
    "target_bucket_size_3_to_4",
    "target_bucket_size_5_plus",
)
LABELS = {
    "recall@100": "Recall@100",
    "ndcg@100": "NDCG@100",
    "mrr@100": "MRR@100",
    "coverage@100": "Coverage@100",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def assemble_report_evidence(
    assembly: Mapping[str, Any], *, output_path: Path | None = None
) -> dict[str, Any]:
    document = _assemble_report_evidence(assembly)
    if output_path is not None:
        persist_immutable_bytes(
            output_path,
            canonical_bytes(document),
            label="native-500M report evidence",
        )
    return document


def _assemble_report_evidence(assembly: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(assembly, Mapping):
        raise ValueError("report assembly must be an object")
    specification = json.loads(json.dumps(dict(assembly)))
    if set(specification) != {
        "schema",
        "original",
        "best_g1",
        "rqs",
        "terminal",
        "terminal_bridge",
        "terminal_bridge_selection_paths",
    }:
        raise ValueError("report assembly schema differs")
    if specification["schema"] != "g6-native500m-report-assembly/v1":
        raise ValueError("report assembly protocol differs")
    original = _assembled_control(specification["original"], "original_baseline")
    best_g1 = _assembled_control(specification["best_g1"], "best_g1_control")
    questions = specification["rqs"]
    if not isinstance(questions, list) or [
        row.get("index") for row in questions
    ] != list(range(4)):
        raise ValueError("report assembly requires ordered RQ0-RQ3")
    rqs = []
    selected_rows = []
    for question in questions:
        candidates = question.get("candidates")
        selection_paths = question.get("selection_paths")
        if (
            not isinstance(question.get("title"), str)
            or not question["title"]
            or not isinstance(candidates, list)
            or not candidates
            or not isinstance(selection_paths, list)
            or not selection_paths
        ):
            raise ValueError("report assembly question is incomplete")
        ledgers = [_load_selection_ledger(binding) for binding in selection_paths]
        treatment_rows = [
            _assembled_treatment(candidate, order)
            for order, candidate in enumerate(candidates)
        ]
        selected_candidate = _rq_selected_candidate(question["index"], ledgers)
        matching = [
            row
            for row in treatment_rows
            if _matches_selected_evidence(row, selected_candidate)
        ]
        if len(matching) != 1:
            raise ValueError("finalist evidence differs from the RQ ledger decision")
        selected_id = matching[0]["evidence_identity"]["evidence_sha256"]
        for row in treatment_rows:
            row["selected"] = row["evidence_identity"]["evidence_sha256"] == selected_id
            decision = promote_against_two_baselines(
                Candidate(
                    row["evidence_identity"]["evidence_sha256"],
                    MetricValues(
                        row["metrics"]["recall@100"],
                        row["metrics"]["ndcg@100"],
                    ),
                    0,
                ),
                original=Candidate(
                    "original",
                    MetricValues(
                        original["metrics"]["recall@100"],
                        original["metrics"]["ndcg@100"],
                    ),
                    0,
                ),
                best_g1=Candidate(
                    "best_g1",
                    MetricValues(
                        best_g1["metrics"]["recall@100"],
                        best_g1["metrics"]["ndcg@100"],
                    ),
                    1,
                ),
                recall_relative_dispersion=NATIVE500M_RELATIVE_DISPERSIONS[
                    "recall@100"
                ],
                ndcg_relative_dispersion=NATIVE500M_RELATIVE_DISPERSIONS["ndcg@100"],
            )
            row["promoted"] = decision.promoted
        selected = [row for row in treatment_rows if row["selected"]][0]
        selected_rows.append(selected)
        rqs.append(
            {
                "index": question["index"],
                "title": question["title"],
                "rows": [original, best_g1, *treatment_rows],
                "tuning_rows": _ledger_tuning_rows(ledgers),
            }
        )
    expected_terminal = selected_rows[-1] if selected_rows[-1]["promoted"] else best_g1
    terminal = _assembled_terminal(specification["terminal"])
    if (
        terminal["evidence_identity"]["evidence_sha256"]
        != expected_terminal["evidence_identity"]["evidence_sha256"]
    ):
        raise ValueError("terminal evidence differs from derived promotion decision")
    bridge_value = specification["terminal_bridge"]
    bridge_selection_paths = specification["terminal_bridge_selection_paths"]
    if not isinstance(bridge_selection_paths, list):
        raise ValueError("terminal bridge selection bindings are invalid")
    terminal_is_best_g1 = (
        terminal["evidence_identity"]["evidence_sha256"]
        == best_g1["evidence_identity"]["evidence_sha256"]
    )
    if terminal_is_best_g1:
        if bridge_value is not None:
            raise ValueError("a no-SID terminal must not declare a SID bridge")
        if bridge_selection_paths:
            raise ValueError("a no-SID terminal must not declare bridge selection")
        terminal_bridge = None
    else:
        if not bridge_selection_paths:
            raise ValueError(
                "a SID terminal requires an authenticated bridge selection"
            )
        bridge_ledgers = [
            _load_selection_ledger(binding) for binding in bridge_selection_paths
        ]
        bridge_winner = _bridge_selected_candidate(bridge_ledgers)
        terminal_bridge = _assembled_terminal(bridge_value)
        if (
            not _matches_selected_evidence(terminal_bridge, bridge_winner)
            or terminal_bridge["parameters"].get("backbone") != "original_g1"
            or not _same_atomic_sid_bundle(
                terminal_bridge["parameters"], terminal["parameters"]
            )
        ):
            raise ValueError("terminal bridge differs from the atomic SID bundle")
    best_g1_gain = {
        metric: best_g1["metrics"][metric] - original["metrics"][metric]
        for metric in METRICS
    }
    sid_marginal = {
        metric: (
            0.0
            if terminal_bridge is None
            else terminal_bridge["metrics"][metric] - original["metrics"][metric]
        )
        for metric in METRICS
    }
    component_gains = {
        metric: best_g1_gain[metric] + sid_marginal[metric] for metric in METRICS
    }
    body = {
        "schema": "g6-native500m-report-evidence/v2",
        "dataset_size": "native-500m",
        "relative_dispersion": dict(NATIVE500M_RELATIVE_DISPERSIONS),
        "assembly": specification,
        "rqs": rqs,
        "aggregate": {
            "original": original,
            "best_g1": best_g1,
            "terminal": terminal,
            "terminal_bridge": terminal_bridge,
            "best_g1_gain": best_g1_gain,
            "sid_marginal": sid_marginal,
            "component_gains": component_gains,
        },
    }
    return {
        **body,
        "assembly_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest(),
    }


def _assembled_control(value: object, role: str) -> dict[str, Any]:
    row = _assembled_run(value)
    row["role"] = role
    return row


def _assembled_treatment(value: object, order: int) -> dict[str, Any]:
    row = _assembled_run(value)
    final_evaluation = row.pop("final_evaluation")
    parameters = row["parameters"]
    artifacts = final_evaluation.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("semantic final evaluation diagnostics are missing")
    diagnostics = artifacts.get("semantic_id_diagnostics")
    slices = artifacts.get("slice_diagnostics")
    if not isinstance(diagnostics, dict) or not isinstance(slices, dict):
        raise ValueError("semantic report diagnostics are incomplete")
    row.update(
        {
            "role": "treatment",
            "semantic_levels": parameters["levels"],
            "sid_diagnostics": diagnostics,
            "slice_diagnostics": [
                _assembled_slice(name, slices[name]) for name in SLICE_NAMES
            ],
            "manifest_order": order,
        }
    )
    return row


def _assembled_slice(name: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("semantic slice diagnostics are incomplete")
    control = value.get("control")
    semantic = value.get("semantic")
    if not isinstance(control, dict) or not isinstance(semantic, dict):
        raise ValueError("semantic slice metrics are incomplete")
    return {
        "slice": name,
        "control_recall@100": control.get("recall@100"),
        "treatment_recall@100": semantic.get("recall@100"),
        "num_users": value.get("num_users"),
        "num_targets": value.get("num_targets"),
    }


def _assembled_terminal(value: object) -> dict[str, Any]:
    row = _assembled_run(value)
    row.pop("final_evaluation", None)
    return row


def _assembled_run(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"method", "path"}:
        raise ValueError("assembled run binding differs")
    method = value["method"]
    path_value = value["path"]
    if not isinstance(method, str) or not method or not isinstance(path_value, str):
        raise ValueError("assembled run identity is invalid")
    path = Path(path_value).resolve(strict=True)
    document, content = _load_collected_run(path)
    contract_path = Path(document["artifacts"]["g6_native500m_job.json"]["path"])
    contract = json.loads(contract_path.read_text())
    parameters = contract["job"]["parameters"]
    final_path = Path(document["artifacts"]["final_evaluation.json"]["path"])
    final_evaluation = json.loads(final_path.read_text())
    return {
        "method": method,
        "job_id": document["job_id"],
        "stage": document["stage"],
        "metrics": document["metrics"],
        "parameters": parameters,
        "best_epoch": document["best_epoch"],
        "trained_epochs": document["trained_epochs"],
        "final_evaluation": final_evaluation,
        "evidence_identity": {
            "schema": document["schema"],
            "manifest_logical_sha256": document["manifest_logical_sha256"],
            "job_logical_sha256": document["job_logical_sha256"],
            "evidence_sha256": document["evidence_sha256"],
            "path": str(path),
            "physical_sha256": hashlib.sha256(content).hexdigest(),
        },
    }


def _load_collected_run(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("collected run must be a regular file")
    content = path.read_bytes()
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("collected run is invalid JSON") from error
    required = {
        "schema",
        "stage",
        "job_id",
        "run_name",
        "manifest_logical_sha256",
        "manifest_physical_sha256",
        "job_logical_sha256",
        "batch_id",
        "best_epoch",
        "trained_epochs",
        "metrics",
        "artifacts",
        "evidence_sha256",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("collected run schema differs")
    body = {key: value for key, value in document.items() if key != "evidence_sha256"}
    if (
        document["schema"] != "g6-native500m-collected-run/v1"
        or document["evidence_sha256"]
        != hashlib.sha256(canonical_bytes(body)).hexdigest()
        or document["trained_epochs"] != 26
        or not isinstance(document["best_epoch"], int)
        or not 1 <= document["best_epoch"] <= 26
    ):
        raise ValueError("collected run identity or horizon differs")
    artifacts = document["artifacts"]
    expected_artifacts = {
        "g6_native500m_job.json",
        "training_metadata.json",
        "final_metrics.json",
        "ranking_evidence.pt",
        "top100_item_evidence.pt",
        "final_evaluation.json",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("collected run artifact set differs")
    for name, identity in artifacts.items():
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
            raise ValueError("collected run artifact identity is invalid")
        artifact_path = Path(identity["path"]).resolve(strict=True)
        if (
            artifact_path.is_symlink()
            or not artifact_path.is_file()
            or hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            != identity["sha256"]
        ):
            raise ValueError(f"collected run artifact differs: {name}")
    contract = json.loads(Path(artifacts["g6_native500m_job.json"]["path"]).read_text())
    metadata = json.loads(Path(artifacts["training_metadata.json"]["path"]).read_text())
    final_metrics = json.loads(
        Path(artifacts["final_metrics.json"]["path"]).read_text()
    )
    if (
        contract.get("manifest_logical_sha256") != document["manifest_logical_sha256"]
        or contract.get("job_logical_sha256") != document["job_logical_sha256"]
        or contract.get("job", {}).get("job_id") != document["job_id"]
        or metadata.get("best_epoch") != document["best_epoch"]
        or metadata.get("epochs_trained") != 26
        or final_metrics != document["metrics"]
    ):
        raise ValueError("collected run evidence chain differs")
    return document, content


def _load_selection_ledger(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "manifest_path",
        "logs_root",
        "queue_state_directory",
    }:
        raise ValueError("stage selection ledger binding differs")
    path = Path(value["path"])
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("stage selection ledger must be a regular file")
    try:
        document = json.loads(resolved.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("stage selection ledger is invalid JSON") from error
    required = {
        "schema",
        "stage",
        "manifest_logical_sha256",
        "manifest_physical_sha256",
        "batch_id",
        "recall_relative_dispersion",
        "selection_group_field",
        "selected_job_ids",
        "candidates",
        "selection_sha256",
    }
    body = {key: value for key, value in document.items() if key != "selection_sha256"}
    if (
        not isinstance(document, dict)
        or set(document) != required
        or document["schema"] != "g6-native500m-stage-selection/v1"
        or document["recall_relative_dispersion"]
        != NATIVE500M_RELATIVE_DISPERSIONS["recall@100"]
        or document["selection_sha256"]
        != hashlib.sha256(canonical_bytes(body)).hexdigest()
    ):
        raise ValueError("stage selection ledger identity differs")
    candidates = document["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("stage selection ledger has no candidates")
    manifest = load_queue_manifest(Path(value["manifest_path"]))
    if (
        manifest.stage != document["stage"]
        or manifest.logical_sha256 != document["manifest_logical_sha256"]
        or manifest.physical_sha256 != document["manifest_physical_sha256"]
    ):
        raise ValueError("stage selection manifest binding differs")
    rederived = collect_stage_candidates(
        manifest=manifest,
        logs_root=Path(value["logs_root"]),
        queue_state_directory=Path(value["queue_state_directory"]),
        batch_id=document["batch_id"],
        output_path=None,
    )
    if rederived != document:
        raise ValueError("stage selection differs from authenticated artifacts")
    return document


def _rq_selected_candidate(index: int, ledgers: list[dict[str, Any]]) -> dict[str, Any]:
    if index == 0:
        ledger = ledgers[-1]
        selected = set(ledger["selected_job_ids"].values())
        rows = [row for row in ledger["candidates"] if row["job_id"] in selected]
        winner = select_by_quality(
            [
                Candidate(
                    row["job_id"],
                    MetricValues(
                        row["validation_metrics"]["recall@100"],
                        row["validation_metrics"]["ndcg@100"],
                    ),
                    row["manifest_order"],
                )
                for row in rows
            ],
            recall_relative_dispersion=NATIVE500M_RELATIVE_DISPERSIONS["recall@100"],
        ).identifier
        return _ledger_candidate(
            next(row for row in rows if row["job_id"] == winner), ledger
        )
    if index == 1:
        ledger = _one_stage_ledger(ledgers, "rq1_confirmation")
        grouped = {
            initialization: sorted(
                [
                    row
                    for row in ledger["candidates"]
                    if row["parameters"]["sid_initialization"] == initialization
                ],
                key=lambda row: row["parameters"]["seed"],
            )
            for initialization in ("random", "content_pca")
        }
        decision = decide_rq1_initialization(
            [_ledger_seed(row) for row in grouped["random"]],
            [_ledger_seed(row) for row in grouped["content_pca"]],
            recall_relative_dispersion=NATIVE500M_RELATIVE_DISPERSIONS["recall@100"],
            ndcg_relative_dispersion=NATIVE500M_RELATIVE_DISPERSIONS["ndcg@100"],
        )
        return _ledger_candidate(
            next(
                row
                for row in grouped[decision.selected]
                if row["parameters"]["seed"] == 42
            ),
            ledger,
        )
    ledger = _one_stage_ledger(ledgers, "rq2_rq3_confirmation")
    grouped = {
        "rq0": sorted(
            [row for row in ledger["candidates"] if "rq0_anchor" in row["job_id"]],
            key=lambda row: row["parameters"]["seed"],
        ),
        "suffix": sorted(
            [
                row
                for row in ledger["candidates"]
                if "rq0_anchor" not in row["job_id"]
                and row["parameters"]["collision_policy"] == "suffix"
            ],
            key=lambda row: row["parameters"]["seed"],
        ),
        "none": sorted(
            [
                row
                for row in ledger["candidates"]
                if row["parameters"]["collision_policy"] == "none"
            ],
            key=lambda row: row["parameters"]["seed"],
        ),
    }
    decision = decide_rq23(
        [_ledger_seed(row) for row in grouped["rq0"]],
        [_ledger_seed(row) for row in grouped["suffix"]],
        [_ledger_seed(row) for row in grouped["none"]],
        recall_relative_dispersion=NATIVE500M_RELATIVE_DISPERSIONS["recall@100"],
        ndcg_relative_dispersion=NATIVE500M_RELATIVE_DISPERSIONS["ndcg@100"],
    )
    selected = decision.rq2_selected if index == 2 else decision.terminal_selected
    group = "rq0" if selected == "rq0" else selected
    return _ledger_candidate(
        next(row for row in grouped[group] if row["parameters"]["seed"] == 42),
        ledger,
    )


def _one_stage_ledger(ledgers: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    matches = [ledger for ledger in ledgers if ledger["stage"] == stage]
    if len(matches) != 1:
        raise ValueError(f"report assembly requires exactly one {stage} ledger")
    return matches[0]


def _bridge_selected_candidate(
    ledgers: list[dict[str, Any]],
) -> dict[str, Any]:
    stages = [ledger["stage"] for ledger in ledgers]
    if (
        not stages
        or len(set(stages)) != len(stages)
        or any(
            stage not in {"terminal_bridge", "terminal_bridge_boundary"}
            for stage in stages
        )
        or stages[-1] not in {"terminal_bridge", "terminal_bridge_boundary"}
        or (
            "terminal_bridge_boundary" in stages
            and stages[-1] != "terminal_bridge_boundary"
        )
    ):
        raise ValueError("terminal bridge selection chain differs")
    ledger = ledgers[-1]
    selected = set(ledger["selected_job_ids"].values())
    rows = [row for row in ledger["candidates"] if row["job_id"] in selected]
    if len(rows) != 1:
        raise ValueError("terminal bridge selection is ambiguous")
    return _ledger_candidate(rows[0], ledger)


def _ledger_candidate(row: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "stage": ledger["stage"],
        "manifest_logical_sha256": ledger["manifest_logical_sha256"],
    }


def _ledger_seed(row: dict[str, Any]) -> SeedEvidence:
    convergence = row.get("convergence")
    if not isinstance(convergence, dict):
        raise ValueError("confirmation ledger lacks convergence evidence")
    return SeedEvidence(
        seed=row["parameters"]["seed"],
        metrics=MetricValues(
            row["validation_metrics"]["recall@100"],
            row["validation_metrics"]["ndcg@100"],
        ),
        first_epoch_at_95_percent=convergence["first_epoch_at_95_percent"],
        normalized_recall_auc=convergence["normalized_recall_auc"],
    )


def _same_report_treatment(
    actual: dict[str, object], selected: dict[str, object]
) -> bool:
    fields = (
        "backbone",
        "embedding_learning_rate",
        "deep_learning_rate",
        "seed",
        "representation",
        "levels",
        "shared_codes",
        "collision_policy",
        "sid_initialization",
    )
    return all(actual.get(field) == selected.get(field) for field in fields)


def _matches_selected_evidence(
    collected: dict[str, Any], selected: dict[str, Any]
) -> bool:
    expected = _selected_evidence_identity(selected)
    identity = collected["evidence_identity"]
    return (
        collected["job_id"] == expected["job_id"]
        and collected["stage"] == expected["stage"]
        and identity["job_logical_sha256"] == expected["job_logical_sha256"]
        and identity["manifest_logical_sha256"] == expected["manifest_logical_sha256"]
        and _same_report_treatment(collected["parameters"], selected["parameters"])
    )


def _selected_evidence_identity(selected: dict[str, Any]) -> dict[str, str]:
    reused = selected.get("reused_from")
    if reused is None:
        return {
            "job_id": selected["job_id"],
            "stage": selected["stage"],
            "job_logical_sha256": selected["job_logical_sha256"],
            "manifest_logical_sha256": selected["manifest_logical_sha256"],
        }
    artifacts = selected.get("artifacts")
    contract_identity = (
        artifacts.get("g6_native500m_job.json") if isinstance(artifacts, dict) else None
    )
    if (
        not isinstance(reused, dict)
        or set(reused) != {"selection_sha256", "job_id", "job_logical_sha256"}
        or not _SHA256.fullmatch(str(reused["selection_sha256"]))
        or not isinstance(reused["job_id"], str)
        or not reused["job_id"]
        or not _SHA256.fullmatch(str(reused["job_logical_sha256"]))
        or not isinstance(contract_identity, dict)
    ):
        raise ValueError("selected exact-reuse evidence is incomplete")
    contract_binding = Path(contract_identity["path"])
    if contract_binding.is_symlink() or not contract_binding.is_file():
        raise ValueError("selected exact-reuse source contract differs")
    contract_path = contract_binding.resolve(strict=True)
    contract_content = contract_path.read_bytes()
    if hashlib.sha256(contract_content).hexdigest() != contract_identity.get("sha256"):
        raise ValueError("selected exact-reuse source contract differs")
    contract = json.loads(contract_content)
    source_job = contract.get("job")
    if (
        not isinstance(source_job, dict)
        or not isinstance(source_job.get("job_id"), str)
        or not source_job["job_id"]
        or not isinstance(source_job.get("stage"), str)
        or not source_job["stage"]
        or not _SHA256.fullmatch(str(contract.get("job_logical_sha256")))
        or not _SHA256.fullmatch(str(contract.get("manifest_logical_sha256")))
    ):
        raise ValueError("selected exact-reuse source identity differs")
    return {
        "job_id": source_job["job_id"],
        "stage": source_job["stage"],
        "job_logical_sha256": contract["job_logical_sha256"],
        "manifest_logical_sha256": contract["manifest_logical_sha256"],
    }


def _same_atomic_sid_bundle(
    actual: dict[str, object], selected: dict[str, object]
) -> bool:
    fields = (
        "representation",
        "representation_width",
        "levels",
        "shared_codes",
        "collision_policy",
        "sid_initialization",
    )
    return all(actual.get(field) == selected.get(field) for field in fields)


def _ledger_tuning_rows(ledgers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ledger in ledgers:
        for candidate in ledger["candidates"]:
            logical = candidate["job_logical_sha256"]
            if logical in seen:
                continue
            seen.add(logical)
            parameters = candidate["parameters"]
            rows.append(
                {
                    "method": candidate["job_id"],
                    "embedding_learning_rate": parameters["embedding_learning_rate"],
                    "deep_learning_rate": parameters["deep_learning_rate"],
                    "horizon": candidate["training_horizon"],
                    "best_epoch": candidate["best_epoch"],
                    "metrics": candidate["validation_metrics"],
                }
            )
    return rows


def _tuning_row(row: dict[str, Any]) -> dict[str, Any]:
    parameters = row["parameters"]
    return {
        "method": row["method"],
        "embedding_learning_rate": parameters["embedding_learning_rate"],
        "deep_learning_rate": parameters["deep_learning_rate"],
        "horizon": row["trained_epochs"],
        "best_epoch": row["best_epoch"],
        "metrics": row["metrics"],
    }


def render_compact_report(evidence: Mapping[str, Any]) -> str:
    validated = _validated(evidence)
    blocks = [
        "# G6: RQ-KMeans semantic IDs in history — native Yambda-500M",
        (
            "Native Yambda-500M; operational bands use the reviewed native-500M "
            "relative dispersions scaled by each table's own unrounded reference."
        ),
        _dispersion_table(validated["relative_dispersion"]),
    ]
    blocks.extend(
        _rq_block(rq, validated["relative_dispersion"]) for rq in validated["rqs"]
    )
    blocks.append(_aggregate_block(validated))
    return "\n\n".join(_flatten(blocks)) + "\n"


def render_tuning_report(evidence: Mapping[str, Any]) -> str:
    validated = _validated(evidence)
    blocks = ["# G6 native Yambda-500M tuning"]
    for rq in validated["rqs"]:
        blocks.extend(
            (
                f"## RQ{rq['index']}: {rq['title']}",
                _tuning_table(rq["tuning_rows"]),
            )
        )
    return "\n\n".join(blocks) + "\n"


def write_reports(
    evidence: Mapping[str, Any], output_directory: Path
) -> tuple[Path, ...]:
    validated = _validated(evidence)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "rq0_reader_native500m.md": _reader_subset(validated, (0,), aggregate=False),
        "rq0_tuning_native500m.md": _tuning_subset(validated, (0,)),
        "rq1_rq3_reader_native500m.md": _reader_subset(
            validated, (1, 2, 3), aggregate=True
        ),
        "rq1_rq3_tuning_native500m.md": _tuning_subset(validated, (1, 2, 3)),
    }
    paths = []
    for name, content in outputs.items():
        path = output_directory / name
        path.write_text(content)
        paths.append(path)
    return tuple(paths)


def _reader_subset(
    evidence: dict[str, Any], indices: tuple[int, ...], *, aggregate: bool
) -> str:
    blocks = [
        "# G6: RQ-KMeans semantic IDs in history — native Yambda-500M",
        _dispersion_table(evidence["relative_dispersion"]),
    ]
    blocks.extend(
        _rq_block(rq, evidence["relative_dispersion"])
        for rq in evidence["rqs"]
        if rq["index"] in indices
    )
    if aggregate:
        blocks.append(_aggregate_block(evidence))
    return "\n\n".join(_flatten(blocks)) + "\n"


def _tuning_subset(evidence: dict[str, Any], indices: tuple[int, ...]) -> str:
    blocks = ["# G6 native Yambda-500M tuning"]
    for rq in evidence["rqs"]:
        if rq["index"] not in indices:
            continue
        blocks.extend(
            (f"## RQ{rq['index']}: {rq['title']}", _tuning_table(rq["tuning_rows"]))
        )
    return "\n\n".join(blocks) + "\n"


def _rq_block(rq: dict[str, Any], dispersion: dict[str, float]) -> tuple[str, ...]:
    original = _one_role(rq["rows"], "original_baseline")
    best_g1 = _one_role(rq["rows"], "best_g1_control")
    treatments = [row for row in rq["rows"] if row["role"] == "treatment"]
    return (
        f"## RQ{rq['index']}: {rq['title']}",
        "### Original-G1/SASRec comparison",
        _quality_table([original, *treatments], original, dispersion),
        "### Best-G1 local comparison",
        _quality_table([best_g1, *treatments], best_g1, dispersion),
        "### SID retrieval diagnostics",
        _sid_retrieval_table(treatments),
        "### Intrinsic SID diagnostics",
        _intrinsic_sid_table(treatments),
        "### Eligible target-frequency and collision slices",
        _slice_table(treatments),
    )


def _dispersion_table(dispersion: dict[str, float]) -> str:
    rows = [
        "| Metric | Reviewed relative dispersion |",
        "| --- | ---: |",
    ]
    rows.extend(
        f"| {metric} | {100 * dispersion[metric]:.3f}% |"
        for metric in NATIVE500M_RELATIVE_DISPERSIONS
    )
    return "\n".join(rows)


def _sid_retrieval_table(rows: list[dict[str, Any]]) -> str:
    maximum_levels = max(row["semantic_levels"] for row in rows)
    columns = ["Method", "Exact@10 / @50 / @100"] + [
        f"Prefix L{level} @10 / @50 / @100" for level in range(1, maximum_levels + 1)
    ]
    rendered = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---", *["---:" for _ in columns[1:]]]) + " |",
    ]
    for row in rows:
        metrics = row["metrics"]
        values = [
            row["method"],
            _metric_triplet(metrics, "sid_exact_recall"),
            *(
                (
                    _metric_triplet(metrics, f"sid_prefix_recall", level=level)
                    if level <= row["semantic_levels"]
                    else "—"
                )
                for level in range(1, maximum_levels + 1)
            ),
        ]
        rendered.append("| " + " | ".join(values) + " |")
    return "\n".join(rendered)


def _intrinsic_sid_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Method | ICR | Collided items | Unique tuples | "
        "Bucket p50 / p95 / p99 / max | Occupied p95 by level | "
        "Occupied p95/mean by level | Intra-code cosine by level | "
        "Reconstruction residual by depth |\n"
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |"
    )
    rendered = [header]
    for row in rows:
        values = row["sid_diagnostics"]
        rendered.append(
            "| "
            + " | ".join(
                (
                    row["method"],
                    _score(values["identifier_collision_rate"]),
                    _score(values["collided_item_fraction"]),
                    str(values["unique_base_tuples"]),
                    " / ".join(
                        _score(values[name])
                        for name in (
                            "collision_bucket_size_p50",
                            "collision_bucket_size_p95",
                            "collision_bucket_size_p99",
                            "collision_bucket_size_max",
                        )
                    ),
                    _level_values(values["p95_occupied_load"]),
                    _level_values(values["p95_to_mean_occupied_load"]),
                    _level_values(values["intra_code_cosine_similarity"]),
                    _level_values(values["reconstruction_mse_by_depth"]),
                )
            )
            + " |"
        )
    return "\n".join(rendered)


def _slice_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Method | Slice | Control Recall@100 | Treatment Recall@100 | "
        "Point delta | Users | Targets |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    rendered = [header]
    for row in rows:
        for evidence in row["slice_diagnostics"]:
            control = evidence["control_recall@100"]
            treatment = evidence["treatment_recall@100"]
            rendered.append(
                "| "
                + " | ".join(
                    (
                        row["method"],
                        evidence["slice"],
                        _score(control),
                        _score(treatment),
                        _points(treatment - control),
                        str(evidence["num_users"]),
                        str(evidence["num_targets"]),
                    )
                )
                + " |"
            )
    return "\n".join(rendered)


def _quality_table(
    rows: list[dict[str, Any]],
    reference: dict[str, Any],
    dispersion: dict[str, float],
) -> str:
    header = (
        "| Method | Recall@100 | Delta Recall@100 | NDCG@100 | Delta NDCG@100 | "
        "MRR@100 | Coverage@100 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rendered = [header]
    for row in rows:
        metrics = row["metrics"]
        reference_metrics = reference["metrics"]
        values = [
            row["method"],
            _score(metrics["recall@100"]),
            _delta(
                metrics["recall@100"],
                reference_metrics["recall@100"],
                dispersion["recall@100"],
            ),
            _score(metrics["ndcg@100"]),
            _delta(
                metrics["ndcg@100"],
                reference_metrics["ndcg@100"],
                dispersion["ndcg@100"],
            ),
            _score(metrics["mrr@100"]),
            _score(metrics["coverage@100"]),
        ]
        if row.get("selected") is True:
            values = [f"**{value}**" for value in values]
        rendered.append("| " + " | ".join(values) + " |")
    return "\n".join(rendered)


def _tuning_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Method | Recall@100 | NDCG@100 | Embedding LR | Deep LR | Horizon | "
        "Best epoch |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rendered = [header]
    for row in rows:
        metrics = row["metrics"]
        rendered.append(
            "| "
            + " | ".join(
                (
                    row["method"],
                    _score(metrics["recall@100"]),
                    _score(metrics["ndcg@100"]),
                    _raw(row["embedding_learning_rate"]),
                    _raw(row["deep_learning_rate"]),
                    str(row["horizon"]),
                    str(row["best_epoch"]),
                )
            )
            + " |"
        )
    return "\n".join(rendered)


def _aggregate_block(evidence: dict[str, Any]) -> tuple[str, str]:
    aggregate = evidence["aggregate"]
    original = aggregate["original"]["metrics"]
    terminal = aggregate["terminal"]["metrics"]
    components = aggregate["component_gains"]
    best_g1_gain = aggregate["best_g1_gain"]
    sid_marginal = aggregate["sid_marginal"]
    dispersion = evidence["relative_dispersion"]
    rows = [
        (
            LABELS[metric],
            _score(original[metric]),
            _score(terminal[metric]),
            _points(terminal[metric] - original[metric]),
            _points(best_g1_gain[metric]),
            _points(sid_marginal[metric]),
            _points(components[metric]),
            _points(terminal[metric] - original[metric] - components[metric]),
            _interaction_label(
                terminal[metric] - original[metric] - components[metric],
                original[metric] * dispersion[metric],
            ),
        )
        for metric in METRICS
    ]
    table = (
        "| Metric | Original G1 | Aggregate | Aggregate gain | Best-G1 gain | "
        "SID bridge marginal | Component gain sum | Interaction gap | Interaction |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in rows)
    )
    return "## Aggregated improvement", table


def _validated(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise ValueError("report evidence must be an object")
    document = dict(evidence)
    if (
        document.get("schema") != "g6-native500m-report-evidence/v2"
        or document.get("dataset_size") != "native-500m"
    ):
        raise ValueError("report evidence protocol identity differs")
    assembly = document.get("assembly")
    if (
        not isinstance(assembly, dict)
        or _assemble_report_evidence(assembly) != document
    ):
        raise ValueError("report evidence differs from artifact-backed reassembly")
    dispersion = document.get("relative_dispersion")
    if dispersion != NATIVE500M_RELATIVE_DISPERSIONS:
        raise ValueError("native-500M relative dispersion differs from reviewed values")
    for metric, value in dispersion.items():
        _finite_positive(value, metric)
    rqs = document.get("rqs")
    if not isinstance(rqs, list) or [rq.get("index") for rq in rqs] != list(range(4)):
        raise ValueError("report requires consecutive RQ0-RQ3 evidence")
    for rq in rqs:
        if not isinstance(rq.get("title"), str) or not rq["title"]:
            raise ValueError("research question title is missing")
        rows = rq.get("rows")
        tuning = rq.get("tuning_rows")
        if not isinstance(rows, list) or not isinstance(tuning, list) or not tuning:
            raise ValueError("research question rows are incomplete")
        _one_role(rows, "original_baseline")
        _one_role(rows, "best_g1_control")
        if not any(row.get("role") == "treatment" for row in rows):
            raise ValueError("research question has no treatment")
        selected = [
            row
            for row in rows
            if row.get("role") == "treatment" and row.get("selected") is True
        ]
        if len(selected) != 1:
            raise ValueError(
                "research question requires exactly one selected treatment"
            )
        for row in rows:
            _validate_metrics(row.get("metrics"))
            _validate_evidence_identity(row.get("evidence_identity"), row["metrics"])
            if row.get("role") == "treatment":
                _validate_sid_evidence(row)
        for row in tuning:
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError("tuning metrics are incomplete")
            for metric in ("recall@100", "ndcg@100"):
                value = _finite(metrics.get(metric), metric)
                if not 0 <= value <= 1:
                    raise ValueError(f"{metric} is outside [0, 1]")
            for name in ("embedding_learning_rate", "deep_learning_rate"):
                _finite_positive(row.get(name), name)
            if row.get("horizon") != 26:
                raise ValueError("tuning row did not complete horizon 26")
            best_epoch = row.get("best_epoch")
            if (
                not isinstance(best_epoch, int)
                or isinstance(best_epoch, bool)
                or not 1 <= best_epoch <= 26
            ):
                raise ValueError("tuning row best epoch is invalid")
    aggregate = document.get("aggregate")
    if not isinstance(aggregate, dict) or set(aggregate) != {
        "original",
        "best_g1",
        "terminal",
        "terminal_bridge",
        "best_g1_gain",
        "sid_marginal",
        "component_gains",
    }:
        raise ValueError("aggregate evidence is incomplete")
    for role in ("original", "best_g1", "terminal"):
        _validate_metrics(aggregate[role].get("metrics"))
        _validate_evidence_identity(
            aggregate[role].get("evidence_identity"), aggregate[role]["metrics"]
        )
    if aggregate["terminal_bridge"] is not None:
        _validate_metrics(aggregate["terminal_bridge"].get("metrics"))
        _validate_evidence_identity(
            aggregate["terminal_bridge"].get("evidence_identity"),
            aggregate["terminal_bridge"]["metrics"],
        )
    for field in ("best_g1_gain", "sid_marginal", "component_gains"):
        if set(aggregate[field]) != set(METRICS):
            raise ValueError(f"aggregate {field} is incomplete")
        for metric, value in aggregate[field].items():
            _finite(value, metric)
    return document


def _validate_metrics(metrics: object) -> None:
    if not isinstance(metrics, dict) or not set(METRICS) <= set(metrics):
        raise ValueError("report metrics are incomplete")
    for metric in METRICS:
        value = metrics[metric]
        _finite(value, metric)
        if not 0 <= value <= 1:
            raise ValueError(f"{metric} is outside [0, 1]")


def _validate_sid_evidence(row: dict[str, Any]) -> None:
    levels = row.get("semantic_levels")
    metrics = row.get("metrics")
    if levels not in (3, 4) or not isinstance(metrics, dict):
        raise ValueError("treatment semantic levels are not approved")
    for cutoff in SID_CUTOFFS:
        names = [f"sid_exact_recall@{cutoff}"] + [
            f"sid_prefix_recall@{cutoff}_l{level}" for level in range(1, levels + 1)
        ]
        for name in names:
            value = _finite(metrics.get(name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} is outside [0, 1]")
    diagnostics = row.get("sid_diagnostics")
    scalar_names = (
        "identifier_collision_rate",
        "collided_item_fraction",
        "unique_base_tuples",
        "collision_bucket_size_p50",
        "collision_bucket_size_p95",
        "collision_bucket_size_p99",
        "collision_bucket_size_max",
    )
    level_names = (
        "p95_occupied_load",
        "p95_to_mean_occupied_load",
        "occupied_codes",
        "dead_code_fraction",
        "intra_code_cosine_similarity",
        "reconstruction_mse_by_depth",
    )
    if not isinstance(diagnostics, dict) or set(diagnostics) != set(
        (*scalar_names, *level_names)
    ):
        raise ValueError("intrinsic SID diagnostics are incomplete")
    for name in scalar_names:
        _finite(diagnostics[name], name)
    for name in level_names:
        values = diagnostics[name]
        if not isinstance(values, list) or len(values) != levels:
            raise ValueError(f"{name} does not cover every SID depth")
        for value in values:
            _finite(value, name)
    slices = row.get("slice_diagnostics")
    if not isinstance(slices, list) or [item.get("slice") for item in slices] != list(
        SLICE_NAMES
    ):
        raise ValueError("eligible SID slices are incomplete or reordered")
    for item in slices:
        for name in ("control_recall@100", "treatment_recall@100"):
            value = _finite(item.get(name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} is outside [0, 1]")
        for name in ("num_users", "num_targets"):
            value = item.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"slice {name} is invalid")


def _validate_evidence_identity(value: object, metrics: dict[str, float]) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "manifest_logical_sha256",
        "job_logical_sha256",
        "evidence_sha256",
        "path",
        "physical_sha256",
    }:
        raise ValueError("collector evidence identity is incomplete")
    if value["schema"] != "g6-native500m-collected-run/v1" or any(
        not isinstance(value[name], str) or not _SHA256.fullmatch(value[name])
        for name in (
            "manifest_logical_sha256",
            "job_logical_sha256",
            "evidence_sha256",
            "physical_sha256",
        )
    ):
        raise ValueError("collector evidence identity is invalid")
    path_value = value["path"]
    if not isinstance(path_value, str):
        raise ValueError("collector evidence path is invalid")
    path = Path(path_value).resolve(strict=True)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != value["physical_sha256"]:
        raise ValueError("collector evidence physical identity differs")
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("collector evidence is invalid JSON") from error
    for name in (
        "schema",
        "manifest_logical_sha256",
        "job_logical_sha256",
        "evidence_sha256",
    ):
        if document.get(name) != value[name]:
            raise ValueError("collector evidence logical identity differs")
    if document.get("metrics") != metrics:
        raise ValueError("reported metrics differ from collector evidence")


def _one_role(rows: Iterable[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {role}")
    return matches[0]


def _delta(value: float, reference: float, relative_dispersion: float) -> str:
    if value == reference:
        return "—"
    percent = (value / reference - 1) * 100
    rendered = f"{percent:+.3f}%"
    threshold = reference * relative_dispersion
    if value > reference + threshold:
        return f'<span style="color: green">{rendered}</span>'
    if value < reference - threshold:
        return f'<span style="color: red">{rendered}</span>'
    return rendered


def _interaction_label(gap: float, resolution: float) -> str:
    if gap > resolution:
        return "positive"
    if gap < -resolution:
        return "negative"
    return "unresolved"


def _score(value: float) -> str:
    return f"{value:.3f}"


def _points(value: float) -> str:
    return f"{value:+.3f}"


def _raw(value: float | int) -> str:
    return f"{value:.12g}"


def _metric_triplet(
    metrics: dict[str, float], prefix: str, *, level: int | None = None
) -> str:
    suffix = "" if level is None else f"_l{level}"
    return " / ".join(
        _score(metrics[f"{prefix}@{cutoff}{suffix}"]) for cutoff in SID_CUTOFFS
    )


def _level_values(values: list[float]) -> str:
    return " / ".join(_score(value) for value in values)


def _finite(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _finite_positive(value: object, name: str) -> float:
    number = _finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _flatten(values: Iterable[str | tuple[str, ...]]) -> list[str]:
    result = []
    for value in values:
        result.extend(value if isinstance(value, tuple) else (value,))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    evidence = json.loads(arguments.evidence.read_text())
    for path in write_reports(evidence, arguments.output_directory):
        print(path)


if __name__ == "__main__":
    main()
