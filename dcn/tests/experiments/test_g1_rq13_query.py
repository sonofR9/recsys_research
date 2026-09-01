import os
import runpy
import subprocess
from pathlib import Path

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import (
    DEEP_LRS,
    candidate_by_run,
    initial_candidates,
    make_boundary_candidate,
    make_selected_cap_candidates,
    rq13_cap4_candidates,
    rq13_initial_candidates,
    validated_cap4_candidates,
    validated_required_boundary_candidates,
    validated_selected_cap_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq8_reinvestigation_candidates import (
    query_initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact
from utils.global_config import config as global_config


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/g1_sasrec_item_ids_likes"
LAUNCHER = EXPERIMENT / "launchers/architecture/rq13_rq14_query_500m.sh"
BOUNDARY_LAUNCHER = EXPERIMENT / "launchers/architecture/rq13_query_boundary_500m.sh"
CAP4_LAUNCHER = EXPERIMENT / "launchers/architecture/rq13_query_cap4_500m.sh"
SELECTED_CAP_LAUNCHER = (
    EXPERIMENT / "launchers/architecture/rq13_query_selected_cap_500m.sh"
)
CONFIG = EXPERIMENT / "configs/rq13_rq14_query_variant.py"
RQ8_CONFIG = EXPERIMENT / "configs/rq8_reinvestigation_variant.py"


def test_rq13_manifest_is_the_exact_approved_fifteen_run_grid() -> None:
    candidates = rq13_initial_candidates()

    assert len(candidates) == 15
    assert len({candidate.run_name for candidate in candidates}) == 15
    assert {candidate.study for candidate in candidates} == {"rq13"}
    assert {candidate.treatment for candidate in candidates} == {
        "one_example",
        "truncated_8",
        "truncated_16",
        "required_8",
        "required_16",
    }
    assert {candidate.deep_lr for candidate in candidates} == set(DEEP_LRS)
    assert all(
        candidate_by_run(candidate.run_name) == candidate for candidate in candidates
    )


def test_rq13_cap4_screen_is_a_separate_exact_three_run_grid() -> None:
    candidates = rq13_cap4_candidates()

    assert len(candidates) == 3
    assert {candidate.treatment for candidate in candidates} == {"truncated_4"}
    assert {candidate.deep_lr for candidate in candidates} == set(DEEP_LRS)
    assert {candidate.stage for candidate in candidates} == {"cap_anchor"}
    assert all(
        candidate_by_run(candidate.run_name) == candidate for candidate in candidates
    )
    assert not set(candidates).intersection(rq13_initial_candidates())


def test_rq13_selected_cap_grid_and_boundaries_are_canonical() -> None:
    candidates = make_selected_cap_candidates(32)
    boundary = make_boundary_candidate(candidates[-1], "high", 1)

    assert len(candidates) == 3
    assert {candidate.treatment for candidate in candidates} == {"selected_cap_32"}
    assert {candidate.deep_lr for candidate in candidates} == set(DEEP_LRS)
    assert {candidate.stage for candidate in candidates} == {"selected_cap"}
    assert all(
        candidate_by_run(candidate.run_name) == candidate for candidate in candidates
    )
    assert candidate_by_run(boundary.run_name) == boundary


def test_rq13_selected_cap_manifest_is_bound_to_validation_only_fit_evidence() -> None:
    candidates = make_selected_cap_candidates(32)
    evidence = {
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "required_followups": [candidate.run_name for candidate in candidates],
        "cap_fit": {
            "status": "selected_cap_pending",
            "metric": "validation Recall@100 from validation-selected checkpoints",
            "selected_cap": 32,
            "selection_target": {
                "metric": "mean validation Recall@100",
                "control_values": [0.1367, 0.1343, 0.1363],
                "control_mean": sum((0.1367, 0.1343, 0.1363)) / 3,
                "multiplier": 1.10,
                "value": 1.10 * (sum((0.1367, 0.1343, 0.1363)) / 3),
            },
            "reader_success_target": {
                "metric": "mean full-user Recall@100",
                "control_mean": 0.13468336146286186,
                "multiplier": 1.10,
                "value": 1.10 * 0.13468336146286186,
            },
            "input_bindings": {
                "contributing_artifacts": {
                    name: {
                        "artifact_sha256": {
                            "training_metadata.json": "a",
                            "final_metrics.json": "b",
                            "sweep.log": "c",
                        },
                        "source_manifest_sha256": "source",
                    }
                    for name in (
                        "one_example",
                        "truncated_4",
                        "truncated_8",
                        "truncated_16",
                    )
                },
                "eligible_target_counts_sha256": "d" * 64,
                "stage_one_correctness_audit": {
                    "status": "passed",
                    "schema_version": 1,
                    "artifact_sha256": "e" * 64,
                },
            },
            "practical_ceiling": {"selected": 32},
            "target_cap": 37,
        },
    }

    assert (
        validated_selected_cap_candidates(
            evidence, [candidate.run_name for candidate in reversed(candidates)]
        )
        == candidates
    )

    invalid = {
        **evidence,
        "cap_fit": {
            **evidence["cap_fit"],
            "metric": "full-user Recall@100",
        },
    }
    with pytest.raises(ValueError, match="validation"):
        validated_selected_cap_candidates(
            invalid, [candidate.run_name for candidate in candidates]
        )


def test_rq13_boundary_candidates_are_canonical_and_roundtrip() -> None:
    low = next(
        candidate
        for candidate in rq13_initial_candidates()
        if candidate.treatment == "truncated_8" and candidate.deep_lr == min(DEEP_LRS)
    )
    high = next(
        candidate
        for candidate in rq13_initial_candidates()
        if candidate.treatment == "required_16" and candidate.deep_lr == max(DEEP_LRS)
    )

    candidates = (
        make_boundary_candidate(low, "low", 1),
        make_boundary_candidate(low, "low", 2),
        make_boundary_candidate(high, "high", 1),
        make_boundary_candidate(high, "high", 2),
    )

    assert [candidate.deep_lr for candidate in candidates] == [
        0.003,
        0.0015,
        0.048,
        0.096,
    ]
    assert all(candidate.stage == "lr_boundary" for candidate in candidates)
    assert all(
        candidate_by_run(candidate.run_name) == candidate for candidate in candidates
    )
    assert len({candidate.run_name for candidate in candidates}) == len(candidates)


def test_rq13_launcher_inspection_lists_no_later_rq() -> None:
    result = subprocess.run(
        [str(LAUNCHER), "--list-initial"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        candidate.run_name for candidate in rq13_initial_candidates()
    ]
    assert "rq14" not in result.stdout


def test_rq13_cap4_launcher_lists_exact_screen() -> None:
    result = subprocess.run(
        [str(CAP4_LAUNCHER), "--list"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        candidate.run_name for candidate in rq13_cap4_candidates()
    ]


def test_rq13_cap4_stage_requires_resolved_audited_original_surface() -> None:
    candidates = rq13_cap4_candidates()
    names = [candidate.run_name for candidate in candidates]
    evidence = {
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "missing_initial_artifacts": names,
        "required_followups": names,
        "required_boundary_followups": [],
        "cap_fit": {"status": "pending_cap4"},
        "correctness_audit": {"status": "passed"},
        "surface_winners": {
            treatment: {}
            for treatment in {
                "one_example",
                "truncated_8",
                "truncated_16",
                "required_8",
                "required_16",
            }
        },
    }

    assert validated_cap4_candidates(evidence, names) == candidates
    with pytest.raises(ValueError, match="audit"):
        validated_cap4_candidates(
            {**evidence, "correctness_audit": {"status": "failed"}}, names
        )


def test_rq13_launcher_submits_only_the_rq13_manifest() -> None:
    launcher = LAUNCHER.read_text()

    assert "rq13_initial_candidates" in launcher
    assert "exactly 15 runs" in launcher
    assert "utils/training_queue/queue.sh" in launcher
    assert 'enqueue "$run"' in launcher
    assert "g1_require_config_compatible_or_absent" in launcher
    assert "g1_require_config_recipe_compatible_or_absent" not in launcher

    cap4_launcher = CAP4_LAUNCHER.read_text()
    assert "rq13_prefix_expansion_report" in cap4_launcher
    assert "validated_cap4_candidates" in cap4_launcher


def test_rq13_assignment_is_accepted_by_the_artifact_verifier() -> None:
    candidate = rq13_initial_candidates()[0]

    assert verify_artifact._config_assignments(
        [f"G1_QUERY_RUN={candidate.run_name}"]
    ) == {"G1_QUERY_RUN": candidate.run_name}


def test_rq13_artifact_verifier_initializes_the_config_base_path(monkeypatch) -> None:
    candidate = rq13_initial_candidates()[0]
    monkeypatch.setattr(global_config, "_base_path", None)

    experiment = verify_artifact._config_experiment(
        CONFIG, {"G1_QUERY_RUN": candidate.run_name}
    )

    assert global_config.base_path == Path(experiment.base_path)


def test_rq13_boundary_manifest_accepts_only_the_exact_current_followups() -> None:
    initial = rq13_initial_candidates()[0]
    candidates = (
        make_boundary_candidate(initial, "low", 1),
        make_boundary_candidate(initial, "low", 2),
    )

    evidence = {
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "missing_initial_artifacts": [],
        "required_boundary_followups": [item.run_name for item in candidates],
        "required_followups": [item.run_name for item in candidates],
    }

    assert (
        validated_required_boundary_candidates(
            evidence, [item.run_name for item in reversed(candidates)]
        )
        == candidates
    )


def test_rq13_boundary_manifest_rejects_missing_stale_extra_and_duplicate_runs() -> (
    None
):
    initial = rq13_initial_candidates()[0]
    candidate = make_boundary_candidate(initial, "low", 1)
    current = {
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "missing_initial_artifacts": [],
        "required_boundary_followups": [candidate.run_name],
        "required_followups": [candidate.run_name],
    }

    invalid = (
        (
            {**current, "missing_initial_artifacts": [initial.run_name]},
            [candidate.run_name],
        ),
        (current, []),
        (
            current,
            [candidate.run_name, make_boundary_candidate(initial, "low", 2).run_name],
        ),
        (current, [candidate.run_name, candidate.run_name]),
        ({**current, "required_boundary_followups": []}, [candidate.run_name]),
    )
    for evidence, requested in invalid:
        try:
            validated_required_boundary_candidates(evidence, requested)
        except ValueError:
            continue
        raise AssertionError("invalid RQ13 boundary manifest was accepted")


def test_rq13_boundary_candidate_reconstructs_the_exact_training_config(
    monkeypatch,
) -> None:
    initial = next(
        candidate
        for candidate in rq13_initial_candidates()
        if candidate.treatment == "truncated_16" and candidate.deep_lr == 0.024
    )
    candidate = make_boundary_candidate(initial, "high", 1)
    monkeypatch.setenv("G1_DATASET_SIZE", "500m")
    monkeypatch.setenv("G1_MAX_EPOCHS", "20")
    monkeypatch.setenv("G1_VARIANT", "baseline")
    monkeypatch.setenv("G1_QUERY_RUN", candidate.run_name)

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.run_name == candidate.run_name
    assert experiment.deep_learning_rate == 0.048
    assert experiment.query_architecture == "encoder_decoder"
    assert experiment.prefix_length_rule == "truncated"
    assert experiment.prefix_cap == 16


def test_rq13_selected_cap_reconstructs_exact_truncated_training_config(
    monkeypatch,
) -> None:
    candidate = make_selected_cap_candidates(32)[1]
    monkeypatch.setenv("G1_QUERY_RUN", candidate.run_name)

    experiment = runpy.run_path(str(CONFIG))["experiment"]

    assert experiment.run_name == candidate.run_name
    assert experiment.deep_learning_rate == 0.012
    assert experiment.prefix_length_rule == "truncated"
    assert experiment.prefix_cap == 32


def test_rq13_selected_cap_launcher_regenerates_and_validates_fit_evidence() -> None:
    launcher = SELECTED_CAP_LAUNCHER.read_text()

    assert "rq13_prefix_expansion_report" in launcher
    assert "rq13_prefix_expansion_audit" in launcher
    assert "stage_one_audit_required" in launcher
    assert "validate_bound_stage_one_audit" in launcher
    assert launcher.count("rq13_prefix_expansion_report") == 2
    assert "validated_selected_cap_candidates" in launcher
    assert "utils/training_queue/queue.sh" in launcher
    assert "g1_require_config_compatible_or_absent" in launcher


def _fake_selected_cap_python(tmp_path: Path) -> Path:
    binary = tmp_path / "bin/python"
    binary.parent.mkdir()
    binary.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

scenario = os.environ["FAKE_RQ13_SCENARIO"]
evidence = Path(os.environ["FAKE_RQ13_EVIDENCE"])
calls = Path(os.environ["FAKE_RQ13_CALLS"])
arguments = sys.argv[1:]
source = sys.stdin.read() if arguments and arguments[0] == "-" else ""
with calls.open("a") as stream:
    if any("rq13_prefix_expansion_report" in argument for argument in arguments):
        label = "report"
    elif any("rq13_prefix_expansion_audit" in argument for argument in arguments):
        label = "audit"
    elif "validate_bound_stage_one_audit" in source:
        label = "validate"
    else:
        label = "stage"
    stream.write(label + "\\n")

if label == "report":
    report_count = calls.read_text().splitlines().count("report")
    if scenario == "wrong_stage":
        status = "pending_cap4"
    elif scenario == "failed_rebind" or report_count == 1:
        status = "stage_one_audit_required"
    else:
        status = "selected_cap_pending"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({"cap_fit": {"status": status}}))
elif label == "audit":
    Path(os.environ["FAKE_RQ13_AUDIT"]).write_text("{}")
elif label == "stage":
    status = json.loads(evidence.read_text())["cap_fit"]["status"]
    if status not in {"stage_one_audit_required", "selected_cap_pending"}:
        raise SystemExit(2)
    print(status)
else:
    if scenario in {"missing_audit", "stale_binding", "failed_rebind"}:
        raise SystemExit(2)
    print(os.environ["FAKE_RQ13_CANDIDATES"].replace(",", "\\n"))
"""
    )
    binary.chmod(0o755)
    return binary


@pytest.mark.parametrize(
    "scenario",
    ["wrong_stage", "missing_audit", "stale_binding", "failed_rebind"],
)
def test_rq13_selected_cap_launcher_fails_closed_before_listing_candidates(
    tmp_path: Path, scenario: str
) -> None:
    binary = _fake_selected_cap_python(tmp_path)
    evidence = tmp_path / "evidence/rq13_prefix_expansion_results.json"
    audit = tmp_path / "evidence/rq13_prefix_expansion_correctness.json"
    calls = tmp_path / "calls"
    environment = {
        **os.environ,
        "PATH": f"{binary.parent}:{os.environ['PATH']}",
        "G1_QUERY_EVIDENCE_DIR": str(evidence.parent),
        "G1_QUERY_SCRATCHPAD_DIR": str(tmp_path / "scratchpad"),
        "FAKE_RQ13_SCENARIO": scenario,
        "FAKE_RQ13_EVIDENCE": str(evidence),
        "FAKE_RQ13_AUDIT": str(audit),
        "FAKE_RQ13_CALLS": str(calls),
        "FAKE_RQ13_CANDIDATES": ",".join(
            candidate.run_name for candidate in make_selected_cap_candidates(32)
        ),
    }

    result = subprocess.run(
        [str(SELECTED_CAP_LAUNCHER), "--list"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert result.stdout == ""


def test_rq13_selected_cap_launcher_lists_only_after_fresh_audit_rebinding(
    tmp_path: Path,
) -> None:
    binary = _fake_selected_cap_python(tmp_path)
    evidence = tmp_path / "evidence/rq13_prefix_expansion_results.json"
    calls = tmp_path / "calls"
    candidates = make_selected_cap_candidates(32)
    environment = {
        **os.environ,
        "PATH": f"{binary.parent}:{os.environ['PATH']}",
        "G1_QUERY_EVIDENCE_DIR": str(evidence.parent),
        "G1_QUERY_SCRATCHPAD_DIR": str(tmp_path / "scratchpad"),
        "FAKE_RQ13_SCENARIO": "success",
        "FAKE_RQ13_EVIDENCE": str(evidence),
        "FAKE_RQ13_AUDIT": str(
            tmp_path / "evidence/rq13_prefix_expansion_correctness.json"
        ),
        "FAKE_RQ13_CALLS": str(calls),
        "FAKE_RQ13_CANDIDATES": ",".join(
            candidate.run_name for candidate in candidates
        ),
    }

    result = subprocess.run(
        [str(SELECTED_CAP_LAUNCHER), "--list"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        candidate.run_name for candidate in candidates
    ]
    assert calls.read_text().splitlines() == [
        "report",
        "stage",
        "audit",
        "report",
        "validate",
    ]


def test_rq13_keeps_the_frozen_rq8_scoring_and_evaluator_surface(
    monkeypatch,
) -> None:
    rq8_candidate = next(
        candidate
        for candidate in query_initial_candidates()
        if candidate.query_method == "standard" and candidate.deep_lr == 0.006
    )
    rq13_candidate = next(
        candidate
        for candidate in rq13_initial_candidates()
        if candidate.treatment == "one_example" and candidate.deep_lr == 0.006
    )
    monkeypatch.setenv("G1_DATASET_SIZE", "500m")
    monkeypatch.setenv("G1_MAX_EPOCHS", "20")
    monkeypatch.setenv("G1_VARIANT", "baseline")
    monkeypatch.setenv("G1_RQ8_RUN", rq8_candidate.run_name)
    rq8 = runpy.run_path(str(RQ8_CONFIG))["experiment"]
    monkeypatch.setenv("G1_QUERY_RUN", rq13_candidate.run_name)
    rq13 = runpy.run_path(str(CONFIG))["experiment"]

    fields = (
        "negative_sampling",
        "num_in_batch_negatives",
        "logq_correction",
        "random_negative_fraction",
        "logq_alpha",
        "correct_positive_logq",
        "mask_false_negatives",
        "exclude_own_group_negatives",
        "dense_random_negative_scores",
        "eval_ks",
        "eval_max_users",
        "eval_every_n_epochs",
        "selection_k",
        "evaluation_catalog",
        "exclude_seen_from_evaluation",
        "restore_best_weights",
    )
    assert {name: getattr(rq13, name) for name in fields} == {
        name: getattr(rq8, name) for name in fields
    }
    assert (
        rq13.checkpointing.best_metric_name,
        rq13.checkpointing.best_metric_prefix,
        rq13.checkpointing.best_metric_mode,
    ) == (
        rq8.checkpointing.best_metric_name,
        rq8.checkpointing.best_metric_prefix,
        rq8.checkpointing.best_metric_mode,
    )


def test_rq13_boundary_launcher_rejects_initial_and_later_rq_candidates(
    tmp_path: Path,
) -> None:
    rq13_initial = rq13_initial_candidates()[0]
    rq14_initial = next(
        candidate for candidate in initial_candidates() if candidate.study == "rq14"
    )

    for candidate in (rq13_initial, make_boundary_candidate(rq14_initial, "low", 1)):
        environment = os.environ.copy()
        environment["G1_QUERY_LOGS"] = str(tmp_path / "logs")
        environment["G1_QUERY_EVIDENCE_DIR"] = str(tmp_path / "evidence")
        environment["G1_QUERY_SCRATCHPAD_DIR"] = str(tmp_path / "scratchpad")
        result = subprocess.run(
            [str(BOUNDARY_LAUNCHER), "--list", candidate.run_name],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        assert (tmp_path / "scratchpad/rq13_prefix_expansion_tuning_500m.md").is_file()


def test_rq13_boundary_launcher_regenerates_evidence_before_queueing() -> None:
    launcher = BOUNDARY_LAUNCHER.read_text()

    assert "rq13_prefix_expansion_report" in launcher
    assert "validated_required_boundary_candidates" in launcher
    assert "utils/training_queue/queue.sh" in launcher
    assert "g1_require_config_compatible_or_absent" in launcher
    assert "g1_require_config_recipe_compatible_or_absent" not in launcher
