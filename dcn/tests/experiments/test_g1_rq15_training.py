import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from dcn.config.query_retrieval_training import (
    MuTransferRq15CrossAttentionGenerationExperiment,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import (
    Rq15Candidate,
    candidate_by_run,
    candidate_followup_record,
    initial_candidates,
    launch_initial_candidates,
    make_auxiliary_weight_candidate,
    make_boundary_candidate,
    selected_source_candidate,
    selected_source_checkpoint,
    source_candidate_by_run,
    source_candidates,
    validated_required_followup_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


EXPERIMENT = Path("experiments/g1_sasrec_item_ids_likes")
CONFIG = EXPERIMENT / "configs/rq15_decoder_training_variant.py"
SOURCE_CONFIG = EXPERIMENT / "configs/rq15_rq8_checkpoint_variant.py"
LAUNCHER = EXPERIMENT / "launchers/architecture/rq15_decoder_training_500m.sh"
SOURCE_LAUNCHER = EXPERIMENT / "launchers/architecture/rq15_rq8_checkpoint_500m.sh"
FOLLOWUP_LAUNCHER = (
    EXPERIMENT / "launchers/architecture/rq15_decoder_training_followups_500m.sh"
)


def _followup_evidence(candidates: list[Rq15Candidate]) -> dict[str, object]:
    stage = candidates[0].stage
    return {
        "schema_version": 1,
        "research_question": "RQ15 decoder-decoder training method",
        "dataset_size": "500m",
        "claims_status": (
            "pending_boundary"
            if stage == "lr_boundary"
            else "pending_auxiliary_weights"
        ),
        "result_claims_user_validated": False,
        "missing_artifacts": [],
        "required_followups": [
            candidate_followup_record(candidate) for candidate in candidates
        ],
        "scratch_control": {"run_name": "scratch"},
        "checkpoint_pretraining": {"run_name": "checkpoint"},
        "treatments": {},
        "artifact_audit": {"status": "passed"},
    }


def _stub_queue(tmp_path: Path) -> tuple[Path, Path]:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "[[ \"${G1_DATASET_SIZE:-}\" == 500m ]] || return 91\n"
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; }\n"
        "drain() { return 0; }\n"
    )
    artifacts = tmp_path / "artifacts.sh"
    artifacts.write_text(
        "g1_require_config_compatible_or_absent() { return ${STUB_ARTIFACT_STATUS:-1}; }\n"
        "g1_verify_config_recipe_artifact() { return 0; }\n"
        "g1_stop_artifact_verifier() { return 0; }\n"
    )
    return queue, artifacts


def _stub_report_environment(tmp_path: Path, evidence: Path) -> dict[str, str]:
    (tmp_path / "sitecustomize.py").write_text(
        "import json, os, sys, types\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "name = 'experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_report'\n"
        "module = types.ModuleType(name)\n"
        "def collect_report_bundle(*args, **kwargs):\n"
        "    print('INFO report collection diagnostic')\n"
        "    path = Path(os.environ['STUB_RQ15_AUTHORITATIVE_EVIDENCE'])\n"
        "    return SimpleNamespace(evidence=json.loads(path.read_text()))\n"
        "module.collect_report_bundle = collect_report_bundle\n"
        "sys.modules[name] = module\n"
    )
    return {
        "PYTHONPATH": str(tmp_path)
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
        "STUB_RQ15_AUTHORITATIVE_EVIDENCE": str(evidence),
    }


def _write_source_artifacts(logs: Path) -> None:
    for candidate in source_candidates():
        directory = logs / candidate.run_name
        directory.mkdir(parents=True)
        recall = 0.2 if candidate.deep_lr in {0.024, 0.048} else 0.19
        ndcg = 0.08 if candidate.deep_lr in {0.024, 0.048} else 0.09
        epoch_time = 1.0 if candidate.deep_lr == 0.048 else 2.0
        metadata = {
            "embedding_learning_rate": candidate.embedding_lr,
            "deep_learning_rate": candidate.deep_lr,
            "best_epoch": 20,
            "stopped_epoch": 20,
            "lr_horizon_complete": True,
            "selection_resolved": True,
        }
        (directory / "training_metadata.json").write_text(json.dumps(metadata))
        (directory / "final_metrics.json").write_text(
            json.dumps({"recall@100": recall, "ndcg@100": ndcg})
        )
        (directory / candidate.checkpoint_name).touch()
        lines = []
        for epoch in range(20):
            epoch_recall = recall - (19 - epoch) * 0.001
            epoch_ndcg = ndcg - (19 - epoch) * 0.001
            lines.append(
                f"epoch {epoch} finished "
                f"timing.train_epoch_time={epoch_time:.4f} "
                "timing.val_inference_time=0.2500 timing.val_save_time=0.0100 "
                f"epoch/val_true.ndcg@100={epoch_ndcg:.4f} "
                f"epoch/val_true.recall@100={epoch_recall:.4f}"
            )
        (directory / "sweep.log").write_text("\n".join(lines))


def _experiment(candidate: Rq15Candidate):
    previous = os.environ.get("G1_RQ15_RUN")
    previous_source = os.environ.get("G1_RQ15_SOURCE_RUN")
    try:
        os.environ["G1_RQ15_RUN"] = candidate.run_name
        if candidate.training_method == "pretrained_finetune":
            os.environ["G1_RQ15_SOURCE_RUN"] = source_candidates()[1].run_name
        return runpy.run_path(str(CONFIG))["experiment"]
    finally:
        if previous is None:
            os.environ.pop("G1_RQ15_RUN", None)
        else:
            os.environ["G1_RQ15_RUN"] = previous
        if previous_source is None:
            os.environ.pop("G1_RQ15_SOURCE_RUN", None)
        else:
            os.environ["G1_RQ15_SOURCE_RUN"] = previous_source


def test_initial_surface_exposes_both_optimizer_rates() -> None:
    candidates = initial_candidates()

    assert len(candidates) == 27
    assert {candidate.training_method for candidate in candidates} == {
        "scratch_candidate_only",
        "pretrained_finetune",
        "auxiliary_ntp",
    }
    assert {candidate.embedding_lr for candidate in candidates} == {
        0.032,
        0.064,
        0.128,
    }
    for method in ("scratch_candidate_only", "pretrained_finetune"):
        assert {
            candidate.deep_lr
            for candidate in candidates
            if candidate.training_method == method
        } == {0.00075, 0.0015, 0.003}
    assert {
        candidate.deep_lr
        for candidate in candidates
        if candidate.training_method == "auxiliary_ntp"
    } == {0.003, 0.012, 0.048}
    assert len({candidate.run_name for candidate in candidates}) == 27


def test_initial_launch_reuses_only_the_historical_scratch_center() -> None:
    launched = launch_initial_candidates()

    assert len(launched) == 26
    assert not any(
        candidate.training_method == "scratch_candidate_only"
        and candidate.embedding_lr == 0.064
        and candidate.deep_lr == 0.0015
        for candidate in launched
    )
    assert set(launched).issubset(initial_candidates())


def test_source_surface_has_three_deep_rates_and_canonical_recipes() -> None:
    candidates = source_candidates()

    assert len(candidates) == 3
    assert {candidate.embedding_lr for candidate in candidates} == {0.064}
    assert {candidate.deep_lr for candidate in candidates} == {0.024, 0.048, 0.096}
    assert all(source_candidate_by_run(candidate.run_name) == candidate for candidate in candidates)
    assert all(candidate.checkpoint_name == "rq15_first_stage_checkpoint.pt" for candidate in candidates)


def test_source_selection_uses_validation_metrics_then_full_horizon_time(
    tmp_path: Path,
) -> None:
    _write_source_artifacts(tmp_path)

    selected = selected_source_candidate(tmp_path)

    assert selected.deep_lr == 0.048
    assert selected_source_checkpoint(tmp_path) == selected.checkpoint_path(tmp_path)


@pytest.mark.parametrize("candidate", initial_candidates())
def test_rq15_recipe_is_distinct_cls_only(candidate: Rq15Candidate) -> None:
    experiment = _experiment(candidate)

    assert isinstance(experiment, MuTransferRq15CrossAttentionGenerationExperiment)
    assert experiment.size == "500m"
    assert experiment.query_architecture == "decoder_decoder"
    assert experiment.query_slots_shared is False
    assert experiment.include_history_memory is False
    assert experiment.num_query_slots == 4
    assert experiment.window == "bounded_prefix"
    assert experiment.prefix_cap == 1
    assert experiment.max_seq_len == 128
    assert experiment.embedding_learning_rate == candidate.embedding_lr
    assert experiment.deep_learning_rate == candidate.deep_lr
    assert experiment.training_method == candidate.training_method
    assert experiment.auxiliary_ntp_weight == (
        1.0 if candidate.training_method == "auxiliary_ntp" else 0.0
    )


def test_verifier_accepts_only_the_rq15_launcher_assignments(tmp_path: Path) -> None:
    candidate = next(
        item
        for item in initial_candidates()
        if item.training_method == "pretrained_finetune"
    )
    checkpoint = tmp_path / "first-stage.pt"
    assignments = verify_artifact._config_assignments(
        [
            f"G1_RQ15_RUN={candidate.run_name}",
            f"G1_RQ15_SOURCE_RUN={source_candidates()[1].run_name}",
            f"G1_RQ15_FIRST_STAGE_CHECKPOINT={checkpoint}",
        ]
    )

    assert assignments == {
        "G1_RQ15_RUN": candidate.run_name,
        "G1_RQ15_SOURCE_RUN": source_candidates()[1].run_name,
        "G1_RQ15_FIRST_STAGE_CHECKPOINT": str(checkpoint),
    }
    with pytest.raises(ValueError, match="unsupported config-verifier assignment"):
        verify_artifact._config_assignments(["G1_RQ15_STAGE=initial"])


def test_verifier_reconstructs_rq15_checkpoint_candidate(tmp_path: Path) -> None:
    candidate = next(
        item
        for item in initial_candidates()
        if item.training_method == "pretrained_finetune"
    )
    checkpoint = tmp_path / "first-stage.pt"
    assignments = verify_artifact._config_assignments(
        [
            f"G1_RQ15_RUN={candidate.run_name}",
            f"G1_RQ15_SOURCE_RUN={source_candidates()[1].run_name}",
            f"G1_RQ15_FIRST_STAGE_CHECKPOINT={checkpoint}",
        ]
    )

    experiment = verify_artifact._config_experiment(CONFIG, assignments)

    assert experiment.run_name == candidate.run_name
    assert experiment.first_stage_checkpoint == checkpoint
    assert experiment.training_method == "pretrained_finetune"


def test_source_recipe_is_exact_selected_rq8_standard_ntp() -> None:
    previous = os.environ.get("G1_RQ15_SOURCE_RUN")
    try:
        for candidate in source_candidates():
            os.environ["G1_RQ15_SOURCE_RUN"] = candidate.run_name
            experiment = runpy.run_path(str(SOURCE_CONFIG))["experiment"]

            assert experiment.size == "500m"
            assert experiment.max_seq_len == 128
            assert experiment.effective_cls_token_mode == "none"
            assert experiment.embedding_learning_rate == candidate.embedding_lr
            assert experiment.deep_learning_rate == candidate.deep_lr
            assert experiment.dataloader.effective_batch_size == 1280
            assert experiment.num_epochs == 20
            assert experiment.run_name == candidate.run_name
            assert (
                experiment.checkpoint_export_metadata["source_recipe_run_name"]
                == candidate.source_recipe_run_name
            )
    finally:
        if previous is None:
            os.environ.pop("G1_RQ15_SOURCE_RUN", None)
        else:
            os.environ["G1_RQ15_SOURCE_RUN"] = previous


def test_auxiliary_weight_followup_is_consumed_by_the_existing_recipe() -> None:
    anchor = next(
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "auxiliary_ntp"
        and candidate.deep_lr == 0.012
    )
    candidate = make_auxiliary_weight_candidate(anchor, 0.3)

    experiment = _experiment(candidate)

    assert experiment.run_name == candidate.run_name
    assert experiment.deep_learning_rate == 0.012
    assert experiment.auxiliary_ntp_weight == 0.3


def test_deep_boundary_uses_each_methods_initial_grid_ratio() -> None:
    scratch = next(
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "scratch_candidate_only"
        and candidate.deep_lr == 0.003
    )
    auxiliary = next(
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "auxiliary_ntp"
        and candidate.deep_lr == 0.048
    )

    assert make_boundary_candidate(scratch, "deep", "high", 1).deep_lr == 0.006
    assert make_boundary_candidate(auxiliary, "deep", "high", 1).deep_lr == 0.192


def test_pretrained_low_embedding_boundary_ends_at_frozen_embedding() -> None:
    anchors = [
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "pretrained_finetune"
        and candidate.embedding_lr == 0.032
    ]

    candidates = [
        make_boundary_candidate(anchor, "embedding", "low", 8) for anchor in anchors
    ]

    assert {candidate.embedding_lr for candidate in candidates} == {0.0}
    assert {candidate.deep_lr for candidate in candidates} == {
        0.00075,
        0.0015,
        0.003,
    }
    assert all(candidate.boundary_step == 8 for candidate in candidates)
    assert all(
        candidate_by_run(candidate.run_name) == candidate for candidate in candidates
    )
    assert all("_e0_" in candidate.run_name for candidate in candidates)
    assert all(
        _experiment(candidate).embedding_learning_rate == 0.0
        for candidate in candidates
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"training_method": "scratch_candidate_only"},
        {"embedding_lr": -0.0},
        {
            "stage": "initial",
            "boundary_axis": None,
            "boundary_direction": None,
            "boundary_step": None,
        },
        {"boundary_direction": "high"},
        {"boundary_step": 7},
    ],
)
def test_zero_embedding_lr_is_only_the_exact_pretrained_terminal_boundary(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "training_method": "pretrained_finetune",
        "embedding_lr": 0.0,
        "deep_lr": 0.0015,
        "stage": "lr_boundary",
        "boundary_axis": "embedding",
        "boundary_direction": "low",
        "boundary_step": 8,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="zero embedding LR"):
        Rq15Candidate(**values)


def test_frozen_embedding_deep_probe_and_terminal_continuation() -> None:
    deep_probe = Rq15Candidate(
        "pretrained_finetune",
        embedding_lr=0.0,
        deep_lr=0.000375,
        stage="lr_boundary",
        boundary_axis="deep",
        boundary_direction="low",
        boundary_step=1,
    )

    assert candidate_by_run(deep_probe.run_name) == deep_probe
    anchor = next(
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "pretrained_finetune"
        and candidate.embedding_lr == 0.032
    )
    with pytest.raises(ValueError, match="terminal"):
        make_boundary_candidate(anchor, "embedding", "low", 9)
    for foreign_embedding_lr in (0.064, 0.128):
        foreign_anchor = next(
            candidate
            for candidate in initial_candidates()
            if candidate.training_method == "pretrained_finetune"
            and candidate.embedding_lr == foreign_embedding_lr
            and candidate.deep_lr == 0.0015
        )
        with pytest.raises(ValueError, match="low-edge anchor"):
            make_boundary_candidate(foreign_anchor, "embedding", "low", 8)


def test_launchers_are_queue_only_and_do_not_submit_on_source_import() -> None:
    launcher = LAUNCHER.read_text()
    source = SOURCE_LAUNCHER.read_text()

    assert "utils/training_queue/queue.sh" in launcher
    assert "utils/training_queue/queue.sh" in source
    assert "selected_source_candidate" in launcher
    assert "launch_initial_candidates" in launcher
    assert "source_candidates" in source
    assert "python -m dcn.main" not in launcher
    assert "python -m dcn.main" not in source


def test_launcher_enqueues_scratch_candidates_without_checkpoint(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "[[ \"${G1_DATASET_SIZE:-}\" == 500m ]] || return 91\n"
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; }\n"
        "drain() { return 0; }\n"
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_RQ15_LOGS": str(tmp_path / "logs"),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_RQ15_FIRST_STAGE_CHECKPOINT": str(tmp_path / "missing.pt"),
        },
    )

    enqueued = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    assert result.returncode == 2
    assert len(enqueued) == 17
    assert sum("scratch_candidate_only" in line for line in enqueued) == 8
    assert sum("auxiliary_ntp" in line for line in enqueued) == 9
    assert all("G1_RQ15_FIRST_STAGE_CHECKPOINT" not in line for line in enqueued)
    assert result.stderr.count("skipped missing first-stage checkpoint") == 9


def test_launcher_supplies_checkpoint_only_to_pretrained_candidates(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "[[ \"${G1_DATASET_SIZE:-}\" == 500m ]] || return 91\n"
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; }\n"
        "drain() { return 0; }\n"
    )
    logs = tmp_path / "logs"
    _write_source_artifacts(logs)
    selected = selected_source_candidate(logs)
    checkpoint = selected.checkpoint_path(logs)
    artifacts = tmp_path / "artifacts.sh"
    artifacts.write_text(
        "g1_verify_config_recipe_artifact() { return 0; }\n"
        "g1_require_config_recipe_compatible_or_absent() { return 1; }\n"
        "g1_stop_artifact_verifier() { return 0; }\n"
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_RQ15_LOGS": str(logs),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_RQ15_ARTIFACTS_LIBRARY": str(artifacts),
            "G1_RQ15_FIRST_STAGE_CHECKPOINT": str(checkpoint),
            "G1_RQ15_SOURCE_RUN": selected.run_name,
        },
    )

    enqueued = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    pretrained = [line for line in enqueued if "pretrained_finetune" in line]
    auxiliary = [line for line in enqueued if "auxiliary_ntp" in line]
    assert result.returncode == 0
    scratch = [line for line in enqueued if "scratch_candidate_only" in line]
    assert len(pretrained) == 9
    assert len(auxiliary) == 9
    assert len(scratch) == 8
    assert all("G1_RQ15_FIRST_STAGE_CHECKPOINT" in line for line in pretrained)
    assert all("G1_RQ15_FIRST_STAGE_CHECKPOINT" not in line for line in auxiliary)


def test_launcher_can_enqueue_only_pretrained_candidates(tmp_path: Path) -> None:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "[[ \"${G1_DATASET_SIZE:-}\" == 500m ]] || return 91\n"
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; }\n"
        "drain() { return 0; }\n"
    )
    logs = tmp_path / "logs"
    _write_source_artifacts(logs)
    selected = selected_source_candidate(logs)
    checkpoint = selected.checkpoint_path(logs)
    artifacts = tmp_path / "artifacts.sh"
    artifacts.write_text(
        "g1_verify_config_recipe_artifact() { return 0; }\n"
        "g1_require_config_recipe_compatible_or_absent() { return 1; }\n"
        "g1_stop_artifact_verifier() { return 0; }\n"
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--pretrained-only"],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_RQ15_LOGS": str(logs),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_RQ15_ARTIFACTS_LIBRARY": str(artifacts),
            "G1_RQ15_FIRST_STAGE_CHECKPOINT": str(checkpoint),
            "G1_RQ15_SOURCE_RUN": selected.run_name,
        },
    )

    enqueued = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    assert result.returncode == 0
    assert len(enqueued) == 9
    assert all("pretrained_finetune" in line for line in enqueued)
    assert all("G1_RQ15_FIRST_STAGE_CHECKPOINT" in line for line in enqueued)


def test_launcher_rejects_recipe_incompatible_source_before_pretrained(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    _write_source_artifacts(logs)
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "[[ \"${G1_DATASET_SIZE:-}\" == 500m ]] || return 91\n"
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; }\n"
        "drain() { return 0; }\n"
    )
    artifacts = tmp_path / "artifacts.sh"
    artifacts.write_text(
        "g1_verify_config_recipe_artifact() { return 1; }\n"
        "g1_require_config_recipe_compatible_or_absent() { return 1; }\n"
        "g1_stop_artifact_verifier() { return 0; }\n"
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_RQ15_LOGS": str(logs),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_RQ15_ARTIFACTS_LIBRARY": str(artifacts),
        },
    )

    enqueued = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    assert result.returncode == 2
    assert len(enqueued) == 17
    assert not any("pretrained_finetune" in line for line in enqueued)
    assert result.stderr.count("skipped missing first-stage checkpoint") == 9


def test_followup_launcher_enqueues_only_evidence_auxiliary_candidates(
    tmp_path: Path,
) -> None:
    anchor = next(
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "auxiliary_ntp"
        and candidate.deep_lr == 0.012
    )
    candidates = [
        make_auxiliary_weight_candidate(anchor, weight) for weight in (0.1, 0.3)
    ]
    evidence = tmp_path / "results.json"
    evidence.write_text(json.dumps(_followup_evidence(candidates)))
    queue, artifacts = _stub_queue(tmp_path)

    result = subprocess.run(
        ["bash", str(FOLLOWUP_LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ
        | _stub_report_environment(tmp_path, evidence)
        | {
            "G1_RQ15_EVIDENCE": str(evidence),
            "G1_RQ15_LOGS": str(tmp_path / "logs"),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_RQ15_ARTIFACTS_LIBRARY": str(artifacts),
        },
    )

    enqueued = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    assert result.returncode == 0
    assert len(enqueued) == 2
    assert all("auxiliary_ntp" in line and "G1_RQ15_RUN=" in line for line in enqueued)
    assert all("G1_RQ15_FIRST_STAGE_CHECKPOINT" not in line for line in enqueued)


def test_followup_launcher_list_is_not_contaminated_by_report_stdout(
    tmp_path: Path,
) -> None:
    anchor = next(
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "auxiliary_ntp"
        and candidate.deep_lr == 0.012
    )
    candidate = make_auxiliary_weight_candidate(anchor, 0.1)
    evidence = tmp_path / "results.json"
    evidence.write_text(json.dumps(_followup_evidence([candidate])))

    result = subprocess.run(
        ["bash", str(FOLLOWUP_LAUNCHER), "--list"],
        capture_output=True,
        text=True,
        env=os.environ
        | _stub_report_environment(tmp_path, evidence)
        | {
            "G1_RQ15_EVIDENCE": str(evidence),
            "G1_RQ15_LOGS": str(tmp_path / "logs"),
        },
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [candidate.run_name]
    assert "INFO report collection diagnostic" in result.stderr


def test_followup_launcher_supplies_checkpoint_only_to_pretrained_boundary(
    tmp_path: Path,
) -> None:
    anchor = next(
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "pretrained_finetune"
        and candidate.deep_lr == 0.003
    )
    candidate = make_boundary_candidate(anchor, "deep", "high", 1)
    evidence = tmp_path / "results.json"
    evidence.write_text(json.dumps(_followup_evidence([candidate])))
    queue, artifacts = _stub_queue(tmp_path)
    logs = tmp_path / "logs"
    _write_source_artifacts(logs)
    selected = selected_source_candidate(logs)
    checkpoint = selected.checkpoint_path(logs)

    result = subprocess.run(
        ["bash", str(FOLLOWUP_LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ
        | _stub_report_environment(tmp_path, evidence)
        | {
            "G1_RQ15_EVIDENCE": str(evidence),
            "G1_RQ15_LOGS": str(logs),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_RQ15_ARTIFACTS_LIBRARY": str(artifacts),
            "G1_RQ15_FIRST_STAGE_CHECKPOINT": str(checkpoint),
            "G1_RQ15_SOURCE_RUN": selected.run_name,
        },
    )

    enqueued = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    assert result.returncode == 0
    assert len(enqueued) == 1
    assert candidate.run_name in enqueued[0]
    assert f"G1_RQ15_FIRST_STAGE_CHECKPOINT={checkpoint}" in enqueued[0]


def test_followup_launcher_skips_artifact_verified_candidate(tmp_path: Path) -> None:
    anchor = next(
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "auxiliary_ntp"
        and candidate.deep_lr == 0.012
    )
    candidate = make_auxiliary_weight_candidate(anchor, 0.1)
    evidence = tmp_path / "results.json"
    evidence.write_text(json.dumps(_followup_evidence([candidate])))
    queue, artifacts = _stub_queue(tmp_path)

    result = subprocess.run(
        ["bash", str(FOLLOWUP_LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ
        | _stub_report_environment(tmp_path, evidence)
        | {
            "G1_RQ15_EVIDENCE": str(evidence),
            "G1_RQ15_LOGS": str(tmp_path / "logs"),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_RQ15_ARTIFACTS_LIBRARY": str(artifacts),
            "STUB_ARTIFACT_STATUS": "0",
        },
    )

    assert result.returncode == 0
    assert "ENQUEUE " not in result.stderr
    assert f"skipped compatible {candidate.run_name}" in result.stdout


def test_followup_validator_rejects_fabricated_canonical_candidate() -> None:
    anchor = next(
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "auxiliary_ntp"
        and candidate.deep_lr == 0.048
    )
    approved = _followup_evidence(
        [make_boundary_candidate(anchor, "deep", "high", 1)]
    )
    fabricated = _followup_evidence(
        [
            Rq15Candidate(
                "auxiliary_ntp",
                embedding_lr=0.123,
                deep_lr=0.777,
                stage="lr_boundary",
                boundary_axis="deep",
                boundary_direction="high",
                boundary_step=7,
            )
        ]
    )

    with pytest.raises(ValueError, match="artifact-derived report"):
        validated_required_followup_candidates(
            fabricated,
            authoritative_evidence=approved,
        )


def test_followup_launcher_rejects_empty_foreign_stale_and_manual_candidates(
    tmp_path: Path,
) -> None:
    anchor = next(
        candidate
        for candidate in initial_candidates()
        if candidate.training_method == "auxiliary_ntp"
        and candidate.deep_lr == 0.012
    )
    candidate = make_auxiliary_weight_candidate(anchor, 0.1)
    valid = _followup_evidence([candidate])
    documents = [
        {**valid, "required_followups": []},
        {**valid, "claims_status": "ready_for_user_validation"},
        {
            **valid,
            "required_followups": [
                {**valid["required_followups"][0], "deep_lr": 0.024}
            ],
        },
        {
            **valid,
            "required_followups": [
                {
                    **valid["required_followups"][0],
                    "run_name": anchor.run_name,
                    "stage": "initial",
                }
            ],
        },
    ]
    for index, document in enumerate(documents):
        evidence = tmp_path / f"invalid-{index}.json"
        evidence.write_text(json.dumps(document))
        result = subprocess.run(
            ["bash", str(FOLLOWUP_LAUNCHER), "--list"],
            capture_output=True,
            text=True,
            env=os.environ | {"G1_RQ15_EVIDENCE": str(evidence)},
        )
        assert result.returncode == 2
        assert result.stdout == ""

    manual = subprocess.run(
        ["bash", str(FOLLOWUP_LAUNCHER), candidate.run_name],
        capture_output=True,
        text=True,
    )
    assert manual.returncode == 2
    assert "usage:" in manual.stderr
