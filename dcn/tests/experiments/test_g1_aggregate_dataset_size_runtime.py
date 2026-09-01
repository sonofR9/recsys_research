from __future__ import annotations

import fcntl
import math
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from experiments.g1_aggregate_dataset_size.protocol.candidates import (
    BATCH_LR_CALIBRATION_PAIRS,
    FIXED_MEMBERS,
    INITIAL_JOINT_LR_PAIRS,
    MAX_APPROVED_RUNS,
    MAX_HORIZON_CORRECTION_RUNS,
    MAX_PRE_HORIZON_RUNS,
    ApprovalRequired,
    aggregate_initial_candidates,
    baseline_initial_candidates,
    batch_followup_candidates,
    batch_initial_candidates,
    batch_lr_calibration_candidates,
    bridge_candidates,
    candidate_by_run,
    local_lr_candidates,
    optimizer_boundary_candidates,
    repeat_candidates,
)
from experiments.g1_aggregate_dataset_size.launchers.runtime import (
    CandidateResult,
    InfeasibleBatchCell,
    archive_infeasible_batch_artifact,
    archive_retry_artifact,
    completion_is_valid,
    load_infeasible_batch_cells,
    load_verified_results,
    stage_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_candidates import (
    initial_candidates as frozen_500m_candidates,
)


FROZEN_500M_INITIAL_RUN_NAMES = (
    "g1_aggregate_baseline_none_l2_e0p064_d0p006_hnone_c0_initial_ts2_r1_500m",
    "g1_aggregate_baseline_none_l2_e0p064_d0p012_hnone_c0_initial_ts2_r1_500m",
    "g1_aggregate_baseline_none_l2_e0p064_d0p024_hnone_c0_initial_ts2_r1_500m",
    "g1_aggregate_aggregate_none_l4_e0p064_d0p048_h15_c0_initial_ts2_r1_500m",
    "g1_aggregate_aggregate_none_l4_e0p07764674795069047_d0p02484672863178322_h15_c0_initial_ts2_r1_500m",
    "g1_aggregate_aggregate_none_l4_e0p0468526465053628_d0p032703745675187676_h15_c0_initial_ts2_r1_500m",
    "g1_aggregate_aggregate_none_l6_e0p064_d0p048_h15_c0_initial_ts2_r1_500m",
    "g1_aggregate_aggregate_none_l6_e0p07764674795069047_d0p02484672863178322_h15_c0_initial_ts2_r1_500m",
    "g1_aggregate_aggregate_none_l6_e0p0468526465053628_d0p032703745675187676_h15_c0_initial_ts2_r1_500m",
    "g1_aggregate_aggregate_none_l8_e0p064_d0p048_h15_c0_initial_ts2_r1_500m",
    "g1_aggregate_aggregate_none_l8_e0p07764674795069047_d0p02484672863178322_h15_c0_initial_ts2_r1_500m",
    "g1_aggregate_aggregate_none_l8_e0p0468526465053628_d0p032703745675187676_h15_c0_initial_ts2_r1_500m",
)


def _result(candidate, recall: float = 0.1, ndcg: float = 0.04, best: int = 4):
    return CandidateResult(candidate, recall, ndcg, best)


def _load_config(candidate):
    path = (
        Path(__file__).parents[3]
        / "experiments/g1_aggregate_dataset_size/configs/aggregate_variant.py"
    )
    previous = {
        name: value for name, value in os.environ.items() if name.startswith("G1_")
    }
    os.environ["G1_AGGREGATE_RUN"] = candidate.run_name
    try:
        return runpy.run_path(str(path))["experiment"]
    finally:
        for name in tuple(os.environ):
            if name.startswith("G1_"):
                os.environ.pop(name)
        os.environ.update(previous)


def test_initial_50m_path_has_exact_approved_composition_and_37_runs() -> None:
    calibration = batch_lr_calibration_candidates()
    selected = calibration[3]
    repeats = repeat_candidates(selected)
    bridges = bridge_candidates(selected)
    aggregates = aggregate_initial_candidates(1280)

    assert [candidate.batch_size for candidate in calibration] == [
        512,
        512,
        512,
        1280,
        1280,
        1280,
    ]
    assert [candidate.seed for candidate in repeats] == list(range(43, 52))
    assert tuple(candidate.member for candidate in bridges[:10]) == FIXED_MEMBERS
    assert [candidate.num_layers for candidate in bridges[10:]] == [4, 6, 8]
    assert {
        (candidate.num_layers, candidate.embedding_lr, candidate.deep_lr)
        for candidate in aggregates
    } == {
        (layers, embedding_lr, deep_lr)
        for layers in (4, 6, 8)
        for embedding_lr, deep_lr in INITIAL_JOINT_LR_PAIRS
    }
    assert len({
        candidate.run_name
        for candidate in (*calibration, *repeats, *bridges, *aggregates)
    }) == 37


def test_corrected_batch_lr_manifest_has_equal_three_candidate_budgets() -> None:
    candidates = batch_lr_calibration_candidates()

    assert len(candidates) == 6
    assert {candidate.batch_size for candidate in candidates} == {512, 1280}
    assert {
        candidate.batch_size: tuple(
            (row.embedding_lr, row.deep_lr)
            for row in candidates
            if row.batch_size == candidate.batch_size
        )
        for candidate in candidates
    } == {512: BATCH_LR_CALIBRATION_PAIRS, 1280: BATCH_LR_CALIBRATION_PAIRS}
    assert {candidate.stage for candidate in candidates} == {"batch_lr_calibration"}
    assert stage_candidates("batch_lr_calibration", {}) == candidates


def test_corrected_batch_lr_selection_waits_for_all_six_and_propagates_pair() -> None:
    calibration = batch_lr_calibration_candidates()
    results = {
        candidate.run_name: _result(
            candidate,
            recall=0.4 if candidate.batch_size == 512 and index == 1 else 0.2,
            ndcg=0.05,
        )
        for index, candidate in enumerate(calibration)
    }

    with pytest.raises(RuntimeError, match="six-cell batch/LR calibration"):
        stage_candidates("repeats", dict(list(results.items())[:-1]))

    repeats = stage_candidates("repeats", results)
    assert len(repeats) == 9
    assert {candidate.batch_size for candidate in repeats} == {512}
    assert {candidate.embedding_lr for candidate in repeats} == {
        calibration[1].embedding_lr
    }
    assert {candidate.deep_lr for candidate in repeats} == {calibration[1].deep_lr}

    old = batch_initial_candidates()[0]
    results[old.run_name] = _result(old, recall=0.99, ndcg=0.99)
    assert stage_candidates("repeats", results) == repeats


def test_50m_names_are_isolated_and_500m_identities_are_unchanged() -> None:
    frozen_names = FROZEN_500M_INITIAL_RUN_NAMES
    candidates = (*batch_initial_candidates(), *aggregate_initial_candidates(1280))

    assert tuple(
        candidate.run_name for candidate in frozen_500m_candidates()
    ) == FROZEN_500M_INITIAL_RUN_NAMES
    assert all(candidate.run_name.endswith("_50m") for candidate in candidates)
    assert not set(frozen_names).intersection(candidate.run_name for candidate in candidates)
    assert all(candidate_by_run(candidate.run_name) == candidate for candidate in candidates)
    with pytest.raises(ValueError, match="unknown native-50M aggregate run"):
        candidate_by_run(frozen_names[0])


def test_native_50m_config_materializes_baseline_and_all_eleven_members() -> None:
    baseline_candidate = baseline_initial_candidates(1280)[0]
    aggregate_candidate = aggregate_initial_candidates(1280)[0]
    baseline = _load_config(baseline_candidate)
    aggregate = _load_config(aggregate_candidate)

    assert baseline.size == aggregate.size == "50m"
    assert baseline.dataloader.batch_size == aggregate.dataloader.batch_size == 1280
    assert baseline.num_epochs == 80
    assert baseline.early_stopping_patience == 3
    assert baseline.lr_schedule.shape == "constant"
    assert aggregate.num_epochs == aggregate.lr_schedule_horizon_epochs == 15
    assert aggregate.transformer.num_layers == 4
    assert aggregate.transformer.ffn == "swiglu"
    assert aggregate.transformer.ffn_intermediate_dim == 192
    assert aggregate.transformer.alibi
    assert aggregate.transformer.learned_positions == ("forward", "reverse")
    assert aggregate.transformer.norm_place == "post"
    assert aggregate.transformer.input_norm == "rms"
    assert aggregate.transformer.final_norm == "rms"
    assert aggregate.cls_token_mode == "end_only"
    assert aggregate.timestamp_num_bins == 32
    assert aggregate.transformer.rope == "timestamp_reverse"
    assert aggregate.negative_sampling == "random_offline_logq"
    assert aggregate.num_in_batch_negatives == 2048
    assert aggregate.transformer.num_kv_heads == 1
    assert aggregate.bos


def test_bounded_followups_match_the_approved_limits() -> None:
    batch_outer = batch_followup_candidates(batch_initial_candidates()[0])
    baseline_noncentral = baseline_initial_candidates(1280)[1]
    local = local_lr_candidates(baseline_noncentral)
    boundary = optimizer_boundary_candidates(local[1])

    assert [candidate.batch_size for candidate in batch_outer] == [160, 320, 480]
    assert len(local) == 3
    assert len(boundary) in {3, 6}
    assert (MAX_PRE_HORIZON_RUNS, MAX_HORIZON_CORRECTION_RUNS, MAX_APPROVED_RUNS) == (
        64,
        74,
        138,
    )
    with pytest.raises(ApprovalRequired, match="outer batch boundary"):
        batch_followup_candidates(batch_outer[0])


def test_stage_dependencies_fail_closed_before_repeats_and_bridges() -> None:
    with pytest.raises(RuntimeError, match="batch/LR calibration"):
        stage_candidates("baseline_initial", {})

    calibration = batch_lr_calibration_candidates()
    calibration_results = {
        candidate.run_name: _result(
            candidate,
            recall=0.3 if candidate.batch_size == 1280 and index == 3 else 0.2,
        )
        for index, candidate in enumerate(calibration)
    }
    assert stage_candidates("baseline_initial", calibration_results) == ()
    repeats = stage_candidates("repeats", calibration_results)
    with pytest.raises(RuntimeError, match="ten baseline repeats"):
        stage_candidates("bridges", calibration_results)
    assert len(repeats) == 9


@pytest.mark.parametrize(
    ("candidate_kind", "metadata", "num_users", "expected"),
    [
        (
            "constant",
            {
                "max_epochs": 80,
                "stopped_epoch": 12,
                "best_epoch": 9,
                "early_stopped": True,
                "selection_resolved": True,
                "best_epoch_at_cap": False,
            },
            3414,
            True,
        ),
        (
            "constant",
            {
                "max_epochs": 80,
                "stopped_epoch": 80,
                "best_epoch": 79,
                "early_stopped": False,
                "selection_resolved": False,
                "best_epoch_at_cap": False,
            },
            3414,
            False,
        ),
        (
            "h15",
            {
                "max_epochs": 15,
                "stopped_epoch": 15,
                "best_epoch": 14,
                "epochs_trained": 15,
                "lr_schedule_horizon_epochs": 15,
                "lr_horizon_complete": True,
            },
            3414,
            True,
        ),
        (
            "h15",
            {
                "max_epochs": 15,
                "stopped_epoch": 14,
                "best_epoch": 14,
                "epochs_trained": 14,
                "lr_schedule_horizon_epochs": 15,
                "lr_horizon_complete": False,
            },
            3414,
            False,
        ),
        (
            "constant",
            {
                "max_epochs": 80,
                "stopped_epoch": 12,
                "best_epoch": 9,
                "early_stopped": True,
                "selection_resolved": True,
                "best_epoch_at_cap": False,
            },
            37018,
            False,
        ),
    ],
)
def test_completion_rules_require_native_user_count_and_schedule_or_cap_contract(
    candidate_kind: str, metadata: dict[str, object], num_users: int, expected: bool
) -> None:
    if candidate_kind == "constant":
        candidate = baseline_initial_candidates(1280)[0]
    else:
        candidate = aggregate_initial_candidates(1280)[0]
        if expected:
            metadata.update(_scheduled_trace_metadata(candidate))
    assert completion_is_valid(candidate, metadata, {"num_users": num_users}) is expected


def test_scheduled_completion_rejects_missing_or_incorrect_deep_only_traces() -> None:
    candidate = aggregate_initial_candidates(1280)[0]
    metadata = {
        "max_epochs": 15,
        "stopped_epoch": 15,
        "best_epoch": 14,
        "epochs_trained": 15,
        "lr_schedule_horizon_epochs": 15,
        "lr_horizon_complete": True,
        **_scheduled_trace_metadata(candidate),
    }
    metrics = {"num_users": 3414}

    assert completion_is_valid(candidate, metadata, metrics)
    missing = dict(metadata)
    missing.pop("lr_group_traces")
    assert not completion_is_valid(candidate, missing, metrics)
    changed_embedding = dict(metadata)
    changed_embedding["lr_group_traces"] = {
        **metadata["lr_group_traces"],
        "embedding": [candidate.embedding_lr * 0.9] * 15,
    }
    assert not completion_is_valid(candidate, changed_embedding, metrics)
    changed_deep = dict(metadata)
    deep_trace = list(metadata["lr_group_traces"]["deep"])
    deep_trace[4] *= 0.9
    changed_deep["lr_group_traces"] = {
        **metadata["lr_group_traces"],
        "deep": deep_trace,
    }
    assert not completion_is_valid(candidate, changed_deep, metrics)
    wrong_horizon = dict(metadata)
    wrong_horizon["lr_schedule_horizon_steps"] -= 1
    assert not completion_is_valid(candidate, wrong_horizon, metrics)


def test_h15_h24_h36_endpoint_rules_are_bounded() -> None:
    initial = aggregate_initial_candidates(1280)[:3]
    results = _complete_aggregate_prerequisites()
    results.update({
        candidate.run_name: _result(candidate, recall=0.3 if index == 0 else 0.2, best=15)
        for index, candidate in enumerate(initial)
    })
    h24 = stage_candidates("aggregate_followups", results, aggregate_depth=4)
    assert len(h24) == 3
    assert {candidate.horizon_epochs for candidate in h24} == {24}

    h24_results = dict(results)
    h24_results.update(
        {
            candidate.run_name: _result(candidate, recall=0.3 if index == 0 else 0.2, best=24)
            for index, candidate in enumerate(h24)
        }
    )
    h36 = stage_candidates("aggregate_followups", h24_results, aggregate_depth=4)
    assert len(h36) == 3
    assert {candidate.horizon_epochs for candidate in h36} == {36}

    h36_results = dict(h24_results)
    h36_results.update(
        {
            candidate.run_name: _result(candidate, recall=0.3 if index == 0 else 0.2, best=36)
            for index, candidate in enumerate(h36)
        }
    )
    with pytest.raises(ApprovalRequired, match="H36"):
        stage_candidates("aggregate_followups", h36_results, aggregate_depth=4)


def test_batch_selection_requires_explicit_persisted_oom_resolution(
    tmp_path: Path,
) -> None:
    candidates = batch_initial_candidates()
    logs = tmp_path / "logs"
    logs.mkdir()
    failed = logs / candidates[-1].run_name
    failed.mkdir()
    (failed / "sweep.log").write_text(
        "torch.OutOfMemoryError: CUDA out of memory while allocating tensor\n"
    )
    ledger = logs / ".g1-aggregate-50m-infeasible.json"

    cell = archive_infeasible_batch_artifact(failed, candidates[-1], ledger)
    outcomes = {
        candidates[0].run_name: _result(candidates[0], recall=0.2),
        candidates[1].run_name: _result(candidates[1], recall=0.3),
        cell.candidate.run_name: cell,
    }

    assert load_infeasible_batch_cells(ledger) == {
        candidates[-1].run_name: cell
    }
    with pytest.raises(ValueError, match="audit-only"):
        stage_candidates("batch_initial", outcomes)
    with pytest.raises(RuntimeError, match="batch/LR calibration"):
        stage_candidates("baseline_initial", outcomes)


def test_arbitrary_failure_is_archived_for_retry_but_never_marked_infeasible(
    tmp_path: Path,
) -> None:
    candidate = batch_initial_candidates()[0]
    logs = tmp_path / "logs"
    logs.mkdir()
    failed = logs / candidate.run_name
    failed.mkdir()
    (failed / "sweep.log").write_text("RuntimeError: data worker stopped\n")
    ledger = logs / ".g1-aggregate-50m-infeasible.json"

    with pytest.raises(ValueError, match="recognized CUDA OOM"):
        archive_infeasible_batch_artifact(failed, candidate, ledger)

    lock_path = logs / ".run-locks" / f"{candidate.run_name}.lock"
    lock_path.parent.mkdir()
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active run"):
            archive_retry_artifact(failed)
        assert failed.is_dir()

    archived = archive_retry_artifact(failed)
    assert archived.parent == logs / "old"
    assert archived.is_dir()
    assert (archived / "sweep.log").read_text() == "RuntimeError: data worker stopped\n"
    assert not ledger.exists()


def test_parseable_off_manifest_lr_never_enters_baseline_selection() -> None:
    batches = batch_lr_calibration_candidates()
    results = {
        candidate.run_name: _result(
            candidate,
            recall=0.4 if candidate.batch_size == 1280 and index == 3 else 0.2,
        )
        for index, candidate in enumerate(batches)
    }
    forged = type(batches[0])(
        "baseline",
        0.025,
        0.013,
        1280,
        "baseline_local",
    )
    assert candidate_by_run(forged.run_name) == forged
    results[forged.run_name] = _result(forged, recall=0.99)

    repeats = stage_candidates("repeats", results)

    assert {candidate.embedding_lr for candidate in repeats} == {
        batches[3].embedding_lr
    }
    assert {candidate.deep_lr for candidate in repeats} == {batches[3].deep_lr}


def test_parseable_off_manifest_aggregate_surface_cannot_replace_approved_grid() -> None:
    approved = aggregate_initial_candidates(1280)[:3]
    results = {
        candidate.run_name: _result(candidate, recall=0.3 if index == 0 else 0.2)
        for index, candidate in enumerate(approved)
    }
    forged = tuple(
        type(approved[0])(
            "aggregate",
            embedding_lr,
            deep_lr,
            1280,
            "aggregate_initial",
            num_layers=4,
            horizon_epochs=15,
        )
        for embedding_lr, deep_lr in ((0.02, 0.01), (0.03, 0.015), (0.04, 0.02))
    )
    results.update(
        {
            candidate.run_name: _result(candidate, recall=0.99)
            for candidate in forged
        }
    )
    prerequisites = _complete_aggregate_prerequisites()
    prerequisites.update(results)

    assert stage_candidates(
        "aggregate_followups", prerequisites, aggregate_depth=4
    ) == ()

    forged_only = _complete_aggregate_prerequisites()
    forged_only.update(
        {
            candidate.run_name: _result(candidate, recall=0.99)
            for candidate in forged
        }
    )
    with pytest.raises(RuntimeError, match="surface is incomplete"):
        stage_candidates(
            "aggregate_followups",
            forged_only,
            aggregate_depth=4,
        )


def test_old_batch_boundary_is_audit_only(tmp_path: Path) -> None:
    initial = batch_initial_candidates()
    results = {
        candidate.run_name: _result(
            candidate, recall=0.4 if candidate.batch_size == 640 else 0.2
        )
        for candidate in initial
    }
    followups = batch_followup_candidates(initial[0])
    results.update({candidate.run_name: _result(candidate, recall=0.9) for candidate in followups})

    with pytest.raises(ValueError, match="audit-only"):
        stage_candidates("batch_followup", results)
    with pytest.raises(RuntimeError, match="batch/LR calibration"):
        stage_candidates("baseline_initial", results)


def test_aggregate_followups_require_every_upstream_selected_prerequisite() -> None:
    aggregate = aggregate_initial_candidates(1280)[:3]
    isolated = {
        candidate.run_name: _result(
            candidate, recall=0.4 if index == 0 else 0.3, best=15
        )
        for index, candidate in enumerate(aggregate)
    }

    with pytest.raises(RuntimeError, match="batch/LR calibration"):
        stage_candidates(
            "aggregate_followups", isolated, aggregate_depth=4
        )

    prerequisites = _complete_aggregate_prerequisites()
    prerequisites.update(isolated)
    assert {
        candidate.horizon_epochs
        for candidate in stage_candidates(
            "aggregate_followups", prerequisites, aggregate_depth=4
        )
    } == {24}

    unresolved_scheduler = dict(prerequisites)
    scheduler = next(
        outcome.candidate
        for outcome in unresolved_scheduler.values()
        if outcome.candidate.family == "bridge"
        and outcome.candidate.member == "scheduler"
    )
    unresolved_scheduler[scheduler.run_name] = _result(scheduler, best=15)
    with pytest.raises(RuntimeError, match="scheduler bridge horizon"):
        stage_candidates(
            "aggregate_followups", unresolved_scheduler, aggregate_depth=4
        )


def test_infeasible_transaction_uses_ledger_lock_and_override_relative_paths(
    tmp_path: Path,
) -> None:
    candidate = batch_initial_candidates()[0]
    logs = tmp_path / "artifacts" / "logs"
    failed = logs / candidate.run_name
    failed.mkdir(parents=True)
    (failed / "sweep.log").write_text(
        "torch.OutOfMemoryError: CUDA out of memory while allocating tensor\n"
    )
    ledger = tmp_path / "state" / "infeasible.json"
    ledger.parent.mkdir()
    ledger_lock = ledger.with_name(f"{ledger.name}.lock")

    with ledger_lock.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="infeasible ledger is locked"):
            archive_infeasible_batch_artifact(failed, candidate, ledger)
        assert failed.is_dir()
        assert not ledger.exists()

    cell = archive_infeasible_batch_artifact(failed, candidate, ledger)
    assert not Path(cell.archive_path).is_absolute()
    assert load_infeasible_batch_cells(ledger) == {candidate.run_name: cell}
    assert load_verified_results(logs, ledger) == {candidate.run_name: cell}
    assert load_verified_results(logs) == {}
    assert load_verified_results(tmp_path / "missing-logs", ledger) == {
        candidate.run_name: cell
    }


def test_launcher_rejects_noncanonical_logs_before_queue_execution(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[3]
    launcher = (
        root
        / "experiments/g1_aggregate_dataset_size/launchers/aggregate_50m.sh"
    )
    environment = dict(os.environ)
    environment["G1_AGGREGATE_LOGS"] = str(tmp_path / "redirected-logs")

    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "G1_AGGREGATE_LOGS must resolve to" in result.stderr


def _scheduled_trace_metadata(candidate) -> dict[str, object]:
    horizon = candidate.horizon_epochs
    assert horizon is not None
    steps_per_epoch = 10
    total_steps = steps_per_epoch * horizon
    warmup_steps = int(total_steps * 0.05)
    decay_steps = total_steps - warmup_steps - 1
    factors = []
    for epoch in range(1, horizon + 1):
        step = epoch * steps_per_epoch - 1
        if step < warmup_steps:
            factors.append((step + 1) / warmup_steps)
        else:
            progress = min(1.0, (step - warmup_steps) / decay_steps)
            factors.append(
                0.0
                if progress == 1
                else 0.5 * (1 + math.cos(math.pi * progress))
            )
    return {
        "optimizer_steps_per_epoch": steps_per_epoch,
        "lr_schedule_horizon_steps": total_steps,
        "embedding_learning_rate": candidate.embedding_lr,
        "deep_learning_rate": candidate.deep_lr,
        "lr_group_traces": {
            "embedding": [candidate.embedding_lr] * horizon,
            "deep": [candidate.deep_lr * factor for factor in factors],
        },
    }


def _complete_aggregate_prerequisites() -> dict[str, CandidateResult]:
    calibration = batch_lr_calibration_candidates()
    results = {
        candidate.run_name: _result(
            candidate,
            recall=0.5 if candidate.batch_size == 1280 and index == 3 else 0.2,
        )
        for index, candidate in enumerate(calibration)
    }
    selected = calibration[3]
    results.update(
        {
            candidate.run_name: _result(candidate)
            for candidate in repeat_candidates(selected)
        }
    )
    results.update(
        {
            candidate.run_name: _result(candidate)
            for candidate in bridge_candidates(selected)
        }
    )
    return results
