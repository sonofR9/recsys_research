import json
from pathlib import Path

import pytest

from experiments.g6_rqkmeans_history.launchers.remediation_compiled import (
    build_remediation_experiment,
    decode_remediation_job,
    encode_remediation_job,
)
from experiments.g6_rqkmeans_history.launchers.remediation_manifest import (
    load_remediation_jobs,
)
from experiments.g6_rqkmeans_history.launchers.remediation_workflow import (
    CONTROL_SELECTION_PATH,
    RemediationLedger,
    RemediationWorkflow,
    load_control_reference,
)
from experiments.g6_rqkmeans_history.protocol.manifest import CompiledJob
from experiments.g6_rqkmeans_history.protocol.remediation import (
    CARRYOVER_MANIFEST_SHA256,
    CONTROL_JOB_ID,
    CONTROL_RUN_NAME,
    FROZEN_EVENT_WIDTH,
    RemediationDriver,
    RunBudgetApprovalRequired,
    carryover_compiled_jobs,
    compile_remediation_cap_continuation,
    remediation_manifest,
    validate_remediation_job,
)
from experiments.g6_rqkmeans_history.protocol.evidence import (
    BoundaryApprovalRequired,
)
from experiments.g6_rqkmeans_history.protocol.remediation_evidence import (
    RemediationArtifact,
    require_remediation_boundary_resolved,
)


def _driver(path: Path) -> RemediationDriver:
    driver = RemediationDriver(path)
    driver.register_carryovers(
        [
            (compiled, 0.03 + index / 1000, Path(f"artifact-{index}"))
            for index, compiled in enumerate(carryover_compiled_jobs())
        ]
    )
    return driver


def _adaptive_treatment(driver: RemediationDriver) -> CompiledJob:
    for index in range(3):
        anchor = driver.next_treatment()
        assert anchor is not None
        driver.tell(anchor, 0.04 + index / 1000, Path(f"anchor-{index}"))
    treatment = driver.next_treatment()
    assert treatment is not None
    return treatment


def test_remediation_manifest_is_versioned_and_accounts_for_the_approved_budget() -> (
    None
):
    manifest = remediation_manifest()

    document = manifest.to_dict()
    assert document["version"] == 3
    assert manifest.stage_counts == {
        "remediation_tuning": 16,
        "remediation_lr_boundary": 8,
        "remediation_bridge_tuning": 12,
        "remediation_bridge_lr_boundary": 8,
    }
    assert document["initial_runs"] == 16
    assert document["maximum_runs"] == 44
    assert len({job.id for job in manifest.jobs}) == 44
    tuning = manifest.jobs_for_stage("remediation_tuning")
    assert ["remediation_v2" in job.run_name for job in tuning] == [
        True,
        True,
        *([False] * 14),
    ]
    assert all("remediation_v3" in job.run_name for job in manifest.jobs[2:])
    assert document["carryovers"] == [
        {
            "job_id": compiled.approved.id,
            "parameters": compiled.parameters,
            "source_manifest_sha256": CARRYOVER_MANIFEST_SHA256,
        }
        for compiled in carryover_compiled_jobs()
    ]
    assert [anchor["trial"] for anchor in document["anchors"]] == [2, 3, 4]
    assert document["sampler"]["restart_stable_trial_seed"] == {
        "hash": "sha256",
        "material": "{seed}:{stage}:{trial_number}",
        "integer": "first_4_bytes_big_endian",
    }


def test_treatment_study_fixes_the_control_path_and_tunes_only_approved_axes(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    compiled = driver.next_treatment()
    assert compiled is not None

    validate_remediation_job(compiled)
    assert compiled.parameters["batch_size"] == 256
    assert compiled.parameters["validation_batch_size"] == 8192
    assert compiled.parameters["num_levels"] == 3
    assert compiled.parameters["num_codes"] == 512
    assert compiled.parameters["frozen_event_width"] == FROZEN_EVENT_WIDTH == 128
    assert compiled.parameters["source_control_job_id"] == CONTROL_JOB_ID
    assert compiled.parameters["source_control_run_name"] == CONTROL_RUN_NAME
    assert compiled.parameters["representation_width"] == 32
    assert compiled.parameters["embedding_learning_rate"] == 0.256
    assert compiled.parameters["deep_learning_rate"] == 0.03463626154088337

    driver.tell(compiled, 0.1, tmp_path / "artifact")
    second = driver.next_treatment()
    assert second is not None
    assert second.parameters["representation_width"] == 64
    assert second.parameters["embedding_learning_rate"] == 0.256


def test_carryover_and_anchor_parameters_are_manifest_enforced(tmp_path: Path) -> None:
    carryover = carryover_compiled_jobs()[0]
    changed = CompiledJob(
        carryover.approved,
        carryover.parameters | {"deep_learning_rate": 0.01},
    )

    with pytest.raises(ValueError, match="carryover parameters changed"):
        validate_remediation_job(changed)

    driver = _driver(tmp_path / "study.sqlite3")
    anchor = driver.next_treatment()
    assert anchor is not None
    changed = CompiledJob(
        anchor.approved,
        anchor.parameters | {"embedding_learning_rate": 0.128},
    )
    with pytest.raises(ValueError, match="anchor parameters changed"):
        validate_remediation_job(changed)


def test_tpe_sequence_is_identical_across_driver_restarts(tmp_path: Path) -> None:
    def collect(path: Path, *, restart: bool) -> list[dict]:
        driver = _driver(path)
        parameters = []
        for index in range(7):
            if restart and index:
                driver = _driver(path)
            compiled = driver.next_treatment()
            assert compiled is not None
            parameters.append(compiled.parameters)
            driver.tell(compiled, 0.04 + index / 1000, Path(f"result-{index}"))
        return parameters

    uninterrupted = collect(tmp_path / "continuous.sqlite3", restart=False)
    interrupted = collect(tmp_path / "restarted.sqlite3", restart=True)

    assert interrupted == uninterrupted
    assert (
        len(
            {
                (
                    row["representation_width"],
                    row["embedding_learning_rate"],
                    row["deep_learning_rate"],
                )
                for row in interrupted[3:]
            }
        )
        == 4
    )


def test_sampler_seed_and_study_manifest_identity_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sampler seed"):
        RemediationDriver(tmp_path / "wrong-seed.sqlite3", seed=99)

    path = tmp_path / "study.sqlite3"
    _driver(path)
    study = __import__("optuna").load_study(
        study_name="g6-rq0-remediation-v3-remediation_tuning",
        storage=f"sqlite:///{path.resolve()}",
    )
    assert study.user_attrs["protocol_identity"]["manifest_sha256"] == (
        remediation_manifest().sha256
    )
    study.set_user_attr("protocol_identity", {"manifest_sha256": "changed"})
    with pytest.raises(ValueError, match="study protocol identity changed"):
        _driver(path)


def test_physical_run_budget_counts_cap_continuations(tmp_path: Path) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    bridge = driver.next_bridge(driver.next_treatment())  # type: ignore[arg-type]
    assert bridge is not None
    ledger = RemediationLedger(tmp_path / "ledger.json")
    compiled = bridge
    for _ in range(44):
        ledger.append(compiled)
        compiled = compile_remediation_cap_continuation(compiled)

    with pytest.raises(RunBudgetApprovalRequired, match="44 physical runs"):
        ledger.append(compiled)


def test_crossing_batch_is_rejected_before_any_job_is_reserved(tmp_path: Path) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    treatment = driver.next_treatment()
    assert treatment is not None
    bridge = driver.next_bridge(treatment)
    assert bridge is not None
    ledger_path = tmp_path / "ledger.json"
    ledger = RemediationLedger(ledger_path)
    compiled = bridge
    for _ in range(42):
        ledger.append(compiled)
        compiled = compile_remediation_cap_continuation(compiled)
    crossing = []
    for _ in range(3):
        crossing.append(compiled)
        compiled = compile_remediation_cap_continuation(compiled)
    submitted = []
    workflow = RemediationWorkflow(
        driver,
        logs_root=tmp_path / "logs",
        ledger_path=ledger_path,
        submit=submitted.append,  # type: ignore[arg-type]
    )

    with pytest.raises(RunBudgetApprovalRequired, match="44 physical runs"):
        workflow.run(tuple(crossing))

    assert len(load_remediation_jobs(ledger_path)) == 42
    assert submitted == []


def test_cap_extension_is_capacity_checked_before_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    treatment = driver.next_treatment()
    assert treatment is not None
    bridge = driver.next_bridge(treatment)
    assert bridge is not None
    ledger_path = tmp_path / "ledger.json"
    ledger = RemediationLedger(ledger_path)
    compiled = bridge
    for index in range(44):
        ledger.append(compiled)
        if index < 43:
            compiled = compile_remediation_cap_continuation(compiled)
    submitted = []
    workflow = RemediationWorkflow(
        driver,
        logs_root=tmp_path / "logs",
        ledger_path=ledger_path,
        submit=submitted.append,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        "experiments.g6_rqkmeans_history.launchers.remediation_workflow."
        "remediation_artifact_state",
        lambda *_: "extend_cap",
    )

    with pytest.raises(RunBudgetApprovalRequired, match="44 physical runs"):
        workflow.run((compiled,))

    assert len(load_remediation_jobs(ledger_path)) == 44
    assert submitted == []


def test_remediation_builder_uses_the_residual_representation_and_separate_widths(
    tmp_path: Path,
) -> None:
    compiled = _driver(tmp_path / "study.sqlite3").next_treatment()
    assert compiled is not None

    experiment = build_remediation_experiment(compiled)

    assert experiment.history_representation == (
        "item_frozen_sid_learned_residual_event"
    )
    assert experiment.semantic.num_levels == 3
    assert experiment.semantic.num_codes == 512
    assert experiment.frozen_event_width == 128
    assert experiment.representation_width in {32, 64, 128}
    assert experiment.transformer.num_layers == 4


def test_remediation_contract_round_trips_without_accepting_old_manifest_identity(
    tmp_path: Path,
) -> None:
    compiled = _driver(tmp_path / "study.sqlite3").next_treatment()
    assert compiled is not None

    encoded = encode_remediation_job(compiled)

    assert decode_remediation_job(encoded) == compiled
    contract = json.loads(__import__("base64").urlsafe_b64decode(encoded).decode())
    contract["manifest_sha256"] = "old"
    changed = (
        __import__("base64").urlsafe_b64encode(json.dumps(contract).encode()).decode()
    )
    with pytest.raises(RuntimeError, match="different approved remediation manifest"):
        decode_remediation_job(changed)


def test_boundary_freezes_all_non_lr_treatment_parameters(tmp_path: Path) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    source = _adaptive_treatment(driver)
    source = CompiledJob(
        source.approved,
        source.parameters
        | {"embedding_learning_rate": 1e-4, "deep_learning_rate": 0.01},
    )

    boundaries = driver.compile_lr_boundaries(source)

    assert len(boundaries) == 4
    assert all(
        boundary.parameters["source_parameters"] == source.parameters
        for boundary in boundaries
    )
    assert all(
        boundary.parameters["embedding_learning_rate"] < 1e-4 for boundary in boundaries
    )
    for boundary in boundaries:
        validate_remediation_job(boundary)
        experiment = build_remediation_experiment(boundary)
        assert experiment.history_representation == (
            "item_frozen_sid_learned_residual_event"
        )


def test_outermost_boundary_winner_requires_new_approval(tmp_path: Path) -> None:
    source = _adaptive_treatment(_driver(tmp_path / "study.sqlite3"))
    source = CompiledJob(
        source.approved,
        source.parameters
        | {"embedding_learning_rate": 1e-4, "deep_learning_rate": 0.01},
    )
    outer = _driver(tmp_path / "other.sqlite3").compile_lr_boundaries(source)[-1]
    artifact = RemediationArtifact(outer, tmp_path, {}, {}, {})

    with pytest.raises(BoundaryApprovalRequired, match="outermost"):
        require_remediation_boundary_resolved(artifact)


def test_original_bridge_can_extend_an_unresolved_epoch_cap(tmp_path: Path) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    treatment = driver.next_treatment()
    assert treatment is not None
    bridge = driver.next_bridge(treatment)
    assert bridge is not None

    continuation = compile_remediation_cap_continuation(bridge)

    assert continuation.attempt == 1
    assert continuation.cap_epochs == 60
    assert decode_remediation_job(encode_remediation_job(continuation)) == continuation
    assert build_remediation_experiment(continuation).num_epochs == 60

    boundary_source = CompiledJob(
        bridge.approved,
        bridge.parameters
        | {"embedding_learning_rate": 1e-4, "deep_learning_rate": 0.01},
    )
    boundary = driver.compile_lr_boundaries(boundary_source)[0]
    boundary_continuation = compile_remediation_cap_continuation(boundary)

    assert boundary_continuation.attempt == 1
    assert boundary_continuation.cap_epochs == 60
    assert (
        decode_remediation_job(encode_remediation_job(boundary_continuation))
        == boundary_continuation
    )
    assert build_remediation_experiment(boundary_continuation).num_epochs == 60


def test_original_bridge_authenticates_the_selected_treatment(tmp_path: Path) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    treatment = driver.next_treatment()
    assert treatment is not None
    bridge = driver.next_bridge(treatment)
    assert bridge is not None
    assert bridge.parameters["selected_treatment_run_name"] == treatment.run_name
    assert bridge.parameters["selected_treatment_parameters"] == treatment.parameters

    changed = CompiledJob(
        bridge.approved,
        bridge.parameters | {"selected_treatment_job_id": "not-approved"},
    )
    with pytest.raises(ValueError, match="selected treatment"):
        validate_remediation_job(changed)


def test_promotion_uses_the_approved_absolute_metric_bands() -> None:
    driver = RemediationDriver

    assert driver.promotion_eligible(
        control_recall=0.13018,
        control_ndcg=0.05168,
        treatment_recall=0.13219,
        treatment_ndcg=0.04968,
    )
    assert not driver.promotion_eligible(
        control_recall=0.13018,
        control_ndcg=0.05168,
        treatment_recall=0.13218,
        treatment_ndcg=0.05168,
    )
    assert not driver.promotion_eligible(
        control_recall=0.13018,
        control_ndcg=0.05168,
        treatment_recall=0.133,
        treatment_ndcg=0.04967,
    )


def test_control_reference_is_bound_to_the_audited_selection(tmp_path: Path) -> None:
    reference = load_control_reference(CONTROL_SELECTION_PATH)

    assert reference["control"]["job_id"] == CONTROL_JOB_ID
    assert reference["control"]["run_name"] == CONTROL_RUN_NAME
    assert reference["control"]["parameters"]["num_levels"] == 3
    assert reference["control"]["parameters"]["num_codes"] == 512
    assert reference["control"]["parameters"]["representation_width"] == 128

    changed = json.loads(CONTROL_SELECTION_PATH.read_text())
    changed["semantic_winner"]["metrics"]["recall@100"] = 0.5
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="control"):
        load_control_reference(path)

    changed = json.loads(CONTROL_SELECTION_PATH.read_text())
    changed["original_control"]["metrics"]["recall@100"] = 0.5
    path = tmp_path / "changed-original.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="original control row"):
        load_control_reference(path)
