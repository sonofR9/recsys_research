import base64
import json
from pathlib import Path

import pytest

from experiments.g6_rqkmeans_history.launchers.remediation_bounded import (
    build_bounded_gate_experiment,
    decode_bounded_gate_job,
    encode_bounded_gate_job,
)
from experiments.g6_rqkmeans_history.launchers.bounded_gate_manifest import (
    load_bounded_gate_jobs,
    write_bounded_gate_jobs,
)
from experiments.g6_rqkmeans_history.launchers.bounded_gate_workflow import (
    _record_launch_or_reuse_complete,
)
from experiments.g6_rqkmeans_history.protocol.manifest import CompiledJob
from experiments.g6_rqkmeans_history.protocol.remediation_bounded import (
    BOUNDED_GATE_SCALES,
    SOURCE_SELECTION_PATH,
    bounded_gate_jobs,
    bounded_gate_manifest,
    load_bounded_gate_source,
    validate_bounded_gate_job,
)


def test_bounded_gate_manifest_binds_the_five_approved_cells() -> None:
    manifest = bounded_gate_manifest()

    assert manifest.to_dict()["version"] == 1
    assert manifest.to_dict()["prior_physical_runs"] == 20
    assert manifest.to_dict()["new_physical_runs"] == 5
    assert manifest.to_dict()["maximum_total_physical_runs"] == 44
    assert [
        job.parameters["learned_residual_max_scale"] for job in bounded_gate_jobs()
    ] == list(BOUNDED_GATE_SCALES)
    assert len({job.approved.id for job in bounded_gate_jobs()}) == 5
    assert len({job.run_name for job in bounded_gate_jobs()}) == 5


def test_bounded_gate_builder_changes_only_the_approved_bound() -> None:
    jobs = bounded_gate_jobs()

    experiments = [build_bounded_gate_experiment(job) for job in jobs]

    assert [
        experiment.learned_residual_max_scale for experiment in experiments
    ] == list(BOUNDED_GATE_SCALES)
    assert {experiment.representation_width for experiment in experiments} == {32}
    assert {experiment.frozen_event_width for experiment in experiments} == {128}
    assert {experiment.embedding_learning_rate for experiment in experiments} == {0.256}
    assert {experiment.deep_learning_rate for experiment in experiments} == {
        0.03463626154088337
    }


def test_bounded_gate_contract_round_trips_and_rejects_parameter_changes() -> None:
    compiled = bounded_gate_jobs()[2]

    assert decode_bounded_gate_job(encode_bounded_gate_job(compiled)) == compiled

    changed = CompiledJob(
        compiled.approved,
        compiled.parameters | {"learned_residual_max_scale": 0.03},
    )
    with pytest.raises(ValueError, match="bounded-gate parameters changed"):
        validate_bounded_gate_job(changed)

    contract = json.loads(base64.urlsafe_b64decode(encode_bounded_gate_job(compiled)))
    contract["manifest_sha256"] = "changed"
    encoded = base64.urlsafe_b64encode(json.dumps(contract).encode()).decode()
    with pytest.raises(RuntimeError, match="manifest"):
        decode_bounded_gate_job(encoded)


def test_bounded_gate_ledger_requires_the_exact_approved_grid(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    jobs = bounded_gate_jobs()

    write_bounded_gate_jobs(path, jobs)
    assert load_bounded_gate_jobs(path) == jobs

    document = json.loads(path.read_text())
    document["jobs"].pop()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="exact approved grid"):
        load_bounded_gate_jobs(path)


def test_bounded_gate_source_is_bound_to_the_completed_v3_selection(
    tmp_path: Path,
) -> None:
    source = load_bounded_gate_source(SOURCE_SELECTION_PATH)

    assert source["run_counts"]["total_including_carryovers"] == 20
    assert source["treatment_winner"]["job_id"] == (
        "remediation_tuning:learned_sid_residual_trial_02"
    )

    changed = json.loads(SOURCE_SELECTION_PATH.read_text())
    changed["treatment_winner"]["metrics"]["recall@100"] = 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="source selection changed"):
        load_bounded_gate_source(path)


def test_recorded_bounded_gate_launch_is_never_resubmitted(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    logs = tmp_path / "logs"

    assert _record_launch_or_reuse_complete(ledger_path=ledger, logs_root=logs) is True

    with pytest.raises(RuntimeError, match="launch is recorded"):
        _record_launch_or_reuse_complete(ledger_path=ledger, logs_root=logs)


def test_partial_bounded_gate_artifact_requires_audit(tmp_path: Path) -> None:
    run = tmp_path / "logs" / bounded_gate_jobs()[0].run_name
    run.mkdir(parents=True)
    (run / "stdout.log").write_text("partial")

    with pytest.raises(RuntimeError, match="partial artifact"):
        _record_launch_or_reuse_complete(
            ledger_path=tmp_path / "ledger.json", logs_root=tmp_path / "logs"
        )
