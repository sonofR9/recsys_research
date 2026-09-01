import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.g3_pretrained_item_embeddings.launchers import (
    rq4_capacity_extension,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    G3_CPU_THREAD_ENVIRONMENT,
    PROJECT_ROOT,
    decode_control_job,
    encode_control_job,
    find_existing_ledger_batch,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4 import (
    RQ4_METADATA_FAMILIES,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_capacity_extension_ledger import (
    RQ4_CAPACITY_SELECTION_SHA256,
    RQ4_CAPACITY_EXTENSION_LEDGER_PATH,
    compile_rq4_capacity_extension_ledger,
    load_rq4_capacity_extension_ledger,
    persist_rq4_capacity_extension_ledger,
    validate_rq4_capacity_extension_ledger_document,
)


def test_width128_extension_is_deterministic_equal_and_exactly_nine_rows() -> None:
    first = compile_rq4_capacity_extension_ledger(PROJECT_ROOT)
    second = compile_rq4_capacity_extension_ledger(PROJECT_ROOT)

    assert first == second
    assert first.capacity_selection.logical_sha256 == RQ4_CAPACITY_SELECTION_SHA256
    assert len(first.rows) == 9
    for family_id in RQ4_METADATA_FAMILIES:
        rows = [row for row in first.rows if row.family_id == family_id]
        assert len(rows) == 3
        assert len(
            {
                (row.embedding_learning_rate, row.deep_learning_rate)
                for row in rows
            }
        ) == 3
        assert {row.horizon_epochs for row in rows} == {25}
        assert {
            row.to_dict()["representation"]["metadata_dim"] for row in rows
        } == {128}


def test_width128_extension_persistence_is_immutable(tmp_path: Path) -> None:
    ledger = compile_rq4_capacity_extension_ledger(PROJECT_ROOT)
    canonical = PROJECT_ROOT / RQ4_CAPACITY_EXTENSION_LEDGER_PATH

    persist_rq4_capacity_extension_ledger(canonical, ledger, root=PROJECT_ROOT)
    persist_rq4_capacity_extension_ledger(canonical, ledger, root=PROJECT_ROOT)

    with pytest.raises(ValueError, match="canonical project path"):
        persist_rq4_capacity_extension_ledger(
            tmp_path / "ledger.json", ledger, root=PROJECT_ROOT
        )


def test_width128_queue_surface_contains_exactly_nine_jobs(tmp_path: Path) -> None:
    ledger = compile_rq4_capacity_extension_ledger(PROJECT_ROOT)
    commands = rq4_capacity_extension.compile_rq4_capacity_extension_queue_commands(
        ledger_path=PROJECT_ROOT
        / "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
        "rq4_metadata_capacity_width128.json",
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )

    assert len(commands) == 12
    assert sum("enqueue-run" in command for command in commands) == 9
    assert {
        command[command.index("--run") + 1] for command in commands[2:11]
    } == {row.run_name for row in ledger.rows}


def test_width128_training_builder_uses_exact_ledger_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = compile_rq4_capacity_extension_ledger(PROJECT_ROOT)
    compiled = decode_control_job(
        encode_control_job(ledger, ledger.rows[0].id), ledger
    )
    captured = {}

    def fake_builder(**arguments):
        captured.update(arguments)
        return SimpleNamespace(base_path="unused")

    monkeypatch.setattr(rq4_capacity_extension, "build_g3_experiment", fake_builder)
    rq4_capacity_extension.build_training_experiment(
        compiled,
        ledger=ledger,
        feature_data_path=PROJECT_ROOT / "feature-data",
    )

    job = compiled.job
    training = job["training"]
    representation = job["representation"]
    assert captured["embedding_learning_rate"] == training[
        "embedding_learning_rate"
    ]
    assert captured["deep_learning_rate"] == training["deep_learning_rate"]
    assert captured["lr_schedule_horizon_epochs"] == 25
    assert captured["representation"].metadata_dim == 128
    assert captured["representation"].metadata == tuple(
        representation["metadata"]
    )


def test_width128_loader_rejects_exact_copy_at_another_path(tmp_path: Path) -> None:
    ledger = compile_rq4_capacity_extension_ledger(PROJECT_ROOT)
    canonical = PROJECT_ROOT / RQ4_CAPACITY_EXTENSION_LEDGER_PATH
    copied = tmp_path / canonical.name
    copied.write_bytes(canonical.read_bytes())

    with pytest.raises(ValueError, match="canonical project path"):
        load_rq4_capacity_extension_ledger(
            copied,
            root=PROJECT_ROOT,
            expected_ledger_sha256=ledger.sha256,
        )


@pytest.mark.parametrize(
    ("field", "drifted"),
    (("schema_version", True), ("logical_total", 9.0)),
)
def test_width128_validator_rejects_exact_json_type_drift(
    field: str, drifted: object
) -> None:
    ledger = compile_rq4_capacity_extension_ledger(PROJECT_ROOT)
    document = ledger.to_dict()
    if field == "schema_version":
        document[field] = drifted
    else:
        accounting = document["opportunity_accounting"]
        assert isinstance(accounting, dict)
        accounting[field] = drifted

    with pytest.raises(ValueError, match="differs from authenticated inputs"):
        validate_rq4_capacity_extension_ledger_document(
            document,
            root=PROJECT_ROOT,
            expected_ledger_sha256=ledger.sha256,
        )


def test_width128_existing_batch_identity_ignores_copied_ledger_path(
    tmp_path: Path,
) -> None:
    ledger = compile_rq4_capacity_extension_ledger(PROJECT_ROOT)
    canonical = PROJECT_ROOT / RQ4_CAPACITY_EXTENSION_LEDGER_PATH
    copied = tmp_path / canonical.name
    copied.write_bytes(canonical.read_bytes())
    state_dir = tmp_path / "queue"
    completed = state_dir / "completed"
    batches = state_dir / "batches"
    completed.mkdir(parents=True)
    batches.mkdir()
    runner = PROJECT_ROOT / (
        "experiments/g3_pretrained_item_embeddings/launchers/"
        "run_rq4_capacity_extension.py"
    )
    batch_id = "existing-width128"
    job_ids = []
    for index, row in enumerate(ledger.rows):
        job_id = f"job-{index}"
        job_ids.append(job_id)
        environment = [
            f"{rq4_capacity_extension.JOB_ENVIRONMENT}="
            f"{encode_control_job(ledger, row.id)}",
            f"{rq4_capacity_extension.LEDGER_ENVIRONMENT}={copied}",
            "WANDB_MODE=offline",
            *G3_CPU_THREAD_ENVIRONMENT,
        ]
        (completed / f"{job_id}.json").write_text(
            json.dumps(
                {
                    "id": job_id,
                    "batch_id": batch_id,
                    "run": row.run_name,
                    "script": str(runner),
                    "data_group": "g3-native50m-likes",
                    "environment": environment,
                }
            )
        )
    (batches / f"{batch_id}.json").write_text(
        json.dumps({"id": batch_id, "sealed": True, "jobs": job_ids})
    )

    assert find_existing_ledger_batch(
        state_dir=state_dir,
        ledger_path=canonical,
        ledger=ledger,
        runner_script=runner,
        job_environment=rq4_capacity_extension.JOB_ENVIRONMENT,
        ledger_environment=rq4_capacity_extension.LEDGER_ENVIRONMENT,
        ledger_path_sensitive=False,
    ) == batch_id


def test_width128_compiled_job_codec_rejects_numeric_type_drift() -> None:
    ledger = compile_rq4_capacity_extension_ledger(PROJECT_ROOT)
    encoded = encode_control_job(ledger, ledger.rows[0].id)
    document = json.loads(base64.urlsafe_b64decode(encoded).decode())
    document["job"]["training"]["horizon_epochs"] = 25.0
    drifted = base64.urlsafe_b64encode(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).decode()

    with pytest.raises(ValueError, match="differs from its approved ledger row"):
        decode_control_job(drifted, ledger)
