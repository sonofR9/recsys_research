import copy
import json
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL,
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.control_ledger import (
    ManifestReference,
    initial_control_ledger,
    load_control_ledger,
    persist_control_ledger,
    validate_control_ledger_document,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    G3_CPU_THREAD_ENVIRONMENT,
    build_training_experiment,
    compile_queue_commands,
    decode_control_job,
    encode_control_job,
    verify_ledger_inputs,
    write_job_contract,
)
from experiments.g3_pretrained_item_embeddings.launchers import (
    control as control_launcher,
)
from experiments.g3_pretrained_item_embeddings.protocol import manifests
from experiments.g3_pretrained_item_embeddings.protocol.manifests import (
    build_artifact_manifest,
    persist_artifact_manifest,
)


def _reference(kind: str, sha256: str) -> ManifestReference:
    return ManifestReference(
        kind=kind,
        path=f"experiments/g3_pretrained_item_embeddings/protocol/artifacts/{kind}.json",
        sha256=sha256,
    )


def _ledger():
    return initial_control_ledger(
        g4_control=ManifestReference(
            kind="g4_selected_control",
            path="experiments/g4_future_items/protocol/selected_control_manifest.json",
            sha256=APPROVED_PROTOCOL.control.manifest_sha256,
        ),
        content=ManifestReference(
            kind="native50m_content",
            path=(
                "experiments/g3_pretrained_item_embeddings/protocol/artifacts/"
                "native50m_content.json"
            ),
            sha256=("5e24e5db5d3a5635433abd962b1de0753599618c2c0ab67edab6801b967ab070"),
        ),
        features=ManifestReference(
            kind="native50m_features",
            path=(
                "experiments/g3_pretrained_item_embeddings/protocol/artifacts/"
                "native50m_features.json"
            ),
            sha256=("02e919339094e5091e77d09bd77ea669b665c7f6f49a29b6f27d6708ee9cf021"),
        ),
    )


def test_initial_control_ledger_compiles_the_approved_ten_coordinates() -> None:
    ledger = _ledger()

    assert ledger.schema_version == 1
    assert ledger.kind == "g3_untied_control"
    assert ledger.protocol_sha256 == APPROVED_PROTOCOL_SHA256
    assert len(ledger.rows) == 10
    assert len({row.id for row in ledger.rows}) == 10
    assert len({row.run_name for row in ledger.rows}) == 10
    assert [row.role for row in ledger.rows].count("search") == 9
    assert [row.role for row in ledger.rows].count("transfer_check") == 1
    assert all(row.dataset_size == "native-50m" for row in ledger.rows)
    assert all(row.batch_size == 512 and row.seed == 42 for row in ledger.rows)
    assert all(row.representation == "untied_learned_item_id" for row in ledger.rows)
    assert (
        sum(
            row.embedding_learning_rate
            == APPROVED_PROTOCOL.control.embedding_learning_rate
            and row.deep_learning_rate == APPROVED_PROTOCOL.control.deep_learning_rate
            and row.horizon_epochs == APPROVED_PROTOCOL.control.horizon_epochs
            for row in ledger.rows
        )
        == 1
    )
    assert validate_control_ledger_document(ledger.to_dict()) == ledger


def test_control_ledger_schema_and_coordinates_fail_closed_on_drift() -> None:
    document = _ledger().to_dict()
    with_unknown = copy.deepcopy(document)
    with_unknown["unknown"] = True
    with pytest.raises(ValueError, match="ledger keys"):
        validate_control_ledger_document(with_unknown)

    changed_coordinate = copy.deepcopy(document)
    changed_coordinate["rows"][0]["training"]["embedding_learning_rate"] = 0.2
    with pytest.raises(ValueError, match="approved control coordinates"):
        validate_control_ledger_document(changed_coordinate)

    changed_representation = copy.deepcopy(document)
    changed_representation["rows"][0]["representation"]["tied"] = True
    with pytest.raises(ValueError, match="approved control coordinates"):
        validate_control_ledger_document(changed_representation)


def test_control_ledger_persistence_refuses_overwrite(tmp_path) -> None:
    destination = tmp_path / "untied_control.json"
    ledger = _ledger()

    persist_control_ledger(destination, ledger)
    first = destination.read_bytes()
    persist_control_ledger(destination, ledger)
    assert destination.read_bytes() == first

    changed = initial_control_ledger(
        g4_control=ledger.g4_control,
        content=ledger.content,
        features=_reference("native50m_features", "c" * 64),
    )
    with pytest.raises(RuntimeError, match="immutable control ledger"):
        persist_control_ledger(destination, changed)


def test_loader_rejects_a_self_consistent_replacement_ledger(tmp_path) -> None:
    ledger = _ledger_with_different_feature_hash()
    destination = tmp_path / "replacement.json"
    persist_control_ledger(destination, ledger)

    with pytest.raises(ValueError, match="approved immutable hash"):
        load_control_ledger(destination)


def test_control_job_codec_is_bound_to_the_exact_ledger_row() -> None:
    ledger = _ledger()
    encoded = encode_control_job(ledger, ledger.rows[0].id)

    compiled = decode_control_job(encoded, ledger)
    assert compiled.ledger_sha256 == ledger.sha256
    assert compiled.row_id == ledger.rows[0].id
    assert compiled.job == ledger.rows[0].to_dict()

    with pytest.raises(ValueError, match="approved ledger row"):
        decode_control_job(encoded, _ledger_with_different_feature_hash())


def test_control_queue_commands_are_ten_granular_persistent_jobs(tmp_path) -> None:
    ledger = _ledger()
    ledger_path = tmp_path / "untied_control.json"
    commands = compile_queue_commands(
        ledger_path=ledger_path,
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )

    enqueue = [command for command in commands if "enqueue-run" in command]
    assert len(enqueue) == 10
    assert len({command[command.index("--run") + 1] for command in enqueue}) == 10
    assert all(
        command[command.index("--data-group") + 1] == "g3-native50m-likes"
        for command in enqueue
    )
    assert all(
        Path(command[command.index("--script") + 1]).name == "run_control.py"
        for command in enqueue
    )
    assert all(
        any(value.startswith("G3_CONTROL_JOB_B64=") for value in command)
        for command in enqueue
    )
    assert all(
        any(value == "WANDB_MODE=offline" for value in command) for command in enqueue
    )
    assert all(
        all(value in command for value in G3_CPU_THREAD_ENVIRONMENT)
        for command in enqueue
    )
    assert commands[0][-2:] == ["status", "--json"]
    assert commands[1][-1] == "new-batch"
    assert commands[-1][-2:] == ["seal-batch", "DRY_RUN_BATCH"]


def test_control_runtime_uses_bound_feature_file_and_exact_contract_path(
    tmp_path,
) -> None:
    ledger = _ledger()
    ledger_path = tmp_path / "untied_control.json"
    persist_control_ledger(ledger_path, ledger)
    compiled = decode_control_job(encode_control_job(ledger, ledger.rows[0].id), ledger)
    item_features = tmp_path / "bound" / "item_features.parquet"

    experiment = build_training_experiment(
        compiled,
        feature_data_path=item_features,
    )
    assert experiment.feature_data_path == item_features
    destination = write_job_contract(compiled, ledger_path, tmp_path / "logs")
    assert destination == (
        tmp_path / "logs" / ledger.rows[0].run_name / "g3_control_job.json"
    )


def test_runtime_input_check_skips_raw_content_but_checks_consumed_compact(
    tmp_path,
    monkeypatch,
) -> None:
    files = {}
    for role in (
        "content_source",
        "compaction_implementation",
        "compact_remap",
        "compact_output",
        "events_source",
        "materialization_implementation",
        "item_features",
        "training_user_histories",
        "artist_vocab",
        "album_vocab",
    ):
        path = tmp_path / "inputs" / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode())
        files[role] = path
    content = build_artifact_manifest(
        root=tmp_path,
        artifacts={
            role: files[role]
            for role in (
                "content_source",
                "compaction_implementation",
                "compact_remap",
                "compact_output",
            )
        },
        metadata={"dataset_size": "native-50m"},
    )
    feature_roles = (
        "events_source",
        "compact_remap",
        "materialization_implementation",
        "item_features",
        "training_user_histories",
        "artist_vocab",
        "album_vocab",
    )
    features = build_artifact_manifest(
        root=tmp_path,
        artifacts={role: files[role] for role in feature_roles},
        metadata={
            "dataset_size": "native-50m",
            "validation_interval_seconds": 604800,
            "num_items": 3,
            "training_rows": 10,
            "training_users": 2,
            "artist_vocab_size": 4,
            "album_vocab_size": 5,
            "artist_unknown_rate": 0.1,
            "album_unknown_rate": 0.2,
            "artist_max_cardinality": 2,
            "album_max_cardinality": 3,
        },
    )
    content_path = tmp_path / "protocol/content.json"
    features_path = tmp_path / "protocol/features.json"
    persist_artifact_manifest(content_path, content)
    persist_artifact_manifest(features_path, features)
    g4_path = (
        tmp_path / "experiments/g4_future_items/protocol/selected_control_manifest.json"
    )
    g4_path.parent.mkdir(parents=True)
    g4_path.write_text(
        json.dumps(
            {
                "seed_42_configuration": {
                    "run_name": APPROVED_PROTOCOL.control.run_name,
                    "selected": {
                        "batch_size": 512,
                        "embedding_learning_rate": APPROVED_PROTOCOL.control.embedding_learning_rate,
                        "deep_learning_rate": APPROVED_PROTOCOL.control.deep_learning_rate,
                        "lr_schedule_horizon_epochs": APPROVED_PROTOCOL.control.horizon_epochs,
                    },
                }
            }
        )
    )
    ledger = initial_control_ledger(
        g4_control=ManifestReference(
            "g4_selected_control",
            "experiments/g4_future_items/protocol/selected_control_manifest.json",
            APPROVED_PROTOCOL.control.manifest_sha256,
        ),
        content=ManifestReference(
            "native50m_content",
            content_path.relative_to(tmp_path).as_posix(),
            content.sha256,
        ),
        features=ManifestReference(
            "native50m_features",
            features_path.relative_to(tmp_path).as_posix(),
            features.sha256,
        ),
    )
    approved_compact_hash = next(
        binding.sha256
        for binding in content.artifacts
        if binding.role == "compact_output"
    )
    monkeypatch.setattr(
        type(APPROVED_PROTOCOL),
        "content_hash",
        lambda self, dataset_size: approved_compact_hash,
    )
    monkeypatch.setattr(
        control_launcher,
        "_file_sha256",
        lambda path: APPROVED_PROTOCOL.control.manifest_sha256,
    )
    original = manifests._file_sha256
    raw_source = files["content_source"].resolve()

    def forbid_raw_source(path):
        if path.resolve() == raw_source:
            raise AssertionError("runtime rehashed the raw content source")
        return original(path)

    monkeypatch.setattr(manifests, "_file_sha256", forbid_raw_source)
    item_features = verify_ledger_inputs(
        tmp_path,
        ledger,
        full_validation=False,
    )
    assert item_features == files["item_features"]

    files["compact_output"].write_bytes(b"drifted_compact")
    with pytest.raises(ValueError, match="compact_output"):
        verify_ledger_inputs(tmp_path, ledger, full_validation=False)


def _ledger_with_different_feature_hash():
    ledger = _ledger()
    return initial_control_ledger(
        g4_control=ledger.g4_control,
        content=ledger.content,
        features=_reference("native50m_features", "c" * 64),
    )
