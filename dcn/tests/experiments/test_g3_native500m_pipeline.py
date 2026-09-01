from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from dcn.eval.ranking_evidence import RankingEvidence, write_ranking_evidence
from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
    authenticate_compatibility_resolution,
    derive_compatibility_resolution,
    derive_compatibility_transition,
    derive_conditional_family_selection,
    collect_batch_evidence,
    derive_family_selection,
    load_batch_evidence,
    load_family_selection,
    load_compatibility_resolution,
    persist_batch_evidence,
    persist_family_selection,
    persist_compatibility_resolution,
)
import experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence as evidence_module
from experiments.g3_pretrained_item_embeddings.configs.model import (
    G3Representation,
    G3GenerationExperiment,
    _publish_final_evaluation_bundle,
    _state_dict_sha256 as model_state_dict_sha256,
)
from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
    PROTOCOL_SHA256 as NATIVE500M_PROTOCOL_SHA256,
    compile_baseline_rows,
)
from experiments.g4_future_items.configs.control import G4GenerationExperiment
from experiments.g3_pretrained_item_embeddings.analysis.native500m_report import (
    render_native500m_reports,
)
import experiments.g3_pretrained_item_embeddings.analysis.native500m_report as report_module
from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
    DATA_GROUP,
    JOB_ENVIRONMENT,
    MANIFEST_ENVIRONMENT,
    MANIFEST_LOGICAL_SHA256_ENVIRONMENT,
    MANIFEST_PHYSICAL_SHA256_ENVIRONMENT,
    InputManifestReference,
    authenticate_training_queue_service,
    build_batch_specification,
    build_implementation_identity,
    freeze_execution_manifest,
    load_execution_manifest,
    materialize_baseline_execution_manifest,
    materialize_conditional_boundary_execution_manifest,
    materialize_conditional_execution_manifest,
    materialize_continuation_execution_manifest,
    materialize_selected_execution_manifest,
    persist_batch_specification,
    persist_queue_submission_binding,
    submit_execution_manifest,
)
import experiments.g3_pretrained_item_embeddings.launchers.native500m as launcher_module
import experiments.g3_pretrained_item_embeddings.launchers.run_native500m as runner_module
from experiments.g3_pretrained_item_embeddings.launchers.run_native500m import (
    build_training_experiment,
    load_compiled_job,
    write_job_contract,
)


PROTOCOL_SHA256 = "a" * 64
TEST_POPULATION = {
    "num_users": 2,
    "user_ids_sha256": hashlib.sha256(b"[11,22]").hexdigest(),
}


@pytest.fixture(autouse=True)
def _approved_test_population(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launcher_module,
        "APPROVED_EVALUATION_POPULATION",
        TEST_POPULATION,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evidence_import_does_not_execute_training_worker() -> None:
    environment = os.environ.copy()
    environment[JOB_ENVIRONMENT] = "poisoned-worker-payload"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence",
        ],
        cwd=Path(__file__).parents[3],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _write_input_manifest(path: Path, role: str) -> InputManifestReference:
    payload = {
        "kind": role,
        "rows": [1, 2, 3],
        **(
            {"metadata": {"evaluation_user_count": 2, "num_items": 157_357}}
            if role == "dataset"
            else {}
        ),
    }
    logical_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    path.write_bytes(_canonical_bytes({**payload, "sha256": logical_sha256}))
    return InputManifestReference.from_path(
        role=role,
        root=path.parent,
        path=path,
        logical_sha256=logical_sha256,
    )


def _job() -> dict[str, object]:
    implementation_prefix = str(build_implementation_identity()["sha256"])[:12]
    job = compile_baseline_rows()[0].to_dict()
    return job | {
        "run_name": f"{launcher_module._run_name(job)}_i{implementation_prefix}",
        "resolved_representation": G3Representation(item_id_tying="tied").to_dict(),
        "predecessor_artifacts": [],
    }


def _freeze_manifest(root: Path, *, protocol_sha256: str = PROTOCOL_SHA256) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    references = tuple(
        _write_input_manifest(root / f"{role}.json", role)
        for role in ("dataset", "content", "features")
    )
    path = root / "baseline.execution.json"
    return freeze_execution_manifest(
        path,
        stage="baseline",
        rows=compile_baseline_rows(),
        input_manifests=references,
        job_payloads={
            row.id: {
                "resolved_representation": G3Representation(
                    item_id_tying="tied"
                ).to_dict(),
                "predecessor_artifacts": [],
            }
            for row in compile_baseline_rows()
        },
        protocol_sha256=protocol_sha256,
    )


def test_execution_manifest_and_atomic_batch_bind_logical_and_physical_identity(
    tmp_path: Path,
) -> None:
    manifest_path = _freeze_manifest(tmp_path)
    manifest = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
        validate_inputs=True,
    )

    assert manifest.stage == "baseline"
    assert manifest.source_snapshot.name == manifest.implementation_identity["sha256"]
    assert (
        manifest.source_snapshot
        / "experiments/g3_pretrained_item_embeddings/launchers/run_native500m.py"
    ).is_file()
    assert manifest.logical_sha256 == json.loads(manifest_path.read_text())["sha256"]
    assert manifest.physical_sha256 == _sha256(manifest_path)
    specification = build_batch_specification(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
    )
    assert specification.document["version"] == 1
    assert len(specification.document["jobs"]) == 9
    queue_job = specification.document["jobs"][0]
    assert queue_job["data_group"] == DATA_GROUP
    environment = dict(value.split("=", 1) for value in queue_job["environment"])
    assert Path(queue_job["script"]) == (
        manifest.source_snapshot
        / "experiments/g3_pretrained_item_embeddings/launchers/run_native500m.py"
    )
    assert environment["PYTHONSAFEPATH"] == "1"
    assert environment["PYTHONPATH"] == str(manifest.source_snapshot)
    assert environment[MANIFEST_ENVIRONMENT] == str(manifest_path.resolve())
    assert environment[MANIFEST_LOGICAL_SHA256_ENVIRONMENT] == manifest.logical_sha256
    assert environment[MANIFEST_PHYSICAL_SHA256_ENVIRONMENT] == manifest.physical_sha256
    payload = json.loads(
        base64.b64decode(
            environment[JOB_ENVIRONMENT], altchars=b"-_", validate=True
        ).decode()
    )
    assert payload == {
        "job": _job(),
        "manifest_logical_sha256": manifest.logical_sha256,
        "manifest_physical_sha256": manifest.physical_sha256,
        "row_id": _job()["id"],
    }

    specification_path = persist_batch_specification(tmp_path / "specs", specification)
    assert specification_path.name == f"{specification.sha256}.json"
    assert specification_path.read_bytes() == _canonical_bytes(specification.document)
    persist_batch_specification(tmp_path / "specs", specification)


def test_queue_verifies_snapshot_before_importing_training_code(tmp_path: Path) -> None:
    manifest_path = _freeze_manifest(tmp_path / "manifest")
    manifest = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
        validate_inputs=True,
    )
    queue_job = build_batch_specification(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
    ).document["jobs"][0]
    binaries = tmp_path / "bin"
    binaries.mkdir()
    nvidia = binaries / "nvidia-smi"
    nvidia.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == --query-gpu=index* ]]; then printf '0, GPU-a\\n'; fi\n"
    )
    nvidia.chmod(0o755)
    training_python = binaries / "python"
    training_python.write_text(
        "#!/usr/bin/env bash\n"
        'touch "$DCN_PREPARED_MARKER"\n'
        'while [ ! -e "$DCN_TRAINING_RELEASE" ]; do sleep 0.01; done\n'
        "printf 'training-imported\\n' >> \"$QUEUE_RECORD\"\n"
    )
    training_python.chmod(0o755)
    queue = launcher_module.PROJECT_ROOT / "utils/training_queue/queue.sh"
    assignments = " ".join(shlex.quote(value) for value in queue_job["environment"])

    def run_queue(work: Path, run_name: str) -> subprocess.CompletedProcess[str]:
        work.mkdir()
        return subprocess.run(
            [
                "bash",
                "-c",
                (
                    f"source {shlex.quote(str(queue))} && "
                    f"enqueue {run_name} {assignments} && drain"
                ),
            ],
            cwd=work,
            env={
                **os.environ,
                "PATH": f"{binaries}:{os.environ['PATH']}",
                "QUEUE_RECORD": str(tmp_path / "training-record"),
                "TRAINING_QUEUE_IN_FLIGHT": "1",
                "TRAINING_QUEUE_SCRIPT": str(queue_job["script"]),
            },
            capture_output=True,
            text=True,
        )

    completed = run_queue(tmp_path / "valid-queue", "valid")
    assert completed.returncode == 0, completed.stderr
    record = tmp_path / "training-record"
    assert record.read_text() == "training-imported\n"

    runner = Path(queue_job["script"])
    runner.parent.chmod(0o755)
    runner.chmod(0o644)
    runner.write_bytes(runner.read_bytes() + b"\n")
    failed = run_queue(tmp_path / "corrupt-queue", "corrupt")

    assert failed.returncode != 0
    assert record.read_text() == "training-imported\n"
    assert (
        "failed immutable source snapshot verification"
        in (tmp_path / "corrupt-queue/generated/logs/corrupt/sweep.log").read_text()
    )


def test_snapshot_loaded_launcher_is_unchanged_by_live_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _freeze_manifest(tmp_path)
    manifest = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
        validate_inputs=True,
    )
    specification = build_batch_specification(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
    )
    environment = os.environ.copy()
    environment.update(
        value.split("=", 1)
        for value in specification.document["jobs"][0]["environment"]
    )
    environment.pop(JOB_ENVIRONMENT)
    monkeypatch.setattr(
        launcher_module,
        "build_implementation_identity",
        lambda: (_ for _ in ()).throw(AssertionError("live source was inspected")),
    )

    reloaded = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
        validate_inputs=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from experiments.g3_pretrained_item_embeddings.launchers.native500m "
                "import PROJECT_ROOT, build_implementation_identity; "
                "print(PROJECT_ROOT); print(build_implementation_identity()['sha256'])"
            ),
        ],
        cwd=launcher_module.PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert reloaded.implementation_identity == manifest.implementation_identity
    assert completed.stdout.splitlines() == [
        str(manifest.source_snapshot),
        str(manifest.implementation_identity["sha256"]),
    ]


def test_read_only_snapshot_runner_binds_writable_state_to_manifest_root(
    tmp_path: Path,
) -> None:
    manifest_path = _freeze_manifest(tmp_path)
    manifest = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
        validate_inputs=True,
    )
    queue_job = build_batch_specification(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
    ).document["jobs"][0]
    environment = os.environ.copy()
    environment.update(value.split("=", 1) for value in queue_job["environment"])
    environment["G3_TEST_JOB"] = environment.pop(JOB_ENVIRONMENT)
    program = """
import os
from pathlib import Path
from types import SimpleNamespace
import experiments.g3_pretrained_item_embeddings.configs.model as model
import experiments.g3_pretrained_item_embeddings.protocol.native500m.artifacts as artifacts
import experiments.g3_pretrained_item_embeddings.launchers.native500m as launcher
import experiments.g3_pretrained_item_embeddings.launchers.run_native500m as runner

feature_manifest = SimpleNamespace(
    artifacts=[SimpleNamespace(role="item_features", path="features.pt")]
)
artifacts.load_artifact_manifest = lambda path: path
artifacts.validate_content_manifest = lambda *args, **kwargs: SimpleNamespace(artifacts=[])
artifacts.validate_dataset_manifest = lambda *args, **kwargs: SimpleNamespace(artifacts=[])
artifacts.validate_feature_manifest = lambda *args, **kwargs: feature_manifest
model.build_native500m_job = lambda *args, **kwargs: SimpleNamespace()
launcher.APPROVED_EVALUATION_POPULATION = {
    "num_users": 2,
    "user_ids_sha256": "6f4d4ce6b7defe110fa3c5b995d392d039d47b11281252fca49e4a8e7ab1c7fb",
}
os.environ[runner.JOB_ENVIRONMENT] = os.environ.pop("G3_TEST_JOB")
compiled = runner.load_compiled_job(expected_protocol_sha256="a" * 64)
experiment = runner.build_training_experiment(compiled)
contract = runner.write_job_contract(compiled, Path(experiment.base_path) / "logs")
print(experiment.base_path)
print(contract)
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    expected_base = tmp_path / "generated"
    expected_contract = (
        expected_base
        / "logs"
        / str(queue_job["run"])
        / "g3_native500m_job.json"
    )
    assert completed.stdout.splitlines() == [str(expected_base), str(expected_contract)]
    assert expected_contract.is_file()
    assert not (manifest.source_snapshot / "generated").exists()


def test_manifest_semantic_replay_uses_bound_snapshot_not_live_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _freeze_manifest(tmp_path)
    launcher_module._BOUND_SOURCE_REPLAY_CACHE.clear()
    monkeypatch.setattr(
        launcher_module,
        "_validate_canonical_execution_manifest",
        lambda **_: (_ for _ in ()).throw(AssertionError("LIVE-COMPILER-USED")),
    )

    manifest = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
        validate_inputs=True,
    )

    assert manifest.source_snapshot.name == manifest.implementation_identity["sha256"]


def test_manifest_rejects_failed_bound_snapshot_semantic_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _freeze_manifest(tmp_path)
    launcher_module._BOUND_SOURCE_REPLAY_CACHE.clear()
    original_run = launcher_module.subprocess.run

    def fail_snapshot_replay(command, **arguments):
        if command[:2] == [sys.executable, "-c"]:
            return SimpleNamespace(returncode=9, stdout="", stderr="replay failed")
        return original_run(command, **arguments)

    monkeypatch.setattr(launcher_module.subprocess, "run", fail_snapshot_replay)

    with pytest.raises(ValueError, match="bound source semantic replay failed"):
        load_execution_manifest(
            manifest_path,
            expected_protocol_sha256=PROTOCOL_SHA256,
            validate_inputs=True,
        )


def test_manifest_location_defines_data_repository_independently_of_snapshot(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    inputs = repository / "inputs"
    inputs.mkdir(parents=True)
    references = tuple(
        _write_input_manifest(inputs / f"{role}.json", role)
        for role in ("dataset", "content", "features")
    )
    references = tuple(
        InputManifestReference.from_path(
            role=reference.role,
            root=repository,
            path=inputs / f"{reference.role}.json",
            logical_sha256=reference.logical_sha256,
        )
        for reference in references
    )
    manifest_path = freeze_execution_manifest(
        repository
        / "generated/g3-native500m/execution-manifests/baseline.execution.json",
        stage="baseline",
        rows=compile_baseline_rows(),
        input_manifests=references,
        job_payloads={
            row.id: {
                "resolved_representation": G3Representation(
                    item_id_tying="tied"
                ).to_dict(),
                "predecessor_artifacts": [],
            }
            for row in compile_baseline_rows()
        },
        protocol_sha256=PROTOCOL_SHA256,
    )

    manifest = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
        validate_inputs=True,
    )
    compiled = SimpleNamespace(manifest_path=manifest.path)

    assert runner_module._project_root(compiled.manifest_path) == repository
    assert manifest.source_snapshot.parent == (
        repository / "generated/g3-native500m/source-snapshots"
    )


@pytest.mark.parametrize("damage", ["partial", "corrupt"])
def test_execution_manifest_rejects_partial_or_corrupt_source_snapshot(
    tmp_path: Path, damage: str
) -> None:
    manifest_path = _freeze_manifest(tmp_path)
    manifest = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
        validate_inputs=True,
    )
    source = (
        manifest.source_snapshot
        / "experiments/g3_pretrained_item_embeddings/launchers/run_native500m.py"
    )
    source.parent.chmod(0o755)
    source.chmod(0o644)
    if damage == "partial":
        source.unlink()
    else:
        source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="source snapshot"):
        load_execution_manifest(
            manifest_path,
            expected_protocol_sha256=PROTOCOL_SHA256,
            validate_inputs=True,
        )


def test_snapshot_materialization_rejects_symlinked_generated_ancestor(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    (repository / "generated").symlink_to(outside, target_is_directory=True)
    references = tuple(
        _write_input_manifest(repository / f"{role}.json", role)
        for role in ("dataset", "content", "features")
    )

    with pytest.raises(ValueError, match="traverses a symlink"):
        freeze_execution_manifest(
            repository / "generated/g3-native500m/execution-manifests/baseline.json",
            stage="baseline",
            rows=compile_baseline_rows(),
            input_manifests=references,
            job_payloads={
                row.id: {
                    "resolved_representation": G3Representation(
                        item_id_tying="tied"
                    ).to_dict(),
                    "predecessor_artifacts": [],
                }
                for row in compile_baseline_rows()
            },
            protocol_sha256=PROTOCOL_SHA256,
        )
    assert not (outside / "g3-native500m/source-snapshots").exists()


def test_manifest_loading_rejects_symlinked_snapshot_ancestor(tmp_path: Path) -> None:
    manifest_path = _freeze_manifest(tmp_path)
    snapshots = tmp_path / "generated/g3-native500m/source-snapshots"
    relocated = tmp_path / "relocated-source-snapshots"
    snapshots.rename(relocated)
    snapshots.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(ValueError, match="traverses a symlink"):
        load_execution_manifest(
            manifest_path,
            expected_protocol_sha256=PROTOCOL_SHA256,
            validate_inputs=True,
        )


def test_baseline_materializer_compiles_nine_real_rows_and_explicit_representation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.g3_pretrained_item_embeddings.protocol.native500m.artifacts as artifacts

    calls = []
    for name in (
        "validate_content_manifest",
        "validate_dataset_manifest",
        "validate_feature_manifest",
    ):
        original = getattr(artifacts, name)

        def validate(manifest, *, _original=original, **arguments):
            calls.append(arguments["validate_files"])
            return _original(
                manifest,
                root=arguments["root"],
                validate_files=False,
                validate_semantics=arguments["validate_semantics"],
            )

        monkeypatch.setattr(artifacts, name, validate)
    launcher_module.native500m_input_manifest_references()
    references = tuple(
        _write_input_manifest(tmp_path / f"{role}.json", role)
        for role in ("dataset", "content", "features")
    )
    monkeypatch.setattr(
        launcher_module,
        "native500m_input_manifest_references",
        lambda: references,
    )
    manifest_path = materialize_baseline_execution_manifest(output_directory=tmp_path)
    manifest = load_execution_manifest(manifest_path, validate_inputs=True)

    assert manifest.stage == "baseline"
    assert len(manifest.rows) == 9
    assert {row["job"]["family_id"] for row in manifest.rows} == {"baseline"}
    assert {row["job"]["horizon_epochs"] for row in manifest.rows} == {10, 20, 40}
    assert {reference.role for reference in manifest.input_manifests} == {
        "content",
        "dataset",
        "features",
    }
    for row in manifest.rows:
        assert row["job"]["resolved_representation"] == {
            "history_representation": "learned_id",
            "catalog_representation": "learned_id",
            "history_hidden_dim": None,
            "content_gate": "fixed",
            "gate_hidden_dim": None,
            "metadata": [],
            "metadata_dim": None,
            "extra_item_id_dim": None,
            "item_id_tying": "tied",
        }
        assert row["job"]["predecessor_artifacts"] == []

    specification = build_batch_specification(manifest_path)
    assert len(specification.document["jobs"]) == 9
    assert calls == [True, True, True]


def test_manifest_and_runner_fail_closed_on_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _freeze_manifest(tmp_path)
    specification = build_batch_specification(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
    )
    environment = dict(
        value.split("=", 1)
        for value in specification.document["jobs"][0]["environment"]
    )
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    compiled = load_compiled_job(expected_protocol_sha256=PROTOCOL_SHA256)
    assert compiled.job == _job()
    contract_path = write_job_contract(compiled, tmp_path / "logs")
    assert contract_path.name == "g3_native500m_job.json"
    write_job_contract(compiled, tmp_path / "logs")

    document = json.loads(manifest_path.read_text())
    document["rows"][0]["job"]["batch_size"] = 1280
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(_canonical_bytes(document))
    with pytest.raises(ValueError, match="physical SHA-256"):
        load_compiled_job(expected_protocol_sha256=PROTOCOL_SHA256)


def test_runner_revalidates_source_after_lazy_model_construction_and_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.g3_pretrained_item_embeddings.configs.model as model_module
    import experiments.g3_pretrained_item_embeddings.protocol.native500m.artifacts as artifacts_module

    manifest_path = tmp_path / "execution.json"
    manifest_path.write_text("{}")
    references = tuple(
        _write_input_manifest(tmp_path / f"{role}.json", role)
        for role in ("content", "dataset", "features")
    )
    feature_path = tmp_path / "features.pt"
    feature_path.write_bytes(b"features")
    feature_manifest = SimpleNamespace(
        artifacts=[SimpleNamespace(role="item_features", path="features.pt")]
    )
    monkeypatch.setattr(artifacts_module, "load_artifact_manifest", lambda path: path)
    monkeypatch.setattr(
        artifacts_module,
        "validate_content_manifest",
        lambda *_, **__: SimpleNamespace(artifacts=[]),
    )
    monkeypatch.setattr(
        artifacts_module,
        "validate_dataset_manifest",
        lambda *_, **__: SimpleNamespace(artifacts=[]),
    )
    monkeypatch.setattr(
        artifacts_module,
        "validate_feature_manifest",
        lambda *_, **__: feature_manifest,
    )
    events = []
    experiment = SimpleNamespace()

    def build(*_, **__):
        events.append("build")
        return experiment

    def validate(_: object) -> dict[str, object]:
        events.append("validate")
        return {}

    monkeypatch.setattr(model_module, "build_native500m_job", build)
    monkeypatch.setattr(runner_module, "validate_current_source_ledger", validate)
    compiled = SimpleNamespace(
        manifest_path=manifest_path,
        input_manifests=references,
        job={"id": "job"},
        implementation_identity=build_implementation_identity(),
        evaluation_population=TEST_POPULATION,
    )

    assert build_training_experiment(compiled) is experiment
    assert experiment.base_path == tmp_path / "generated"
    assert events == ["build", "validate", "validate"]


def test_execution_manifest_rejects_a_rehashed_but_drifted_code_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _freeze_manifest(tmp_path)
    document = json.loads(manifest_path.read_text())
    identity = document["implementation_identity"]
    identity["files"][0]["sha256"] = "0" * 64
    identity_body = {key: value for key, value in identity.items() if key != "sha256"}
    identity["sha256"] = hashlib.sha256(_canonical_bytes(identity_body)).hexdigest()
    implementation_prefix = identity["sha256"][:12]
    for row in document["rows"]:
        row["job"][
            "run_name"
        ] = f"{launcher_module._run_name(row['job'])}_i{implementation_prefix}"
    manifest_body = {key: value for key, value in document.items() if key != "sha256"}
    document["sha256"] = hashlib.sha256(_canonical_bytes(manifest_body)).hexdigest()
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(_canonical_bytes(document))

    with pytest.raises(ValueError, match="source snapshot"):
        load_execution_manifest(
            manifest_path,
            expected_protocol_sha256=PROTOCOL_SHA256,
            validate_inputs=True,
        )


def test_execution_manifest_rejects_rehashed_partial_and_forged_ledgers(
    tmp_path: Path,
) -> None:
    partial_path = _freeze_manifest(tmp_path / "partial")
    partial = json.loads(partial_path.read_text())
    partial["rows"] = partial["rows"][:-1]
    partial["sha256"] = hashlib.sha256(
        _canonical_bytes(
            {key: value for key, value in partial.items() if key != "sha256"}
        )
    ).hexdigest()
    partial_path.chmod(0o644)
    partial_path.write_bytes(_canonical_bytes(partial))
    with pytest.raises(ValueError, match="bound source semantic replay failed"):
        load_execution_manifest(
            partial_path,
            expected_protocol_sha256=PROTOCOL_SHA256,
            validate_inputs=True,
        )

    forged_path = _freeze_manifest(tmp_path / "forged")
    forged = json.loads(forged_path.read_text())
    forged["rows"][0]["job"]["resolved_representation"]["item_id_tying"] = "untied"
    forged["sha256"] = hashlib.sha256(
        _canonical_bytes(
            {key: value for key, value in forged.items() if key != "sha256"}
        )
    ).hexdigest()
    forged_path.chmod(0o644)
    forged_path.write_bytes(_canonical_bytes(forged))
    with pytest.raises(ValueError, match="bound source semantic replay failed"):
        load_execution_manifest(
            forged_path,
            expected_protocol_sha256=PROTOCOL_SHA256,
            validate_inputs=True,
        )


def test_conditional_protocol_module_is_bound_into_immutable_execution_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _freeze_manifest(tmp_path)
    relative = (
        "experiments/g3_pretrained_item_embeddings/protocol/native500m/conditional.py"
    )
    target = launcher_module.PROJECT_ROOT / relative
    assert relative in {
        entry["path"] for entry in build_implementation_identity()["files"]
    }
    original_sha256 = launcher_module._file_sha256
    monkeypatch.setattr(
        launcher_module,
        "_file_sha256",
        lambda path: "e" * 64 if path == target else original_sha256(path),
    )
    manifest = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
        validate_inputs=True,
    )
    assert manifest.implementation_identity != build_implementation_identity()


def test_execution_identity_covers_live_g3_runtime_sources_and_excludes_non_sources() -> (
    None
):
    identity = build_implementation_identity()
    ordered_paths = [entry["path"] for entry in identity["files"]]
    paths = set(ordered_paths)

    assert {
        "experiments/g3_pretrained_item_embeddings/analysis/queue_attribution.py",
        "experiments/g3_pretrained_item_embeddings/analysis/native500m_report.py",
        "utils/global_config.py",
        "utils/locks.py",
        "utils/report_file_facts.py",
        "utils/training_queue/service.py",
        "utils/training_queue/gpu_check.py",
        "utils/training_queue/queue_depth.py",
        "utils/training_queue/queue.sh",
        "utils/training_queue/service_scheduler.sh",
        "experiments/g4_future_items/__init__.py",
        "experiments/g4_future_items/configs/__init__.py",
        "experiments/g4_future_items/configs/control.py",
        "experiments/g4_future_items/protocol/control_manifest.json",
        "experiments/generation_protocol.py",
    }.issubset(paths)
    assert {
        "dcn/tests/test_training_queue.py",
        "neuralrec/tests/test_recall.py",
        "data/tests/test_preprocessing.py",
        "utils/tests/test_report_file_facts.py",
        "dcn/data/old/dataset_binary.py",
        "utils/training_queue/README.md",
        "experiments/g3_pretrained_item_embeddings/protocol/plan.md",
        "experiments/g3_pretrained_item_embeddings/protocol/native500m/dataset_manifest.json",
        "experiments/g4_future_items/protocol/native500m/manifest.py",
    }.isdisjoint(paths)
    assert ordered_paths == sorted(ordered_paths)
    assert all(
        not {"tests", "old", "__pycache__"}.intersection(Path(path).parts)
        and not Path(path).name.startswith("test_")
        and not path.startswith("generated/")
        for path in paths
    )


def test_every_representative_runtime_dependency_leaves_frozen_manifest_loadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = (
        "experiments/g3_pretrained_item_embeddings/analysis/queue_attribution.py",
        "experiments/g3_pretrained_item_embeddings/analysis/native500m_report.py",
        "utils/global_config.py",
        "utils/locks.py",
        "utils/report_file_facts.py",
        "utils/training_queue/service.py",
        "utils/training_queue/queue.sh",
        "utils/training_queue/service_scheduler.sh",
        "experiments/g4_future_items/__init__.py",
        "experiments/g4_future_items/configs/__init__.py",
    )
    stale_manifest = _freeze_manifest(tmp_path / "stale")
    stale_document = json.loads(stale_manifest.read_text())
    stale_identity = stale_document["implementation_identity"]
    stale_prefix = stale_identity["sha256"][:12]
    assert all(
        row["job"]["run_name"].endswith(f"_i{stale_prefix}")
        for row in stale_document["rows"]
    )

    original_sha256 = launcher_module._file_sha256
    for index, relative in enumerate(dependencies):
        target = launcher_module.PROJECT_ROOT / relative
        replacement = f"{index + 1:064x}"
        with monkeypatch.context() as patch:
            patch.setattr(
                launcher_module,
                "_file_sha256",
                lambda path, *, _target=target, _replacement=replacement: (
                    _replacement if path == _target else original_sha256(path)
                ),
            )
            changed_identity = build_implementation_identity()
            assert changed_identity["sha256"] != stale_identity["sha256"]
            assert (
                next(
                    entry
                    for entry in changed_identity["files"]
                    if entry["path"] == relative
                )["sha256"]
                == replacement
            )
            loaded = load_execution_manifest(
                stale_manifest,
                expected_protocol_sha256=PROTOCOL_SHA256,
                validate_inputs=True,
            )
            assert loaded.implementation_identity == stale_identity
            with pytest.raises(RuntimeError, match="changed while snapshotting"):
                _freeze_manifest(tmp_path / f"fresh-{index}")


def test_source_ledger_detects_added_removed_and_symlink_parent_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.setattr(launcher_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(launcher_module, "_IMPLEMENTATION_SOURCE_ROOTS", ("package",))
    monkeypatch.setattr(launcher_module, "_IMPLEMENTATION_EXPLICIT_PATHS", set())

    initial = build_implementation_identity()
    (package / "added.py").write_text("VALUE = 1\n")
    added = build_implementation_identity()
    assert added != initial
    with pytest.raises(ValueError, match="implementation drifted"):
        launcher_module.validate_current_source_ledger(initial)
    assert [item["path"] for item in added["files"]] == [
        "package/__init__.py",
        "package/added.py",
    ]
    (package / "__init__.py").unlink()
    removed = build_implementation_identity()
    assert removed != added
    with pytest.raises(ValueError, match="implementation drifted"):
        launcher_module.validate_current_source_ledger(added)

    real = tmp_path / "real"
    real.mkdir()
    (real / "binding.json").write_text("{}")
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(launcher_module, "_IMPLEMENTATION_SOURCE_ROOTS", ())
    monkeypatch.setattr(
        launcher_module,
        "_IMPLEMENTATION_EXPLICIT_PATHS",
        {"linked/binding.json"},
    )
    with pytest.raises(ValueError, match="traverses a symlink"):
        build_implementation_identity()


def test_existing_only_uses_the_queue_find_batch_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _freeze_manifest(tmp_path)
    commands = []
    events = []

    def run(command, **arguments):
        events.append("submit")
        commands.append((command, arguments))
        return SimpleNamespace(stdout="existing-batch\n")

    def authenticate(**_):
        events.append("authenticate")
        return {"service": "authenticated"}

    def load_binding(**_):
        events.append("load-binding")
        return {}

    monkeypatch.setattr(launcher_module.subprocess, "run", run)
    monkeypatch.setattr(
        launcher_module,
        "authenticate_training_queue_service",
        authenticate,
    )
    monkeypatch.setattr(
        launcher_module,
        "load_queue_submission_binding",
        load_binding,
    )
    batch_id = submit_execution_manifest(
        manifest_path=manifest_path,
        state_directory=tmp_path / "queue",
        specification_directory=tmp_path / "specifications",
        expected_protocol_sha256=PROTOCOL_SHA256,
        existing_only=True,
    )

    assert batch_id == "existing-batch"
    assert "find-batch" in commands[0][0]
    assert "submit-batch" not in commands[0][0]
    assert events == [
        "authenticate",
        "authenticate",
        "submit",
        "authenticate",
        "load-binding",
    ]


def _write_fake_queue_process_tree(
    proc_root: Path,
    *,
    state_directory: Path,
    daemon_pid: int = 100,
    scheduler_pid: int = 101,
    start_ticks: int = 100,
    scheduler_parent_pid: int | None = None,
    daemon_cwd: Path | None = None,
    scheduler_cwd: Path | None = None,
) -> dict[str, object]:
    scheduler_parent_pid = (
        daemon_pid if scheduler_parent_pid is None else scheduler_parent_pid
    )
    daemon = proc_root / str(daemon_pid)
    scheduler = proc_root / str(scheduler_pid)
    (daemon / "task" / str(daemon_pid)).mkdir(parents=True, exist_ok=True)
    scheduler.mkdir(parents=True, exist_ok=True)
    daemon_fields = ["S", "1", *(["0"] * 17), str(start_ticks)]
    scheduler_fields = [
        "S",
        str(scheduler_parent_pid),
        *(["0"] * 17),
        str(start_ticks + 1),
    ]
    (daemon / "stat").write_text(f"{daemon_pid} (python) {' '.join(daemon_fields)}")
    (scheduler / "stat").write_text(
        f"{scheduler_pid} (bash) {' '.join(scheduler_fields)}"
    )
    service = launcher_module.PROJECT_ROOT / "utils/training_queue/service.py"
    queue_scheduler = (
        launcher_module.PROJECT_ROOT / "utils/training_queue/service_scheduler.sh"
    )
    token = "queue-instance"
    (daemon / "cmdline").write_bytes(
        b"\0".join(
            value.encode()
            for value in (
                sys.executable,
                str(service),
                "--state-dir",
                str(state_directory.resolve()),
                "_serve",
                "--instance-token",
                token,
            )
        )
        + b"\0"
    )
    (scheduler / "cmdline").write_bytes(
        b"\0".join(
            value.encode()
            for value in (
                "bash",
                str(queue_scheduler),
                str(state_directory.resolve()),
            )
        )
        + b"\0"
    )
    (daemon / "cwd").unlink(missing_ok=True)
    (scheduler / "cwd").unlink(missing_ok=True)
    (daemon / "cwd").symlink_to(
        launcher_module.PROJECT_ROOT if daemon_cwd is None else daemon_cwd,
        target_is_directory=True,
    )
    (scheduler / "cwd").symlink_to(
        launcher_module.PROJECT_ROOT if scheduler_cwd is None else scheduler_cwd,
        target_is_directory=True,
    )
    (daemon / "task" / str(daemon_pid) / "children").write_text(str(scheduler_pid))
    newest_source = max(
        (launcher_module.PROJECT_ROOT / path).stat().st_mtime
        for path in launcher_module._QUEUE_RUNTIME_PATHS
    )
    boot_time = int(newest_source) + 10
    (proc_root / "stat").write_text(f"btime {boot_time}\n")
    return {
        "running": True,
        "pid": daemon_pid,
        "pid_start_time": start_ticks,
        "instance_token": token,
    }


def test_queue_authentication_binds_exact_daemon_and_scheduler_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    state_directory = tmp_path / "queue"
    status = _write_fake_queue_process_tree(proc_root, state_directory=state_directory)
    implementation_identity = build_implementation_identity()
    monkeypatch.setattr(
        launcher_module,
        "build_implementation_identity",
        lambda: (_ for _ in ()).throw(AssertionError("unrelated live source scan")),
    )
    identity = authenticate_training_queue_service(
        state_directory=state_directory,
        implementation_identity=implementation_identity,
        proc_root=proc_root,
        status_loader=lambda _: status,
    )

    assert identity["pid"] == 100
    assert identity["scheduler_pid"] == 101
    assert identity["working_directory"] == str(launcher_module.PROJECT_ROOT)
    assert identity["scheduler_working_directory"] == str(
        launcher_module.PROJECT_ROOT
    )
    assert identity["scheduler_cmdline"][1].endswith("service_scheduler.sh")
    assert [item["path"] for item in identity["sources"]] == list(
        launcher_module._QUEUE_RUNTIME_PATHS
    )


def test_queue_authenticates_runtime_before_executing_real_status_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    state_directory = tmp_path / "queue"
    status = _write_fake_queue_process_tree(proc_root, state_directory=state_directory)
    implementation_identity = build_implementation_identity()
    authenticated = set()
    original_sha256 = launcher_module._file_sha256

    def sha256(path: Path) -> str:
        try:
            relative = path.relative_to(launcher_module.PROJECT_ROOT).as_posix()
        except ValueError:
            relative = ""
        if relative in launcher_module._QUEUE_RUNTIME_PATHS:
            authenticated.add(relative)
        return original_sha256(path)

    def run(command, **arguments):
        assert authenticated == set(launcher_module._QUEUE_RUNTIME_PATHS)
        assert command[-2:] == ["status", "--json"]
        return SimpleNamespace(stdout=json.dumps(status))

    monkeypatch.setattr(launcher_module, "_file_sha256", sha256)
    monkeypatch.setattr(launcher_module.subprocess, "run", run)

    identity = authenticate_training_queue_service(
        state_directory=state_directory,
        implementation_identity=implementation_identity,
        proc_root=proc_root,
    )

    assert identity["pid"] == status["pid"]


def test_queue_authentication_rejects_scheduler_reparenting_and_source_staleness(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    state_directory = tmp_path / "queue"
    status = _write_fake_queue_process_tree(
        proc_root,
        state_directory=state_directory,
        scheduler_parent_pid=999,
    )
    with pytest.raises(RuntimeError, match="process identity differs"):
        authenticate_training_queue_service(
            state_directory=state_directory,
            implementation_identity=build_implementation_identity(),
            proc_root=proc_root,
            status_loader=lambda _: status,
        )

    status = _write_fake_queue_process_tree(proc_root, state_directory=state_directory)
    (proc_root / "stat").write_text("btime 0\n")
    with pytest.raises(RuntimeError, match="predates its source"):
        authenticate_training_queue_service(
            state_directory=state_directory,
            implementation_identity=build_implementation_identity(),
            proc_root=proc_root,
            status_loader=lambda _: status,
        )


def test_queue_authentication_rejects_runtime_outside_project_root(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    state_directory = tmp_path / "queue"
    wrong_root = tmp_path / "wrong-root"
    wrong_root.mkdir()
    status = _write_fake_queue_process_tree(
        proc_root,
        state_directory=state_directory,
        daemon_cwd=wrong_root,
    )
    with pytest.raises(RuntimeError, match="process identity differs"):
        authenticate_training_queue_service(
            state_directory=state_directory,
            implementation_identity=build_implementation_identity(),
            proc_root=proc_root,
            status_loader=lambda _: status,
        )

    status = _write_fake_queue_process_tree(
        proc_root,
        state_directory=state_directory,
        scheduler_cwd=wrong_root,
    )
    with pytest.raises(RuntimeError, match="process identity differs"):
        authenticate_training_queue_service(
            state_directory=state_directory,
            implementation_identity=build_implementation_identity(),
            proc_root=proc_root,
            status_loader=lambda _: status,
        )


def test_family_selection_is_rederived_from_complete_compiler_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence_module,
        "PROTOCOL",
        SimpleNamespace(num_items=157_357, evaluation_user_count=2),
    )
    monkeypatch.setattr(
        evidence_module,
        "_load_slice_inputs",
        lambda root, manifest: ({1: 1, 101: 1}, {11: 1, 22: 2}),
    )
    monkeypatch.setattr(
        evidence_module,
        "compute_ranking_slices",
        lambda **arguments: SimpleNamespace(slices=()),
    )
    baseline_manifest_path = _freeze_manifest(
        tmp_path, protocol_sha256=NATIVE500M_PROTOCOL_SHA256
    )
    baseline_batch, baseline_specification = _write_completed_run(
        tmp_path, baseline_manifest_path
    )
    evidence = collect_batch_evidence(
        root=tmp_path,
        manifest_path=baseline_manifest_path,
        batch_specification_path=baseline_specification,
        batch_id=baseline_batch,
        expected_protocol_sha256=NATIVE500M_PROTOCOL_SHA256,
    )
    runs = evidence["runs"]
    evidence_path = tmp_path / "baseline-evidence.json"
    persist_batch_evidence(evidence_path, evidence)
    selection = derive_family_selection(
        family_id="baseline",
        evidence_paths=(evidence_path,),
        root=tmp_path,
    )
    selection_path = tmp_path / "baseline-selection.json"
    persist_family_selection(selection_path, selection)

    assert load_family_selection(selection_path, root=tmp_path) == selection
    assert selection["winner"]["row_id"] == "baseline:01"
    inputs = tuple(
        _write_input_manifest(tmp_path / f"selected-{role}.json", role)
        for role in ("dataset", "content", "features")
    )
    monkeypatch.setattr(
        launcher_module,
        "native500m_input_manifest_references",
        lambda *, root: inputs,
    )
    selected_manifest_path = materialize_selected_execution_manifest(
        tmp_path / "untied.execution.json",
        family_id="untied_control",
        predecessor_selection_path=selection_path,
        root=tmp_path,
    )
    selected_manifest = json.loads(selected_manifest_path.read_text())
    load_execution_manifest(selected_manifest_path, validate_inputs=True)
    assert len(selected_manifest["rows"]) == 9
    assert {row["job"]["predecessor_id"] for row in selected_manifest["rows"]} == {
        selection["winner"]["row_id"]
    }
    assert {
        row["job"]["resolved_representation"]["item_id_tying"]
        for row in selected_manifest["rows"]
    } == {"untied"}

    rq2_manifest_path = materialize_selected_execution_manifest(
        tmp_path / "rq2.execution.json",
        family_id="rq2_content_concat",
        predecessor_selection_path=selection_path,
        root=tmp_path,
    )
    rq2_manifest = json.loads(rq2_manifest_path.read_text())

    def execution_evidence(
        manifest_path: Path, *, winner_index: int, name: str
    ) -> Path:
        batch_id, specification_path = _write_completed_run(
            tmp_path, manifest_path, winner_index=winner_index
        )
        continuation_document = collect_batch_evidence(
            root=tmp_path,
            manifest_path=manifest_path,
            batch_specification_path=specification_path,
            batch_id=batch_id,
            expected_protocol_sha256=NATIVE500M_PROTOCOL_SHA256,
        )
        continuation_path = tmp_path / name
        persist_batch_evidence(continuation_path, continuation_document)
        return continuation_path

    rq2_initial_evidence = execution_evidence(
        rq2_manifest_path, winner_index=0, name="rq2-initial.json"
    )
    rq2_followup_path = materialize_continuation_execution_manifest(
        tmp_path / "rq2-followup.execution.json",
        family_id="rq2_content_concat",
        continuation="followup",
        evidence_paths=(rq2_initial_evidence,),
        predecessor_selection_path=selection_path,
        root=tmp_path,
    )
    rq2_followup = json.loads(rq2_followup_path.read_text())
    assert len(rq2_followup["rows"]) == 3
    assert {row["job"]["stage"] for row in rq2_followup["rows"]} == {
        "capacity_followup"
    }
    load_execution_manifest(rq2_followup_path, validate_inputs=True)

    rq2_followup_evidence = execution_evidence(
        rq2_followup_path, winner_index=-1, name="rq2-followup.json"
    )
    rq2_boundary_path = materialize_continuation_execution_manifest(
        tmp_path / "rq2-boundary.execution.json",
        family_id="rq2_content_concat",
        continuation="boundary",
        evidence_paths=(rq2_initial_evidence, rq2_followup_evidence),
        predecessor_selection_path=selection_path,
        root=tmp_path,
    )
    rq2_boundary = json.loads(rq2_boundary_path.read_text())
    assert len(rq2_boundary["rows"]) >= 3
    assert {row["job"]["stage"] for row in rq2_boundary["rows"]} == {"boundary"}
    load_execution_manifest(rq2_boundary_path, validate_inputs=True)

    with pytest.raises(ValueError, match="exactly the complete initial"):
        materialize_continuation_execution_manifest(
            tmp_path / "rq2-extra.execution.json",
            family_id="rq2_content_concat",
            continuation="followup",
            evidence_paths=(rq2_initial_evidence, rq2_followup_evidence),
            predecessor_selection_path=selection_path,
            root=tmp_path,
        )

    incomplete = json.loads(json.dumps(evidence))
    incomplete["runs"] = incomplete["runs"][:-1]
    incomplete["sha256"] = hashlib.sha256(
        _canonical_bytes(
            {key: value for key, value in incomplete.items() if key != "sha256"}
        )
    ).hexdigest()
    incomplete_path = tmp_path / "incomplete.json"
    persist_batch_evidence(incomplete_path, incomplete)
    with pytest.raises(ValueError, match="authenticated queue artifacts"):
        derive_family_selection(
            family_id="baseline",
            evidence_paths=(incomplete_path,),
            root=tmp_path,
        )

    forged = json.loads(json.dumps(selection))
    forged["winner"] = forged["candidates"][1] | {"job": runs[1]["job"]}
    forged_body = {key: value for key, value in forged.items() if key != "sha256"}
    forged["sha256"] = hashlib.sha256(_canonical_bytes(forged_body)).hexdigest()
    forged_path = tmp_path / "forged-selection.json"
    persist_family_selection(forged_path, forged)
    with pytest.raises(ValueError, match="complete protocol evidence"):
        load_family_selection(forged_path, root=tmp_path)

    boundary_root = tmp_path / "boundary-root"
    boundary_manifest_path = _freeze_manifest(
        boundary_root, protocol_sha256=NATIVE500M_PROTOCOL_SHA256
    )
    boundary_batch, boundary_specification = _write_completed_run(
        boundary_root, boundary_manifest_path, winner_index=2
    )
    boundary = collect_batch_evidence(
        root=boundary_root,
        manifest_path=boundary_manifest_path,
        batch_specification_path=boundary_specification,
        batch_id=boundary_batch,
        expected_protocol_sha256=NATIVE500M_PROTOCOL_SHA256,
    )
    boundary_path = boundary_root / "boundary-initial.json"
    persist_batch_evidence(boundary_path, boundary)
    boundary_inputs = tuple(
        _write_input_manifest(boundary_root / f"selected-{role}.json", role)
        for role in ("dataset", "content", "features")
    )
    monkeypatch.setattr(
        launcher_module,
        "native500m_input_manifest_references",
        lambda *, root: boundary_inputs,
    )
    baseline_boundary_path = materialize_continuation_execution_manifest(
        boundary_root / "baseline-boundary.execution.json",
        family_id="baseline",
        continuation="boundary",
        evidence_paths=(boundary_path,),
        predecessor_selection_path=None,
        root=boundary_root,
    )
    baseline_boundary = json.loads(baseline_boundary_path.read_text())
    assert len(baseline_boundary["rows"]) == 3
    assert {row["job"]["stage"] for row in baseline_boundary["rows"]} == {"boundary"}
    assert all(
        [artifact["role"] for artifact in row["job"]["predecessor_artifacts"]]
        == ["continuation_authorization"]
        for row in baseline_boundary["rows"]
    )
    load_execution_manifest(baseline_boundary_path, validate_inputs=True)
    forged_authorization = json.loads(json.dumps(baseline_boundary))
    for row in forged_authorization["rows"]:
        row["job"]["predecessor_artifacts"][0]["row_id"] = "baseline:forged"
    forged_authorization["sha256"] = hashlib.sha256(
        _canonical_bytes(
            {
                key: value
                for key, value in forged_authorization.items()
                if key != "sha256"
            }
        )
    ).hexdigest()
    forged_authorization_path = boundary_root / "baseline-boundary-forged.json"
    forged_authorization_path.write_bytes(_canonical_bytes(forged_authorization))
    with pytest.raises(ValueError, match="bound source semantic replay failed"):
        load_execution_manifest(forged_authorization_path, validate_inputs=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _full_metrics() -> dict[str, float]:
    values = {}
    for cutoff in (10, 50, 100):
        values |= {
            f"recall@{cutoff}": 1.0,
            f"capped_recall@{cutoff}": 1.0,
            f"ndcg@{cutoff}": 1.0,
            f"mrr@{cutoff}": 1.0,
            f"coverage@{cutoff}": 2 * cutoff / 157_357,
        }
    return values | {"num_users": 2.0}


def _cosine_trace(horizon: int, learning_rate: float) -> list[float]:
    total_steps = horizon * 10
    warmup_steps = int(total_steps * 0.05)
    decay_steps = total_steps - warmup_steps - 1
    values = []
    for epoch in range(1, horizon + 1):
        step = epoch * 10 - 1
        if step < warmup_steps:
            factor = (step + 1) / warmup_steps
        else:
            progress = min(1.0, (step - warmup_steps) / decay_steps)
            factor = (
                0.0 if progress == 1.0 else 0.5 * (1 + math.cos(math.pi * progress))
            )
        values.append(learning_rate * factor)
    return values


def _write_completed_run(
    root: Path, manifest_path: Path, *, winner_index: int = 0
) -> tuple[str, Path]:
    protocol_sha256 = json.loads(manifest_path.read_text())["protocol_sha256"]
    manifest = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=protocol_sha256,
        validate_inputs=True,
    )
    specification = build_batch_specification(
        manifest_path,
        expected_protocol_sha256=protocol_sha256,
    )
    specification_path = persist_batch_specification(
        root / "generated/g3-native500m/batch-specifications", specification
    )
    batch_id = f"batch-{manifest_path.stem}"
    job_ids = [f"{batch_id}-job{index + 1}" for index in range(len(manifest.rows))]
    queue_root = root / "generated/training-queue-service"
    _write_json(
        queue_root / "batches" / f"{batch_id}.json",
        {
            "id": batch_id,
            "jobs": job_ids,
            "sealed": True,
            "atomic_submission": True,
            "expected_job_count": len(job_ids),
            "specification_sha256": specification.sha256,
        },
    )
    persist_queue_submission_binding(
        specification_path=specification_path,
        manifest=manifest,
        batch_id=batch_id,
        queue_service_identity=_synthetic_queue_service_identity(
            root / "generated/training-queue-service"
        ),
    )
    for index, job_id in enumerate(job_ids):
        _write_completed_job(
            root=root,
            manifest=manifest,
            queue_root=queue_root,
            queue_row=specification.document["jobs"][index],
            batch_id=batch_id,
            job_id=job_id,
            job=manifest.rows[index]["job"],
            selection_recall=0.2 if index == winner_index else 0.1,
            selection_ndcg=0.1 if index == winner_index else 0.05,
        )
    return batch_id, specification_path


def _synthetic_queue_service_identity(state_directory: Path) -> dict[str, object]:
    implementation_identity = build_implementation_identity()
    files = {item["path"]: item for item in implementation_identity["files"]}
    project_root = launcher_module._queue_project_root(state_directory)
    body = {
        "schema_version": 2,
        "pid": 100,
        "pid_start_time_ticks": 200,
        "boot_time_unix_seconds": 300,
        "clock_ticks_per_second": 100,
        "instance_token": "test-instance",
        "state_directory": str(state_directory.resolve()),
        "working_directory": str(project_root),
        "cmdline": ["python", "service.py"],
        "scheduler_pid": 101,
        "scheduler_start_time_ticks": 201,
        "scheduler_cmdline": ["bash", "service_scheduler.sh"],
        "scheduler_working_directory": str(project_root),
        "sources": [files[path] for path in launcher_module._QUEUE_RUNTIME_PATHS],
    }
    return {**body, "sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest()}


def _write_completed_job(
    *,
    root: Path,
    manifest: object,
    queue_root: Path,
    queue_row: dict[str, object],
    batch_id: str,
    job_id: str,
    job: dict[str, object],
    selection_recall: float,
    selection_ndcg: float,
) -> None:
    completed = {
        "id": job_id,
        "batch_id": batch_id,
        **queue_row,
        "exit_code": 0,
        "dispatched_at": 0.0,
        "finished_at": 4_000_000_000.0,
    }
    _write_json(queue_root / "completed" / f"{job_id}.json", completed)

    compiled_environment = dict(
        value.split("=", 1) for value in queue_row["environment"]
    )
    for name, value in compiled_environment.items():
        os.environ[name] = value
    try:
        compiled = load_compiled_job(expected_protocol_sha256=manifest.protocol_sha256)
    finally:
        for name in compiled_environment:
            os.environ.pop(name, None)
    run_directory = root / "generated/logs" / str(job["run_name"])
    write_job_contract(compiled, root / "generated/logs")
    horizon = int(job["horizon_epochs"])
    embedding_lr = float(job["embedding_learning_rate"])
    deep_lr = float(job["deep_learning_rate"])
    total_steps = horizon * 10
    metadata = {
        "dataset_size": "500m",
        "g3_dataset_size": "native-500m",
        "seed": 42,
        "batch_size": 512,
        "physical_batch_size": 512,
        "effective_batch_size": 512,
        "num_epochs": horizon,
        "max_epochs": horizon,
        "epochs_trained": horizon,
        "stopped_epoch": horizon,
        "best_epoch": min(7, horizon),
        "early_stopped": False,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "embedding_learning_rate": embedding_lr,
        "deep_learning_rate": deep_lr,
        "lr_schedule_horizon_epochs": horizon,
        "lr_schedule_horizon_steps": total_steps,
        "optimizer_steps_per_epoch": 10,
        "lr_group_traces": {
            "embedding": [embedding_lr] * horizon,
            "deep": _cosine_trace(horizon, deep_lr),
        },
        "g3_execution_identity": manifest.implementation_identity,
        "g3_evaluation_population": manifest.evaluation_population,
        "transfer_invariants": {
            "batch_size": 512,
            "dataset_size": "500m",
            "event_type_filter": "like",
            "eval_max_users": 20_000,
            "evaluation_catalog": "all",
            "exclude_seen_from_evaluation": False,
            "restore_best_weights": True,
            "user_sample": None,
            "lr_schedule": {
                "shape": "cosine",
                "warmup_fraction": 0.05,
                "cycles": 1,
                "min_lr_fraction": 0.0,
                "optimizer_group_scope": "deep_only",
            },
        },
    }
    _write_json(run_directory / "training_metadata.json", metadata)
    _write_json(run_directory / "final_metrics.json", _full_metrics())
    _write_json(
        run_directory / "top_item_rankings.json",
        {
            "schema_version": 1,
            "catalog_sha256": hashlib.sha256(
                json.dumps(list(range(1, 157_358)), separators=(",", ":")).encode()
            ).hexdigest(),
            "catalog_size": 157_357,
            "exclude_seen": False,
            "max_k": 100,
            "rankings": [
                {"user_id": 11, "item_ids": list(range(1, 101))},
                {"user_id": 22, "item_ids": list(range(101, 201))},
            ],
        },
    )
    _write_json(run_directory / "g3_training_diagnostics.json", {"ok": True})
    (run_directory / "sweep.log").write_text(
        f"epoch {min(7, horizon) - 1} finished "
        f"epoch/val_true.recall@100={selection_recall} "
        f"epoch/val_true.ndcg@100={selection_ndcg}\n"
    )
    evidence = RankingEvidence(
        user_ids=torch.tensor([11, 22]),
        history_item_ids=torch.tensor([1, 2]),
        history_offsets=torch.tensor([0, 1, 2]),
        relevant_item_ids=torch.tensor([1, 101]),
        relevance_offsets=torch.tensor([0, 1, 2]),
        relevant_train_frequencies=torch.tensor([3, 4]),
        relevant_ranks=torch.tensor([1, 1]),
        max_k=100,
    )
    context_path = (
        root / "generated/logs/.ranking-evidence/g3-native500m-likes/context.pt"
    )
    write_ranking_evidence(
        evidence,
        context_path=context_path,
        ranking_path=run_directory / "ranking_evidence.pt",
    )
    checkpoint_state = {
        "step": torch.tensor(1.0),
        "weight": torch.tensor([1.0, 2.0]),
    }
    checkpoint_state_sha256 = _state_dict_sha256(checkpoint_state)
    checkpoint_path = run_directory / "restored_best_checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "best_epoch": min(7, horizon),
            "state_sha256": checkpoint_state_sha256,
            "state_dict": checkpoint_state,
        },
        checkpoint_path,
    )
    proof_body = {
        "schema_version": 1,
        "best_epoch": min(7, horizon),
        "checkpoint": _artifact_identity(checkpoint_path),
        "checkpoint_state_sha256": checkpoint_state_sha256,
        "execution_identity_sha256": manifest.implementation_identity["sha256"],
        "evaluation_population": manifest.evaluation_population,
        "final_metrics": _artifact_identity(run_directory / "final_metrics.json"),
        "ranking_evidence": _artifact_identity(run_directory / "ranking_evidence.pt"),
        "top_item_rankings": _artifact_identity(
            run_directory / "top_item_rankings.json"
        ),
    }
    _write_json(
        run_directory / "final_evaluation_proof.json",
        {
            **proof_body,
            "sha256": hashlib.sha256(_canonical_bytes(proof_body)).hexdigest(),
        },
    )


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _artifact_identity(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def test_evidence_collection_authenticates_queue_schedule_ranking_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence_module,
        "PROTOCOL",
        SimpleNamespace(num_items=157_357, evaluation_user_count=2),
    )
    monkeypatch.setattr(
        evidence_module,
        "_load_slice_inputs",
        lambda root, manifest: ({1: 1, 101: 1}, {11: 1, 22: 2}),
    )
    monkeypatch.setattr(
        evidence_module,
        "compute_ranking_slices",
        lambda **arguments: SimpleNamespace(slices=()),
    )
    manifest_path = _freeze_manifest(tmp_path)
    batch_id, specification_path = _write_completed_run(tmp_path, manifest_path)

    evidence = collect_batch_evidence(
        root=tmp_path,
        manifest_path=manifest_path,
        batch_specification_path=specification_path,
        batch_id=batch_id,
        expected_protocol_sha256=PROTOCOL_SHA256,
    )

    assert evidence["data_group"] == DATA_GROUP
    assert evidence["execution_manifest"]["logical_sha256"]
    assert evidence["execution_manifest"]["physical_sha256"]
    assert evidence["batch_specification"]["sha256"]
    assert evidence["queue_submission_binding"]["queue_service_identity_sha256"]
    binding_path = tmp_path / evidence["queue_submission_binding"]["path"]
    binding_bytes = binding_path.read_bytes()
    binding_path.unlink()
    with pytest.raises(ValueError, match="submission binding is absent"):
        collect_batch_evidence(
            root=tmp_path,
            manifest_path=manifest_path,
            batch_specification_path=specification_path,
            batch_id=batch_id,
            expected_protocol_sha256=PROTOCOL_SHA256,
        )
    binding_path.write_bytes(binding_bytes)
    run = evidence["runs"][0]
    assert run["row_id"] == _job()["id"]
    assert run["restored_checkpoint"] == {
        "best_epoch": 7,
        "artifact_sha256": _sha256(
            tmp_path
            / "generated/logs"
            / str(_job()["run_name"])
            / "restored_best_checkpoint.pt"
        ),
        "state_sha256": _state_dict_sha256(
            {
                "step": torch.tensor(1.0),
                "weight": torch.tensor([1.0, 2.0]),
            }
        ),
        "final_evaluation_after_restore": True,
    }
    assert run["full_user_ranking"]["num_users"] == 2
    assert run["full_user_ranking"]["max_k"] == 100
    assert set(run["artifacts"]) == {
        "final_metrics",
        "job_contract",
        "ranking_evidence",
        "sweep_log",
        "top_item_rankings",
        "training_diagnostics",
        "training_metadata",
        "restored_best_checkpoint",
        "final_evaluation_proof",
    }
    output = tmp_path / "evidence.json"
    persist_batch_evidence(output, evidence)
    persist_batch_evidence(output, evidence)
    diagnostic_path = (
        tmp_path
        / "generated/logs"
        / str(_job()["run_name"])
        / "g3_training_diagnostics.json"
    )
    diagnostic_bytes = diagnostic_path.read_bytes()
    collect = evidence_module.collect_batch_evidence
    recollections = 0

    def count_recollection(**arguments):
        nonlocal recollections
        recollections += 1
        return collect(**arguments)

    monkeypatch.setattr(evidence_module, "collect_batch_evidence", count_recollection)
    with pytest.raises(ValueError, match="changed while evidence was verified"):
        with evidence_module._authentication_scope():
            assert (
                load_batch_evidence(
                    output,
                    expected_protocol_sha256=PROTOCOL_SHA256,
                    root=tmp_path,
                )
                == evidence
            )
            assert load_batch_evidence(output, root=tmp_path) == evidence
            assert recollections == 1
            diagnostic_path.write_text('{"memoized-source":"changed"}')
    assert recollections == 1
    diagnostic_path.write_bytes(diagnostic_bytes)
    monkeypatch.setattr(evidence_module, "collect_batch_evidence", collect)

    wrong_batch = json.loads(json.dumps(evidence))
    wrong_batch["queue_batch"]["batch_id"] = "another-batch"
    wrong_batch["sha256"] = hashlib.sha256(
        _canonical_bytes(
            {key: value for key, value in wrong_batch.items() if key != "sha256"}
        )
    ).hexdigest()
    wrong_batch_path = tmp_path / "wrong-batch-evidence.json"
    persist_batch_evidence(wrong_batch_path, wrong_batch)
    with pytest.raises(ValueError, match="another-batch|queue|artifact"):
        load_batch_evidence(
            wrong_batch_path,
            expected_protocol_sha256=PROTOCOL_SHA256,
            root=tmp_path,
        )

    verify_window = evidence_module.verify_artifacts_in_job_window
    mutated = False

    def mutate_after_verification(*arguments, **keywords):
        nonlocal mutated
        verify_window(*arguments, **keywords)
        if not mutated:
            diagnostic_path.write_text('{"changed":true}')
            mutated = True

    monkeypatch.setattr(
        evidence_module,
        "verify_artifacts_in_job_window",
        mutate_after_verification,
    )
    with pytest.raises(ValueError, match="changed while evidence was verified"):
        collect_batch_evidence(
            root=tmp_path,
            manifest_path=manifest_path,
            batch_specification_path=specification_path,
            batch_id=batch_id,
            expected_protocol_sha256=PROTOCOL_SHA256,
        )
    diagnostic_path.write_bytes(diagnostic_bytes)
    monkeypatch.setattr(
        evidence_module, "verify_artifacts_in_job_window", verify_window
    )

    metadata_path = (
        tmp_path / "generated/logs" / str(_job()["run_name"]) / "training_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    metadata["lr_group_traces"]["embedding"][-1] = 0.0
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="constant embedding learning rate"):
        collect_batch_evidence(
            root=tmp_path,
            manifest_path=manifest_path,
            batch_specification_path=specification_path,
            batch_id=batch_id,
            expected_protocol_sha256=PROTOCOL_SHA256,
        )


def test_evidence_rejects_same_count_wrong_users_and_missing_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence_module,
        "PROTOCOL",
        SimpleNamespace(num_items=157_357, evaluation_user_count=2),
    )
    monkeypatch.setattr(
        evidence_module,
        "_load_slice_inputs",
        lambda root, manifest: ({1: 1, 101: 1}, {11: 1, 22: 2, 23: 3}),
    )
    monkeypatch.setattr(
        evidence_module,
        "compute_ranking_slices",
        lambda **arguments: SimpleNamespace(slices=()),
    )
    manifest_path = _freeze_manifest(tmp_path)
    batch_id, specification_path = _write_completed_run(tmp_path, manifest_path)
    run_directory = tmp_path / "generated/logs" / str(_job()["run_name"])
    context_path = (
        tmp_path / "generated/logs/.ranking-evidence/g3-native500m-likes/context.pt"
    )
    ranking_path = run_directory / "ranking_evidence.pt"
    context_path.unlink()
    ranking_path.unlink()
    wrong_users = RankingEvidence(
        user_ids=torch.tensor([11, 23]),
        history_item_ids=torch.tensor([1, 2]),
        history_offsets=torch.tensor([0, 1, 2]),
        relevant_item_ids=torch.tensor([1, 101]),
        relevance_offsets=torch.tensor([0, 1, 2]),
        relevant_train_frequencies=torch.tensor([3, 4]),
        relevant_ranks=torch.tensor([1, 1]),
        max_k=100,
    )
    write_ranking_evidence(
        wrong_users,
        context_path=context_path,
        ranking_path=ranking_path,
    )
    with pytest.raises(ValueError, match="full-user ranking evidence"):
        collect_batch_evidence(
            root=tmp_path,
            manifest_path=manifest_path,
            batch_specification_path=specification_path,
            batch_id=batch_id,
            expected_protocol_sha256=PROTOCOL_SHA256,
        )

    context_path.unlink()
    ranking_path.unlink()
    write_ranking_evidence(
        RankingEvidence(
            user_ids=torch.tensor([11, 22]),
            history_item_ids=torch.tensor([1, 2]),
            history_offsets=torch.tensor([0, 1, 2]),
            relevant_item_ids=torch.tensor([1, 101]),
            relevance_offsets=torch.tensor([0, 1, 2]),
            relevant_train_frequencies=torch.tensor([3, 4]),
            relevant_ranks=torch.tensor([1, 1]),
            max_k=100,
        ),
        context_path=context_path,
        ranking_path=ranking_path,
    )
    (run_directory / "restored_best_checkpoint.pt").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        collect_batch_evidence(
            root=tmp_path,
            manifest_path=manifest_path,
            batch_specification_path=specification_path,
            batch_id=batch_id,
            expected_protocol_sha256=PROTOCOL_SHA256,
        )


def test_native500m_report_derives_rows_winner_bands_and_arithmetic_from_selections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = {
        "baseline",
        "aggregate",
        "untied_control",
        "rq1_content_input",
        "rq2_content_concat",
        "rq3_output_learned",
        "rq3_output_frozen_content",
        "rq3_output_trainable_content",
        "rq3_output_learned_frozen_content",
        "rq3_output_learned_trainable_content",
        "rq4_artist",
        "rq4_album",
        "rq4_artist_album",
        "rq5_global_gate",
        "rq5_frequency_gate",
    }

    def selection(family: str, order: int) -> dict[str, object]:
        validation_recall = 0.12 + order / 10_000
        final_recall = 0.121 + order / 10_000
        return {
            "family_id": family,
            "winner": {
                "row_id": f"{family}:winner",
                "job": {
                    "id": f"{family}:winner",
                    "family_id": family,
                    "manifest_order": order,
                    "predecessor_artifacts": [],
                },
                "selection_metrics": {
                    "recall@100": validation_recall,
                    "ndcg@100": 0.05 + order / 20_000,
                },
                "metrics": {
                    "recall@100": final_recall,
                    "ndcg@100": 0.05 + order / 20_000,
                },
            },
            "predecessor": None,
        }

    documents = {
        family: selection(family, index)
        for index, family in enumerate(sorted(families))
    }
    documents["baseline"] = selection("baseline", 0)
    for document in documents.values():
        document["winner"]["slices"] = {
            "item_frequency": {
                name: {"metrics": {"recall@100": value}}
                for name, value in (("head", 0.2), ("mid", 0.1), ("tail", 0.01))
            },
            "user_history": {
                name: {"metrics": {"recall@100": value}}
                for name, value in (("low", 0.1), ("mid", 0.12), ("high", 0.14))
            },
        }
    documents["untied_control"]["winner"]["metrics"]["recall@100"] = 0.140
    documents["rq1_content_input"]["winner"]["selection_metrics"]["recall@100"] = 0.150
    documents["rq1_content_input"]["winner"]["metrics"]["recall@100"] = 0.130
    documents["aggregate"]["winner"]["metrics"] = {
        "recall@100": 0.151,
        "ndcg@100": 0.060,
    }
    input_winner = documents["rq2_content_concat"]["winner"]
    documents["aggregate"]["winner"]["job"]["predecessor_artifacts"] = [
        {
            "role": "aggregate_input",
            "path": "rq2_content_concat.json",
            "row_id": input_winner["row_id"],
        }
    ]
    monkeypatch.setattr(
        report_module,
        "load_family_selection",
        lambda path: documents[Path(path).stem],
    )
    rendered = render_native500m_reports(
        selection_paths={family: Path(f"{family}.json") for family in families},
        conclusions={
            number: (
                f"Selection: hand-written RQ{number} decision.",
                f"Observed result: hand-written RQ{number} interpretation.",
                f"Conclusion: hand-written RQ{number} outcome.",
            )
            for number in range(1, 6)
        },
    )

    for number in range(1, 6):
        assert f"## RQ{number}:" in rendered.reader
    assert rendered.reader.index(
        "two-layer G1-best tied baseline"
    ) < rendered.reader.index("normalized frozen content")
    assert "**normalized frozen content**" in rendered.reader
    assert "**untied learned-ID control**" not in rendered.reader
    assert "secondary mechanism control" in rendered.reader
    rq5 = rendered.reader.split("## RQ5:", 1)[1].split("## Aggregated improvement", 1)[
        0
    ]
    assert "item ID + normalized content" in rq5
    assert "one learned scalar controls" in rq5
    assert "Item-frequency Recall@100" in rq5
    assert "History-length Recall@100" in rq5
    for number in range(1, 6):
        section = rendered.reader.split(f"## RQ{number}:", 1)[1]
        if number < 5:
            section = section.split(f"## RQ{number + 1}:", 1)[0]
        else:
            section = section.split("## Aggregated improvement", 1)[0]
        assert section.count("Selection:") == 1
        assert section.count("Observed result:") == 1
        assert section.count("Conclusion:") == 1
    assert "## Aggregated improvement" in rendered.reader
    assert "resolution band" in rendered.reader
    assert "0.030000" in rendered.reader
    forbidden = ("runtime", "throughput", "GPU memory", "parameters", "embedding lr")
    assert not any(value in rendered.reader for value in forbidden)


def test_g3_inherits_the_exact_full_ranking_snapshot_emitter() -> None:
    assert issubclass(G3GenerationExperiment, G4GenerationExperiment)


def test_compatibility_report_reuses_final_natural_bridge_without_duplicate_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def row(family: str, recall: float, ndcg: float):
        return {
            "row_id": f"{family}:winner",
            "job": {"family_id": family},
            "metrics": {"recall@100": recall, "ndcg@100": ndcg},
        }

    documents = {
        "baseline": {"winner": row("baseline", 0.10, 0.05)},
        "input": {"winner": row("rq1_content_input", 0.12, 0.055)},
        "learned": {"winner": row("rq3_output_learned", 0.11, 0.052)},
        "bridge": {"winner": row("bridge_rq3_output", 0.13, 0.058)},
    }
    initial = {
        "prior_state": None,
        "component_targets": {
            "input": {"path": "input.json", "row_id": "rq1_content_input:winner"},
            "output": {"path": "output.json", "row_id": "rq3:winner"},
            "metadata": None,
        },
        "included": {
            "input": {"path": "input.json", "row_id": "rq1_content_input:winner"},
            "output": None,
            "metadata": None,
        },
        "standalone_selections": {"rq3_output_learned": {"path": "learned.json"}},
    }
    final = {
        "prior_state": {"path": "state0.json"},
        "completed_transition": {
            "decision": "accept",
            "selected_selection": {
                "path": "bridge.json",
                "row_id": "bridge_rq3_output:winner",
            },
            "predecessor_reference": {
                "path": "input.json",
                "row_id": "rq1_content_input:winner",
            },
        },
        "most_specific_selection": {
            "path": "bridge.json",
            "row_id": "bridge_rq3_output:winner",
        },
    }
    monkeypatch.setattr(
        report_module,
        "load_compatibility_resolution",
        lambda path: initial if Path(path).stem == "state0" else final,
    )
    monkeypatch.setattr(
        report_module,
        "load_family_selection",
        lambda path: documents[Path(path).stem],
    )
    table = report_module._compatibility_aggregate_table(
        baseline=documents["baseline"]["winner"],
        final_state_path=Path("state1.json"),
        thresholds={"recall@100": 0.001, "ndcg@100": 0.001},
    )
    assert "0.130000" in table
    assert "+0.030000" in table
    assert "unresolved" in table


def test_report_promotion_gates_use_baseline_tail_band_and_atomic_rq5() -> None:
    def row(family: str, recall: float, tail: float, ndcg: float, order: int):
        return {
            "job": {"family_id": family, "manifest_order": order},
            "selection_metrics": {"recall@100": recall, "ndcg@100": ndcg},
            "metrics": {"recall@100": recall, "ndcg@100": ndcg},
            "slices": {"item_frequency": {"tail": {"metrics": {"recall@100": tail}}}},
        }

    thresholds = {"recall@100": 0.01, "ndcg@100": 0.01}
    rq1 = [
        row("baseline", 0.10, 0.020, 0.05, 0),
        row("untied_control", 0.10, 0.001, 0.05, 0),
        row("rq1_content_input", 0.095, 0.022, 0.05, 0),
    ]
    assert (
        report_module._approved_winner_index(
            1,
            rq1,
            ("baseline", "untied_control", "rq1_content_input"),
            thresholds,
            0.002,
        )
        == 0
    )
    rq1[2]["slices"]["item_frequency"]["tail"]["metrics"]["recall@100"] = 0.022001
    assert (
        report_module._approved_winner_index(
            1,
            rq1,
            ("baseline", "untied_control", "rq1_content_input"),
            thresholds,
            0.002,
        )
        == 2
    )

    rq5 = [
        row("baseline", 0.100, 0.020, 0.050, 0),
        row("rq2_content_concat", 0.099, 0.020, 0.050, 0),
        row("rq5_global_gate", 0.112, 0.024, 0.051, 0),
        row("rq5_frequency_gate", 0.111, 0.027, 0.052, 0),
    ]
    assert (
        report_module._approved_winner_index(
            5,
            rq5,
            (
                "baseline",
                "rq2_content_concat",
                "rq5_global_gate",
                "rq5_frequency_gate",
            ),
            thresholds,
            0.002,
        )
        == 3
    )
    rq5[3]["slices"]["item_frequency"]["tail"]["metrics"]["recall@100"] = 0.025
    assert (
        report_module._approved_winner_index(
            5,
            rq5,
            (
                "baseline",
                "rq2_content_concat",
                "rq5_global_gate",
                "rq5_frequency_gate",
            ),
            thresholds,
            0.002,
        )
        == 2
    )


def test_compatibility_resolution_supports_no_treatment_without_aggregate_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = {
        "baseline",
        "untied_control",
        "rq1_content_input",
        "rq2_content_concat",
        "rq3_output_learned",
        "rq3_output_frozen_content",
        "rq3_output_trainable_content",
        "rq3_output_learned_frozen_content",
        "rq3_output_learned_trainable_content",
        "rq4_artist",
        "rq4_album",
        "rq4_artist_album",
        "rq5_global_gate",
        "rq5_frequency_gate",
    }
    documents = {}
    paths = {}
    for family in families:
        winner = {
            "row_id": f"{family}:winner",
            "job": {"family_id": family, "manifest_order": 0},
            "selection_metrics": {"recall@100": 0.1, "ndcg@100": 0.05},
            "metrics": {"recall@100": 0.1, "ndcg@100": 0.05},
            "slices": {"item_frequency": {"tail": {"metrics": {"recall@100": 0.02}}}},
        }
        body = {
            "schema_version": 1,
            "kind": "g3_native500m_family_selection",
            "protocol_sha256": NATIVE500M_PROTOCOL_SHA256,
            "family_id": family,
            "winner": winner,
        }
        document = {
            **body,
            "sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
        }
        path = tmp_path / f"{family}.json"
        path.write_bytes(_canonical_bytes(document))
        documents[family] = document
        paths[family] = path
    monkeypatch.setattr(
        evidence_module,
        "authenticate_family_selection",
        lambda path, root: (documents[Path(path).stem], object()),
    )
    resolution = derive_compatibility_resolution(selection_paths=paths, root=tmp_path)
    assert resolution["next_conditional_family"] is None
    assert resolution["most_specific_selection"]["row_id"] == "baseline:winner"
    assert resolution["included"]["output"] is None
    assert resolution["included"]["metadata"] is None
    assert len(resolution["standalone_decisions"]) == len(families)
    path = tmp_path / "compatibility.json"
    persist_compatibility_resolution(path, resolution)
    assert load_compatibility_resolution(path, root=tmp_path) == resolution
    forged = json.loads(json.dumps(resolution))
    forged["standalone_decisions"].pop()
    forged_body = {key: value for key, value in forged.items() if key != "sha256"}
    forged["sha256"] = hashlib.sha256(_canonical_bytes(forged_body)).hexdigest()
    forged_path = tmp_path / "forged-compatibility.json"
    persist_compatibility_resolution(forged_path, forged)
    with pytest.raises(ValueError, match="differs from selected evidence"):
        load_compatibility_resolution(forged_path, root=tmp_path)


def test_compatibility_transitions_authenticate_accept_and_omit_advancement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        CandidateResult,
        authenticate_resolved_conditional_predecessor,
        authenticate_selected_coordinate,
        compile_nine_cell_family,
        family_spec,
        required_boundary_extensions,
        select_preliminary_winner,
    )

    families = {
        "baseline",
        "untied_control",
        "rq1_content_input",
        "rq2_content_concat",
        "rq3_output_learned",
        "rq3_output_frozen_content",
        "rq3_output_trainable_content",
        "rq3_output_learned_frozen_content",
        "rq3_output_learned_trainable_content",
        "rq4_artist",
        "rq4_album",
        "rq4_artist_album",
        "rq5_global_gate",
        "rq5_frequency_gate",
    }

    def authenticated(rows, predecessor=None):
        results = tuple(
            CandidateResult(
                row=row,
                recall_at_100=0.2 if index == 0 else 0.1,
                ndcg_at_100=0.1 if index == 0 else 0.05,
                best_epoch=1,
                epochs_trained=row.horizon_epochs,
            )
            for index, row in enumerate(rows)
        )
        return authenticate_selected_coordinate(
            results, expected_rows=rows, predecessor=predecessor
        )

    baseline_rows = compile_baseline_rows()
    baseline_authenticated = authenticated(baseline_rows)
    rq1_rows = compile_nine_cell_family(
        family_spec("rq1_content_input"), baseline_authenticated.coordinate
    )
    rq1_authenticated = authenticated(rq1_rows, baseline_authenticated)
    scores = {family: 0.1 for family in families}
    scores |= {
        "rq1_content_input": 0.14,
        "rq2_content_concat": 0.11,
        "rq3_output_learned": 0.11,
        "rq3_output_frozen_content": 0.13,
        "rq3_output_trainable_content": 0.125,
        "rq4_artist": 0.13,
        "rq4_album": 0.125,
    }
    documents = {}
    authentications = {}
    paths = {}
    for family in families:
        row_id = (
            rq1_authenticated.selected_result.row.id
            if family == "rq1_content_input"
            else f"{family}:winner"
        )
        if family == "rq1_content_input":
            representation = G3Representation(
                history_representation="content"
            ).to_dict()
        elif family.startswith("rq3_output_"):
            catalog = (
                "learned_id" if family == "rq3_output_learned" else "frozen_content"
            )
            representation = G3Representation(
                history_representation="id_content",
                history_hidden_dim=64,
                catalog_representation=catalog,
                item_id_tying="untied",
            ).to_dict()
        elif family == "rq2_content_concat":
            representation = G3Representation(
                history_representation="id_content",
                history_hidden_dim=64,
                item_id_tying="tied",
            ).to_dict()
        elif family == "rq4_artist":
            representation = G3Representation(
                metadata=("artist",), metadata_dim=32, item_id_tying="tied"
            ).to_dict()
        else:
            representation = G3Representation(item_id_tying="tied").to_dict()
        winner = {
            "row_id": row_id,
            "job": {
                "family_id": family,
                "manifest_order": 0,
                "resolved_representation": representation,
            },
            "selection_metrics": {"recall@100": scores[family], "ndcg@100": 0.05},
            "metrics": {"recall@100": scores[family], "ndcg@100": 0.05},
            "slices": {"item_frequency": {"tail": {"metrics": {"recall@100": 0.02}}}},
        }
        body = {
            "schema_version": 1,
            "kind": "g3_native500m_family_selection",
            "protocol_sha256": NATIVE500M_PROTOCOL_SHA256,
            "family_id": family,
            "winner": winner,
        }
        document = {
            **body,
            "sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
        }
        path = tmp_path / f"{family}.json"
        path.write_bytes(_canonical_bytes(document))
        documents[path.name] = document
        authentications[path.name] = (
            rq1_authenticated
            if family == "rq1_content_input"
            else baseline_authenticated
        )
        paths[family] = path

    def authenticate(path, root):
        del root
        name = Path(path).name
        return documents[name], authentications[name]

    monkeypatch.setattr(evidence_module, "authenticate_family_selection", authenticate)
    initial = derive_compatibility_resolution(selection_paths=paths, root=tmp_path)
    assert initial["next_conditional_family"] == "bridge_rq3_output"
    decisions = {
        decision["family_id"]: decision for decision in initial["standalone_decisions"]
    }
    assert decisions["rq3_output_trainable_content"]["eligible"] is True
    assert (
        decisions["rq3_output_trainable_content"]["reason"] == "eligible_but_superseded"
    )
    assert decisions["rq4_album"]["eligible"] is True
    assert decisions["rq4_album"]["reason"] == "eligible_but_superseded"
    initial_path = tmp_path / "compatibility-0.json"
    persist_compatibility_resolution(initial_path, initial)
    _, initial_authenticated = authenticate_compatibility_resolution(
        initial_path, root=tmp_path
    )
    inputs = tuple(
        _write_input_manifest(tmp_path / f"conditional-{role}.json", role)
        for role in ("dataset", "content", "features")
    )
    monkeypatch.setattr(
        launcher_module,
        "native500m_input_manifest_references",
        lambda *, root: inputs,
    )
    conditional_manifest_path = materialize_conditional_execution_manifest(
        tmp_path / "bridge.execution.json",
        family_id="bridge_rq3_output",
        compatibility_state_path=initial_path,
        root=tmp_path,
    )
    conditional_manifest = json.loads(conditional_manifest_path.read_text())
    assert len(conditional_manifest["rows"]) == 9
    forged_conditional = json.loads(json.dumps(conditional_manifest))
    for row in forged_conditional["rows"]:
        row["job"]["resolved_representation"]["catalog_representation"] = "learned_id"
    forged_conditional["sha256"] = hashlib.sha256(
        _canonical_bytes(
            {key: value for key, value in forged_conditional.items() if key != "sha256"}
        )
    ).hexdigest()
    forged_conditional_path = tmp_path / "bridge-forged.execution.json"
    forged_conditional_path.write_bytes(_canonical_bytes(forged_conditional))
    with pytest.raises(ValueError, match="bound source semantic replay failed"):
        load_execution_manifest(forged_conditional_path, validate_inputs=True)
    monkeypatch.setattr(
        evidence_module,
        "PROTOCOL",
        SimpleNamespace(num_items=157_357, evaluation_user_count=2),
    )
    monkeypatch.setattr(
        evidence_module,
        "_load_slice_inputs",
        lambda root, manifest: ({1: 1, 101: 1}, {11: 1, 22: 2}),
    )
    monkeypatch.setattr(
        evidence_module,
        "compute_ranking_slices",
        lambda **arguments: SimpleNamespace(slices=()),
    )

    def conditional_evidence(winner_index: int, name: str) -> Path:
        batch_id, specification_path = _write_completed_run(
            tmp_path, conditional_manifest_path, winner_index=winner_index
        )
        document = collect_batch_evidence(
            root=tmp_path,
            manifest_path=conditional_manifest_path,
            batch_specification_path=specification_path,
            batch_id=batch_id,
            expected_protocol_sha256=NATIVE500M_PROTOCOL_SHA256,
        )
        path = tmp_path / name
        persist_batch_evidence(path, document)
        return path

    interior_evidence = conditional_evidence(0, "bridge-interior.json")
    with pytest.raises(ValueError, match="interior.*no continuation"):
        materialize_conditional_boundary_execution_manifest(
            tmp_path / "bridge-interior-boundary.execution.json",
            family_id="bridge_rq3_output",
            evidence_paths=(interior_evidence,),
            compatibility_state_path=initial_path,
            root=tmp_path,
        )
    resolved_for_boundary = authenticate_resolved_conditional_predecessor(
        target_family_id="bridge_rq3_output",
        compatibility_state=initial_authenticated,
    )
    compiled_for_boundary = compile_nine_cell_family(
        family_spec("bridge_rq3_output"), resolved_for_boundary.coordinate
    )
    boundary_index = next(
        index
        for index in range(len(compiled_for_boundary))
        if required_boundary_extensions(
            select_preliminary_winner(
                tuple(
                    CandidateResult(
                        row=row,
                        recall_at_100=0.2 if row_index == index else 0.1,
                        ndcg_at_100=0.1 if row_index == index else 0.05,
                        best_epoch=1,
                        epochs_trained=row.horizon_epochs,
                    )
                    for row_index, row in enumerate(compiled_for_boundary)
                ),
                expected_rows=compiled_for_boundary,
                predecessor=resolved_for_boundary,
            ),
            compiled_for_boundary,
        )
    )
    boundary_evidence = conditional_evidence(boundary_index, "bridge-boundary.json")
    original_state_bytes = initial_path.read_bytes()
    original_authenticate_state = evidence_module.authenticate_compatibility_resolution

    def mutate_after_authentication(path, *, root):
        result = original_authenticate_state(path, root=root)
        initial_path.write_text("{}")
        return result

    monkeypatch.setattr(
        evidence_module,
        "authenticate_compatibility_resolution",
        mutate_after_authentication,
    )
    with pytest.raises(ValueError, match="changed|identity"):
        materialize_conditional_execution_manifest(
            tmp_path / "bridge-toctou.execution.json",
            family_id="bridge_rq3_output",
            compatibility_state_path=initial_path,
            root=tmp_path,
        )
    initial_path.write_bytes(original_state_bytes)
    with pytest.raises(ValueError, match="changed|identity|artifact differs"):
        materialize_conditional_boundary_execution_manifest(
            tmp_path / "bridge-boundary-toctou.execution.json",
            family_id="bridge_rq3_output",
            evidence_paths=(boundary_evidence,),
            compatibility_state_path=initial_path,
            root=tmp_path,
        )
    initial_path.write_bytes(original_state_bytes)
    monkeypatch.setattr(
        evidence_module,
        "authenticate_compatibility_resolution",
        original_authenticate_state,
    )
    boundary_manifest_path = materialize_conditional_boundary_execution_manifest(
        tmp_path / "bridge-boundary.execution.json",
        family_id="bridge_rq3_output",
        evidence_paths=(boundary_evidence,),
        compatibility_state_path=initial_path,
        root=tmp_path,
    )
    boundary_manifest = json.loads(boundary_manifest_path.read_text())
    assert boundary_manifest["rows"]
    assert {row["job"]["stage"] for row in boundary_manifest["rows"]} == {"boundary"}
    load_execution_manifest(boundary_manifest_path, validate_inputs=True)
    boundary_batch, boundary_specification = _write_completed_run(
        tmp_path, boundary_manifest_path, winner_index=0
    )
    boundary_results = collect_batch_evidence(
        root=tmp_path,
        manifest_path=boundary_manifest_path,
        batch_specification_path=boundary_specification,
        batch_id=boundary_batch,
        expected_protocol_sha256=NATIVE500M_PROTOCOL_SHA256,
    )
    boundary_results_path = tmp_path / "bridge-boundary-results.json"
    persist_batch_evidence(boundary_results_path, boundary_results)
    with pytest.raises(
        ValueError, match="missing.*boundary|missing a compiler-derived row"
    ):
        derive_conditional_family_selection(
            family_id="bridge_rq3_output",
            evidence_paths=(boundary_evidence,),
            compatibility_state_path=initial_path,
            root=tmp_path,
        )
    conditional_selection = derive_conditional_family_selection(
        family_id="bridge_rq3_output",
        evidence_paths=(boundary_evidence, boundary_results_path),
        compatibility_state_path=initial_path,
        root=tmp_path,
    )
    assert set(conditional_selection["expected_row_ids"]) == {
        *(row["job"]["id"] for row in conditional_manifest["rows"]),
        *(row["job"]["id"] for row in boundary_manifest["rows"]),
    }
    resolved_output = authenticate_resolved_conditional_predecessor(
        target_family_id="bridge_rq3_output",
        compatibility_state=initial_authenticated,
    )
    bridge_rows = compile_nine_cell_family(
        family_spec("bridge_rq3_output"), resolved_output.coordinate
    )
    bridge_authenticated = authenticated(bridge_rows, resolved_output)

    def conditional_document(name: str, recall: float, auth) -> Path:
        row = auth.selected_result.row
        winner = {
            "row_id": row.id,
            "job": row.to_dict()
            | {
                "resolved_representation": G3Representation(
                    history_representation="content",
                    catalog_representation="frozen_content",
                    item_id_tying="untied",
                ).to_dict()
            },
            "selection_metrics": {"recall@100": recall, "ndcg@100": 0.06},
            "metrics": {"recall@100": recall, "ndcg@100": 0.06},
            "slices": {"item_frequency": {"tail": {"metrics": {"recall@100": 0.02}}}},
        }
        body = {
            "schema_version": 1,
            "kind": "g3_native500m_family_selection",
            "protocol_sha256": NATIVE500M_PROTOCOL_SHA256,
            "family_id": row.family_id,
            "compatibility_state": evidence_module._selection_file_reference(
                initial_path,
                initial,
                "compatibility_state",
                root=tmp_path,
            ),
            "winner": winner,
        }
        document = {
            **body,
            "sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
        }
        path = tmp_path / name
        path.write_bytes(_canonical_bytes(document))
        documents[path.name] = document
        authentications[path.name] = auth
        return path

    accepted_selection = conditional_document(
        "bridge-accepted.json", 0.16, bridge_authenticated
    )
    accepted = derive_compatibility_transition(
        prior_resolution_path=initial_path,
        completed_selection_path=accepted_selection,
        root=tmp_path,
    )
    assert accepted["completed_transition"]["decision"] == "accept"
    assert accepted["next_conditional_family"] == "bridge_rq4_metadata"
    accepted_path = tmp_path / "compatibility-accepted.json"
    persist_compatibility_resolution(accepted_path, accepted)
    authenticate_compatibility_resolution(accepted_path, root=tmp_path)

    omitted_selection = conditional_document(
        "bridge-omitted.json", 0.14, bridge_authenticated
    )
    omitted = derive_compatibility_transition(
        prior_resolution_path=initial_path,
        completed_selection_path=omitted_selection,
        root=tmp_path,
    )
    assert omitted["completed_transition"]["decision"] == "omit"
    assert omitted["next_conditional_family"] == "bridge_rq4_metadata"
    omitted_path = tmp_path / "compatibility-omitted.json"
    persist_compatibility_resolution(omitted_path, omitted)
    _, omitted_authenticated = authenticate_compatibility_resolution(
        omitted_path, root=tmp_path
    )
    resolved_metadata = authenticate_resolved_conditional_predecessor(
        target_family_id="bridge_rq4_metadata",
        compatibility_state=omitted_authenticated,
    )
    metadata_rows = compile_nine_cell_family(
        family_spec("bridge_rq4_metadata"), resolved_metadata.coordinate
    )
    metadata_authenticated = authenticated(metadata_rows, resolved_metadata)
    metadata_path = conditional_document(
        "metadata-omitted.json", 0.14, metadata_authenticated
    )
    metadata_document = documents[metadata_path.name]
    metadata_document["compatibility_state"] = (
        evidence_module._selection_file_reference(
            omitted_path, omitted, "compatibility_state", root=tmp_path
        )
    )
    metadata_body = {
        key: value for key, value in metadata_document.items() if key != "sha256"
    }
    metadata_document["sha256"] = hashlib.sha256(
        _canonical_bytes(metadata_body)
    ).hexdigest()
    metadata_path.write_bytes(_canonical_bytes(metadata_document))
    terminal = derive_compatibility_transition(
        prior_resolution_path=omitted_path,
        completed_selection_path=metadata_path,
        root=tmp_path,
    )
    assert terminal["completed_transition"]["decision"] == "omit"
    assert terminal["next_conditional_family"] is None
    terminal_path = tmp_path / "compatibility-terminal.json"
    persist_compatibility_resolution(terminal_path, terminal)
    authenticate_compatibility_resolution(terminal_path, root=tmp_path)


def test_final_evaluation_bundle_reuses_identical_retry_and_rejects_changed_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"weight": torch.tensor([1.0, 2.0])}
    state_sha256 = model_state_dict_sha256(state)
    calls = []

    def evaluate() -> None:
        calls.append("evaluate")
        (tmp_path / "final_metrics.json").write_text("{}")
        (tmp_path / "ranking_evidence.pt").write_bytes(b"ranking")
        (tmp_path / "top_item_rankings.json").write_text("{}")

    arguments = {
        "run_directory": tmp_path,
        "best_epoch": 7,
        "state": state,
        "state_sha256": state_sha256,
        "execution_identity": build_implementation_identity(),
        "evaluation_population": TEST_POPULATION,
        "evaluate": evaluate,
    }
    _publish_final_evaluation_bundle(**arguments)
    _publish_final_evaluation_bundle(**arguments)

    assert calls == ["evaluate"]
    original_proof = (tmp_path / "final_evaluation_proof.json").read_bytes()
    changed = {"weight": torch.tensor([3.0, 4.0])}
    with pytest.raises(RuntimeError, match="proof differs"):
        _publish_final_evaluation_bundle(
            **arguments
            | {
                "state": changed,
                "state_sha256": model_state_dict_sha256(changed),
            }
        )
    assert (tmp_path / "final_evaluation_proof.json").read_bytes() == original_proof
    assert calls == ["evaluate"]
    monkeypatch.setattr(
        launcher_module,
        "validate_current_source_ledger",
        lambda _: (_ for _ in ()).throw(ValueError("implementation drifted")),
    )
    with pytest.raises(ValueError, match="implementation drifted"):
        _publish_final_evaluation_bundle(**arguments)
    assert (tmp_path / "final_evaluation_proof.json").read_bytes() == original_proof


def test_state_dict_sha256_accepts_scalar_tensors() -> None:
    state = {
        "scalar": torch.tensor(1.0),
        "vector": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
    }
    digest = model_state_dict_sha256(state)

    assert digest == model_state_dict_sha256(state)
    assert digest == evidence_module._state_dict_sha256(state)
    assert digest != model_state_dict_sha256(state | {"scalar": torch.tensor(2.0)})


def test_metric_reconciliation_accepts_only_machine_roundoff() -> None:
    assert evidence_module._same_metric(0.1, 0.1 + 3e-15)
    assert not evidence_module._same_metric(0.1, 0.1 + 1e-12)


def test_final_evaluation_bundle_serializes_concurrent_publication(
    tmp_path: Path,
) -> None:
    state = {"weight": torch.tensor([1.0, 2.0])}
    entered = Event()
    release = Event()
    calls = []

    def evaluate() -> None:
        calls.append("evaluate")
        entered.set()
        assert release.wait(timeout=5)
        (tmp_path / "final_metrics.json").write_text("{}")
        (tmp_path / "ranking_evidence.pt").write_bytes(b"ranking")
        (tmp_path / "top_item_rankings.json").write_text("{}")

    arguments = {
        "run_directory": tmp_path,
        "best_epoch": 7,
        "state": state,
        "state_sha256": model_state_dict_sha256(state),
        "execution_identity": build_implementation_identity(),
        "evaluation_population": TEST_POPULATION,
        "evaluate": evaluate,
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_publish_final_evaluation_bundle, **arguments)
        assert entered.wait(timeout=5)
        second = executor.submit(_publish_final_evaluation_bundle, **arguments)
        release.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert calls == ["evaluate"]
    proof = json.loads((tmp_path / "final_evaluation_proof.json").read_text())
    assert proof["checkpoint_state_sha256"] == arguments["state_sha256"]


def test_final_evaluation_refuses_proof_when_source_drifts_during_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"weight": torch.tensor([1.0, 2.0])}
    calls = []

    def validate(_: object) -> dict[str, object]:
        calls.append("validate")
        if calls.count("validate") == 2:
            raise ValueError("execution implementation drifted")
        return {}

    def evaluate() -> None:
        calls.append("evaluate")
        (tmp_path / "final_metrics.json").write_text("{}")
        (tmp_path / "ranking_evidence.pt").write_bytes(b"ranking")
        (tmp_path / "top_item_rankings.json").write_text("{}")

    monkeypatch.setattr(launcher_module, "validate_current_source_ledger", validate)
    with pytest.raises(ValueError, match="implementation drifted"):
        _publish_final_evaluation_bundle(
            run_directory=tmp_path,
            best_epoch=7,
            state=state,
            state_sha256=model_state_dict_sha256(state),
            execution_identity=build_implementation_identity(),
            evaluation_population=TEST_POPULATION,
            evaluate=evaluate,
        )

    assert calls == ["validate", "evaluate", "validate"]
    assert not (tmp_path / "final_evaluation_proof.json").exists()
