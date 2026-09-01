from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from experiments.g6_rqkmeans_history.native500m.launchers import materialize
from experiments.g6_rqkmeans_history.native500m.analysis import (
    collect as collect_analysis,
)
from experiments.g6_rqkmeans_history.native500m.configs.runtime import build_control
from experiments.g6_rqkmeans_history.native500m.analysis.collect import (
    candidate_selection_group,
    collect_stage_candidates,
    reuse_source_is_eligible,
)
from experiments.g6_rqkmeans_history.native500m.analysis.report import (
    _load_selection_ledger,
)
from experiments.g6_rqkmeans_history.native500m.launchers.queue import (
    batch_shard_job_ids,
    build_batch_specification,
    canonical_bytes,
    load_admitted_queue_manifest,
    load_queue_manifest,
    QueueJob,
    resolve_or_submit_batch,
)
from experiments.g6_rqkmeans_history.native500m.launchers.materialize import (
    build_cachefix_rq1_confirmation_manifest,
    build_rq23_confirmation_manifest,
    build_rq0_first_surface_manifest,
    build_controls_manifest,
    compile_and_persist_stage,
    persist_stage_manifest,
    resolve_rq23_confirmation_reuse,
)
from experiments.g6_rqkmeans_history.native500m.launchers.runtime import (
    build_experiment,
    experiment_logical_sha256,
    load_runtime_job,
    source_identity_sha256,
)
from experiments.g6_rqkmeans_history.native500m.protocol.contracts import (
    ExactReuse,
    JobContract,
    SelectionBinding,
    StageManifest,
)


PLAN_SHA256 = "3561064c58087cff75b0029c62eb477104b8bce51ff77b4868f3733d0b218910"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_queue_job_exposes_a_validated_seed() -> None:
    job = QueueJob(
        job_id="job",
        run_name="run",
        runner="runner.py",
        config_logical_sha256="1" * 64,
        data_group="data",
        logical_sha256="2" * 64,
        payload={"parameters": {"seed": 42}},
        environment={},
    )

    assert job.seed == 42
    with pytest.raises(ValueError, match="seed"):
        replace(job, payload={"parameters": {"seed": True}}).seed


@pytest.mark.parametrize("retry_revision", [-1, True, 1.5])
def test_cachefix_confirmation_rejects_invalid_retry_revision(
    retry_revision: object,
) -> None:
    with pytest.raises(ValueError, match="retry revision"):
        build_cachefix_rq1_confirmation_manifest(retry_revision=retry_revision)


def test_cachefix_confirmation_retry_revision_is_replayable_and_science_neutral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = SelectionBinding("rq1_surface", "1" * 64, True)
    reuse = ExactReuse("source", "2" * 64, ("seed",))
    parameters = {
        "builder": "semantic",
        "runner": materialize.RUNNER,
        "run_name": "confirmation_random_42",
        "config_logical_sha256": "3" * 64,
        "data_group": "confirmation-random-42",
        "environment": {
            "G6_NATIVE500M_SOURCE_SHA256": "4" * 64,
            "G6_NATIVE500M_TOKENIZER_BINDING_REVISION": "shared-base-v2",
            "G6_NATIVE500M_TOKENIZER_REGISTRY_SHA256": "5" * 64,
            "G6_NATIVE500M_TOKENIZER_BASE_CACHE_KEY": "base",
            "G6_NATIVE500M_TOKENIZER_FIT_SHA256": "6" * 64,
            "G6_NATIVE500M_TOKENIZER_CODES_SHA256": "7" * 64,
            "G6_NATIVE500M_TOKENIZER_MATERIALIZATION_SHA256": "8" * 64,
        },
        "backbone": "best_g1",
        "embedding_learning_rate": 0.02,
        "deep_learning_rate": 0.01,
        "seed": 42,
        "representation": "learned_sid_tokens",
        "levels": 4,
        "shared_codes": 8192,
        "representation_width": 128,
        "collision_policy": "suffix",
        "sid_initialization": "random",
    }
    base = StageManifest.create(
        stage="rq1_confirmation",
        jobs=(
            JobContract.create(
                job_id="rq1_confirmation:random:42",
                stage="rq1_confirmation",
                parameters=parameters,
                source_selection=predecessor,
                exact_reuse=(reuse,),
            ),
        ),
        predecessor=predecessor,
    )
    monkeypatch.setattr(
        materialize,
        "build_rq1_confirmation_manifest",
        lambda **_: base,
    )

    manifests = []
    recipes = []
    for revision in (0, 1, 2):
        path, recipe_path = compile_and_persist_stage(
            tmp_path / f"confirmation-{revision}.json",
            compiler_name="cachefix_rq1_confirmation",
            arguments={"retry_revision": revision},
        )
        manifests.append(load_queue_manifest(path))
        recipes.append(json.loads(recipe_path.read_text()))

    repeated_path, repeated_recipe_path = compile_and_persist_stage(
        tmp_path / "confirmation-1-repeated.json",
        compiler_name="cachefix_rq1_confirmation",
        arguments={"retry_revision": 1},
    )
    assert (
        load_queue_manifest(repeated_path).logical_sha256 == manifests[1].logical_sha256
    )
    assert (
        json.loads(repeated_recipe_path.read_text())["recipe_sha256"]
        == recipes[1]["recipe_sha256"]
    )
    assert len({manifest.logical_sha256 for manifest in manifests}) == 3
    assert len({recipe["recipe_sha256"] for recipe in recipes}) == 3
    jobs = [manifest.jobs[0] for manifest in manifests]
    assert len({job.job_id for job in jobs}) == 3
    assert len({job.run_name for job in jobs}) == 3
    assert len({job.data_group for job in jobs}) == 3
    assert [job.payload["exact_reuse"] for job in jobs] == [
        jobs[0].payload["exact_reuse"]
    ] * 3
    scientific_fields = {
        "backbone",
        "embedding_learning_rate",
        "deep_learning_rate",
        "seed",
        "representation",
        "levels",
        "shared_codes",
        "representation_width",
        "collision_policy",
        "sid_initialization",
    }
    assert [
        {field: job.payload["parameters"][field] for field in scientific_fields}
        for job in jobs
    ] == [
        {field: jobs[0].payload["parameters"][field] for field in scientific_fields}
    ] * 3


def test_rq23_surface_keeps_rq0_anchor_out_of_policy_selection() -> None:
    anchor = QueueJob(
        job_id="rq2_rq3_surface:random:99",
        run_name="anchor",
        runner="runner.py",
        config_logical_sha256="1" * 64,
        data_group="data",
        logical_sha256="2" * 64,
        payload={"parameters": {"collision_policy": "suffix", "seed": 42}},
        environment={},
    )
    suffix = replace(anchor, job_id="rq2_rq3_surface:suffix:00")

    assert (
        candidate_selection_group("rq2_rq3_surface", anchor, "collision_policy") is None
    )
    assert (
        candidate_selection_group(
            "rq2_rq3_surface",
            replace(anchor, job_id="rq2_rq3_surface:random:99:cachefix01"),
            "collision_policy",
        )
        is None
    )
    assert (
        candidate_selection_group(
            "rq2_rq3_surface",
            replace(anchor, job_id="rq2_rq3_surface:rq0_anchor:99"),
            "collision_policy",
        )
        is None
    )
    assert (
        candidate_selection_group(
            "rq2_rq3_refinement",
            replace(
                anchor,
                job_id="rq2_rq3_refinement:rq0_anchor:99:cachefix01",
            ),
            "collision_policy",
        )
        is None
    )
    assert (
        candidate_selection_group("rq2_rq3_surface", suffix, "collision_policy")
        == "suffix"
    )
    assert (
        candidate_selection_group(
            "rq2_rq3_confirmation",
            replace(anchor, job_id="rq2_rq3_confirmation:rq0_anchor:42"),
            "collision_policy",
        )
        == "rq0_anchor"
    )


def test_only_rq23_comparator_can_reuse_a_nonselected_candidate() -> None:
    job = QueueJob(
        job_id="rq2_rq3_refinement:rq0_anchor:99",
        run_name="anchor",
        runner="runner.py",
        config_logical_sha256="1" * 64,
        data_group="data",
        logical_sha256="2" * 64,
        payload={"stage": "rq2_rq3_refinement", "parameters": {"seed": 42}},
        environment={},
    )

    assert reuse_source_is_eligible(job, "source", {"suffix": "other"})
    assert not reuse_source_is_eligible(
        replace(job, job_id="rq2_rq3_refinement:suffix:00"),
        "source",
        {"suffix": "other"},
    )
    assert reuse_source_is_eligible(job, "source", {"suffix": "source"})

    overlap = replace(
        job,
        job_id="rq2_rq3_confirmation:suffix:43",
        payload={"stage": "rq2_rq3_confirmation", "parameters": {"seed": 43}},
    )
    assert reuse_source_is_eligible(
        overlap,
        "source",
        {"suffix": "other"},
        source_selection_stage="rq1_confirmation",
    )
    assert not reuse_source_is_eligible(
        overlap,
        "source",
        {"suffix": "other"},
        source_selection_stage="rq1_surface",
    )


def _write_manifest(path: Path, *, jobs: int = 2) -> Path:
    approval_path = Path(
        "experiments/g6_rqkmeans_history/protocol/native500m_approval.json"
    )
    body = {
        "schema": "g6-native500m-stage-manifest/v1",
        "stage": "controls",
        "dataset_size": "native-500m",
        "batch_size": 512,
        "training_horizon": 26,
        "plan_sha256": PLAN_SHA256,
        "approval_sha256": _sha256(approval_path.read_bytes()),
        "predecessor": None,
        "jobs": [
            {
                "schema": "g6-native500m-job/v1",
                "job_id": f"controls:{index}",
                "stage": "controls",
                "dataset_size": "native-500m",
                "batch_size": 512,
                "training_horizon": 26,
                "schedule": "annealed" if index % 2 == 0 else "constant",
                "plan_sha256": PLAN_SHA256,
                "approval_sha256": _sha256(approval_path.read_bytes()),
                "parameters": {
                    "builder": "control",
                    "run_name": f"g6_native500m_control_{index}",
                    "runner": (
                        "experiments/g6_rqkmeans_history/native500m/launchers/"
                        "run_native500m.py"
                    ),
                    "config_logical_sha256": f"{index + 1:064x}",
                    "data_group": "g6-native500m-controls-v1",
                    "environment": {
                        "G6_NATIVE500M_ROW_ID": f"controls:{index}",
                        "G6_NATIVE500M_SOURCE_SHA256": source_identity_sha256(),
                    },
                    "backbone": "best_g1" if index % 2 == 0 else "original_g1",
                    "embedding_learning_rate": 0.001,
                    "deep_learning_rate": 0.002,
                    "seed": 42,
                },
                "source_selection": None,
                "exact_reuse": [],
            }
            for index in range(jobs)
        ],
    }
    document = {**body, "sha256": _sha256(canonical_bytes(body))}
    path.write_bytes(canonical_bytes(document))
    return path


def test_batch_specification_is_versioned_and_manifest_bound(tmp_path: Path) -> None:
    manifest = replace(
        load_queue_manifest(_write_manifest(tmp_path / "manifest.json")),
        compiler_recipe_sha256="a" * 64,
    )

    specification = build_batch_specification(manifest)

    assert specification.document["version"] == 1
    assert specification.sha256 == _sha256(canonical_bytes(specification.document))
    assert [job["run"] for job in specification.document["jobs"]] == [
        "g6_native500m_control_0",
        "g6_native500m_control_1",
    ]
    for job in specification.document["jobs"]:
        assert job["data_group"] == "g6-native500m-controls-v1"
        environment = dict(value.split("=", 1) for value in job["environment"])
        assert environment["G6_NATIVE500M_MANIFEST_LOGICAL_SHA256"] == (
            manifest.logical_sha256
        )
        assert environment["G6_NATIVE500M_MANIFEST_PHYSICAL_SHA256"] == (
            manifest.physical_sha256
        )
        assert environment["G6_NATIVE500M_PLAN_SHA256"] == PLAN_SHA256
        assert environment["WANDB_MODE"] == "offline"


def test_batch_specification_supports_exact_nonempty_job_shards(
    tmp_path: Path,
) -> None:
    manifest = load_queue_manifest(_write_manifest(tmp_path / "manifest.json", jobs=4))
    included = frozenset((manifest.jobs[1].job_id, manifest.jobs[3].job_id))

    specification = build_batch_specification(manifest, included_job_ids=included)

    assert [job["run"] for job in specification.document["jobs"]] == [
        manifest.jobs[1].run_name,
        manifest.jobs[3].run_name,
    ]
    with pytest.raises(ValueError, match="nonempty subset"):
        build_batch_specification(manifest, included_job_ids=frozenset())
    with pytest.raises(ValueError, match="absent"):
        build_batch_specification(manifest, included_job_ids=frozenset({"foreign"}))


def test_batch_shards_are_deterministic_balanced_partitions(tmp_path: Path) -> None:
    manifest = load_queue_manifest(_write_manifest(tmp_path / "manifest.json", jobs=6))

    shards = [
        batch_shard_job_ids(manifest, shard_index=index, shard_count=2)
        for index in range(2)
    ]

    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == {job.job_id for job in manifest.jobs}
    assert [
        [job.job_id for job in manifest.jobs if job.job_id in shard] for shard in shards
    ] == [
        [manifest.jobs[index].job_id for index in (0, 2, 4)],
        [manifest.jobs[index].job_id for index in (1, 3, 5)],
    ]
    with pytest.raises(ValueError, match="coordinates"):
        batch_shard_job_ids(manifest, shard_index=2, shard_count=2)
    with pytest.raises(ValueError, match="exceeds"):
        batch_shard_job_ids(manifest, shard_index=0, shard_count=7)
    with pytest.raises(ValueError, match="maximum"):
        batch_shard_job_ids(manifest, shard_index=0, shard_count=1)


def test_manifest_adapter_rejects_partial_or_foreign_jobs(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path / "manifest.json")
    document = json.loads(path.read_text())
    document["jobs"][0]["batch_size"] = 640
    body = {key: value for key, value in document.items() if key != "sha256"}
    document["sha256"] = _sha256(canonical_bytes(body))
    path.write_bytes(canonical_bytes(document))

    with pytest.raises(ValueError, match="512"):
        load_queue_manifest(path)

    _write_manifest(path)
    document = json.loads(path.read_text())
    document["jobs"] = document["jobs"][:1]
    path.write_bytes(canonical_bytes(document))
    with pytest.raises(ValueError, match="logical SHA-256"):
        load_queue_manifest(path)


def test_resolve_finds_exact_batch_before_submit_and_recovers_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = replace(
        load_queue_manifest(_write_manifest(tmp_path / "manifest.json")),
        compiler_recipe_sha256="a" * 64,
    )
    calls: list[str] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        action = command[-2]
        calls.append(action)
        if action == "find-batch" and calls.count("find-batch") == 1:
            return subprocess.CompletedProcess(command, 3, "", "")
        return subprocess.CompletedProcess(command, 0, "a" * 32 + "\n", "")

    monkeypatch.setattr(subprocess, "run", run)

    first = resolve_or_submit_batch(
        manifest,
        state_directory=tmp_path / "state",
        specification_directory=tmp_path / "specifications",
    )
    second = resolve_or_submit_batch(
        manifest,
        state_directory=tmp_path / "state",
        specification_directory=tmp_path / "specifications",
    )

    assert first == second == "a" * 32
    assert calls == ["find-batch", "submit-batch", "find-batch"]


def test_resolve_submits_only_the_authenticated_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = replace(
        load_queue_manifest(_write_manifest(tmp_path / "manifest.json", jobs=4)),
        compiler_recipe_sha256="a" * 64,
    )
    included = batch_shard_job_ids(manifest, shard_index=1, shard_count=2)
    observed_runs: list[str] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        action = command[-2]
        specification = json.loads(Path(command[-1]).read_text())
        observed_runs[:] = [row["run"] for row in specification["jobs"]]
        return subprocess.CompletedProcess(
            command,
            3 if action == "find-batch" else 0,
            "" if action == "find-batch" else "a" * 32 + "\n",
            "",
        )

    monkeypatch.setattr(subprocess, "run", run)

    assert (
        resolve_or_submit_batch(
            manifest,
            state_directory=tmp_path / "state",
            specification_directory=tmp_path / "specifications",
            included_job_ids=included,
        )
        == "a" * 32
    )
    assert observed_runs == [manifest.jobs[index].run_name for index in (1, 3)]


def test_resolve_rejects_oversized_direct_submission(tmp_path: Path) -> None:
    manifest = replace(
        load_queue_manifest(_write_manifest(tmp_path / "manifest.json", jobs=5)),
        compiler_recipe_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="4-job maximum"):
        resolve_or_submit_batch(
            manifest,
            state_directory=tmp_path / "state",
            specification_directory=tmp_path / "specifications",
            dry_run=True,
        )


def test_dry_run_never_contacts_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = replace(
        load_queue_manifest(_write_manifest(tmp_path / "manifest.json")),
        compiler_recipe_sha256="a" * 64,
    )

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("dry-run contacted the queue")

    monkeypatch.setattr(subprocess, "run", forbidden)
    rendered = resolve_or_submit_batch(
        manifest,
        state_directory=tmp_path / "state",
        specification_directory=tmp_path / "specifications",
        dry_run=True,
    )

    assert json.loads(rendered)["version"] == 1


def test_runtime_round_trip_authenticates_manifest_job_and_config(
    tmp_path: Path,
) -> None:
    run_name = "g6_native500m_original_control"
    expected = build_control(
        backbone="original_g1",
        embedding_learning_rate=0.001,
        deep_learning_rate=0.002,
        run_name=run_name,
    )
    job = JobContract.create(
        job_id="controls:original",
        stage="controls",
        schedule="constant",
        parameters={
            "run_name": run_name,
            "builder": "control",
            "runner": (
                "experiments/g6_rqkmeans_history/native500m/launchers/"
                "run_native500m.py"
            ),
            "config_logical_sha256": experiment_logical_sha256(expected),
            "data_group": "g6-native500m-controls-v1",
            "environment": {"G6_NATIVE500M_SOURCE_SHA256": source_identity_sha256()},
            "backbone": "original_g1",
            "embedding_learning_rate": 0.001,
            "deep_learning_rate": 0.002,
            "seed": 42,
        },
    )
    document = StageManifest.create(stage="controls", jobs=(job,)).to_document()
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_bytes(document))
    manifest = load_queue_manifest(path)
    specification = build_batch_specification(manifest)
    environment = dict(
        value.split("=", 1)
        for value in specification.document["jobs"][0]["environment"]
    )

    runtime_manifest, runtime_job = load_runtime_job(environment)
    actual = build_experiment(runtime_job)

    assert runtime_manifest.logical_sha256 == manifest.logical_sha256
    assert runtime_job.job_id == job.job_id
    assert experiment_logical_sha256(actual) == experiment_logical_sha256(expected)


def test_controls_materializer_builds_runnable_approved_jobs(tmp_path: Path) -> None:
    compiled = build_controls_manifest()

    assert compiled.stage == "controls"
    assert len(compiled.jobs) == 24
    assert {job.backbone for job in compiled.jobs} == {"original_g1", "best_g1"}
    assert {job.schedule for job in compiled.jobs if job.backbone == "original_g1"} == {
        "constant"
    }
    assert {job.schedule for job in compiled.jobs if job.backbone == "best_g1"} == {
        "annealed"
    }
    assert all(
        experiment_logical_sha256(
            build_experiment(
                load_queue_manifest(
                    persist_stage_manifest(tmp_path / "controls.json", compiled)
                ).jobs[index]
            )
        )
        == job.parameters["config_logical_sha256"]
        for index, job in enumerate(compiled.jobs)
    )


def test_control_retry_uses_unique_operational_identities_only() -> None:
    original = build_controls_manifest()
    retry = build_controls_manifest(retry_revision=1)

    assert len(retry.jobs) == 24
    assert {job.job_id for job in original.jobs}.isdisjoint(
        {job.job_id for job in retry.jobs}
    )
    assert {job.run_name for job in original.jobs}.isdisjoint(
        {job.run_name for job in retry.jobs}
    )
    assert [
        (
            job.backbone,
            job.embedding_learning_rate,
            job.deep_learning_rate,
            job.seed,
        )
        for job in retry.jobs
    ] == [
        (
            job.backbone,
            job.embedding_learning_rate,
            job.deep_learning_rate,
            job.seed,
        )
        for job in original.jobs
    ]


def test_candidate_collection_selects_from_authenticated_best_epoch(
    tmp_path: Path,
) -> None:
    manifest = load_queue_manifest(_write_manifest(tmp_path / "manifest.json", jobs=4))
    logs_root = tmp_path / "logs"
    state = tmp_path / "state"
    batch_ids = ("b" * 32, "c" * 32)
    service_job_ids = []
    scores = (
        (0.120, 0.040),
        (0.130, 0.030),
        (0.119, 0.050),
        (0.128, 0.060),
    )
    for index, (job, score) in enumerate(zip(manifest.jobs, scores, strict=True)):
        directory = logs_root / job.run_name
        directory.mkdir(parents=True)
        (directory / "g6_native500m_job.json").write_bytes(
            canonical_bytes(
                {
                    "schema": "g6-native500m-run-contract/v1",
                    "manifest_logical_sha256": manifest.logical_sha256,
                    "manifest_physical_sha256": manifest.physical_sha256,
                    "job_logical_sha256": job.logical_sha256,
                    "config_logical_sha256": job.config_logical_sha256,
                    "job": job.payload,
                }
            )
        )
        schedule = job.payload["schedule"]
        checkpoint_path = directory / "best_model_state.pt"
        checkpoint_path.write_bytes(f"checkpoint-{index}".encode())
        checkpoint_sha256 = _sha256(checkpoint_path.read_bytes())
        (directory / "training_metadata.json").write_text(
            json.dumps(
                {
                    "dataset_size": "500m",
                    "batch_size": 512,
                    "physical_batch_size": 512,
                    "effective_batch_size": 512,
                    "gradient_accumulation_steps": 1,
                    "num_epochs": 26,
                    "max_epochs": 26,
                    "epochs_trained": 26,
                    "stopped_epoch": 26,
                    "early_stopped": False,
                    "best_epoch": 20,
                    "best_model_artifact": {
                        "schema": "g6-best-model-state/v1",
                        "path": str(checkpoint_path),
                        "sha256": checkpoint_sha256,
                    },
                    "lr_horizon_complete": schedule == "annealed",
                    "transfer_invariants": {
                        "restore_best_weights": True,
                        "lr_schedule": {
                            "shape": (
                                "constant" if schedule == "constant" else "cosine"
                            )
                        },
                    },
                }
            )
        )
        epochs = [
            {
                "epoch": epoch,
                "recall@100": score[0] if epoch == 20 else score[0] - 0.01,
                "ndcg@100": score[1] if epoch == 20 else score[1] - 0.01,
            }
            for epoch in range(1, 27)
        ]
        (directory / "validation_history.json").write_bytes(
            canonical_bytes(
                {
                    "schema": "g6-native500m-validation-history/v1",
                    "job_id": job.job_id,
                    "job_logical_sha256": job.logical_sha256,
                    "config_logical_sha256": job.config_logical_sha256,
                    "selection_metric": "recall@100",
                    "best_epoch": 20,
                    "epochs": epochs,
                }
            )
        )
        service_job_id = f"{index + 1:032x}"
        service_job_ids.append(service_job_id)
        (state / "completed").mkdir(parents=True, exist_ok=True)
        queued = build_batch_specification(manifest).document["jobs"][index]
        (state / "completed" / f"{service_job_id}.json").write_text(
            json.dumps({**queued, "exit_code": 0})
        )
    (state / "batches").mkdir(parents=True)
    for batch_id, indexes in zip(batch_ids, ((0, 1), (2, 3)), strict=True):
        included = frozenset(manifest.jobs[index].job_id for index in indexes)
        (state / "batches" / f"{batch_id}.json").write_text(
            json.dumps(
                {
                    "sealed": True,
                    "atomic_submission": True,
                    "specification_sha256": build_batch_specification(
                        manifest, included_job_ids=included
                    ).sha256,
                    "jobs": [service_job_ids[index] for index in indexes],
                }
            )
        )

    selection_path = tmp_path / "selection.json"
    selection = collect_stage_candidates(
        manifest=manifest,
        logs_root=logs_root,
        queue_state_directory=state,
        batch_id=batch_ids,
        recall_relative_dispersion=0.01685,
        output_path=selection_path,
    )

    assert selection["selected_job_ids"] == {
        "best_g1": manifest.jobs[2].job_id,
        "original_g1": manifest.jobs[3].job_id,
    }
    assert len(selection["candidates"]) == 4
    assert selection["batch_id"] == list(batch_ids)
    assert len(selection["selection_sha256"]) == 64
    assert (
        collect_stage_candidates(
            manifest=manifest,
            logs_root=logs_root,
            queue_state_directory=state,
            batch_id=tuple(reversed(batch_ids)),
            recall_relative_dispersion=0.01685,
            output_path=None,
        )
        == selection
    )

    binding = {
        "path": str(selection_path),
        "manifest_path": str(manifest.path),
        "logs_root": str(logs_root),
        "queue_state_directory": str(state),
    }
    assert _load_selection_ledger(binding) == selection

    forged = json.loads(selection_path.read_text())
    forged["candidates"][0]["validation_metrics"]["recall@100"] = 0.99
    forged_body = {
        key: value for key, value in forged.items() if key != "selection_sha256"
    }
    forged["selection_sha256"] = _sha256(canonical_bytes(forged_body))
    forged_path = tmp_path / "forged-selection.json"
    forged_path.write_bytes(canonical_bytes(forged))
    with pytest.raises(ValueError, match="authenticated artifacts"):
        _load_selection_ledger({**binding, "path": str(forged_path)})

    history_path = logs_root / manifest.jobs[0].run_name / "validation_history.json"
    history = json.loads(history_path.read_text())
    history["epochs"][0]["recall@100"] = 0.99
    history_path.write_bytes(canonical_bytes(history))
    with pytest.raises(ValueError, match="best epoch"):
        collect_stage_candidates(
            manifest=manifest,
            logs_root=logs_root,
            queue_state_directory=state,
            batch_id=batch_ids,
            recall_relative_dispersion=0.01685,
            output_path=tmp_path / "tampered-selection.json",
        )


def test_final_collection_normalizes_batch_ids_for_queue_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_queue_manifest(_write_manifest(tmp_path / "manifest.json", jobs=1))
    job = manifest.jobs[0]
    directory = tmp_path / "logs" / job.run_name
    directory.mkdir(parents=True)
    (directory / "g6_native500m_job.json").write_bytes(
        canonical_bytes(
            {
                "schema": "g6-native500m-run-contract/v1",
                "manifest_logical_sha256": manifest.logical_sha256,
                "manifest_physical_sha256": manifest.physical_sha256,
                "job_logical_sha256": job.logical_sha256,
                "config_logical_sha256": job.config_logical_sha256,
                "job": job.payload,
            }
        )
    )
    (directory / "training_metadata.json").write_text(
        json.dumps(
            {
                "dataset_size": "500m",
                "batch_size": 512,
                "physical_batch_size": 512,
                "effective_batch_size": 512,
                "gradient_accumulation_steps": 1,
                "num_epochs": 26,
                "max_epochs": 26,
                "epochs_trained": 26,
                "stopped_epoch": 26,
                "early_stopped": False,
                "best_epoch": 20,
                "lr_horizon_complete": True,
                "transfer_invariants": {
                    "restore_best_weights": True,
                    "lr_schedule": {"shape": "cosine"},
                },
            }
        )
    )
    for name in (
        "final_metrics.json",
        "ranking_evidence.pt",
        "top100_item_evidence.pt",
        "final_evaluation.json",
    ):
        (directory / name).touch()

    def verified(
        state: Path,
        *,
        batch_ids: tuple[str, ...],
        manifest: object,
        job: object,
    ) -> None:
        assert state == tmp_path / "state"
        assert batch_ids == ("a" * 32,)
        raise RuntimeError("queue verification reached")

    monkeypatch.setattr(collect_analysis, "_verify_queue_success", verified)

    with pytest.raises(RuntimeError, match="queue verification reached"):
        collect_analysis.collect_final_run(
            manifest=manifest,
            job_id=job.job_id,
            logs_root=tmp_path / "logs",
            queue_state_directory=tmp_path / "state",
            batch_id="a" * 32,
            selection_sha256="b" * 64,
            ranking_context_path=tmp_path / "ranking-context.pt",
            semantic_codes=torch.empty((0, 0), dtype=torch.int64),
            metric_tolerance=0.0,
            output_path=tmp_path / "evidence.json",
        )


def test_dependent_manifest_binds_authenticated_selection(tmp_path: Path) -> None:
    source_path = _write_manifest(tmp_path / "source-manifest.json")
    source = load_queue_manifest(source_path)
    body = {
        "schema": "g6-native500m-stage-selection/v1",
        "stage": "controls",
        "manifest_logical_sha256": source.logical_sha256,
        "manifest_physical_sha256": source.physical_sha256,
        "batch_id": "3" * 32,
        "recall_relative_dispersion": 0.01685,
        "selection_group_field": "backbone",
        "selected_job_ids": {
            "best_g1": source.jobs[0].job_id,
            "original_g1": source.jobs[1].job_id,
        },
        "candidates": [
            {
                "job_id": job.job_id,
                "job_logical_sha256": job.logical_sha256,
                "parameters": job.payload["parameters"],
                "validation_metrics": {
                    "recall@100": 0.12 - index * 0.01,
                    "ndcg@100": 0.05 - index * 0.01,
                },
            }
            for index, job in enumerate(source.jobs)
        ],
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_bytes(
        canonical_bytes({**body, "selection_sha256": _sha256(canonical_bytes(body))})
    )
    with pytest.raises(ValueError, match="missing candidate artifact"):
        build_rq0_first_surface_manifest(
            selection_path,
            source_manifest_path=source_path,
            logs_root=tmp_path / "logs",
            queue_state_directory=tmp_path / "state",
        )


def test_queue_admission_replays_compiler_recipe_and_rejects_forged_manifest(
    tmp_path: Path,
) -> None:
    path, recipe_path = compile_and_persist_stage(
        tmp_path / "controls.json", compiler_name="controls", arguments={}
    )

    admitted = load_admitted_queue_manifest(path)

    assert admitted.compiler_recipe_sha256 is not None
    assert recipe_path.is_file()
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="compiler recipe replay"):
        load_admitted_queue_manifest(path)


def test_queue_admission_rejects_forged_compiler_provenance(tmp_path: Path) -> None:
    path, recipe_path = compile_and_persist_stage(
        tmp_path / "controls.json", compiler_name="controls", arguments={}
    )
    recipe = json.loads(recipe_path.read_text())
    recipe["compiler"] = "arbitrary_rows"
    body = {key: value for key, value in recipe.items() if key != "recipe_sha256"}
    recipe["recipe_sha256"] = _sha256(canonical_bytes(body))
    recipe_path.chmod(0o644)
    recipe_path.write_bytes(canonical_bytes(recipe))

    with pytest.raises(ValueError, match="unknown compiler"):
        load_admitted_queue_manifest(path)


def test_rq23_overlap_resolution_allows_only_approved_two_four_six_new_runs() -> None:
    def parameters(*, policy: str, initialization: str = "random") -> dict[str, object]:
        return {
            "backbone": "best_g1",
            "embedding_learning_rate": 0.02,
            "deep_learning_rate": 0.01,
            "seed": 42,
            "representation": "learned_sid_event",
            "levels": 3,
            "shared_codes": 512,
            "representation_width": 128,
            "collision_policy": policy,
            "sid_initialization": initialization,
        }

    arms = {
        "rq0_anchor": parameters(policy="suffix"),
        "suffix": parameters(policy="suffix", initialization="content_pca"),
        "none": parameters(policy="none", initialization="content_pca"),
    }
    rq0 = tuple(
        {
            "job_id": f"rq1:random:{seed}",
            "parameters": arms["rq0_anchor"] | {"seed": seed},
        }
        for seed in (43, 44)
    )
    suffix = tuple(
        {"job_id": f"rq1:content:{seed}", "parameters": arms["suffix"] | {"seed": seed}}
        for seed in (43, 44)
    )

    assert sum(row[3] is None for row in resolve_rq23_confirmation_reuse(arms, ())) == 6
    assert (
        sum(row[3] is None for row in resolve_rq23_confirmation_reuse(arms, rq0)) == 4
    )
    assert (
        sum(
            row[3] is None
            for row in resolve_rq23_confirmation_reuse(arms, (*rq0, *suffix))
        )
        == 2
    )


def test_rq23_confirmation_recovers_revisioned_anchor_from_authenticated_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def parameters(*, policy: str, initialization: str) -> dict[str, object]:
        return {
            "backbone": "best_g1",
            "embedding_learning_rate": 0.02,
            "deep_learning_rate": 0.01,
            "seed": 42,
            "representation": "learned_sid_event",
            "levels": 3,
            "shared_codes": 512,
            "representation_width": 128,
            "collision_policy": policy,
            "sid_initialization": initialization,
        }

    refined = {
        "stage": "rq2_rq3_refinement",
        "selection_sha256": "1" * 64,
        "selected_job_ids": {
            "suffix": "rq2_rq3_refinement:suffix:00:cachefix01",
            "none": "rq2_rq3_refinement:none:00:cachefix01",
        },
        "candidates": [
            {
                "job_id": "rq2_rq3_refinement:suffix:00:cachefix01",
                "parameters": parameters(policy="suffix", initialization="content_pca"),
            },
            {
                "job_id": "rq2_rq3_refinement:none:00:cachefix01",
                "parameters": parameters(policy="none", initialization="content_pca"),
            },
        ],
    }
    rq0 = {
        "stage": "rq2_rq3_surface",
        "selection_sha256": "2" * 64,
        "selected_job_ids": {},
        "candidates": [
            {
                "job_id": "rq2_rq3_surface:random:99:cachefix01",
                "parameters": parameters(policy="suffix", initialization="random"),
            }
        ],
    }

    def authenticate(
        selection_path: Path, *_: object
    ) -> tuple[SelectionBinding, dict[str, object]]:
        selection = rq0 if selection_path.name == "rq0.json" else refined
        if selection_path.name == "boundary.json":
            selection = dict(refined) | {
                "stage": "rq2_rq3_boundary",
                "selection_sha256": "4" * 64,
            }
        return (
            SelectionBinding(selection["stage"], selection["selection_sha256"], True),
            selection,
        )

    carried_calls: list[dict[str, object]] = []

    def carry(**arguments: object) -> JobContract:
        carried_calls.append(arguments)
        selection = arguments["selection"]
        source_id = arguments["source_id"]
        source = next(
            row for row in selection["candidates"] if row["job_id"] == source_id
        )
        row = materialize._semantic_row_from_parameters(
            stage=str(arguments["stage"]),
            family=str(arguments["family"]),
            index=int(arguments["index"]),
            parameters=source["parameters"],
            seed=int(source["parameters"]["seed"]),
        )
        return materialize._dependent_job(
            str(arguments["stage"]),
            arguments["predecessor"],
            row,
            source_sha256=source_identity_sha256(),
            exact_reuse=(ExactReuse("source", "2" * 64, ("seed",)),),
        )

    monkeypatch.setattr(materialize, "_authenticated_selection", authenticate)
    monkeypatch.setattr(materialize, "_carried_candidate_job", carry)
    monkeypatch.setattr(
        materialize,
        "load_stage_manifest",
        lambda path: SimpleNamespace(
            predecessor=(
                SelectionBinding("rq2_rq3_refinement", "1" * 64, True)
                if path.name == "boundary-manifest.json"
                else SelectionBinding("rq2_rq3_surface", "2" * 64, True)
            )
        ),
    )

    manifest = build_rq23_confirmation_manifest(
        tmp_path / "refined.json",
        source_manifest_path=tmp_path / "refined-manifest.json",
        logs_root=tmp_path / "logs",
        queue_state_directory=tmp_path / "queue",
        rq0_selection_path=tmp_path / "rq0.json",
        rq0_source_manifest_path=tmp_path / "rq0-manifest.json",
    )

    anchor = next(job for job in manifest.jobs if ":rq0_anchor:42" in job.job_id)
    assert anchor.parameters["sid_initialization"] == "random"
    assert any(
        call["selection"] is rq0
        and call["source_id"] == "rq2_rq3_surface:random:99:cachefix01"
        for call in carried_calls
    )

    rq0["selection_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="lineage"):
        build_rq23_confirmation_manifest(
            tmp_path / "refined.json",
            source_manifest_path=tmp_path / "refined-manifest.json",
            logs_root=tmp_path / "logs",
            queue_state_directory=tmp_path / "queue",
            rq0_selection_path=tmp_path / "rq0.json",
            rq0_source_manifest_path=tmp_path / "rq0-manifest.json",
        )

    rq0["selection_sha256"] = "2" * 64
    boundary_manifest = build_rq23_confirmation_manifest(
        tmp_path / "boundary.json",
        source_manifest_path=tmp_path / "boundary-manifest.json",
        logs_root=tmp_path / "logs",
        queue_state_directory=tmp_path / "queue",
        rq0_selection_path=tmp_path / "rq0.json",
        rq0_source_manifest_path=tmp_path / "rq0-manifest.json",
        rq0_lineage_selection_path=tmp_path / "refined.json",
        rq0_lineage_source_manifest_path=tmp_path / "refined-manifest.json",
    )
    assert any(":rq0_anchor:42" in job.job_id for job in boundary_manifest.jobs)
