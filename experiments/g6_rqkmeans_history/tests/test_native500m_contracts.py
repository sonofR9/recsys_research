import dataclasses
import json
from pathlib import Path

import pytest

from experiments.g6_rqkmeans_history.native500m.configs.runtime import (
    build_control,
    build_semantic_treatment,
)
from experiments.g6_rqkmeans_history.native500m.launchers.queue import QueueJob
from experiments.g6_rqkmeans_history.native500m.launchers.runtime import (
    build_experiment,
    experiment_logical_sha256,
)
from experiments.g6_rqkmeans_history.native500m.launchers import runtime
from experiments.g6_rqkmeans_history.native500m.protocol.contracts import (
    APPROVED_STAGES,
    APPROVED_PREDECESSORS,
    APPROVAL_SHA256,
    PLAN_SHA256,
    EnvironmentContract,
    ExactReuse,
    JobContract,
    SelectionBinding,
    StageManifest,
    canonical_json,
    document_identity,
    load_approval_binding,
)


def _parameters(stage: str, *, learning_rate: float = 0.01) -> dict[str, object]:
    parameters: dict[str, object] = {
        "builder": "control" if stage.startswith("controls") else "semantic",
        "runner": "experiments/g6_rqkmeans_history/native500m/launchers/run_native500m.py",
        "run_name": f"g6_native500m_{stage}",
        "config_logical_sha256": "c" * 64,
        "data_group": "g6-native500m-v1",
        "environment": {},
        "backbone": (
            "original_g1"
            if stage.startswith("rq0_bridge") or stage.startswith("terminal_bridge")
            else "best_g1"
        ),
        "embedding_learning_rate": learning_rate,
        "deep_learning_rate": 0.02,
        "seed": 42,
    }
    if not stage.startswith("controls"):
        parameters["environment"] = {
            "G6_NATIVE500M_TOKENIZER_BINDING_REVISION": "shared-base-v2",
            "G6_NATIVE500M_TOKENIZER_REGISTRY_SHA256": "1" * 64,
            "G6_NATIVE500M_TOKENIZER_BASE_CACHE_KEY": "kmeans_3x512_test",
            "G6_NATIVE500M_TOKENIZER_FIT_SHA256": "2" * 64,
            "G6_NATIVE500M_TOKENIZER_CODES_SHA256": "3" * 64,
            "G6_NATIVE500M_TOKENIZER_MATERIALIZATION_SHA256": "4" * 64,
        }
        parameters |= {
            "representation": "learned_sid_event",
            "levels": 3,
            "shared_codes": 512,
            "representation_width": 128,
            "collision_policy": "suffix",
            "sid_initialization": "random",
        }
    if stage.endswith("confirmation"):
        parameters["seed"] = 43
    return parameters


def _job(
    job_id: str = "control:0",
    stage: str = "controls",
    *,
    source: SelectionBinding | None = None,
    learning_rate: float = 0.01,
    reuse: tuple[ExactReuse, ...] = (),
) -> JobContract:
    return JobContract.create(
        job_id=job_id,
        stage=stage,
        parameters=_parameters(stage, learning_rate=learning_rate),
        source_selection=source,
        exact_reuse=reuse,
    )


def test_approval_binding_authenticates_exact_plan_and_approval_files() -> None:
    binding = load_approval_binding()

    assert binding.plan_sha256 == PLAN_SHA256
    assert binding.approval_sha256 == APPROVAL_SHA256
    assert binding.expected_run_totals == (130, 132, 134, 138, 140, 142)
    assert binding.maximum_runs == 262


def test_approval_binding_rejects_tampered_plan(tmp_path: Path) -> None:
    approval = tmp_path / "approval.json"
    plan = tmp_path / "plan.md"
    plan.write_text("tampered")
    approval.write_text(
        json.dumps(
            {
                "schema": "g6-native500m-approval/v1",
                "plan_sha256": PLAN_SHA256,
                "dataset_size": "native-500m",
                "expected_run_totals": [130, 132, 134, 138, 140, 142],
                "maximum_runs": 262,
                "fixed_training_horizon": 26,
                "representation_width": 128,
                "tokenizer_levels": [3, 4],
                "shared_codebook_sizes": [512, 2048, 8192],
                "fixed_kmeans_max_iterations": 300,
                "fixed_kmeans_tolerance": 0.0001,
                "approved_at": "2026-08-31",
            }
        )
    )

    with pytest.raises(ValueError, match="plan SHA-256"):
        load_approval_binding(plan_path=plan, approval_path=approval)


def test_canonical_and_physical_hashes_have_distinct_meanings() -> None:
    compact = b'{"a":1,"b":2}'
    pretty = b'{\n  "b": 2,\n  "a": 1\n}\n'

    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert (
        document_identity(compact).logical_sha256
        == document_identity(pretty).logical_sha256
    )
    assert (
        document_identity(compact).physical_sha256
        != document_identity(pretty).physical_sha256
    )


def test_job_contract_is_immutable_and_rejects_protocol_drift() -> None:
    job = _job()

    assert job.batch_size == 512
    assert job.training_horizon == 26
    assert job.dataset_size == "native-500m"
    assert job.plan_sha256 == PLAN_SHA256
    assert job.approval_sha256 == APPROVAL_SHA256
    with pytest.raises(dataclasses.FrozenInstanceError):
        job.job_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="batch size"):
        JobContract.create(
            job_id="bad", stage="controls", parameters={}, batch_size=256
        )


def test_current_semantic_job_requires_complete_tokenizer_binding() -> None:
    parameters = _parameters("rq0_surface")
    parameters["environment"] = {}

    with pytest.raises(ValueError, match="lacks its tokenizer artifact binding"):
        JobContract.create(
            job_id="rq0:unbound",
            stage="rq0_surface",
            parameters=parameters,
            source_selection=SelectionBinding("controls", "a" * 64, True),
        )


def test_legacy_semantic_contract_is_readable_but_not_executable() -> None:
    current = _job(
        "rq0:legacy",
        "rq0_surface",
        source=SelectionBinding("controls", "a" * 64, True),
    )
    document = current.to_dict()
    document["schema"] = "g6-native500m-job/v1"
    document["parameters"]["environment"] = {}

    legacy = JobContract.from_dict(document)
    queue_job = QueueJob(
        job_id=legacy.job_id,
        run_name=legacy.run_name,
        runner=legacy.parameters["runner"],
        config_logical_sha256=legacy.parameters["config_logical_sha256"],
        data_group=legacy.parameters["data_group"],
        logical_sha256=legacy.logical_sha256,
        payload=legacy.to_dict(),
        environment={},
    )

    assert legacy.schema == "g6-native500m-job/v1"
    with pytest.raises(RuntimeError, match="current job contract"):
        build_experiment(queue_job)


def test_schedule_shape_is_bound_to_the_backbone() -> None:
    original = JobContract.create(
        job_id="original:0",
        stage="controls",
        parameters=_parameters("controls") | {"backbone": "original_g1"},
        schedule="constant",
    )
    assert original.schedule == "constant"
    with pytest.raises(ValueError, match="constant-LR"):
        JobContract.create(
            job_id="original:bad",
            stage="controls",
            parameters=_parameters("controls") | {"backbone": "original_g1"},
            schedule="annealed",
        )
    with pytest.raises(ValueError, match="annealed"):
        JobContract.create(
            job_id="best:bad",
            stage="controls",
            parameters=_parameters("controls") | {"backbone": "best_g1"},
            schedule="constant",
        )


def test_stage_manifest_requires_the_exact_resolved_predecessor() -> None:
    predecessor = SelectionBinding("rq0_surface", "a" * 64, True)
    job = _job("rq1:0", "rq1_surface", source=predecessor)
    manifest = StageManifest.create(
        stage="rq1_surface",
        jobs=(job,),
        predecessor=predecessor,
        requires_predecessor=True,
    )

    assert manifest.predecessor == predecessor
    with pytest.raises(ValueError, match="source selection"):
        StageManifest.create(
            stage="rq1_surface",
            jobs=(job,),
            predecessor=SelectionBinding("rq0_surface", "b" * 64, True),
            requires_predecessor=True,
        )
    with pytest.raises(ValueError, match="resolved"):
        StageManifest.create(
            stage="rq1_surface",
            jobs=(job,),
            predecessor=SelectionBinding("rq0_surface", "a" * 64, False),
            requires_predecessor=True,
        )


def test_exact_reuse_authenticates_source_and_equal_fields() -> None:
    source = _job(learning_rate=0.02)
    reuse = ExactReuse(
        source_job_id=source.job_id,
        source_contract_sha256=source.logical_sha256,
        fields=("embedding_learning_rate", "seed"),
    )
    predecessor = SelectionBinding("controls", "a" * 64, True)
    target = _job(
        "rq0:reuse",
        "rq0_surface",
        source=predecessor,
        learning_rate=0.02,
        reuse=(reuse,),
    )
    target.validate_reuse({source.job_id: source})

    changed = _job(
        "rq0:changed",
        "rq0_surface",
        source=predecessor,
        learning_rate=0.03,
        reuse=(reuse,),
    )
    with pytest.raises(ValueError, match="exact-reuse field"):
        changed.validate_reuse({source.job_id: source})


def test_environment_contract_round_trips_and_rejects_foreign_values() -> None:
    manifest = StageManifest.create(stage="controls", jobs=(_job(),))
    environment = EnvironmentContract.from_manifest(manifest)

    assert EnvironmentContract.from_environ(environment.to_environ()) == environment
    changed = environment.to_environ() | {"G6_DATASET_SIZE": "native-50m"}
    with pytest.raises(ValueError, match="dataset"):
        EnvironmentContract.from_environ(changed)


@pytest.mark.parametrize("stage", APPROVED_STAGES)
def test_job_contract_enforces_exact_runnable_schema_for_each_training_stage(
    stage: str,
) -> None:
    if stage == "aggregate":
        with pytest.raises(ValueError, match="not a runnable"):
            JobContract.create(
                job_id="aggregate:0",
                stage=stage,
                parameters=_parameters(stage),
            )
        return
    schedule = (
        "constant" if _parameters(stage)["backbone"] == "original_g1" else "annealed"
    )
    job = JobContract.create(
        job_id=f"{stage}:0",
        stage=stage,
        parameters=_parameters(stage),
        schedule=schedule,
        source_selection=(
            None
            if stage == "controls"
            else SelectionBinding(
                {
                    "controls_boundary": "controls",
                    "rq0_surface": "controls",
                    "rq0_boundary": "rq0_surface",
                    "rq0_bridge": "rq0_surface",
                    "rq0_bridge_boundary": "rq0_bridge",
                    "rq1_surface": "rq0_surface",
                    "rq1_boundary": "rq1_surface",
                    "rq1_confirmation": "rq1_surface",
                    "rq2_rq3_surface": "rq1_confirmation",
                    "rq2_rq3_refinement": "rq2_rq3_surface",
                    "rq2_rq3_boundary": "rq2_rq3_refinement",
                    "rq2_rq3_confirmation": "rq2_rq3_refinement",
                    "terminal_bridge": "rq2_rq3_confirmation",
                    "terminal_bridge_boundary": "terminal_bridge",
                }[stage],
                "a" * 64,
                True,
            )
        ),
    )
    assert job.stage == stage

    missing = _parameters(stage)
    missing.pop("runner")
    with pytest.raises(ValueError, match="parameter fields"):
        JobContract.create(
            job_id=f"{stage}:missing",
            stage=stage,
            parameters=missing,
            schedule=schedule,
            source_selection=job.source_selection,
        )
    extra = _parameters(stage) | {"unapproved": True}
    with pytest.raises(ValueError, match="parameter fields"):
        JobContract.create(
            job_id=f"{stage}:extra",
            stage=stage,
            parameters=extra,
            schedule=schedule,
            source_selection=job.source_selection,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("backbone", "foreign"),
        ("representation", "foreign"),
        ("levels", 2),
        ("shared_codes", 1024),
        ("representation_width", 64),
        ("collision_policy", "foreign"),
        ("sid_initialization", "foreign"),
        ("seed", True),
    ),
)
def test_semantic_job_contract_rejects_values_outside_approved_domains(
    field: str, value: object
) -> None:
    predecessor = SelectionBinding("controls", "a" * 64, True)
    parameters = _parameters("rq0_surface") | {field: value}

    with pytest.raises(ValueError):
        JobContract.create(
            job_id="rq0_surface:bad",
            stage="rq0_surface",
            parameters=parameters,
            source_selection=predecessor,
        )


def test_manifest_enforces_the_approved_predecessor_dag() -> None:
    valid = SelectionBinding("controls", "a" * 64, True)
    job = _job("rq0_surface:0", "rq0_surface", source=valid)
    StageManifest.create(stage="rq0_surface", jobs=(job,), predecessor=valid)

    invalid = SelectionBinding("rq2_rq3_confirmation", "b" * 64, True)
    invalid_job = dataclasses.replace(job, source_selection=invalid)
    with pytest.raises(ValueError, match="predecessor stage"):
        StageManifest.create(
            stage="rq0_surface", jobs=(invalid_job,), predecessor=invalid
        )
    with pytest.raises(ValueError, match="unknown stage"):
        SelectionBinding("foreign", "a" * 64, True)


def test_rq23_refinement_can_follow_the_initial_surface() -> None:
    predecessor = SelectionBinding("rq2_rq3_surface", "a" * 64, True)
    job = _job(
        "rq2_rq3_refinement:suffix:00",
        "rq2_rq3_refinement",
        source=predecessor,
    )

    manifest = StageManifest.create(
        stage="rq2_rq3_refinement", jobs=(job,), predecessor=predecessor
    )

    assert manifest.predecessor == predecessor

    repeated = SelectionBinding("rq2_rq3_refinement", "b" * 64, True)
    with pytest.raises(ValueError, match="predecessor stage"):
        _job(
            "rq2_rq3_refinement:suffix:01",
            "rq2_rq3_refinement",
            source=repeated,
        )


def test_environment_stage_is_bound_to_the_manifest() -> None:
    manifest = StageManifest.create(stage="controls", jobs=(_job(),))
    environment = EnvironmentContract.from_manifest(manifest).to_environ()
    environment["G6_STAGE"] = "rq0_surface"

    with pytest.raises(ValueError, match="manifest stage"):
        EnvironmentContract.from_environ(environment, manifest=manifest)


def test_aggregate_manifest_is_dependency_bound_but_has_no_runnable_jobs() -> None:
    predecessor = SelectionBinding("rq2_rq3_confirmation", "a" * 64, True)

    manifest = StageManifest.create(stage="aggregate", jobs=(), predecessor=predecessor)

    assert manifest.jobs == ()
    assert manifest.predecessor == predecessor


def test_job_identity_and_runner_must_be_strict_runnable_strings() -> None:
    with pytest.raises(ValueError, match="job ID"):
        JobContract.create(
            job_id=1,  # type: ignore[arg-type]
            stage="controls",
            parameters=_parameters("controls"),
        )
    with pytest.raises(ValueError, match="runner"):
        JobContract.create(
            job_id="controls:bad_runner",
            stage="controls",
            parameters=_parameters("controls") | {"runner": "missing.py"},
        )


@pytest.mark.parametrize(
    ("stage", "seed"),
    (("controls", 43), ("rq0_surface", 43), ("rq1_confirmation", 42)),
)
def test_job_seed_must_match_the_stage(stage: str, seed: int) -> None:
    parameters = _parameters(stage) | {"seed": seed}
    predecessor = (
        None if stage == "controls" else SelectionBinding("controls", "a" * 64, True)
    )

    with pytest.raises(ValueError, match="seed"):
        JobContract.create(
            job_id=f"{stage}:bad_seed",
            stage=stage,
            parameters=parameters,
            source_selection=predecessor,
        )


def test_content_initialization_requires_a_trainable_sid_representation() -> None:
    parameters = _parameters("rq2_rq3_surface") | {
        "representation": "frozen_sid_tokens",
        "sid_initialization": "content_pca",
    }
    predecessor = SelectionBinding("rq1_confirmation", "a" * 64, True)

    with pytest.raises(ValueError, match="trainable-SID"):
        JobContract.create(
            job_id="rq2_rq3_surface:bad_content",
            stage="rq2_rq3_surface",
            parameters=parameters,
            source_selection=predecessor,
        )


@pytest.mark.parametrize(
    "stage", tuple(stage for stage in APPROVED_STAGES if stage != "aggregate")
)
def test_every_signed_stage_schema_builds_the_bound_runtime_experiment(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameters = _parameters(stage)
    if stage.startswith("controls"):
        expected = build_control(
            backbone=parameters["backbone"],
            embedding_learning_rate=parameters["embedding_learning_rate"],
            deep_learning_rate=parameters["deep_learning_rate"],
            run_name=parameters["run_name"],
            seed=parameters["seed"],
        )
    else:
        expected = build_semantic_treatment(
            backbone=parameters["backbone"],
            representation=parameters["representation"],
            embedding_learning_rate=parameters["embedding_learning_rate"],
            deep_learning_rate=parameters["deep_learning_rate"],
            num_levels=parameters["levels"],
            num_codes=parameters["shared_codes"],
            run_name=parameters["run_name"],
            seed=parameters["seed"],
            representation_width=parameters["representation_width"],
            collision_policy=parameters["collision_policy"],
            sid_lookup_initialization=parameters["sid_initialization"],
        )
    parameters["config_logical_sha256"] = experiment_logical_sha256(expected)
    predecessor_stage = next(iter(APPROVED_PREDECESSORS[stage]), None)
    predecessor = (
        None
        if predecessor_stage is None
        else SelectionBinding(predecessor_stage, "a" * 64, True)
    )
    schedule = "constant" if parameters["backbone"] == "original_g1" else "annealed"
    contract = JobContract.create(
        job_id=f"{stage}:buildable",
        stage=stage,
        parameters=parameters,
        source_selection=predecessor,
        schedule=schedule,
    )
    queue_job = QueueJob(
        job_id=contract.job_id,
        run_name=contract.run_name,
        runner=parameters["runner"],
        config_logical_sha256=parameters["config_logical_sha256"],
        data_group=parameters["data_group"],
        logical_sha256=contract.logical_sha256,
        payload=contract.to_dict(),
        environment=dict(parameters["environment"]),
    )
    monkeypatch.setattr(
        runtime,
        "load_tokenizer_registry",
        lambda *args, **kwargs: {"sha256": "1" * 64},
    )
    monkeypatch.setattr(runtime, "verify_tokenizer_binding", lambda **kwargs: None)

    assert experiment_logical_sha256(build_experiment(queue_job)) == (
        parameters["config_logical_sha256"]
    )
