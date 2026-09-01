from __future__ import annotations

import base64
from copy import deepcopy
from functools import lru_cache
import json
from dataclasses import fields, replace
from pathlib import Path
import runpy
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from experiments.g4_future_items.protocol.native500m import manifest as native_manifest
from experiments.g4_future_items.configs.native500m import (
    DEEP_LEARNING_RATE_ANCHOR,
    EMBEDDING_LEARNING_RATE,
    build_native500m_control,
    build_native500m_treatment,
)
from experiments.g4_future_items.launchers.native500m import build_queue_specification
from experiments.g4_future_items.launchers.run_native500m import (
    build_experiment,
    load_compiled_job,
    validate_compiled_data_identity,
)
from experiments.g4_future_items.protocol.manifest import MATERIALIZATION_COST_LIMITS
from experiments.g4_future_items.protocol.materialization import write_period_artifact
from experiments.g4_future_items.protocol.native500m.manifest import (
    BASE_DEEP_LEARNING_RATES,
    build_native_source_closure,
    canonical_bytes,
    canonical_sha256,
    compile_base_ledger,
    compile_boundary_ledger,
    load_frozen_ledger,
    resolve_native500m_data_identity,
    validate_native500m_data_identity,
    validate_native_source_closure,
    validate_runtime_experiment,
)
from dcn.config import MuTransferGenerationExperiment
from experiments.g4_future_items.targets import (
    OCCURRENCE_POSITION_COLUMN,
    FutureEvent,
)
from dcn.nn.precomputed_embeddings import PrecomputedEmbeddingLookup
from dcn.tests.helpers import scalar_feature
from dcn.models.sequence_targets import NextItemTargets
from neuralrec.utils import LOSS_DENOMINATOR


ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_ROOT = ROOT / "experiments/g4_future_items/protocol/native500m"
G1_SELECTED_RUN = (
    "g1_aggregate_aggregate_none_l4_e0p0468526465053628_"
    "d0p032703745675187676_h15_c0_initial_ts2_r1_500m"
)


def _write_resealed(path: Path, document: dict[str, object]) -> None:
    unsealed = {key: value for key, value in document.items() if key != "sha256"}
    path.write_bytes(canonical_bytes(unsealed | {"sha256": canonical_sha256(unsealed)}))


def _compiled_payload(specification: dict[str, object], index: int = 0) -> dict:
    environment = specification["jobs"][index]["environment"]
    encoded = next(
        value.split("=", 1)[1]
        for value in environment
        if value.startswith("G4_NATIVE500M_JOB_B64=")
    )
    return json.loads(base64.urlsafe_b64decode(encoded))


@lru_cache(maxsize=1)
def _authenticated_execution_identity() -> tuple[dict, dict]:
    payload = _compiled_payload(
        build_queue_specification(PROTOCOL_ROOT / "ledgers/control_tuning.json")
    )
    return payload["source_closure"], payload["data_identity"]


def _materialization_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object]]:
    measurement_id = "e" * 64
    artifact_root = tmp_path / "selector_artifacts"
    deterministic = write_period_artifact(
        [],
        selector_kind="deterministic",
        selected_configuration={},
        provenance={},
        cost={"measurement_id": measurement_id},
        output_root=artifact_root,
    )
    learned = write_period_artifact(
        [],
        selector_kind="learned",
        selected_configuration={},
        provenance={},
        cost={"measurement_id": measurement_id},
        output_root=artifact_root,
    )
    evidence = {
        "version": "g4-materialization-cost-v1",
        "measurement_id": measurement_id,
        "passes": True,
        "deterministic_artifact_sha256": deterministic.sha256,
        "learned_artifact_sha256": learned.sha256,
        "runtime": {"wall_seconds": 1.0, "peak_aggregate_rss_bytes": 1},
        "logical_output_scratch_bytes": 1,
        "timed_load_valid": True,
        "limits": dict(MATERIALIZATION_COST_LIMITS),
    }
    evidence_path = tmp_path / "materialization_cost.json"
    evidence_path.write_bytes(canonical_bytes(evidence))
    monkeypatch.setattr(native_manifest, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(native_manifest, "NATIVE_PERIOD_ARTIFACT_ROOT", artifact_root)
    return evidence_path, evidence


def _completed_predecessor(
    tmp_path: Path,
    ledger: dict[str, object],
    *,
    winner_row_id: str,
    winner_recall: float = 0.2,
) -> tuple[Path, list[Path]]:
    ledger_path = tmp_path / f"{ledger['stage']}.json"
    ledger_path.write_bytes(canonical_bytes(ledger))
    source_closure, data_identity = _authenticated_execution_identity()
    run_directories = []
    for row in ledger["rows"]:
        payload = {
            "ledger_sha256": ledger["sha256"],
            "row_id": row["id"],
            "job": row["job"],
            "source_closure": deepcopy(source_closure),
            "data_identity": deepcopy(data_identity),
        }
        run_directory = tmp_path / row["job"]["run_name"]
        run_directory.mkdir()
        contract = payload | {"ledger_path": str(ledger_path.resolve())}
        (run_directory / "g4_job.json").write_text(json.dumps(contract))
        horizon = row["job"]["lr_schedule_horizon_epochs"]
        (run_directory / "training_metadata.json").write_text(
            json.dumps(
                {
                    "lr_schedule_horizon_epochs": horizon,
                    "num_epochs": horizon,
                    "max_epochs": horizon,
                    "epochs_trained": horizon,
                    "lr_horizon_complete": True,
                    "selection_resolved": True,
                    "best_epoch": 1,
                    "batch_size": 512,
                    "embedding_learning_rate": row["job"]["embedding_learning_rate"],
                    "deep_learning_rate": row["job"]["deep_learning_rate"],
                }
            )
        )
        recall = winner_recall if row["id"] == winner_row_id else 0.1
        (run_directory / "sweep.log").write_text(
            f"epoch 0 finished epoch/val_true.recall@100={recall} "
            "epoch/val.loss=0.5\n"
        )
        run_directories.append(run_directory)
    return ledger_path, run_directories


def test_native500m_control_is_exact_two_layer_g1_aggregate() -> None:
    experiment = build_native500m_control(
        run_name="g4_native500m_test",
        deep_learning_rate=DEEP_LEARNING_RATE_ANCHOR,
    )

    assert experiment.size == "500m"
    assert experiment.user_sample is None
    assert experiment.dataloader.batch_size == 512
    assert experiment.embedding_learning_rate == EMBEDDING_LEARNING_RATE
    assert experiment.deep_learning_rate == DEEP_LEARNING_RATE_ANCHOR
    assert experiment.lr_schedule_horizon_epochs == 15
    assert experiment.num_epochs == 15
    assert experiment.lr_schedule.shape == "cosine"
    assert experiment.lr_schedule.warmup_fraction == 0.05
    assert experiment.lr_schedule.optimizer_group_scope == "deep_only"
    assert experiment.transformer.dim == 64
    assert experiment.item_embedding_dim == 64
    assert experiment.transformer.num_layers == 2
    assert experiment.transformer.nhead == 2
    assert experiment.transformer.num_kv_heads == 1
    assert experiment.transformer.ffn == "swiglu"
    assert experiment.transformer.ffn_intermediate_dim == 192
    assert experiment.transformer.attention_window is None
    assert experiment.mup_base_dim == 16
    assert experiment.mup_delta_dim == 32
    assert experiment.num_in_batch_negatives == 2048
    assert experiment.final_ranking_evidence_group == "g4-native500m"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["baseline"].__setitem__("attention_heads", 4),
        lambda document: document["baseline"].__setitem__("unexpected", True),
        lambda document: document["historical_lineage_policy"].__setitem__(
            "native50m_metrics_reused", True
        ),
        lambda document: document["historical_lineage_policy"].__setitem__(
            "unexpected", True
        ),
    ],
    ids=["baseline-value", "baseline-extra", "history-value", "history-extra"],
)
def test_control_manifest_rejects_partial_or_extended_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    document = json.loads(native_manifest.CONTROL_MANIFEST_PATH.read_text())
    mutation(document)
    path = tmp_path / "control_manifest.json"
    path.write_bytes(canonical_bytes(document))
    monkeypatch.setattr(native_manifest, "CONTROL_MANIFEST_PATH", path)

    with pytest.raises(ValueError, match="control manifest"):
        native_manifest.load_control_manifest()


def test_native500m_control_has_authoritative_selected_g1_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G1_AGGREGATE_RUN", G1_SELECTED_RUN)
    monkeypatch.setenv("G1_DATASET_SIZE", "500m")
    monkeypatch.setenv("G1_MAX_EPOCHS", "20")
    monkeypatch.setenv("G1_VARIANT", "baseline")
    g1 = runpy.run_path(
        str(ROOT / "experiments/g1_sasrec_item_ids_likes/configs/aggregate_variant.py")
    )["experiment"]
    g4 = build_native500m_control(
        run_name="g4_control_trial_02_native500m",
        deep_learning_rate=DEEP_LEARNING_RATE_ANCHOR,
    )
    comparable_g4 = replace(
        g4,
        run_name=g1.run_name,
        dataloader=replace(g4.dataloader, batch_size=1280),
        transformer=replace(g4.transformer, num_layers=4),
        final_ranking_evidence_group=None,
    )
    base_fields = fields(MuTransferGenerationExperiment)

    assert g1.run_name == G1_SELECTED_RUN
    assert g1.transformer.ffn == "swiglu"
    assert g1.transformer.ffn_intermediate_dim == 192
    assert {
        field.name: getattr(comparable_g4, field.name) for field in base_fields
    } == {field.name: getattr(g1, field.name) for field in base_fields}


def test_native500m_treatment_changes_only_positive_construction() -> None:
    control = build_native500m_control(
        run_name="control",
        deep_learning_rate=DEEP_LEARNING_RATE_ANCHOR,
    )
    treatment = build_native500m_treatment(
        run_name="rq2",
        deep_learning_rate=DEEP_LEARNING_RATE_ANCHOR,
        objective={"id": "rq2_next10", "event_lookahead": 10},
        valid_positive_mask_mode="next_10_unique",
    )

    assert treatment.objective_id == "rq2_next10"
    assert treatment.objective_event_lookahead == 10
    for name in (
        "size",
        "dataloader",
        "embedding_learning_rate",
        "deep_learning_rate",
        "lr_schedule_horizon_epochs",
        "num_epochs",
        "transformer",
        "item_embedding_dim",
        "mup_base_dim",
        "mup_delta_dim",
        "negative_sampling",
        "num_in_batch_negatives",
    ):
        assert getattr(treatment, name) == getattr(control, name)


def test_native500m_rq3_period_count_is_fixed_to_one() -> None:
    with pytest.raises(ValueError, match="period_count must be 1"):
        build_native500m_treatment(
            run_name="rq3",
            deep_learning_rate=DEEP_LEARNING_RATE_ANCHOR,
            objective={
                "id": "rq3_learned_hard",
                "selector_artifact_sha256": "a" * 64,
                "period_count": 2,
            },
            valid_positive_mask_mode="selected_period_union_unique",
        )


@pytest.mark.parametrize("stage", ["control_tuning", "rq1_tuning", "rq2_tuning"])
def test_frozen_base_ledgers_move_only_deep_learning_rate(stage: str) -> None:
    ledger = load_frozen_ledger(PROTOCOL_ROOT / "ledgers" / f"{stage}.json")

    assert ledger["dataset_size"] == "500m"
    assert ledger["stage"] == stage
    assert tuple(row["job"]["deep_learning_rate"] for row in ledger["rows"]) == (
        BASE_DEEP_LEARNING_RATES
    )
    assert all(
        row["job"]["dataloader"] == {"batch_size": 512} for row in ledger["rows"]
    )
    assert all(
        row["job"]["embedding_learning_rate"] == EMBEDDING_LEARNING_RATE
        for row in ledger["rows"]
    )
    assert all(row["job"]["lr_schedule_horizon_epochs"] == 15 for row in ledger["rows"])
    assert all(
        row["job"]["objective"].get("period_count") in {None, 1}
        for row in ledger["rows"]
    )


def test_control_retry_revision_changes_only_operational_identity() -> None:
    base = compile_base_ledger("control_tuning")
    retry = compile_base_ledger("control_tuning", retry_revision=1)

    assert (
        load_frozen_ledger(PROTOCOL_ROOT / "ledgers/control_tuning_retry1.json")
        == retry
    )
    assert retry["retry_revision"] == 1
    assert retry["retry_incident"]["batch_id"] == ("fa9ae431997d4b968ea3596bb4c0aa6f")
    assert [row["id"] for row in retry["rows"]] == [
        "control_tuning:retry1:01",
        "control_tuning:retry1:02",
        "control_tuning:retry1:03",
    ]
    assert [row["job"]["run_name"] for row in retry["rows"]] == [
        "g4_control_trial_01_retry1_native500m",
        "g4_control_trial_02_retry1_native500m",
        "g4_control_trial_03_retry1_native500m",
    ]
    base_contract = deepcopy(base)
    retry_contract = deepcopy(retry)
    for document in (base_contract, retry_contract):
        document.pop("rows")
        document.pop("sha256")
    retry_contract.pop("retry_revision")
    retry_contract.pop("retry_incident")
    assert retry_contract == base_contract
    for base_row, retry_row in zip(base["rows"], retry["rows"]):
        base_job = deepcopy(base_row["job"])
        retry_job = deepcopy(retry_row["job"])
        base_job.pop("run_name")
        retry_job.pop("run_name")
        retry_job["protocol"].pop("retry_revision")
        assert retry_job == base_job

    specification = build_queue_specification(
        PROTOCOL_ROOT / "ledgers/control_tuning_retry1.json"
    )
    payloads = [
        _compiled_payload(specification, index)
        for index in range(len(specification["jobs"]))
    ]
    assert [payload["row_id"] for payload in payloads] == [
        row["id"] for row in retry["rows"]
    ]
    assert all(payload["ledger_sha256"] == retry["sha256"] for payload in payloads)
    assert all(
        payload["source_closure"] == payloads[0]["source_closure"]
        and payload["data_identity"] == payloads[0]["data_identity"]
        for payload in payloads
    )
    validate_native_source_closure(payloads[0]["source_closure"])
    validate_native500m_data_identity(
        payloads[0]["data_identity"], build_experiment(payloads[0]["job"])
    )


def test_control_retry2_changes_only_operational_identity_and_selects_one_row() -> None:
    base = compile_base_ledger("control_tuning")
    retry = compile_base_ledger("control_tuning", retry_revision=2)

    assert (
        load_frozen_ledger(PROTOCOL_ROOT / "ledgers/control_tuning_retry2.json")
        == retry
    )
    assert retry["retry_incident"]["batch_id"] == "a02f5ffba08d4fb9bcb9260ac149bb40"
    for base_row, retry_row in zip(base["rows"], retry["rows"]):
        base_job = deepcopy(base_row["job"])
        retry_job = deepcopy(retry_row["job"])
        base_job.pop("run_name")
        retry_job.pop("run_name")
        retry_job["protocol"].pop("retry_revision")
        assert retry_job == base_job

    selected = retry["rows"][0]
    specification = build_queue_specification(
        PROTOCOL_ROOT / "ledgers/control_tuning_retry2.json",
        row_id=selected["id"],
    )
    assert len(specification["jobs"]) == 1
    assert _compiled_payload(specification)["row_id"] == selected["id"]


def test_queue_specification_rejects_unknown_row() -> None:
    with pytest.raises(ValueError, match="no unique row"):
        build_queue_specification(
            PROTOCOL_ROOT / "ledgers/control_tuning.json", row_id="missing"
        )


def test_control_retry2_queue_specification_requires_one_explicit_row() -> None:
    with pytest.raises(ValueError, match="requires one explicit row"):
        build_queue_specification(PROTOCOL_ROOT / "ledgers/control_tuning_retry2.json")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ledger: ledger.__setitem__("retry_revision", 0),
        lambda ledger: ledger["retry_incident"].__setitem__("sha256", "0" * 64),
        lambda ledger: ledger["rows"][0]["job"].__setitem__(
            "run_name", "g4_control_trial_01_native500m"
        ),
        lambda ledger: ledger["rows"][0]["job"].__setitem__(
            "deep_learning_rate", 0.123
        ),
    ],
    ids=["revision", "incident", "run-name", "scientific-setting"],
)
def test_frozen_control_retry_rejects_resealed_mutations(
    tmp_path: Path, mutation
) -> None:
    ledger = compile_base_ledger("control_tuning", retry_revision=1)
    mutation(ledger)
    path = tmp_path / "forged-retry.json"
    _write_resealed(path, ledger)

    with pytest.raises(ValueError, match="compiler-equivalent"):
        load_frozen_ledger(path)


@pytest.mark.parametrize(
    ("stage", "revision"),
    [("rq1_tuning", 1), ("control_tuning", 3), ("control_tuning", True)],
)
def test_control_retry_revision_is_narrowly_authorized(
    stage: str, revision: int
) -> None:
    with pytest.raises(ValueError, match="retry revision"):
        compile_base_ledger(stage, retry_revision=revision)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ledger: ledger["rows"][0]["job"].__setitem__("seed", 43),
        lambda ledger: ledger["rows"][0]["job"].__setitem__(
            "deep_learning_rate", 0.123
        ),
        lambda ledger: ledger.__setitem__("stage", "rq2_tuning"),
        lambda ledger: ledger["rows"][0]["job"]["objective"].__setitem__(
            "id", "rq2_next10"
        ),
        lambda ledger: ledger.__setitem__("unexpected", True),
        lambda ledger: ledger["rows"].pop(),
        lambda ledger: ledger["rows"].reverse(),
        lambda ledger: ledger["rows"][0].__setitem__("id", "forged"),
    ],
    ids=[
        "seed",
        "learning-rate",
        "stage",
        "objective",
        "extra",
        "row-count",
        "row-order",
        "row-id",
    ],
)
def test_frozen_ledger_rejects_resealed_noncompiler_documents(
    tmp_path: Path, mutation
) -> None:
    ledger = compile_base_ledger("control_tuning")
    mutation(ledger)
    path = tmp_path / "malicious.json"
    _write_resealed(path, ledger)

    with pytest.raises(ValueError, match="compiler-equivalent"):
        load_frozen_ledger(path)


def test_frozen_ledger_accepts_dynamic_rq3_materialization_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path, evidence = _materialization_fixture(tmp_path, monkeypatch)
    ledger = compile_base_ledger(
        stage="rq3_learned_hard_tuning",
        selector_artifact_sha256=evidence["learned_artifact_sha256"],
        materialization_evidence_path=evidence_path,
    )
    path = tmp_path / "rq3.json"
    path.write_bytes(canonical_bytes(ledger))

    assert load_frozen_ledger(path) == ledger
    assert all(
        row["job"]["materialization_evidence"] == ledger["materialization_evidence"]
        for row in ledger["rows"]
    )


def test_rq3_ledger_rejects_noncanonical_failing_or_changed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path, evidence = _materialization_fixture(tmp_path, monkeypatch)
    evidence_path.write_bytes(canonical_bytes(evidence) + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        compile_base_ledger(
            "rq3_learned_hard_tuning",
            selector_artifact_sha256=evidence["learned_artifact_sha256"],
            materialization_evidence_path=evidence_path,
        )

    evidence_path.write_bytes(canonical_bytes(evidence))
    ledger = compile_base_ledger(
        "rq3_learned_hard_tuning",
        selector_artifact_sha256=evidence["learned_artifact_sha256"],
        materialization_evidence_path=evidence_path,
    )
    ledger_path = tmp_path / "rq3.json"
    ledger_path.write_bytes(canonical_bytes(ledger))
    evidence_path.write_bytes(canonical_bytes(evidence | {"passes": False}))
    with pytest.raises(ValueError, match="compiler-equivalent"):
        load_frozen_ledger(ledger_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ledger: ledger.__setitem__("direction", "upper"),
        lambda ledger: ledger.__setitem__("round_number", 2),
        lambda ledger: ledger.__setitem__("base_stage", "rq1_tuning"),
    ],
    ids=["direction", "round", "base-stage"],
)
def test_frozen_boundary_ledger_rejects_resealed_contract_mutations(
    tmp_path: Path, mutation
) -> None:
    base = compile_base_ledger("control_tuning")
    ledger_path, run_directories = _completed_predecessor(
        tmp_path, base, winner_row_id=base["rows"][0]["id"]
    )
    ledger = compile_boundary_ledger(
        stage="control_tuning",
        direction="lower",
        round_number=1,
        predecessor_ledger_paths=[ledger_path],
        candidate_run_directories=run_directories,
    )
    mutation(ledger)
    path = tmp_path / "malicious-boundary.json"
    _write_resealed(path, ledger)

    with pytest.raises(ValueError, match="compiler-equivalent"):
        load_frozen_ledger(path)


def test_boundary_ledgers_are_directional_bounded_and_evidence_gated(
    tmp_path: Path,
) -> None:
    lower_base = compile_base_ledger("control_tuning")
    lower_path, lower_runs = _completed_predecessor(
        tmp_path, lower_base, winner_row_id=lower_base["rows"][0]["id"]
    )
    upper_root = tmp_path / "upper"
    upper_root.mkdir()
    upper_base = compile_base_ledger("rq1_tuning")
    upper_path, upper_runs = _completed_predecessor(
        upper_root, upper_base, winner_row_id=upper_base["rows"][-1]["id"]
    )
    lower = compile_boundary_ledger(
        stage="control_tuning",
        direction="lower",
        round_number=1,
        predecessor_ledger_paths=[lower_path],
        candidate_run_directories=lower_runs,
    )
    upper = compile_boundary_ledger(
        stage="rq1_tuning",
        direction="upper",
        round_number=1,
        predecessor_ledger_paths=[upper_path],
        candidate_run_directories=upper_runs,
    )

    assert [row["job"]["deep_learning_rate"] for row in lower["rows"]] == [
        DEEP_LEARNING_RATE_ANCHOR / 8,
        DEEP_LEARNING_RATE_ANCHOR / 4,
    ]
    assert [row["job"]["deep_learning_rate"] for row in upper["rows"]] == [
        DEEP_LEARNING_RATE_ANCHOR * 4,
        DEEP_LEARNING_RATE_ANCHOR * 8,
    ]
    assert "predecessor_evidence" in lower
    with pytest.raises(ValueError, match="round_number"):
        compile_boundary_ledger(
            stage="control_tuning", direction="lower", round_number=3
        )


def test_boundary_requires_complete_outer_winner_and_round_sequence(
    tmp_path: Path,
) -> None:
    base = compile_base_ledger("control_tuning")
    base_path, base_runs = _completed_predecessor(
        tmp_path, base, winner_row_id=base["rows"][0]["id"]
    )
    with pytest.raises(ValueError, match="incomplete"):
        compile_boundary_ledger(
            stage="control_tuning",
            direction="lower",
            round_number=1,
            predecessor_ledger_paths=[base_path],
            candidate_run_directories=base_runs[:-1],
        )
    with pytest.raises(ValueError, match="requested outer edge"):
        compile_boundary_ledger(
            stage="control_tuning",
            direction="upper",
            round_number=1,
            predecessor_ledger_paths=[base_path],
            candidate_run_directories=base_runs,
        )

    round_one = compile_boundary_ledger(
        stage="control_tuning",
        direction="lower",
        round_number=1,
        predecessor_ledger_paths=[base_path],
        candidate_run_directories=base_runs,
    )
    round_one_path, round_one_runs = _completed_predecessor(
        tmp_path,
        round_one,
        winner_row_id=round_one["rows"][0]["id"],
        winner_recall=0.3,
    )
    round_two = compile_boundary_ledger(
        stage="control_tuning",
        direction="lower",
        round_number=2,
        predecessor_ledger_paths=[base_path, round_one_path],
        candidate_run_directories=[*base_runs, *round_one_runs],
    )

    assert [row["job"]["deep_learning_rate"] for row in round_two["rows"]] == [
        DEEP_LEARNING_RATE_ANCHOR / 32,
        DEEP_LEARNING_RATE_ANCHOR / 16,
    ]


@pytest.mark.parametrize("identity", ["source_closure", "data_identity"])
def test_boundary_rejects_unauthenticated_predecessor_execution_identity(
    tmp_path: Path, identity: str
) -> None:
    base = compile_base_ledger("control_tuning")
    ledger_path, run_directories = _completed_predecessor(
        tmp_path, base, winner_row_id=base["rows"][0]["id"]
    )
    for run_directory in run_directories:
        contract_path = run_directory / "g4_job.json"
        contract = json.loads(contract_path.read_text())
        contract[identity] = {"fixture": "forged"}
        contract_path.write_text(json.dumps(contract))

    with pytest.raises(ValueError, match="execution identity"):
        compile_boundary_ledger(
            stage="control_tuning",
            direction="lower",
            round_number=1,
            predecessor_ledger_paths=[ledger_path],
            candidate_run_directories=run_directories,
        )


def test_conditional_rq3_ledger_fixes_period_count_and_authenticates_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path, evidence = _materialization_fixture(tmp_path, monkeypatch)
    ledger = compile_base_ledger(
        "rq3_learned_hard_tuning",
        selector_artifact_sha256=evidence["learned_artifact_sha256"],
        materialization_evidence_path=evidence_path,
    )

    assert ledger["selector_artifact_sha256"] == evidence["learned_artifact_sha256"]
    assert ledger["materialization_evidence"]["sha256"] == canonical_sha256(evidence)
    assert len(ledger["rows"]) == 3
    assert all(row["job"]["objective"]["period_count"] == 1 for row in ledger["rows"])
    assert tuple(row["job"]["deep_learning_rate"] for row in ledger["rows"]) == (
        BASE_DEEP_LEARNING_RATES
    )


def test_queue_specification_is_atomic_native500m_and_ledger_bound() -> None:
    ledger_path = PROTOCOL_ROOT / "ledgers/control_tuning.json"
    specification = build_queue_specification(ledger_path)

    assert specification["version"] == 1
    assert len(specification["jobs"]) == 3
    assert all(
        job["data_group"] == "g4-native500m-likes" for job in specification["jobs"]
    )
    assert all(job["run"].endswith("_native500m") for job in specification["jobs"])
    assert all(
        any(
            value.startswith("G4_NATIVE500M_LEDGER_PATH=")
            for value in job["environment"]
        )
        for job in specification["jobs"]
    )
    json.dumps(specification, allow_nan=False)
    payload = _compiled_payload(specification)
    assert payload["source_closure"] == build_native_source_closure()
    assert payload["data_identity"]["dataset_size"] == "500m"
    assert {
        "experiments/g4_future_items/configs/native500m.py",
        "experiments/g4_future_items/targets.py",
        "experiments/g4_future_items/protocol/native500m/manifest.py",
        "experiments/g4_future_items/launchers/native500m.py",
        "experiments/g4_future_items/launchers/run_native500m.py",
    } <= set(payload["source_closure"]["paths"])
    encoded_job = next(
        value
        for value in specification["jobs"][0]["environment"]
        if value.startswith("G4_NATIVE500M_JOB_B64=")
    )
    assert len(encoded_job.encode()) < 64 * 1024


def test_queue_specification_is_byte_deterministic_for_exact_retry() -> None:
    ledger_path = PROTOCOL_ROOT / "ledgers/control_tuning.json"

    assert canonical_bytes(build_queue_specification(ledger_path)) == canonical_bytes(
        build_queue_specification(ledger_path)
    )


def test_source_closure_and_runner_reject_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path = PROTOCOL_ROOT / "ledgers/control_tuning.json"
    payload = _compiled_payload(build_queue_specification(ledger_path))
    relative = "experiments/g4_future_items/targets.py"
    payload["source_closure"]["sources"][relative] = "0" * 64
    encoded = base64.urlsafe_b64encode(canonical_bytes(payload)).decode()
    monkeypatch.setenv("G4_NATIVE500M_JOB_B64", encoded)
    monkeypatch.setenv("G4_NATIVE500M_LEDGER_PATH", str(ledger_path))

    with pytest.raises(ValueError, match="source closure"):
        validate_native_source_closure(payload["source_closure"])
    with pytest.raises(RuntimeError, match="source closure"):
        load_compiled_job()


def test_data_identity_binds_artifacts_split_and_catalog(tmp_path: Path) -> None:
    main = tmp_path / "events_remapped.parquet"
    remap = tmp_path / "item_id_remap.parquet"
    main.write_bytes(b"events-v1")
    pl.DataFrame({"compact_id": [0, 3, 2, 3]}).write_parquet(remap)
    experiment = SimpleNamespace(
        artifacts=SimpleNamespace(main_parquet=main),
        dataset_key="native-test",
        validation_cutoff_timestamp=12345,
    )
    identity = resolve_native500m_data_identity(experiment)

    assert identity["split_cutoff_timestamp"] == 12345
    assert identity["mapped_catalog_sha256"] == canonical_sha256([2, 3])
    validate_native500m_data_identity(identity, experiment)
    main.write_bytes(b"events-v2")
    with pytest.raises(ValueError, match="data identity"):
        validate_native500m_data_identity(identity, experiment)


def test_runner_rejects_tampered_data_identity_before_training() -> None:
    ledger_path = PROTOCOL_ROOT / "ledgers/control_tuning.json"
    payload = _compiled_payload(build_queue_specification(ledger_path))
    payload["data_identity"]["split_cutoff_timestamp"] += 1
    experiment = build_experiment(payload["job"])

    with pytest.raises(RuntimeError, match="data identity"):
        validate_compiled_data_identity(payload, experiment)


def test_frozen_ledger_row_builds_the_exact_runtime_experiment() -> None:
    ledger = load_frozen_ledger(PROTOCOL_ROOT / "ledgers/rq1_tuning.json")
    job = ledger["rows"][1]["job"]

    experiment = build_experiment(job)

    assert experiment.run_name == "g4_rq1_24h_trial_02_native500m"
    assert experiment.size == "500m"
    assert experiment.dataloader.batch_size == 512
    assert experiment.embedding_learning_rate == EMBEDDING_LEARNING_RATE
    assert experiment.deep_learning_rate == DEEP_LEARNING_RATE_ANCHOR
    assert experiment.lr_schedule_horizon_epochs == 15
    assert experiment.objective_id == "rq1_24h"


@pytest.mark.parametrize(
    ("component", "field", "value"),
    [
        ("transformer", "dropout", 0.2),
        ("transformer", "norm", "rms"),
        ("dataloader", "val_batch_size", 4096),
        ("runtime", "compile", True),
        ("lr_schedule", "min_lr_fraction", 0.1),
        (None, "restore_best_weights", False),
        (None, "initializer_std", 0.01),
        (None, "mask_false_negatives", True),
        (None, "weight_decay", 0.1),
        (None, "seed", 43),
        (None, "run_name", "another_native500m"),
    ],
)
def test_runtime_validation_rejects_material_control_mutations(
    component: str | None,
    field: str,
    value: object,
) -> None:
    job = compile_base_ledger("control_tuning")["rows"][1]["job"]
    experiment = build_experiment(job)
    if component is None:
        setattr(experiment, field, value)
    else:
        setattr(
            experiment,
            component,
            replace(getattr(experiment, component), **{field: value}),
        )

    with pytest.raises(ValueError, match="runtime experiment"):
        validate_runtime_experiment(experiment, job)


def test_native_treatment_uses_real_prefix_metadata_for_bos_end_cls(
    cpu_attention: None,
) -> None:
    class RecordingFutureIndex:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def candidates(self, **values):
            self.calls.append(values)
            return (
                FutureEvent(values["prefix_timestamp"] + 1, 23),
                FutureEvent(values["prefix_timestamp"] + 2, 24),
            )

    experiment = build_native500m_treatment(
        run_name="g4_native500m_bos_cls_forward",
        deep_learning_rate=DEEP_LEARNING_RATE_ANCHOR,
        objective={"id": "rq2_next10", "event_lookahead": 10},
        valid_positive_mask_mode="next_10_unique",
    )
    experiment.__dict__["artifacts"] = SimpleNamespace(
        item_id_column="compact_item_id",
        user_column="uid",
        timestamp_column="timestamp",
    )
    experiment.__dict__["item_embeddings"] = PrecomputedEmbeddingLookup(
        torch.randn(31, 64), learnable_default=False, strict=False
    )
    experiment.__dict__["device"] = torch.device("cpu")
    experiment.__dict__["validation_cutoff_timestamp"] = 10_000
    experiment.__dict__["_offline_item_probabilities"] = torch.full((32,), 1 / 32)
    future_index = RecordingFutureIndex()
    experiment.__dict__["future_event_index"] = future_index
    criterion = experiment.create_criterion().train()
    model_outputs: list[dict[str, torch.Tensor]] = []
    criterion.model.register_forward_hook(
        lambda _module, _inputs, output: model_outputs.append(output)
    )
    observed_queries: list[torch.Tensor] = []
    observed_positive_ids: list[torch.Tensor] = []
    observed_acceptable: list[tuple[torch.Tensor, torch.Tensor]] = []
    rng_prefixes: list[tuple[int, int, int]] = []
    original_logits = criterion.loss.logits
    original_query_generator = criterion.targets._query_generator

    def capture_logits(query_repr, positive_repr, positive_ids, *args, **kwargs):
        observed_queries.append(query_repr.detach().clone())
        observed_positive_ids.append(positive_ids.detach().clone())
        observed_acceptable.append(
            (
                kwargs["acceptable_positive_ids"].detach().clone(),
                kwargs["acceptable_positive_offsets"].detach().clone(),
            )
        )
        return original_logits(query_repr, positive_repr, positive_ids, *args, **kwargs)

    def capture_query_generator(uid, prefix_timestamp, prefix_item_id):
        rng_prefixes.append((uid, prefix_timestamp, prefix_item_id))
        return original_query_generator(uid, prefix_timestamp, prefix_item_id)

    criterion.loss.logits = capture_logits
    criterion.targets._query_generator = capture_query_generator
    batch = {
        "int_columns": {
            "compact_item_id": scalar_feature(
                torch.tensor([9, 10, 11, 19, 20, 21, 29])
            ),
            "uid": scalar_feature(torch.tensor([17, 17, 17, 27, 27, 27, 37])),
            OCCURRENCE_POSITION_COLUMN: scalar_feature(
                torch.tensor([4, 5, 6, 8, 9, 10, 12])
            ),
        },
        "float_columns": {},
        "cumulative_lens": torch.tensor([0, 3, 6, 7], dtype=torch.int32),
        "timestamp": torch.tensor([100, 101, 102, 200, 201, 202, 300]),
    }

    result = criterion(batch)
    result["loss"].backward()

    assert torch.isfinite(result["loss"])
    gradients = [
        parameter.grad
        for parameter in criterion.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    assert result[LOSS_DENOMINATOR] == 7
    assert [
        (call["uid"], call["occurrence_position"], call["prefix_item_id"])
        for call in future_index.calls
    ] == [(17, 4, 9), (17, 5, 10), (27, 8, 19), (27, 9, 20)]
    assert rng_prefixes == [
        (17, 100, 9),
        (17, 101, 10),
        (27, 200, 19),
        (27, 201, 20),
    ]
    assert observed_positive_ids[0][[0, 3, 6]].tolist() == [9, 19, 29]
    assert set(observed_positive_ids[0][[1, 2, 4, 5]].tolist()) <= {23, 24}
    acceptable_ids, acceptable_offsets = observed_acceptable[0]
    for index, positive_id in enumerate(observed_positive_ids[0].tolist()):
        start, end = acceptable_offsets[index : index + 2].tolist()
        acceptable = set(acceptable_ids[start:end].tolist())
        assert positive_id in acceptable
        assert acceptable == ({23, 24} if index in {1, 2, 4, 5} else {positive_id})
    output = model_outputs[0]
    control_pairs = NextItemTargets()(output)
    assert control_pairs.group_sizes.tolist() == [3, 3, 1]
    torch.testing.assert_close(observed_queries[0], control_pairs.query_repr)
