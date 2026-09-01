from __future__ import annotations

from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import subprocess

import pytest
import torch

from dcn.config import SemanticHistoryExperiment
from dcn.config.settings import transformer_metadata
from dcn.eval.ranking_evidence import RankingEvidence, write_ranking_evidence
from dcn.semantic import SemanticCodes
from dcn.training_metadata import (
    GENERATION_TRAINING_SEMANTICS_REVISION,
    TIMESTAMP_BIN_SEMANTICS_REVISION,
)
from experiments.g6_rqkmeans_history.configs.rq0 import (
    build_control,
    build_semantic_treatment,
)
from experiments.g6_rqkmeans_history.launchers.compiled import (
    build_experiment,
    decode_compiled_job,
    encode_compiled_job,
)
from experiments.g6_rqkmeans_history.launchers.optuna_workflow import (
    CompiledManifestWriter,
    OptunaStudyWorkflow,
    ProgramResult,
    _write_program_slices,
    semantic_promotion_eligible,
)
from experiments.g6_rqkmeans_history.protocol.evidence import (
    VerifiedArtifact,
    archive_run_artifact,
    artifact_state,
    inference_cost_contract,
    load_verified_artifact,
    select_best,
)
from experiments.g6_rqkmeans_history.protocol.manifest import (
    APPROVED_MANIFEST_PATH,
    INITIAL_RUNS,
    MAX_RUNS,
    CompiledJob,
    RANKING_EVIDENCE_GROUP,
    approved_manifest,
    compile_cap_continuation,
    validate_compiled_job,
    validate_approved_manifest,
)
from experiments.g6_rqkmeans_history.protocol.optuna_driver import (
    G6Rq0OptunaDriver,
    Selection,
)


SEMANTIC_METHODS = (
    "learned_sid_event",
    "item_frozen_sid_event",
    "item_learned_frozen_sid_event",
    "learned_sid_tokens",
    "learned_frozen_sid_tokens",
    "frozen_sid_tokens",
    "interleaved_item_sid_tokens",
)


def _selection(compiled: CompiledJob, objective: float = 0.2) -> Selection:
    return Selection(compiled=compiled, objective=objective, selection_resolved=True)


def _driver(path: Path, *, seed: int = 42) -> G6Rq0OptunaDriver:
    return G6Rq0OptunaDriver(
        path,
        feasible_training_batches=(128, 256, 512),
        validation_batch_size=1024,
        seed=seed,
    )


def _primary_selection(tmp_path: Path) -> Selection:
    driver = _driver(tmp_path / "primary.sqlite3")
    compiled = driver.next_primary_control()
    assert compiled is not None
    driver.tell(compiled, 0.2, tmp_path / "primary.json")
    return _selection(compiled)


def _write_artifact(
    compiled: CompiledJob,
    logs_root: Path,
    *,
    recall: float = 0.2,
    ndcg: float = 0.1,
    early_stopped: bool | None = None,
) -> None:
    directory = logs_root / compiled.run_name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "g6_rq0_job.json").write_text(
        json.dumps(compiled.to_contract(approved_manifest()))
    )
    metrics = {
        f"{metric}@{cutoff}": (
            recall
            if metric == "recall" and cutoff == 100
            else ndcg if metric == "ndcg" and cutoff == 100 else 0.1
        )
        for metric in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
        for cutoff in (10, 50, 100)
    }
    builder = compiled.parameters.get("builder")
    semantic = compiled.approved.stage in {"treatment_tuning", "bridge_tuning"} or (
        compiled.approved.stage == "lr_boundary" and builder in {"treatment", "bridge"}
    )
    if semantic:
        levels = compiled.parameters["num_levels"]
        metrics |= {f"sid_exact_recall@{cutoff}": 0.1 for cutoff in (10, 50, 100)}
        metrics |= {
            f"sid_prefix_recall@{cutoff}_l{level}": 0.1
            for cutoff in (10, 50, 100)
            for level in range(1, levels + 1)
        }
    (directory / "final_metrics.json").write_text(json.dumps(metrics))
    best_backbone = compiled.approved.stage in {
        "primary_control_tuning",
        "primary_control_repeats",
        "treatment_tuning",
    } or (
        compiled.approved.stage == "lr_boundary"
        and builder in {"primary_control", "treatment"}
    )
    experiment = build_experiment(compiled)
    transfer = {
        "experiment_class": type(experiment).__name__,
        "mup_base_dim": experiment.mup_base_dim,
        "mup_delta_dim": experiment.mup_delta_dim,
        "mup_base_ffn_dim": experiment.mup_base_ffn_dim,
        "mup_delta_ffn_dim": experiment.mup_delta_ffn_dim,
        "dataset_size": experiment.size,
        "user_sample": None,
        "event_type_filter": experiment.event_type_filter,
        "min_item_interactions_per_item": experiment.min_item_interactions_per_item,
        "drop_unmapped_items": experiment.drop_unmapped_items,
        "validation_interval_seconds": experiment.validation_interval_seconds,
        "day_range": asdict(experiment.day_range),
        "batch_size": experiment.dataloader.batch_size,
        "physical_batch_size": experiment.dataloader.batch_size,
        "gradient_accumulation_steps": (
            experiment.dataloader.gradient_accumulation_steps
        ),
        "effective_batch_size": experiment.dataloader.effective_batch_size,
        "model_dim": experiment.model_dim,
        "item_embedding_dim": experiment.item_embedding_dim,
        "max_seq_len": experiment.max_seq_len,
        "window": experiment.window,
        "bos": experiment.bos,
        "cls_token": experiment.effective_cls_token_mode != "none",
        "cls_token_mode": experiment.effective_cls_token_mode,
        "timestamp_delta": experiment.timestamp_delta,
        "timestamp_combination": experiment.timestamp_combination,
        "timestamp_num_bins": experiment.timestamp_num_bins,
        "per_layer_item_embeddings": experiment.per_layer_item_embeddings,
        "per_layer_item_features": experiment.effective_per_layer_item_features,
        "per_layer_item_feature_dim": experiment.per_layer_item_feature_dim,
        "negative_sampling": experiment.negative_sampling,
        "num_in_batch_negatives": experiment.num_in_batch_negatives,
        "logq_correction": experiment.logq_correction,
        "random_negative_fraction": experiment.random_negative_fraction,
        "logq_alpha": experiment.logq_alpha,
        "correct_positive_logq": experiment.correct_positive_logq,
        "mask_false_negatives": experiment.mask_false_negatives,
        "exclude_own_group_negatives": experiment.exclude_own_group_negatives,
        "dense_random_negative_scores": experiment.dense_random_negative_scores,
        "eval_ks": list(experiment.eval_ks),
        "eval_max_users": experiment.eval_max_users,
        "eval_every_n_epochs": experiment.eval_every_n_epochs,
        "early_stopping_patience": experiment.early_stopping_patience,
        "early_stopping_min_delta": experiment.early_stopping_min_delta,
        "early_stopping_metric": experiment.checkpointing.best_metric_name,
        "early_stopping_metric_prefix": experiment.checkpointing.best_metric_prefix,
        "selection_k": experiment.selection_k,
        "evaluation_catalog": experiment.evaluation_catalog,
        "exclude_seen_from_evaluation": experiment.exclude_seen_from_evaluation,
        "restore_best_weights": experiment.restore_best_weights,
        "adaptive_schedule_early_stopping": (
            experiment.adaptive_schedule_early_stopping
        ),
        "transformer": transformer_metadata(experiment.transformer),
        "lr_schedule": asdict(experiment.lr_schedule),
    }
    if experiment.timestamp_delta == "bins":
        transfer["timestamp_bin_semantics_revision"] = TIMESTAMP_BIN_SEMANTICS_REVISION
    if experiment.lr_schedule.requires_horizon:
        transfer["lr_schedule_horizon_epochs"] = experiment.lr_schedule_horizon_epochs
    resolved_early_stop = not best_backbone if early_stopped is None else early_stopped
    metadata = {
        "selection_resolved": best_backbone or resolved_early_stop,
        "training_semantics_revision": GENERATION_TRAINING_SEMANTICS_REVISION,
        "dataset_size": "50m",
        "seed": compiled.approved.seed,
        "batch_size": compiled.parameters["batch_size"],
        "val_batch_size": compiled.parameters["validation_batch_size"],
        "embedding_learning_rate": compiled.parameters["embedding_learning_rate"],
        "deep_learning_rate": compiled.parameters["deep_learning_rate"],
        "num_epochs": experiment.num_epochs,
        "max_epochs": experiment.num_epochs,
        "epochs_trained": (
            15 if best_backbone else 8 if resolved_early_stop else experiment.num_epochs
        ),
        "early_stopped": resolved_early_stop,
        "best_epoch_at_cap": False,
        "transfer_invariants": transfer,
    }
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    write_ranking_evidence(
        RankingEvidence(
            user_ids=torch.tensor([1]),
            history_item_ids=torch.tensor([1, 2]),
            history_offsets=torch.tensor([0, 2]),
            relevant_item_ids=torch.tensor([3]),
            relevance_offsets=torch.tensor([0, 1]),
            relevant_train_frequencies=torch.tensor([4]),
            relevant_ranks=torch.tensor([1]),
            max_k=100,
        ),
        context_path=(
            logs_root / ".ranking-evidence" / RANKING_EVIDENCE_GROUP / "context.pt"
        ),
        ranking_path=directory / "ranking_evidence.pt",
    )
    if semantic:
        from dcn.config import SemanticIdConfig

        levels = compiled.parameters["num_levels"]
        codes = compiled.parameters["num_codes"]
        semantic_config = SemanticIdConfig(num_levels=levels, num_codes=codes)
        diagnostics = {
            "semantic_cache_key": semantic_config.cache_key,
            "num_levels": levels,
            "shared_num_codes": codes,
            "semantic_content_width": 128,
            "identifier_collision_rate": 0.1,
            "collided_item_fraction": 0.2,
            "p95_occupied_load": [2.0] * levels,
            "p95_to_mean_occupied_load": [1.1] * levels,
            "intra_code_cosine_similarity": [0.4] * levels,
            "collision_suffix_symbols": 17,
        }
        (directory / "semantic_id_diagnostics.json").write_text(json.dumps(diagnostics))


def test_manifest_accounts_for_the_approved_initial_and_conditional_budgets() -> None:
    manifest = approved_manifest()

    assert INITIAL_RUNS == 165
    assert MAX_RUNS == 245
    assert len(manifest.jobs) == MAX_RUNS
    assert manifest.stage_counts == {
        "primary_control_tuning": 20,
        "original_control_tuning": 12,
        "primary_control_repeats": 9,
        "treatment_tuning": 112,
        "bridge_tuning": 12,
        "lr_boundary": 80,
    }
    assert sum(job.conditional for job in manifest.jobs) == 80
    assert len({job.id for job in manifest.jobs}) == MAX_RUNS
    assert len({job.run_name for job in manifest.jobs}) == MAX_RUNS


def test_manifest_has_seven_equal_budget_treatments_and_four_anchors() -> None:
    jobs = approved_manifest().jobs_for_stage("treatment_tuning")

    assert {job.method for job in jobs} == set(SEMANTIC_METHODS)
    for method in SEMANTIC_METHODS:
        method_jobs = [job for job in jobs if job.method == method]
        assert len(method_jobs) == 16
        assert [job.forced_parameters for job in method_jobs[:4]] == [
            {"num_levels": 2, "num_codes": 256},
            {"num_levels": 3, "num_codes": 64},
            {"num_levels": 3, "num_codes": 128},
            {"num_levels": 4, "num_codes": 32},
        ]


def test_committed_manifest_is_the_exact_approved_document() -> None:
    document = json.loads(APPROVED_MANIFEST_PATH.read_text())
    digest = document.pop("sha256")

    validate_approved_manifest(document)
    assert digest == approved_manifest().sha256


def test_driver_is_deterministic_resumable_and_requires_tell(tmp_path: Path) -> None:
    database = tmp_path / "study.sqlite3"
    driver = _driver(database, seed=17)

    first = driver.next_primary_control()
    assert first is not None
    assert first.parameters["batch_size"] in {128, 256, 512}
    assert first.parameters["validation_batch_size"] == 1024
    assert driver.next_primary_control() == first

    driver.tell(first, 0.31, tmp_path / "first.json")
    second = driver.next_primary_control()
    assert second is not None
    assert second.approved.trial == 1

    resumed = _driver(database, seed=17)
    assert resumed.next_primary_control() == second

    copy = _driver(tmp_path / "copy.sqlite3", seed=17)
    copy_first = copy.next_primary_control()
    assert copy_first == first
    assert copy_first is not None
    copy.tell(copy_first, 0.31, tmp_path / "first.json")
    assert copy.next_primary_control() == second

    changed_preflight = G6Rq0OptunaDriver(
        database,
        feasible_training_batches=(128,),
        validation_batch_size=512,
        seed=17,
    )
    with pytest.raises(ValueError, match="feasible_training_batches changed"):
        changed_preflight.next_primary_control()


def test_dependencies_freeze_the_global_batch_and_selected_representation(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    primary = _primary_selection(tmp_path)

    original = driver.next_original_control(primary)
    treatment = driver.next_treatment("learned_sid_event", primary)
    assert original is not None and treatment is not None
    assert (
        original.parameters["batch_size"] == primary.compiled.parameters["batch_size"]
    )
    assert (
        treatment.parameters["batch_size"] == primary.compiled.parameters["batch_size"]
    )
    assert treatment.parameters["num_codes"] in {32, 64, 128, 256, 512}
    assert "codes_per_level" not in treatment.parameters

    driver.tell(original, 0.19, tmp_path / "original.json")
    driver.tell(treatment, 0.22, tmp_path / "treatment.json")
    bridge = driver.next_bridge(
        primary,
        _selection(original, 0.19),
        _selection(treatment, 0.22),
    )
    assert bridge is not None
    assert bridge.parameters["representation"] == "learned_sid_event"
    assert bridge.parameters["num_levels"] == treatment.parameters["num_levels"]
    assert bridge.parameters["num_codes"] == treatment.parameters["num_codes"]
    assert (
        bridge.parameters["representation_width"]
        == treatment.parameters["representation_width"]
    )


def test_unresolved_or_wrong_dependencies_are_rejected(tmp_path: Path) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    primary = _primary_selection(tmp_path)
    unresolved = replace(primary, selection_resolved=False)

    with pytest.raises(ValueError, match="selection-resolved"):
        driver.next_treatment("learned_sid_event", unresolved)
    with pytest.raises(ValueError, match="approved representation"):
        driver.next_treatment("unknown", primary)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="treatment selection"):
        driver.next_bridge(primary, primary, primary)


def test_primary_repeats_are_seeds_43_through_51(tmp_path: Path) -> None:
    primary = _primary_selection(tmp_path)
    driver = _driver(tmp_path / "driver.sqlite3")

    repeats = driver.compile_primary_repeats(primary)

    assert [job.approved.seed for job in repeats] == list(range(43, 52))
    assert all(
        job.parameters["selected_primary_control_job_id"]
        == primary.compiled.approved.id
        for job in repeats
    )
    assert all(
        job.parameters["batch_size"] == primary.compiled.parameters["batch_size"]
        for job in repeats
    )


def test_lr_boundary_uses_outer_log_tenth_and_four_slots_per_rate(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    primary = _primary_selection(tmp_path)
    source = primary.compiled
    low_and_high = CompiledJob(
        source.approved,
        source.parameters
        | {"embedding_learning_rate": 1e-4, "deep_learning_rate": 0.128},
    )

    boundary = driver.compile_lr_boundaries(_selection(low_and_high))

    assert len(boundary) == 8
    assert [job.approved.forced_parameters["learning_rate"] for job in boundary] == [
        "embedding_learning_rate",
    ] * 4 + ["deep_learning_rate"] * 4
    assert all(
        job.parameters["source_job_id"] == source.approved.id for job in boundary
    )
    assert all(
        job.parameters["batch_size"] == source.parameters["batch_size"]
        for job in boundary
    )
    assert boundary[0].parameters["embedding_learning_rate"] < 1e-4
    assert boundary[-1].parameters["deep_learning_rate"] > 0.128

    middle = CompiledJob(
        source.approved,
        source.parameters
        | {"embedding_learning_rate": 0.01, "deep_learning_rate": 0.01},
    )
    assert driver.compile_lr_boundaries(_selection(middle)) == ()


def test_lr_boundary_contract_freezes_the_selected_source_configuration(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    primary = _primary_selection(tmp_path)
    source = driver.next_treatment("learned_sid_event", primary)
    assert source is not None
    source = CompiledJob(
        source.approved,
        source.parameters | {"embedding_learning_rate": 1e-4},
    )
    boundary = driver.compile_lr_boundaries(_selection(source))[0]
    assert boundary.parameters["source_parameters"] == source.parameters

    alternatives = {
        "batch_size": next(
            value
            for value in (128, 256, 512)
            if value != boundary.parameters["batch_size"]
        ),
        "num_codes": next(
            value
            for value in (32, 64, 128, 256, 512)
            if value != boundary.parameters["num_codes"]
        ),
        "deep_learning_rate": boundary.parameters["deep_learning_rate"] / 2,
        "embedding_learning_rate": boundary.parameters["embedding_learning_rate"] / 2,
        "selected_primary_control_job_id": (
            approved_manifest().jobs_for_stage("primary_control_tuning")[-1].id
        ),
    }
    for name, value in alternatives.items():
        changed = replace(
            boundary,
            parameters=boundary.parameters | {name: value},
        )
        with pytest.raises(ValueError, match="boundary"):
            validate_compiled_job(changed)

    writer = CompiledManifestWriter(tmp_path / "compiled.json")
    writer.append(source)
    changed_source = dict(boundary.parameters["source_parameters"])
    changed_source["batch_size"] = alternatives["batch_size"]
    self_consistent_drift = replace(
        boundary,
        parameters=boundary.parameters
        | {
            "batch_size": alternatives["batch_size"],
            "source_parameters": changed_source,
        },
    )
    validate_compiled_job(self_consistent_drift)
    with pytest.raises(ValueError, match="source contract changed"):
        writer.append(self_consistent_drift)


def test_compiled_contract_round_trips_and_rejects_parameter_drift(
    tmp_path: Path,
) -> None:
    primary = _primary_selection(tmp_path)
    encoded = encode_compiled_job(primary.compiled)

    assert decode_compiled_job(encoded) == primary.compiled
    changed = CompiledJob(
        primary.compiled.approved,
        primary.compiled.parameters | {"num_levels": 3},
    )
    with pytest.raises(ValueError, match="unexpected parameters"):
        validate_compiled_job(changed)


def test_config_plumbing_reconstructs_both_controls_and_semantic_treatments() -> None:
    original = build_control(
        "original_g1",
        batch_size=256,
        validation_batch_size=1024,
        embedding_learning_rate=0.01,
        deep_learning_rate=0.02,
        run_name="original",
    )
    strongest = build_control(
        "best_g1",
        batch_size=256,
        validation_batch_size=1024,
        embedding_learning_rate=0.01,
        deep_learning_rate=0.02,
        run_name="strongest",
    )
    treatment = build_semantic_treatment(
        "learned_sid_event",
        backbone="best_g1",
        batch_size=256,
        validation_batch_size=1024,
        embedding_learning_rate=0.01,
        deep_learning_rate=0.02,
        num_levels=3,
        num_codes=128,
        representation_width=64,
        run_name="treatment",
    )

    assert original.size == strongest.size == treatment.size == "50m"
    assert original.transformer.num_layers == 2
    assert strongest.transformer.num_layers == 4
    assert strongest.transformer.ffn == "swiglu"
    assert strongest.transformer.ffn_intermediate_dim == 192
    assert treatment.semantic.num_levels == 3
    assert treatment.semantic.num_codes == 128
    assert treatment.history_representation == "learned_sid_event"
    assert isinstance(treatment, SemanticHistoryExperiment)
    assert original.final_ranking_evidence_group == RANKING_EVIDENCE_GROUP
    assert treatment.final_ranking_evidence_group == RANKING_EVIDENCE_GROUP


def test_compiled_builder_uses_stage_and_lineage_to_choose_the_backbone(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    primary = _primary_selection(tmp_path)
    treatment = driver.next_treatment("frozen_sid_tokens", primary)
    assert treatment is not None

    experiment = build_experiment(treatment)

    assert isinstance(experiment, SemanticHistoryExperiment)
    assert experiment.run_name == treatment.approved.run_name
    assert experiment.seed == treatment.approved.seed
    assert experiment.history_representation == "frozen_sid_tokens"
    assert experiment.transformer.num_layers == 4


def test_verified_semantic_artifact_requires_sid_metrics_and_diagnostics(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    treatment = driver.next_treatment("learned_sid_event", _primary_selection(tmp_path))
    assert treatment is not None
    _write_artifact(treatment, tmp_path)

    verified = load_verified_artifact(treatment, tmp_path)
    assert verified.semantic_diagnostics is not None
    assert (
        verified.semantic_diagnostics["shared_num_codes"]
        == treatment.parameters["num_codes"]
    )

    (verified.path / "semantic_id_diagnostics.json").unlink()
    with pytest.raises(ValueError, match="cannot read"):
        load_verified_artifact(treatment, tmp_path)


def test_artifact_validation_rejects_incomplete_metrics_and_protocol_drift(
    tmp_path: Path,
) -> None:
    compiled = _primary_selection(tmp_path).compiled
    _write_artifact(compiled, tmp_path)
    directory = tmp_path / compiled.approved.run_name
    metrics_path = directory / "final_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    del metrics["coverage@50"]
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="missing coverage@50"):
        load_verified_artifact(compiled, tmp_path)

    _write_artifact(compiled, tmp_path)
    metadata_path = directory / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["transfer_invariants"]["max_seq_len"] = 99
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="transfer invariant max_seq_len changed"):
        load_verified_artifact(compiled, tmp_path)


def test_artifact_validation_requires_immutable_ranking_evidence(
    tmp_path: Path,
) -> None:
    compiled = _primary_selection(tmp_path).compiled
    _write_artifact(compiled, tmp_path)
    (tmp_path / compiled.run_name / "ranking_evidence.pt").unlink()

    assert artifact_state(compiled, tmp_path) == "partial"
    with pytest.raises(ValueError, match="cannot read ranking evidence"):
        load_verified_artifact(compiled, tmp_path)


def test_program_writes_selected_run_slice_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    primary = _primary_selection(tmp_path).compiled
    semantic = driver.next_treatment("learned_sid_event", _selection(primary))
    assert semantic is not None
    logs_root = tmp_path / "logs"
    _write_artifact(primary, logs_root)
    _write_artifact(semantic, logs_root)
    primary_artifact = load_verified_artifact(primary, logs_root)
    semantic_artifact = load_verified_artifact(semantic, logs_root)
    result = ProgramResult(
        primary_control=primary_artifact,
        original_control=primary_artifact,
        treatment_winners={"learned_sid_event": semantic_artifact},
        semantic_winner=semantic_artifact,
        semantic_promoted=True,
        selected_primary_method=semantic_artifact,
        bridge=semantic_artifact,
        bands={"recall@100": 0.01, "ndcg@100": 0.01},
    )
    experiment = build_experiment(semantic)
    levels = experiment.semantic.num_levels
    base_codes = torch.zeros(3, levels, dtype=torch.int64)
    base_codes[2] = 1
    experiment.__dict__["semantic_codes"] = SemanticCodes(
        item_ids=torch.tensor([1, 2, 3]),
        codes=torch.cat(
            [base_codes, torch.tensor([[1], [2], [1]], dtype=torch.int64)], dim=1
        ),
        codes_per_level=(experiment.semantic.num_codes,) * levels + (3,),
    )
    monkeypatch.setattr(experiment, "setup", lambda: None)
    monkeypatch.setattr(
        "experiments.g6_rqkmeans_history.launchers.optuna_workflow.build_experiment",
        lambda compiled: experiment,
    )
    destination = tmp_path / "slices.json"

    _write_program_slices(destination, result, logs_root=logs_root)

    document = json.loads(destination.read_text())
    assert document["selected_job_ids"] == {
        "primary_control": primary.approved.id,
        "semantic_winner": semantic.approved.id,
    }
    assert document["collision_slice"]["collided_item_count"] == 2


def test_selection_uses_recall_band_then_ndcg_then_token_cost(tmp_path: Path) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    primary = _primary_selection(tmp_path)
    event = driver.next_treatment("learned_sid_event", primary)
    tokens = driver.next_treatment("learned_sid_tokens", primary)
    assert event is not None and tokens is not None
    event_artifact = VerifiedArtifact(
        event,
        tmp_path / "event",
        {"recall@100": 0.205, "ndcg@100": 0.1},
        {},
        {"semantic_content_width": 128},
    )
    token_artifact = VerifiedArtifact(
        tokens,
        tmp_path / "tokens",
        {"recall@100": 0.2, "ndcg@100": 0.12},
        {},
        {"semantic_content_width": 128},
    )

    assert (
        select_best(
            [event_artifact, token_artifact],
            recall_band=0.01,
            ndcg_band=0.001,
        )
        == token_artifact
    )
    assert (
        select_best(
            [event_artifact, token_artifact],
            recall_band=0.001,
            ndcg_band=0.001,
        )
        == event_artifact
    )
    assert semantic_promotion_eligible(
        token_artifact,
        event_artifact,
        recall_band=0.001,
        ndcg_band=0.03,
    )
    assert not semantic_promotion_eligible(
        token_artifact,
        event_artifact,
        recall_band=0.01,
        ndcg_band=0.03,
    )


def test_selection_cost_contract_distinguishes_same_token_representations(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    primary = _primary_selection(tmp_path)
    learned = driver.next_treatment("learned_sid_event", primary)
    item_frozen = driver.next_treatment("item_frozen_sid_event", primary)
    assert learned is not None and item_frozen is not None
    learned = CompiledJob(
        learned.approved,
        learned.parameters | {"representation_width": 128},
    )
    shared = learned.parameters | {
        "representation": "item_frozen_sid_event",
    }
    item_frozen = CompiledJob(item_frozen.approved, shared)
    validate_compiled_job(item_frozen)
    learned_artifact = VerifiedArtifact(
        learned,
        tmp_path / "learned",
        {"recall@100": 0.2, "ndcg@100": 0.1},
        {},
        {"semantic_content_width": 128},
    )
    item_frozen_artifact = VerifiedArtifact(
        item_frozen,
        tmp_path / "item_frozen",
        {"recall@100": 0.2, "ndcg@100": 0.1},
        {},
        {"semantic_content_width": 128},
    )

    learned_cost = inference_cost_contract(learned_artifact)
    item_frozen_cost = inference_cost_contract(item_frozen_artifact)

    assert learned_cost.sequence_tokens == 102
    assert learned_cost.sequence_tokens == item_frozen_cost.sequence_tokens
    assert learned_cost != item_frozen_cost
    wider_frozen = replace(
        item_frozen_artifact,
        semantic_diagnostics={"semantic_content_width": 256},
    )
    assert inference_cost_contract(wider_frozen) != item_frozen_cost
    assert item_frozen_cost.sort_key < learned_cost.sort_key
    assert select_best(
        [learned_artifact, item_frozen_artifact],
        recall_band=0,
        ndcg_band=0,
    ) == min(
        (learned_artifact, item_frozen_artifact),
        key=lambda artifact: inference_cost_contract(artifact).sort_key,
    )


def test_compiled_ledger_is_atomic_idempotent_and_fail_closed(tmp_path: Path) -> None:
    compiled = _primary_selection(tmp_path).compiled
    path = tmp_path / "compiled.json"
    writer = CompiledManifestWriter(path)

    writer.append(compiled)
    writer.append(compiled)
    assert path.exists()

    changed = CompiledJob(
        compiled.approved,
        compiled.parameters | {"batch_size": 128},
    )
    with pytest.raises(ValueError, match="compiled job changed"):
        writer.append(changed)


def test_workflow_advances_a_complete_study_through_verified_artifacts(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    logs_root = tmp_path / "logs"

    def submit(compiled_jobs: tuple[CompiledJob, ...]) -> None:
        for index, compiled in enumerate(compiled_jobs):
            _write_artifact(compiled, logs_root, recall=0.1 + index / 1000)

    workflow = OptunaStudyWorkflow(
        driver,
        logs_root=logs_root,
        compiled_path=tmp_path / "compiled.json",
        submit=submit,
    )

    assert workflow.advance(driver.next_primary_control) == 20
    assert driver.next_primary_control() is None
    assert (
        len(
            workflow.artifacts(
                approved_manifest().jobs_for_stage("primary_control_tuning")
            )
        )
        == 20
    )


def test_workflow_never_submits_the_same_semantic_cache_concurrently(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    primary = _primary_selection(tmp_path)
    first = driver.next_treatment("learned_sid_event", primary)
    second = driver.next_treatment("learned_sid_tokens", primary)
    assert first is not None and second is not None
    assert (first.parameters["num_levels"], first.parameters["num_codes"]) == (
        second.parameters["num_levels"],
        second.parameters["num_codes"],
    )
    submitted: list[int] = []

    def submit(compiled_jobs: tuple[CompiledJob, ...]) -> None:
        submitted.append(len(compiled_jobs))
        for compiled in compiled_jobs:
            _write_artifact(compiled, tmp_path / "logs")

    workflow = OptunaStudyWorkflow(
        driver,
        logs_root=tmp_path / "logs",
        compiled_path=tmp_path / "compiled.json",
        submit=submit,
    )

    workflow.run_compiled((first, second))

    assert submitted == [1, 1]


def test_workflow_archives_partial_attempt_under_lock_then_resubmits(
    tmp_path: Path,
) -> None:
    compiled = _primary_selection(tmp_path).compiled
    logs_root = tmp_path / "logs"
    partial = logs_root / compiled.run_name
    partial.mkdir(parents=True)
    (partial / "sweep.log").write_text("preempted")
    submitted: list[str] = []

    def submit(compiled_jobs: tuple[CompiledJob, ...]) -> None:
        for job in compiled_jobs:
            submitted.append(job.run_name)
            _write_artifact(job, logs_root)

    workflow = OptunaStudyWorkflow(
        _driver(tmp_path / "workflow.sqlite3"),
        logs_root=logs_root,
        compiled_path=tmp_path / "compiled.json",
        submit=submit,
    )

    artifact = workflow.run_compiled((compiled,))[0]

    archived = logs_root / "old" / f"{compiled.run_name}.incomplete-001"
    assert submitted == [compiled.run_name]
    assert artifact.path == logs_root / compiled.run_name
    assert (archived / "sweep.log").read_text() == "preempted"


def test_workflow_recovers_a_failed_queue_submission(tmp_path: Path) -> None:
    compiled = _primary_selection(tmp_path).compiled
    logs_root = tmp_path / "logs"
    submissions = 0

    def submit(compiled_jobs: tuple[CompiledJob, ...]) -> None:
        nonlocal submissions
        submissions += 1
        if submissions == 1:
            directory = logs_root / compiled_jobs[0].run_name
            directory.mkdir(parents=True)
            (directory / "sweep.log").write_text("failed")
            raise subprocess.CalledProcessError(1, "queue")
        _write_artifact(compiled_jobs[0], logs_root)

    workflow = OptunaStudyWorkflow(
        _driver(tmp_path / "workflow.sqlite3"),
        logs_root=logs_root,
        compiled_path=tmp_path / "compiled.json",
        submit=submit,
    )

    artifact = workflow.run_compiled((compiled,))[0]

    assert submissions == 2
    assert artifact.path == logs_root / compiled.run_name
    assert (
        logs_root / "old" / f"{compiled.run_name}.incomplete-001" / "sweep.log"
    ).read_text() == "failed"


def test_partial_archive_refuses_an_active_run_lock(tmp_path: Path) -> None:
    compiled = _primary_selection(tmp_path).compiled
    logs_root = tmp_path / "logs"
    directory = logs_root / compiled.run_name
    directory.mkdir(parents=True)
    (directory / "sweep.log").write_text("active")
    lock_path = logs_root / ".run-locks" / f"{compiled.run_name}.lock"
    lock_path.parent.mkdir(parents=True)

    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active training process"):
            archive_run_artifact(compiled, logs_root, reason="incomplete")

    assert directory.is_dir()


def test_partial_archives_use_immutable_attempt_names(tmp_path: Path) -> None:
    compiled = _primary_selection(tmp_path).compiled
    logs_root = tmp_path / "logs"
    first_directory = logs_root / compiled.run_name
    first_directory.mkdir(parents=True)
    (first_directory / "sweep.log").write_text("first")

    first = archive_run_artifact(compiled, logs_root, reason="incomplete")
    second_directory = logs_root / compiled.run_name
    second_directory.mkdir()
    (second_directory / "sweep.log").write_text("second")
    second = archive_run_artifact(compiled, logs_root, reason="incomplete")

    assert first.name.endswith("incomplete-001")
    assert second.name.endswith("incomplete-002")
    assert (first / "sweep.log").read_text() == "first"
    assert (second / "sweep.log").read_text() == "second"


def test_original_backbone_extends_cap_until_early_stopping_resolves(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    original = driver.next_original_control(_primary_selection(tmp_path))
    assert original is not None
    logs_root = tmp_path / "logs"
    submitted: list[CompiledJob] = []

    def submit(compiled_jobs: tuple[CompiledJob, ...]) -> None:
        for job in compiled_jobs:
            submitted.append(job)
            _write_artifact(job, logs_root, early_stopped=job.attempt == 2)

    workflow = OptunaStudyWorkflow(
        driver,
        logs_root=logs_root,
        compiled_path=tmp_path / "compiled.json",
        submit=submit,
    )

    artifact = workflow.run_compiled((original,))[0]

    assert [(job.attempt, job.cap_epochs) for job in submitted] == [
        (0, None),
        (1, 60),
        (2, 90),
    ]
    assert artifact.compiled == compile_cap_continuation(
        compile_cap_continuation(original)
    )
    assert decode_compiled_job(encode_compiled_job(artifact.compiled)) == (
        artifact.compiled
    )
    assert artifact.compiled.parameters == original.parameters
    assert artifact.compiled.approved.seed == original.approved.seed
    assert build_experiment(artifact.compiled).num_epochs == 90
    assert (logs_root / "old" / f"{original.run_name}.cap-exhausted-001").is_dir()
    first_extension = compile_cap_continuation(original)
    assert (
        logs_root / "old" / f"{first_extension.run_name}.cap-exhausted-001"
    ).is_dir()
    assert len(approved_manifest().jobs) == MAX_RUNS


def test_cap_continuations_cover_bridge_and_original_boundary(tmp_path: Path) -> None:
    driver = _driver(tmp_path / "study.sqlite3")
    primary = _primary_selection(tmp_path)
    original = driver.next_original_control(primary)
    treatment = driver.next_treatment("learned_sid_event", primary)
    assert original is not None and treatment is not None
    bridge = driver.next_bridge(primary, _selection(original), _selection(treatment))
    assert bridge is not None
    boundary_source = CompiledJob(
        original.approved,
        original.parameters | {"embedding_learning_rate": 1e-4},
    )
    boundaries = driver.compile_lr_boundaries(_selection(boundary_source))
    boundary = next(
        job
        for job in boundaries
        if job.approved.forced_parameters["learning_rate"] == "embedding_learning_rate"
    )

    for compiled in (bridge, boundary):
        continuation = compile_cap_continuation(compiled)
        assert continuation.cap_epochs == 60
        assert continuation.parameters == compiled.parameters
        assert continuation.approved.seed == compiled.approved.seed
        experiment = build_experiment(continuation)
        assert experiment.transformer.num_layers == 2
        assert experiment.num_epochs == 60
