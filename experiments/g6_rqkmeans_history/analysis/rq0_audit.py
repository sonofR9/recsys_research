from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import optuna
import torch
from optuna.trial import TrialState

from dcn.eval.ranking_evidence import RankingEvidence, load_ranking_evidence
from dcn.semantic import (
    ResidualCodebooks,
    SemanticCodes,
    semantic_id_diagnostics,
)
from dcn.semantic.artifacts import load_item_embeddings
from experiments.g6_rqkmeans_history.analysis.preflight import (
    PreflightEvidence,
    load_preflight_evidence,
    preflight_contract_metadata,
)
from experiments.g6_rqkmeans_history.protocol.evidence import (
    VerifiedArtifact,
    empirical_bands,
    inference_cost_contract,
    load_verified_artifact,
    require_resolved_boundary,
    select_best,
)
from experiments.g6_rqkmeans_history.protocol.manifest import (
    DEEP_LR_BOUNDS,
    EMBEDDING_LR_BOUNDS,
    NUM_CODES,
    NUM_LEVELS,
    RANKING_EVIDENCE_GROUP,
    REPRESENTATIONS,
    REPRESENTATION_WIDTHS,
    CompiledJob,
    approved_manifest,
    boundary_side,
    load_compiled_jobs,
    outside_boundary_rates,
    validate_boundary_source,
)
from experiments.g6_rqkmeans_history.protocol.optuna_driver import (
    G6Rq0OptunaDriver,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_FILES = (
    "g6_rq0_job.json",
    "final_metrics.json",
    "training_metadata.json",
    "ranking_evidence.pt",
    "semantic_id_diagnostics.json",
)
BASE_PARAMETERS = (
    "batch_size",
    "validation_batch_size",
    "embedding_learning_rate",
    "deep_learning_rate",
)
BAND_ROUNDING = "sample standard deviation rounded upward to one significant digit"
CONTENT_EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "generated/datasets/yambda/50m_like_core5_knownitems/embeddings_compact.parquet"
)


def build_audit_evidence(
    *,
    database_path: Path,
    compiled_path: Path,
    logs_root: Path,
    preflight_path: Path,
    bands_path: Path,
    selection_path: Path,
) -> dict[str, object]:
    manifest = approved_manifest()
    preflight = load_preflight_evidence(preflight_path)
    compiled = load_compiled_jobs(compiled_path)
    latest = {}
    for job in compiled:
        previous = latest.get(job.approved.id)
        if previous is None or job.attempt > previous.attempt:
            latest[job.approved.id] = job
    manifest_ids = {job.id for job in manifest.jobs}
    unknown_ids = set(latest) - manifest_ids
    if unknown_ids:
        raise ValueError(
            f"compiled jobs are absent from the manifest: {sorted(unknown_ids)}"
        )
    required_jobs = [job for job in manifest.jobs if not job.conditional]
    missing_ids = {job.id for job in required_jobs} - set(latest)
    if missing_ids:
        raise ValueError(
            f"G6 RQ0 audit omits unconditional manifest jobs: {sorted(missing_ids)}"
        )
    ledger = list(latest.values())
    for job in ledger:
        validate_boundary_source(job, ledger)
        if job.parameters["batch_size"] not in preflight.feasible_training_batches:
            raise ValueError(f"{job.approved.id}: batch was not preflight-feasible")
        if job.parameters["validation_batch_size"] != preflight.validation_batch_size:
            raise ValueError(
                f"{job.approved.id}: validation batch changed from preflight"
            )
    artifacts = [
        load_verified_artifact(job, logs_root)
        for job in sorted(
            latest.values(),
            key=lambda candidate: (
                _manifest_order()[candidate.approved.id],
                candidate.attempt,
            ),
        )
    ]
    ranking_context_path = (
        logs_root / ".ranking-evidence" / RANKING_EVIDENCE_GROUP / "context.pt"
    )
    for artifact in artifacts:
        _verify_ranking_metrics(artifact, ranking_context_path)
    semantic_codebooks = _verify_semantic_codebooks(artifacts)
    bands_document = _read_json(bands_path)
    selection = _read_json(selection_path)
    if selection.get("dataset_size") != "native-50m":
        raise ValueError("G6 RQ0 selection references a different dataset size")
    if selection.get("manifest_sha256") != manifest.sha256:
        raise ValueError("G6 RQ0 selection references a different approved manifest")
    artifacts_by_id = {
        artifact.compiled.approved.id: artifact for artifact in artifacts
    }
    primary_job_id = _verify_selected_row(
        _required_dict(selection, "primary_control"), artifacts_by_id
    )
    band_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.compiled.approved.id == primary_job_id
        or artifact.compiled.approved.stage == "primary_control_repeats"
    ]
    band_seeds = sorted(artifact.compiled.approved.seed for artifact in band_artifacts)
    if bands_document.get("dataset_size") != "native-50m":
        raise ValueError("G6 RQ0 bands reference a different dataset size")
    if bands_document.get("rounding") != BAND_ROUNDING:
        raise ValueError("G6 RQ0 bands use different rounding provenance")
    if bands_document.get("seeds") != band_seeds:
        raise ValueError("published G6 RQ0 band seeds do not match their run artifacts")
    recomputed_bands = empirical_bands(band_artifacts)
    if bands_document.get("bands") != recomputed_bands:
        raise ValueError("published G6 RQ0 bands do not match their run artifacts")
    if selection.get("bands") != recomputed_bands:
        raise ValueError("G6 RQ0 selection uses different empirical bands")
    selected_job_ids = _verify_selection(
        selection,
        artifacts,
        artifacts_by_id,
        recomputed_bands,
    )
    optuna_studies = _verify_optuna_history(
        database_path, artifacts, manifest.sha256, preflight
    )
    _replay_optuna_history(preflight, artifacts_by_id, selected_job_ids)
    stage_counts = Counter(artifact.compiled.approved.stage for artifact in artifacts)
    required_stage_counts = Counter(job.stage for job in required_jobs)
    incomplete_stages = {
        stage: {"expected": expected, "actual": stage_counts[stage]}
        for stage, expected in required_stage_counts.items()
        if stage_counts[stage] < expected
    }
    if incomplete_stages:
        raise ValueError(f"G6 RQ0 stage counts are incomplete: {incomplete_stages}")
    return {
        "schema_version": 1,
        "dataset_size": "native-50m",
        "manifest_sha256": manifest.sha256,
        "preflight": _file_reference(preflight_path),
        "optuna_database": _file_reference(database_path),
        "optuna_studies": optuna_studies,
        "workflow_implementation": {
            name: _file_reference(PROJECT_ROOT / path)
            for name, path in {
                "manifest": "experiments/g6_rqkmeans_history/protocol/manifest.py",
                "driver": "experiments/g6_rqkmeans_history/protocol/optuna_driver.py",
                "selection": "experiments/g6_rqkmeans_history/protocol/evidence.py",
                "audit": "experiments/g6_rqkmeans_history/analysis/rq0_audit.py",
            }.items()
        },
        "candidate_count": len(artifacts),
        "stage_counts": dict(sorted(stage_counts.items())),
        "required_candidate_count": len(required_jobs),
        "required_stage_counts": dict(sorted(required_stage_counts.items())),
        "selection_sha256": _sha256(selection_path),
        "selected_job_ids": selected_job_ids,
        "ranking_context": _file_reference(ranking_context_path),
        "semantic_codebooks": semantic_codebooks,
        "metric_verification": {
            "ranking_derived": list(_ranking_metric_names()),
            "artifact_only": sorted(
                set().union(*(artifact.metrics.keys() for artifact in artifacts))
                - set(_ranking_metric_names())
            ),
        },
        "bands": {
            "published_sha256": _sha256(bands_path),
            "rounding": bands_document.get("rounding"),
            "values": recomputed_bands,
            "inputs": [_artifact_row(artifact) for artifact in band_artifacts],
        },
        "candidates": [_artifact_row(artifact) for artifact in artifacts],
    }


def write_audit_evidence(path: Path, document: dict[str, object]) -> None:
    content = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"existing G6 RQ0 audit evidence differs: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _artifact_row(artifact: VerifiedArtifact) -> dict[str, object]:
    compiled = artifact.compiled
    required_files = ARTIFACT_FILES[:4]
    if artifact.semantic_diagnostics is not None:
        required_files = ARTIFACT_FILES
    missing_files = [
        name for name in required_files if not (artifact.path / name).is_file()
    ]
    if missing_files:
        raise ValueError(
            f"G6 RQ0 artifact {compiled.approved.id} omits files: {missing_files}"
        )
    files = {name: _file_reference(artifact.path / name) for name in required_files}
    inference_cost = inference_cost_contract(artifact)
    return {
        "job_id": compiled.approved.id,
        "stage": compiled.approved.stage,
        "method": compiled.approved.method,
        "seed": compiled.approved.seed,
        "run_name": compiled.run_name,
        "attempt": compiled.attempt,
        "cap_epochs": compiled.cap_epochs,
        "parameters": compiled.parameters,
        "metrics": artifact.metrics,
        "inference_cost": {
            **asdict(inference_cost),
            "total_multiply_accumulates": inference_cost.total_multiply_accumulates,
        },
        "training": {
            name: artifact.metadata.get(name)
            for name in (
                "best_epoch",
                "stopped_epoch",
                "epochs_trained",
                "early_stopped",
                "selection_resolved",
                "best_epoch_at_cap",
            )
        },
        "artifacts": files,
    }


def _verify_ranking_metrics(artifact: VerifiedArtifact, context_path: Path) -> None:
    evidence = load_ranking_evidence(
        context_path, artifact.path / "ranking_evidence.pt"
    )
    recomputed = _ranking_metrics(evidence)
    mismatches = [
        name
        for name, expected in recomputed.items()
        if name not in artifact.metrics
        or not math.isclose(
            artifact.metrics[name], expected, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    if mismatches:
        raise ValueError(
            f"G6 RQ0 artifact {artifact.compiled.approved.id} has ranking-metric "
            f"mismatches: {mismatches}"
        )


def _ranking_metrics(evidence: RankingEvidence) -> dict[str, float]:
    if evidence.max_k < 100:
        raise ValueError("G6 RQ0 ranking evidence does not reach cutoff 100")
    counts = evidence.relevance_offsets[1:] - evidence.relevance_offsets[:-1]
    if bool((counts < 1).any()):
        raise ValueError("G6 RQ0 ranking evidence contains a user without targets")
    num_users = int(evidence.user_ids.shape[0])
    owners = torch.repeat_interleave(torch.arange(num_users), counts)
    ranks = evidence.relevant_ranks
    metrics = {"num_users": float(num_users)}
    for cutoff in (10, 50, 100):
        hits = (ranks > 0) & (ranks <= cutoff)
        hit_counts = torch.zeros(num_users, dtype=torch.float64)
        hit_counts.scatter_add_(0, owners, hits.to(torch.float64))
        dcg = torch.zeros(num_users, dtype=torch.float64)
        discounts = torch.zeros_like(ranks, dtype=torch.float64)
        discounts[hits] = 1.0 / torch.log2(ranks[hits].to(torch.float64) + 1)
        dcg.scatter_add_(0, owners, discounts)
        ideal_discounts = (
            1.0 / torch.log2(torch.arange(cutoff, dtype=torch.float64) + 2)
        ).cumsum(0)
        ideal_dcg = ideal_discounts[counts.clamp_max(cutoff) - 1]
        first_ranks = torch.full((num_users,), cutoff + 1, dtype=torch.int64)
        eligible_ranks = torch.where(hits, ranks, cutoff + 1)
        first_ranks.scatter_reduce_(
            0, owners, eligible_ranks, reduce="amin", include_self=True
        )
        metrics |= {
            f"recall@{cutoff}": float((hit_counts / counts).mean()),
            f"capped_recall@{cutoff}": float(
                (hit_counts / counts.clamp_max(cutoff)).mean()
            ),
            f"ndcg@{cutoff}": float((dcg / ideal_dcg).mean()),
            f"mrr@{cutoff}": float(
                torch.where(
                    first_ranks <= cutoff,
                    first_ranks.to(torch.float64).reciprocal(),
                    0.0,
                ).mean()
            ),
        }
    return metrics


def _ranking_metric_names() -> tuple[str, ...]:
    return (
        "num_users",
        *(
            f"{metric}@{cutoff}"
            for metric in ("recall", "capped_recall", "ndcg", "mrr")
            for cutoff in (10, 50, 100)
        ),
    )


def _verify_semantic_codebooks(
    artifacts: list[VerifiedArtifact],
) -> dict[str, object]:
    verified: dict[str, object] = {}
    dataset_cache_key = preflight_contract_metadata()["workload"]["dataset_cache_key"]
    item_ids, embeddings = load_item_embeddings(CONTENT_EMBEDDINGS_PATH)
    grouped: dict[str, list[VerifiedArtifact]] = {}
    for artifact in artifacts:
        if artifact.semantic_diagnostics is not None:
            grouped.setdefault(
                str(artifact.semantic_diagnostics["semantic_cache_key"]), []
            ).append(artifact)
    for cache_key, cache_artifacts in grouped.items():
        artifact = cache_artifacts[0]
        diagnostics = artifact.semantic_diagnostics
        assert diagnostics is not None
        semantic_dir = (
            PROJECT_ROOT
            / "generated/preprocessed/dataset"
            / str(dataset_cache_key)
            / "semantic"
            / cache_key
        )
        codebooks_path = semantic_dir / "codebooks.pt"
        codes_path = semantic_dir / "codes.pt"
        codebooks = ResidualCodebooks.load(codebooks_path)
        codes = SemanticCodes.load(codes_path)
        if not codes.item_ids.equal(item_ids):
            raise ValueError(f"G6 RQ0 semantic cache {cache_key} changed catalog items")
        expected_shape = (
            artifact.compiled.parameters["num_levels"],
            artifact.compiled.parameters["num_codes"],
            diagnostics["semantic_content_width"],
        )
        if tuple(codebooks.centroids.shape) != expected_shape:
            raise ValueError(
                f"G6 RQ0 artifact {artifact.compiled.approved.id} has a changed "
                "semantic codebook shape"
            )
        expected_levels = artifact.compiled.parameters["num_levels"] + 1
        if (
            codes.num_levels != expected_levels
            or tuple(codes.codes_per_level[:-1])
            != (artifact.compiled.parameters["num_codes"],) * (expected_levels - 1)
            or codes.codes_per_level[-1] != diagnostics["collision_suffix_symbols"]
        ):
            raise ValueError(
                f"G6 RQ0 artifact {artifact.compiled.approved.id} has changed "
                "semantic codes"
            )
        expected_codes = SemanticCodes.with_collision_suffix(
            item_ids,
            codes.codes[:, : int(artifact.compiled.parameters["num_levels"])],
            int(artifact.compiled.parameters["num_codes"]),
        )
        if (
            not expected_codes.item_ids.equal(codes.item_ids)
            or not expected_codes.codes.equal(codes.codes)
            or expected_codes.codes_per_level != codes.codes_per_level
        ):
            raise ValueError(
                f"G6 RQ0 semantic cache {cache_key} changed collision suffixes"
            )
        assignment_check = _verify_nearest_centroid_codes(codebooks, codes, embeddings)
        recomputed_diagnostics = json.loads(
            json.dumps(
                asdict(
                    semantic_id_diagnostics(
                        codes,
                        embeddings,
                        num_base_levels=int(artifact.compiled.parameters["num_levels"]),
                    )
                )
            )
        )
        for candidate in cache_artifacts:
            candidate_diagnostics = candidate.semantic_diagnostics
            assert candidate_diagnostics is not None
            expected_candidate_diagnostics = {
                **recomputed_diagnostics,
                "semantic_cache_key": cache_key,
                "num_levels": codebooks.num_levels,
                "shared_num_codes": codebooks.num_codes,
                "semantic_content_width": codebooks.dim,
                "collision_suffix_symbols": codes.codes_per_level[-1],
            }
            if any(
                candidate_diagnostics.get(name) != value
                for name, value in expected_candidate_diagnostics.items()
            ):
                raise ValueError(
                    f"G6 RQ0 artifact {candidate.compiled.approved.id} has changed "
                    "semantic diagnostics"
                )
            cache_mtime = max(
                codebooks_path.stat().st_mtime_ns, codes_path.stat().st_mtime_ns
            )
            if cache_mtime > (candidate.path / "final_metrics.json").stat().st_mtime_ns:
                raise ValueError(
                    f"G6 RQ0 semantic cache {cache_key} postdates a run artifact"
                )
        contract = {
            "shape": list(codebooks.centroids.shape),
            "codes_per_level": list(codes.codes_per_level),
            "assignment_check": assignment_check,
            "codebooks": _file_reference(codebooks_path),
            "codes": _file_reference(codes_path),
        }
        verified[cache_key] = contract
    return {
        "content_embeddings": _file_reference(CONTENT_EMBEDDINGS_PATH),
        "caches": dict(sorted(verified.items())),
    }


def _verify_nearest_centroid_codes(
    codebooks: ResidualCodebooks,
    codes: SemanticCodes,
    embeddings: torch.Tensor,
) -> list[dict[str, float | int]]:
    residual = embeddings.to(codebooks.centroids.dtype)
    base_codes = codes.codes[:, : codebooks.num_levels]
    checks = []
    for level in range(codebooks.num_levels):
        centroids = codebooks.centroids[level]
        assigned_codes = base_codes[:, level]
        disagreements = 0
        maximum_relative_excess = 0.0
        for start in range(0, len(residual), 4096):
            chunk = residual[start : start + 4096]
            chunk_codes = assigned_codes[start : start + len(chunk)]
            distances = torch.cdist(chunk, centroids)
            assigned = distances.gather(1, chunk_codes[:, None]).squeeze(1)
            minimum = distances.amin(1)
            close = torch.isclose(
                assigned.square(), minimum.square(), rtol=5e-3, atol=1e-6
            )
            if not bool(close.all()):
                raise ValueError("G6 RQ0 semantic codes are not nearest centroids")
            disagreements += int((assigned > minimum).sum())
            maximum_relative_excess = max(
                maximum_relative_excess,
                float(
                    (
                        (assigned.square() - minimum.square())
                        / minimum.square().clamp_min(1e-12)
                    ).max()
                ),
            )
        checks.append(
            {
                "level": level,
                "strict_argmin_disagreements": disagreements,
                "maximum_relative_squared_distance_excess": maximum_relative_excess,
            }
        )
        residual = residual - centroids[assigned_codes]
    return checks


def _verify_optuna_history(
    database_path: Path,
    artifacts: list[VerifiedArtifact],
    manifest_sha256: str,
    preflight: PreflightEvidence,
) -> list[dict[str, object]]:
    if not database_path.is_file():
        raise ValueError(f"G6 RQ0 Optuna database is absent: {database_path}")
    storage = f"sqlite:///{database_path.resolve()}"
    tuning_stages = {
        "primary_control_tuning",
        "original_control_tuning",
        "treatment_tuning",
        "bridge_tuning",
    }
    grouped: dict[tuple[str, str], list[VerifiedArtifact]] = {}
    for artifact in artifacts:
        approved = artifact.compiled.approved
        if approved.stage in tuning_stages:
            grouped.setdefault((approved.stage, approved.method), []).append(artifact)
    contracts = []
    for (stage, method), candidates in sorted(grouped.items()):
        candidates.sort(key=lambda artifact: artifact.compiled.approved.trial or 0)
        study = optuna.load_study(
            storage=storage,
            study_name=f"g6-rq0-{stage}-{method}",
        )
        fixed = _study_fixed_parameters(stage, candidates[0].compiled.parameters)
        expected_attrs = {
            "manifest_sha256": manifest_sha256,
            "driver_seed": 42,
            "feasible_training_batches": list(preflight.feasible_training_batches),
            "validation_batch_size": preflight.validation_batch_size,
            "fixed_parameters": fixed,
        }
        if study.user_attrs != expected_attrs:
            raise ValueError(
                f"G6 RQ0 Optuna study contract changed: {study.study_name}"
            )
        if study.direction != optuna.study.StudyDirection.MAXIMIZE or not isinstance(
            study.sampler, optuna.samplers.TPESampler
        ):
            raise ValueError(
                f"G6 RQ0 Optuna study semantics changed: {study.study_name}"
            )
        trials = study.get_trials(deepcopy=False)
        if len(trials) != len(candidates):
            raise ValueError(f"G6 RQ0 Optuna trial count changed: {study.study_name}")
        expected_trial_parameters = {
            "primary_control_tuning": {
                "batch_size",
                "embedding_learning_rate",
                "deep_learning_rate",
            },
            "original_control_tuning": {
                "embedding_learning_rate",
                "deep_learning_rate",
            },
            "treatment_tuning": {
                "embedding_learning_rate",
                "deep_learning_rate",
                "num_levels",
                "num_codes",
                "representation_width",
            },
            "bridge_tuning": {
                "embedding_learning_rate",
                "deep_learning_rate",
            },
        }[stage]
        expected_distributions = _study_distributions(stage, preflight)
        for trial, artifact in zip(trials, candidates, strict=True):
            compiled = artifact.compiled
            if (
                trial.number != compiled.approved.trial
                or trial.state != TrialState.COMPLETE
                or trial.value != artifact.metrics["recall@100"]
                or trial.user_attrs.get("job_id") != compiled.approved.id
                or trial.user_attrs.get("compiled_parameters") != compiled.parameters
                or Path(str(trial.user_attrs.get("artifact"))).resolve()
                != artifact.path.resolve()
                or set(trial.params) != expected_trial_parameters
                or trial.distributions != expected_distributions
                or any(
                    compiled.parameters.get(name) != value
                    for name, value in trial.params.items()
                )
            ):
                raise ValueError(f"G6 RQ0 Optuna trial changed: {compiled.approved.id}")
        contracts.append(
            {
                "study_name": study.study_name,
                "direction": "maximize",
                "sampler": "TPESampler",
                "derived_seed": int.from_bytes(
                    hashlib.sha256(f"42:{study.study_name}".encode()).digest()[:4],
                    "big",
                ),
                "trial_count": len(trials),
                "distributions": {
                    name: repr(distribution)
                    for name, distribution in sorted(expected_distributions.items())
                },
            }
        )
    return contracts


def _study_distributions(
    stage: str, preflight: PreflightEvidence
) -> dict[str, optuna.distributions.BaseDistribution]:
    distributions: dict[str, optuna.distributions.BaseDistribution] = {
        "embedding_learning_rate": optuna.distributions.FloatDistribution(
            *EMBEDDING_LR_BOUNDS, log=True
        ),
        "deep_learning_rate": optuna.distributions.FloatDistribution(
            *DEEP_LR_BOUNDS, log=True
        ),
    }
    if stage == "primary_control_tuning":
        distributions["batch_size"] = optuna.distributions.CategoricalDistribution(
            preflight.feasible_training_batches
        )
    elif stage == "treatment_tuning":
        distributions |= {
            "num_levels": optuna.distributions.CategoricalDistribution(NUM_LEVELS),
            "num_codes": optuna.distributions.CategoricalDistribution(NUM_CODES),
            "representation_width": optuna.distributions.CategoricalDistribution(
                REPRESENTATION_WIDTHS
            ),
        }
    return distributions


def _replay_optuna_history(
    preflight: PreflightEvidence,
    artifacts_by_id: dict[str, VerifiedArtifact],
    selected_job_ids: dict[str, object],
) -> None:
    scratchpad = PROJECT_ROOT / "experiments/g6_rqkmeans_history/scratchpad"
    prior_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with tempfile.TemporaryDirectory(dir=scratchpad) as temporary:
        driver = G6Rq0OptunaDriver(
            Path(temporary) / "replay.sqlite3",
            feasible_training_batches=preflight.feasible_training_batches,
            validation_batch_size=preflight.validation_batch_size,
        )
        artifacts = list(artifacts_by_id.values())
        _replay_study(
            driver,
            driver.next_primary_control,
            _stage_artifacts(artifacts, "primary_control_tuning"),
        )
        primary = artifacts_by_id[str(selected_job_ids["primary_control"])]
        _replay_study(
            driver,
            lambda: driver.next_original_control(primary.selection()),
            _stage_artifacts(artifacts, "original_control_tuning"),
        )
        for representation in REPRESENTATIONS:
            _replay_study(
                driver,
                lambda representation=representation: driver.next_treatment(
                    representation, primary.selection()
                ),
                [
                    artifact
                    for artifact in _stage_artifacts(artifacts, "treatment_tuning")
                    if artifact.compiled.approved.method == representation
                ],
            )
        original = artifacts_by_id[str(selected_job_ids["original_control"])]
        semantic = artifacts_by_id[str(selected_job_ids["semantic_winner"])]
        _replay_study(
            driver,
            lambda: driver.next_bridge(
                primary.selection(), original.selection(), semantic.selection()
            ),
            _stage_artifacts(artifacts, "bridge_tuning"),
        )
    optuna.logging.set_verbosity(prior_verbosity)


def _replay_study(
    driver: G6Rq0OptunaDriver,
    next_job: Any,
    artifacts: list[VerifiedArtifact],
) -> None:
    artifacts.sort(key=lambda artifact: artifact.compiled.approved.trial or 0)
    for artifact in artifacts:
        replayed = next_job()
        if (
            replayed is None
            or replayed.approved != artifact.compiled.approved
            or replayed.parameters != artifact.compiled.parameters
        ):
            raise ValueError(
                f"G6 RQ0 seeded TPE replay diverged at {artifact.compiled.approved.id}"
            )
        driver.tell(
            replayed,
            artifact.metrics["recall@100"],
            artifact.path,
        )
    if next_job() is not None:
        raise ValueError("G6 RQ0 seeded TPE replay has extra trials")


def _study_fixed_parameters(stage: str, parameters: dict[str, Any]) -> dict[str, Any]:
    if stage == "primary_control_tuning":
        return {}
    names = [
        "batch_size",
        "validation_batch_size",
        "selected_primary_control_job_id",
    ]
    if stage == "treatment_tuning":
        names.append("representation")
    elif stage == "bridge_tuning":
        names.extend(
            (
                "representation",
                "num_levels",
                "num_codes",
                "representation_width",
                "selected_original_control_job_id",
                "selected_treatment_job_id",
            )
        )
    return {name: parameters[name] for name in names}


def _verify_selected_row(
    row: dict[str, Any], artifacts: dict[str, VerifiedArtifact]
) -> str:
    job_id = row.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("G6 RQ0 selection has an invalid job id")
    artifact = artifacts.get(job_id)
    if artifact is None:
        raise ValueError(f"G6 RQ0 selected job is absent from candidates: {job_id}")
    compiled = artifact.compiled
    expected = {
        "run_name": compiled.run_name,
        "attempt": compiled.attempt,
        "cap_epochs": compiled.cap_epochs,
        "parameters": compiled.parameters,
        "metrics": artifact.metrics,
        "inference_cost": asdict(inference_cost_contract(artifact)),
    }
    mismatches = [name for name, value in expected.items() if row.get(name) != value]
    if mismatches:
        raise ValueError(
            f"G6 RQ0 selected row {job_id} disagrees with artifacts: {mismatches}"
        )
    return job_id


def _verify_selection(
    selection: dict[str, Any],
    artifacts: list[VerifiedArtifact],
    artifacts_by_id: dict[str, VerifiedArtifact],
    bands: dict[str, float],
) -> dict[str, object]:
    recall_band = bands["recall@100"]
    ndcg_band = bands["ndcg@100"]
    primary_initial = _stage_artifacts(artifacts, "primary_control_tuning")
    primary_source = select_best(primary_initial, recall_band=0, ndcg_band=0)
    primary = _winner_with_required_boundaries(
        primary_source,
        primary_initial,
        artifacts_by_id,
        recall_band=0,
        ndcg_band=0,
    )
    _require_selected_artifact(selection, "primary_control", primary, artifacts_by_id)
    _verify_repeats(primary, artifacts)
    primary_dependencies = {
        "batch_size": primary.compiled.parameters["batch_size"],
        "validation_batch_size": primary.compiled.parameters["validation_batch_size"],
        "selected_primary_control_job_id": primary.compiled.approved.id,
    }

    original_initial = _stage_artifacts(artifacts, "original_control_tuning")
    _verify_dependencies(original_initial, primary_dependencies)
    original_source = select_best(
        original_initial, recall_band=recall_band, ndcg_band=ndcg_band
    )
    original = _winner_with_required_boundaries(
        original_source,
        original_initial,
        artifacts_by_id,
        recall_band=recall_band,
        ndcg_band=ndcg_band,
    )
    _require_selected_artifact(selection, "original_control", original, artifacts_by_id)

    published_treatments = _required_dict(selection, "treatment_winners")
    if set(published_treatments) != set(REPRESENTATIONS):
        raise ValueError("G6 RQ0 selection has the wrong treatment-winner keys")
    treatment_winners: dict[str, VerifiedArtifact] = {}
    boundary_sources = [primary_source, original_source]
    for representation in REPRESENTATIONS:
        initial = [
            artifact
            for artifact in _stage_artifacts(artifacts, "treatment_tuning")
            if artifact.compiled.approved.method == representation
        ]
        _verify_dependencies(initial, primary_dependencies)
        source = select_best(initial, recall_band=recall_band, ndcg_band=ndcg_band)
        boundary_sources.append(source)
        winner = _winner_with_required_boundaries(
            source,
            initial,
            artifacts_by_id,
            recall_band=recall_band,
            ndcg_band=ndcg_band,
        )
        row = published_treatments[representation]
        if not isinstance(row, dict):
            raise ValueError(f"G6 RQ0 treatment winner {representation} is invalid")
        _require_row_artifact(row, winner, artifacts_by_id)
        treatment_winners[representation] = winner

    semantic = select_best(
        list(treatment_winners.values()),
        recall_band=recall_band,
        ndcg_band=ndcg_band,
    )
    _require_selected_artifact(selection, "semantic_winner", semantic, artifacts_by_id)
    promoted = (
        semantic.metrics["recall@100"] > primary.metrics["recall@100"] + recall_band
        and semantic.metrics["ndcg@100"] >= primary.metrics["ndcg@100"] - ndcg_band
    )
    if selection.get("semantic_promoted") is not promoted:
        raise ValueError("G6 RQ0 semantic-promotion decision is inconsistent")
    selected_primary = semantic if promoted else primary
    _require_selected_artifact(
        selection, "selected_primary_method", selected_primary, artifacts_by_id
    )

    bridge_initial = _stage_artifacts(artifacts, "bridge_tuning")
    bridge_dependencies = primary_dependencies | {
        "selected_original_control_job_id": original.compiled.approved.id,
        "selected_treatment_job_id": semantic.compiled.approved.id,
        **{
            name: semantic.compiled.parameters[name]
            for name in (
                "representation",
                "num_levels",
                "num_codes",
                "representation_width",
            )
        },
    }
    _verify_dependencies(bridge_initial, bridge_dependencies)
    bridge_source = select_best(
        bridge_initial, recall_band=recall_band, ndcg_band=ndcg_band
    )
    boundary_sources.append(bridge_source)
    bridge = _winner_with_required_boundaries(
        bridge_source,
        bridge_initial,
        artifacts_by_id,
        recall_band=recall_band,
        ndcg_band=ndcg_band,
    )
    _require_selected_artifact(selection, "bridge", bridge, artifacts_by_id)
    _verify_conditional_boundaries(boundary_sources, artifacts)
    return {
        "primary_control": primary.compiled.approved.id,
        "original_control": original.compiled.approved.id,
        "semantic_winner": semantic.compiled.approved.id,
        "selected_primary_method": selected_primary.compiled.approved.id,
        "bridge": bridge.compiled.approved.id,
        "treatment_winners": {
            name: treatment_winners[name].compiled.approved.id
            for name in sorted(treatment_winners)
        },
    }


def _winner_with_required_boundaries(
    source: VerifiedArtifact,
    initial: list[VerifiedArtifact],
    artifacts_by_id: dict[str, VerifiedArtifact],
    *,
    recall_band: float,
    ndcg_band: float,
) -> VerifiedArtifact:
    boundaries = []
    for expected in _expected_boundaries(source.compiled):
        artifact = artifacts_by_id.get(expected.approved.id)
        if artifact is None:
            raise ValueError(
                f"G6 RQ0 required boundary is absent: {expected.approved.id}"
            )
        if artifact.compiled.parameters != expected.parameters:
            raise ValueError(
                f"G6 RQ0 boundary parameters changed: {expected.approved.id}"
            )
        boundaries.append(artifact)
    winner = select_best(
        [*initial, *boundaries],
        recall_band=recall_band,
        ndcg_band=ndcg_band,
    )
    require_resolved_boundary(winner)
    return winner


def _expected_boundaries(source: CompiledJob) -> list[CompiledJob]:
    stage = source.approved.stage
    if stage == "primary_control_tuning":
        builder, surface = "primary_control", "primary_control"
    elif stage == "original_control_tuning":
        builder, surface = "original_control", "original_control"
    elif stage == "treatment_tuning":
        builder = "treatment"
        surface = str(source.parameters["representation"])
    elif stage == "bridge_tuning":
        builder, surface = "bridge", "bridge"
    else:
        raise ValueError(f"G6 RQ0 cannot extend boundary for stage {stage}")
    boundaries = []
    for learning_rate, bounds in (
        ("embedding_learning_rate", EMBEDDING_LR_BOUNDS),
        ("deep_learning_rate", DEEP_LR_BOUNDS),
    ):
        side = boundary_side(float(source.parameters[learning_rate]), bounds)
        if side is None:
            continue
        jobs = sorted(
            (
                job
                for job in approved_manifest().jobs_for_stage("lr_boundary")
                if job.method == surface
                and job.forced_parameters["learning_rate"] == learning_rate
            ),
            key=lambda job: int(job.forced_parameters["boundary_slot"]),
        )
        rates = outside_boundary_rates(bounds, side)
        if len(jobs) != len(rates):
            raise ValueError(
                f"G6 RQ0 manifest boundary family is incomplete: {surface}"
            )
        for job, rate in zip(jobs, rates, strict=True):
            parameters = dict(source.parameters)
            parameters |= {
                "builder": builder,
                "source_job_id": source.approved.id,
                "source_parameters": dict(source.parameters),
                "boundary_side": side,
                learning_rate: rate,
            }
            boundaries.append(CompiledJob(job, parameters))
    return boundaries


def _verify_repeats(
    primary: VerifiedArtifact, artifacts: list[VerifiedArtifact]
) -> None:
    expected = {name: primary.compiled.parameters[name] for name in BASE_PARAMETERS} | {
        "selected_primary_control_job_id": primary.compiled.approved.id
    }
    for repeat in _stage_artifacts(artifacts, "primary_control_repeats"):
        if repeat.compiled.parameters != expected:
            raise ValueError(
                f"G6 RQ0 repeat {repeat.compiled.approved.id} changed control parameters"
            )


def _verify_dependencies(
    artifacts: list[VerifiedArtifact], expected: dict[str, object]
) -> None:
    for artifact in artifacts:
        mismatches = [
            name
            for name, value in expected.items()
            if artifact.compiled.parameters.get(name) != value
        ]
        if mismatches:
            raise ValueError(
                f"G6 RQ0 job {artifact.compiled.approved.id} changed fixed "
                f"dependencies: {mismatches}"
            )


def _verify_conditional_boundaries(
    sources: list[VerifiedArtifact], artifacts: list[VerifiedArtifact]
) -> None:
    expected = {
        boundary.approved.id: boundary.parameters
        for source in sources
        for boundary in _expected_boundaries(source.compiled)
    }
    actual = {
        artifact.compiled.approved.id: artifact.compiled.parameters
        for artifact in artifacts
        if artifact.compiled.approved.stage == "lr_boundary"
    }
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        changed = sorted(
            job_id
            for job_id in expected.keys() & actual.keys()
            if expected[job_id] != actual[job_id]
        )
        raise ValueError(
            "G6 RQ0 conditional boundary plan changed: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def _require_selected_artifact(
    selection: dict[str, Any],
    name: str,
    expected: VerifiedArtifact,
    artifacts_by_id: dict[str, VerifiedArtifact],
) -> None:
    _require_row_artifact(_required_dict(selection, name), expected, artifacts_by_id)


def _require_row_artifact(
    row: dict[str, Any],
    expected: VerifiedArtifact,
    artifacts_by_id: dict[str, VerifiedArtifact],
) -> None:
    selected_id = _verify_selected_row(row, artifacts_by_id)
    if selected_id != expected.compiled.approved.id:
        raise ValueError(
            f"G6 RQ0 published winner {selected_id} differs from recomputed winner "
            f"{expected.compiled.approved.id}"
        )


def _stage_artifacts(
    artifacts: list[VerifiedArtifact], stage: str
) -> list[VerifiedArtifact]:
    return [
        artifact for artifact in artifacts if artifact.compiled.approved.stage == stage
    ]


def _required_dict(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"G6 RQ0 evidence omits {name}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_reference(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"G6 RQ0 evidence file is absent: {path}")
    return {"path": _project_relative(path), "sha256": _sha256(path)}


def _project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _manifest_order() -> dict[str, int]:
    return {job.id: index for index, job in enumerate(approved_manifest().jobs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, default=Path("generated/logs"))
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    document = build_audit_evidence(
        database_path=arguments.database,
        compiled_path=arguments.compiled,
        logs_root=arguments.logs_root,
        preflight_path=arguments.preflight,
        bands_path=arguments.bands,
        selection_path=arguments.selection,
    )
    write_audit_evidence(arguments.output, document)


if __name__ == "__main__":
    main()
