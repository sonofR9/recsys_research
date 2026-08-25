from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

from experiments.g2_esasrec.configs.local import (
    COMPONENT_METHODS,
    DEEP_LR_BOUNDS,
    EMBEDDING_LR_BOUNDS,
)
from experiments.g2_esasrec.protocol.manifest import (
    ApprovedJob,
    CompiledJob,
    approved_manifest,
    validate_compiled_job,
)
from experiments.g2_esasrec.analysis.fit_evidence import FitEvidence
from experiments.g2_esasrec.launchers.compiled import persisted_job_contract
from experiments.g2_esasrec.official.provenance import (
    OFFICIAL_HYPERPARAMETERS,
    OFFICIAL_PACKAGE_VERSIONS,
    OFFICIAL_PYTHON_VERSION,
    OFFICIAL_PROTOCOL,
    RECTOOLS_SOURCE_SHA256,
)

RANKING_METRICS = ("recall", "ndcg", "mrr", "capped_recall", "coverage")
METRICS = tuple(
    f"{metric}@{cutoff}" for metric in RANKING_METRICS for cutoff in (10, 50, 100)
)
OFFICIAL_LOCAL_SOURCE_SHA256 = {
    "catalog_data": (
        "6621a27cb6f0d6c9f53e0a7d2cd4da0342b083b26f7a36d1fc48df59f7ff3936"
    ),
    "runner": "de6fdbdd5c6315ad2099617e68bd064d9f33fba7b6bc52a23374b69a7125258f",
    "protocol": "dd3d5fa349a7384576fe0ba8c70fce5fcaa76bb2411d7a8b44a5827cf72c359b",
    "provenance": ("007f67b74bf3c1a9c0b57fac9e3dc108974c475b7f49bfc00d752ca82f175e05"),
}


@dataclass(frozen=True)
class VerifiedArtifact:
    job: ApprovedJob
    path: Path
    parameters: dict[str, Any]
    metrics: dict[str, float]
    metadata: dict[str, Any]
    costs: dict[str, float]

    @property
    def wall_seconds(self) -> float:
        return self.costs["wall_seconds"]


@dataclass(frozen=True)
class MetricBand:
    sample_standard_deviation: float
    reader_threshold: float

    def to_dict(self) -> dict[str, float]:
        return {
            "sample_standard_deviation": self.sample_standard_deviation,
            "reader_threshold": self.reader_threshold,
        }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"missing or invalid artifact file {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"artifact file must contain an object: {path}")
    return value


def _finite_metric(value: Any, name: str, *, maximum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"metric {name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"metric {name} is invalid")
    if maximum is not None and result > maximum:
        raise ValueError(f"metric {name} must be in [0, {maximum:g}]")
    return result


def _require_local_contract(
    compiled: CompiledJob,
    contract: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    expected = persisted_job_contract(compiled)
    if contract != expected:
        raise ValueError(
            f"{compiled.approved.run_name}: local implementation contract changed"
        )
    evidence = expected["local_implementation"]
    transfer = metadata.get("transfer_invariants")
    if not isinstance(transfer, dict):
        raise ValueError(
            f"{compiled.approved.run_name}: local transfer invariants are missing"
        )
    for name, value in evidence["transfer_invariants"].items():
        if transfer.get(name) != value:
            raise ValueError(
                f"{compiled.approved.run_name}: local transfer invariant {name} changed"
            )
    for name in (
        "max_epochs",
        "initializer_std",
        "weight_decay",
        "runtime_dtype",
        "runtime_compile",
    ):
        if metadata.get(name) != evidence["training"][name]:
            raise ValueError(
                f"{compiled.approved.run_name}: local training contract {name} changed"
            )
    if evidence["training"]["layer_family"] != "g1_control":
        expected_recipe = {
            name: evidence["training"][name]
            for name in ("layer_family", "loss_kind", "gbce_t")
        }
        if metadata.get("g2_recipe") != expected_recipe:
            raise ValueError(
                f"{compiled.approved.run_name}: executed G2 recipe changed"
            )
    parameters = compiled.parameters
    for name in (
        "batch_size",
        "embedding_learning_rate",
        "deep_learning_rate",
    ):
        if metadata.get(name) != parameters[name]:
            raise ValueError(
                f"{compiled.approved.run_name}: contract parameter {name} changed"
            )


def _require_official_contract(job: ApprovedJob, metadata: dict[str, Any]) -> None:
    prefix = f"{job.run_name}:"
    if metadata.get("implementation") != "RecTools SASRecModel with LiGRLayers":
        raise ValueError(f"{prefix} official implementation changed")
    if metadata.get("hyperparameters") != OFFICIAL_HYPERPARAMETERS:
        raise ValueError(f"{prefix} official hyperparameters changed")
    if metadata.get("max_epochs") != 100 or metadata.get("patience") != 10:
        raise ValueError(f"{prefix} official training hyperparameters changed")
    if metadata.get("protocol") != OFFICIAL_PROTOCOL:
        raise ValueError(f"{prefix} official protocol changed")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{prefix} official provenance is missing")
    environment = provenance.get("environment")
    packages = environment.get("packages") if isinstance(environment, dict) else None
    if (
        not isinstance(environment, dict)
        or environment.get("python") != OFFICIAL_PYTHON_VERSION
    ):
        raise ValueError(f"{prefix} official Python version changed")
    if packages != OFFICIAL_PACKAGE_VERSIONS:
        raise ValueError(f"{prefix} official package versions changed")
    sources = provenance.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"{prefix} official RecTools source hashes are missing")
    expected_source_names = (
        RECTOOLS_SOURCE_SHA256.keys() | OFFICIAL_LOCAL_SOURCE_SHA256.keys()
    )
    if sources.keys() != expected_source_names:
        raise ValueError(f"{prefix} official source set changed")
    observed = {
        name: source.get("sha256") if isinstance(source, dict) else None
        for name, source in sources.items()
        if name in RECTOOLS_SOURCE_SHA256
    }
    if observed != RECTOOLS_SOURCE_SHA256:
        raise ValueError(f"{prefix} official RecTools source hashes changed")
    local_observed = {
        name: source.get("sha256") if isinstance(source, dict) else None
        for name, source in sources.items()
        if name in OFFICIAL_LOCAL_SOURCE_SHA256
    }
    if local_observed != OFFICIAL_LOCAL_SOURCE_SHA256:
        raise ValueError(f"{prefix} local official source hashes changed")


def _unresolved_selection_error(
    job: ApprovedJob, directory: Path, metadata: dict[str, Any]
) -> ValueError:
    max_epochs = metadata.get("max_epochs")
    epochs_trained = metadata.get("epochs_trained")
    reached_cap = (
        isinstance(max_epochs, int)
        and not isinstance(max_epochs, bool)
        and max_epochs > 0
        and (
            metadata.get("best_epoch_at_cap") is True
            or (
                isinstance(epochs_trained, int)
                and not isinstance(epochs_trained, bool)
                and epochs_trained >= max_epochs
            )
        )
    )
    if reached_cap:
        extended_cap = math.ceil(max_epochs * 1.5)
        requested_name = f"{job.run_name}_cap{extended_cap}"
        return ValueError(
            f"{job.run_name}: unresolved at max_epochs={max_epochs}; preserve raw "
            f"artifact {directory}; request approval for a new run named "
            f"{requested_name} with max_epochs={extended_cap}. No approved manifest "
            "slot exists; no extension job was created."
        )
    return ValueError(f"{job.run_name}: artifact is not selection-resolved")


def _approved_job_by_id(job_id: str) -> ApprovedJob:
    matches = [job for job in approved_manifest().jobs if job.id == job_id]
    if len(matches) != 1:
        raise ValueError(f"artifact prerequisite {job_id!r} is not approved")
    return matches[0]


def _require_verified_prerequisites(compiled: CompiledJob, logs_root: Path) -> None:
    parameters = compiled.parameters
    selected_control_id = parameters.get("selected_control_job_id")
    if isinstance(selected_control_id, str):
        selected_control = load_verified_artifact(
            _approved_job_by_id(selected_control_id), logs_root
        )
        selected_names = {"batch_size"}
        if compiled.approved.stage == "control_repeats":
            selected_names |= {
                "embedding_learning_rate",
                "deep_learning_rate",
            }
        _require_propagated(compiled, selected_control, selected_names)
    source_id = parameters.get("source_job_id")
    if not isinstance(source_id, str):
        return
    source = load_verified_artifact(_approved_job_by_id(source_id), logs_root)
    _require_source_parameter_lineage(compiled, source)


def _require_propagated(
    compiled: CompiledJob,
    source: VerifiedArtifact,
    names: Iterable[str],
) -> None:
    for name in names:
        if compiled.parameters.get(name) != source.parameters.get(name):
            raise ValueError(
                f"{compiled.approved.run_name}: prerequisite {name} was not propagated"
            )


def _boundary_rate_name(source: VerifiedArtifact) -> str:
    edges = []
    for name, (lower, upper) in {
        "embedding_learning_rate": EMBEDDING_LR_BOUNDS,
        "deep_learning_rate": DEEP_LR_BOUNDS,
    }.items():
        value = float(source.parameters[name])
        if value <= lower * 1.05:
            edges.append(name)
        if value >= upper * 0.95:
            edges.append(name)
    if len(edges) != 1:
        raise ValueError("verified LR-boundary source has no unique boundary")
    return edges[0]


def _require_boundary_lineage(compiled: CompiledJob, source: VerifiedArtifact) -> None:
    changed_name = _boundary_rate_name(source)
    _require_propagated(
        compiled,
        source,
        {
            "batch_size",
            "selected_control_job_id",
            "ligr_multiplier",
            "gbce_t",
            "uniform_fraction",
            "logq_correction",
            *({"deep_learning_rate", "embedding_learning_rate"} - {changed_name}),
        }
        & source.parameters.keys(),
    )
    lower, upper = {
        "embedding_learning_rate": EMBEDDING_LR_BOUNDS,
        "deep_learning_rate": DEEP_LR_BOUNDS,
    }[changed_name]
    source_value = float(source.parameters[changed_name])
    if source_value <= lower * 1.05:
        expected = (lower / 3, lower / math.sqrt(3))
    else:
        expected = (upper * 3, upper * math.sqrt(3))
    trial = compiled.approved.trial
    if trial not in {0, 1} or compiled.parameters[changed_name] != expected[trial]:
        raise ValueError(
            f"{compiled.approved.run_name}: prerequisite {changed_name} boundary changed"
        )


def _require_source_parameter_lineage(
    compiled: CompiledJob, source: VerifiedArtifact
) -> None:
    job = compiled.approved
    if job.stage == "component_tuning":
        names = {"batch_size", "selected_control_job_id", "ligr_multiplier"}
        _require_propagated(compiled, source, names)
        return
    if job.stage == "mixed_tuning":
        names = {
            "batch_size",
            "embedding_learning_rate",
            "deep_learning_rate",
            "selected_control_job_id",
            "ligr_multiplier",
        }
        _require_propagated(compiled, source, names)
        return
    if job.stage == "lr_boundary":
        _require_boundary_lineage(compiled, source)
        return
    inherited = {
        "batch_size",
        "embedding_learning_rate",
        "deep_learning_rate",
        "selected_control_job_id",
        "ligr_multiplier",
        "gbce_t",
        "uniform_fraction",
        "logq_correction",
        "builder",
        "method",
    }
    _require_propagated(compiled, source, inherited & source.parameters.keys())


def load_verified_artifact(job: ApprovedJob, logs_root: Path) -> VerifiedArtifact:
    directory = logs_root / job.run_name
    contract = _load_object(directory / "g2_job.json")
    metadata = _load_object(directory / "training_metadata.json")
    raw_metrics = _load_object(directory / "final_metrics.json")
    manifest = approved_manifest()
    if contract.get("manifest_sha256") != manifest.sha256:
        raise ValueError(f"{job.run_name}: artifact manifest hash changed")
    if contract.get("job") != job.to_dict():
        raise ValueError(f"{job.run_name}: artifact job identity changed")
    parameters = contract.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{job.run_name}: artifact parameters are missing")
    compiled = CompiledJob(job, parameters)
    validate_compiled_job(compiled)
    expected_dataset = "native-50m" if job.stage == "official" else "50m"
    if metadata.get("dataset_size") != expected_dataset:
        raise ValueError(f"{job.run_name}: artifact is not native 50M")
    if metadata.get("seed") != job.seed:
        raise ValueError(f"{job.run_name}: seed changed")
    if job.stage == "official":
        if contract != compiled.to_contract(manifest):
            raise ValueError(f"{job.run_name}: official job contract changed")
        _require_official_contract(job, metadata)
    else:
        _require_local_contract(compiled, contract, metadata)
    if job.stage == "official":
        resolved = (
            metadata.get("early_stopped") is True
            and metadata.get("best_epoch_at_cap") is False
            and isinstance(metadata.get("best_epoch"), int)
        )
    else:
        resolved = metadata.get("selection_resolved") is True
    if not resolved:
        raise _unresolved_selection_error(job, directory, metadata)
    num_users = _finite_metric(raw_metrics.get("num_users"), "num_users")
    if num_users != 3414:
        raise ValueError(
            f"{job.run_name}: metrics must use the full 3,414-user denominator"
        )
    metrics = {
        name: _finite_metric(raw_metrics.get(name), name, maximum=1.0)
        for name in METRICS
    }
    if job.stage == "official":
        costs = {
            "wall_seconds": _finite_metric(metadata.get("wall_seconds"), "wall_seconds")
        }
    else:
        raw_costs = _load_object(directory / "cost_metrics.json")
        costs = {
            name: _finite_metric(raw_costs.get(name), name)
            for name in (
                "params_total",
                "params_deep",
                "training_seconds",
                "wall_seconds",
                "median_train_epoch_seconds",
                "peak_memory_gb",
                "targets_per_second",
                "best_epoch",
            )
        }
        if costs["best_epoch"] != float(metadata["best_epoch"]):
            raise ValueError(f"{job.run_name}: cost evidence best epoch changed")
    _require_verified_prerequisites(compiled, logs_root)
    return VerifiedArtifact(job, directory, parameters, metrics, metadata, costs)


def load_exact_artifacts(
    jobs: Iterable[ApprovedJob], logs_root: Path
) -> list[VerifiedArtifact]:
    expected = list(jobs)
    if len({job.id for job in expected}) != len(expected):
        raise ValueError("artifact request contains duplicate jobs")
    return [load_verified_artifact(job, logs_root) for job in expected]


def _requires_selected_ligr(artifact: VerifiedArtifact) -> bool:
    method = artifact.job.method
    if artifact.job.stage == "component_tuning":
        return method.startswith("matched_standard_") or method == "ligr_gbce"
    if artifact.job.stage == "mixed_tuning":
        return True
    if artifact.job.stage == "lr_boundary":
        return method != "ligr_sampled_softmax" and "ligr_multiplier" in (
            artifact.parameters
        )
    return artifact.job.stage == "reversal_confirmation" and (
        "ligr_multiplier" in artifact.parameters
    )


def _selected_ligr_ancestor(
    artifact: VerifiedArtifact,
    artifacts_by_id: Mapping[str, VerifiedArtifact],
) -> str:
    visited = set()
    current = artifact
    while True:
        if current.job.id in visited:
            raise ValueError("selected LiGR prerequisite lineage contains a cycle")
        visited.add(current.job.id)
        if current.job.method == "ligr_sampled_softmax" and current.job.stage in {
            "component_tuning",
            "lr_boundary",
        }:
            return current.job.id
        source_id = current.parameters.get("source_job_id")
        source = artifacts_by_id.get(source_id)
        if source is None:
            raise ValueError(
                f"{artifact.job.run_name}: selected LiGR prerequisite is missing"
            )
        current = source


def require_selected_ligr_lineage(
    artifacts: Iterable[VerifiedArtifact], selected_ligr: VerifiedArtifact
) -> None:
    rows = list(artifacts)
    if not (
        selected_ligr.job.method == "ligr_sampled_softmax"
        and selected_ligr.job.stage in {"component_tuning", "lr_boundary"}
    ):
        raise ValueError("selected LiGR artifact has the wrong family")
    artifacts_by_id = {artifact.job.id: artifact for artifact in rows}
    if len(artifacts_by_id) != len(rows):
        raise ValueError("selected LiGR lineage contains duplicate artifacts")
    if artifacts_by_id.get(selected_ligr.job.id) != selected_ligr:
        raise ValueError("selected LiGR artifact is absent from the verified program")
    for artifact in rows:
        if (
            _requires_selected_ligr(artifact)
            and _selected_ligr_ancestor(artifact, artifacts_by_id)
            != selected_ligr.job.id
        ):
            raise ValueError(
                f"{artifact.job.run_name}: prerequisite is not the selected LiGR winner"
            )


def _selected_control_ancestor(
    artifact: VerifiedArtifact,
    artifacts_by_id: Mapping[str, VerifiedArtifact],
) -> str:
    visited = set()
    current = artifact
    while True:
        if current.job.id in visited:
            raise ValueError("selected control prerequisite lineage contains a cycle")
        visited.add(current.job.id)
        if current.job.stage == "control_tuning" or (
            current.job.stage == "lr_boundary" and current.job.method == "control"
        ):
            return current.job.id
        source_id = current.parameters.get("source_job_id")
        source = artifacts_by_id.get(source_id)
        if source is None:
            raise ValueError(
                f"{artifact.job.run_name}: selected control prerequisite is missing"
            )
        current = source


def require_selected_control_lineage(
    artifacts: Iterable[VerifiedArtifact], selected_control: VerifiedArtifact
) -> None:
    rows = list(artifacts)
    if not (
        selected_control.job.stage == "control_tuning"
        or (
            selected_control.job.stage == "lr_boundary"
            and selected_control.job.method == "control"
        )
    ):
        raise ValueError("selected control artifact has the wrong family")
    artifacts_by_id = {artifact.job.id: artifact for artifact in rows}
    if len(artifacts_by_id) != len(rows):
        raise ValueError("selected control lineage contains duplicate artifacts")
    if artifacts_by_id.get(selected_control.job.id) != selected_control:
        raise ValueError(
            "selected control artifact is absent from the verified program"
        )
    for artifact in rows:
        selected_id = artifact.parameters.get("selected_control_job_id")
        if isinstance(selected_id, str):
            if selected_id != selected_control.job.id:
                raise ValueError(
                    f"{artifact.job.run_name}: prerequisite is not the selected "
                    "control winner"
                )
            continue
        if artifact.job.stage == "reversal_confirmation" and (
            artifact.parameters.get("builder") == "control"
            and _selected_control_ancestor(artifact, artifacts_by_id)
            != selected_control.job.id
        ):
            raise ValueError(
                f"{artifact.job.run_name}: prerequisite is not the selected control "
                "winner"
            )


def bundle_promotion_reason(
    baseline: VerifiedArtifact,
    candidate: VerifiedArtifact,
    metric_bands: Mapping[str, float],
) -> str | None:
    recall_gain = candidate.metrics["recall@100"] - baseline.metrics["recall@100"]
    ndcg_gain = candidate.metrics["ndcg@100"] - baseline.metrics["ndcg@100"]
    if recall_gain > float(metric_bands["recall@100"]):
        return "recall_gain_beyond_band"
    if (
        recall_gain >= -float(metric_bands["recall@100"])
        and ndcg_gain > float(metric_bands["ndcg@100"])
        and candidate.wall_seconds <= baseline.wall_seconds
    ):
        return "recall_tie_ndcg_gain_on_quality_cost_pareto_frontier"
    return None


def select_aggregate_bundle(
    baseline: VerifiedArtifact,
    component_winners: Iterable[VerifiedArtifact],
    mixed_winner: VerifiedArtifact | None,
    metric_bands: Mapping[str, float],
) -> VerifiedArtifact:
    candidates = list(component_winners)
    if len(candidates) != len(COMPONENT_METHODS) or {
        candidate.job.method for candidate in candidates
    } != set(COMPONENT_METHODS):
        raise ValueError("aggregate selection requires all six component winners")
    if mixed_winner is not None:
        if mixed_winner.job.method != "mixed_sampler":
            raise ValueError("aggregate mixed candidate has the wrong family")
        candidates.append(mixed_winner)
    qualified = [
        candidate
        for candidate in candidates
        if bundle_promotion_reason(baseline, candidate, metric_bands) is not None
    ]
    if not qualified:
        return baseline
    return select_best(qualified, metric_bands=metric_bands)


def _artifact_identity(artifact: VerifiedArtifact) -> dict[str, str]:
    return {
        "job_id": artifact.job.id,
        "run_name": artifact.job.run_name,
        "method": artifact.job.method,
        "stage": artifact.job.stage,
    }


def build_composition_evidence(
    baseline: VerifiedArtifact,
    selected: VerifiedArtifact,
    component_winners: Iterable[VerifiedArtifact],
    mixed_winner: VerifiedArtifact | None,
    metric_bands: Mapping[str, float],
) -> dict[str, Any]:
    components = list(component_winners)
    expected = select_aggregate_bundle(baseline, components, mixed_winner, metric_bands)
    if selected != expected:
        raise ValueError("composition evidence selected bundle changed")
    candidates = [*components, *([] if mixed_winner is None else [mixed_winner])]
    candidate_rows = []
    omissions = []
    for candidate in candidates:
        reason = bundle_promotion_reason(baseline, candidate, metric_bands)
        row: dict[str, Any] = _artifact_identity(candidate) | {
            "candidate_kind": (
                "mixed" if candidate.job.method == "mixed_sampler" else "component"
            ),
            "qualified": reason is not None,
            "qualification_reason": reason,
            "selected": candidate == selected,
        }
        candidate_rows.append(row)
        if candidate != selected:
            omissions.append(
                _artifact_identity(candidate)
                | {
                    "reason": (
                        "qualified_but_not_selected_by_band_aware_rule"
                        if reason is not None
                        else "not_qualified_for_promotion"
                    )
                }
            )
    if mixed_winner is None:
        omissions.append(
            {
                "job_id": None,
                "run_name": None,
                "method": "mixed_sampler",
                "stage": "mixed_tuning",
                "reason": "no_eligible_mixed_winner",
            }
        )
    metrics = {}
    for metric in ("recall@100", "ndcg@100", "coverage@100"):
        baseline_value = baseline.metrics[metric]
        aggregate_value = selected.metrics[metric]
        gain = aggregate_value - baseline_value
        metrics[metric] = {
            "baseline": baseline_value,
            "aggregate": aggregate_value,
            "aggregate_gain_points": gain,
            "aggregate_gain_percent": (
                None if baseline_value == 0 else gain / baseline_value * 100
            ),
            "standalone_sum_points": gain,
            "interaction_gap_points": 0.0,
            "interaction_band": float(metric_bands[metric]),
            "interaction_label": "unresolved",
        }
    return {
        "manifest_sha256": approved_manifest().sha256,
        "dataset_size": "native-50m",
        "selection_rule": "band_aware_promotion_then_recall_ndcg_cost",
        "baseline": _artifact_identity(baseline),
        "baseline_fallback": _artifact_identity(baseline)
        | {
            "selected": selected == baseline,
            "selection_condition": "no_atomic_bundle_qualified",
        },
        "included_bundle": _artifact_identity(selected),
        "mixed_candidate_eligible": mixed_winner is not None,
        "mixed_candidate_status": (
            {
                "status": "eligible",
                **_artifact_identity(mixed_winner),
                "qualified": (
                    bundle_promotion_reason(baseline, mixed_winner, metric_bands)
                    is not None
                ),
                "selected": mixed_winner == selected,
            }
            if mixed_winner is not None
            else {
                "status": "omitted",
                "method": "mixed_sampler",
                "reason": "no_eligible_mixed_winner",
            }
        ),
        "candidates": candidate_rows,
        "omissions": omissions,
        "metrics": metrics,
        "interaction_resolution": (
            "unresolved against native-50m size-matched empirical bands"
        ),
    }


def write_composition_evidence(document: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def build_reversal_confirmation_evidence(
    confirmations: Iterable[VerifiedArtifact],
    sources: Iterable[VerifiedArtifact],
) -> dict[str, Any]:
    rows = list(confirmations)
    expected_jobs = approved_manifest().jobs_for_stage("reversal_confirmation")
    if len(rows) != len(expected_jobs) or {row.job.id for row in rows} != {
        job.id for job in expected_jobs
    }:
        raise ValueError("reversal evidence requires all four approved confirmations")
    sources_by_id = {source.job.id: source for source in sources}
    if len(sources_by_id) != 2:
        raise ValueError("reversal evidence requires exactly two source artifacts")
    confirmation_rows = []
    slot_sources: dict[int, str] = {}
    for row in sorted(rows, key=lambda item: (int(item.job.trial or 0), item.job.seed)):
        slot = row.job.trial
        source_id = row.parameters.get("source_job_id")
        source = sources_by_id.get(source_id)
        if slot not in {0, 1} or source is None:
            raise ValueError("reversal confirmation source changed")
        prior = slot_sources.setdefault(slot, source.job.id)
        if prior != source.job.id:
            raise ValueError("reversal configuration does not share one exact source")
        confirmation_rows.append(
            {
                "job_id": row.job.id,
                "run_name": row.job.run_name,
                "configuration_slot": slot,
                "source_job_id": source.job.id,
                "source_run_name": source.job.run_name,
                "source_method": source.job.method,
                "seed": row.job.seed,
                "metrics": {metric: row.metrics[metric] for metric in METRICS},
            }
        )
    if {(row["configuration_slot"], row["seed"]) for row in confirmation_rows} != {
        (0, 43),
        (0, 44),
        (1, 43),
        (1, 44),
    }:
        raise ValueError("reversal confirmation seeds changed")
    if len(set(slot_sources.values())) != 2:
        raise ValueError("reversal configurations must use distinct sources")
    return {
        "manifest_sha256": approved_manifest().sha256,
        "dataset_size": "native-50m",
        "interpretation_state": "explicit_validation_required",
        "sources": [
            _artifact_identity(source) | {"parameters": source.parameters}
            for source in sorted(sources_by_id.values(), key=lambda item: item.job.id)
        ],
        "confirmations": confirmation_rows,
    }


def persist_reversal_confirmation_evidence(
    document: Mapping[str, Any], destination: Path
) -> None:
    serialized = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        try:
            existing = json.loads(destination.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("reversal confirmation evidence is unreadable") from error
        if existing != document:
            raise ValueError("reversal confirmation evidence changed")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(serialized)
    temporary.replace(destination)


def _round_up_one_significant_digit(value: float) -> float:
    if value == 0:
        return 0.0
    magnitude = 10.0 ** math.floor(math.log10(value))
    return math.ceil(value / magnitude - 1e-12) * magnitude


def control_band_artifacts(
    selected_control: VerifiedArtifact,
    repeat_artifacts: Iterable[VerifiedArtifact],
) -> tuple[VerifiedArtifact, ...]:
    rows = list(repeat_artifacts)
    seed_42_repeats = [row for row in rows if row.job.seed == 42]
    if seed_42_repeats and (
        seed_42_repeats[0].parameters.get("selected_control_job_id")
        != selected_control.job.id
        or _control_configuration(seed_42_repeats[0])
        != _control_configuration(selected_control)
    ):
        raise ValueError(
            "empirical bands require the exact selected control configuration"
        )
    if selected_control.job.seed == 42 and not seed_42_repeats:
        rows.append(selected_control)
    rows.sort(key=lambda artifact: artifact.job.seed)
    empirical_bands(rows)
    return tuple(rows)


def _control_configuration(artifact: VerifiedArtifact) -> dict[str, Any]:
    return {
        name: artifact.parameters.get(name)
        for name in (
            "batch_size",
            "embedding_learning_rate",
            "deep_learning_rate",
        )
    }


def _validate_control_band_artifacts(rows: list[VerifiedArtifact]) -> None:
    if [artifact.job.seed for artifact in rows] != list(range(42, 52)):
        raise ValueError("empirical bands require exact control seeds 42 through 51")
    first = rows[0]
    if first.job.stage == "control_repeats":
        selected_control_job_id = first.parameters.get("selected_control_job_id")
    elif first.job.stage == "control_tuning" or (
        first.job.stage == "lr_boundary" and first.job.method == "control"
    ):
        selected_control_job_id = first.job.id
    else:
        raise ValueError("seed-42 band artifact is not the selected control")
    expected_configuration = _control_configuration(first)
    for artifact in rows[1:]:
        if artifact.job.stage != "control_repeats":
            raise ValueError("seeds 43 through 51 must be unchanged-control repeats")
        if (
            artifact.parameters.get("selected_control_job_id")
            != selected_control_job_id
            or _control_configuration(artifact) != expected_configuration
        ):
            raise ValueError(
                "empirical bands require the exact selected control configuration"
            )


def empirical_bands(artifacts: Iterable[VerifiedArtifact]) -> dict[str, MetricBand]:
    rows = list(artifacts)
    _validate_control_band_artifacts(rows)
    result: dict[str, MetricBand] = {}
    for metric in METRICS:
        spread = statistics.stdev(artifact.metrics[metric] for artifact in rows)
        result[metric] = MetricBand(spread, _round_up_one_significant_digit(spread))
    return result


def write_empirical_bands(
    artifacts: Iterable[VerifiedArtifact], destination: Path
) -> None:
    rows = list(artifacts)
    bands = empirical_bands(rows)
    document = {
        "dataset_size": "native-50m",
        "description": (
            "Sample standard deviations from ten unchanged control seeds; "
            "not confidence intervals."
        ),
        "seeds": [artifact.job.seed for artifact in rows],
        "run_names": [artifact.job.run_name for artifact in rows],
        "metrics": {metric: band.to_dict() for metric, band in bands.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def select_best(
    artifacts: Iterable[VerifiedArtifact],
    *,
    metric_bands: Mapping[str, float] | None = None,
) -> VerifiedArtifact:
    candidates = list(artifacts)
    if not candidates:
        raise ValueError("cannot select from no verified artifacts")
    bands = metric_bands or {}
    best_recall = max(candidate.metrics["recall@100"] for candidate in candidates)
    recall_band = float(bands.get("recall@100", 0.0))
    candidates = [
        candidate
        for candidate in candidates
        if best_recall - candidate.metrics["recall@100"] <= recall_band
    ]
    best_ndcg = max(candidate.metrics["ndcg@100"] for candidate in candidates)
    candidates = [
        candidate
        for candidate in candidates
        if candidate.metrics["ndcg@100"] == best_ndcg
    ]
    return min(
        candidates, key=lambda candidate: (candidate.wall_seconds, candidate.job.id)
    )


def select_control_with_fit_gate(
    artifacts: Iterable[VerifiedArtifact],
    fit_evidence: FitEvidence,
) -> VerifiedArtifact:
    candidates = list(artifacts)
    if any(
        candidate.job.stage not in {"control_tuning", "lr_boundary"}
        or (
            candidate.job.stage == "lr_boundary"
            and candidate.parameters.get("builder") != "control"
        )
        for candidate in candidates
    ):
        raise ValueError("control fit gate accepts only control tuning artifacts")
    eligible = [
        candidate
        for candidate in candidates
        if candidate.parameters.get("batch_size") in fit_evidence.eligible_batches
    ]
    if not eligible:
        raise ValueError("no control candidate has a successful max-LiGR fit probe")
    return select_best(eligible)


def aggregate_artifacts(
    artifacts: Iterable[VerifiedArtifact], *, run_name: str
) -> VerifiedArtifact:
    rows = list(artifacts)
    if not rows:
        raise ValueError("cannot aggregate no artifacts")
    method = rows[0].job.method
    if any(row.job.method != method for row in rows):
        raise ValueError("aggregate rows must use one method")
    synthetic_job = ApprovedJob(
        id=f"aggregate:{method}",
        run_name=run_name,
        stage=rows[0].job.stage,
        method=method,
        seed=rows[0].job.seed,
    )
    metrics = {
        metric: statistics.fmean(row.metrics[metric] for row in rows)
        for metric in METRICS
    }
    common_costs = set.intersection(*(set(row.costs) for row in rows))
    costs = {
        name: statistics.fmean(row.costs[name] for row in rows) for name in common_costs
    }
    return VerifiedArtifact(
        synthetic_job,
        rows[0].path,
        {"seeds": [row.job.seed for row in rows]},
        metrics,
        {"dataset_size": rows[0].metadata["dataset_size"]},
        costs,
    )


def mixed_sampler_winner(
    baseline: VerifiedArtifact,
    candidates: Iterable[VerifiedArtifact],
    metric_bands: Mapping[str, float],
) -> VerifiedArtifact | None:
    recall_band = float(metric_bands["recall@100"])
    coverage_band = float(metric_bands["coverage@100"])
    eligible = [
        candidate
        for candidate in candidates
        if candidate.metrics["recall@100"]
        >= baseline.metrics["recall@100"] - recall_band
        and candidate.metrics["coverage@100"]
        > baseline.metrics["coverage@100"] + coverage_band
    ]
    return None if not eligible else select_best(eligible, metric_bands=metric_bands)
