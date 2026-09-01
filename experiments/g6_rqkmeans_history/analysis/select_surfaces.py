from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol, Sequence

from dcn.config import SemanticIdConfig
from dcn.eval.ranking_evidence import load_ranking_evidence
from experiments.g6_rqkmeans_history.protocol.manifest import RANKING_EVIDENCE_GROUP
from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    CollisionSearchJob,
    collision_search_manifest,
    validate_collision_diagnostics,
)
from experiments.g6_rqkmeans_history.protocol.collision_recovery import (
    SOURCE_JOB_ID as RECOVERY_SOURCE_JOB_ID,
    recovery_job,
)
from experiments.g6_rqkmeans_history.protocol.rq1_manifest import (
    Rq1SearchJob,
    rq1_search_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_ROOT = PROJECT_ROOT / "generated/logs"
OUTPUT = (
    PROJECT_ROOT
    / "experiments/g6_rqkmeans_history/evidence/rq1_rq3_surface_native50m.json"
)


class SearchJob(Protocol):
    id: str
    physical_run_name: str
    reused: bool
    parameters: dict[str, Any]


@dataclass(frozen=True)
class SurfaceArtifact:
    job: Rq1SearchJob | CollisionSearchJob
    run_name: str
    metrics: dict[str, float]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]
    artifact_sha256: dict[str, str]

    def row(self) -> dict[str, Any]:
        return {
            "job_id": self.job.id,
            "run_name": self.run_name,
            "reused": self.job.reused,
            "parameters": self.job.parameters,
            "metrics": self.metrics,
            "training": {
                "best_epoch": self.metadata["best_epoch"],
                "epochs_trained": self.metadata["epochs_trained"],
                "peak_memory_gb": self.metadata.get("peak_memory_gb"),
            },
            "diagnostics": self.diagnostics,
            "artifact_sha256": self.artifact_sha256,
        }


def select_by_recall_ndcg(
    artifacts: Sequence[SurfaceArtifact], *, recall_band: float
) -> SurfaceArtifact:
    if not artifacts:
        raise ValueError("selection requires artifacts")
    best_recall = max(artifact.metrics["recall@100"] for artifact in artifacts)
    recall_tied = [
        artifact
        for artifact in artifacts
        if artifact.metrics["recall@100"] >= best_recall - recall_band
    ]
    return max(
        recall_tied,
        key=lambda artifact: (
            artifact.metrics["ndcg@100"],
            -artifacts.index(artifact),
        ),
    )


def collect_surfaces(logs_root: Path = LOGS_ROOT) -> dict[str, Any]:
    rq1_manifest = rq1_search_manifest()
    collision_manifest = collision_search_manifest()
    rq1 = [_load_artifact(job, logs_root, rq1_manifest.sha256) for job in rq1_manifest.jobs]
    collision = [
        _load_artifact(job, logs_root, collision_manifest.sha256)
        for job in collision_manifest.jobs
    ]
    rq1_selected = {
        mode: select_by_recall_ndcg(
            [artifact for artifact in rq1 if artifact.job.initialization == mode],
            recall_band=0.002,
        )
        for mode in ("random", "content_pca")
    }
    collision_selected = {
        policy: select_by_recall_ndcg(
            [artifact for artifact in collision if artifact.job.policy == policy],
            recall_band=0.002,
        )
        for policy in ("suffix", "none")
    }
    return {
        "schema": "g6-rq1-rq3-surface/v1",
        "dataset_size": "native-50m",
        "rq1_manifest_sha256": rq1_manifest.sha256,
        "collision_manifest_sha256": collision_manifest.sha256,
        "rq1": {
            "selected": {
                mode: artifact.row() for mode, artifact in rq1_selected.items()
            },
            "boundary_triggered": {
                mode: _boundary_groups(artifact.job, rq1_manifest.jobs)
                for mode, artifact in rq1_selected.items()
            },
            "rows": [artifact.row() for artifact in rq1],
        },
        "rq2_rq3": {
            "selected": {
                policy: artifact.row()
                for policy, artifact in collision_selected.items()
            },
            "lr_boundary_triggered": {
                policy: _boundary_groups(artifact.job, collision_manifest.jobs)
                for policy, artifact in collision_selected.items()
            },
            "rows": [artifact.row() for artifact in collision],
        },
    }


def _load_artifact(
    job: Rq1SearchJob | CollisionSearchJob,
    logs_root: Path,
    manifest_sha256: str,
) -> SurfaceArtifact:
    run_name, contract_name, expected_contract = artifact_identity(
        job, manifest_sha256
    )
    directory = logs_root / run_name
    paths = {
        "job_contract": directory / contract_name,
        "final_metrics": directory / "final_metrics.json",
        "training_metadata": directory / "training_metadata.json",
        "ranking_evidence": directory / "ranking_evidence.pt",
        "sid_diagnostics": directory / "semantic_id_diagnostics.json",
        "sweep_log": directory / "sweep.log",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"{run_name}: missing artifacts {missing}")
    if not job.reused:
        contract = _read_json(paths["job_contract"])
        if contract != expected_contract:
            raise ValueError(f"{run_name}: job contract changed")
    else:
        _validate_rq0_contract(job, _read_json(paths["job_contract"]))
    metrics = _numeric_metrics(_read_json(paths["final_metrics"]))
    for name in ("recall@100", "ndcg@100", "mrr@100", "coverage@100"):
        if name not in metrics:
            raise ValueError(f"{job.physical_run_name}: missing {name}")
    metadata = _read_json(paths["training_metadata"])
    _validate_metadata(job, metadata)
    if not job.reused:
        _validate_rq1_initialization(job, metadata)
    diagnostics = _read_json(paths["sid_diagnostics"])
    _validate_diagnostics(job, diagnostics, require_policy=not job.reused)
    load_ranking_evidence(
        logs_root / ".ranking-evidence" / RANKING_EVIDENCE_GROUP / "context.pt",
        paths["ranking_evidence"],
    )
    return SurfaceArtifact(
        job=job,
        run_name=run_name,
        metrics=metrics,
        metadata=metadata,
        diagnostics=diagnostics,
        artifact_sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
    )


def artifact_identity(
    job: Rq1SearchJob | CollisionSearchJob, manifest_sha256: str
) -> tuple[str, str, dict[str, Any] | None]:
    if isinstance(job, CollisionSearchJob) and job.id == RECOVERY_SOURCE_JOB_ID:
        recovery = recovery_job()
        if recovery.source_job != job or recovery.source_manifest_sha256 != manifest_sha256:
            raise ValueError(f"{job.id}: recovery source changed")
        return (
            recovery.run_name,
            "g6_rq2_rq3_recovery_job.json",
            {
                "recovery_manifest_sha256": recovery.manifest_sha256,
                "job": recovery.to_dict(),
            },
        )
    contract_name = (
        "g6_rq0_job.json"
        if job.reused
        else (
            "g6_rq1_job.json"
            if isinstance(job, Rq1SearchJob)
            else "g6_rq2_rq3_job.json"
        )
    )
    expected = (
        None
        if job.reused
        else {"manifest_sha256": manifest_sha256, "job": job.to_dict()}
    )
    return job.physical_run_name, contract_name, expected


def _validate_metadata(job: SearchJob, metadata: dict[str, Any]) -> None:
    parameters = job.parameters
    expected = {
        "dataset_size": "50m",
        "batch_size": parameters["batch_size"],
        "seed": parameters["training_seed"],
        "embedding_learning_rate": parameters["embedding_learning_rate"],
        "deep_learning_rate": parameters["deep_learning_rate"],
        "num_epochs": 15,
        "lr_schedule_horizon_epochs": 15,
        "lr_horizon_complete": True,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(
                f"{job.physical_run_name}: metadata {name!r} changed"
            )


def _validate_rq0_contract(
    job: Rq1SearchJob | CollisionSearchJob, contract: dict[str, Any]
) -> None:
    source_job = contract.get("job")
    source_parameters = contract.get("parameters")
    if (
        contract.get("manifest_sha256")
        != "f8694eb0503e47a25fe6f278f66598ba6c3fdebc8800433339b8cc93ef8650b1"
        or not isinstance(source_job, dict)
        or source_job.get("id") != job.physical_job_id
        or source_job.get("run_name") != job.physical_run_name
        or not isinstance(source_parameters, dict)
    ):
        raise ValueError(f"{job.physical_run_name}: RQ0 source contract changed")
    expected = {
        "batch_size": job.parameters["batch_size"],
        "validation_batch_size": job.parameters["validation_batch_size"],
        "representation": job.parameters["representation"],
        "representation_width": job.parameters["representation_width"],
        "num_levels": job.parameters["num_levels"],
        "num_codes": job.parameters["num_codes"],
        "embedding_learning_rate": job.parameters["embedding_learning_rate"],
        "deep_learning_rate": job.parameters["deep_learning_rate"],
    }
    for name, value in expected.items():
        if source_parameters.get(name) != value:
            raise ValueError(
                f"{job.physical_run_name}: RQ0 source parameter {name!r} changed"
            )


def _validate_diagnostics(
    job: Rq1SearchJob | CollisionSearchJob,
    diagnostics: dict[str, Any],
    *,
    require_policy: bool,
) -> None:
    parameters = job.parameters
    if (
        diagnostics.get("num_levels") != parameters["num_levels"]
        or diagnostics.get("shared_num_codes") != parameters["num_codes"]
    ):
        raise ValueError(f"{job.physical_run_name}: SID diagnostics changed")
    policy = job.policy if isinstance(job, CollisionSearchJob) else "suffix"
    expected_cache_key = SemanticIdConfig(
        quantizer="kmeans",
        num_levels=parameters["num_levels"],
        num_codes=parameters["num_codes"],
        kmeans_iterations=parameters.get("kmeans_iterations", 20),
        collision_policy=policy,
        seed=42,
    ).cache_key
    if diagnostics.get("semantic_cache_key") != expected_cache_key:
        raise ValueError(f"{job.physical_run_name}: semantic cache key changed")
    if require_policy:
        validate_collision_diagnostics(policy=policy, diagnostics=diagnostics)
    if isinstance(job, CollisionSearchJob) and job.policy == "none":
        bucket_max = diagnostics.get("collision_bucket_size_max")
        if (
            not isinstance(bucket_max, int)
            or isinstance(bucket_max, bool)
            or bucket_max + 1 > 8192
        ):
            raise ValueError(
                f"{job.physical_run_name}: counterfactual suffix exceeds cap"
            )


def _validate_rq1_initialization(
    job: Rq1SearchJob | CollisionSearchJob, metadata: dict[str, Any]
) -> None:
    if not isinstance(job, Rq1SearchJob):
        return
    expected_projection = (
        "per_level_centered_pca_v1"
        if job.initialization == "content_pca"
        else None
    )
    if (
        metadata.get("sid_lookup_initialization") != job.initialization
        or metadata.get("sid_lookup_projection") != expected_projection
    ):
        raise ValueError(f"{job.physical_run_name}: SID initialization changed")
    diagnostics = metadata.get("sid_initialization_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"{job.physical_run_name}: SID initialization is missing")
    required_hashes = (
        "base_rows_before_sha256",
        "base_rows_after_sha256",
        "non_base_rows_sha256",
        "codebook_centroids_sha256",
    )
    if (
        diagnostics.get("mode") != job.initialization
        or diagnostics.get("projection") != expected_projection
        or diagnostics.get("rng_nonadvancing") is not True
        or any(
            not isinstance(diagnostics.get(name), str)
            or len(diagnostics[name]) != 64
            for name in required_hashes
        )
    ):
        raise ValueError(f"{job.physical_run_name}: SID initialization is invalid")
    changed = (
        diagnostics["base_rows_before_sha256"]
        != diagnostics["base_rows_after_sha256"]
    )
    if changed != (job.initialization == "content_pca"):
        raise ValueError(f"{job.physical_run_name}: SID overwrite does not match mode")
    levels = diagnostics.get("levels")
    if not isinstance(levels, list) or len(levels) != job.parameters["num_levels"]:
        raise ValueError(f"{job.physical_run_name}: SID level scales are missing")
    for level in levels:
        if (
            not isinstance(level, dict)
            or not math.isclose(
                level.get("random_rms", math.nan),
                level.get("initialized_rms", math.inf),
                rel_tol=1e-6,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(f"{job.physical_run_name}: SID RMS changed")


def _boundary_groups(
    job: Rq1SearchJob | CollisionSearchJob,
    jobs: Sequence[Rq1SearchJob | CollisionSearchJob],
) -> list[str]:
    groups = []
    for name in ("embedding_learning_rate", "deep_learning_rate"):
        value = getattr(job.coordinate, name)
        tested = [getattr(candidate.coordinate, name) for candidate in jobs]
        if value in {min(tested), max(tested)}:
            groups.append(name)
    return groups


def _numeric_metrics(document: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    for name, value in document.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"metric {name!r} is not finite")
            metrics[name] = numeric
    return metrics


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected an object")
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    document = collect_surfaces()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if arguments.output.exists() and arguments.output.read_text() != content:
        raise RuntimeError(f"surface evidence already differs: {arguments.output}")
    arguments.output.write_text(content)
    for question, selections in (
        ("rq1", document["rq1"]["selected"]),
        ("rq2_rq3", document["rq2_rq3"]["selected"]),
    ):
        print(question)
        for name, row in selections.items():
            print(name, row["job_id"], row["metrics"]["recall@100"])


if __name__ == "__main__":
    main()
