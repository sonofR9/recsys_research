from types import SimpleNamespace

import pytest

from dcn.config import SemanticIdConfig
from experiments.g6_rqkmeans_history.analysis.select_surfaces import (
    SurfaceArtifact,
    artifact_identity,
    _validate_diagnostics,
    select_by_recall_ndcg,
)
from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    collision_search_manifest,
)
from experiments.g6_rqkmeans_history.protocol.collision_recovery import recovery_job


def _artifact(job_id: str, recall: float, ndcg: float) -> SurfaceArtifact:
    return SurfaceArtifact(
        job=SimpleNamespace(id=job_id),
        run_name=job_id,
        metrics={"recall@100": recall, "ndcg@100": ndcg},
        metadata={},
        diagnostics={},
        artifact_sha256={},
    )


def test_recovery_replaces_only_the_approved_physical_artifact() -> None:
    manifest = collision_search_manifest()
    recovery = recovery_job()
    source = recovery.source_job

    run_name, contract_name, contract = artifact_identity(source, manifest.sha256)

    assert run_name == recovery.run_name
    assert contract_name == "g6_rq2_rq3_recovery_job.json"
    assert contract == {
        "recovery_manifest_sha256": recovery.manifest_sha256,
        "job": recovery.to_dict(),
    }

    ordinary = next(job for job in manifest.jobs if not job.reused and job != source)
    run_name, contract_name, contract = artifact_identity(ordinary, manifest.sha256)
    assert run_name == ordinary.physical_run_name
    assert contract_name == "g6_rq2_rq3_job.json"
    assert contract == {"manifest_sha256": manifest.sha256, "job": ordinary.to_dict()}


def test_selection_uses_ndcg_only_inside_the_recall_band() -> None:
    lower_recall = _artifact("lower", 0.129, 0.06)
    best_recall = _artifact("best", 0.130, 0.05)

    assert (
        select_by_recall_ndcg([lower_recall, best_recall], recall_band=0.002)
        == lower_recall
    )


def test_selection_rejects_ndcg_outside_the_recall_band() -> None:
    lower_recall = _artifact("lower", 0.127, 0.06)
    best_recall = _artifact("best", 0.130, 0.05)

    assert (
        select_by_recall_ndcg([lower_recall, best_recall], recall_band=0.002)
        == best_recall
    )


def test_diagnostics_bind_collision_semantics_and_cache_identity() -> None:
    job = next(job for job in collision_search_manifest().jobs if job.policy == "suffix")
    config = SemanticIdConfig(
        quantizer="kmeans",
        num_levels=job.coordinate.num_levels,
        num_codes=job.coordinate.num_codes,
        kmeans_iterations=job.coordinate.kmeans_iterations,
        collision_policy="suffix",
        seed=42,
    )
    diagnostics = {
        "num_levels": job.coordinate.num_levels,
        "shared_num_codes": job.coordinate.num_codes,
        "semantic_cache_key": config.cache_key,
        "collision_policy": "suffix",
        "collision_suffix_symbols": 1,
    }
    _validate_diagnostics(job, diagnostics, require_policy=True)
    with pytest.raises(ValueError, match="suffix"):
        _validate_diagnostics(
            job,
            {**diagnostics, "collision_suffix_symbols": -1},
            require_policy=True,
        )
    with pytest.raises(ValueError, match="cache key"):
        _validate_diagnostics(
            job,
            {**diagnostics, "semantic_cache_key": "wrong"},
            require_policy=True,
        )
