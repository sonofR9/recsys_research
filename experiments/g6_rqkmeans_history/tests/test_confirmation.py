import json
from pathlib import Path

import pytest

from experiments.g6_rqkmeans_history.configs.confirmation import (
    build_confirmation_experiment,
)
from experiments.g6_rqkmeans_history.launchers.confirmation_runtime import (
    decode_job,
    encode_job,
)
from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    collision_search_manifest,
)
from experiments.g6_rqkmeans_history.protocol.confirmation import (
    load_confirmation_manifest,
)
from experiments.g6_rqkmeans_history.protocol.rq1_manifest import (
    rq1_search_manifest,
)


def _surface(tmp_path: Path, *, boundary: bool = False) -> Path:
    rq1 = rq1_search_manifest()
    collision = collision_search_manifest()
    random = rq1.jobs_for_initialization("random")[1]
    content = rq1.jobs_for_initialization("content_pca")[1]
    suffix = next(
        job for job in collision.jobs if job.policy == "suffix" and job.coordinate.trial == 1
    )
    none = next(
        job for job in collision.jobs if job.policy == "none" and job.coordinate.trial == 1
    )
    row = lambda job: {"job_id": job.id, "parameters": job.parameters}
    document = {
        "schema": "g6-rq1-rq3-surface/v1",
        "rq1_manifest_sha256": rq1.sha256,
        "collision_manifest_sha256": collision.sha256,
        "rq1": {
            "selected": {"random": row(random), "content_pca": row(content)},
            "boundary_triggered": {
                "random": ["deep_learning_rate"] if boundary else [],
                "content_pca": [],
            },
        },
        "rq2_rq3": {
            "selected": {"suffix": row(suffix), "none": row(none)},
            "lr_boundary_triggered": {"suffix": [], "none": []},
        },
    }
    path = tmp_path / "surface.json"
    path.write_text(json.dumps(document))
    return path


def test_confirmation_manifest_freezes_all_required_seed_repeats(
    tmp_path: Path,
) -> None:
    manifest = load_confirmation_manifest(_surface(tmp_path))

    assert len(manifest.jobs) == 12
    assert {job.seed for job in manifest.jobs if job.family == "rq1"} == {43, 44, 45}
    assert {job.seed for job in manifest.jobs if job.family == "collision"} == {43, 44}
    assert len({job.id for job in manifest.jobs}) == len(manifest.jobs)
    assert all("_t" in job.variant for job in manifest.jobs)


def test_confirmation_manifest_refuses_unresolved_lr_boundaries(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="boundary extension"):
        load_confirmation_manifest(_surface(tmp_path, boundary=True))


def test_confirmation_contract_and_config_preserve_seed(tmp_path: Path) -> None:
    manifest = load_confirmation_manifest(_surface(tmp_path))
    job = manifest.jobs[0]

    assert decode_job(encode_job(job, manifest), manifest) == job
    experiment = build_confirmation_experiment(job)
    assert experiment.seed == job.seed
    assert experiment.run_name == job.run_name
