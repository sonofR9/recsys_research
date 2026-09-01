from dataclasses import asdict
import json

import pytest

from experiments.g6_rqkmeans_history.configs.collision_policy import (
    build_collision_policy_experiment,
)
from experiments.g6_rqkmeans_history.configs.collision_recovery import (
    build_collision_recovery_experiment,
)
from experiments.g6_rqkmeans_history.launchers.collision_recovery_runtime import (
    decode_recovery_job,
    encode_recovery_job,
    write_recovery_contract,
)
from experiments.g6_rqkmeans_history.protocol.collision_recovery import (
    recovery_job,
)


def test_recovery_changes_only_physical_run_identity() -> None:
    job = recovery_job()
    source = job.source_job
    recovery = build_collision_recovery_experiment(job)
    original = build_collision_policy_experiment(source)

    original_fields = asdict(original)
    recovery_fields = asdict(recovery)
    assert recovery_fields.pop("run_name") == job.run_name
    assert original_fields.pop("run_name") == source.run_name
    assert recovery_fields == original_fields


def test_recovery_payload_and_contract_are_manifest_bound(tmp_path) -> None:
    job = recovery_job()
    encoded = encode_recovery_job(job)

    assert decode_recovery_job(encoded) == job
    path = write_recovery_contract(job, tmp_path)
    assert json.loads(path.read_text()) == {
        "recovery_manifest_sha256": job.manifest_sha256,
        "job": job.to_dict(),
    }

    changed = encoded[:-2] + "AA"
    with pytest.raises(RuntimeError, match="invalid|absent"):
        decode_recovery_job(changed)
