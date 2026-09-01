import base64
import json

import pytest

from experiments.g6_rqkmeans_history.launchers.rq1_runtime import (
    decode_rq1_job,
    encode_rq1_job,
    initial_rq1_jobs,
    rq1_cache_safe_phase,
)
from experiments.g6_rqkmeans_history.protocol.rq1_manifest import (
    rq1_search_manifest,
)


def test_rq1_job_encoding_is_bound_to_the_approved_manifest() -> None:
    job = initial_rq1_jobs()[0]

    assert decode_rq1_job(encode_rq1_job(job)) == job

    contract = json.loads(base64.urlsafe_b64decode(encode_rq1_job(job)).decode())
    contract["job"]["initialization"] = "random"
    mutated = base64.urlsafe_b64encode(json.dumps(contract).encode()).decode()
    with pytest.raises(RuntimeError, match="approved manifest"):
        decode_rq1_job(mutated)


def test_rq1_runtime_emits_only_the_22_new_physical_jobs() -> None:
    jobs = initial_rq1_jobs()

    assert jobs == rq1_search_manifest().new_physical_jobs
    assert len(jobs) == 22
    assert not any(job.reused for job in jobs)


def test_rq1_runtime_rejects_rebuilding_a_bound_carryover() -> None:
    carryover = rq1_search_manifest().jobs_for_initialization("random")[0]

    with pytest.raises(ValueError, match="carryover"):
        encode_rq1_job(carryover)


def test_rq1_cache_safe_phases_prime_the_shared_tokenizer_first() -> None:
    jobs = initial_rq1_jobs()

    assert rq1_cache_safe_phase(0) == jobs[:1]
    assert rq1_cache_safe_phase(1) == jobs[1:]
    with pytest.raises(ValueError, match="phase index"):
        rq1_cache_safe_phase(2)
