import base64
from collections import Counter
import json

import pytest

from experiments.g6_rqkmeans_history.launchers.collision_runtime import (
    cache_safe_waves,
    decode_collision_job,
    encode_collision_job,
    initial_physical_jobs,
    select_wave,
)
from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    collision_search_manifest,
)


def test_collision_job_encoding_is_bound_to_the_approved_manifest() -> None:
    job = collision_search_manifest().jobs[1]

    assert decode_collision_job(encode_collision_job(job)) == job

    contract = json.loads(base64.urlsafe_b64decode(encode_collision_job(job)).decode())
    contract["job"]["policy"] = "suffix"
    mutated = base64.urlsafe_b64encode(json.dumps(contract).encode()).decode()
    with pytest.raises(RuntimeError, match="approved manifest"):
        decode_collision_job(mutated)


def test_initial_physical_jobs_reuse_only_the_exact_rq0_suffix_anchor() -> None:
    jobs = initial_physical_jobs()

    assert len(jobs) == 79
    assert all(not (job.policy == "suffix" and job.coordinate.trial == 0) for job in jobs)
    assert any(job.policy == "none" and job.coordinate.trial == 0 for job in jobs)


def test_collision_runtime_rejects_rebuilding_the_rq0_carryover() -> None:
    carryover = collision_search_manifest().jobs[0]

    assert carryover.reused
    with pytest.raises(ValueError, match="carryover"):
        encode_collision_job(carryover)


def test_cache_safe_waves_never_fit_one_base_tokenizer_twice_concurrently() -> None:
    jobs = initial_physical_jobs()
    waves = cache_safe_waves(jobs)

    assert Counter(job.id for wave in waves for job in wave) == Counter(
        job.id for job in jobs
    )
    for wave in waves:
        keys = {
            (
                job.coordinate.num_levels,
                job.coordinate.num_codes,
                job.coordinate.kmeans_iterations,
            )
            for job in wave
        }
        assert len(keys) == len(wave)


def test_cache_safe_waves_use_the_minimum_number_of_rounds() -> None:
    jobs = initial_physical_jobs()
    waves = cache_safe_waves(jobs)
    multiplicities: dict[tuple[int, int, int], int] = {}
    for job in jobs:
        coordinate = job.coordinate
        key = (
            coordinate.num_levels,
            coordinate.num_codes,
            coordinate.kmeans_iterations,
        )
        multiplicities[key] = multiplicities.get(key, 0) + 1

    assert len(waves) == max(multiplicities.values())


def test_select_wave_rejects_an_out_of_range_round() -> None:
    waves = cache_safe_waves(initial_physical_jobs())

    assert select_wave(0) == waves[0]
    with pytest.raises(ValueError, match="wave index"):
        select_wave(len(waves))


def test_select_wave_partitions_an_explicit_subset_safely() -> None:
    jobs = initial_physical_jobs()[:7]
    waves = cache_safe_waves(jobs)

    assert select_wave(0, [job.id for job in jobs]) == waves[0]
