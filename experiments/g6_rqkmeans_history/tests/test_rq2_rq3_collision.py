import pytest
import torch

from dcn.config import SemanticHistoryExperiment
from dcn.semantic import SemanticCodes

from experiments.g6_rqkmeans_history.configs.collision_policy import (
    build_collision_policy_experiment,
)
from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    COLLISION_POLICIES,
    collision_search_coordinates,
    collision_search_manifest,
    validate_collision_diagnostics,
    validate_collision_symbol_cap,
)


def test_manifest_exposes_the_approved_paired_search() -> None:
    manifest = collision_search_manifest()
    coordinates = collision_search_coordinates()

    assert manifest.to_dict()["dataset_size"] == "native-50m"
    assert manifest.to_dict()["source_selection_sha256"] == (
        "8391def6cfddbeb4cb1b048f3d4fed62e4bf0e304270e8d21a7d2de4ded0646b"
    )
    assert manifest.to_dict()["search_space"] == {
        "num_levels": [2, 3, 4, 5],
        "shared_num_codes": [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        "kmeans_iterations": [10, 20, 40],
        "embedding_learning_rate": [0.008, 0.512, "log_uniform"],
        "deep_learning_rate": [0.002, 0.128, "log_uniform"],
    }
    assert len(coordinates) == 40
    assert len(manifest.jobs) == 80
    assert len(manifest.new_physical_jobs) == 79
    assert [job.policy for job in manifest.jobs[:2]] == list(COLLISION_POLICIES)
    assert all(
        manifest.jobs[2 * index].coordinate
        == manifest.jobs[2 * index + 1].coordinate
        for index in range(40)
    )
    reused = [job for job in manifest.jobs if job.reused]
    assert len(reused) == 1
    assert reused[0].physical_job_id == (
        "lr_boundary:boundary_item_frozen_sid_event_embedding_learning_rate_0"
    )
    assert reused[0].physical_run_name == (
        "g6_rq0_boundary_item_frozen_sid_event_"
        "embedding_learning_rate_0_native50m"
    )


def test_manifest_preserves_the_four_approved_anchors() -> None:
    anchors = collision_search_coordinates()[:4]

    assert [
        (coordinate.num_levels, coordinate.num_codes, coordinate.kmeans_iterations)
        for coordinate in anchors
    ] == [(3, 512, 20), (2, 64, 20), (4, 1024, 20), (5, 4096, 40)]
    assert anchors[0].embedding_learning_rate == 0.3620386719675124
    assert anchors[0].deep_learning_rate == 0.03463626154088337
    assert len({coordinate.identity for coordinate in collision_search_coordinates()}) == 40


@pytest.mark.parametrize("policy", COLLISION_POLICIES)
def test_approved_collision_job_builds_the_selected_rq0_representation(
    policy: str,
) -> None:
    job = next(
        job
        for job in collision_search_manifest().jobs
        if job.policy == policy and job.coordinate.trial == 0
    )

    experiment = build_collision_policy_experiment(job)

    assert experiment.history_representation == "item_frozen_sid_event"
    assert experiment.representation_width == 128
    assert experiment.semantic.collision_policy == policy
    assert experiment.semantic.num_levels == 3
    assert experiment.semantic.num_codes == 512
    assert experiment.semantic.kmeans_iterations == 20
    assert experiment.dataloader.batch_size == 256
    assert experiment.seed == 42


def test_collision_builder_rejects_a_job_outside_the_manifest() -> None:
    job = collision_search_manifest().jobs[0]
    changed = job.with_parameters(num_codes=7)

    with pytest.raises(ValueError, match="approved paired search"):
        build_collision_policy_experiment(changed)


def test_collision_builder_rejects_oversized_suffix_before_creating_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = collision_search_manifest().jobs[0]
    experiment = build_collision_policy_experiment(job)
    experiment.__dict__["semantic_codes"] = SemanticCodes(
        item_ids=torch.tensor([1]),
        codes=torch.tensor([[0, 0, 0, 8192]]),
        codes_per_level=(512, 512, 512, 8193),
    )
    monkeypatch.setattr(
        SemanticHistoryExperiment,
        "create_runner",
        lambda self: pytest.fail("training runner was created"),
    )

    with pytest.raises(ValueError, match="suffix level exceeds"):
        experiment.create_runner()


def test_collision_symbol_cap_includes_the_suffix_level() -> None:
    codes = SemanticCodes(
        item_ids=torch.tensor([1]),
        codes=torch.tensor([[0, 0]]),
        codes_per_level=(8192, 8193),
    )

    with pytest.raises(ValueError, match="suffix level exceeds"):
        validate_collision_symbol_cap(codes, policy="suffix", base_levels=1)


def test_paired_search_rejects_none_when_counterfactual_suffix_exceeds_cap() -> None:
    codes = SemanticCodes.without_collision_suffix(
        torch.arange(1, 8194),
        torch.zeros((8193, 2), dtype=torch.int64),
        num_codes=32,
    )

    with pytest.raises(ValueError, match="counterfactual suffix"):
        validate_collision_symbol_cap(
            codes,
            policy="none",
            base_levels=2,
            require_suffix_feasibility=True,
        )


@pytest.mark.parametrize(
    ("policy", "suffix_symbols"),
    [("suffix", 17), ("none", 0)],
)
def test_collision_diagnostics_match_the_configured_policy(
    policy: str,
    suffix_symbols: int,
) -> None:
    validate_collision_diagnostics(
        policy=policy,
        diagnostics={
            "collision_policy": policy,
            "collision_suffix_symbols": suffix_symbols,
        },
    )


def test_collision_diagnostics_reject_base_width_as_a_disabled_suffix() -> None:
    with pytest.raises(ValueError, match="must report zero suffix symbols"):
        validate_collision_diagnostics(
            policy="none",
            diagnostics={
                "collision_policy": "none",
                "collision_suffix_symbols": 512,
            },
        )
