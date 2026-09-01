from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from dcn.config import SemanticHistoryExperiment, SemanticIdConfig
from dcn.semantic import ResidualCodebooks, SemanticCodes
from experiments.g6_rqkmeans_history.configs.rq1 import (
    Rq1InitializationExperiment,
    build_rq1_initialization,
    build_rq1_search_experiment,
    project_centroids_with_pca,
    rq1_learned_sid_embedding,
)
from experiments.g6_rqkmeans_history.protocol.rq1_manifest import (
    DEEP_LR_BOUNDS,
    EMBEDDING_LR_BOUNDS,
    rq1_search_manifest,
)


def _semantic_artifacts() -> tuple[SemanticCodes, ResidualCodebooks]:
    codes = SemanticCodes.with_collision_suffix(
        item_ids=torch.tensor([1, 2, 3, 4]),
        codes=torch.tensor([[0, 0], [0, 0], [1, 1], [2, 2]]),
        num_codes=3,
    )
    codebooks = ResidualCodebooks(
        torch.tensor(
            [
                [
                    [1.0, 2.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 2.0, 1.0],
                ],
                [
                    [0.5, 0.0, 1.0, 2.0],
                    [1.5, 1.0, 0.0, 1.0],
                    [0.0, 2.0, 1.0, 0.5],
                ],
            ]
        )
    )
    return codes, codebooks


def _small_experiment(
    initialization: str,
    *,
    representation_width: int = 2,
) -> Rq1InitializationExperiment:
    codes, codebooks = _semantic_artifacts()
    experiment = Rq1InitializationExperiment(
        sid_lookup_initialization=initialization,
        history_representation="item_learned_frozen_sid_event",
        representation_width=representation_width,
        semantic=SemanticIdConfig(num_levels=2, num_codes=3),
        transformer=replace(
            SemanticHistoryExperiment.transformer,
            dim=8,
            num_layers=1,
            nhead=2,
            num_kv_heads=1,
            ffn_intermediate_dim=16,
        ),
        item_embedding_dim=4,
        mup_base_dim=4,
        mup_delta_dim=6,
        initializer_std=0.02,
    )
    experiment.__dict__["item_embeddings"] = SimpleNamespace(num_known_ids=4)
    experiment.__dict__["artifacts"] = SimpleNamespace(
        item_id_column="compact_item_id"
    )
    experiment.__dict__["semantic_codes"] = codes
    experiment.__dict__["semantic_codebooks"] = codebooks
    return experiment


def _model_with_rng(
    initialization: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Rq1InitializationExperiment, torch.nn.Module, torch.Tensor]:
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "0")
    codes, codebooks = _semantic_artifacts()
    monkeypatch.setattr(
        Rq1InitializationExperiment, "semantic_codes", property(lambda _: codes)
    )
    monkeypatch.setattr(
        Rq1InitializationExperiment,
        "semantic_codebooks",
        property(lambda _: codebooks),
    )
    torch.manual_seed(37)
    experiment = _small_experiment(initialization)
    model = experiment.base_model
    return experiment, model, torch.get_rng_state()


def test_rq1_builder_freezes_the_approved_family() -> None:
    experiment = build_rq1_initialization(
        "content_pca",
        embedding_learning_rate=0.1,
        deep_learning_rate=0.01,
        run_name="rq1_content",
    )

    assert experiment.size == "50m"
    assert experiment.dataloader.batch_size == 256
    assert experiment.history_representation == "item_learned_frozen_sid_event"
    assert experiment.semantic.num_levels == 4
    assert experiment.semantic.num_codes == 512
    assert experiment.representation_width == 32
    assert experiment.sid_lookup_initialization == "content_pca"

    with pytest.raises(ValueError, match="unknown SID lookup initialization"):
        build_rq1_initialization(
            "unknown",  # type: ignore[arg-type]
            embedding_learning_rate=0.1,
            deep_learning_rate=0.01,
            run_name="rq1_unknown",
        )


def test_rq1_manifest_fixes_paired_coordinates_and_physical_jobs() -> None:
    manifest = rq1_search_manifest()
    random_jobs = manifest.jobs_for_initialization("random")
    content_jobs = manifest.jobs_for_initialization("content_pca")

    assert len(manifest.coordinates) == 16
    assert len(manifest.jobs) == 32
    assert len(manifest.new_physical_jobs) == 22
    assert len({job.physical_job_id for job in manifest.jobs}) == 32
    assert all(
        manifest.jobs[2 * trial].coordinate
        == manifest.jobs[2 * trial + 1].coordinate
        for trial in range(16)
    )
    assert [job.coordinate for job in random_jobs] == [
        job.coordinate for job in content_jobs
    ]
    assert [job.physical_job_id for job in random_jobs[:6]] == [
        f"treatment_tuning:item_learned_frozen_sid_event_trial_{trial:02d}"
        for trial in range(10, 16)
    ]
    assert [job.physical_job_id for job in random_jobs[6:10]] == [
        "lr_boundary:boundary_item_learned_frozen_sid_event_"
        f"embedding_learning_rate_{slot}"
        for slot in range(4)
    ]
    assert all(job.reused for job in random_jobs[:10])
    assert not any(job.reused for job in content_jobs)
    assert not any(job.reused for job in random_jobs[10:])
    assert all(
        EMBEDDING_LR_BOUNDS[0]
        <= job.coordinate.embedding_learning_rate
        <= EMBEDDING_LR_BOUNDS[1]
        and DEEP_LR_BOUNDS[0]
        <= job.coordinate.deep_learning_rate
        <= DEEP_LR_BOUNDS[1]
        for job in random_jobs[10:]
    )
    assert manifest.sha256 == rq1_search_manifest().sha256


def test_rq1_manifest_fixes_the_sobol_coordinates() -> None:
    coordinates = rq1_search_manifest().coordinates[10:]

    assert [coordinate.identity for coordinate in coordinates] == pytest.approx(
        [
            (0.2510526895883004, 0.00021100283643412708),
            (0.0002392056409105946, 0.028080757402472973),
            (0.0017564987189965285, 0.0012522746964537435),
            (0.03418660345450424, 0.004665226263955906),
            (0.009648560986601922, 0.0032786872587464884),
            (0.0037384540174461565, 0.013659276622634593),
        ],
        rel=1e-15,
        abs=0.0,
    )


def test_rq1_search_builder_uses_manifest_physical_identity() -> None:
    manifest = rq1_search_manifest()
    job = manifest.jobs_for_initialization("content_pca")[10]
    experiment = build_rq1_search_experiment(job)

    assert experiment.run_name == job.physical_run_name
    assert experiment.embedding_learning_rate == (
        job.coordinate.embedding_learning_rate
    )
    assert experiment.deep_learning_rate == job.coordinate.deep_learning_rate
    assert experiment.sid_lookup_initialization == "content_pca"

    with pytest.raises(ValueError, match="carryover must not be rebuilt"):
        build_rq1_search_experiment(
            manifest.jobs_for_initialization("random")[0]
        )


def test_content_pca_uses_stable_component_signs() -> None:
    centroids = torch.tensor(
        [[2.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=torch.float64
    )

    scores = project_centroids_with_pca(centroids, 2)

    torch.testing.assert_close(
        scores,
        centroids,
        rtol=1e-12,
        atol=1e-12,
    )


def test_content_initialization_changes_only_base_code_rows_after_mup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random_experiment, random_model, random_rng = _model_with_rng(
        "random", monkeypatch
    )
    content_experiment, content_model, content_rng = _model_with_rng(
        "content_pca", monkeypatch
    )
    random_embedding = rq1_learned_sid_embedding(random_model)
    content_embedding = rq1_learned_sid_embedding(content_model)
    random_weight = random_embedding.embedding.weight.detach()
    content_weight = content_embedding.embedding.weight.detach()
    base_rows = torch.zeros(len(random_weight), dtype=torch.bool)

    for level in range(random_experiment.semantic_codebooks.num_levels):
        first, last = random_experiment.semantic_codes.vocabulary.level_range(level)
        base_rows[first:last] = True
        assert not torch.equal(random_weight[first:last], content_weight[first:last])
        assert torch.allclose(
            random_weight[first:last].square().mean().sqrt(),
            content_weight[first:last].square().mean().sqrt(),
            rtol=1e-6,
            atol=1e-8,
        )

    assert torch.equal(random_weight[~base_rows], content_weight[~base_rows])
    assert content_embedding.embedding.weight.requires_grad
    random_lookup_parameter = id(random_embedding.embedding.weight)
    content_lookup_parameter = id(content_embedding.embedding.weight)
    random_other = {
        name: parameter
        for name, parameter in random_model.named_parameters()
        if id(parameter) != random_lookup_parameter
    }
    content_other = {
        name: parameter
        for name, parameter in content_model.named_parameters()
        if id(parameter) != content_lookup_parameter
    }
    assert random_other.keys() == content_other.keys()
    assert all(
        torch.equal(random_other[name], content_other[name]) for name in random_other
    )
    assert torch.equal(random_rng, content_rng)
    random_diagnostics = random_experiment.sid_initialization_diagnostics
    content_diagnostics = content_experiment.sid_initialization_diagnostics
    assert random_diagnostics["base_rows_after_sha256"] == content_diagnostics[
        "base_rows_before_sha256"
    ]
    assert random_diagnostics["non_base_rows_sha256"] == content_diagnostics[
        "non_base_rows_sha256"
    ]
    assert content_diagnostics["rng_nonadvancing"] is True
    assert content_experiment.generation_architecture_metadata()[
        "sid_lookup_initialization"
    ] == "content_pca"


def test_content_initialization_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_experiment, first_model, first_rng = _model_with_rng(
        "content_pca", monkeypatch
    )
    second_experiment, second_model, second_rng = _model_with_rng(
        "content_pca", monkeypatch
    )

    assert torch.equal(
        rq1_learned_sid_embedding(first_model).embedding.weight,
        rq1_learned_sid_embedding(second_model).embedding.weight,
    )
    assert first_experiment.sid_initialization_diagnostics == (
        second_experiment.sid_initialization_diagnostics
    )
    assert torch.equal(first_rng, second_rng)


def test_content_initialization_rejects_a_projection_wider_than_centroids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "0")
    codes, codebooks = _semantic_artifacts()
    monkeypatch.setattr(
        Rq1InitializationExperiment, "semantic_codes", property(lambda _: codes)
    )
    monkeypatch.setattr(
        Rq1InitializationExperiment,
        "semantic_codebooks",
        property(lambda _: codebooks),
    )
    experiment = _small_experiment("content_pca", representation_width=5)

    with pytest.raises(ValueError, match="cannot project 4-dimensional centroids to 5"):
        _ = experiment.base_model
