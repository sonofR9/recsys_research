import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from dcn.config import SemanticExperiment, SemanticHistoryExperiment, SemanticIdConfig
from dcn.config.semantic import KMeansIdStage
from dcn.semantic import ResidualCodebooks, SemanticCodes
from dcn.tests.helpers import packed_batch


def _artifacts(
    policy: str,
) -> tuple[SemanticCodes, ResidualCodebooks]:
    item_ids = torch.tensor([1, 2, 3, 4])
    base_codes = torch.tensor([[0, 0], [0, 0], [0, 1], [1, 0]])
    codes = (
        SemanticCodes.with_collision_suffix(item_ids, base_codes, num_codes=2)
        if policy == "suffix"
        else SemanticCodes.without_collision_suffix(item_ids, base_codes, num_codes=2)
    )
    codebooks = ResidualCodebooks(torch.randn(2, 2, 3))
    return codes, codebooks


def _experiment(
    policy: str,
    artifacts: tuple[SemanticCodes, ResidualCodebooks],
) -> SemanticHistoryExperiment:
    codes, codebooks = artifacts
    experiment = SemanticHistoryExperiment(
        history_representation="item_frozen_sid_event",
        representation_width=8,
        semantic=SemanticIdConfig(
            num_levels=2,
            num_codes=2,
            collision_policy=policy,
        ),
    )
    experiment.__dict__["item_embeddings"] = SimpleNamespace(num_known_ids=4)
    experiment.__dict__["artifacts"] = SimpleNamespace(item_id_column="compact_item_id")
    experiment.__dict__["semantic_codes"] = codes
    experiment.__dict__["semantic_codebooks"] = codebooks
    return experiment


def test_collision_policies_share_base_codebook_cache_but_not_assignments(
    tmp_path: Path,
) -> None:
    embeddings = tmp_path / "embeddings.parquet"
    pl.DataFrame(
        {
            "compact_id": [1, 2, 3, 4],
            "normalized_embed": [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
        }
    ).write_parquet(embeddings)
    common = {
        "num_levels": 1,
        "num_codes": 2,
        "kmeans_iterations": 2,
        "seed": 42,
    }
    suffix = SemanticHistoryExperiment(
        semantic=SemanticIdConfig(**common, collision_policy="suffix")
    )
    none = SemanticHistoryExperiment(
        semantic=SemanticIdConfig(**common, collision_policy="none")
    )
    for experiment in (suffix, none):
        experiment.__dict__["dataset_cache_dir"] = tmp_path / "cache"
        experiment.__dict__["artifacts"] = SimpleNamespace(
            item_id_column="compact_id",
            precomputed_embeddings={"compact_id": embeddings},
        )
        experiment.device = torch.device("cpu")

    assert suffix.semantic_stage.codebooks_path == none.semantic_stage.codebooks_path
    assert (
        suffix.semantic_stage.materialization_lock_path
        == none.semantic_stage.materialization_lock_path
    )
    assert suffix.semantic_stage.codes_path != none.semantic_stage.codes_path

    suffix.semantic_stage.run()
    none.semantic_stage.run()

    suffix_codebooks = ResidualCodebooks.load(suffix.semantic_stage.codebooks_path)
    none_codebooks = ResidualCodebooks.load(none.semantic_stage.codebooks_path)
    suffix_codes = SemanticCodes.load(suffix.semantic_stage.codes_path)
    none_codes = SemanticCodes.load(none.semantic_stage.codes_path)

    assert torch.equal(suffix_codebooks.centroids, none_codebooks.centroids)
    assert torch.equal(suffix_codes.codes[:, :-1], none_codes.codes)
    assert suffix_codes.num_levels == none_codes.num_levels + 1
    assert not suffix.semantic_stage.materialization_marker_path.exists()
    assert not none.semantic_stage.materialization_marker_path.exists()


def test_suffix_policy_preserves_the_existing_semantic_cache_identity() -> None:
    suffix = SemanticIdConfig(num_levels=3, num_codes=512)
    none = SemanticIdConfig(
        num_levels=3,
        num_codes=512,
        collision_policy="none",
    )

    assert suffix.cache_key == "kmeans_3x512_75bd085d12"
    assert suffix.base_cache_key == none.base_cache_key
    assert suffix.cache_key != none.cache_key


def test_convergent_kmeans_uses_an_isolated_cache_identity() -> None:
    legacy = SemanticIdConfig(
        num_levels=3,
        num_codes=512,
        kmeans_iterations=300,
    )
    convergent = SemanticIdConfig(
        num_levels=3,
        num_codes=512,
        kmeans_iterations=300,
        kmeans_relative_inertia_tolerance=1e-4,
        kmeans_assignment_early_stopping=True,
    )

    assert legacy.base_cache_key != convergent.base_cache_key
    assert convergent.convergence_enabled


def test_convergent_kmeans_writes_authenticated_fit_diagnostics(
    tmp_path: Path,
) -> None:
    embeddings = tmp_path / "embeddings.parquet"
    pl.DataFrame(
        {
            "compact_id": list(range(1, 13)),
            "normalized_embed": [
                [float(index // 4), float(index % 4)] for index in range(12)
            ],
        }
    ).write_parquet(embeddings)
    experiment = SemanticHistoryExperiment(
        semantic=SemanticIdConfig(
            num_levels=2,
            num_codes=3,
            kmeans_iterations=300,
            kmeans_relative_inertia_tolerance=1e-4,
            kmeans_assignment_early_stopping=True,
        )
    )
    experiment.__dict__["dataset_cache_dir"] = tmp_path / "cache"
    experiment.__dict__["artifacts"] = SimpleNamespace(
        item_id_column="compact_id",
        precomputed_embeddings={"compact_id": embeddings},
    )
    experiment.device = torch.device("cpu")

    experiment.semantic_stage.run()
    document = experiment.semantic_diagnostics_document()

    convergence = document["kmeans_convergence"]
    assert convergence["schema"] == "residual-kmeans-convergence/v1"
    assert convergence["fitter_revision"] == "convergent-lloyd-v1"
    assert convergence["max_iterations"] == 300
    assert convergence["relative_inertia_tolerance"] == 1e-4
    assert convergence["assignment_early_stopping"] is True
    assert len(convergence["levels"]) == 2
    assert len(document["kmeans_convergence_sha256"]) == 64

    marker_path = experiment.semantic_stage.materialization_marker_path
    marker = json.loads(marker_path.read_text())
    fit_marker_path = experiment.semantic_stage.fit_materialization_marker_path
    fit_marker = json.loads(fit_marker_path.read_text())
    fit_artifact_paths = {
        "codebooks": experiment.semantic_stage.codebooks_path,
        "base_codes": experiment.semantic_stage.base_codes_path,
        "fit_diagnostics": experiment.semantic_stage.fit_diagnostics_path,
    }
    assert fit_marker == {
        "schema": "residual-kmeans-fit-materialization/v1",
        "fitter_revision": "convergent-lloyd-v1",
        "materialization_revision": "shared-base-v2",
        "semantic_base_cache_key": experiment.semantic.base_cache_key,
        "artifact_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in fit_artifact_paths.items()
        },
    }
    assert marker == {
        "schema": "residual-kmeans-collision-materialization/v1",
        "fitter_revision": "convergent-lloyd-v1",
        "materialization_revision": "shared-base-v2",
        "semantic_base_cache_key": experiment.semantic.base_cache_key,
        "semantic_cache_key": experiment.semantic.cache_key,
        "fit_materialization_sha256": hashlib.sha256(
            fit_marker_path.read_bytes()
        ).hexdigest(),
        "codes_sha256": hashlib.sha256(
            experiment.semantic_stage.codes_path.read_bytes()
        ).hexdigest(),
    }
    assert experiment.semantic_stage.cache_complete
    assert not list(marker_path.parent.glob(".*.tmp"))

    experiment.semantic_stage.fit_diagnostics_path.write_text("{}")
    assert not experiment.semantic_stage.cache_complete
    with pytest.raises(RuntimeError, match="incomplete or corrupted"):
        experiment.semantic_stage.run()


def test_convergent_collision_policies_reuse_one_authenticated_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings = tmp_path / "embeddings.parquet"
    pl.DataFrame(
        {
            "compact_id": list(range(1, 13)),
            "normalized_embed": [
                [float(index // 4), float(index % 4)] for index in range(12)
            ],
        }
    ).write_parquet(embeddings)
    common = {
        "num_levels": 2,
        "num_codes": 3,
        "kmeans_iterations": 300,
        "kmeans_relative_inertia_tolerance": 1e-4,
        "kmeans_assignment_early_stopping": True,
    }
    suffix = SemanticHistoryExperiment(
        semantic=SemanticIdConfig(**common, collision_policy="suffix")
    )
    none = SemanticHistoryExperiment(
        semantic=SemanticIdConfig(**common, collision_policy="none")
    )
    for experiment in (suffix, none):
        experiment.__dict__["dataset_cache_dir"] = tmp_path / "cache"
        experiment.__dict__["artifacts"] = SimpleNamespace(
            item_id_column="compact_id",
            precomputed_embeddings={"compact_id": embeddings},
        )
        experiment.device = torch.device("cpu")

    suffix.semantic_stage.run()
    fit_marker = suffix.semantic_stage.fit_materialization_marker_path.read_bytes()
    monkeypatch.setattr(
        "dcn.config.semantic.fit_residual_kmeans_with_diagnostics",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("refitted")),
    )

    none.semantic_stage.run()

    suffix_codes = SemanticCodes.load(suffix.semantic_stage.codes_path)
    none_codes = SemanticCodes.load(none.semantic_stage.codes_path)
    assert torch.equal(suffix_codes.codes[:, :-1], none_codes.codes)
    assert (
        suffix.semantic_stage.fit_materialization_marker_path.read_bytes() == fit_marker
    )
    assert suffix.semantic_stage.cache_complete
    assert none.semantic_stage.cache_complete


def test_convergent_stage_reuses_fit_assignments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SemanticIdConfig(
        num_levels=2,
        num_codes=3,
        kmeans_iterations=300,
        kmeans_relative_inertia_tolerance=1e-4,
        kmeans_assignment_early_stopping=True,
    )
    stage = KMeansIdStage(
        embeddings_parquet=tmp_path / "unused.parquet",
        codes_path=tmp_path / "codes.pt",
        codebooks_path=tmp_path / "codebooks.pt",
        fit_diagnostics_path=tmp_path / "fit.json",
        config=config,
        device=torch.device("cpu"),
    )
    embeddings = torch.randn(30, 4, generator=torch.Generator().manual_seed(13))
    monkeypatch.setattr(
        ResidualCodebooks,
        "encode",
        lambda self, values: (_ for _ in ()).throw(AssertionError("re-encoded")),
    )

    codebooks, codes = stage.quantize(embeddings)

    assert codebooks.num_levels == 2
    assert codes.shape == (30, 2)


def test_collision_policy_does_not_change_concrete_item_output_semantics() -> None:
    suffix = _experiment("suffix", _artifacts("suffix"))
    none = _experiment("none", _artifacts("none"))
    item_ids = [1, 2, 3]
    batch = packed_batch(item_ids, [2, 1])

    suffix_tokens = suffix.create_tokenizer()(batch)
    none_tokens = none.create_tokenizer()(batch)

    assert suffix_tokens.item_ids.tolist() == item_ids
    assert none_tokens.item_ids.tolist() == item_ids
    assert suffix_tokens.is_target.tolist() == none_tokens.is_target.tolist()
    assert suffix.true_metric_options()["semantic_base_levels"] == 2
    assert none.true_metric_options()["semantic_base_levels"] == 2


@pytest.mark.parametrize(
    ("policy", "expected_suffix_symbols"),
    [("suffix", 3), ("none", 0)],
)
def test_semantic_diagnostics_record_collision_policy_and_suffix_width(
    policy: str,
    expected_suffix_symbols: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings_path = tmp_path / "embeddings.parquet"
    pl.DataFrame(
        {
            "compact_id": [1, 2, 3, 4],
            "normalized_embed": [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    ).write_parquet(embeddings_path)
    experiment = _experiment(policy, _artifacts(policy))
    experiment.base_path = str(tmp_path)
    experiment.__dict__["dataset_cache_dir"] = tmp_path / "cache"
    experiment.run_name = f"diagnostics_{policy}"
    experiment.__dict__["artifacts"] = SimpleNamespace(
        item_id_column="compact_id",
        precomputed_embeddings={"compact_id": embeddings_path},
    )
    (tmp_path / "logs" / experiment.run_name).mkdir(parents=True)
    monkeypatch.setattr(SemanticExperiment, "finish", lambda self, runner: None)

    experiment.finish(SimpleNamespace())

    diagnostics = json.loads(
        (
            tmp_path / "logs" / experiment.run_name / "semantic_id_diagnostics.json"
        ).read_text()
    )
    assert diagnostics["collision_policy"] == policy
    assert diagnostics["collision_suffix_symbols"] == expected_suffix_symbols


def test_semantic_diagnostics_reuse_the_base_tokenizer_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings_path = tmp_path / "embeddings.parquet"
    pl.DataFrame(
        {
            "compact_id": [1, 2, 3, 4],
            "normalized_embed": [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    ).write_parquet(embeddings_path)
    experiment = _experiment("suffix", _artifacts("suffix"))
    experiment.base_path = str(tmp_path)
    experiment.__dict__["dataset_cache_dir"] = tmp_path / "cache"
    experiment.__dict__["artifacts"] = SimpleNamespace(
        item_id_column="compact_id",
        precomputed_embeddings={"compact_id": embeddings_path},
    )
    monkeypatch.setattr(SemanticExperiment, "finish", lambda self, runner: None)
    (tmp_path / "logs" / experiment.run_name).mkdir(parents=True)
    experiment.finish(SimpleNamespace())
    first = json.loads(
        (
            tmp_path / "logs" / experiment.run_name / "semantic_id_diagnostics.json"
        ).read_text()
    )
    cache_path = experiment._semantic_base_dir / "semantic_id_diagnostics_v2.json"
    cached = json.loads(cache_path.read_text())
    cache_path.write_text(json.dumps(cached["diagnostics"]))

    monkeypatch.setattr(
        "dcn.config.semantic_history.semantic_id_diagnostics",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("recomputed")),
    )
    experiment.run_name = "cached_diagnostics"
    (tmp_path / "logs" / experiment.run_name).mkdir(parents=True)
    experiment.finish(SimpleNamespace())
    second = json.loads(
        (
            tmp_path / "logs" / experiment.run_name / "semantic_id_diagnostics.json"
        ).read_text()
    )

    assert second == first

    experiment.invalidate_cache = True
    with pytest.raises(AssertionError, match="recomputed"):
        experiment._cached_semantic_diagnostics(torch.ones(4, 3))
