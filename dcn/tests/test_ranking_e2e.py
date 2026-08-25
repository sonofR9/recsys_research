"""End-to-end runs of the ranking variants on a miniature yambda layout."""

from pathlib import Path
from dataclasses import replace

import pytest

import dcn.config.ranking as ranking_config
from dcn.config import (
    HomeworkRankingExperiment,
    RankingExperiment,
    RankingWithHistoryExperiment,
    SemanticRankingExperiment,
)
from dcn.config.settings import EmbeddingConfig, TRANSFORMER
from dcn.main import run_experiment
from dcn.tests.miniature_yambda import configure, semantic_overrides
from neuralrec.utils import EXTRA_METRICS


def _configured(experiment_class, base_path: Path, **overrides):
    return configure(
        experiment_class,
        base_path,
        embedding=EmbeddingConfig(num_embeddings=16, dim=4, sparse=False),
        **overrides,
    )


@pytest.fixture(autouse=True)
def _small_history_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ranking_config,
        "TRANSFORMER",
        replace(
            TRANSFORMER,
            dim=4,
            num_layers=1,
            nhead=2,
            num_kv_heads=1,
            ffn_intermediate_dim=8,
        ),
    )


@pytest.mark.parametrize(
    "experiment_class, overrides",
    [
        (RankingExperiment, {}),
        (RankingWithHistoryExperiment, {}),
        (SemanticRankingExperiment, semantic_overrides()),
        (
            SemanticRankingExperiment,
            semantic_overrides(quantizer="rqvae", num_epochs=1),
        ),
        (HomeworkRankingExperiment, {}),
    ],
    ids=["plain", "history", "semantic_kmeans", "semantic_rqvae", "homework"],
)
@pytest.mark.training_e2e
def test_ranking_variant_trains(
    experiment_class,
    overrides: dict,
    base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")

    run_experiment(_configured(experiment_class, base_path, **overrides))


@pytest.mark.parametrize(
    "experiment_class, heads",
    [
        (RankingExperiment, {"like", "listen"}),
        (HomeworkRankingExperiment, {"like", "full_play"}),
    ],
    ids=["plain", "homework"],
)
def test_a_ranking_run_scores_both_heads_pairwise(
    experiment_class, heads: set[str], base_path: Path
) -> None:
    experiment = _configured(experiment_class, base_path)
    experiment.setup()
    train_days, val_days = experiment.train_and_validation_days
    (callback,) = experiment.extra_callbacks(train_days, val_days)
    state: dict = {}

    callback.on_epoch_end(state)

    metrics = state[EXTRA_METRICS]["epoch/val_pairwise"]
    assert set(metrics) == heads
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_the_history_variant_actually_adds_a_history_encoder(base_path: Path) -> None:
    plain = _configured(RankingExperiment, base_path)
    with_history = _configured(RankingWithHistoryExperiment, base_path)

    plain.setup()
    assert plain.base_model.history_encoder is None
    assert with_history.base_model.history_encoder is not None


def test_the_semantic_variant_widens_the_event_token(base_path: Path) -> None:
    with_history = _configured(RankingWithHistoryExperiment, base_path)
    semantic = _configured(SemanticRankingExperiment, base_path, **semantic_overrides())

    semantic.setup()
    semantic.semantic_stage.run()
    assert (
        len(semantic.base_model.feature_encoders)
        == len(with_history.base_model.feature_encoders) + 1
    )
