"""End-to-end runs of the generation variants on a miniature yambda layout."""

import json
import math
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
import torch
import mup

from dcn.config import (
    ActionGenerationExperiment,
    CombinedSemanticGenerationExperiment,
    GenerationExperiment,
    RqVaeGenerationExperiment,
    SemanticGenerationExperiment,
    TigerExperiment,
    TimeWindowGenerationExperiment,
    MuTransferGenerationExperiment,
)
from dcn.config.sequence import SequenceExperiment
from dcn.config.settings import LrScheduleConfig, TRANSFORMER
from dcn.training import OPTIMIZER_GROUP_ID
from dcn.main import run_experiment
from dcn.tests.miniature_yambda import SECONDS_IN_DAY, configure, semantic_overrides
from dcn.tests.helpers import packed_batch
from neuralrec.run.train import TrainRunner
from utils.global_config import config as global_config


def _configured(experiment_class, base_path: Path, **overrides):
    return configure(
        experiment_class,
        base_path,
        **{"transformer": _small_transformer(), **overrides},
    )


def _small_transformer(dim: int = 4):
    return replace(
        TRANSFORMER,
        dim=dim,
        num_layers=1,
        nhead=2,
        num_kv_heads=1,
        ffn_intermediate_dim=2 * dim,
    )


def _semantic(**overrides) -> dict:
    return {**semantic_overrides(**overrides), "beam_width": 2}


def _sampled_softmax() -> dict:
    return {"num_in_batch_negatives": 4}


_VARIANTS = [
    (GenerationExperiment, {**_sampled_softmax(), "bos": True}),
    (
        TimeWindowGenerationExperiment,
        {**_sampled_softmax(), "window_seconds": SECONDS_IN_DAY},
    ),
    (ActionGenerationExperiment, _sampled_softmax()),
    (SemanticGenerationExperiment, _semantic()),
    (TigerExperiment, _semantic()),
    (CombinedSemanticGenerationExperiment, _semantic()),
    (RqVaeGenerationExperiment, _semantic(quantizer="rqvae", num_epochs=1)),
]


@pytest.mark.parametrize(
    "experiment_class, overrides",
    _VARIANTS,
    ids=[
        "next_like_bos",
        "likes_in_24h",
        "likes_and_listens",
        "semantic",
        "tiger",
        "semantic_combined",
        "semantic_rqvae",
    ],
)
@pytest.mark.training_e2e
def test_generation_variant_trains(
    experiment_class,
    overrides: dict,
    base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")

    run_experiment(_configured(experiment_class, base_path, **overrides))


@pytest.mark.training_e2e
def test_the_likes_variant_reports_the_full_catalog_metrics(
    base_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    experiment = _configured(GenerationExperiment, base_path, **_sampled_softmax())

    run_experiment(experiment)

    report = json.loads(
        (base_path / "logs" / experiment.run_name / "final_metrics.json").read_text()
    )
    assert report["num_users"] > 0
    for name in ("ndcg@10", "recall@100", "capped_recall@100", "mrr@10", "coverage@10"):
        assert 0.0 <= report[name] <= 1.0

    metadata = json.loads(
        (base_path / "logs" / experiment.run_name / "training_metadata.json").read_text()
    )
    assert metadata["targets_per_epoch"] > 0
    assert metadata["training_horizon"] == (
        metadata["targets_per_epoch"] * experiment.num_epochs
    )
    assert metadata["physical_batch_size"] == experiment.dataloader.batch_size
    assert (
        metadata["gradient_accumulation_steps"]
        == experiment.dataloader.gradient_accumulation_steps
    )
    assert metadata["effective_batch_size"] == (
        experiment.dataloader.effective_batch_size
    )
    assert metadata["optimizer_steps"] == (
        math.ceil(
            len(experiment.sequence_train_loader)
            / experiment.dataloader.gradient_accumulation_steps
        )
        * experiment.num_epochs
    )
    assert metadata["validation_loss"] > 0
    assert metadata["tokens_per_epoch"] >= metadata["targets_per_epoch"]
    assert metadata["token_horizon"] == (
        metadata["tokens_per_epoch"] * experiment.num_epochs
    )
    assert metadata["tokens_seen"] == metadata["token_horizon"]


@pytest.mark.training_e2e
def test_queued_generation_prewarms_before_device_activation(
    base_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "0")
    monkeypatch.setenv("DCN_GPU_LOCK_DEVICE", "GPU-test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    experiment = _configured(GenerationExperiment, base_path, **_sampled_softmax())
    activate = experiment.activate_runner_device

    def activate_after_warmup(runner: TrainRunner) -> None:
        assert experiment.callbacks.validation._prepared_batches is not None
        assert experiment.true_metric._prepared_query_batches is not None
        assert not experiment.sequence_val_loader.pin_memory
        assert not experiment.sequence_train_loader.pin_memory
        activate(runner)

    monkeypatch.setattr(experiment, "activate_runner_device", activate_after_warmup)

    run_experiment(experiment)


def test_queued_generation_transfers_prepared_iterator_to_trainer(
    base_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    experiment = _configured(GenerationExperiment, base_path, **_sampled_softmax())
    experiment.setup()
    experiment.prebuild_runner_components()
    prepared_iterator = experiment._prepared_train_iterator

    trainer = experiment.create_runner()

    assert trainer._prepared_train_iterator is prepared_iterator
    assert experiment._prepared_train_iterator is None


@pytest.mark.training_e2e
def test_mu_transfer_variant_trains_with_width_aware_parameters(
    base_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    experiment = _configured(
        MuTransferGenerationExperiment,
        base_path,
        **_sampled_softmax(),
        transformer=replace(
            _small_transformer(dim=8), nhead=4, num_kv_heads=2
        ),
        mup_base_dim=4,
        mup_delta_dim=8,
    )

    run_experiment(experiment)

    assert experiment.base_model.query_multiplier == 1.0
    assert experiment.base_model.item_embedding.embedding_dim == 64
    assert experiment.base_model.tokenizer.out_dim == 8
    assert isinstance(experiment.base_model.query_projection, mup.MuReadout)
    assert experiment.base_model.item_embedding.weight.infshape.ninf() == 0
    assert all(
        layer.softmax_scale == pytest.approx(0.5)
        for layer in experiment.base_model.sequence_model.layers
    )
    assert all(
        hasattr(parameter, "infshape")
        for parameter in experiment.base_model.parameters()
    )


def test_mu_transfer_widths_must_fit_the_attention_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MuTransferGenerationExperiment(mup_base_dim=3)


def test_mu_transfer_keeps_projection_shapes_when_target_matches_item_width(
    base_path: Path,
) -> None:
    experiment = _configured(
        MuTransferGenerationExperiment,
        base_path,
        **_sampled_softmax(),
        item_embedding_dim=8,
        mup_base_dim=4,
        mup_delta_dim=16,
    )
    experiment.setup()

    model = experiment.base_model

    assert model.tokenizer.projection is not None
    assert model.query_projection is not None
    assert all(hasattr(parameter, "infshape") for parameter in model.parameters())


def _ffn_width_multipliers(
    base_path: Path, ffn_intermediate_dim: int, **mup_overrides: int
) -> dict[str, float]:
    experiment = _configured(
        MuTransferGenerationExperiment,
        base_path,
        **_sampled_softmax(),
        transformer=replace(
            _small_transformer(dim=8), ffn_intermediate_dim=ffn_intermediate_dim
        ),
        mup_base_dim=4,
        mup_delta_dim=8,
        **mup_overrides,
    )
    experiment.setup()
    return {
        name: parameter.infshape.width_mult()
        for name, parameter in experiment.base_model.named_parameters()
        if name.endswith(("ffn.w1.weight", "ffn.w3.weight"))
    }


def test_mu_transfer_scales_the_ffn_output_projection_with_its_own_width(
    base_path: Path,
) -> None:
    narrow = _ffn_width_multipliers(
        base_path, 8, mup_base_ffn_dim=8, mup_delta_ffn_dim=16
    )
    wide = _ffn_width_multipliers(
        base_path, 32, mup_base_ffn_dim=8, mup_delta_ffn_dim=16
    )

    output_projection = next(name for name in narrow if name.endswith("w3.weight"))
    input_projection = next(name for name in narrow if name.endswith("w1.weight"))
    assert wide[output_projection] == 4 * narrow[output_projection]
    assert wide[input_projection] == narrow[input_projection]


def test_mu_transfer_learning_rate_falls_with_the_ffn_width(base_path: Path) -> None:
    def output_projection_rate(ffn_intermediate_dim: int) -> float:
        experiment = _configured(
            MuTransferGenerationExperiment,
            base_path,
            **_sampled_softmax(),
            transformer=replace(
                _small_transformer(dim=8), ffn_intermediate_dim=ffn_intermediate_dim
            ),
            mup_base_dim=4,
            mup_delta_dim=8,
            mup_base_ffn_dim=8,
            mup_delta_ffn_dim=16,
            deep_learning_rate=0.012,
        )
        experiment.setup()
        output_projection = next(
            parameter
            for name, parameter in experiment.base_model.named_parameters()
            if name.endswith("ffn.w3.weight")
        )
        optimizer = experiment.create_optimizers()
        return next(
            group["lr"]
            for group in optimizer.param_groups
            if any(parameter is output_projection for parameter in group["params"])
        )

    assert output_projection_rate(8) == pytest.approx(4 * output_projection_rate(32))


@pytest.mark.training_e2e
def test_mu_transfer_deep_only_schedule_trains_with_multiple_deep_groups(
    base_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    experiment = _configured(
        MuTransferGenerationExperiment,
        base_path,
        **_sampled_softmax(),
        transformer=_small_transformer(dim=8),
        mup_base_dim=4,
        mup_delta_dim=8,
        num_epochs=1,
        lr_schedule=LrScheduleConfig(
            "linear", optimizer_group_scope="deep_only"
        ),
        lr_schedule_horizon_epochs=1,
        adaptive_schedule_early_stopping=True,
        early_stopping_patience=3,
    )
    experiment.setup()
    optimizer = experiment.create_optimizers()

    assert [group["schedule_group"] for group in optimizer.param_groups].count(
        "deep"
    ) > 1
    assert len({group[OPTIMIZER_GROUP_ID] for group in optimizer.param_groups}) == len(
        optimizer.param_groups
    )

    run_experiment(experiment)

    metadata = json.loads(
        (base_path / "logs" / experiment.run_name / "training_metadata.json").read_text()
    )
    assert set(metadata["lr_group_traces"]) == {"embedding", "deep"}
    assert {len(trace) for trace in metadata["lr_group_traces"].values()} == {1}
    assert metadata["lr_group_traces"]["embedding"] == pytest.approx(
        [experiment.embedding_learning_rate]
    )
    assert metadata["lr_group_traces"]["deep"] == pytest.approx(
        [experiment.deep_learning_rate]
    )


def test_mu_transfer_without_an_ffn_base_reads_the_ffn_width_off_the_ratio(
    base_path: Path,
) -> None:
    narrow = _ffn_width_multipliers(base_path, 8)
    wide = _ffn_width_multipliers(base_path, 32)

    assert wide == narrow


def test_mu_transfer_ffn_base_needs_its_delta(base_path: Path) -> None:
    with pytest.raises(ValueError, match="mup_delta_ffn_dim"):
        _ffn_width_multipliers(base_path, 8, mup_base_ffn_dim=8)


def test_mu_transfer_ffn_base_and_delta_must_differ(base_path: Path) -> None:
    with pytest.raises(ValueError, match="differ"):
        _ffn_width_multipliers(
            base_path, 8, mup_base_ffn_dim=8, mup_delta_ffn_dim=8
        )


def test_mu_transfer_keeps_normalization_scales_at_one(base_path: Path) -> None:
    experiment = _configured(
        MuTransferGenerationExperiment,
        base_path,
        **_sampled_softmax(),
        initializer_std=0.02,
        mup_base_dim=4,
        mup_delta_dim=8,
    )
    experiment.setup()

    norm_weights = [
        parameter
        for name, parameter in experiment.base_model.named_parameters()
        if "norm" in name and name.endswith("weight")
    ]

    assert norm_weights
    assert all(torch.equal(weight, torch.ones_like(weight)) for weight in norm_weights)


def test_unknown_negative_sampling_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative_sampling"):
        GenerationExperiment(negative_sampling="typo")  # type: ignore[arg-type]


def test_random_offline_logq_corrects_by_expected_negative_count(
    base_path: Path,
) -> None:
    experiment = _configured(
        GenerationExperiment,
        base_path,
        **_sampled_softmax(),
        negative_sampling="random_offline_logq",
    )
    experiment.setup()

    criterion = experiment.create_criterion().loss

    assert criterion.q.sum().item() == pytest.approx(4.0)
    assert criterion.random_negatives.probabilities.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["random", "in_batch_no_logq"])
def test_uncorrected_negative_modes_disable_logq(
    base_path: Path, mode: str
) -> None:
    experiment = _configured(
        GenerationExperiment,
        base_path,
        **_sampled_softmax(),
        negative_sampling=mode,
    )
    experiment.setup()

    assert experiment.create_criterion().loss.correction == "none"


@pytest.mark.parametrize("mode", ["mixed_online_logq", "mixed_offline_logq"])
def test_mixed_negatives_split_the_budget_and_correct_only_in_batch(
    base_path: Path, mode: str
) -> None:
    experiment = _configured(
        GenerationExperiment,
        base_path,
        **_sampled_softmax(),
        negative_sampling=mode,
    )
    experiment.setup()

    criterion = experiment.create_criterion().loss
    assert criterion.num_in_batch_negatives == 2
    assert criterion.random_negatives.num_negatives == 2
    assert criterion.correction == "yi2019"
    assert not criterion.correct_random_negatives


@pytest.mark.parametrize(
    "fraction,expected_random",
    [(0.25, 2), (0.75, 6)],
)
def test_mixed_negative_fraction_controls_the_budget_split(
    base_path: Path, fraction: float, expected_random: int
) -> None:
    experiment = _configured(
        GenerationExperiment,
        base_path,
        num_in_batch_negatives=8,
        negative_sampling="mixed_online_logq",
        random_negative_fraction=fraction,
    )
    experiment.setup()

    criterion = experiment.create_criterion().loss

    assert criterion.random_negatives.num_negatives == expected_random
    assert criterion.num_in_batch_negatives == 8 - expected_random


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_mixed_negative_fraction_must_leave_both_sources(
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match="random_negative_fraction"):
        GenerationExperiment(random_negative_fraction=fraction)


def test_mixed_negative_count_must_leave_both_sources(base_path: Path) -> None:
    experiment = _configured(
        GenerationExperiment,
        base_path,
        num_in_batch_negatives=1,
        negative_sampling="mixed_online_logq",
        random_negative_fraction=0.25,
    )
    experiment.setup()

    with pytest.raises(ValueError, match="at least one negative"):
        experiment.create_criterion()


@pytest.mark.training_e2e
def test_the_best_epoch_is_picked_on_a_metric_the_eval_reports(
    base_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selection rule names a metric by string, so only a run proves it
    lands where the epoch is chosen by."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    # The miniature config replaces the checkpointing group wholesale, which
    # would take the variant's own selection rule with it.
    experiment = _configured(
        GenerationExperiment,
        base_path,
        **_sampled_softmax(),
        checkpointing=GenerationExperiment().checkpointing,
    )

    run_experiment(experiment)

    assert experiment.callbacks.best_weights.best_epoch is not None


def test_a_selection_metric_the_eval_never_computes_is_refused() -> None:
    with pytest.raises(ValueError, match="eval_ks"):
        GenerationExperiment(eval_ks=(10, 20))


def test_only_the_likes_variant_drops_listens(base_path: Path) -> None:
    likes = _configured(GenerationExperiment, base_path, **_sampled_softmax())
    actions = _configured(ActionGenerationExperiment, base_path, **_sampled_softmax())
    likes.setup()

    def event_count(experiment) -> int:
        loader = experiment.make_sequence_loader(
            experiment.train_and_validation_days[0],
            split="train",
            batch_size=2,
            shuffle=False,
        )
        return sum(int(batch["cumulative_lens"][-1]) for batch in loader)

    assert 0 < event_count(likes) < event_count(actions)


def test_second_exact_validation_interval_splits_the_boundary_day(
    base_path: Path,
) -> None:
    experiment = _configured(
        GenerationExperiment,
        base_path,
        **_sampled_softmax(),
        validation_interval_seconds=86_400,
    )
    experiment.setup()

    train_days, val_days = experiment.train_and_validation_days
    boundary_day = (set(train_days) & set(val_days)).pop()
    frame = pl.read_parquet(experiment.dataset_manager.day_to_path[boundary_day])
    train = frame.filter(experiment.row_filter_for_split("train"))
    validation = frame.filter(experiment.row_filter_for_split("validation"))

    assert train["timestamp"].max() < experiment.validation_cutoff_timestamp
    assert validation["timestamp"].min() >= experiment.validation_cutoff_timestamp


def test_cls_metric_queries_train_history_and_scores_validation_relevance(
    base_path: Path,
) -> None:
    experiment = _configured(
        GenerationExperiment,
        base_path,
        **_sampled_softmax(),
        cls_token=True,
        window="next_item",
    )
    experiment.setup()
    callback = experiment.true_metric
    train_days, validation_days = experiment.train_and_validation_days

    query_items: dict[int, set[int]] = {}
    for batch in callback.query_loader:
        lengths = batch["cumulative_lens"].diff().tolist()
        users = batch["int_columns"][experiment.user_column].dense().split(lengths)
        items = batch["int_columns"][experiment.item_id_column].dense().split(lengths)
        for user_values, item_values in zip(users, items):
            query_items[int(user_values[0])] = set(item_values.tolist())

    assert query_items == experiment._interactions(train_days, split="train")
    assert callback.relevance == experiment._interactions(
        validation_days, split="validation"
    )
    assert set(train_days).isdisjoint(validation_days)


def test_interleaved_cls_counts_every_item_and_query_token(base_path: Path) -> None:
    experiment = _configured(
        GenerationExperiment,
        base_path,
        **_sampled_softmax(),
        cls_token_mode="interleaved",
        window="next_item",
    )
    experiment.setup()

    event_count = sum(
        len(sample["int_columns"][experiment.item_id_column])
        for sample in experiment.sequence_train_loader.dataset
    )

    assert experiment.training_tokens_per_epoch == 2 * event_count
    assert experiment.training_targets_per_epoch == event_count - len(
        experiment.sequence_train_loader.dataset
    )


def test_interleaved_cls_model_accepts_the_configured_event_horizon(
    base_path: Path,
    cpu_attention: None,
) -> None:
    experiment = _configured(
        GenerationExperiment,
        base_path,
        **_sampled_softmax(),
        cls_token_mode="interleaved",
        window="next_item",
    )
    experiment.setup()
    global_config.set_cpu_attention(True)
    model = experiment._create_model().eval()
    batch = packed_batch([1, 2, 3, 4, 1], [5])

    assert model(batch)["query_repr"].shape[0] == 10


def test_legacy_cls_flag_cannot_conflict_with_interleaved_mode() -> None:
    with pytest.raises(ValueError, match="conflicts with interleaved"):
        GenerationExperiment(cls_token=True, cls_token_mode="interleaved")


def test_interleaved_cls_rejects_bos_target_semantics() -> None:
    with pytest.raises(ValueError, match="does not support BOS"):
        GenerationExperiment(
            bos=True,
            cls_token_mode="interleaved",
            window="next_item",
        )


def test_homework_evaluation_ranks_the_full_split_catalog_without_seen_mask(
    base_path: Path,
) -> None:
    experiment = _configured(
        GenerationExperiment,
        base_path,
        **_sampled_softmax(),
        validation_interval_seconds=SECONDS_IN_DAY,
        evaluation_catalog="all",
        exclude_seen_from_evaluation=False,
    )
    experiment.setup()

    callback = experiment.true_metric
    item_ids = callback.item_batch["int_columns"][experiment.item_id_column].dense()
    train_days, val_days = experiment.train_and_validation_days
    expected = set().union(
        *experiment._interactions(sorted(set(train_days + val_days))).values()
    )

    assert set(item_ids.tolist()) == expected
    assert not callback.exclude_seen


def test_homework_initializer_also_resets_normalization_scales(
    base_path: Path,
) -> None:
    experiment = _configured(
        GenerationExperiment,
        base_path,
        **_sampled_softmax(),
        initializer_std=0.02,
    )
    experiment.setup()

    norm = next(
        module
        for module in experiment.base_model.modules()
        if isinstance(module, torch.nn.LayerNorm)
    )

    assert not torch.equal(norm.weight, torch.ones_like(norm.weight))
    assert norm.weight.abs().max() <= 0.04


def test_the_action_variant_puts_two_tokens_on_every_event(base_path: Path) -> None:
    actions = _configured(ActionGenerationExperiment, base_path, **_sampled_softmax())
    actions.setup()

    assert actions.base_model.tokenizer.tokens_per_event == 2


def test_no_generation_variant_falls_back_to_the_framework_epoch_count() -> None:
    """These sit on two bases, and only one of them sets the epoch count.

    Dataclass fields resolve in reverse MRO, so the base that merely inherits
    the framework default silently wins unless every variant states its own.
    """
    epochs = {variant().num_epochs for variant, _ in _VARIANTS}

    assert len(epochs) == 1
    assert epochs != {SequenceExperiment.num_epochs}
