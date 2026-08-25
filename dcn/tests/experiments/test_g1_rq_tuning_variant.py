from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from dcn.config import MuTransferGenerationExperiment


SCRIPT = (
    Path(__file__).parents[3]
    / "experiments"
    / "g1_sasrec_item_ids_likes"
    / "configs/rq_tuning_variant.py"
)


def _load(monkeypatch: pytest.MonkeyPatch, **environment: str):
    defaults = {
        "G1_DATASET_SIZE": "50m",
        "G1_TUNE_RUN": "test_ts2_r2",
        "G1_TUNE_RUN_REVISION": "2",
        "G1_TUNE_SOURCE_VARIANT": "neg_online_logq",
        "G1_TUNE_EXPERIMENT_FIELDS": "negative_sampling",
        "G1_TUNE_EMBEDDING_LR": "0.008",
        "G1_TUNE_DEEP_LR": "0.006",
    }
    for name, value in {**defaults, **environment}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("G1_VARIANT", "stale_invalid_value")
    return runpy.run_path(str(SCRIPT))["experiment"]


def test_negative_family_copies_only_requested_method_and_tuning_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TUNE_BATCH_SIZE="1024",
        G1_TUNE_NUM_NEGATIVES="256",
        G1_TUNE_LOGQ_ALPHA="0.02",
        G1_TUNE_CORRECT_POSITIVE_LOGQ="1",
    )

    assert experiment.run_name == "g1_rqtune_test_ts2_r2_50m"
    assert experiment.negative_sampling == "online_logq"
    assert experiment.dataloader.batch_size == 1024
    assert experiment.num_in_batch_negatives == 256
    assert experiment.logq_alpha == 0.02
    assert experiment.correct_positive_logq
    assert experiment.embedding_learning_rate == 0.008
    assert experiment.deep_learning_rate == 0.006
    assert experiment.transformer.ffn == "swiglu"
    assert type(experiment) is MuTransferGenerationExperiment
    assert experiment.item_embedding_dim == 64
    assert experiment.num_epochs == 20
    assert experiment.eval_every_n_epochs == 1
    assert experiment.early_stopping_patience == 3
    assert experiment.early_stopping_min_delta == 0
    assert experiment.restore_best_weights


def test_architecture_fields_copy_as_one_exact_treatment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TUNE_SOURCE_VARIANT="heads_4",
        G1_TUNE_TRANSFORMER_FIELDS="nhead,num_kv_heads",
        G1_TUNE_EXPERIMENT_FIELDS="",
    )

    assert experiment.transformer.nhead == 4
    assert experiment.transformer.num_kv_heads == 4
    assert experiment.transformer.ffn == "swiglu"
    assert experiment.dataloader.batch_size == 1280
    assert experiment.num_in_batch_negatives == 512


def test_cls_mode_is_copied_as_an_explicit_treatment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TUNE_SOURCE_VARIANT="cls_interleaved",
        G1_TUNE_TRANSFORMER_FIELDS="",
        G1_TUNE_EXPERIMENT_FIELDS="cls_token_mode",
    )

    assert experiment.effective_cls_token_mode == "interleaved"


def test_physical_and_effective_batch_are_configured_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TUNE_SOURCE_VARIANT="seq_512",
        G1_TUNE_TRANSFORMER_FIELDS="",
        G1_TUNE_EXPERIMENT_FIELDS="max_seq_len",
        G1_TUNE_BATCH_SIZE="640",
        G1_TUNE_GRADIENT_ACCUMULATION_STEPS="2",
    )

    assert experiment.max_seq_len == 512
    assert experiment.dataloader.batch_size == 640
    assert experiment.dataloader.gradient_accumulation_steps == 2
    assert experiment.dataloader.effective_batch_size == 1280


def test_ffn_width_can_be_tuned_with_the_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TUNE_SOURCE_VARIANT="ffn_swiglu",
        G1_TUNE_TRANSFORMER_FIELDS="ffn",
        G1_TUNE_EXPERIMENT_FIELDS="",
        G1_TUNE_FFN_DIM="224",
    )

    assert experiment.transformer.ffn == "swiglu"
    assert experiment.transformer.ffn_intermediate_dim == 224


def test_ffn_depth_can_be_tuned_with_the_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TUNE_SOURCE_VARIANT="ffn_geglu",
        G1_TUNE_TRANSFORMER_FIELDS="ffn,gated_ffn_dropout",
        G1_TUNE_EXPERIMENT_FIELDS="",
        G1_TUNE_FFN_DIM="114",
        G1_TUNE_NUM_LAYERS="8",
    )

    assert experiment.transformer.ffn == "geglu"
    assert experiment.transformer.ffn_intermediate_dim == 114
    assert experiment.transformer.num_layers == 8
    assert experiment.transformer.ffn_dropout == 0.1
    assert experiment.transformer.gated_ffn_dropout
    assert experiment.mup_base_dim == 16
    assert experiment.mup_delta_dim == 32
    assert experiment.item_embedding_dim == 64


def test_unknown_copy_field_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unknown transformer field"):
        _load(monkeypatch, G1_TUNE_TRANSFORMER_FIELDS="dim,typo")


def test_tuning_epoch_cap_can_only_extend_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="at least 20"):
        _load(monkeypatch, G1_TUNE_EPOCHS="19")


def test_extended_cap_requires_collision_safe_run_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TUNE_EPOCHS="30",
        G1_TUNE_RUN_REVISION="3",
        G1_TUNE_RUN="test_cap30_ts2_r3",
    )
    assert experiment.num_epochs == 30

    with pytest.raises(ValueError, match="cap30_ts2_r3"):
        _load(
            monkeypatch,
            G1_TUNE_EPOCHS="30",
            G1_TUNE_RUN_REVISION="3",
            G1_TUNE_RUN="test_r3",
        )


def test_logq_correction_is_an_explicit_tuning_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TUNE_SOURCE_VARIANT="baseline",
        G1_TUNE_EXPERIMENT_FIELDS="negative_sampling",
        G1_TUNE_LOGQ_CORRECTION="baseline",
    )

    assert experiment.negative_sampling == "offline_logq"
    assert experiment.logq_correction == "baseline"


def test_architecture_treatment_is_preserved_under_mup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TUNE_SOURCE_VARIANT="heads_4",
        G1_TUNE_TRANSFORMER_FIELDS="nhead,num_kv_heads",
        G1_TUNE_EXPERIMENT_FIELDS="",
    )

    assert type(experiment) is MuTransferGenerationExperiment
    assert experiment.transformer.nhead == 4
    assert experiment.transformer.num_kv_heads == 4
    assert experiment.mup_base_dim == 16
    assert experiment.mup_delta_dim == 32
