import hashlib
from dataclasses import replace
from pathlib import Path

import mup
import polars as pl
import pytest

from dcn.config.query_retrieval import (
    CrossAttentionGenerationExperiment,
    MuTransferCrossAttentionGenerationExperiment,
)
from dcn.config.query_retrieval_training import (
    MuTransferRq15CrossAttentionGenerationExperiment,
)
from dcn.config.settings import TRANSFORMER
from dcn.models.history_tokens import EndQuerySlots
from dcn.tests.miniature_yambda import configure
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact
from utils.global_config import config as global_config
from utils.report_file_facts import report_file_fact_scope


def _history_transformer(dim: int = 8):
    return replace(
        TRANSFORMER,
        dim=dim,
        num_layers=2,
        nhead=2,
        num_kv_heads=2,
        ffn="swiglu",
        ffn_intermediate_dim=4 * dim,
        attention_window=None,
    )


def _retrieval_decoder(dim: int = 8):
    return replace(
        _history_transformer(dim),
        num_layers=1,
        ffn_intermediate_dim=32 if dim == 8 else 2 * dim,
        learned_positions="forward",
    )


def _experiment(base_path: Path, **overrides):
    settings = {
        "transformer": _history_transformer(),
        "retrieval_decoder": _retrieval_decoder(),
        "num_in_batch_negatives": 4,
        **overrides,
    }
    return configure(
        CrossAttentionGenerationExperiment,
        base_path,
        **settings,
    )


@pytest.fixture(autouse=True)
def initialize_config(base_path: Path) -> None:
    global_config.initialize(base_path)


def test_encoder_decoder_uses_bidirectional_history_and_one_query(
    base_path: Path,
) -> None:
    model = _experiment(base_path).base_model

    assert all(not layer.is_causal for layer in model.memory_encoder.layers)
    assert model.query_slots is None
    assert len(model.decoder.self_attention_layers) == 1
    assert model.decoder_query.shape == (8,)


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("include_history", [True, False])
def test_decoder_decoder_exposes_the_approved_query_memory_cross(
    base_path: Path, shared: bool, include_history: bool
) -> None:
    model = _experiment(
        base_path,
        query_architecture="decoder_decoder",
        query_slots_shared=shared,
        include_history_memory=include_history,
    ).base_model

    assert all(layer.is_causal for layer in model.memory_encoder.layers)
    assert isinstance(model.query_slots, EndQuerySlots)
    assert model.query_slots.embeddings.shape == (1 if shared else 4, 8)
    assert model.include_history_memory is include_history


def test_bounded_prefix_policy_reaches_the_training_loader(base_path: Path) -> None:
    experiment = _experiment(
        base_path,
        window="bounded_prefix",
        prefix_length_rule="truncated",
        prefix_cap=8,
    )

    dataset = experiment.sequence_train_loader.dataset

    assert dataset.window == "bounded_prefix"
    assert dataset.prefix_length_rule == "truncated"
    assert dataset.prefix_cap == 8


def test_prefix_policy_has_distinct_shared_training_counts(
    base_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DCN_RUNNER_DATA_READY", str(base_path / "runner-ready"))
    one_prefix = _experiment(
        base_path,
        validation_days=0,
        window="bounded_prefix",
        prefix_length_rule="truncated",
        prefix_cap=1,
    )
    three_prefixes = _experiment(
        base_path,
        validation_days=0,
        window="bounded_prefix",
        prefix_length_rule="truncated",
        prefix_cap=3,
    )

    assert one_prefix.training_targets_per_epoch == 2
    assert three_prefixes.training_targets_per_epoch == 6


def test_rq15_training_counts_do_not_fetch_individual_sequences(
    base_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = configure(
        MuTransferRq15CrossAttentionGenerationExperiment,
        base_path,
        transformer=_history_transformer(64),
        retrieval_decoder=_retrieval_decoder(64),
        query_architecture="decoder_decoder",
        query_slots_shared=False,
        include_history_memory=False,
        validation_days=0,
        item_embedding_dim=64,
        mup_base_dim=16,
        mup_delta_dim=32,
        num_in_batch_negatives=4,
    )
    dataset = experiment.sequence_train_loader.dataset

    def reject_fetch(dataset: object, index: int) -> dict:
        raise AssertionError(f"unexpected per-sequence fetch {index}")

    monkeypatch.setattr(type(dataset), "__getitem__", reject_fetch)

    assert experiment.training_targets_per_epoch == dataset.event_count
    assert experiment.auxiliary_ntp_targets_per_epoch == (
        dataset.event_count - len(dataset)
    )
    assert experiment.training_tokens_per_epoch == (
        dataset.event_count + len(dataset) * experiment.num_query_slots
    )


def test_rq15_verifier_manifest_does_not_construct_model(base_path: Path) -> None:
    checkpoint = base_path / "selected-first-stage.pt"
    checkpoint.write_bytes(b"selected checkpoint")
    source_metadata = {
        "dataset_size": "500m",
        "source_recipe_run_name": "approved-source",
    }
    experiment = configure(
        MuTransferRq15CrossAttentionGenerationExperiment,
        base_path,
        transformer=_history_transformer(64),
        retrieval_decoder=_retrieval_decoder(64),
        query_architecture="decoder_decoder",
        query_slots_shared=False,
        include_history_memory=False,
        validation_days=0,
        item_embedding_dim=64,
        mup_base_dim=16,
        mup_delta_dim=32,
        num_in_batch_negatives=4,
        training_method="pretrained_finetune",
        first_stage_checkpoint=checkpoint,
        first_stage_checkpoint_metadata=source_metadata,
        auxiliary_ntp_weight=0.0,
    )

    top_level, invariants = verify_artifact._expected_metadata(experiment)

    expected = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_metadata": source_metadata,
        "history_position_count": experiment.max_seq_len,
        "copied_modules": ["item_embedding", "memory_encoder", "tokenizer"],
        "newly_initialized_modules": [
            "decoder",
            "decoder_query",
            "query_projection",
            "query_slots",
        ],
    }
    assert top_level["first_stage_initialization"] == expected
    assert invariants["first_stage_initialization"] == expected
    assert "base_model" not in experiment.__dict__


def test_rq13_metadata_reports_each_required_training_count(base_path: Path) -> None:
    experiment = _experiment(
        base_path,
        validation_days=0,
        window="bounded_prefix",
        prefix_length_rule="truncated",
        prefix_cap=3,
    )

    metadata = experiment.generation_architecture_metadata()

    assert metadata["original_users_per_epoch"] == 2
    assert metadata["expanded_examples_per_epoch"] == 6
    assert metadata["candidate_targets_per_epoch"] == 6
    assert metadata["ntp_targets_per_epoch"] == 0
    assert metadata["input_tokens_per_epoch"] == experiment.training_tokens_per_epoch


def test_report_metadata_reuses_persistent_day_bounds(
    base_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "DCN_REPORT_FILE_FACTS", str(base_path / "report-file-facts.sqlite3")
    )
    with report_file_fact_scope(base_path):
        first = _experiment(base_path, validation_interval_seconds=86_400)
        expected = first.generation_architecture_metadata()
    day_paths = {
        path.resolve() for path in first.dataset_manager.day_to_path.values()
    }
    read_parquet = pl.read_parquet

    def reject_day_read(path: str | Path, *args, **kwargs):
        if Path(path).resolve() in day_paths:
            raise AssertionError(f"re-read cached daily parquet {path}")
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", reject_day_read)
    with report_file_fact_scope(base_path):
        second = _experiment(base_path, validation_interval_seconds=86_400)
        assert second.generation_architecture_metadata() == expected


def test_report_day_bounds_cache_includes_day_ids(base_path: Path) -> None:
    with report_file_fact_scope(base_path):
        first = _experiment(base_path)
        original = first._day_timestamp_bounds
        second = _experiment(base_path)
        second.dataset_manager.day_to_path = {
            day + 100: path for day, path in second.dataset_manager.day_to_path.items()
        }
        shifted = second._day_timestamp_bounds

    assert set(shifted) == {day + 100 for day in original}


def test_report_day_bounds_cache_uses_complete_filter_identity(
    base_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left_values = list(range(100))
    right_values = left_values.copy()
    right_values[2] = 500
    left = pl.col("item_id").is_in(left_values)
    right = pl.col("item_id").is_in(right_values)
    assert str(left) == str(right)

    monkeypatch.setattr(
        CrossAttentionGenerationExperiment, "row_filter", property(lambda _: left)
    )
    with report_file_fact_scope(base_path):
        first = _experiment(base_path)
        first._day_timestamp_bounds

    reads = 0
    read_parquet = pl.read_parquet

    def count_day_read(path: str | Path, *args, **kwargs):
        nonlocal reads
        reads += 1
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", count_day_read)
    monkeypatch.setattr(
        CrossAttentionGenerationExperiment, "row_filter", property(lambda _: right)
    )
    with report_file_fact_scope(base_path):
        second = _experiment(base_path)
        expected_reads = len(second.dataset_manager.day_to_path)
        second._day_timestamp_bounds

    assert reads == expected_reads


def test_mup_parameterization_covers_both_transformer_stages(base_path: Path) -> None:
    experiment = configure(
        MuTransferCrossAttentionGenerationExperiment,
        base_path,
        transformer=_history_transformer(64),
        retrieval_decoder=_retrieval_decoder(64),
        item_embedding_dim=64,
        mup_base_dim=16,
        mup_delta_dim=32,
        num_in_batch_negatives=4,
    )

    model = experiment.base_model

    assert isinstance(model.query_projection, mup.MuReadout)
    assert all(hasattr(parameter, "infshape") for parameter in model.parameters())
    expected_scale = (
        experiment.mup_base_dim // experiment.transformer.nhead
    ) ** 0.5 / (experiment.transformer.dim // experiment.transformer.nhead)
    attention_layers = (
        *model.memory_encoder.layers,
        *model.decoder.self_attention_layers,
        *model.decoder.cross_attention_layers,
    )
    assert all(
        layer.softmax_scale == pytest.approx(expected_scale)
        for layer in attention_layers
    )


def test_artifact_verifier_expects_the_complete_query_recipe(base_path: Path) -> None:
    experiment = _experiment(
        base_path,
        validation_days=0,
        window="bounded_prefix",
        prefix_length_rule="required",
        prefix_cap=8,
    )

    top_level, invariants = verify_artifact._expected_metadata(experiment)

    for metadata in (top_level, invariants):
        assert metadata["query_architecture"] == "encoder_decoder"
        assert metadata["prefix_length_rule"] == "required"
        assert metadata["prefix_cap"] == 8
        assert metadata["retrieval_decoder"]["ffn_intermediate_dim"] == 32
        assert metadata["original_users_per_epoch"] == 2
        assert metadata["expanded_examples_per_epoch"] == 2
        assert metadata["candidate_targets_per_epoch"] == 2
        assert metadata["ntp_targets_per_epoch"] == 0
        assert metadata["input_tokens_per_epoch"] == 8


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"retrieval_decoder": replace(_retrieval_decoder(), ffn="gelu")}, "SwiGLU"),
        (
            {
                "retrieval_decoder": replace(
                    _retrieval_decoder(), ffn_intermediate_dim=48
                )
            },
            "divisible by 32",
        ),
        ({"query_architecture": "encoder_decoder", "query_slots_shared": True}, "only"),
    ],
)
def test_invalid_cross_attention_architectures_are_rejected(
    base_path: Path, overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _experiment(base_path, **overrides)
