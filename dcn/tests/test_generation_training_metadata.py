import json
from types import SimpleNamespace

import pytest
import torch

from dcn.config import GenerationExperiment, TrainingCallbacks
from dcn.config.settings import LrScheduleConfig, TransformerConfig
from neuralrec.run.callbacks import BestWeights, EarlyStopping, ValidationCallback
from utils.global_config import config as global_config


def _write_metadata(
    tmp_path,
    *,
    stopped_epoch_index: int,
    best_epoch_index: int | None,
    early_stopped: bool,
    lr_schedule: LrScheduleConfig | None = None,
    lr_schedule_horizon_epochs: int | None = None,
    num_epochs: int = 20,
    adaptive_schedule_early_stopping: bool = False,
    cls_token_mode: str = "none",
    transformer: TransformerConfig | None = None,
    item_encoder: torch.nn.Module | None = None,
) -> dict:
    global_config.initialize(tmp_path)
    experiment = GenerationExperiment(
        run_name="selection_metadata",
        num_epochs=num_epochs,
        early_stopping_patience=3,
        early_stopping_min_delta=0,
        lr_schedule=lr_schedule,
        lr_schedule_horizon_epochs=lr_schedule_horizon_epochs,
        adaptive_schedule_early_stopping=adaptive_schedule_early_stopping,
        cls_token_mode=cls_token_mode,
        window="next_item" if cls_token_mode != "none" else "sliding",
        **({} if transformer is None else {"transformer": transformer}),
    )
    experiment.__dict__["item_embedding"] = (
        torch.nn.Embedding(5, 64) if item_encoder is None else item_encoder
    )
    experiment.__dict__["training_targets_per_epoch"] = 10
    experiment.__dict__["training_tokens_per_epoch"] = 20
    best_weights = BestWeights(metric_name="recall@100", metric_prefix="epoch/val_true")
    best_weights.best_epoch = best_epoch_index
    stopping = EarlyStopping(
        metric_name="recall@100",
        metric_prefix="epoch/val_true",
        patience=3,
    )
    stopping.should_stop = early_stopped
    experiment.__dict__["callbacks"] = TrainingCallbacks(
        all=[],
        best_weights=best_weights,
        validation=ValidationCallback(),
        early_stopping=stopping,
    )
    runner = SimpleNamespace(
        current_epoch=stopped_epoch_index,
        global_step=17,
        steps_per_epoch=1,
        lr_schedule_total_steps=lr_schedule_horizon_epochs or num_epochs,
        optimizer=SimpleNamespace(param_groups=[]),
        state={},
    )

    experiment._report_training_metadata(runner)

    path = tmp_path / "logs" / experiment.run_name / "training_metadata.json"
    return json.loads(path.read_text())


def test_training_metadata_accepts_a_non_embedding_history_encoder(tmp_path) -> None:
    item_encoder = torch.nn.Sequential(
        torch.nn.Embedding(5, 7),
        torch.nn.Linear(7, 11, bias=False),
    )
    item_encoder.out_dim = 11

    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=4,
        best_epoch_index=1,
        early_stopped=True,
        item_encoder=item_encoder,
    )

    assert metadata["item_embedding_dim"] == 11
    assert metadata["transfer_invariants"]["item_embedding_dim"] == 11


@pytest.mark.parametrize("cls_token_mode", ["end_only", "interleaved"])
def test_training_metadata_distinguishes_cls_query_modes(
    tmp_path,
    cls_token_mode: str,
) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=4,
        best_epoch_index=1,
        early_stopped=True,
        cls_token_mode=cls_token_mode,
    )

    assert metadata["cls_token"] is True
    assert metadata["cls_token_mode"] == cls_token_mode
    assert metadata["transfer_invariants"]["cls_token"] is True
    assert metadata["transfer_invariants"]["cls_token_mode"] == cls_token_mode


def test_default_transformer_metadata_keeps_the_historical_recipe_shape(
    tmp_path,
) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=4,
        best_epoch_index=1,
        early_stopped=True,
    )
    transformer = metadata["transfer_invariants"]["transformer"]

    assert "rope_base" not in transformer
    assert "learned_position_fusion" not in transformer
    assert "learned_position_fusion_normalization" not in transformer
    assert "learned_position_fusion_residual" not in transformer
    assert "learned_position_fusion_semantics_revision" not in transformer
    assert "learned_position_initialization" not in transformer
    assert "learned_position_initialization_semantics_revision" not in transformer
    assert "learned_position_reverse_correction" not in transformer
    assert "learned_position_reverse_max_scale" not in transformer
    assert "learned_position_reverse_correction_semantics_revision" not in transformer
    assert "learned_position_reverse_initializer_rng_nonadvancing" not in transformer
    assert "learned_position_reverse_initializer_semantics_revision" not in transformer


@pytest.mark.parametrize(
    ("transformer", "field", "value"),
    [
        (
            TransformerConfig(
                learned_positions="forward",
                learned_position_fusion="concat",
            ),
            "learned_position_fusion",
            "concat",
        ),
        (TransformerConfig(rope="forward", rope_base=100.0), "rope_base", 100.0),
    ],
)
def test_nondefault_position_fields_are_recorded_in_training_metadata(
    tmp_path, transformer: TransformerConfig, field: str, value: object
) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=4,
        best_epoch_index=1,
        early_stopped=True,
        transformer=transformer,
    )

    assert metadata["transfer_invariants"]["transformer"][field] == value


def test_normalized_concat_records_fusion_semantics_revision(tmp_path) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=4,
        best_epoch_index=1,
        early_stopped=True,
        transformer=TransformerConfig(
            learned_positions="forward",
            learned_position_fusion="concat",
            learned_position_fusion_normalization="input_rms",
        ),
    )
    transformer = metadata["transfer_invariants"]["transformer"]

    assert transformer["learned_position_fusion_normalization"] == "input_rms"
    assert "learned_position_fusion_residual" not in transformer
    assert transformer["learned_position_fusion_semantics_revision"] == 2


def test_rezero_concat_records_fusion_semantics_revision(tmp_path) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=4,
        best_epoch_index=1,
        early_stopped=True,
        transformer=TransformerConfig(
            learned_positions="forward",
            learned_position_fusion="concat",
            learned_position_fusion_residual="rezero",
        ),
    )
    transformer = metadata["transfer_invariants"]["transformer"]

    assert transformer["learned_position_fusion_residual"] == "rezero"
    assert "learned_position_fusion_normalization" not in transformer
    assert transformer["learned_position_fusion_semantics_revision"] == 3


def test_zero_reverse_records_distinct_initialization_semantics(tmp_path) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=4,
        best_epoch_index=1,
        early_stopped=True,
        transformer=TransformerConfig(
            learned_positions=("forward", "reverse"),
            learned_position_initialization="zero_reverse",
        ),
    )
    transformer = metadata["transfer_invariants"]["transformer"]

    assert transformer["learned_position_initialization"] == "zero_reverse"
    assert transformer["learned_position_initialization_semantics_revision"] == 1


def test_bounded_reverse_records_distinct_semantics_and_scale(tmp_path) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=4,
        best_epoch_index=1,
        early_stopped=True,
        transformer=TransformerConfig(
            learned_positions=("forward", "reverse"),
            learned_position_reverse_correction="bounded_tanh",
            learned_position_reverse_max_scale=0.1,
        ),
    )
    transformer = metadata["transfer_invariants"]["transformer"]

    assert transformer["learned_position_reverse_correction"] == "bounded_tanh"
    assert transformer["learned_position_reverse_max_scale"] == 0.1
    assert transformer["learned_position_reverse_correction_semantics_revision"] == 1
    assert "learned_position_initialization" not in transformer


def test_bounded_reverse_records_rng_isolated_initializer_semantics(tmp_path) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=4,
        best_epoch_index=1,
        early_stopped=True,
        transformer=TransformerConfig(
            learned_positions=("forward", "reverse"),
            learned_position_reverse_correction="bounded_tanh",
            learned_position_reverse_max_scale=0.025,
            learned_position_reverse_initializer_rng_nonadvancing=True,
        ),
    )
    transformer = metadata["transfer_invariants"]["transformer"]

    assert transformer["learned_position_reverse_max_scale"] == 0.025
    assert transformer["learned_position_reverse_initializer_rng_nonadvancing"] is True
    assert transformer["learned_position_reverse_initializer_semantics_revision"] == 1


@pytest.mark.parametrize(
    ("schedule", "stops_early"),
    [(LrScheduleConfig("step"), False), (LrScheduleConfig("constant"), True)],
)
def test_only_a_step_by_step_schedule_gets_early_stopping(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: LrScheduleConfig,
    stops_early: bool,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    global_config.initialize(tmp_path)
    experiment = GenerationExperiment(
        run_name="callbacks",
        early_stopping_patience=3,
        lr_schedule=schedule,
        lr_schedule_horizon_epochs=20,
    )
    experiment.__dict__["item_embedding"] = torch.nn.Embedding(5, 64)

    callbacks = experiment.create_callbacks()

    assert (callbacks.early_stopping is not None) == stops_early


def test_rq5_adaptive_schedule_keeps_patience_for_annealed_shape(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    global_config.initialize(tmp_path)
    experiment = GenerationExperiment(
        run_name="rq5_callbacks",
        early_stopping_patience=3,
        lr_schedule=LrScheduleConfig("linear"),
        lr_schedule_horizon_epochs=20,
        adaptive_schedule_early_stopping=True,
    )
    experiment.__dict__["item_embedding"] = torch.nn.Embedding(5, 64)

    callbacks = experiment.create_callbacks()

    assert callbacks.early_stopping is not None
    assert callbacks.lr_schedule is not None
    assert callbacks.lr_schedule.stops_at_horizon is False


@pytest.mark.parametrize(
    ("stopped_epoch_index", "early_stopped", "expected_status", "next_horizon"),
    [
        (16, True, "calibrated", None),
        (9, True, "shorten_horizon", 10),
        (19, False, "extend_horizon", 30),
    ],
)
def test_rq5_annealed_horizon_calibration_status(
    tmp_path,
    stopped_epoch_index: int,
    early_stopped: bool,
    expected_status: str,
    next_horizon: int | None,
) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=stopped_epoch_index,
        best_epoch_index=max(0, stopped_epoch_index - 3),
        early_stopped=early_stopped,
        lr_schedule=LrScheduleConfig("linear"),
        lr_schedule_horizon_epochs=20,
        adaptive_schedule_early_stopping=True,
    )

    assert metadata["horizon_calibration_status"] == expected_status
    assert metadata["next_lr_schedule_horizon_epochs"] == next_horizon
    assert metadata["selection_resolved"] == (expected_status == "calibrated")
    assert metadata["optimizer_steps_per_epoch"] == 1
    assert metadata["lr_schedule_horizon_steps"] == 20


def test_a_completed_annealed_horizon_resolves_its_own_selection(tmp_path) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=19,
        best_epoch_index=19,
        early_stopped=False,
        lr_schedule=LrScheduleConfig("linear"),
        lr_schedule_horizon_epochs=20,
    )

    assert metadata["lr_horizon_complete"]
    assert metadata["selection_resolved"]


def test_an_annealed_run_cut_short_of_its_horizon_is_unresolved(tmp_path) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=10,
        best_epoch_index=7,
        early_stopped=False,
        lr_schedule=LrScheduleConfig("linear"),
        lr_schedule_horizon_epochs=20,
    )

    assert not metadata["lr_horizon_complete"]
    assert not metadata["selection_resolved"]


def test_a_step_by_step_schedule_still_needs_an_early_stop(tmp_path) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=25,
        best_epoch_index=22,
        early_stopped=True,
        lr_schedule=LrScheduleConfig("inverse_sqrt", timescale_steps=100),
        num_epochs=40,
    )

    assert not metadata["lr_horizon_complete"]
    assert metadata["selection_resolved"]


def test_training_metadata_records_actual_early_stopped_selection(tmp_path) -> None:
    metadata = _write_metadata(
        tmp_path, stopped_epoch_index=4, best_epoch_index=1, early_stopped=True
    )

    assert metadata["num_epochs"] == 20
    assert metadata["max_epochs"] == 20
    assert metadata["epochs_trained"] == 5
    assert metadata["best_epoch"] == 2
    assert metadata["stopped_epoch"] == 5
    assert metadata["early_stopped"]
    assert metadata["selection_resolved"]
    assert not metadata["best_epoch_at_cap"]
    assert metadata["training_horizon"] == 50
    assert metadata["token_horizon"] == 100
    assert metadata["optimizer_steps"] == 17
    assert (
        metadata["transfer_invariants"]
        | {
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0,
            "early_stopping_metric": "recall@100",
            "early_stopping_metric_prefix": "epoch/val_true",
        }
        == metadata["transfer_invariants"]
    )


@pytest.mark.parametrize("best_epoch_index", [None, 18, 19])
def test_training_metadata_rejects_unresolved_cap_selection(
    tmp_path, best_epoch_index: int | None
) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=19,
        best_epoch_index=best_epoch_index,
        early_stopped=False,
    )

    assert not metadata["selection_resolved"]
    assert metadata["best_epoch_at_cap"] == (best_epoch_index == 19)


def test_training_metadata_rejects_early_stop_reported_at_cap(tmp_path) -> None:
    metadata = _write_metadata(
        tmp_path,
        stopped_epoch_index=19,
        best_epoch_index=15,
        early_stopped=True,
    )

    assert metadata["early_stopped"]
    assert not metadata["best_epoch_at_cap"]
    assert not metadata["selection_resolved"]
