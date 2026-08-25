import json
import runpy
from pathlib import Path

import pytest


COLLECT = (
    Path(__file__).resolve().parents[3]
    / "experiments/g1_sasrec_item_ids_likes/analysis/collect.py"
)


def _load_status(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    schedule: dict,
    epochs_trained: int,
    horizon_epochs: int | None = 20,
    max_epochs: int = 40,
    early_stopped: bool = True,
    best_epoch: int | None = None,
) -> str:
    namespace = runpy.run_path(str(COLLECT))
    globals_ = namespace["load_report_runs"].__globals__
    monkeypatch.setitem(globals_, "GENERATED", root)
    monkeypatch.setitem(globals_, "has_current_generation_semantics", lambda _: True)
    monkeypatch.setitem(globals_, "_exact_artifact_matches", lambda *_: True)

    name = "g1_rqtune_schedule_step_e16d6_50m"
    directory = root / "logs" / name
    directory.mkdir(parents=True)
    metadata = {
        "training_semantics_revision": 1,
        "dataset_size": "50m",
        "batch_size": 512,
        "embedding_learning_rate": 0.016,
        "deep_learning_rate": 0.006,
        "num_epochs": max_epochs,
        "max_epochs": max_epochs,
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch
        or (epochs_trained - 3 if early_stopped else epochs_trained),
        "stopped_epoch": epochs_trained,
        "early_stopped": early_stopped,
        "best_epoch_at_cap": not early_stopped and epochs_trained == max_epochs,
        "selection_resolved": early_stopped and epochs_trained < max_epochs,
        "transfer_invariants": {
            "experiment_class": "MuTransferGenerationExperiment",
            "eval_every_n_epochs": 1,
            "restore_best_weights": True,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.0,
            "early_stopping_metric": "recall@100",
            "early_stopping_metric_prefix": "epoch/val_true",
            "max_seq_len": 128,
            "lr_schedule": schedule,
            "transformer": {"learned_positions": "reverse", "dropout": 0.1},
        },
    }
    if horizon_epochs is not None:
        metadata["lr_schedule_horizon_epochs"] = horizon_epochs
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text(
        json.dumps({"recall@100": 0.12, "ndcg@100": 0.05})
    )

    [run] = namespace["load_report_runs"]("50m")
    return run.status


def _schedule(shape: str, **overrides) -> dict:
    return {
        "shape": shape,
        "warmup_fraction": 0.0,
        "min_lr_fraction": 0.0,
        "cycles": 1,
        "timescale_fraction": None,
        "timescale_steps": None,
        "power_exponent": -0.51,
        "power_transition_tokens": None,
    } | overrides


@pytest.mark.parametrize(
    "shape", ["linear", "cosine", "polynomial", "warmup_stable_decay", "step"]
)
def test_a_run_that_spent_its_annealing_horizon_is_usable(
    monkeypatch, tmp_path: Path, shape: str
) -> None:
    status = _load_status(
        monkeypatch,
        tmp_path,
        schedule=_schedule(shape, warmup_fraction=0.05),
        epochs_trained=20,
        early_stopped=False,
    )

    assert status == "completed"


def test_a_frozen_tail_past_the_horizon_stays_usable(
    monkeypatch, tmp_path: Path
) -> None:
    status = _load_status(
        monkeypatch,
        tmp_path,
        schedule=_schedule("linear"),
        epochs_trained=23,
    )

    assert status == "completed"


def test_a_best_epoch_past_the_horizon_is_unusable(
    monkeypatch, tmp_path: Path
) -> None:
    status = _load_status(
        monkeypatch,
        tmp_path,
        schedule=_schedule("exponential"),
        epochs_trained=26,
        best_epoch=23,
    )

    assert status == "unusable"


def test_a_run_that_stopped_inside_its_horizon_is_usable(
    monkeypatch, tmp_path: Path
) -> None:
    status = _load_status(
        monkeypatch,
        tmp_path,
        schedule=_schedule("linear"),
        epochs_trained=14,
    )

    assert status == "completed"


def test_a_step_schedule_that_never_reached_its_first_drop_is_unusable(
    monkeypatch, tmp_path: Path
) -> None:
    status = _load_status(
        monkeypatch,
        tmp_path,
        schedule=_schedule("step"),
        epochs_trained=9,
    )

    assert status == "unusable"


def test_a_wsd_schedule_that_never_reached_its_decay_is_unusable(
    monkeypatch, tmp_path: Path
) -> None:
    status = _load_status(
        monkeypatch,
        tmp_path,
        schedule=_schedule("warmup_stable_decay", warmup_fraction=0.05),
        epochs_trained=14,
    )

    assert status == "unusable"


def test_a_step_by_step_schedule_at_its_cap_is_unusable(
    monkeypatch, tmp_path: Path
) -> None:
    status = _load_status(
        monkeypatch,
        tmp_path,
        schedule=_schedule("constant"),
        epochs_trained=40,
        horizon_epochs=None,
        early_stopped=False,
    )

    assert status == "unusable"
