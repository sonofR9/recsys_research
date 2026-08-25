import math
import runpy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dcn.config.settings import LrScheduleConfig
from dcn.models import (
    CriterionSpec,
    LossWrapper,
    MultiCriterion,
    TargetExtractionWrapper,
)
from dcn.tests.helpers import scalar_feature
from dcn.training import EpochTrainer
from neuralrec.run.callbacks.base import Callback
from neuralrec.run.callbacks.clipping import GradientNormClippingCallback
from neuralrec.run.callbacks.lr_schedule import LrSchedule
from neuralrec.utils import LOSS_DENOMINATOR


class _RegressionDataset(Dataset):
    def __init__(self) -> None:
        self.samples = torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, -0.5],
                [1.0, 1.0, 0.25],
                [2.0, -1.0, 1.5],
                [-1.0, 2.0, -1.0],
                [0.5, 0.5, 0.0],
                [3.0, 1.0, 2.0],
                [1.0, 3.0, -0.25],
                [-2.0, -1.0, -0.5],
                [2.0, 2.0, 1.0],
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]


class _RegressionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.1]))

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = batch[:, :2] @ self.weight
        return {
            "loss": (prediction - batch[:, 2]).square().mean(),
            "last_input": batch[-1, 0].detach(),
        }


class _StepRecorder(Callback):
    def __init__(self) -> None:
        self.begin = 0
        self.before_optimizer = 0
        self.end = 0
        self.rates: list[float] = []
        self.losses: list[float] = []
        self.corresponding_outputs: list[bool] = []

    def on_step_begin(self, state: dict[str, Any], batch: Any) -> None:
        self.begin += 1

    def on_before_optimizer_step(self, state: dict[str, Any]) -> None:
        self.before_optimizer += 1
        self.rates.append(state["optimizer"].param_groups[0]["lr"])

    def on_step_end(
        self, state: dict[str, Any], batch: Any, out: dict[str, Any]
    ) -> None:
        self.end += 1
        self.losses.append(float(out["loss"]))
        self.corresponding_outputs.append(
            float(batch[-1, 0]) == float(out["last_input"])
        )


def _train_regression(
    *, physical_batch_size: int, accumulation_steps: int
) -> tuple[EpochTrainer, _StepRecorder, _StepRecorder]:
    model = _RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    recorder = _StepRecorder()
    periodic_recorder = _StepRecorder()
    trainer = EpochTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=DataLoader(
            _RegressionDataset(), batch_size=physical_batch_size, shuffle=False
        ),
        num_epochs=1,
        gradient_accumulation_steps=accumulation_steps,
        callbacks=[
            LrSchedule("linear"),
            GradientNormClippingCallback(0.1),
            recorder,
            periodic_recorder.every_n_steps(2, include_step_zero=True),
        ],
    )
    trainer.train()
    return trainer, recorder, periodic_recorder


def test_separable_loss_accumulation_matches_effective_batch_with_clipping() -> None:
    reference, _, _ = _train_regression(physical_batch_size=4, accumulation_steps=1)
    accumulated, recorder, periodic_recorder = _train_regression(
        physical_batch_size=2, accumulation_steps=2
    )

    torch.testing.assert_close(
        accumulated.model.weight, reference.model.weight, rtol=1e-6, atol=1e-7
    )
    assert accumulated.total_steps == math.ceil(5 / 2)
    assert accumulated.global_step == 3
    assert accumulated.step == 3
    assert recorder.begin == recorder.before_optimizer == recorder.end == 3
    assert recorder.rates == [0.2, 0.1, 0.0]
    assert all(recorder.corresponding_outputs)
    assert (
        periodic_recorder.begin
        == periodic_recorder.before_optimizer
        == periodic_recorder.end
        == 2
    )


def test_undersized_microbatch_is_weighted_by_its_sample_count() -> None:
    reference, reference_recorder, _ = _train_regression(
        physical_batch_size=6, accumulation_steps=1
    )
    accumulated, recorder, _ = _train_regression(
        physical_batch_size=3, accumulation_steps=2
    )

    torch.testing.assert_close(
        accumulated.model.weight, reference.model.weight, rtol=1e-6, atol=1e-7
    )
    assert accumulated.total_steps == 2
    assert all(recorder.corresponding_outputs)
    assert recorder.losses == pytest.approx(reference_recorder.losses)


def _finite_horizon_rates(
    max_epochs: int,
    *,
    horizon_epochs: int = 20,
    schedule: LrScheduleConfig | None = None,
    stop_at_horizon: bool = True,
) -> list[float]:
    model = _RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    recorder = _StepRecorder()
    trainer = EpochTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=DataLoader(
            _RegressionDataset(), batch_size=10, shuffle=False
        ),
        num_epochs=max_epochs,
        lr_schedule_horizon_epochs=horizon_epochs,
        callbacks=[
            (
                LrSchedule("linear", stop_at_horizon=stop_at_horizon)
                if schedule is None
                else LrSchedule(
                    schedule.shape,
                    warmup_fraction=schedule.warmup_fraction,
                    min_lr_fraction=schedule.min_lr_fraction,
                    cycles=schedule.cycles,
                    timescale_steps=schedule.timescale_steps,
                    timescale_fraction=schedule.timescale_fraction,
                    power_exponent=schedule.power_exponent,
                    power_transition_tokens=schedule.power_transition_tokens,
                    stop_at_horizon=stop_at_horizon,
                )
            ),
            recorder,
        ],
    )

    trainer.train()

    return recorder.rates


@pytest.mark.parametrize("extended_cap", [40, 60])
def test_safety_cap_extension_preserves_shared_lr_prefix(
    extended_cap: int,
) -> None:
    initial = _finite_horizon_rates(20)
    extended = _finite_horizon_rates(extended_cap)

    assert extended[: len(initial)] == pytest.approx(initial)
    assert min(initial) > 0


@pytest.mark.parametrize("extended_cap", [40, 60])
def test_a_spent_annealing_horizon_ends_the_run(extended_cap: int) -> None:
    horizon_rates = _finite_horizon_rates(20)

    assert _finite_horizon_rates(extended_cap) == pytest.approx(horizon_rates)
    assert len(horizon_rates) == 19


def test_adaptive_schedule_uses_the_cap_to_reach_the_exact_last_step() -> None:
    rates = _finite_horizon_rates(20, stop_at_horizon=False)

    assert len(rates) == 20
    assert min(rates[:-1]) > 0
    assert rates[-1] == 0


@pytest.mark.parametrize(
    "schedule",
    [
        LrScheduleConfig("linear", min_lr_fraction=0.1),
        LrScheduleConfig("step"),
        LrScheduleConfig("exponential"),
    ],
)
def test_a_floored_or_stepped_horizon_also_ends_the_run(
    schedule: LrScheduleConfig,
) -> None:
    assert len(_finite_horizon_rates(40, schedule=schedule)) == 19


def test_a_never_ending_schedule_trains_to_its_extended_cap() -> None:
    assert len(_finite_horizon_rates(40, schedule=LrScheduleConfig("constant"))) == 40


def test_mup_cap_extension_preserves_cosine_warmup_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (
        Path(__file__).resolve().parents[3]
        / "experiments/g1_sasrec_item_ids_likes/configs/variant.py"
    )
    monkeypatch.setenv("G1_VARIANT", "mup_dim32_lr2e3")
    monkeypatch.setenv("G1_DATASET_SIZE", "50m")
    for name in (
        "G1_MAX_USERS",
        "G1_SEED",
        "G1_TRAIN_BATCH_SIZE",
        "G1_VAL_BATCH_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("G1_MAX_EPOCHS", "20")
    initial = runpy.run_path(str(script))["experiment"]
    monkeypatch.setenv("G1_MAX_EPOCHS", "40")
    extended = runpy.run_path(str(script))["experiment"]

    assert initial.num_epochs == initial.lr_schedule_horizon_epochs == 20
    assert extended.num_epochs == 40
    assert extended.lr_schedule_horizon_epochs == 20
    initial_rates = _finite_horizon_rates(
        initial.num_epochs,
        horizon_epochs=initial.lr_schedule_horizon_epochs,
        schedule=initial.lr_schedule,
    )
    extended_rates = _finite_horizon_rates(
        extended.num_epochs,
        horizon_epochs=extended.lr_schedule_horizon_epochs,
        schedule=extended.lr_schedule,
    )
    assert extended_rates[: len(initial_rates)] == pytest.approx(initial_rates)


class _PackedTargetMeanModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        values = batch["values"]
        is_target = torch.ones_like(values, dtype=torch.bool)
        is_target[batch["cumulative_lens"][1:] - 1] = False
        predictions = self.weight * values[is_target]
        return {
            "loss": (predictions - 2 * values[is_target]).square().mean(),
            LOSS_DENOMINATOR: int(is_target.sum()),
        }


class _PackedBatches:
    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return iter(self.batches)


def test_packed_sequence_accumulation_uses_target_count() -> None:
    microbatches = [
        {
            "values": torch.arange(1, 6, dtype=torch.float32),
            "cumulative_lens": torch.tensor([0, 5]),
        },
        {
            "values": torch.arange(6, 8, dtype=torch.float32),
            "cumulative_lens": torch.tensor([0, 2]),
        },
    ]
    combined = {
        "values": torch.arange(1, 8, dtype=torch.float32),
        "cumulative_lens": torch.tensor([0, 5, 7]),
    }

    def train(
        batches: list[dict[str, torch.Tensor]], accumulation: int
    ) -> torch.Tensor:
        model = _PackedTargetMeanModel()
        trainer = EpochTrainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            train_loader=_PackedBatches(batches),  # type: ignore[arg-type]
            num_epochs=1,
            gradient_accumulation_steps=accumulation,
        )
        trainer.train()
        return model.weight

    torch.testing.assert_close(
        train(microbatches, 2), train([combined], 1), rtol=1e-6, atol=1e-7
    )


class _ZeroDenominatorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: torch.Tensor) -> dict[str, Any]:
        return {"loss": self.weight * 0, LOSS_DENOMINATOR: 0}


def test_all_zero_denominators_take_one_zero_gradient_step() -> None:
    model = _ZeroDenominatorModel()
    trainer = EpochTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=DataLoader(torch.arange(4), batch_size=2),
        num_epochs=1,
        gradient_accumulation_steps=2,
    )

    trainer.train()

    assert trainer.global_step == 1
    assert model.weight.item() == 1.0


class _RankingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        rows = batch["float_columns"]["target_like"].num_rows()
        predictions = self.weight.expand(rows)
        return {
            "like": scalar_feature(predictions),
            "listen": scalar_feature(predictions),
        }


def _ranking_batch(
    like: list[float], listen: list[float], listen_mask: list[bool]
) -> dict[str, Any]:
    rows = len(like)
    return {
        "int_columns": {
            "listen_mask": scalar_feature(torch.tensor(listen_mask)),
        },
        "float_columns": {
            "target_like": scalar_feature(torch.tensor(like, dtype=torch.float32)),
            "target_listen": scalar_feature(torch.tensor(listen, dtype=torch.float32)),
        },
        "cumulative_lens": torch.tensor([0, rows]),
    }


def _ranking_loss() -> LossWrapper:
    criterion = MultiCriterion(
        [
            CriterionSpec(
                "like",
                TargetExtractionWrapper(
                    nn.BCEWithLogitsLoss(),
                    prediction_column="like",
                    target_column="target_like",
                ),
                1.0,
            ),
            CriterionSpec(
                "listen",
                TargetExtractionWrapper(
                    nn.MSELoss(),
                    prediction_column="listen",
                    target_column="target_listen",
                    mask_column="listen_mask",
                ),
                0.5,
            ),
        ]
    )
    return LossWrapper(_RankingModel(), criterion)


def test_packed_ranking_accumulation_weights_each_criterion_by_its_targets() -> None:
    microbatches = [
        _ranking_batch([0, 1, 0], [0.1, 0.7, 0.2], [True, False, True]),
        _ranking_batch([1, 1], [0.8, 0.4], [False, False]),
    ]
    combined = _ranking_batch(
        [0, 1, 0, 1, 1],
        [0.1, 0.7, 0.2, 0.8, 0.4],
        [True, False, True, False, False],
    )

    def train(batches: list[dict[str, Any]], accumulation: int) -> torch.Tensor:
        model = _ranking_loss()
        trainer = EpochTrainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            train_loader=_PackedBatches(batches),  # type: ignore[arg-type]
            num_epochs=1,
            gradient_accumulation_steps=accumulation,
        )
        trainer.train()
        return model.model.weight

    torch.testing.assert_close(
        train(microbatches, 2), train([combined], 1), rtol=1e-6, atol=1e-7
    )


class _BatchSequence:
    def __init__(self) -> None:
        self.batches = [
            {"value": torch.tensor([1.0]), "cumulative_lens": torch.tensor([0, 3])},
            {"value": torch.tensor([2.0]), "cumulative_lens": torch.tensor([0, 5])},
            {"value": torch.tensor([3.0]), "cumulative_lens": torch.tensor([0, 7])},
        ]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return iter(self.batches)


class _PowerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {
            "loss": (batch["value"] * self.weight).mean(),
            LOSS_DENOMINATOR: batch["value"].shape[0],
        }


def test_power_schedule_counts_every_accumulated_microbatch_once() -> None:
    model = _PowerModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = EpochTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=_BatchSequence(),  # type: ignore[arg-type]
        num_epochs=1,
        gradient_accumulation_steps=2,
        callbacks=[LrSchedule("power", power_transition_tokens=1, power_exponent=-0.5)],
    )

    trainer.train()

    assert trainer.total_steps == 2
    assert optimizer.param_groups[0]["power_seen_tokens"] == 15


def test_accumulation_steps_must_be_positive() -> None:
    model = _RegressionModel()
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        EpochTrainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            train_loader=DataLoader(_RegressionDataset(), batch_size=2),
            num_epochs=1,
            gradient_accumulation_steps=0,
        )
