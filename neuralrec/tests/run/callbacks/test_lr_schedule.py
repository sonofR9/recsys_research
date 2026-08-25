import math

import pytest
import torch

from neuralrec.run.callbacks.lr_schedule import LrSchedule, ScheduleShape


class _Runner:
    def __init__(self, total_steps: int | None) -> None:
        self.total_steps = total_steps
        self.global_step = 0


def _optimizer(rates: tuple[float, ...] = (1.0,)) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": rate} for rate in rates]
    )


def _named_optimizer(order: tuple[str, ...]) -> torch.optim.Optimizer:
    rates = {"embedding": 1.0, "deep": 0.1}
    return torch.optim.SGD(
        [
            {
                "params": [torch.nn.Parameter(torch.zeros(1))],
                "lr": rates[name],
                "schedule_group": name,
            }
            for name in order
        ]
    )


def _trace(
    schedule: LrSchedule,
    *,
    steps: int = 10,
    until: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> list[list[float]]:
    optimizer = _optimizer() if optimizer is None else optimizer
    runner = _Runner(steps)
    state = {"train_runner": runner, "optimizer": optimizer}
    schedule.on_train_begin(state)

    seen = []
    for step in range(steps if until is None else until):
        runner.global_step = step
        schedule.on_step_begin(state, None)
        seen.append([group["lr"] for group in optimizer.param_groups])
    return seen


def test_a_constant_schedule_without_warmup_leaves_every_rate_alone() -> None:
    seen = _trace(LrSchedule(), optimizer=_optimizer((1.0, 0.1)))

    assert seen == [[1.0, 0.1]] * 10


def test_warmup_ramps_up_to_the_configured_rate() -> None:
    seen = [row[0] for row in _trace(LrSchedule(warmup_fraction=0.5))]

    assert seen[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert seen[5:] == pytest.approx([1.0] * 5)


def test_every_group_keeps_its_own_rate_through_the_schedule() -> None:
    seen = _trace(LrSchedule("linear"), optimizer=_optimizer((1.0, 0.1)))

    assert all(math.isclose(fast, 10 * slow) for fast, slow in seen)


@pytest.mark.parametrize(
    "order", [("embedding", "deep"), ("deep", "embedding")]
)
def test_deep_only_scope_uses_stable_group_identity(order: tuple[str, ...]) -> None:
    optimizer = _named_optimizer(order)
    seen = _trace(
        LrSchedule("linear", optimizer_group_scope="deep_only"),
        optimizer=optimizer,
    )
    traces = {
        group["schedule_group"]: [row[index] for row in seen]
        for index, group in enumerate(optimizer.param_groups)
    }

    assert traces["embedding"] == pytest.approx([1.0] * 10)
    assert traces["deep"][0] == pytest.approx(0.1)
    assert traces["deep"][-1] == pytest.approx(0.0)


def test_deep_only_scope_rejects_unnamed_optimizer_groups() -> None:
    schedule = LrSchedule("linear", optimizer_group_scope="deep_only")
    state = {"train_runner": _Runner(10), "optimizer": _optimizer((1.0, 0.1))}

    with pytest.raises(ValueError, match="schedule_group"):
        schedule.on_train_begin(state)


def test_group_lr_trace_is_keyed_by_stable_identity() -> None:
    schedule = LrSchedule("linear", optimizer_group_scope="deep_only")
    optimizer = _named_optimizer(("deep", "embedding"))
    runner = _Runner(2)
    state = {"train_runner": runner, "optimizer": optimizer}
    schedule.on_train_begin(state)
    for step in range(2):
        runner.global_step = step
        schedule.on_step_begin(state, None)
        runner.global_step = step + 1
        schedule.on_epoch_end(state)

    assert schedule.group_lr_trace == {
        "deep": pytest.approx([0.1, 0.0]),
        "embedding": pytest.approx([1.0, 1.0]),
    }


def test_duplicate_deep_subgroups_produce_one_logical_trace_value_per_epoch() -> None:
    parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
    optimizer = torch.optim.SGD(
        [
            {
                "params": [parameters[0]],
                "lr": 1.0,
                "schedule_group": "embedding",
            },
            {"params": [parameters[1]], "lr": 0.1, "schedule_group": "deep"},
            {"params": [parameters[2]], "lr": 0.025, "schedule_group": "deep"},
        ]
    )
    schedule = LrSchedule("linear", optimizer_group_scope="deep_only")
    runner = _Runner(4)
    state = {"train_runner": runner, "optimizer": optimizer}
    schedule.on_train_begin(state)
    for epoch in range(2):
        for step in range(epoch * 2, epoch * 2 + 2):
            runner.global_step = step
            schedule.on_step_begin(state, None)
        runner.global_step = (epoch + 1) * 2
        schedule.on_epoch_end(state)

    assert schedule.group_lr_trace == {
        "embedding": pytest.approx([1.0, 1.0]),
        "deep": pytest.approx([0.1 * (2 / 3), 0.0]),
    }


@pytest.mark.parametrize("shape", ["linear", "cosine", "polynomial"])
def test_a_decaying_schedule_runs_from_the_full_rate_down_to_the_floor(
    shape: ScheduleShape,
) -> None:
    seen = [row[0] for row in _trace(LrSchedule(shape, min_lr_fraction=0.25))]

    assert seen[0] == pytest.approx(1.0)
    assert seen[-1] == pytest.approx(0.25)
    assert seen == sorted(seen, reverse=True)


def test_step_schedule_makes_two_discrete_drops() -> None:
    seen = [row[0] for row in _trace(LrSchedule("step"))]

    assert seen == pytest.approx([1.0] * 5 + [0.1] * 2 + [0.01] * 3)


def test_exponential_schedule_decays_smoothly_by_two_orders_of_magnitude() -> None:
    seen = [row[0] for row in _trace(LrSchedule("exponential"))]

    assert seen[0] == pytest.approx(1.0)
    assert seen[-1] == pytest.approx(0.01)
    assert seen == sorted(seen, reverse=True)


def test_cosine_schedule_can_restart_for_multiple_cycles() -> None:
    seen = [row[0] for row in _trace(LrSchedule("cosine", cycles=2), steps=11)]

    assert seen[0] == pytest.approx(1.0)
    assert seen[4] < 0.1
    assert seen[5] == pytest.approx(1.0)
    assert seen[-1] == pytest.approx(0.0)


def test_schedule_requires_at_least_one_cycle() -> None:
    with pytest.raises(ValueError, match="cycles"):
        LrSchedule("cosine", cycles=0)


def test_warmup_stable_decay_holds_then_cosine_decays() -> None:
    seen = [
        row[0]
        for row in _trace(
            LrSchedule("warmup_stable_decay", warmup_fraction=0.2), steps=20
        )
    ]

    assert seen[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert seen[4:16] == pytest.approx([1.0] * 12)
    assert seen[-1] == pytest.approx(0.0)


def test_inverse_sqrt_decays_on_its_timescale_rather_than_on_one_step() -> None:
    """`1/sqrt(t)` is only gentle when read in units of the timescale: it takes
    three of them to halve. Measured per step it collapses inside the first
    epoch, which is the difference between a schedule and a switch."""
    seen = [row[0] for row in _trace(LrSchedule("inverse_sqrt", timescale_steps=4))]

    assert seen[0] == pytest.approx(1.0)
    assert seen[4] == pytest.approx(math.sqrt(1 / 2))
    assert seen[-1] == pytest.approx(math.sqrt(4 / 13))


def test_inverse_sqrt_timescale_can_track_the_run_length() -> None:
    short = [
        row[0]
        for row in _trace(
            LrSchedule("inverse_sqrt", timescale_fraction=0.05), steps=100
        )
    ]
    long = [
        row[0]
        for row in _trace(
            LrSchedule("inverse_sqrt", timescale_fraction=0.05), steps=1_000
        )
    ]

    assert short[5] == pytest.approx(math.sqrt(1 / 2))
    assert long[50] == pytest.approx(math.sqrt(1 / 2))


def test_inverse_sqrt_decays_from_the_end_of_a_warmup_on_the_warmups_timescale() -> (
    None
):
    """Its usual form: the warmup sets both where the peak is and how fast the
    decay reads, so a warmup alone is enough to define the shape."""
    seen = [row[0] for row in _trace(LrSchedule("inverse_sqrt", warmup_fraction=0.4))]

    assert seen[3] == pytest.approx(1.0)
    assert seen[4] == pytest.approx(1.0)
    assert seen[8] == pytest.approx(math.sqrt(1 / 2))


def test_inverse_sqrt_refuses_a_run_that_gives_it_no_timescale() -> None:
    state = {"train_runner": _Runner(10), "optimizer": _optimizer()}

    with pytest.raises(ValueError, match="timescale_steps"):
        LrSchedule("inverse_sqrt").on_train_begin(state)


def test_power_schedule_decays_by_tokens_seen_without_a_run_horizon() -> None:
    schedule = LrSchedule(
        "power", power_exponent=-0.5, power_transition_tokens=100
    )
    optimizer = _optimizer((1.0, 0.1))
    runner = _Runner(None)
    state = {"train_runner": runner, "optimizer": optimizer}
    schedule.on_train_begin(state)

    schedule.on_step_begin(state, {"cumulative_lens": torch.tensor([0, 40, 80])})
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1.0, 0.1]
    )
    schedule.on_step_begin(state, {"cumulative_lens": torch.tensor([0, 120])})

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [math.sqrt(100 / 200), 0.1 * math.sqrt(100 / 200)]
    )
    assert all(group["power_seen_tokens"] == 200 for group in optimizer.param_groups)


def test_power_schedule_restores_seen_tokens_from_optimizer_groups() -> None:
    optimizer = _optimizer()
    optimizer.param_groups[0]["power_seen_tokens"] = 400
    schedule = LrSchedule(
        "power", power_exponent=-0.5, power_transition_tokens=100
    )
    state = {"train_runner": _Runner(None), "optimizer": optimizer}

    schedule.on_train_begin(state)
    schedule.on_step_begin(state, {"cumulative_lens": torch.tensor([0, 100])})

    assert optimizer.param_groups[0]["lr"] == pytest.approx(math.sqrt(100 / 500))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"power_transition_tokens": 0}, "transition"),
        ({"power_exponent": 0.1, "power_transition_tokens": 10}, "negative"),
        (
            {"warmup_fraction": 0.1, "power_transition_tokens": 10},
            "does not use fractional warmup",
        ),
    ],
)
def test_power_schedule_validates_its_shape(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LrSchedule("power", **kwargs)


def test_a_trainer_that_does_not_know_its_length_can_still_run_a_shape_without_one() -> (
    None
):
    state = {"train_runner": _Runner(None), "optimizer": _optimizer()}
    LrSchedule("inverse_sqrt", timescale_steps=4).on_train_begin(state)

    with pytest.raises(ValueError, match="does not know how long"):
        LrSchedule("cosine").on_train_begin(state)
    with pytest.raises(ValueError, match="does not know how long"):
        LrSchedule(warmup_fraction=0.1).on_train_begin(state)


def test_resuming_does_not_compound_the_schedule_onto_an_already_scaled_rate(
    tmp_path,
) -> None:
    """What the optimizer saves is a mid-schedule rate, not the rate the run was
    configured with, so a resumed schedule has to recover the latter."""
    straight_through = _trace(LrSchedule("cosine", warmup_fraction=0.2))

    interrupted = _optimizer()
    _trace(LrSchedule("cosine", warmup_fraction=0.2), optimizer=interrupted, until=6)
    saved = tmp_path / "optimizer.pt"
    torch.save(interrupted.state_dict(), saved)

    reloaded = _optimizer()
    reloaded.load_state_dict(torch.load(saved, weights_only=False))
    resumed = _trace(LrSchedule("cosine", warmup_fraction=0.2), optimizer=reloaded)

    assert resumed[-1] == pytest.approx(straight_through[-1])
