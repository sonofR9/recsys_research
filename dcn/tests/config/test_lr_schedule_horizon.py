import pytest

from dcn.config.settings import LrScheduleConfig


@pytest.mark.parametrize(
    "shape",
    [
        "linear",
        "cosine",
        "polynomial",
        "warmup_stable_decay",
        "step",
        "exponential",
    ],
)
def test_shapes_that_decay_against_the_horizon_anneal_over_it(shape: str) -> None:
    assert LrScheduleConfig(shape).anneals_over_horizon


@pytest.mark.parametrize("shape", ["constant", "inverse_sqrt", "power"])
def test_step_by_step_shapes_do_not_anneal_over_a_horizon(shape: str) -> None:
    assert not LrScheduleConfig(shape).anneals_over_horizon


def test_a_rate_floor_does_not_change_who_decides_the_length() -> None:
    assert LrScheduleConfig("linear", min_lr_fraction=0.05).anneals_over_horizon
