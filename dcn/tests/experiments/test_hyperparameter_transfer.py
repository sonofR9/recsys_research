from __future__ import annotations

import math

import pytest

from experiments.hyperparameter_transfer import (
    Observation,
    fit_horizon_transfer,
)


def _surface_observations() -> list[Observation]:
    observations = []
    for horizon in (1_000.0, 4_000.0, 16_000.0):
        embedding_optimum = 0.08 * horizon**-0.2
        deep_optimum = 0.3 * horizon**-0.4
        for embedding_factor in (0.5, 1.0, 2.0):
            for deep_factor in (0.5, 1.0, 2.0):
                embedding_lr = embedding_optimum * embedding_factor
                deep_lr = deep_optimum * deep_factor
                score = -(
                    math.log(embedding_lr / embedding_optimum) ** 2
                    + 0.6 * math.log(deep_lr / deep_optimum) ** 2
                    + 0.2
                    * math.log(embedding_lr / embedding_optimum)
                    * math.log(deep_lr / deep_optimum)
                )
                observations.append(
                    Observation(
                        horizon=horizon,
                        parameters={
                            "embedding_lr": embedding_lr,
                            "deep_lr": deep_lr,
                        },
                        objective=score,
                    )
                )
    return observations


def test_joint_log_quadratic_then_power_law_recovers_known_transfer() -> None:
    fit = fit_horizon_transfer(_surface_observations(), maximize=True)

    prediction = fit.predict(64_000)

    assert prediction["embedding_lr"] == pytest.approx(0.08 * 64_000**-0.2)
    assert prediction["deep_lr"] == pytest.approx(0.3 * 64_000**-0.4)
    assert fit.parameters["embedding_lr"].beta == pytest.approx(0.2)
    assert fit.parameters["deep_lr"].beta == pytest.approx(0.4)
    assert all(response.r_squared == pytest.approx(1.0) for response in fit.responses)


def test_minimization_surface_is_supported() -> None:
    observations = [
        Observation(
            horizon=observation.horizon,
            parameters=observation.parameters,
            objective=-observation.objective,
        )
        for observation in _surface_observations()
    ]

    fit = fit_horizon_transfer(observations, maximize=False)

    assert fit.predict(64_000)["deep_lr"] == pytest.approx(0.3 * 64_000**-0.4)


def test_horizon_invariant_optimum_has_zero_exponent() -> None:
    observations = []
    for horizon in (1_000.0, 4_000.0, 16_000.0):
        for factor in (0.5, 1.0, 2.0):
            learning_rate = 1e-3 * factor
            observations.append(
                Observation(
                    horizon=horizon,
                    parameters={"lr": learning_rate},
                    objective=-math.log(factor) ** 2,
                )
            )

    fit = fit_horizon_transfer(observations, maximize=True)

    assert fit.parameters["lr"].exponent == pytest.approx(0, abs=1e-12)
    assert fit.parameters["lr"].r_squared == 1


def test_boundary_optimum_requires_a_wider_sweep() -> None:
    observations = []
    for horizon in (1_000.0, 4_000.0):
        for learning_rate in (1e-4, 2e-4, 4e-4):
            observations.append(
                Observation(
                    horizon=horizon,
                    parameters={"lr": learning_rate},
                    objective=-(math.log(learning_rate) - math.log(8e-4)) ** 2,
                )
            )

    with pytest.raises(ValueError, match="outside or on the boundary"):
        fit_horizon_transfer(observations, maximize=True)


def test_observations_must_share_parameter_names() -> None:
    observations = [
        Observation(1, {"embedding_lr": 1e-3}, 0.1),
        Observation(1, {"deep_lr": 1e-3}, 0.2),
        Observation(1, {"embedding_lr": 2e-3}, 0.3),
    ]

    with pytest.raises(ValueError, match="same parameters"):
        fit_horizon_transfer(observations, maximize=True)
