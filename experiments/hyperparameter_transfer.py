from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Observation:
    horizon: float
    parameters: Mapping[str, float]
    objective: float


@dataclass(frozen=True)
class HorizonResponse:
    horizon: float
    optimum: Mapping[str, float]
    objective: float
    r_squared: float
    sampled_bounds: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class PowerLaw:
    amplitude: float
    exponent: float
    r_squared: float

    @property
    def beta(self) -> float:
        return -self.exponent

    def predict(self, horizon: float) -> float:
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError("horizon must be positive and finite")
        return self.amplitude * horizon**self.exponent


@dataclass(frozen=True)
class HorizonTransferFit:
    parameters: Mapping[str, PowerLaw]
    responses: Sequence[HorizonResponse]

    def predict(self, horizon: float) -> dict[str, float]:
        return {
            name: power_law.predict(horizon)
            for name, power_law in self.parameters.items()
        }

    def as_dict(self) -> dict:
        return {
            "parameters": {
                name: {
                    "amplitude": law.amplitude,
                    "exponent": law.exponent,
                    "beta": law.beta,
                    "r_squared": law.r_squared,
                }
                for name, law in self.parameters.items()
            },
            "responses": [
                {
                    "horizon": response.horizon,
                    "optimum": dict(response.optimum),
                    "objective": response.objective,
                    "r_squared": response.r_squared,
                    "sampled_bounds": {
                        name: list(bounds)
                        for name, bounds in response.sampled_bounds.items()
                    },
                }
                for response in self.responses
            ],
        }


def _quadratic_design(log_parameters: np.ndarray) -> np.ndarray:
    columns = [np.ones(len(log_parameters))]
    columns.extend(log_parameters[:, index] for index in range(log_parameters.shape[1]))
    columns.extend(
        log_parameters[:, left] * log_parameters[:, right]
        for left in range(log_parameters.shape[1])
        for right in range(left, log_parameters.shape[1])
    )
    return np.column_stack(columns)


def _response_fit(
    observations: Sequence[Observation],
    parameter_names: tuple[str, ...],
    *,
    maximize: bool,
) -> HorizonResponse:
    values = np.asarray(
        [
            [observation.parameters[name] for name in parameter_names]
            for observation in observations
        ],
        dtype=float,
    )
    objectives = np.asarray([observation.objective for observation in observations])
    log_values = np.log(values)
    design = _quadratic_design(log_values)
    if (
        len(observations) < design.shape[1]
        or np.linalg.matrix_rank(design) < design.shape[1]
    ):
        raise ValueError(
            f"horizon {observations[0].horizon:g} does not span a full "
            "quadratic response surface"
        )
    coefficients, _, _, _ = np.linalg.lstsq(design, objectives, rcond=None)

    dimension = len(parameter_names)
    linear = coefficients[1 : 1 + dimension]
    quadratic = coefficients[1 + dimension :]
    hessian = np.zeros((dimension, dimension))
    offset = 0
    for left in range(dimension):
        for right in range(left, dimension):
            coefficient = quadratic[offset]
            offset += 1
            if left == right:
                hessian[left, right] = 2 * coefficient
            else:
                hessian[left, right] = coefficient
                hessian[right, left] = coefficient

    eigenvalues = np.linalg.eigvalsh(hessian)
    if maximize and eigenvalues.max() >= -1e-10:
        raise ValueError("fitted response does not have a strict maximum")
    if not maximize and eigenvalues.min() <= 1e-10:
        raise ValueError("fitted response does not have a strict minimum")
    optimum_log = np.linalg.solve(hessian, -linear)

    bounds = {
        name: (float(values[:, index].min()), float(values[:, index].max()))
        for index, name in enumerate(parameter_names)
    }
    optimum = {
        name: float(math.exp(optimum_log[index]))
        for index, name in enumerate(parameter_names)
    }
    for name, value in optimum.items():
        lower, upper = bounds[name]
        if value <= lower * (1 + 1e-9) or value >= upper * (1 - 1e-9):
            raise ValueError(
                f"fitted optimum for {name} at horizon {observations[0].horizon:g} "
                "is outside or on the boundary of the sampled range; widen the sweep"
            )

    fitted = design @ coefficients
    residual_sum = float(np.square(objectives - fitted).sum())
    total_sum = float(np.square(objectives - objectives.mean()).sum())
    r_squared = (
        1.0
        if total_sum == 0 and residual_sum == 0
        else 1 - residual_sum / total_sum
    )
    optimum_objective = float(
        (_quadratic_design(optimum_log[None, :]) @ coefficients)[0]
    )
    return HorizonResponse(
        horizon=observations[0].horizon,
        optimum=optimum,
        objective=optimum_objective,
        r_squared=r_squared,
        sampled_bounds=bounds,
    )


def _power_law(horizons: np.ndarray, optima: np.ndarray) -> PowerLaw:
    log_horizons = np.log(horizons)
    log_optima = np.log(optima)
    exponent, intercept = np.polyfit(log_horizons, log_optima, 1)
    fitted = intercept + exponent * log_horizons
    residual_sum = float(np.square(log_optima - fitted).sum())
    total_sum = float(np.square(log_optima - log_optima.mean()).sum())
    scale = max(float(np.square(log_optima).sum()), 1.0)
    if total_sum <= np.finfo(float).eps * scale:
        exponent = 0.0
        intercept = float(log_optima.mean())
        r_squared = 1.0
    else:
        r_squared = 1 - residual_sum / total_sum
    return PowerLaw(
        amplitude=float(math.exp(intercept)),
        exponent=float(exponent),
        r_squared=r_squared,
    )


def fit_horizon_transfer(
    observations: Sequence[Observation], *, maximize: bool
) -> HorizonTransferFit:
    if not observations:
        raise ValueError("at least one observation is required")
    parameter_names = tuple(sorted(observations[0].parameters))
    if not parameter_names or any(
        tuple(sorted(observation.parameters)) != parameter_names
        for observation in observations
    ):
        raise ValueError("all observations must contain the same parameters")
    for observation in observations:
        if not math.isfinite(observation.horizon) or observation.horizon <= 0:
            raise ValueError("horizons must be positive and finite")
        if not math.isfinite(observation.objective):
            raise ValueError("objectives must be finite")
        if any(
            not math.isfinite(value) or value <= 0
            for value in observation.parameters.values()
        ):
            raise ValueError("transferred parameters must be positive and finite")

    grouped: dict[float, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.horizon, []).append(observation)
    responses = [
        _response_fit(grouped[horizon], parameter_names, maximize=maximize)
        for horizon in sorted(grouped)
    ]
    if len(responses) < 3:
        raise ValueError("at least three distinct horizons are required")

    horizons = np.asarray([response.horizon for response in responses])
    laws = {
        name: _power_law(
            horizons,
            np.asarray([response.optimum[name] for response in responses]),
        )
        for name in parameter_names
    }
    return HorizonTransferFit(parameters=laws, responses=responses)
