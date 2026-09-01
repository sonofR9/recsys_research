from __future__ import annotations

from collections.abc import Mapping, Sequence
import itertools
import hashlib
import json
import math

import numpy as np
from scipy.optimize import minimize


_CAPS = (1, 4, 8, 16)
_TREATMENTS = {
    1: "one_example",
    4: "truncated_4",
    8: "truncated_8",
    16: "truncated_16",
}
_FROZEN_VALIDATION = (0.1367, 0.1343, 0.1363)
_FROZEN_FULL = 0.13468336146286186
_NOISE_BAND = 0.003


class Rq13CapFitError(RuntimeError):
    pass


def practical_cap_ceiling(eligible_target_counts: Sequence[int]) -> dict[str, object]:
    counts = _validated_counts(eligible_target_counts)
    support = sorted(counts, reverse=True)[math.ceil(len(counts) / 2) - 1]
    q16 = _input_tokens(counts, 16)
    running_tokens = 0
    compute = 0
    for cap in range(1, max(counts) + 1):
        running_tokens += sum(
            min(129, count + 2 - cap) for count in counts if count >= cap
        )
        if running_tokens > 2 * q16:
            break
        compute = cap
    extrapolation = 32
    selected = min(support, compute, extrapolation)
    if selected < 17:
        raise Rq13CapFitError("practical cap ceiling leaves no new cap above 16")
    return {
        "support": support,
        "support_definition": "largest cap retained by at least half of users",
        "compute": compute,
        "compute_definition": "largest cap with input tokens no greater than 2x cap 16",
        "extrapolation": extrapolation,
        "extrapolation_definition": "one doubling beyond the largest observed cap",
        "selected": selected,
        "input_tokens": {
            "16": q16,
            str(selected): _input_tokens(counts, selected),
        },
        "expanded_examples": {
            "16": sum(min(16, count) for count in counts),
            str(selected): sum(min(selected, count) for count in counts),
        },
    }


def build_cap_fit(
    rq13_evidence: Mapping[str, object],
    rq12_evidence: Mapping[str, object],
    eligible_target_counts: Sequence[int],
    logged_cap16_input_tokens: int,
) -> dict[str, object]:
    points = _validation_points(rq13_evidence)
    validation_control, full_control = _control_targets(rq12_evidence)
    selection_target = 1.10 * validation_control
    reader_target = 1.10 * full_control
    ceiling = practical_cap_ceiling(eligible_target_counts)
    if (
        isinstance(logged_cap16_input_tokens, bool)
        or not isinstance(logged_cap16_input_tokens, int)
        or logged_cap16_input_tokens
        != ceiling["input_tokens"]["16"]  # type: ignore[index]
    ):
        raise Rq13CapFitError(
            "source-derived cap-16 input tokens do not reproduce logged metadata"
        )
    primary = _fit(points, "power")
    comparator = _fit(points, "exponential")
    target_cap = _target_cap(primary, selection_target)
    selected_cap = min(
        ceiling["selected"],
        max(17, target_cap if target_cap is not None else ceiling["selected"]),
    )
    sensitivity = _sensitivity(points, selection_target)
    leave_one_out = _leave_one_out(points, selection_target)
    slopes = _log_slopes(points)
    model_form_uncertain = bool(
        abs(primary["prediction_at_32"] - comparator["prediction_at_32"]) > _NOISE_BAND
        or (primary["asymptote"] > selection_target)
        != (comparator["asymptote"] > selection_target)
    )
    diagnostics = {
        "observed_monotonic": all(
            points[left] <= points[right]
            for left, right in zip(_CAPS[:-1], _CAPS[1:], strict=True)
        ),
        "log2_slopes": slopes,
        "log2_slopes_nonincreasing": all(
            left >= right for left, right in zip(slopes[:-1], slopes[1:], strict=True)
        ),
        "rmse_within_noise_band": primary["rmse"] <= _NOISE_BAND,
        "primary_parameters_interior": _parameters_interior(primary, "power"),
        "target_below_primary_asymptote": primary["asymptote"] > selection_target,
        "model_form_uncertain": model_form_uncertain,
        "model_form_prediction_difference_at_32": abs(
            primary["prediction_at_32"] - comparator["prediction_at_32"]
        ),
    }
    reliable = (
        all(
            diagnostics[key]
            for key in (
                "observed_monotonic",
                "log2_slopes_nonincreasing",
                "rmse_within_noise_band",
                "primary_parameters_interior",
            )
        )
        and not model_form_uncertain
    )
    reaches_within_ceiling = (
        target_cap is not None and target_cap <= ceiling["selected"]
    )
    return {
        "status": "selected_cap_pending",
        "metric": "validation Recall@100 from validation-selected checkpoints",
        "fit_points": {str(cap): points[cap] for cap in _CAPS},
        "selection_target": {
            "metric": "mean validation Recall@100",
            "control_values": list(_FROZEN_VALIDATION),
            "control_mean": validation_control,
            "multiplier": 1.10,
            "value": selection_target,
        },
        "reader_success_target": {
            "metric": "mean full-user Recall@100",
            "control_mean": full_control,
            "multiplier": 1.10,
            "value": reader_target,
        },
        "fit": primary,
        "nonselecting_comparator": comparator,
        "sensitivity": sensitivity,
        "leave_one_out": leave_one_out,
        "diagnostics": diagnostics,
        "extrapolation_reliable": reliable,
        "target_cap": target_cap,
        "target_reached_within_practical_ceiling": reaches_within_ceiling,
        "practical_ceiling": ceiling,
        "selected_cap": selected_cap,
        "selection_reason": (
            "first fitted target-crossing cap within the practical ceiling"
            if reaches_within_ceiling
            else "bounded cap-32 boundary probe; no target-attainment claim"
        ),
        "input_bindings": {
            "contributing_artifacts": _contributing_artifacts(rq13_evidence),
            "eligible_target_counts_sha256": hashlib.sha256(
                json.dumps(
                    list(_validated_counts(eligible_target_counts)),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "cap16_input_tokens_provenance_check": {
                "source_derived": ceiling["input_tokens"]["16"],  # type: ignore[index]
                "logged_metadata": logged_cap16_input_tokens,
                "matches": True,
            },
        },
    }


def _validation_points(evidence: Mapping[str, object]) -> dict[int, float]:
    if (
        evidence.get("research_question") != "RQ13 encoder-decoder prefix expansion"
        or evidence.get("dataset_size") != "500m"
    ):
        raise Rq13CapFitError("RQ13 cap-fit evidence identifies the wrong study")
    if (
        evidence.get("missing_initial_artifacts") != []
        or evidence.get("required_followups") != []
        or evidence.get("required_boundary_followups") != []
    ):
        raise Rq13CapFitError("RQ13 cap-fit LR surfaces are unresolved")
    winners = evidence.get("surface_winners")
    surfaces = evidence.get("treatments")
    if not isinstance(winners, Mapping):
        raise Rq13CapFitError("RQ13 surface winners are absent")
    if not isinstance(surfaces, Mapping):
        raise Rq13CapFitError("RQ13 cap-fit surfaces are absent")
    result = {}
    for cap, treatment in _TREATMENTS.items():
        row = winners.get(treatment)
        validation = row.get("validation") if isinstance(row, Mapping) else None
        value = (
            validation.get("recall@100") if isinstance(validation, Mapping) else None
        )
        if not _finite_unit(value):
            raise Rq13CapFitError(
                f"cap {cap} validation-selected Recall@100 is absent or invalid"
            )
        result[cap] = float(value)
        treatment_surface = surfaces.get(treatment)
        artifacts = (
            treatment_surface.get("artifacts")
            if isinstance(treatment_surface, Mapping)
            else None
        )
        rates = (
            [
                artifact.get("deep_learning_rate")
                for artifact in artifacts
                if isinstance(artifact, Mapping)
            ]
            if isinstance(artifacts, list)
            else []
        )
        winner_rate = (
            row.get("deep_learning_rate") if isinstance(row, Mapping) else None
        )
        if (
            len(rates) < 3
            or len(set(rates)) != len(rates)
            or winner_rate not in rates
            or winner_rate in {min(rates), max(rates)}
        ):
            raise Rq13CapFitError(f"cap {cap} LR boundary is unresolved")
    return result


def _control_targets(evidence: Mapping[str, object]) -> tuple[float, float]:
    if (
        evidence.get("research_question") != "RQ12 decoder-only query layout"
        or evidence.get("dataset_size") != "500m"
    ):
        raise Rq13CapFitError("RQ12 control evidence identifies the wrong study")
    methods = evidence.get("methods")
    standard = (
        [
            row
            for row in methods
            if isinstance(row, Mapping) and row.get("method") == "standard"
        ]
        if isinstance(methods, list)
        else []
    )
    if len(standard) != 1:
        raise Rq13CapFitError("RQ12 standard control is not unique")
    artifacts = standard[0].get("artifacts")
    values = []
    if isinstance(artifacts, list):
        for row in artifacts:
            metrics = (
                row.get("validation_metrics") if isinstance(row, Mapping) else None
            )
            value = metrics.get("recall@100") if isinstance(metrics, Mapping) else None
            values.append(value)
    if tuple(values) != _FROZEN_VALIDATION:
        raise Rq13CapFitError("RQ12 frozen validation control values changed")
    full = standard[0].get("mean_full_user_metrics")
    full_recall = full.get("recall@100") if isinstance(full, Mapping) else None
    if full_recall != _FROZEN_FULL:
        raise Rq13CapFitError("RQ12 frozen full-user control value changed")
    return sum(_FROZEN_VALIDATION) / len(_FROZEN_VALIDATION), _FROZEN_FULL


def _fit(
    points: Mapping[int, float], model: str, *, require_unique: bool = True
) -> dict[str, float]:
    point_caps = tuple(sorted(points))
    if len(point_caps) < 3:
        raise Rq13CapFitError(f"{model} cap-response fit needs at least three points")
    caps = np.asarray(point_caps, dtype=np.float64)
    values = np.asarray([points[cap] for cap in point_caps], dtype=np.float64)
    maximum = float(values.max())

    def prediction(parameters: np.ndarray) -> np.ndarray:
        a, b, shape = parameters
        if model == "power":
            return a - b * caps ** (-shape)
        return a - b * np.exp(-shape * (caps - 1))

    def objective(parameters: np.ndarray) -> float:
        residual = prediction(parameters) - values
        return float(np.mean(residual * residual))

    lower_shape = 0.05 if model == "power" else 0.0001
    starts = (
        (max(maximum + 0.01, 0.14), min(0.08, maximum), 0.45),
        (0.25, 0.18, 0.2),
        (0.8, 0.7, 0.08),
        (max(maximum + 0.001, 0.13), min(0.06, maximum), 1.0),
    )
    solutions = []
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=((maximum, 1.0), (0.0, 1.0), (lower_shape, 2.0)),
            constraints={"type": "ineq", "fun": lambda value: value[0] - value[1]},
            options={"ftol": 1e-15, "maxiter": 10_000},
        )
        if result.success and np.isfinite(result.x).all() and math.isfinite(result.fun):
            solutions.append(result)
    if not solutions:
        raise Rq13CapFitError(f"{model} cap-response optimizer failed")
    best = min(solutions, key=lambda result: result.fun)
    near = [result for result in solutions if result.fun <= best.fun + 1e-10]
    predictions = [
        result.x[0]
        - result.x[1]
        * (32 ** (-result.x[2]) if model == "power" else math.exp(-result.x[2] * 31))
        for result in near
    ]
    if require_unique and max(predictions) - min(predictions) > 1e-4:
        raise Rq13CapFitError(f"{model} cap-response fit is nonunique")
    a, b, shape = (float(value) for value in best.x)
    jacobian = np.column_stack(
        (
            np.ones_like(caps),
            -(caps ** (-shape) if model == "power" else np.exp(-shape * (caps - 1))),
            (
                b * caps ** (-shape) * np.log(caps)
                if model == "power"
                else b * (caps - 1) * np.exp(-shape * (caps - 1))
            ),
        )
    )
    if require_unique and np.linalg.matrix_rank(jacobian) != 3:
        raise Rq13CapFitError(f"{model} cap-response fit is rank deficient")
    prediction_32 = (
        a - b * 32 ** (-shape) if model == "power" else a - b * math.exp(-shape * 31)
    )
    return {
        "model": (
            "A - B * cap^(-p)" if model == "power" else "A - B * exp(-k * (cap - 1))"
        ),
        "asymptote": a,
        "B": b,
        "shape": shape,
        "rmse": math.sqrt(float(best.fun)),
        "prediction_at_32": prediction_32,
    }


def _target_cap(fit: Mapping[str, float], target: float) -> int | None:
    a = fit["asymptote"]
    b = fit["B"]
    p = fit["shape"]
    if a <= target or b <= 0:
        return None
    return max(1, math.ceil((b / (a - target)) ** (1 / p)))


def _sensitivity(points: Mapping[int, float], target: float) -> dict[str, object]:
    predictions = []
    target_caps = []
    fits = []
    for signs in itertools.product((-1, 1), repeat=4):
        corner = {
            cap: min(1.0, max(0.0, points[cap] + sign * _NOISE_BAND))
            for cap, sign in zip(_CAPS, signs, strict=True)
        }
        fit = _fit(corner, "power", require_unique=False)
        predictions.append(fit["prediction_at_32"])
        target_cap = _target_cap(fit, target)
        target_caps.append(target_cap)
        fits.append(
            {
                "perturbations": {
                    str(cap): sign * _NOISE_BAND
                    for cap, sign in zip(_CAPS, signs, strict=True)
                },
                "asymptote": fit["asymptote"],
                "B": fit["B"],
                "shape": fit["shape"],
                "rmse": fit["rmse"],
                "prediction_at_32": fit["prediction_at_32"],
                "target_cap": target_cap,
            }
        )
    finite_caps = [cap for cap in target_caps if cap is not None]
    return {
        "kind": "all 16 independent ±0.003 corners; sensitivity, not a confidence interval",
        "fits": fits,
        "prediction_at_32_range": [min(predictions), max(predictions)],
        "target_cap_range": (
            [min(finite_caps), max(finite_caps)] if finite_caps else None
        ),
        "infinite_target_cap_corners": sum(cap is None for cap in target_caps),
        "fraction_reaching_target_by_32": sum(
            prediction >= target for prediction in predictions
        )
        / len(predictions),
    }


def _leave_one_out(points: Mapping[int, float], target: float) -> dict[str, object]:
    fits = {}
    predictions = []
    target_caps = []
    for omitted in _CAPS:
        fit = _fit(
            {cap: value for cap, value in points.items() if cap != omitted},
            "power",
            require_unique=False,
        )
        cap = _target_cap(fit, target)
        fits[str(omitted)] = {
            "prediction_at_32": fit["prediction_at_32"],
            "target_cap": cap,
            "asymptote": fit["asymptote"],
        }
        predictions.append(fit["prediction_at_32"])
        target_caps.append(cap)
    finite = [cap for cap in target_caps if cap is not None]
    return {
        "kind": "four leave-one-cap-out fits; diagnostic envelope, not a confidence interval",
        "fits": fits,
        "prediction_at_32_range": [min(predictions), max(predictions)],
        "target_cap_range": [min(finite), max(finite)] if finite else None,
        "infinite_target_cap_fits": sum(cap is None for cap in target_caps),
    }


def _contributing_artifacts(evidence: Mapping[str, object]) -> dict[str, object]:
    winners = evidence["surface_winners"]
    assert isinstance(winners, Mapping)
    result = {}
    for treatment in _TREATMENTS.values():
        row = winners[treatment]
        assert isinstance(row, Mapping)
        hashes = row.get("artifact_sha256")
        source = row.get("source_manifest_sha256")
        if (
            not isinstance(hashes, Mapping)
            or set(hashes)
            != {"training_metadata.json", "final_metrics.json", "sweep.log"}
            or not all(isinstance(value, str) and value for value in hashes.values())
            or not isinstance(source, str)
            or not source
        ):
            raise Rq13CapFitError(
                f"{treatment}: exact artifact or source-manifest binding is absent"
            )
        result[treatment] = {
            "artifact_sha256": dict(hashes),
            "source_manifest_sha256": source,
        }
    return result


def _parameters_interior(fit: Mapping[str, float], model: str) -> bool:
    lower = 0.05 if model == "power" else 0.0001
    return bool(
        fit["asymptote"] < 1 - 1e-6
        and fit["B"] > 1e-6
        and fit["B"] < fit["asymptote"] - 1e-6
        and fit["shape"] > lower + 1e-6
        and fit["shape"] < 2 - 1e-6
    )


def _log_slopes(points: Mapping[int, float]) -> list[float]:
    return [
        (points[right] - points[left]) / (math.log2(right) - math.log2(left))
        for left, right in zip(_CAPS[:-1], _CAPS[1:], strict=True)
    ]


def _input_tokens(counts: Sequence[int], cap: int) -> int:
    return sum(
        sum(min(129, count + 1 - offset) for offset in range(min(cap, count)))
        for count in counts
    )


def _validated_counts(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        raise Rq13CapFitError("eligible target counts are absent")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise Rq13CapFitError("eligible target counts must be positive integers")
    return tuple(values)


def _finite_unit(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )
