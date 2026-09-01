from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_candidates import (
    AggregateCandidate,
    aggregate_boundary_candidates,
    aggregate_local_candidates,
    baseline_boundary_candidates,
    bridge_candidates,
    candidate_by_run,
    initial_candidates,
    selection_initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers.verify_artifact import (
    verify_config,
    verify_config_completed_historical_horizon,
)


METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")
_BANDS = {
    "recall@100": 0.003,
    "ndcg@100": 0.001,
    "recall@10": 0.003,
    "ndcg@10": 0.001,
    "coverage@100": 0.1,
}
_METRIC_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_CONFIG = (
    Path(__file__).parents[1] / "configs" / "aggregate_variant.py"
)


class AggregateReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class AggregateRun:
    candidate: AggregateCandidate
    best_epoch: int
    stopped_epoch: int
    validation_recall: float
    validation_ndcg: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class AggregateReportBundle:
    reader_markdown: str
    tuning_markdown: str
    evidence: dict[str, Any]


def build_report_bundle(
    runs: Iterable[AggregateRun], *, forced_followups: Iterable[str] = ()
) -> AggregateReportBundle:
    run_list = [run for run in runs if run.candidate.correction == 0]
    _require_unique_runs(run_list)
    followups = list(dict.fromkeys(forced_followups))
    by_recipe = _by_recipe(run_list)
    selection_initial = selection_initial_candidates()

    if followups:
        planned_recipes = {
            _recipe_key(candidate_by_run(run_name)) for run_name in followups
        }
        followups.extend(
            candidate.run_name
            for candidate in selection_initial
            if _recipe_key(candidate) not in by_recipe
            and _recipe_key(candidate) not in planned_recipes
        )
        return _pending(run_list, followups)

    missing_initial = _missing(selection_initial, by_recipe)
    if missing_initial:
        followups.extend(candidate.run_name for candidate in missing_initial)
        return _pending(run_list, followups)

    baseline_surface = _available(selection_initial[:3], by_recipe)
    baseline = _select(baseline_surface)
    baseline_boundaries = baseline_boundary_candidates(baseline.candidate)
    missing_baseline_boundaries = _missing(baseline_boundaries, by_recipe)
    if missing_baseline_boundaries:
        followups.extend(candidate.run_name for candidate in missing_baseline_boundaries)
    elif baseline_boundaries:
        baseline = _select(
            baseline_surface + _available(baseline_boundaries, by_recipe)
        )
        if baseline.candidate.stage == "baseline_boundary":
            baseline_boundary_candidates(baseline.candidate)

    selected_by_depth: list[AggregateRun] = []
    for depth in (4, 6, 8):
        selection_depth = tuple(
            candidate
            for candidate in selection_initial
            if candidate.family == "aggregate" and candidate.num_layers == depth
        )
        initial_depth = tuple(
            candidate
            for candidate in initial_candidates()
            if candidate.family == "aggregate" and candidate.num_layers == depth
        )
        surface = _available(selection_depth, by_recipe)
        winner = _select(surface)
        local = aggregate_local_candidates(
            _surface_candidate(winner, initial_depth)
        )
        missing_local = _missing(local, by_recipe)
        if missing_local:
            followups.extend(candidate.run_name for candidate in missing_local)
            continue
        if local:
            surface += _available(local, by_recipe)
            winner = _select(surface)

        pre_boundary = initial_depth + local
        boundaries = aggregate_boundary_candidates(
            _surface_candidate(winner, pre_boundary)
        )
        missing_boundaries = _missing(boundaries, by_recipe)
        if missing_boundaries:
            followups.extend(candidate.run_name for candidate in missing_boundaries)
            continue
        if boundaries:
            winner = _select(surface + _available(boundaries, by_recipe))
            source = _surface_candidate(winner, pre_boundary + boundaries)
            if source.stage == "optimizer_boundary":
                aggregate_boundary_candidates(source)
        selected_by_depth.append(winner)

    followups = list(dict.fromkeys(followups))
    if followups or len(selected_by_depth) != 3:
        return _pending(run_list, followups, baseline=baseline)

    aggregate = _select(selected_by_depth)
    bridges = bridge_candidates(
        baseline.candidate.deep_lr, selected_depth=aggregate.candidate.num_layers
    )
    missing_bridges = _missing(bridges, by_recipe)
    if missing_bridges:
        return _pending(
            run_list,
            [candidate.run_name for candidate in missing_bridges],
            baseline=baseline,
            aggregate=aggregate,
        )

    bridge_runs = _available(bridges, by_recipe)
    evidence = _ready_evidence(baseline, aggregate, bridge_runs)
    return AggregateReportBundle(
        reader_markdown=_render_reader(evidence),
        tuning_markdown=_render_tuning(run_list, baseline, aggregate),
        evidence=evidence,
    )


def collect_report_bundle(logs: Path) -> AggregateReportBundle:
    runs: list[AggregateRun] = []
    if not logs.exists():
        return build_report_bundle(runs)
    for directory in sorted(logs.glob("g1_aggregate_*_500m")):
        if not directory.is_dir():
            continue
        try:
            candidate = candidate_by_run(directory.name)
        except ValueError:
            continue
        if candidate.correction != 0:
            continue
        assignment = [f"G1_AGGREGATE_RUN={candidate.run_name}"]
        if verify_config(directory, _CONFIG, assignment) or (
            _historical_h15_candidate(candidate)
            and verify_config_completed_historical_horizon(
                directory, _CONFIG, assignment
            )
        ):
            runs.append(_load_run(directory, candidate))
    return build_report_bundle(runs)


def _historical_h15_candidate(candidate: AggregateCandidate) -> bool:
    return candidate.horizon_epochs == 15 and (
        candidate.family == "aggregate"
        and candidate.stage in {"initial", "local", "optimizer_boundary"}
        or candidate.family == "bridge"
        and candidate.member == "scheduler"
        and candidate.stage == "bridge"
    )


def write_report_bundle(
    bundle: AggregateReportBundle, scratchpad: Path, evidence: Path
) -> dict[str, Path]:
    paths = {
        "tuning": scratchpad / "aggregate_improvement_tuning_500m.md",
        "reader": scratchpad / "aggregate_improvement_reader_500m.md",
        "evidence": evidence / "aggregate_improvement_results.json",
    }
    _write(paths["tuning"], bundle.tuning_markdown)
    _write(paths["reader"], bundle.reader_markdown)
    _write(
        paths["evidence"],
        json.dumps(bundle.evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return paths


def _pending(
    runs: list[AggregateRun],
    followups: Iterable[str],
    *,
    baseline: AggregateRun | None = None,
    aggregate: AggregateRun | None = None,
) -> AggregateReportBundle:
    required = list(dict.fromkeys(followups))
    evidence: dict[str, Any] = {
        "claims_status": "pending",
        "required_followups": required,
        "selected_baseline": _candidate_evidence(baseline),
        "selected_aggregate": _candidate_evidence(aggregate),
        "selected_depth": (
            None if aggregate is None else aggregate.candidate.num_layers
        ),
    }
    return AggregateReportBundle(
        reader_markdown=(
            "## Aggregated improvement\n\n"
            "Pending completion of the approved aggregate protocol.\n"
        ),
        tuning_markdown=_render_tuning(runs, baseline, aggregate),
        evidence=evidence,
    )


def _ready_evidence(
    baseline: AggregateRun,
    aggregate: AggregateRun,
    bridges: list[AggregateRun],
) -> dict[str, Any]:
    improvements: dict[str, dict[str, float | str]] = {}
    for metric in METRICS:
        baseline_value = baseline.metrics[metric]
        aggregate_value = aggregate.metrics[metric]
        aggregate_gain = aggregate_value - baseline_value
        standalone = sum(run.metrics[metric] - baseline_value for run in bridges)
        gap = aggregate_gain - standalone
        band = _BANDS[metric]
        interaction = (
            "positive" if gap > band else "negative" if gap < -band else "unresolved"
        )
        improvements[metric] = {
            "baseline": baseline_value,
            "aggregate": aggregate_value,
            "aggregate_gain_points": aggregate_gain,
            "aggregate_gain_percent": 100 * aggregate_gain / baseline_value,
            "summed_standalone_gain_points": standalone,
            "interaction_gap": gap,
            "interaction": interaction,
        }
    outcome = classify_aggregate_outcome(baseline.metrics, aggregate.metrics)
    return {
        "claims_status": "ready",
        "required_followups": [],
        "selected_baseline": _candidate_evidence(baseline),
        "selected_aggregate": _candidate_evidence(aggregate),
        "selected_depth": aggregate.candidate.num_layers,
        "bridges": [_candidate_evidence(run) for run in bridges],
        "aggregate_outcome": outcome,
        "aggregated_improvement": improvements,
    }


def _render_reader(evidence: dict[str, Any]) -> str:
    outcome = evidence["aggregate_outcome"]
    lines = [
        "## Aggregated improvement",
        "",
        f"Outcome: **{outcome['classification']}**. {outcome['sentence']}",
        "",
        "The standalone total uses the eleven one-factor bridges against the frozen baseline.",
        "",
        "| metric | baseline | aggregate | aggregate gain | "
        "summed standalone gain | interaction gap | interaction |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for metric in METRICS:
        row = evidence["aggregated_improvement"][metric]
        lines.append(
            f"| {metric} | {row['baseline']:.3f} | {_reader_metric_cell(metric, row)} | "
            f"{row['aggregate_gain_points']:+.3f} | "
            f"{row['summed_standalone_gain_points']:+.3f} | "
            f"{row['interaction_gap']:+.3f} | {row['interaction']} |"
        )
    return "\n".join(lines) + "\n"


def _reader_metric_cell(metric: str, row: dict[str, Any]) -> str:
    value = f"{row['aggregate_gain_percent']:+.1f}% ({row['aggregate']:.3f})"
    gain = row["aggregate_gain_points"]
    if gain > _BANDS[metric]:
        return f'<span style="color: green">{value}</span>'
    if gain < -_BANDS[metric]:
        return f'<span style="color: red">{value}</span>'
    return value


def classify_aggregate_outcome(
    baseline: dict[str, float], aggregate: dict[str, float]
) -> dict[str, float | str]:
    recall_gain = aggregate["recall@100"] - baseline["recall@100"]
    ndcg_gain = aggregate["ndcg@100"] - baseline["ndcg@100"]
    if recall_gain > _BANDS["recall@100"] and ndcg_gain >= -_BANDS["ndcg@100"]:
        classification = "positive"
        sentence = (
            "Recall@100 clears the operational band and NDCG@100 is non-inferior."
        )
    elif (
        recall_gain > _BANDS["recall@100"] and ndcg_gain < -_BANDS["ndcg@100"]
    ) or (
        recall_gain < -_BANDS["recall@100"] and ndcg_gain > _BANDS["ndcg@100"]
    ):
        classification = "trade-off"
        sentence = "Recall@100 and NDCG@100 move materially in opposing directions."
    elif recall_gain < -_BANDS["recall@100"] or ndcg_gain < -_BANDS["ndcg@100"]:
        classification = "regression"
        sentence = "At least one primary reader metric is materially below baseline."
    else:
        classification = "unresolved"
        sentence = "The aggregate does not satisfy the approved positive-quality rule."
    return {
        "classification": classification,
        "sentence": sentence,
        "recall@100_gain": recall_gain,
        "ndcg@100_gain": ndcg_gain,
    }


def _render_tuning(
    runs: list[AggregateRun],
    baseline: AggregateRun | None,
    aggregate: AggregateRun | None,
) -> str:
    selected = {
        run.candidate.run_name for run in (baseline, aggregate) if run is not None
    }
    lines = [
        "# Aggregate-improvement tuning ledger",
        "",
        "| family | member | depth | embedding LR | deep LR | horizon | "
        "best/stopped | validation Recall@100 | validation NDCG@100 | selected |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in sorted(runs, key=lambda item: item.candidate.run_name):
        candidate = run.candidate
        lines.append(
            f"| {candidate.family} | {candidate.member or '-'} | "
            f"{candidate.num_layers} | {candidate.embedding_lr!r} | "
            f"{candidate.deep_lr!r} | {candidate.horizon_epochs or '-'} | "
            f"{run.best_epoch}/{run.stopped_epoch} | {run.validation_recall:.9f} | "
            f"{run.validation_ndcg:.9f} | "
            f"{'yes' if candidate.run_name in selected else ''} |"
        )
    return "\n".join(lines) + "\n"


def _by_recipe(runs: list[AggregateRun]) -> dict[tuple[Any, ...], AggregateRun]:
    result: dict[tuple[Any, ...], AggregateRun] = {}
    for run in runs:
        key = _recipe_key(run.candidate)
        previous = result.get(key)
        if previous is None or run.candidate.correction > previous.candidate.correction:
            result[key] = run
    return result


def _recipe_key(candidate: AggregateCandidate) -> tuple[Any, ...]:
    return (
        candidate.family,
        candidate.member,
        candidate.num_layers,
        candidate.embedding_lr,
        candidate.deep_lr,
    )


def _available(
    candidates: Iterable[AggregateCandidate],
    by_recipe: dict[tuple[Any, ...], AggregateRun],
) -> list[AggregateRun]:
    return [by_recipe[_recipe_key(candidate)] for candidate in candidates]


def _surface_candidate(
    run: AggregateRun, candidates: Iterable[AggregateCandidate]
) -> AggregateCandidate:
    matches = [
        candidate
        for candidate in candidates
        if _recipe_key(candidate) == _recipe_key(run.candidate)
    ]
    if len(matches) != 1:
        raise AggregateReportError(
            f"{run.candidate.run_name}: expected one approved surface candidate"
        )
    return matches[0]


def _missing(
    candidates: Iterable[AggregateCandidate],
    by_recipe: dict[tuple[Any, ...], AggregateRun],
) -> list[AggregateCandidate]:
    return [
        candidate
        for candidate in candidates
        if _recipe_key(candidate) not in by_recipe
    ]


def _select(runs: list[AggregateRun]) -> AggregateRun:
    if not runs:
        raise AggregateReportError("selection surface is empty")
    return sorted(
        runs,
        key=lambda run: (
            -run.validation_recall,
            -run.validation_ndcg,
            run.candidate.run_name,
        ),
    )[0]


def _candidate_evidence(run: AggregateRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "run": run.candidate.run_name,
        "best_epoch": run.best_epoch,
        "stopped_epoch": run.stopped_epoch,
        "validation_recall@100": run.validation_recall,
        "validation_ndcg@100": run.validation_ndcg,
        "metrics": run.metrics,
    }


def _require_unique_runs(runs: list[AggregateRun]) -> None:
    names = [run.candidate.run_name for run in runs]
    if len(names) != len(set(names)):
        raise AggregateReportError("duplicate aggregate run evidence")


def _load_run(directory: Path, candidate: AggregateCandidate) -> AggregateRun:
    metadata = _load_json(directory / "training_metadata.json")
    metrics = _load_json(directory / "final_metrics.json")
    if not isinstance(metadata, dict) or not isinstance(metrics, dict):
        raise AggregateReportError(f"{candidate.run_name}: malformed evidence")
    if metrics.get("num_users") != 37018:
        raise AggregateReportError(f"{candidate.run_name}: unexpected user count")
    final_metrics = {name: _metric(metrics, name, candidate.run_name) for name in METRICS}
    best_epoch = _positive_int(metadata.get("best_epoch"), candidate.run_name, "best_epoch")
    stopped_epoch = _positive_int(
        metadata.get("stopped_epoch"), candidate.run_name, "stopped_epoch"
    )
    validation_recall, validation_ndcg = _best_epoch_metrics(
        directory, best_epoch
    )
    return AggregateRun(
        candidate=candidate,
        best_epoch=best_epoch,
        stopped_epoch=stopped_epoch,
        validation_recall=validation_recall,
        validation_ndcg=validation_ndcg,
        metrics=final_metrics,
    )


def _best_epoch_metrics(directory: Path, best_epoch: int) -> tuple[float, float]:
    values: list[tuple[float, float]] = []
    for line in (directory / "sweep.log").read_text().splitlines():
        if re.search(rf"\bepoch {best_epoch - 1} finished\b", line) is None:
            continue
        recall = re.search(rf"\bepoch/val_true\.recall@100=({_METRIC_NUMBER})\b", line)
        ndcg = re.search(rf"\bepoch/val_true\.ndcg@100=({_METRIC_NUMBER})\b", line)
        if recall is not None and ndcg is not None:
            values.append((float(recall.group(1)), float(ndcg.group(1))))
    if len(set(values)) != 1:
        raise AggregateReportError(
            f"{directory.name}: missing or conflicting best-epoch validation metrics"
        )
    return values[0]


def _metric(metrics: dict[str, Any], name: str, context: str) -> float:
    value = metrics.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise AggregateReportError(f"{context}: invalid {name}")
    return float(value)


def _positive_int(value: Any, context: str, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AggregateReportError(f"{context}: invalid {name}")
    return value


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AggregateReportError(f"cannot read {path}") from error


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--scratchpad", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--required", choices=("all", "non-bridge", "bridge"))
    args = parser.parse_args()
    bundle = collect_report_bundle(args.logs)
    if args.required:
        names = bundle.evidence["required_followups"]
        for name in names:
            family = candidate_by_run(name).family
            if args.required == "all" or (args.required == "bridge") == (family == "bridge"):
                print(name)
        return
    if args.write:
        if args.scratchpad is None or args.evidence is None:
            parser.error("--write requires --scratchpad and --evidence")
        write_report_bundle(bundle, args.scratchpad, args.evidence)
        return
    print(json.dumps(bundle.evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
