from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile

from experiments.g1_sasrec_item_ids_likes.analysis import reporting
from experiments.g1_sasrec_item_ids_likes.analysis.rq7_reinvestigation_candidates import (
    Rq7Candidate,
    candidate_by_run,
    current_implementation_revision,
    initial_candidates,
    make_boundary_candidate,
    make_confirmation_candidate,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq7_reinvestigation_selection import (
    SelectionEvidenceError,
    filesystem_inspector,
)


_METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")
_LEARNED = (
    "learned_forward_add",
    "learned_forward_concat",
    "learned_forward_reverse_add",
    "learned_forward_reverse_concat",
    "learned_forward_add_alibi",
    "learned_forward_concat_alibi",
    "learned_forward_reverse_add_alibi",
    "learned_forward_reverse_concat_alibi",
)
_LEARNED_LABELS = {
    "learned_forward_add": "forward additive",
    "learned_forward_concat": "forward concatenated to item",
    "learned_forward_reverse_add": "forward + reverse additive",
    "learned_forward_reverse_concat": "forward + reverse concatenated to item",
    "learned_forward_add_alibi": "ALiBi + forward additive",
    "learned_forward_concat_alibi": "ALiBi + forward concatenated to item",
    "learned_forward_reverse_add_alibi": "ALiBi + forward + reverse additive",
    "learned_forward_reverse_concat_alibi": (
        "ALiBi + forward + reverse concatenated to item"
    ),
}
_LEARNED_REFERENCES = {
    treatment: (
        "learned_forward_add_alibi"
        if treatment.endswith("_alibi")
        else "learned_forward_add"
    )
    for treatment in _LEARNED
}


class Rq7ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Run:
    candidate: Rq7Candidate
    best_epoch: int
    stopped_epoch: int
    validation_recall: float
    validation_ndcg: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class Resolution:
    selected: Run | None
    status: str


@dataclass(frozen=True)
class Rq7ReportBundle:
    reader_markdown: str
    tuning_markdown: str
    evidence: dict[str, object]


def collect_report_bundle(logs: Path) -> Rq7ReportBundle:
    inspect = filesystem_inspector(logs)
    runs: list[Run] = []
    for directory in sorted(logs.iterdir()) if logs.is_dir() else ():
        if not directory.is_dir():
            continue
        try:
            candidate = candidate_by_run(directory.name)
        except ValueError:
            continue
        if not _eligible(candidate):
            continue
        try:
            evidence = inspect(candidate)
        except SelectionEvidenceError as error:
            raise Rq7ReportError(str(error)) from error
        if evidence is None:
            continue
        try:
            raw_metrics = json.loads((directory / "final_metrics.json").read_text())
            metadata = json.loads((directory / "training_metadata.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise Rq7ReportError(
                f"{candidate.run_name}: unreadable artifact"
            ) from error
        metrics = {
            name: _metric(raw_metrics, name, candidate.run_name) for name in _METRICS
        }
        runs.append(
            Run(
                candidate,
                int(metadata["best_epoch"]),
                int(metadata["stopped_epoch"]),
                evidence.validation_recall,
                evidence.validation_ndcg,
                metrics,
            )
        )
    return build_report_bundle(runs)


def build_report_bundle(runs: list[Run]) -> Rq7ReportBundle:
    eligible = [run for run in runs if _eligible(run.candidate)]
    by_name = {run.candidate.run_name: run for run in eligible}
    if len(by_name) != len(eligible):
        raise Rq7ReportError("duplicate RQ7 artifact identity")
    resolutions = {
        treatment: _resolve_surface(treatment, by_name)
        for treatment in {candidate.treatment for candidate in initial_candidates()}
    }
    plain = _select_plain_rope(resolutions, by_name)
    displayed = {
        **{treatment: resolutions[treatment] for treatment in _LEARNED},
        "none": resolutions["none"],
        "alibi": resolutions["alibi"],
        "plain_rope": plain,
        "rope_forward_base10000_alibi": resolutions["rope_forward_base10000_alibi"],
    }
    selected_names = {
        resolution.selected.candidate.run_name
        for resolution in displayed.values()
        if resolution.selected is not None
    }
    confirmations_required = _confirmations_required(plain, resolutions["alibi"])
    confirmation_status = {
        treatment: _confirmation_status(displayed[treatment], by_name)
        for treatment in (
            "alibi",
            "plain_rope",
            "rope_forward_base10000_alibi",
        )
    }
    learned_ready = all(
        displayed[treatment].status == "ready" for treatment in _LEARNED
    )
    rope_ready = all(
        displayed[treatment].status == "ready"
        for treatment in (
            "none",
            "alibi",
            "plain_rope",
            "rope_forward_base10000_alibi",
        )
    )
    confirmations_ready = not confirmations_required or all(
        status == "ready" for status in confirmation_status.values()
    )
    evidence = {
        "dataset_size": "500m",
        "eligible_native_runs": len(eligible),
        "claims_status": (
            "ready"
            if learned_ready and rope_ready and confirmations_ready
            else "pending"
        ),
        "learned_claims_status": "ready" if learned_ready else "pending",
        "rope_claims_status": (
            "ready" if rope_ready and confirmations_ready else "pending"
        ),
        "comparability_confirmations": {
            "required": confirmations_required,
            "status": confirmation_status,
        },
        "treatments": {
            treatment: _treatment_evidence(
                treatment,
                resolution,
                by_name,
                confirmations_required=confirmations_required,
            )
            for treatment, resolution in displayed.items()
        },
        "validated_artifacts": [run.candidate.run_name for run in eligible],
    }
    return Rq7ReportBundle(
        reader_markdown=_render_reader(
            displayed, by_name, confirmations_required=confirmations_required
        ),
        tuning_markdown=_render_tuning(eligible, selected_names),
        evidence=evidence,
    )


def write_report_bundle(
    bundle: Rq7ReportBundle, scratchpad: Path, evidence: Path
) -> dict[str, Path]:
    paths = {
        "tuning": scratchpad / "rq7_reinvestigation_tuning_500m.md",
        "reader": scratchpad / "rq7_reinvestigation_reader_500m.md",
        "evidence": evidence / "rq7_reinvestigation_results.json",
    }
    _write(paths["tuning"], bundle.tuning_markdown)
    _write(paths["reader"], bundle.reader_markdown)
    _write(
        paths["evidence"], json.dumps(bundle.evidence, indent=2, sort_keys=True) + "\n"
    )
    return paths


def sync_readme(path: Path, reader_markdown: str) -> None:
    start = "<!-- rq7-reinvestigation-generated:start -->"
    end = "<!-- rq7-reinvestigation-generated:end -->"
    text = path.read_text()
    if (
        text.count(start) != 1
        or text.count(end) != 1
        or text.index(start) > text.index(end)
    ):
        raise Rq7ReportError(f"{path}: malformed RQ7 generated-table markers")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    path.write_text(
        before + start + "\n" + reader_markdown.rstrip() + "\n" + end + after
    )


def _eligible(candidate: Rq7Candidate) -> bool:
    return (
        candidate.dataset_size == "500m"
        and candidate.implementation_revision
        == current_implementation_revision(candidate.treatment)
    )


def _resolve_surface(treatment: str, runs: dict[str, Run]) -> Resolution:
    initial = [
        candidate
        for candidate in initial_candidates()
        if candidate.treatment == treatment
    ]
    loaded = [runs.get(candidate.run_name) for candidate in initial]
    if any(run is None for run in loaded):
        return Resolution(None, "pending native-500M LR surface")
    surface = [run for run in loaded if run is not None]
    for _ in range(16):
        winner = _select(surface)
        rates = sorted(run.candidate.deep_lr for run in surface)
        if winner.candidate.deep_lr not in (rates[0], rates[-1]):
            return Resolution(winner, "ready")
        side = "low" if winner.candidate.deep_lr == rates[0] else "high"
        steps = [
            run.candidate.boundary_step or 0
            for run in surface
            if run.candidate.stage == "boundary" and run.candidate.boundary_side == side
        ]
        probe = make_boundary_candidate(
            winner.candidate, side, max(steps, default=0) + 1
        )
        continuation = runs.get(probe.run_name)
        if continuation is None:
            return Resolution(None, "boundary pending")
        surface.append(continuation)
    raise Rq7ReportError(f"{treatment}: LR boundary did not resolve")


def _select_plain_rope(
    primary: dict[str, Resolution], runs: dict[str, Run]
) -> Resolution:
    base = primary["rope_forward_base10000"]
    alibi = primary["alibi"]
    if base.selected is None or alibi.selected is None:
        return base
    if not _materially_worse(base.selected, alibi.selected):
        return base
    alternatives = [base.selected]
    for treatment in ("rope_forward_base100", "rope_forward_base1000"):
        resolution = _resolve_optional_surface(treatment, runs, "rope_base")
        if resolution.selected is None:
            return resolution
        alternatives.append(resolution.selected)
    axis_winner = _select(alternatives)
    outer = {
        "rope_forward_base100": (
            "rope_forward_base10",
            "rope_base_extension",
        ),
        "rope_forward_base10000": (
            "rope_forward_base100000",
            "rope_base_extension",
        ),
    }.get(axis_winner.candidate.treatment)
    if outer is not None:
        resolution = _resolve_optional_surface(outer[0], runs, outer[1])
        if resolution.selected is None:
            return resolution
        alternatives.append(resolution.selected)
    return Resolution(_select(alternatives), "ready")


def _resolve_optional_surface(
    treatment: str, runs: dict[str, Run], initial_stage: str
) -> Resolution:
    candidates = [run for run in runs.values() if run.candidate.treatment == treatment]
    if not candidates:
        return Resolution(None, "native-500M RoPE-base surface pending")
    initial = [run for run in candidates if run.candidate.stage == initial_stage]
    if len(initial) != 3:
        return Resolution(None, "pending native-500M LR surface")
    surface = list(initial)
    for _ in range(16):
        winner = _select(surface)
        rates = sorted(run.candidate.deep_lr for run in surface)
        if winner.candidate.deep_lr not in (rates[0], rates[-1]):
            return Resolution(winner, "ready")
        side = "low" if winner.candidate.deep_lr == rates[0] else "high"
        steps = [
            run.candidate.boundary_step or 0
            for run in surface
            if run.candidate.stage == "boundary" and run.candidate.boundary_side == side
        ]
        probe = make_boundary_candidate(
            winner.candidate, side, max(steps, default=0) + 1
        )
        continuation = runs.get(probe.run_name)
        if continuation is None:
            return Resolution(None, "boundary pending")
        surface.append(continuation)
    raise Rq7ReportError(f"{treatment}: LR boundary did not resolve")


def _select(runs: list[Run]) -> Run:
    ordered = sorted(
        runs, key=lambda run: (run.validation_recall, run.validation_ndcg), reverse=True
    )
    if len(ordered) > 1 and (
        ordered[0].validation_recall,
        ordered[0].validation_ndcg,
    ) == (ordered[1].validation_recall, ordered[1].validation_ndcg):
        raise Rq7ReportError(f"{ordered[0].candidate.treatment}: exact validation tie")
    return ordered[0]


def _display_metrics(
    resolution: Resolution, runs: dict[str, Run], confirmations: bool
) -> tuple[dict[str, float] | None, str]:
    if resolution.selected is None:
        return None, resolution.status
    winner = resolution.selected
    if not confirmations:
        return winner.metrics, "selected seed 42"
    repeats = [winner]
    for seed in (43, 44):
        candidate = make_confirmation_candidate(winner.candidate, seed)
        repeat = runs.get(candidate.run_name)
        if repeat is None:
            return winner.metrics, "confirmations pending"
        repeats.append(repeat)
    return (
        {
            metric: sum(run.metrics[metric] for run in repeats) / 3
            for metric in _METRICS
        },
        "3-seed mean",
    )


def _confirmation_status(resolution: Resolution, runs: dict[str, Run]) -> str:
    if resolution.selected is None:
        return resolution.status
    for seed in (43, 44):
        candidate = make_confirmation_candidate(resolution.selected.candidate, seed)
        if candidate.run_name not in runs:
            return "pending"
    return "ready"


def _confirmations_required(plain: Resolution, alibi: Resolution) -> bool:
    if plain.selected is None or alibi.selected is None:
        return False
    return not _materially_better(plain.selected, alibi.selected)


def _materially_worse(candidate: Run, reference: Run) -> bool:
    recall_delta = candidate.metrics["recall@100"] - reference.metrics["recall@100"]
    if recall_delta < -0.003 and not math.isclose(recall_delta, -0.003, abs_tol=1e-12):
        return True
    return recall_delta <= 0.003 and (
        candidate.metrics["ndcg@100"] - reference.metrics["ndcg@100"] < -0.001
        and not math.isclose(
            candidate.metrics["ndcg@100"] - reference.metrics["ndcg@100"],
            -0.001,
            abs_tol=1e-12,
        )
    )


def _materially_better(candidate: Run, reference: Run) -> bool:
    recall_delta = candidate.metrics["recall@100"] - reference.metrics["recall@100"]
    if recall_delta > 0.003 and not math.isclose(recall_delta, 0.003, abs_tol=1e-12):
        return True
    return recall_delta >= -0.003 and (
        candidate.metrics["ndcg@100"] - reference.metrics["ndcg@100"] > 0.001
        and not math.isclose(
            candidate.metrics["ndcg@100"] - reference.metrics["ndcg@100"],
            0.001,
            abs_tol=1e-12,
        )
    )


def _render_reader(
    displayed: dict[str, Resolution],
    runs: dict[str, Run],
    *,
    confirmations_required: bool,
) -> str:
    lines = ["### Learned-position fusion comparisons", ""]
    learned_metrics = {
        treatment: _display_metrics(displayed[treatment], runs, False)
        for treatment in _LEARNED
    }
    lines += _comparison_table(
        "learned-position treatment",
        _LEARNED,
        _LEARNED_LABELS,
        learned_metrics,
        _LEARNED_REFERENCES,
    )
    lines += ["", "### RoPE / ALiBi comparison", ""]
    rope_order = ("none", "alibi", "plain_rope", "rope_forward_base10000_alibi")
    plain = displayed["plain_rope"].selected
    plain_label = "plain forward RoPE"
    if plain is not None:
        plain_label += f" (base {plain.candidate.position.rope_base:g})"
    rope_labels = {
        "none": "no position encoding",
        "alibi": "ALiBi",
        "plain_rope": plain_label,
        "rope_forward_base10000_alibi": "forward RoPE + ALiBi",
    }
    rope_metrics = {
        treatment: _display_metrics(
            displayed[treatment],
            runs,
            confirmations_required
            and treatment in {"alibi", "plain_rope", "rope_forward_base10000_alibi"},
        )
        for treatment in rope_order
    }
    lines += _comparison_table(
        "RoPE / ALiBi treatment",
        rope_order,
        rope_labels,
        rope_metrics,
        {treatment: "alibi" for treatment in rope_order},
    )
    return "\n".join(lines) + "\n"


def _comparison_table(
    axis: str,
    order: tuple[str, ...],
    labels: dict[str, str],
    values: dict[str, tuple[dict[str, float] | None, str]],
    references: dict[str, str],
) -> list[str]:
    lines = [
        f"| {axis} | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ready = {
        name: metrics for name, (metrics, _) in values.items() if metrics is not None
    }
    best = max(ready, key=lambda name: ready[name]["recall@100"]) if ready else None
    for treatment in order:
        metrics, _ = values[treatment]
        if metrics is None:
            continue
        label = labels[treatment]
        if treatment == best:
            label = f"**{label}**"
        reference = values.get(references[treatment], (None, ""))[0]
        cells = [label]
        for metric in _METRICS:
            cells.append(
                reporting.absolute(metrics[metric])
                if reference is None or treatment == references[treatment]
                else reporting.change_cell(
                    metrics[metric], reference[metric], metric
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _render_tuning(runs: list[Run], selected: set[str]) -> str:
    lines = [
        "# G1 RQ7 — native Yambda-500M tuning ledger",
        "",
        "Selection uses best-epoch validation recall@100, then same-epoch NDCG@100.",
        "",
        "| treatment | embedding LR | deep LR | batch size | best/stopped epoch | validation recall@100 | validation ndcg@100 | full-user recall@100 | full-user ndcg@100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in sorted(
        runs,
        key=lambda item: (
            item.candidate.treatment,
            item.candidate.seed,
            item.candidate.deep_lr,
        ),
    ):
        if run.candidate.stage == "confirmation":
            continue
        chosen = run.candidate.run_name in selected
        values = [
            _tuning_label(run.candidate),
            f"{run.candidate.embedding_lr:.3f}",
            f"{run.candidate.deep_lr:.6g}",
            str(run.candidate.batch_size),
            f"{run.best_epoch}/{run.stopped_epoch}",
            f"{run.validation_recall:.6f}",
            f"{run.validation_ndcg:.6f}",
            f"{run.metrics['recall@100']:.6f}",
            f"{run.metrics['ndcg@100']:.6f}",
        ]
        if chosen:
            values[1] = f"**{values[1]}**"
            values[2] = f"**{values[2]}**"
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _tuning_label(candidate: Rq7Candidate) -> str:
    if candidate.treatment in _LEARNED_LABELS:
        return _LEARNED_LABELS[candidate.treatment]
    if candidate.treatment == "none":
        return "no position encoding"
    if candidate.treatment == "alibi":
        return "ALiBi"
    label = f"forward RoPE, base {candidate.position.rope_base:g}"
    return f"{label} + ALiBi" if candidate.position.alibi else label


def _resolution_evidence(resolution: Resolution) -> dict[str, object]:
    selected = resolution.selected
    return {
        "status": resolution.status,
        "selected_run": None if selected is None else selected.candidate.run_name,
        "selected_deep_lr": None if selected is None else selected.candidate.deep_lr,
        "selected_best_epoch": None if selected is None else selected.best_epoch,
        "selected_stopped_epoch": None if selected is None else selected.stopped_epoch,
        "selected_validation_recall@100": (
            None if selected is None else selected.validation_recall
        ),
        "selected_validation_ndcg@100": (
            None if selected is None else selected.validation_ndcg
        ),
    }


def _treatment_evidence(
    treatment: str,
    resolution: Resolution,
    runs: dict[str, Run],
    *,
    confirmations_required: bool,
) -> dict[str, object]:
    repeated = confirmations_required and treatment in {
        "alibi",
        "plain_rope",
        "rope_forward_base10000_alibi",
    }
    metrics, reader_evidence = _display_metrics(resolution, runs, repeated)
    return {
        **_resolution_evidence(resolution),
        "reader_evidence": reader_evidence,
        "reader_metrics": metrics,
    }


def _metric(metrics: object, name: str, context: str) -> float:
    if not isinstance(metrics, dict):
        raise Rq7ReportError(f"{context}: invalid final metrics")
    value = metrics.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise Rq7ReportError(f"{context}: invalid final metric {name}")
    return float(value)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("generated/logs"))
    parser.add_argument(
        "--scratchpad",
        type=Path,
        default=Path("experiments/g1_sasrec_item_ids_likes/scratchpad"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("experiments/g1_sasrec_item_ids_likes/evidence"),
    )
    parser.add_argument("--readme", type=Path)
    arguments = parser.parse_args()
    bundle = collect_report_bundle(arguments.logs)
    paths = write_report_bundle(bundle, arguments.scratchpad, arguments.evidence)
    research_questions = arguments.scratchpad / "research_questions_500m.md"
    if research_questions.exists():
        sync_readme(research_questions, bundle.reader_markdown)
    if arguments.readme is not None:
        sync_readme(arguments.readme, bundle.reader_markdown)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
