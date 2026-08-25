from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile

from experiments.g1_sasrec_item_ids_likes.analysis.rq11_mixed_streaming_candidates import (
    PRIMARY_FAMILIES,
    Rq11Candidate,
    candidate_by_run,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq11_mixed_streaming_selection import (
    ArtifactEvidence,
    SelectionEvidenceError,
    build_followup_plan,
    classify_against_control,
    filesystem_inspector,
    select_family_winner,
)


_METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")
_LABELS = {
    "uniform_catalog": "uniform catalog",
    "streaming_global_q": "streaming in-batch global-q",
    "popularity_global_q": "popularity catalog global-q",
    "aggregate_uniform_streaming_global_q": "aggregate uniform + streaming global-q",
    "aggregate_uniform_streaming_global_q_negative_only": (
        "aggregate uniform + streaming global-q, negative-only diagnostic"
    ),
}


class Rq11ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Run:
    candidate: Rq11Candidate
    best_epoch: int
    stopped_epoch: int
    validation_recall: float
    validation_ndcg: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class Rq11ReportBundle:
    reader_markdown: str
    tuning_markdown: str
    evidence: dict[str, object]


def collect_report_bundle(logs: Path) -> Rq11ReportBundle:
    inspect = filesystem_inspector(logs)
    runs = []
    for directory in sorted(logs.iterdir()) if logs.is_dir() else ():
        if not directory.is_dir():
            continue
        try:
            candidate = candidate_by_run(directory.name)
        except ValueError:
            continue
        try:
            artifact = inspect(candidate)
            if artifact is None:
                continue
            metadata = json.loads((directory / "training_metadata.json").read_text())
            raw_metrics = json.loads((directory / "final_metrics.json").read_text())
        except (SelectionEvidenceError, OSError, json.JSONDecodeError) as error:
            raise Rq11ReportError(str(error)) from error
        try:
            metrics = {name: float(raw_metrics[name]) for name in _METRICS}
            best_epoch = int(metadata["best_epoch"])
            stopped_epoch = int(metadata["stopped_epoch"])
        except (KeyError, TypeError, ValueError) as error:
            raise Rq11ReportError(
                f"{candidate.run_name}: incomplete report artifact"
            ) from error
        runs.append(
            Run(
                candidate,
                best_epoch,
                stopped_epoch,
                artifact.validation_recall,
                artifact.validation_ndcg,
                metrics,
            )
        )
    return build_report_bundle(runs)


def build_report_bundle(runs: list[Run]) -> Rq11ReportBundle:
    by_name = {run.candidate.run_name: run for run in runs}
    if len(by_name) != len(runs):
        raise Rq11ReportError("duplicate RQ11 artifact identity")
    grouped = {
        family: [run for run in runs if run.candidate.family == family]
        for family in PRIMARY_FAMILIES
    }
    selected = {
        family: _select_runs(family_runs)
        for family, family_runs in grouped.items()
        if family_runs
    }
    artifact_evidence = {run.candidate.run_name: _as_evidence(run) for run in runs}
    try:
        followup = build_followup_plan(
            lambda candidate: artifact_evidence.get(candidate.run_name)
        )
    except SelectionEvidenceError:
        followup = None
    ready = followup is not None and followup.stage == "complete"
    comparisons: dict[str, str] = {}
    if ready:
        mixture = _as_evidence(selected["aggregate_uniform_streaming_global_q"])
        comparisons = {
            family: classify_against_control(mixture, _as_evidence(selected[family]))
            for family in PRIMARY_FAMILIES
            if family != "aggregate_uniform_streaming_global_q"
        }
    evidence: dict[str, object] = {
        "dataset_size": "500m",
        "claims_status": "ready" if ready else "pending",
        "mixture_answer": (
            aggregate_mixture_outcome(comparisons) if ready else "pending"
        ),
        "control_comparisons": comparisons,
        "selected": {
            family: {
                "run_name": run.candidate.run_name,
                "deep_lr": run.candidate.deep_lr,
                "negative_count": run.candidate.negative_count,
                "alpha": run.candidate.alpha,
                "uniform_fraction": run.candidate.uniform_fraction,
                "best_epoch": run.best_epoch,
                "metrics": run.metrics,
            }
            for family, run in selected.items()
        },
        "validated_native_artifacts": sorted(by_name),
        "legacy_artifacts_eligible": False,
    }
    return Rq11ReportBundle(
        reader_markdown=_render_reader(selected if ready else {}),
        tuning_markdown=_render_tuning(runs, selected),
        evidence=evidence,
    )


def write_report_bundle(
    bundle: Rq11ReportBundle, scratchpad: Path, evidence: Path
) -> dict[str, Path]:
    paths = {
        "reader": scratchpad / "rq11_mixed_streaming_reader_500m.md",
        "tuning": scratchpad / "rq11_mixed_streaming_tuning_500m.md",
        "evidence": evidence / "rq11_mixed_streaming_results.json",
    }
    _write(paths["reader"], bundle.reader_markdown)
    _write(paths["tuning"], bundle.tuning_markdown)
    _write(
        paths["evidence"], json.dumps(bundle.evidence, indent=2, sort_keys=True) + "\n"
    )
    return paths


def sync_readme(path: Path, reader_markdown: str) -> None:
    start = "<!-- rq11-mixed-streaming-generated:start -->"
    end = "<!-- rq11-mixed-streaming-generated:end -->"
    text = path.read_text()
    if (
        text.count(start) != 1
        or text.count(end) != 1
        or text.index(start) > text.index(end)
    ):
        raise Rq11ReportError(f"{path}: malformed RQ11 generated-table markers")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    _write(path, before + start + "\n" + reader_markdown.rstrip() + "\n" + end + after)


def _select_runs(runs: list[Run]) -> Run:
    evidence = select_family_winner(_as_evidence(run) for run in runs)
    return next(run for run in runs if run.candidate == evidence.candidate)


def _as_evidence(run: Run) -> ArtifactEvidence:
    return ArtifactEvidence(
        run.candidate,
        run.validation_recall,
        run.validation_ndcg,
        run.metrics["recall@100"],
        run.metrics["ndcg@100"],
    )


def _render_reader(selected: dict[str, Run]) -> str:
    header = (
        "| negative sampling | negatives | logQ alpha | uniform fraction | "
        "recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    if len(selected) != len(PRIMARY_FAMILIES):
        return header + "\n"
    reference = selected["uniform_catalog"]
    best_recall = max(run.metrics["recall@100"] for run in selected.values())
    rows = []
    for family in PRIMARY_FAMILIES:
        run = selected[family]
        label = _LABELS[family]
        cells = [
            label,
            str(run.candidate.negative_count),
            _optional(run.candidate.alpha),
            _optional(run.candidate.uniform_fraction),
            *(
                metric_cell(
                    name,
                    run.metrics[name],
                    reference.metrics[name],
                    family == "uniform_catalog",
                )
                for name in _METRICS
            ),
        ]
        row = "| " + " | ".join(cells) + " |"
        if run.metrics["recall@100"] == best_recall:
            row = row.replace("| ", "| **", 1).replace(" |", "** |", 1)
        rows.append(row)
    return header + "\n" + "\n".join(rows) + "\n"


def _render_tuning(runs: list[Run], selected: dict[str, Run]) -> str:
    lines = ["# RQ11 native-500M tuning", ""]
    selected_names = {run.candidate.run_name for run in selected.values()}
    for family in (
        *PRIMARY_FAMILIES,
        "aggregate_uniform_streaming_global_q_negative_only",
    ):
        family_runs = sorted(
            (run for run in runs if run.candidate.family == family),
            key=lambda run: run.candidate.run_name,
        )
        if not family_runs:
            continue
        lines.extend(
            [
                f"## {_LABELS[family]}",
                "",
                "| deep LR | negatives | logQ alpha | uniform fraction | validation recall@100 | validation ndcg@100 | recall@100 | ndcg@100 | best/stopped epoch |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in family_runs:
            cells = [
                f"{run.candidate.deep_lr:g}",
                str(run.candidate.negative_count),
                _optional(run.candidate.alpha),
                _optional(run.candidate.uniform_fraction),
                f"{run.validation_recall:.8f}",
                f"{run.validation_ndcg:.8f}",
                f"{run.metrics['recall@100']:.8f}",
                f"{run.metrics['ndcg@100']:.8f}",
                f"{run.best_epoch}/{run.stopped_epoch}",
            ]
            row = "| " + " | ".join(cells) + " |"
            if run.candidate.run_name in selected_names:
                row = "| " + " | ".join(f"**{cell}**" for cell in cells) + " |"
            lines.append(row)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def metric_cell(
    metric: str,
    value: float,
    reference: float,
    is_reference: bool = False,
) -> str:
    if is_reference:
        return f"{value:.3f}"
    percent = 100 * (value / reference - 1) if reference else 0.0
    if abs(percent) < 0.5:
        percent = 0.0
    rendered = f"{percent:+.0f}% ({value:.3f})"
    threshold = (
        0.003
        if metric.startswith(("recall@", "capped_recall@"))
        else (
            0.001
            if metric.startswith(("ndcg@", "mrr@"))
            else 0.1 if metric.startswith("coverage@") else math.inf
        )
    )
    delta = value - reference
    if delta > threshold and not math.isclose(delta, threshold, abs_tol=1e-12):
        return f'<span style="color: green">{rendered}</span>'
    if delta < -threshold and not math.isclose(delta, -threshold, abs_tol=1e-12):
        return f'<span style="color: red">{rendered}</span>'
    return rendered


def aggregate_mixture_outcome(comparisons: dict[str, str]) -> str:
    outcomes = set(comparisons.values())
    if outcomes == {"better"}:
        return "yes"
    if "worse" in outcomes:
        return "worse"
    if "trade-off" in outcomes:
        return "trade-off"
    return "unresolved"


def _optional(value: float | None) -> str:
    return "—" if value is None else f"{value:g}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--scratchpad", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--readme", type=Path)
    args = parser.parse_args()
    bundle = collect_report_bundle(args.logs)
    write_report_bundle(bundle, args.scratchpad, args.evidence)
    if args.readme is not None:
        sync_readme(args.readme, bundle.reader_markdown)


if __name__ == "__main__":
    main()
