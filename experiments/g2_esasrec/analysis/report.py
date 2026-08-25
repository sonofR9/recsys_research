from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.g2_esasrec.analysis.evidence import VerifiedArtifact, select_best
from experiments.g2_esasrec.configs.local import (
    COMPONENT_METHODS,
    LIGR_WIDTHS,
    MATCHED_STANDARD_WIDTHS,
)


def _method_name(artifact: VerifiedArtifact) -> str:
    return artifact.job.method.replace("_", " ")


def _value(value: float) -> str:
    return f"{value:.3f}"


def _percent(value: float, reference: float) -> str:
    if reference == 0:
        return "—"
    return f"{(value / reference - 1) * 100:+.3f}%"


def _signed_value(value: float) -> str:
    return f"{value:+.3f}"


def _cost(artifact: VerifiedArtifact, name: str) -> str:
    value = artifact.costs.get(name)
    return "—" if value is None else f"{value:.3f}"


def _reader_parameter(artifact: VerifiedArtifact, name: str) -> str:
    value = artifact.parameters.get(name)
    return "—" if value is None else _value(float(value))


def _treatment_fields(artifact: VerifiedArtifact) -> tuple[str, ...]:
    parameters = artifact.parameters
    method = str(parameters.get("method", artifact.job.method))
    if method == "control":
        layer, loss, width = "G1 control", "sampled softmax", "171"
    elif method == "official_rectools":
        layer, loss, width = "RecTools LiGR", "sampled softmax", "—"
    elif method == "mixed_sampler":
        layer, loss = "LiGR", "sampled softmax"
        width = str(LIGR_WIDTHS[int(parameters["ligr_multiplier"])])
    elif method.startswith("matched_standard_"):
        layer = "parameter-matched SASRec"
        loss = "gBCE" if method.endswith("_gbce") else "sampled softmax"
        width = str(MATCHED_STANDARD_WIDTHS[int(parameters["ligr_multiplier"])])
    elif method.startswith("standard_"):
        layer = "official SASRec block"
        loss = "gBCE" if method.endswith("_gbce") else "sampled softmax"
        width = "256"
    elif method.startswith("ligr_"):
        layer = "LiGR"
        loss = "gBCE" if method.endswith("_gbce") else "sampled softmax"
        width = str(LIGR_WIDTHS[int(parameters["ligr_multiplier"])])
    else:
        layer, loss, width = method.replace("_", " "), "—", "—"
    return (
        layer,
        loss,
        width,
        str(parameters.get("gbce_t", "—")),
        str(parameters.get("uniform_fraction", "—")),
        str(parameters.get("logq_correction", "—")),
    )


def _metric_cell(
    value: float,
    reference: float,
    band: float | None,
) -> str:
    cell = f"{_percent(value, reference)} ({_value(value)})"
    if band is None or abs(value - reference) <= band:
        return cell
    color = "green" if value > reference else "red"
    return f'<span style="color: {color}">{cell}</span>'


def _is_rq2(heading: str) -> bool:
    return heading == "RQ2" or heading.startswith("RQ2:")


def aggregate_section_heading(
    baseline: VerifiedArtifact, selected: VerifiedArtifact
) -> str:
    return "Aggregated improvement"


def _recipe_label(method: str) -> str:
    labels = {
        "control": "recalibrated G1 control",
        "standard_sampled_softmax": "official SASRec block with sampled softmax",
        "standard_gbce": "official SASRec block with gBCE",
        "matched_standard_sampled_softmax": (
            "parameter-matched SASRec with sampled softmax"
        ),
        "matched_standard_gbce": "parameter-matched SASRec with gBCE",
        "ligr_sampled_softmax": "LiGR with sampled softmax",
        "ligr_gbce": "LiGR with gBCE",
        "mixed_sampler": "LiGR with mixed sampling",
    }
    return labels.get(method, method.replace("_", " "))


def _qualification_text(reason: object) -> str:
    if reason == "recall_gain_beyond_band":
        return "qualified because Recall@100 improved beyond its size-matched band"
    if reason == "recall_tie_ndcg_gain_on_quality_cost_pareto_frontier":
        return (
            "qualified because Recall@100 was non-inferior, NDCG@100 improved "
            "beyond its band, and wall time stayed on the quality/cost Pareto frontier"
        )
    return "did not qualify for promotion"


def _rq2_comparison_table(
    *,
    axis: str,
    context: str,
    reference: VerifiedArtifact,
    treatment: VerifiedArtifact,
    metric_bands: Mapping[str, float] | None,
) -> list[str]:
    reference_fields = _treatment_fields(reference)
    treatment_fields = _treatment_fields(treatment)
    field_index = {"block": 0, "loss": 1, "FFN width": 2}[axis]
    extra_columns: list[tuple[str, str, str]]
    if axis == "loss":
        extra_columns = [
            (
                "gBCE t",
                _reader_parameter(reference, "gbce_t"),
                _reader_parameter(treatment, "gbce_t"),
            ),
            (
                "wall s",
                _cost(reference, "wall_seconds"),
                _cost(treatment, "wall_seconds"),
            ),
            (
                "targets/s",
                _cost(reference, "targets_per_second"),
                _cost(treatment, "targets_per_second"),
            ),
            (
                "peak GB",
                _cost(reference, "peak_memory_gb"),
                _cost(treatment, "peak_memory_gb"),
            ),
        ]
    elif axis == "FFN width":
        extra_columns = [
            (
                "params",
                _cost(reference, "params_total"),
                _cost(treatment, "params_total"),
            ),
            (
                "wall s",
                _cost(reference, "wall_seconds"),
                _cost(treatment, "wall_seconds"),
            ),
            (
                "targets/s",
                _cost(reference, "targets_per_second"),
                _cost(treatment, "targets_per_second"),
            ),
            (
                "peak GB",
                _cost(reference, "peak_memory_gb"),
                _cost(treatment, "peak_memory_gb"),
            ),
        ]
    else:
        extra_columns = [
            ("FFN width", reference_fields[2], treatment_fields[2]),
            (
                "params",
                _cost(reference, "params_total"),
                _cost(treatment, "params_total"),
            ),
            (
                "wall s",
                _cost(reference, "wall_seconds"),
                _cost(treatment, "wall_seconds"),
            ),
            (
                "targets/s",
                _cost(reference, "targets_per_second"),
                _cost(treatment, "targets_per_second"),
            ),
            (
                "peak GB",
                _cost(reference, "peak_memory_gb"),
                _cost(treatment, "peak_memory_gb"),
            ),
        ]

    first_column = f"{axis} ({context})"
    lines = [
        f"| {first_column} | recall@100 | ndcg@100 | coverage@100 | "
        + " | ".join(name for name, _, _ in extra_columns)
        + " |",
        "| :--- | :---: | :---: | :---: | "
        + " | ".join("---:" for _ in extra_columns)
        + " |",
    ]
    best = max((reference, treatment), key=lambda row: row.metrics["recall@100"])
    for artifact, fields, is_reference in (
        (reference, reference_fields, True),
        (treatment, treatment_fields, False),
    ):
        label = fields[field_index]
        if artifact == best:
            label = f"**{label}**"
        metrics = [
            (
                _value(artifact.metrics[metric])
                if is_reference
                else _metric_cell(
                    artifact.metrics[metric],
                    reference.metrics[metric],
                    None if metric_bands is None else metric_bands[metric],
                )
            )
            for metric in ("recall@100", "ndcg@100", "coverage@100")
        ]
        extra_values = [
            reference_value if is_reference else treatment_value
            for _, reference_value, treatment_value in extra_columns
        ]
        lines.append(f"| {label} | {' | '.join([*metrics, *extra_values])} |")
    return lines


def _rq2_comparison_tables(
    rows: Iterable[VerifiedArtifact],
    metric_bands: Mapping[str, float] | None,
) -> list[str]:
    by_method = {row.job.method: row for row in rows}
    comparisons = (
        (
            "loss",
            "standard block, FFN width 256",
            "standard_sampled_softmax",
            "standard_gbce",
        ),
        (
            "loss",
            "parameter-matched SASRec, FFN width 1792",
            "matched_standard_sampled_softmax",
            "matched_standard_gbce",
        ),
        (
            "loss",
            "LiGR, FFN width 1024",
            "ligr_sampled_softmax",
            "ligr_gbce",
        ),
        (
            "FFN width",
            "standard block, sampled softmax",
            "standard_sampled_softmax",
            "matched_standard_sampled_softmax",
        ),
        (
            "FFN width",
            "standard block, gBCE",
            "standard_gbce",
            "matched_standard_gbce",
        ),
        (
            "block",
            "sampled softmax, parameter-matched",
            "matched_standard_sampled_softmax",
            "ligr_sampled_softmax",
        ),
        (
            "block",
            "gBCE, parameter-matched",
            "matched_standard_gbce",
            "ligr_gbce",
        ),
    )
    lines: list[str] = []
    for axis, context, reference_method, treatment_method in comparisons:
        if lines:
            lines.append("")
        lines.extend(
            _rq2_comparison_table(
                axis=axis,
                context=context,
                reference=by_method[reference_method],
                treatment=by_method[treatment_method],
                metric_bands=metric_bands,
            )
        )
    return lines


def _composition_summary(
    document: Mapping[str, Any], benchmark: Mapping[str, Any]
) -> list[str]:
    included = document.get("included_bundle")
    omissions = document.get("omissions")
    candidates = document.get("candidates")
    baseline_fallback = document.get("baseline_fallback")
    metrics = document.get("metrics")
    if (
        not isinstance(included, Mapping)
        or not isinstance(included.get("run_name"), str)
        or not isinstance(omissions, list)
        or any(not isinstance(row, Mapping) for row in omissions)
        or not isinstance(candidates, list)
        or any(not isinstance(row, Mapping) for row in candidates)
        or not isinstance(baseline_fallback, Mapping)
        or not isinstance(metrics, Mapping)
    ):
        raise ValueError("aggregate composition evidence is incomplete")
    lines = [
        "| candidate | qualification | selection | rationale |",
        "| :--- | :--- | :--- | :--- |",
    ]
    if baseline_fallback.get("selected") is True:
        lines.append(
            "| recalibrated G1 control | retained | selected | "
            "no atomic bundle qualified for promotion |"
        )
    for row in candidates:
        label = _recipe_label(str(row.get("method")))
        text = _qualification_text(row.get("qualification_reason"))
        qualification = "qualified" if row.get("qualified") else "not qualified"
        selection = "selected" if row.get("selected") else "omitted"
        lines.append(f"| {label} | {qualification} | {selection} | {text} |")
    if any(row.get("reason") == "no_eligible_mixed_winner" for row in omissions):
        lines.append(
            "| LiGR with mixed sampling | not qualified | omitted | "
            "no candidate met the pre-approved mixed-sampler eligibility rule |"
        )
    p50 = float(benchmark["latency_p50_seconds"]) * 1000
    p95 = float(benchmark["latency_p95_seconds"]) * 1000
    throughput = float(benchmark["queries_per_second"])
    lines.extend(
        [
            "",
            "| deterministic selected-recipe reproduction | p50 latency (ms) | "
            "p95 latency (ms) | queries/s |",
            "| :--- | ---: | ---: | ---: |",
            f"| {_recipe_label(str(included['method']))} | {p50:.3f} | "
            f"{p95:.3f} | {throughput:.3f} |",
            "",
            "| metric | baseline | aggregate | point gain | percent gain | "
            "standalone sum | interaction gap | size-matched band | resolution |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
        ]
    )
    for metric in ("recall@100", "ndcg@100", "coverage@100"):
        row = metrics.get(metric)
        if not isinstance(row, Mapping):
            raise ValueError(f"aggregate composition {metric} is missing")
        percent = row.get("aggregate_gain_percent")
        percent_cell = "—" if percent is None else f"{float(percent):+.3f}%"
        lines.append(
            f"| {metric} | {_value(float(row['baseline']))} | "
            f"{_value(float(row['aggregate']))} | "
            f"{_signed_value(float(row['aggregate_gain_points']))} | "
            f"{percent_cell} | "
            f"{_signed_value(float(row['standalone_sum_points']))} | "
            f"{_signed_value(float(row['interaction_gap_points']))} | "
            f"{_value(float(row['interaction_band']))} | "
            f"{row['interaction_label']} |"
        )
    return lines


def render_reversal_confirmation_report(document: Mapping[str, Any]) -> str:
    rows = document.get("confirmations")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("reversal confirmation report requires four rows")
    lines = [
        "# Reversal confirmation review",
        "",
        "Explicit user validation is required before final aggregate selection.",
        "",
        "| source recipe | seed | recall@100 | ndcg@100 | coverage@100 |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("metrics"), Mapping):
            raise ValueError("reversal confirmation report row is invalid")
        metrics = row["metrics"]
        lines.append(
            f"| {_recipe_label(str(row.get('source_method')))} | {row.get('seed')} | "
            f"{_value(float(metrics['recall@100']))} | "
            f"{_value(float(metrics['ndcg@100']))} | "
            f"{_value(float(metrics['coverage@100']))} |"
        )
    return "\n".join(lines) + "\n"


def persist_reversal_confirmation_report(
    document: Mapping[str, Any], destination: Path
) -> None:
    rendered = render_reversal_confirmation_report(document)
    if destination.exists():
        if destination.read_text() != rendered:
            raise ValueError("reversal confirmation report changed")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(destination)


def render_tuning_ledger(
    artifacts: Iterable[VerifiedArtifact],
    *,
    metric_bands: Mapping[str, float] | None = None,
) -> str:
    grouped: dict[tuple[str, str], list[VerifiedArtifact]] = defaultdict(list)
    for artifact in artifacts:
        if artifact.job.stage not in {
            "control_tuning",
            "component_tuning",
            "mixed_tuning",
            "lr_boundary",
        }:
            continue
        if artifact.job.stage == "lr_boundary":
            rq = "RQ1" if artifact.parameters["builder"] == "control" else "RQ2"
        else:
            rq = {
                "control_tuning": "RQ1",
                "component_tuning": "RQ2",
                "mixed_tuning": "RQ3",
            }.get(artifact.job.stage, "RQ1")
        grouped[(rq, artifact.job.method)].append(artifact)
    lines: list[str] = []
    for (rq, method), rows in sorted(grouped.items()):
        winner = select_best(rows, metric_bands=metric_bands)
        lines.extend(
            [
                f"## {rq}",
                "",
                f"### {method.replace('_', ' ')}",
                "",
                "| run | layer | loss | FFN width | gBCE t | uniform | logQ | "
                "embedding LR | deep LR | batch | params | epoch s | train s | "
                "wall s | targets/s | peak GB | recall@100 | ndcg@100 | "
                "coverage@100 |",
                "| :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: | "
                "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: |",
            ]
        )
        for artifact in sorted(rows, key=lambda row: row.job.trial or 0):
            parameters = artifact.parameters
            layer, loss, width, gbce_t, uniform, logq = _treatment_fields(artifact)
            recall = _value(artifact.metrics["recall@100"])
            if artifact == winner:
                recall = f"**{recall}**"
            lines.append(
                "| "
                + " | ".join(
                    [
                        artifact.job.run_name,
                        layer,
                        loss,
                        width,
                        gbce_t,
                        uniform,
                        logq,
                        str(parameters.get("embedding_learning_rate", "—")),
                        str(parameters.get("deep_learning_rate", "—")),
                        str(parameters.get("batch_size", "—")),
                        _cost(artifact, "params_total"),
                        _cost(artifact, "median_train_epoch_seconds"),
                        _cost(artifact, "training_seconds"),
                        _cost(artifact, "wall_seconds"),
                        _cost(artifact, "targets_per_second"),
                        _cost(artifact, "peak_memory_gb"),
                        recall,
                        _value(artifact.metrics["ndcg@100"]),
                        _value(artifact.metrics["coverage@100"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_compact_report(
    research_questions: Mapping[str, Iterable[VerifiedArtifact]],
    *,
    reference: VerifiedArtifact,
    references: Mapping[str, VerifiedArtifact] | None = None,
    metric_bands: Mapping[str, float] | None = None,
    composition_evidence: Mapping[str, Any] | None = None,
    benchmark_evidence: Mapping[str, Any] | None = None,
) -> str:
    lines: list[str] = ["# G2 eSASRec on native Yambda-50M", ""]
    for heading, artifact_rows in research_questions.items():
        rows = list(artifact_rows)
        if not rows:
            raise ValueError(f"{heading} has no verified result rows")
        rq2 = _is_rq2(heading)
        if rq2 and (
            len(rows) != len(COMPONENT_METHODS)
            or {row.job.method for row in rows} != set(COMPONENT_METHODS)
        ):
            raise ValueError("RQ2 requires exactly its six component winners")
        question_reference = None
        if not rq2:
            question_reference = (references or {}).get(heading, reference)
        lines.extend([f"## {heading}", ""])
        if rq2:
            lines.extend(_rq2_comparison_tables(rows, metric_bands))
            lines.append("")
            continue
        header = (
            "| variant | layer | loss | FFN width | gBCE t | uniform | logQ | "
            "params | epoch s | train s | wall s | targets/s | peak GB |"
        )
        alignment = (
            "| :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: | "
            "---: | ---: | ---: | ---: | ---: |"
        )
        if not rq2:
            header = header[:-1] + "| recall@100 | ndcg@100 | coverage@100 |"
            alignment = alignment[:-1] + "| :---: | :---: | ---: |"
        lines.extend([header, alignment])
        for artifact in rows:
            layer, loss, width, gbce_t, uniform, logq = _treatment_fields(artifact)
            cells = (
                []
                if rq2
                else [
                    _metric_cell(
                        artifact.metrics[metric],
                        question_reference.metrics[metric],
                        None if metric_bands is None else metric_bands[metric],
                    )
                    for metric in ("recall@100", "ndcg@100", "coverage@100")
                ]
            )
            values = [
                _method_name(artifact),
                layer,
                loss,
                width,
                gbce_t,
                uniform,
                logq,
                _cost(artifact, "params_total"),
                _cost(artifact, "median_train_epoch_seconds"),
                _cost(artifact, "training_seconds"),
                _cost(artifact, "wall_seconds"),
                _cost(artifact, "targets_per_second"),
                _cost(artifact, "peak_memory_gb"),
            ]
            if not rq2:
                values.extend(cells)
            lines.append(f"| {' | '.join(values)} |")
        if heading == "Aggregated improvement":
            if composition_evidence is None or benchmark_evidence is None:
                raise ValueError("Aggregated improvement requires composition evidence")
            lines.extend(
                ["", *_composition_summary(composition_evidence, benchmark_evidence)]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    artifacts: Iterable[VerifiedArtifact],
    research_questions: Mapping[str, Iterable[VerifiedArtifact]],
    reference: VerifiedArtifact,
    *,
    ledger_path: Path,
    compact_path: Path,
    references: Mapping[str, VerifiedArtifact] | None = None,
    metric_bands: Mapping[str, float] | None = None,
    composition_evidence: Mapping[str, Any] | None = None,
    benchmark_evidence: Mapping[str, Any] | None = None,
) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(render_tuning_ledger(artifacts, metric_bands=metric_bands))
    compact_path.write_text(
        render_compact_report(
            research_questions,
            reference=reference,
            references=references,
            metric_bands=metric_bands,
            composition_evidence=composition_evidence,
            benchmark_evidence=benchmark_evidence,
        )
    )
