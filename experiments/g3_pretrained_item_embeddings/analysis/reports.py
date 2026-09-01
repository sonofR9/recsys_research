from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Sequence


@dataclass(frozen=True)
class TuningRow:
    research_question: str
    family: str
    trial_id: str
    status: Literal["usable", "failed", "interrupted", "incompatible"]
    embedding_learning_rate: float
    deep_learning_rate: float
    declared_horizon_epochs: int
    completed_horizon_epochs: int
    restored_best_epoch: int
    capacity: int | None
    validation_recall_at_100: float
    validation_ndcg_at_100: float
    training_seconds: float


@dataclass(frozen=True)
class ReaderRow:
    variant: str
    metrics: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ReaderSection:
    question: str
    reference_variant: str
    rows: tuple[ReaderRow, ...]


def build_tuning_report(rows: Sequence[TuningRow]) -> str:
    usable = []
    identities = set()
    for row in rows:
        identity = (row.research_question, row.family, row.trial_id)
        if identity in identities:
            raise ValueError(f"duplicate tuning row {identity}")
        identities.add(identity)
        _validate_tuning_row(row)
        if row.status == "usable":
            usable.append(row)
    grouped: dict[str, dict[str, list[TuningRow]]] = {}
    for row in usable:
        grouped.setdefault(row.research_question, {}).setdefault(row.family, []).append(
            row
        )
    sections = []
    for research_question, families in grouped.items():
        sections.append(f"## {research_question}")
        for family, family_rows in families.items():
            sections.append(f"### {family}")
            sections.append(_tuning_table(family_rows))
    return "\n\n".join(sections) + ("\n" if sections else "")


def build_compact_report(
    sections: Sequence[ReaderSection],
    *,
    relative_dispersions: Mapping[str, float],
) -> str:
    rendered = [
        _reader_section(section, relative_dispersions=relative_dispersions)
        for section in sections
    ]
    return "\n\n".join(rendered) + ("\n" if rendered else "")


def build_reader_scaffold(
    *,
    title: str,
    description: str,
    sections: Sequence[ReaderSection],
    aggregate: ReaderSection,
    relative_dispersions: Mapping[str, float],
) -> str:
    if aggregate.question != "Aggregated improvement":
        raise ValueError("reader scaffold requires the closing aggregated improvement")
    if any(section.question == aggregate.question for section in sections):
        raise ValueError("aggregated improvement must appear only as the closing section")
    if not title.strip() or not description.strip():
        raise ValueError("reader scaffold requires a title and description")
    body = build_compact_report(
        (*sections, aggregate), relative_dispersions=relative_dispersions
    ).rstrip()
    return f"# {title}\n\n{description}\n\n{body}\n"


def _tuning_table(rows: Sequence[TuningRow]) -> str:
    winner_index = max(
        range(len(rows)),
        key=lambda index: (
            rows[index].validation_recall_at_100,
            rows[index].validation_ndcg_at_100,
            -rows[index].training_seconds,
            -index,
        ),
    )
    include_capacity = any(row.capacity is not None for row in rows)
    headers = [
        "trial",
        "embedding lr",
        "deep lr",
        "declared horizon",
        "restored epoch",
    ]
    if include_capacity:
        headers.append("capacity")
    headers.extend(
        ("validation recall@100", "validation ndcg@100", "training seconds")
    )
    rendered = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(':---:' for _ in headers)} |",
    ]
    for index, row in enumerate(rows):
        trial = f"**{row.trial_id}**" if index == winner_index else row.trial_id
        values = [
            trial,
            f"{row.embedding_learning_rate:.10g}",
            f"{row.deep_learning_rate:.10g}",
            str(row.declared_horizon_epochs),
            str(row.restored_best_epoch),
        ]
        if include_capacity:
            values.append(str(row.capacity) if row.capacity is not None else "—")
        values.extend(
            (
                f"{row.validation_recall_at_100:.6f}",
                f"{row.validation_ndcg_at_100:.6f}",
                f"{row.training_seconds:.1f}",
            )
        )
        rendered.append(f"| {' | '.join(values)} |")
    return "\n".join(rendered)


def _reader_section(
    section: ReaderSection,
    *,
    relative_dispersions: Mapping[str, float],
) -> str:
    if not section.question or not section.rows:
        raise ValueError("reader section requires a question and rows")
    by_variant = {row.variant: row for row in section.rows}
    if len(by_variant) != len(section.rows):
        raise ValueError("reader section contains duplicate variants")
    try:
        reference = by_variant[section.reference_variant]
    except KeyError as error:
        raise ValueError("reader section reference row is absent") from error
    metric_names = tuple(name for name, _ in reference.metrics)
    if not metric_names or len(set(metric_names)) != len(metric_names):
        raise ValueError("reader reference has duplicate or absent metrics")
    if set(relative_dispersions) < set(metric_names):
        raise ValueError("reader report lacks a relative dispersion")
    reference_metrics = dict(reference.metrics)
    for row in section.rows:
        if tuple(name for name, _ in row.metrics) != metric_names:
            raise ValueError("reader rows do not share the reference metric schema")
        if not row.variant or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for _, value in row.metrics
        ):
            raise ValueError("reader row has an invalid variant or metric")

    rendered = [
        f"## {section.question}",
        "",
        f"| variant | {' | '.join(metric_names)} |",
        f"| :--- | {' | '.join(':---:' for _ in metric_names)} |",
    ]
    for row in section.rows:
        cells = [row.variant]
        for metric_name, value in row.metrics:
            if row.variant == section.reference_variant:
                cells.append(f"{value:.3f}")
            else:
                cells.append(
                    _reader_metric(
                        value,
                        reference=reference_metrics[metric_name],
                        relative_dispersion=relative_dispersions[metric_name],
                    )
                )
        rendered.append(f"| {' | '.join(cells)} |")
    return "\n".join(rendered)


def _reader_metric(
    value: float,
    *,
    reference: float,
    relative_dispersion: float,
) -> str:
    if not math.isfinite(relative_dispersion) or relative_dispersion < 0:
        raise ValueError("relative dispersions must be finite and nonnegative")
    if reference == 0:
        raise ValueError("reader percentage comparison requires a nonzero reference")
    change = 100.0 * (value / reference - 1.0)
    rendered = f"{change:+.1f}% ({value:.3f})"
    band = abs(reference) * relative_dispersion
    delta = value - reference
    if delta > band:
        return f'<span style="color: green">{rendered}</span>'
    if delta < -band:
        return f'<span style="color: red">{rendered}</span>'
    return rendered


def _validate_tuning_row(row: TuningRow) -> None:
    if row.status not in {"usable", "failed", "interrupted", "incompatible"}:
        raise ValueError(f"unknown tuning status {row.status!r}")
    numeric = (
        row.embedding_learning_rate,
        row.deep_learning_rate,
        row.validation_recall_at_100,
        row.validation_ndcg_at_100,
        row.training_seconds,
    )
    if not row.research_question or not row.family or not row.trial_id:
        raise ValueError("tuning row identity must be nonempty")
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("tuning row values must be finite")
    if row.status == "usable" and (
        row.completed_horizon_epochs != row.declared_horizon_epochs
        or not 1 <= row.restored_best_epoch <= row.declared_horizon_epochs
    ):
        raise ValueError("usable tuning row did not complete and restore its horizon")
