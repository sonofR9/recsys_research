"""Shared rendering rule for reader-facing metric cells.

Two results count as different only past a fixed per-metric threshold. The
thresholds are a practical approximation rather than a significance test: seed
spread belongs to the runs that produced it and falls as repeats accumulate, so
a fixed threshold is preferred to one recomputed per comparison.
"""

from __future__ import annotations

METRIC_DECIMALS = 3
_FAMILY_THRESHOLDS = (("recall", 0.003), ("coverage", 0.1))
_RANKING_THRESHOLD = 0.001


def difference_threshold(metric: str) -> float:
    family = metric.split("@")[0]
    for name, threshold in _FAMILY_THRESHOLDS:
        if name in family:
            return threshold
    return _RANKING_THRESHOLD


def absolute(value: float) -> str:
    return f"{value:.{METRIC_DECIMALS}f}"


def colored(cell: str, metric: str, difference: float) -> str:
    if abs(difference) <= difference_threshold(metric):
        return cell
    color = "green" if difference > 0 else "red"
    return f'<span style="color: {color}">{cell}</span>'


def change_cell(value: float, reference: float, metric: str) -> str:
    percent = 100 * (value - reference) / reference
    rendered = "0%" if round(percent) == 0 else f"{percent:+.0f}%"
    return colored(f"{rendered} ({absolute(value)})", metric, value - reference)
