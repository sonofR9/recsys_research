from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Literal, Sequence


@dataclass(frozen=True)
class MetricValues:
    recall_at_100: float
    ndcg_at_100: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and 0 <= value <= 1
            for value in (self.recall_at_100, self.ndcg_at_100)
        ):
            raise ValueError("Recall@100 and NDCG@100 must be finite values in [0, 1]")


@dataclass(frozen=True)
class Candidate:
    identifier: str
    metrics: MetricValues
    manifest_order: int

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("candidate identifier must be nonempty")
        if self.manifest_order < 0:
            raise ValueError("manifest order must be nonnegative")


@dataclass(frozen=True)
class SeedEvidence:
    seed: int
    metrics: MetricValues
    first_epoch_at_95_percent: float | None = None
    normalized_recall_auc: float | None = None

    def __post_init__(self) -> None:
        if self.first_epoch_at_95_percent is not None and (
            not math.isfinite(self.first_epoch_at_95_percent)
            or self.first_epoch_at_95_percent <= 0
        ):
            raise ValueError("epoch-to-95% must be positive finite")
        if self.normalized_recall_auc is not None and (
            not math.isfinite(self.normalized_recall_auc)
            or not 0 <= self.normalized_recall_auc <= 1
        ):
            raise ValueError("normalized Recall AUC must be finite and in [0, 1]")


@dataclass(frozen=True)
class MeanSeedEvidence:
    metrics: MetricValues
    mean_first_epoch_at_95_percent: float | None
    mean_normalized_recall_auc: float | None
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class PromotionDecision:
    candidate_id: str
    promoted: bool
    failed_references: tuple[str, ...]


@dataclass(frozen=True)
class Rq1Decision:
    selected: Literal["random", "content_pca"]
    reason: Literal["quality", "faster_convergence", "retain_random"]
    random: MeanSeedEvidence
    content_pca: MeanSeedEvidence


@dataclass(frozen=True)
class Rq23Decision:
    rq2_selected: Literal["rq0", "suffix"]
    terminal_selected: Literal["rq0", "suffix", "none"]
    promoted: bool
    rq0: MeanSeedEvidence
    suffix: MeanSeedEvidence
    none: MeanSeedEvidence


@dataclass(frozen=True)
class TerminalResolution:
    diagnostic_bundle_id: str
    aggregate_bundle_id: str
    sid_promoted: bool
    requires_terminal_bridge: bool
    launch_aggregate_run: bool


def comparison_band(reference: float, relative_dispersion: float) -> float:
    if not math.isfinite(reference):
        raise ValueError("comparison reference must be finite")
    if not math.isfinite(relative_dispersion) or relative_dispersion < 0:
        raise ValueError("relative dispersion must be finite and nonnegative")
    return abs(reference) * relative_dispersion


def select_by_quality(
    candidates: Sequence[Candidate],
    *,
    recall_relative_dispersion: float,
) -> Candidate:
    if not candidates:
        raise ValueError("quality selection requires candidates")
    if len({candidate.identifier for candidate in candidates}) != len(candidates):
        raise ValueError("quality selection contains duplicate candidate IDs")
    best_recall = max(candidates, key=lambda candidate: candidate.metrics.recall_at_100)
    recall_band = comparison_band(
        best_recall.metrics.recall_at_100, recall_relative_dispersion
    )
    recall_equivalent = tuple(
        candidate
        for candidate in candidates
        if candidate.metrics.recall_at_100
        >= best_recall.metrics.recall_at_100 - recall_band
    )
    return min(
        recall_equivalent,
        key=lambda candidate: (
            -candidate.metrics.ndcg_at_100,
            candidate.manifest_order,
        ),
    )


def promote_against_two_baselines(
    candidate: Candidate,
    *,
    original: Candidate,
    best_g1: Candidate,
    recall_relative_dispersion: float,
    ndcg_relative_dispersion: float,
) -> PromotionDecision:
    failed = tuple(
        reference.identifier
        for reference in (original, best_g1)
        if not _improves_without_ndcg_regression(
            candidate.metrics,
            reference.metrics,
            recall_relative_dispersion=recall_relative_dispersion,
            ndcg_relative_dispersion=ndcg_relative_dispersion,
        )
    )
    return PromotionDecision(candidate.identifier, not failed, failed)


def mean_seed_evidence(
    rows: Sequence[SeedEvidence],
    *,
    expected_seeds: tuple[int, ...],
) -> MeanSeedEvidence:
    seeds = tuple(row.seed for row in rows)
    if seeds != expected_seeds:
        raise ValueError(
            f"confirmation seeds must be exactly {expected_seeds}, got {seeds}"
        )
    epoch_values = tuple(row.first_epoch_at_95_percent for row in rows)
    auc_values = tuple(row.normalized_recall_auc for row in rows)
    if any(value is None for value in epoch_values) != any(
        value is None for value in auc_values
    ):
        raise ValueError("convergence evidence is incomplete")
    if any(value is None for value in epoch_values):
        if not all(value is None for value in (*epoch_values, *auc_values)):
            raise ValueError("convergence evidence is incomplete")
        mean_epoch = None
        mean_auc = None
    else:
        mean_epoch = statistics.fmean(
            value for value in epoch_values if value is not None
        )
        mean_auc = statistics.fmean(value for value in auc_values if value is not None)
    return MeanSeedEvidence(
        metrics=MetricValues(
            statistics.fmean(row.metrics.recall_at_100 for row in rows),
            statistics.fmean(row.metrics.ndcg_at_100 for row in rows),
        ),
        mean_first_epoch_at_95_percent=mean_epoch,
        mean_normalized_recall_auc=mean_auc,
        seeds=seeds,
    )


def decide_rq1_initialization(
    random_rows: Sequence[SeedEvidence],
    content_rows: Sequence[SeedEvidence],
    *,
    recall_relative_dispersion: float,
    ndcg_relative_dispersion: float,
) -> Rq1Decision:
    expected_seeds = (42, 43, 44, 45)
    random = mean_seed_evidence(random_rows, expected_seeds=expected_seeds)
    content = mean_seed_evidence(content_rows, expected_seeds=expected_seeds)
    recall_band = comparison_band(
        random.metrics.recall_at_100, recall_relative_dispersion
    )
    ndcg_band = comparison_band(random.metrics.ndcg_at_100, ndcg_relative_dispersion)
    ndcg_nonregressed = (
        content.metrics.ndcg_at_100 >= random.metrics.ndcg_at_100 - ndcg_band
    )
    if (
        content.metrics.recall_at_100 > random.metrics.recall_at_100 + recall_band
        and ndcg_nonregressed
    ):
        return Rq1Decision("content_pca", "quality", random, content)
    quality_within_bands = (
        abs(content.metrics.recall_at_100 - random.metrics.recall_at_100) <= recall_band
        and abs(content.metrics.ndcg_at_100 - random.metrics.ndcg_at_100) <= ndcg_band
    )
    if quality_within_bands and _content_is_faster(random, content):
        return Rq1Decision("content_pca", "faster_convergence", random, content)
    return Rq1Decision("random", "retain_random", random, content)


def decide_rq23(
    rq0_rows: Sequence[SeedEvidence],
    suffix_rows: Sequence[SeedEvidence],
    none_rows: Sequence[SeedEvidence],
    *,
    recall_relative_dispersion: float,
    ndcg_relative_dispersion: float,
) -> Rq23Decision:
    expected_seeds = (42, 43, 44)
    rq0 = mean_seed_evidence(rq0_rows, expected_seeds=expected_seeds)
    suffix = mean_seed_evidence(suffix_rows, expected_seeds=expected_seeds)
    none = mean_seed_evidence(none_rows, expected_seeds=expected_seeds)
    suffix_nonnegative = suffix.metrics.recall_at_100 >= rq0.metrics.recall_at_100
    suffix_promoted = suffix_nonnegative and _improves_without_ndcg_regression(
        suffix.metrics,
        rq0.metrics,
        recall_relative_dispersion=recall_relative_dispersion,
        ndcg_relative_dispersion=ndcg_relative_dispersion,
    )
    none_promoted = _improves_without_ndcg_regression(
        none.metrics,
        rq0.metrics,
        recall_relative_dispersion=recall_relative_dispersion,
        ndcg_relative_dispersion=ndcg_relative_dispersion,
    )
    promoted = [
        Candidate("suffix", suffix.metrics, 0) for _ in range(int(suffix_promoted))
    ] + [Candidate("none", none.metrics, 1) for _ in range(int(none_promoted))]
    terminal = (
        select_by_quality(
            promoted, recall_relative_dispersion=recall_relative_dispersion
        ).identifier
        if promoted
        else "rq0"
    )
    return Rq23Decision(
        rq2_selected="suffix" if suffix_promoted else "rq0",
        terminal_selected=terminal,  # type: ignore[arg-type]
        promoted=bool(promoted),
        rq0=rq0,
        suffix=suffix,
        none=none,
    )


def resolve_terminal_bundle(
    diagnostic: Candidate,
    *,
    rq0_bridge_bundle_id: str,
    original: Candidate,
    best_g1: Candidate,
    recall_relative_dispersion: float,
    ndcg_relative_dispersion: float,
) -> TerminalResolution:
    promotion = promote_against_two_baselines(
        diagnostic,
        original=original,
        best_g1=best_g1,
        recall_relative_dispersion=recall_relative_dispersion,
        ndcg_relative_dispersion=ndcg_relative_dispersion,
    )
    return TerminalResolution(
        diagnostic_bundle_id=diagnostic.identifier,
        aggregate_bundle_id=(
            diagnostic.identifier if promotion.promoted else best_g1.identifier
        ),
        sid_promoted=promotion.promoted,
        requires_terminal_bridge=diagnostic.identifier != rq0_bridge_bundle_id,
        launch_aggregate_run=False,
    )


def boundary_action(
    *,
    at_outer_boundary: bool,
    extension_round: int,
    boundary_won: bool,
) -> Literal["resolved", "extend", "requires_approval"]:
    if extension_round not in (0, 1):
        raise ValueError("only one boundary extension round is approved")
    if extension_round == 1 and boundary_won:
        return "requires_approval"
    if extension_round == 0 and at_outer_boundary:
        return "extend"
    return "resolved"


def _improves_without_ndcg_regression(
    candidate: MetricValues,
    reference: MetricValues,
    *,
    recall_relative_dispersion: float,
    ndcg_relative_dispersion: float,
) -> bool:
    return candidate.recall_at_100 > reference.recall_at_100 + comparison_band(
        reference.recall_at_100, recall_relative_dispersion
    ) and candidate.ndcg_at_100 >= reference.ndcg_at_100 - comparison_band(
        reference.ndcg_at_100, ndcg_relative_dispersion
    )


def _content_is_faster(random: MeanSeedEvidence, content: MeanSeedEvidence) -> bool:
    values = (
        random.mean_first_epoch_at_95_percent,
        random.mean_normalized_recall_auc,
        content.mean_first_epoch_at_95_percent,
        content.mean_normalized_recall_auc,
    )
    if any(value is None for value in values):
        raise ValueError("RQ1 convergence comparison requires complete curve evidence")
    random_epoch, random_auc, content_epoch, content_auc = values
    assert random_epoch is not None
    assert random_auc is not None
    assert content_epoch is not None
    assert content_auc is not None
    return content_epoch < random_epoch and content_auc > random_auc
