from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


DAY_SECONDS = 86_400
MAX_LOOKAHEAD_SECONDS = 28 * DAY_SECONDS
FEATURE_NAMES = (
    "circular_time_similarity",
    "content_similarity",
    "item_jaccard",
    "artist_jaccard",
    "album_jaccard",
    "log1p_time_gap_seconds",
    "log1p_past_like_count",
    "log1p_candidate_like_count",
    "log1p_trailing_7d_like_count",
    "log1p_trailing_28d_like_count",
    "log1p_trailing_28d_active_days",
    "prefix_hour_sine",
    "prefix_hour_cosine",
    "prefix_weekday_sine",
    "prefix_weekday_cosine",
    "candidate_hour_sine",
    "candidate_hour_cosine",
    "candidate_weekday_sine",
    "candidate_weekday_cosine",
)
STANDARDIZED_FEATURE_INDICES = tuple(range(5, 11))
SelectorFamily = Literal["time", "content", "frequency", "learned"]
FrequencyEntity = Literal["item", "artist", "album"]


@dataclass(frozen=True)
class TimePartition:
    name: Literal["train", "validation", "test"]
    start: int
    end: int

    def contains(self, timestamp: int) -> bool:
        return self.start <= timestamp < self.end


@dataclass(frozen=True)
class ChronologicalBounds:
    train: TimePartition
    validation: TimePartition
    test: TimePartition

    @classmethod
    def from_interval(cls, start: int, cutoff: int) -> ChronologicalBounds:
        if cutoff <= start:
            raise ValueError("cutoff must be greater than start")
        duration = cutoff - start
        train_end = start + (70 * duration) // 100
        validation_end = start + (85 * duration) // 100
        return cls(
            train=TimePartition("train", start, train_end),
            validation=TimePartition("validation", train_end, validation_end),
            test=TimePartition("test", validation_end, cutoff),
        )

    @property
    def partitions(self) -> tuple[TimePartition, TimePartition, TimePartition]:
        return self.train, self.validation, self.test

    def partition_at(self, timestamp: int) -> TimePartition | None:
        for partition in self.partitions:
            if partition.contains(timestamp):
                return partition
        return None


@dataclass(frozen=True)
class LikeEvent:
    uid: int
    timestamp: int
    item_id: int
    artist_ids: tuple[int, ...] = ()
    album_ids: tuple[int, ...] = ()
    content_embedding: NDArray[np.floating] | None = None


@dataclass(frozen=True)
class ListenEvent:
    uid: int
    timestamp: int
    artist_ids: tuple[int, ...] = ()


@dataclass(frozen=True, order=True)
class QueryIdentity:
    uid: int
    timestamp: int
    item_id: int
    occurrence_ordinal: int


@dataclass(frozen=True, order=True)
class CandidateIdentity:
    uid: int
    timestamp: int
    item_id: int
    occurrence_ordinal: int


@dataclass(frozen=True)
class SelectorConfiguration:
    family: SelectorFamily
    period_width_seconds: int
    lookahead_seconds: int
    minimum_liked_events: int
    time_tolerance_seconds: int | None = None
    frequency_entity: FrequencyEntity | None = None
    max_leaf_nodes: int | None = None
    learning_rate: float | None = None
    l2_regularization: float | None = None

    def __post_init__(self) -> None:
        if self.family not in {"time", "content", "frequency", "learned"}:
            raise ValueError(f"unknown selector family {self.family!r}")
        if self.period_width_seconds not in {3_600, 21_600, DAY_SECONDS}:
            raise ValueError("period width must be 1h, 6h, or 24h")
        if self.minimum_liked_events not in {1, 2, 4}:
            raise ValueError("minimum liked events must be 1, 2, or 4")
        if (
            self.lookahead_seconds <= 0
            or self.lookahead_seconds > MAX_LOOKAHEAD_SECONDS
        ):
            raise ValueError("lookahead must be in (0, 28d]")
        if self.family == "time":
            if self.time_tolerance_seconds not in {0, 3_600, 7_200}:
                raise ValueError("time tolerance must be 0h, 1h, or 2h")
        elif self.time_tolerance_seconds is not None:
            raise ValueError("time tolerance applies only to the time family")
        if self.family == "frequency":
            if self.frequency_entity not in {"item", "artist", "album"}:
                raise ValueError("frequency selector requires an entity")
        elif self.frequency_entity is not None:
            raise ValueError("frequency entity applies only to the frequency family")
        learned_values = (
            self.max_leaf_nodes,
            self.learning_rate,
            self.l2_regularization,
        )
        if self.family == "learned":
            if self.max_leaf_nodes not in {7, 15, 31}:
                raise ValueError("learned selector requires 7, 15, or 31 leaves")
            if self.learning_rate is None or self.learning_rate <= 0:
                raise ValueError("learned selector requires a positive learning rate")
            if self.l2_regularization is None or self.l2_regularization < 0:
                raise ValueError("learned selector requires non-negative L2")
        elif any(value is not None for value in learned_values):
            raise ValueError("classifier settings apply only to the learned family")


@dataclass(frozen=True)
class SelectorExample:
    query: QueryIdentity
    candidate: CandidateIdentity
    period_start: int
    period_end: int
    eligible: bool
    relevance_outcome: float
    circular_time_similarity: float
    content_similarity: float
    item_jaccard: float
    artist_jaccard: float
    album_jaccard: float
    time_gap_seconds: int
    past_like_count: int
    candidate_like_count: int
    trailing_7d_like_count: int
    trailing_28d_like_count: int
    trailing_28d_active_days: int
    prefix_hour_sine: float
    prefix_hour_cosine: float
    prefix_weekday_sine: float
    prefix_weekday_cosine: float
    candidate_hour_sine: float
    candidate_hour_cosine: float
    candidate_weekday_sine: float
    candidate_weekday_cosine: float

    def feature_vector(self) -> NDArray[np.float64]:
        return np.asarray(
            (
                self.circular_time_similarity,
                self.content_similarity,
                self.item_jaccard,
                self.artist_jaccard,
                self.album_jaccard,
                math.log1p(self.time_gap_seconds),
                math.log1p(self.past_like_count),
                math.log1p(self.candidate_like_count),
                math.log1p(self.trailing_7d_like_count),
                math.log1p(self.trailing_28d_like_count),
                math.log1p(self.trailing_28d_active_days),
                self.prefix_hour_sine,
                self.prefix_hour_cosine,
                self.prefix_weekday_sine,
                self.prefix_weekday_cosine,
                self.candidate_hour_sine,
                self.candidate_hour_cosine,
                self.candidate_weekday_sine,
                self.candidate_weekday_cosine,
            ),
            dtype=np.float64,
        )

    def deterministic_score(
        self, family: SelectorFamily, entity: FrequencyEntity | None = None
    ) -> float:
        if not self.eligible:
            return 0.0
        if family == "time":
            return 1.0
        if family == "content":
            return self.content_similarity
        if family == "frequency":
            scores = {
                "item": self.item_jaccard,
                "artist": self.artist_jaccard,
                "album": self.album_jaccard,
            }
            if entity not in scores:
                raise ValueError("frequency score requires item, artist, or album")
            return scores[entity]
        raise ValueError(f"family {family!r} is not deterministic")


@dataclass(frozen=True)
class _CandidatePeriodFeatures:
    content_similarity: float
    item_jaccard: float
    artist_jaccard: float
    album_jaccard: float
    cycles: tuple[float, float, float, float]


def fold_for_user(uid: int, seed: int = 42, folds: int = 5) -> int:
    if folds <= 1:
        raise ValueError("fold count must be greater than one")
    payload = json.dumps(
        ["g4-fold-v1", int(uid), int(seed)],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % folds


def weighted_jaccard(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    keys = set(left) | set(right)
    denominator = sum(
        max(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in keys
    )
    if denominator == 0.0:
        return 0.0
    numerator = sum(
        min(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in keys
    )
    return numerator / denominator


def continuous_hour_of_week(timestamp: int) -> float:
    day, second = divmod(int(timestamp), DAY_SECONDS)
    return 24 * ((day + 3) % 7) + second / 3_600


def circular_hour_distance(left_timestamp: int, right_timestamp: int) -> float:
    absolute = abs(
        continuous_hour_of_week(left_timestamp)
        - continuous_hour_of_week(right_timestamp)
    )
    return min(absolute, 168 - absolute)


def time_similarity(left_timestamp: int, right_timestamp: int) -> float:
    return 1.0 - circular_hour_distance(left_timestamp, right_timestamp) / 84


def content_similarity(
    left_embeddings: Iterable[NDArray[np.floating]],
    right_embeddings: Iterable[NDArray[np.floating]],
) -> float:
    left = _mean_normalized_embedding(left_embeddings)
    right = _mean_normalized_embedding(right_embeddings)
    if left is None or right is None:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    cosine = float(np.dot(left, right) / denominator)
    return (min(1.0, max(-1.0, cosine)) + 1.0) / 2.0


def fit_relevance_threshold(outcomes: Sequence[float]) -> float:
    values = np.asarray(outcomes, dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one relevance outcome is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("relevance outcomes must be finite")
    rank = math.ceil(0.8 * values.size) - 1
    return float(np.sort(values)[rank])


def build_selector_examples(
    likes: Sequence[LikeEvent],
    listens: Sequence[ListenEvent],
    bounds: ChronologicalBounds,
    configuration: SelectorConfiguration,
) -> tuple[SelectorExample, ...]:
    likes_by_user = _indexed_likes_by_user(likes)
    listens_by_user: dict[int, list[ListenEvent]] = defaultdict(list)
    for listen in listens:
        listens_by_user[listen.uid].append(listen)
    for user_listens in listens_by_user.values():
        user_listens.sort(key=lambda event: (event.timestamp, event.artist_ids))

    examples: list[SelectorExample] = []
    for uid in sorted(likes_by_user):
        user_likes = likes_by_user[uid]
        user_listens = listens_by_user.get(uid, [])
        for partition in bounds.partitions:
            partition_likes = [
                indexed
                for indexed in user_likes
                if partition.contains(indexed[1].timestamp)
            ]
            partition_listens = [
                listen
                for listen in user_listens
                if partition.contains(listen.timestamp)
            ]
            examples.extend(
                _build_partition_examples(
                    partition_likes,
                    partition_listens,
                    partition,
                    configuration,
                )
            )
    return tuple(examples)


@dataclass(frozen=True)
class FeatureNormalizer:
    means: NDArray[np.float64]
    standard_deviations: NDArray[np.float64]

    @classmethod
    def fit(cls, feature_matrix: NDArray[np.float64]) -> FeatureNormalizer:
        selected = feature_matrix[:, STANDARDIZED_FEATURE_INDICES]
        means = selected.mean(axis=0)
        standard_deviations = selected.std(axis=0, ddof=0)
        standard_deviations = np.where(
            standard_deviations == 0.0, 1.0, standard_deviations
        )
        return cls(means=means, standard_deviations=standard_deviations)

    def transform(self, feature_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        transformed = np.asarray(feature_matrix, dtype=np.float64).copy()
        transformed[:, STANDARDIZED_FEATURE_INDICES] = (
            transformed[:, STANDARDIZED_FEATURE_INDICES] - self.means
        ) / self.standard_deviations
        return transformed


class UnfitSelectorError(ValueError):
    pass


@dataclass(frozen=True)
class LearnedSelector:
    configuration: SelectorConfiguration
    relevance_threshold: float
    normalizer: FeatureNormalizer
    class_weights: Mapping[int, float]
    estimator: object

    def score(self, examples: Sequence[SelectorExample]) -> NDArray[np.float64]:
        features = np.stack([example.feature_vector() for example in examples])
        eligible = np.asarray(
            [example.eligible for example in examples], dtype=np.bool_
        )
        return self.score_matrix(features, eligible)

    def score_matrix(
        self,
        features: NDArray[np.float64],
        eligible: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        scores = np.zeros(len(features), dtype=np.float64)
        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size == 0:
            return scores
        probabilities = self.estimator.predict_proba(
            self.normalizer.transform(features[eligible_indices])
        )[:, 1]
        scores[eligible_indices] = probabilities
        return scores


def fit_learned_selector(
    examples: Sequence[SelectorExample], configuration: SelectorConfiguration
) -> LearnedSelector:
    if configuration.family != "learned":
        raise ValueError("learned fitting requires the learned family")
    if not examples:
        raise UnfitSelectorError("selector fit has no rows")
    return fit_learned_feature_matrix(
        np.stack([example.feature_vector() for example in examples]),
        np.asarray([example.relevance_outcome for example in examples]),
        configuration,
    )


def fit_learned_feature_matrix(
    features: NDArray[np.float64],
    relevance_outcomes: NDArray[np.float64],
    configuration: SelectorConfiguration,
) -> LearnedSelector:
    if configuration.family != "learned":
        raise ValueError("learned fitting requires the learned family")
    if len(features) == 0:
        raise UnfitSelectorError("selector fit has no rows")
    if features.shape != (len(relevance_outcomes), len(FEATURE_NAMES)):
        raise ValueError("learned feature matrix has the wrong shape")
    relevance_threshold = fit_relevance_threshold(relevance_outcomes)
    labels = np.asarray(
        relevance_outcomes > relevance_threshold,
        dtype=np.int64,
    )
    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts == 0):
        raise UnfitSelectorError("selector fit requires both label classes")
    normalizer = FeatureNormalizer.fit(features)
    class_weights = {
        label: len(labels) / (2 * int(class_counts[label])) for label in (0, 1)
    }
    sample_weights = np.asarray([class_weights[int(label)] for label in labels])

    from sklearn.ensemble import HistGradientBoostingClassifier

    estimator = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=200,
        min_samples_leaf=20,
        max_bins=255,
        early_stopping=False,
        random_state=42,
        max_leaf_nodes=configuration.max_leaf_nodes,
        learning_rate=configuration.learning_rate,
        l2_regularization=configuration.l2_regularization,
    )
    estimator.fit(normalizer.transform(features), labels, sample_weight=sample_weights)
    return LearnedSelector(
        configuration=configuration,
        relevance_threshold=relevance_threshold,
        normalizer=normalizer,
        class_weights=class_weights,
        estimator=estimator,
    )


def example_is_eligible(
    example: SelectorExample, configuration: SelectorConfiguration
) -> bool:
    eligible = (
        example.eligible
        and example.candidate.timestamp
        <= example.query.timestamp + configuration.lookahead_seconds
        and example.candidate_like_count >= configuration.minimum_liked_events
    )
    if configuration.family == "time":
        eligible = eligible and (
            circular_hour_distance(example.query.timestamp, example.period_start)
            * 3_600
            <= int(configuration.time_tolerance_seconds or 0)
        )
    return eligible


def score_deterministic_examples(
    examples: Sequence[SelectorExample], configuration: SelectorConfiguration
) -> NDArray[np.float64]:
    if configuration.family not in {"time", "content", "frequency"}:
        raise ValueError("configuration is not deterministic")
    scores = np.zeros(len(examples), dtype=np.float64)
    for index, example in enumerate(examples):
        if not example_is_eligible(example, configuration):
            continue
        if configuration.family == "time":
            scores[index] = 1.0
        elif configuration.family == "content":
            scores[index] = example.content_similarity
        else:
            scores[index] = example.deterministic_score(
                "frequency", configuration.frequency_entity
            )
    return scores


@dataclass(frozen=True)
class SelectorMetrics:
    user_balanced_ndcg_at_10: float
    auroc: float | None
    query_count: int
    user_count: int
    pair_count: int
    positive_count: int
    negative_count: int
    positive_rate: float


def evaluate_selector(
    examples: Sequence[SelectorExample],
    scores: Sequence[float],
    relevance_threshold: float,
) -> SelectorMetrics:
    if len(examples) != len(scores):
        raise ValueError("examples and scores must have equal length")
    labels = np.asarray(
        [example.relevance_outcome > relevance_threshold for example in examples],
        dtype=np.int64,
    )
    scores_array = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores_array)):
        raise ValueError("selector scores must be finite")
    query_ndcg = _query_ndcg_values(examples, scores_array, labels)
    by_user: dict[int, list[float]] = defaultdict(list)
    for query, value in query_ndcg.items():
        by_user[query.uid].append(value)
    user_means = [float(np.mean(values)) for values in by_user.values()]
    positives = int(labels.sum())
    negatives = len(labels) - positives
    return SelectorMetrics(
        user_balanced_ndcg_at_10=float(np.mean(user_means)) if user_means else 0.0,
        auroc=_binary_auroc(labels, scores_array),
        query_count=len(query_ndcg),
        user_count=len(by_user),
        pair_count=len(examples),
        positive_count=positives,
        negative_count=negatives,
        positive_rate=positives / len(labels) if len(labels) else 0.0,
    )


@dataclass(frozen=True)
class BootstrapGate:
    user_count: int
    mean_difference: float
    lower_95: float
    upper_95: float
    passes: bool


def paired_user_bootstrap_gate(
    examples: Sequence[SelectorExample],
    learned_scores: Sequence[float],
    deterministic_scores: Sequence[float],
    relevance_threshold: float,
    *,
    replicates: int = 10_000,
    seed: int = 42,
) -> BootstrapGate:
    if len(examples) != len(learned_scores) or len(examples) != len(
        deterministic_scores
    ):
        raise ValueError("examples and both score vectors must have equal length")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    labels = np.asarray(
        [example.relevance_outcome > relevance_threshold for example in examples],
        dtype=np.int64,
    )
    learned = _query_ndcg_values(examples, np.asarray(learned_scores), labels)
    deterministic = _query_ndcg_values(
        examples, np.asarray(deterministic_scores), labels
    )
    if learned.keys() != deterministic.keys():
        raise ValueError("selector pipelines must have identical query universes")
    by_user: dict[int, list[float]] = defaultdict(list)
    for query in learned:
        by_user[query.uid].append(learned[query] - deterministic[query])
    user_differences = np.asarray(
        [np.mean(by_user[uid]) for uid in sorted(by_user)], dtype=np.float64
    )
    if user_differences.size == 0:
        raise ValueError("bootstrap requires at least one user")
    generator = np.random.Generator(np.random.PCG64(seed))
    samples = generator.choice(
        user_differences,
        size=(replicates, user_differences.size),
        replace=True,
    ).mean(axis=1)
    samples.sort()
    lower = float(samples[math.ceil(0.025 * replicates) - 1])
    upper = float(samples[math.ceil(0.975 * replicates) - 1])
    mean = float(user_differences.mean())
    return BootstrapGate(
        user_count=user_differences.size,
        mean_difference=mean,
        lower_95=lower,
        upper_95=upper,
        passes=lower > 0.0,
    )


@dataclass(frozen=True)
class FoldArtifact:
    scored_fold: int
    fit_user_ids: frozenset[int]
    scored_indices: tuple[int, ...]
    selector: LearnedSelector


@dataclass(frozen=True)
class CrossFitResult:
    scores: NDArray[np.float64]
    artifacts: tuple[FoldArtifact, ...]


def cross_fit_learned_selector(
    examples: Sequence[SelectorExample],
    configuration: SelectorConfiguration,
    *,
    folds: int = 5,
    seed: int = 42,
) -> CrossFitResult:
    scores = np.zeros(len(examples), dtype=np.float64)
    artifacts: list[FoldArtifact] = []
    for scored_fold in range(folds):
        fit_indices = [
            index
            for index, example in enumerate(examples)
            if fold_for_user(example.query.uid, seed, folds) != scored_fold
        ]
        scored_indices = [
            index
            for index, example in enumerate(examples)
            if fold_for_user(example.query.uid, seed, folds) == scored_fold
        ]
        if not scored_indices:
            raise UnfitSelectorError(f"fold {scored_fold} has no scored rows")
        selector = fit_learned_selector(
            [examples[index] for index in fit_indices], configuration
        )
        scores[scored_indices] = selector.score(
            [examples[index] for index in scored_indices]
        )
        artifacts.append(
            FoldArtifact(
                scored_fold=scored_fold,
                fit_user_ids=frozenset(
                    examples[index].query.uid for index in fit_indices
                ),
                scored_indices=tuple(scored_indices),
                selector=selector,
            )
        )
    return CrossFitResult(scores=scores, artifacts=tuple(artifacts))


def _indexed_likes_by_user(
    likes: Sequence[LikeEvent],
) -> dict[int, list[tuple[int, LikeEvent]]]:
    grouped: dict[int, list[tuple[int, LikeEvent]]] = defaultdict(list)
    for input_ordinal, like in enumerate(likes):
        grouped[like.uid].append((input_ordinal, like))
    for uid, user_likes in grouped.items():
        sorted_likes = sorted(
            user_likes,
            key=lambda pair: (pair[1].timestamp, pair[0]),
        )
        grouped[uid] = [
            (occurrence_ordinal, like)
            for occurrence_ordinal, (_, like) in enumerate(sorted_likes)
        ]
    return grouped


def _build_partition_examples(
    indexed_likes: Sequence[tuple[int, LikeEvent]],
    listens: Sequence[ListenEvent],
    partition: TimePartition,
    configuration: SelectorConfiguration,
) -> list[SelectorExample]:
    examples: list[SelectorExample] = []
    for prefix_position, (prefix_ordinal, prefix) in enumerate(indexed_likes):
        if prefix.timestamp - DAY_SECONDS < partition.start:
            continue
        candidates = [
            candidate
            for candidate in indexed_likes[prefix_position + 1 :]
            if prefix.timestamp < candidate[1].timestamp
            and candidate[1].timestamp <= prefix.timestamp + MAX_LOOKAHEAD_SECONDS
            and _utc_day_end(candidate[1].timestamp) <= partition.end
        ]
        if not candidates:
            continue
        past_likes = [
            event
            for _, event in indexed_likes
            if prefix.timestamp - configuration.period_width_seconds
            < event.timestamp
            <= prefix.timestamp
        ]
        trailing_7d = [
            event
            for _, event in indexed_likes
            if prefix.timestamp - 7 * DAY_SECONDS < event.timestamp <= prefix.timestamp
        ]
        trailing_28d = [
            event
            for _, event in indexed_likes
            if prefix.timestamp - 28 * DAY_SECONDS < event.timestamp <= prefix.timestamp
        ]
        listened_past = _listen_artist_frequency(
            listens,
            prefix.timestamp - DAY_SECONDS,
            prefix.timestamp,
            include_end=True,
        )
        past_embeddings = [
            event.content_embedding
            for event in past_likes
            if event.content_embedding is not None
        ]
        past_item = _like_frequency(past_likes, "item")
        past_artist = _like_frequency(past_likes, "artist")
        past_album = _like_frequency(past_likes, "album")
        prefix_cycles = _cyclic_features(prefix.timestamp)
        query = QueryIdentity(
            prefix.uid,
            prefix.timestamp,
            prefix.item_id,
            prefix_ordinal,
        )
        period_cache: dict[int, list[LikeEvent]] = {}
        period_feature_cache: dict[int, _CandidatePeriodFeatures] = {}
        outcome_cache: dict[int, float] = {}
        for candidate_ordinal, candidate in candidates:
            period_start = (
                candidate.timestamp // configuration.period_width_seconds
            ) * configuration.period_width_seconds
            period_end = period_start + configuration.period_width_seconds
            candidate_likes = period_cache.get(period_start)
            if candidate_likes is None:
                candidate_likes = [
                    event
                    for _, event in indexed_likes
                    if period_start <= event.timestamp < period_end
                ]
                period_cache[period_start] = candidate_likes
            day_start = (candidate.timestamp // DAY_SECONDS) * DAY_SECONDS
            relevance_outcome = outcome_cache.get(day_start)
            if relevance_outcome is None:
                relevance_outcome = weighted_jaccard(
                    listened_past,
                    _listen_artist_frequency(
                        listens,
                        max(prefix.timestamp, day_start),
                        day_start + DAY_SECONDS,
                        include_end=False,
                    ),
                )
                outcome_cache[day_start] = relevance_outcome
            eligible = (
                candidate.timestamp
                <= prefix.timestamp + configuration.lookahead_seconds
                and period_start > prefix.timestamp
                and period_end <= partition.end
                and len(candidate_likes) >= configuration.minimum_liked_events
            )
            time_score = time_similarity(prefix.timestamp, period_start)
            if configuration.family == "time":
                eligible = eligible and (
                    circular_hour_distance(prefix.timestamp, period_start) * 3_600
                    <= int(configuration.time_tolerance_seconds or 0)
                )
            period_features = period_feature_cache.get(period_start)
            if period_features is None:
                candidate_embeddings = [
                    event.content_embedding
                    for event in candidate_likes
                    if event.content_embedding is not None
                ]
                period_features = _CandidatePeriodFeatures(
                    content_similarity=content_similarity(
                        past_embeddings, candidate_embeddings
                    ),
                    item_jaccard=weighted_jaccard(
                        past_item, _like_frequency(candidate_likes, "item")
                    ),
                    artist_jaccard=weighted_jaccard(
                        past_artist, _like_frequency(candidate_likes, "artist")
                    ),
                    album_jaccard=weighted_jaccard(
                        past_album, _like_frequency(candidate_likes, "album")
                    ),
                    cycles=_cyclic_features(period_start),
                )
                period_feature_cache[period_start] = period_features
            examples.append(
                SelectorExample(
                    query=query,
                    candidate=CandidateIdentity(
                        candidate.uid,
                        candidate.timestamp,
                        candidate.item_id,
                        candidate_ordinal,
                    ),
                    period_start=period_start,
                    period_end=period_end,
                    eligible=eligible,
                    relevance_outcome=relevance_outcome,
                    circular_time_similarity=time_score,
                    content_similarity=period_features.content_similarity,
                    item_jaccard=period_features.item_jaccard,
                    artist_jaccard=period_features.artist_jaccard,
                    album_jaccard=period_features.album_jaccard,
                    time_gap_seconds=max(0, period_start - prefix.timestamp),
                    past_like_count=len(past_likes),
                    candidate_like_count=len(candidate_likes),
                    trailing_7d_like_count=len(trailing_7d),
                    trailing_28d_like_count=len(trailing_28d),
                    trailing_28d_active_days=len(
                        {event.timestamp // DAY_SECONDS for event in trailing_28d}
                    ),
                    prefix_hour_sine=prefix_cycles[0],
                    prefix_hour_cosine=prefix_cycles[1],
                    prefix_weekday_sine=prefix_cycles[2],
                    prefix_weekday_cosine=prefix_cycles[3],
                    candidate_hour_sine=period_features.cycles[0],
                    candidate_hour_cosine=period_features.cycles[1],
                    candidate_weekday_sine=period_features.cycles[2],
                    candidate_weekday_cosine=period_features.cycles[3],
                )
            )
    return examples


def _utc_day_end(timestamp: int) -> int:
    return (timestamp // DAY_SECONDS + 1) * DAY_SECONDS


def _mass(ids: Iterable[int]) -> dict[int, float]:
    known = sorted({int(value) for value in ids if int(value) != 0})
    if not known:
        return {}
    weight = 1.0 / len(known)
    return {value: weight for value in known}


def _add_mass(target: dict[int, float], source: Mapping[int, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + value


def _like_frequency(
    events: Iterable[LikeEvent], entity: FrequencyEntity
) -> dict[int, float]:
    result: dict[int, float] = {}
    for event in events:
        if entity == "item":
            source = {event.item_id: 1.0} if event.item_id != 0 else {}
        elif entity == "artist":
            source = _mass(event.artist_ids)
        else:
            source = _mass(event.album_ids)
        _add_mass(result, source)
    return result


def _listen_artist_frequency(
    events: Iterable[ListenEvent],
    start: int,
    end: int,
    *,
    include_end: bool,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for event in events:
        in_window = (
            start < event.timestamp <= end
            if include_end
            else start < event.timestamp < end
        )
        if in_window:
            _add_mass(result, _mass(event.artist_ids))
    return result


def _mean_normalized_embedding(
    embeddings: Iterable[NDArray[np.floating]],
) -> NDArray[np.float64] | None:
    normalized: list[NDArray[np.float64]] = []
    for embedding in embeddings:
        array = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(array))
        if norm > 0.0:
            normalized.append(array / norm)
    if not normalized:
        return None
    mean = np.mean(normalized, axis=0)
    if float(np.linalg.norm(mean)) == 0.0:
        return None
    return mean


def _cyclic_features(timestamp: int) -> tuple[float, float, float, float]:
    day, second = divmod(timestamp, DAY_SECONDS)
    hour_phase = 2 * math.pi * second / DAY_SECONDS
    weekday_phase = 2 * math.pi * ((day + 3) % 7) / 7
    return (
        math.sin(hour_phase),
        math.cos(hour_phase),
        math.sin(weekday_phase),
        math.cos(weekday_phase),
    )


def _query_ndcg_values(
    examples: Sequence[SelectorExample],
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    cutoff: int = 10,
) -> dict[QueryIdentity, float]:
    grouped: dict[QueryIdentity, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.query].append(index)
    result: dict[QueryIdentity, float] = {}
    for query, indices in grouped.items():
        ranked = sorted(
            indices,
            key=lambda index: (
                -float(scores[index]),
                examples[index].period_start,
                examples[index].candidate.item_id,
                examples[index].candidate.occurrence_ordinal,
            ),
        )[:cutoff]
        dcg = sum(
            int(labels[index]) / math.log2(rank + 2)
            for rank, index in enumerate(ranked)
        )
        positive_count = int(labels[indices].sum())
        ideal_count = min(positive_count, cutoff)
        if ideal_count == 0:
            result[query] = 0.0
            continue
        ideal = sum(1 / math.log2(rank + 2) for rank in range(ideal_count))
        result[query] = dcg / ideal
    return result


def _binary_auroc(
    labels: NDArray[np.int64], scores: NDArray[np.float64]
) -> float | None:
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2
        position = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )
