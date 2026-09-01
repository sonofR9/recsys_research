from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import json
import math
import mmap
import multiprocessing
import os
import pickle
import resource
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from dcn.config import Experiment
from experiments.g4_future_items.configs.selectors import (
    SelectorTrial,
    SelectorTrialResult,
    compile_selector_search,
    select_family_winner,
    select_strongest_deterministic,
    selector_trial_from_job,
)
from experiments.g4_future_items.protocol.materialization import (
    DEFAULT_PERIOD_ARTIFACT_ROOT,
    PeriodArtifactIdentity,
    ScoredOccurrence,
    ScoredPeriod,
    ScoredQuery,
    SelectorInputPaths,
    SelectorUserEvents,
    iter_selector_users,
    write_period_artifact,
)
from experiments.g4_future_items.protocol.manifest import MATERIALIZATION_COST_LIMITS
from experiments.g4_future_items.selectors import (
    DAY_SECONDS,
    FEATURE_NAMES,
    BootstrapGate,
    ChronologicalBounds,
    LearnedSelector,
    SelectorConfiguration,
    SelectorExample,
    SelectorMetrics,
    TimePartition,
    build_selector_examples,
    fit_learned_feature_matrix,
    fit_relevance_threshold,
    fold_for_user,
)

CANONICAL_MODULE = "experiments.g4_future_items.launchers.run_selectors"
if __name__ in {"__main__", "dcn_experiment_script"}:
    sys.modules[CANONICAL_MODULE] = sys.modules[__name__]

WIDTHS = (3_600, 21_600, DAY_SECONDS)
PREPARED_VERSION = "g4-selector-columnar-v1"
SEARCH_VERSION = "g4-selector-search-v1"
GATE_VERSION = "g4-selector-gate-v1"
FOLD_VERSION = "g4-selector-fold-v1"
CANDIDATE_IDENTITY_COLUMNS = (
    "query_uid",
    "query_timestamp",
    "query_item",
    "query_position",
    "candidate_timestamp",
    "candidate_item",
    "candidate_position",
    "period_start",
    "period_end",
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREPARED_ROOT = PROJECT_ROOT / "generated/g4_selector_prepared"
SEARCH_ROOT = PROJECT_ROOT / "generated/g4_selector_search"
GATE_ROOT = PROJECT_ROOT / "generated/g4_selector_gate"
FOLD_ROOT = PROJECT_ROOT / "generated/g4_selector_folds"
NATIVE_MATERIALIZATION_LOCK_PATH = (
    PROJECT_ROOT / "generated/g4-native-materialization.lock"
)
PROTOCOL_ROOT = PROJECT_ROOT / "experiments/g4_future_items/protocol"
CONTROL_SEMANTICS_MANIFEST_PATH = PROTOCOL_ROOT / "control_semantics_manifest.json"
SELECTED_CONTROL_MANIFEST_PATH = PROTOCOL_ROOT / "selected_control_manifest.json"
TREATMENT_SEMANTICS_MANIFEST_PATH = PROTOCOL_ROOT / "treatment_semantics_manifest.json"
TREATMENT_SCHEMA_REVISIONS = {
    "fold_assignment": "g4-fold-v1",
    "mask_schema": "g4-acceptable-positive-mask-v1",
    "materialization_cost_artifact": "g4-materialization-cost-v1",
    "materialization_measurement": "g4-materialization-gate-v1",
    "period_artifact": "g4-period-artifact-v1",
    "selector_fold_artifact": FOLD_VERSION,
    "selector_gate_artifact": GATE_VERSION,
    "selector_prepared_artifact": PREPARED_VERSION,
    "selector_search_artifact": SEARCH_VERSION,
    "target_fold_rng": "g4-fold-v1",
    "target_rng": "g4-target-v1",
    "target_schema": "g4-future-target-v1",
}
TREATMENT_FIXTURE_PATHS = {
    "objective_and_mask": "dcn/tests/experiments/g4_future_items/test_targets.py",
    "period_materialization_and_artifact": (
        "dcn/tests/experiments/g4_future_items/test_materialization.py"
    ),
    "selector_and_fold": "dcn/tests/experiments/g4_future_items/test_selectors.py",
    "selector_pipeline_artifacts": (
        "dcn/tests/experiments/g4_future_items/test_selector_pipeline.py"
    ),
}
FEASIBILITY_SAMPLE_PERCENTS = (5, 10)
FEASIBILITY_RSS_LIMIT_BYTES = 200 * 1024**3
FEASIBILITY_RSS_GROWTH_LIMIT = 0.25
FEASIBILITY_PROJECTION_FRACTION = 0.8


@dataclass(frozen=True)
class PreparedSelectorData:
    path: Path
    sha256: str
    manifest: Mapping[str, Any]

    @property
    def semantics_sha256(self) -> str:
        return str(self.manifest["semantics_sha256"])

    def width_path(self, width: int) -> Path:
        return self.path / self.manifest["widths"][str(width)]["file"]

    @property
    def query_path(self) -> Path:
        return self.path / self.manifest["queries"]["file"]


@dataclass(frozen=True)
class SelectorSearchResult:
    trial: SelectorTrial
    metrics: SelectorMetrics
    relevance_threshold: float
    artifact_sha256: str
    artifact_payload_sha256: str
    artifact_path: Path
    prepared_sha256: str
    prepared_semantics_sha256: str
    wall_seconds: float

    def to_trial_result(self) -> SelectorTrialResult:
        return SelectorTrialResult(self.trial, self.metrics, self.artifact_sha256)


def native500m_feasibility_user_selected(uid: int, percent: int) -> bool:
    if percent not in FEASIBILITY_SAMPLE_PERCENTS:
        raise ValueError("feasibility sample percent must be 5 or 10")
    payload = json.dumps(
        ["g4-feasibility-v1", int(uid), 42],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (100 // percent) == 0


def native500m_feasibility_trials() -> tuple[SelectorTrial, SelectorTrial]:
    common = {
        "period_width_seconds": 3_600,
        "lookahead_seconds": 7 * DAY_SECONDS,
        "minimum_liked_events": 1,
    }
    return (
        SelectorTrial(
            family="content",
            trial_id=1,
            configuration=SelectorConfiguration(family="content", **common),
        ),
        SelectorTrial(
            family="learned",
            trial_id=1,
            configuration=SelectorConfiguration(
                family="learned",
                max_leaf_nodes=31,
                learning_rate=0.05,
                l2_regularization=1e-5,
                **common,
            ),
        ),
    )


def evaluate_native500m_feasibility(
    five_percent: Mapping[str, Any], ten_percent: Mapping[str, Any]
) -> dict[str, Any]:
    measurements = {"5": dict(five_percent), "10": dict(ten_percent)}
    if (
        measurements["5"].get("sample_percent") != 5
        or measurements["10"].get("sample_percent") != 10
    ):
        raise ValueError("feasibility measurements must be the 5% and 10% samples")

    def projection(name: str) -> float:
        five = float(measurements["5"][name])
        ten = float(measurements["10"][name])
        if not math.isfinite(five) or not math.isfinite(ten) or min(five, ten) < 0:
            raise ValueError("feasibility measurements must be finite and non-negative")
        return max(20 * five, 10 * ten, ten + 18 * max(0.0, ten - five))

    wall_projection = projection("wall_seconds")
    logical_projection = projection("logical_bytes")
    five_rss = int(measurements["5"]["peak_rss_bytes"])
    ten_rss = int(measurements["10"]["peak_rss_bytes"])
    if five_rss <= 0 or ten_rss < 0:
        raise ValueError("feasibility RSS measurements are invalid")
    rss_growth = (ten_rss - five_rss) / five_rss
    failures = []
    if wall_projection > (
        FEASIBILITY_PROJECTION_FRACTION * MATERIALIZATION_COST_LIMITS["wall_seconds"]
    ):
        failures.append("wall_projection")
    if logical_projection > (
        FEASIBILITY_PROJECTION_FRACTION
        * MATERIALIZATION_COST_LIMITS["logical_output_scratch_bytes"]
    ):
        failures.append("logical_projection")
    if ten_rss > FEASIBILITY_RSS_LIMIT_BYTES:
        failures.append("ten_percent_rss")
    if rss_growth > FEASIBILITY_RSS_GROWTH_LIMIT:
        failures.append("rss_growth")
    return {
        "version": "g4-native500m-feasibility-decision-v1",
        "measurements": measurements,
        "wall_seconds_projection": wall_projection,
        "logical_bytes_projection": int(logical_projection),
        "ten_percent_peak_rss_bytes": ten_rss,
        "peak_rss_growth": rss_growth,
        "failed_conditions": failures,
        "passes": not failures,
        "selection_eligible": False,
    }


def classify_native500m_materialization_cost(
    *,
    wall_seconds: float,
    peak_rss_bytes: int,
    logical_bytes: int,
    post_launch_contention: bool,
    attempt: int,
    limits: Mapping[str, int | float] = MATERIALIZATION_COST_LIMITS,
) -> str:
    if attempt not in {1, 2} or isinstance(attempt, bool):
        raise ValueError("materialization attempt must be 1 or 2")
    measurements = (wall_seconds, peak_rss_bytes, logical_bytes)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in measurements
    ):
        raise ValueError("materialization costs must be finite and non-negative")
    if (
        peak_rss_bytes > limits["peak_aggregate_rss_bytes"]
        or logical_bytes > limits["logical_output_scratch_bytes"]
    ):
        return "stop"
    if wall_seconds <= limits["wall_seconds"]:
        return "pass"
    if post_launch_contention and attempt == 1:
        return "inconclusive"
    return "stop"


def prepare_selector_data(
    paths: SelectorInputPaths,
    bounds: ChronologicalBounds,
    output_root: Path,
    *,
    provenance: Mapping[str, Any] | None = None,
    workers: int = 1,
    feasibility_percent: int | None = None,
) -> PreparedSelectorData:
    if workers < 1:
        raise ValueError("selector preparation workers must be positive")
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    partial = output_root / f".prepare-{os.getpid()}-{time.time_ns()}"
    partial.mkdir()
    width_writers: dict[int, pq.ParquetWriter | None] = {
        width: None for width in WIDTHS
    }
    width_rows = {width: 0 for width in WIDTHS}
    width_queries = {width: 0 for width in WIDTHS}
    query_writer: pq.ParquetWriter | None = None
    query_rows = 0
    try:
        for (
            query_documents,
            documents_by_width,
            query_counts_by_width,
        ) in _iter_prepared_user_documents(
            paths,
            bounds,
            workers,
            feasibility_percent=feasibility_percent,
        ):
            if query_documents:
                query_table = pa.Table.from_pylist(query_documents)
                if query_writer is None:
                    query_writer = pq.ParquetWriter(
                        partial / "queries.parquet", query_table.schema
                    )
                query_writer.write_table(query_table, row_group_size=len(query_table))
                query_rows += len(query_table)
            for width in WIDTHS:
                documents = documents_by_width[width]
                if not documents:
                    continue
                table = pa.Table.from_pylist(documents)
                if width_writers[width] is None:
                    width_writers[width] = pq.ParquetWriter(
                        partial / f"width_{width}.parquet", table.schema
                    )
                width_writers[width].write_table(table, row_group_size=len(table))
                width_rows[width] += len(table)
                width_queries[width] += query_counts_by_width[width]
        if query_writer is None or any(
            writer is None for writer in width_writers.values()
        ):
            raise ValueError("selector preparation produced an empty artifact")
        query_writer.close()
        query_writer = None
        for width, writer in width_writers.items():
            assert writer is not None
            writer.close()
            width_writers[width] = None
        if len(set(width_queries.values())) != 1:
            raise ValueError(
                "selector widths produced different common query universes"
            )
        bounds_document = {
            partition.name: {"start": partition.start, "end": partition.end}
            for partition in bounds.partitions
        }
        pair_budget = {
            "control_prefixes": query_rows,
            "selector_common_queries": width_queries[WIDTHS[0]],
            "next_item_fallback_queries": query_rows - width_queries[WIDTHS[0]],
        }
        input_identities = {
            name: _file_identity(path)
            for name, path in zip(
                (
                    "control_likes",
                    "raw_events",
                    "item_id_remap",
                    "compact_embeddings",
                ),
                paths.paths,
            )
        }
        query_identity = _file_identity(partial / "queries.parquet")
        width_identities = {
            width: _file_identity(partial / f"width_{width}.parquet")
            for width in WIDTHS
        }
        semantics = {
            "version": PREPARED_VERSION,
            "bounds": bounds_document,
            "input_sha256": {
                name: identity["sha256"] for name, identity in input_identities.items()
            },
            "query_sha256": query_identity["sha256"],
            "width_sha256": {
                width: identity["sha256"]
                for width, identity in width_identities.items()
            },
            "width_rows": width_rows,
            "width_queries": width_queries,
            "pair_budget": pair_budget,
        }
        user_sample = _native500m_feasibility_sample(feasibility_percent)
        if user_sample is not None:
            semantics["user_sample"] = user_sample
        semantics_sha256 = hashlib.sha256(_canonical_bytes(semantics)).hexdigest()
        manifest = {
            "version": PREPARED_VERSION,
            "semantics_sha256": semantics_sha256,
            "bounds": bounds_document,
            "queries": query_identity | {"rows": query_rows},
            "widths": {
                str(width): width_identities[width]
                | {"rows": width_rows[width], "queries": width_queries[width]}
                for width in WIDTHS
            },
            "pair_budget": pair_budget,
            "inputs": input_identities,
            "provenance": dict(provenance or {}),
            "wall_seconds": time.perf_counter() - started,
        }
        if user_sample is not None:
            manifest["user_sample"] = user_sample
        content = _canonical_bytes(manifest)
        digest = hashlib.sha256(content).hexdigest()
        (partial / "manifest.json").write_bytes(content)
        destination = output_root / digest
        if destination.exists():
            _verify_prepared(destination, digest)
            _remove_tree(partial)
        else:
            os.replace(partial, destination)
        return PreparedSelectorData(destination, digest, manifest)
    except BaseException:
        if query_writer is not None:
            query_writer.close()
        for writer in width_writers.values():
            if writer is not None:
                writer.close()
        if partial.exists():
            _remove_tree(partial)
        raise


def open_prepared_selector_data(
    path: Path, expected_sha256: str
) -> PreparedSelectorData:
    if path.name != expected_sha256:
        path = path / expected_sha256
    manifest = _verify_prepared(path, expected_sha256)
    return PreparedSelectorData(path, expected_sha256, manifest)


def run_search_trial(
    prepared: PreparedSelectorData,
    trial: SelectorTrial,
    output_root: Path,
    *,
    output_artifact_sha256: str | None = None,
) -> SelectorSearchResult:
    if output_artifact_sha256 is not None:
        destination = output_root / output_artifact_sha256
        if destination.exists():
            existing = load_search_result(output_root, output_artifact_sha256)
            if (
                existing.trial != trial
                or existing.prepared_semantics_sha256 != prepared.semantics_sha256
            ):
                raise RuntimeError("immutable selector search artifact differs")
            return existing
    started = time.perf_counter()
    configuration = trial.configuration
    train = _load_partition(
        prepared.width_path(configuration.period_width_seconds), "train"
    )
    validation = _load_partition(
        prepared.width_path(configuration.period_width_seconds), "validation"
    )
    threshold = fit_relevance_threshold(train["relevance_outcome"])
    artifact: dict[str, Any] = {
        "version": SEARCH_VERSION,
        "trial": trial.to_dict(),
        "sampler_seed": trial.sampler_seed,
        "classifier_seed": 42,
        "prepared_sha256": prepared.sha256,
        "prepared_semantics_sha256": prepared.semantics_sha256,
        "prepared_input_sha256": {
            name: identity["sha256"]
            for name, identity in prepared.manifest["inputs"].items()
        },
        "relevance_threshold": threshold,
        "output_artifact_sha256": output_artifact_sha256,
    }
    learned: LearnedSelector | None = None
    if configuration.family == "learned":
        learned = fit_learned_feature_matrix(
            train["features"],
            train["relevance_outcome"],
            configuration,
        )
        scores = learned.score_matrix(
            validation["features"], _eligibility(validation, configuration)
        )
        model_bytes = pickle.dumps(learned, protocol=5)
        artifact["model_sha256"] = hashlib.sha256(model_bytes).hexdigest()
    else:
        scores = _deterministic_scores(validation, configuration)
        model_bytes = None
    metrics, _ = _evaluate_columns(validation, scores, threshold)
    artifact["validation_metrics"] = asdict(metrics)
    artifact_content = _canonical_bytes(artifact)
    payload_sha256 = hashlib.sha256(artifact_content).hexdigest()
    artifact_sha256 = output_artifact_sha256 or payload_sha256
    destination = output_root / artifact_sha256
    if (destination / "result.json").is_file():
        existing = load_search_result(output_root, artifact_sha256)
        if (
            existing.trial != trial
            or existing.prepared_semantics_sha256 != prepared.semantics_sha256
        ):
            raise RuntimeError("immutable selector search artifact differs")
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    _write_immutable(destination / "artifact.json", artifact_content)
    _write_immutable(destination / "artifact.sha256", payload_sha256.encode("ascii"))
    if model_bytes is not None:
        _write_immutable(destination / "model.pkl", model_bytes)
    result = SelectorSearchResult(
        trial=trial,
        metrics=metrics,
        relevance_threshold=threshold,
        artifact_sha256=artifact_sha256,
        artifact_payload_sha256=payload_sha256,
        artifact_path=destination,
        prepared_sha256=prepared.sha256,
        prepared_semantics_sha256=prepared.semantics_sha256,
        wall_seconds=time.perf_counter() - started,
    )
    _write_immutable(
        destination / "result.json", _canonical_bytes(_search_result_document(result))
    )
    return result


def run_exact_search(
    prepared: PreparedSelectorData,
    output_root: Path,
) -> tuple[SelectorSearchResult, ...]:
    return tuple(
        run_search_trial(prepared, trial, output_root)
        for trial in compile_selector_search()
    )


def run_gate(
    prepared: PreparedSelectorData,
    results: Sequence[SelectorSearchResult],
    output_path: Path,
    *,
    output_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    _assert_prepared_semantics(prepared, results)
    if len(results) == 48:
        family_counts = {
            family: sum(result.trial.family == family for result in results)
            for family in ("time", "content", "frequency", "learned")
        }
        if set(family_counts.values()) != {12}:
            raise ValueError("selector gate requires 12 trials per family")
        trial_results = [result.to_trial_result() for result in results]
        deterministic = select_strongest_deterministic(trial_results)
        learned = select_family_winner(
            [result for result in trial_results if result.trial.family == "learned"]
        )
        deterministic_result = _result_by_sha(results, deterministic.artifact_sha256)
        learned_result = _result_by_sha(results, learned.artifact_sha256)
    elif len(results) == 2:
        deterministic_matches = [
            result
            for result in results
            if result.trial.family in {"time", "content", "frequency"}
        ]
        learned_matches = [
            result for result in results if result.trial.family == "learned"
        ]
        if len(deterministic_matches) != 1 or len(learned_matches) != 1:
            raise ValueError("locked gate inputs must be deterministic and learned")
        deterministic_result = deterministic_matches[0]
        learned_result = learned_matches[0]
    else:
        raise ValueError(
            "selector gate requires 48 search trials or two locked winners"
        )
    deterministic_test = _load_partition(
        prepared.width_path(
            deterministic_result.trial.configuration.period_width_seconds
        ),
        "test",
    )
    learned_test = _load_partition(
        prepared.width_path(learned_result.trial.configuration.period_width_seconds),
        "test",
    )
    _assert_common_universe(deterministic_test, learned_test)
    deterministic_scores = _deterministic_scores(
        deterministic_test, deterministic_result.trial.configuration
    )
    learned_model = _load_learned_model(learned_result)
    learned_scores = learned_model.score_matrix(
        learned_test["features"],
        _eligibility(learned_test, learned_result.trial.configuration),
    )
    deterministic_metrics, deterministic_queries = _evaluate_columns(
        deterministic_test,
        deterministic_scores,
        deterministic_result.relevance_threshold,
    )
    learned_metrics, learned_queries = _evaluate_columns(
        learned_test,
        learned_scores,
        learned_result.relevance_threshold,
    )
    if deterministic_queries.keys() != learned_queries.keys():
        raise ValueError("gate selectors have different test queries")
    user_differences: dict[int, list[float]] = {}
    for query, learned_ndcg in learned_queries.items():
        user_differences.setdefault(query[0], []).append(
            learned_ndcg - deterministic_queries[query]
        )
    gate = _bootstrap_user_differences(user_differences)
    document = {
        "version": GATE_VERSION,
        "output_artifact_sha256": output_artifact_sha256,
        "prepared_sha256": prepared.sha256,
        "prepared_semantics_sha256": prepared.semantics_sha256,
        "deterministic": {
            "artifact_sha256": deterministic_result.artifact_sha256,
            "artifact_payload_sha256": (deterministic_result.artifact_payload_sha256),
            "trial": deterministic_result.trial.to_dict(),
            "test_metrics": asdict(deterministic_metrics),
        },
        "learned": {
            "artifact_sha256": learned_result.artifact_sha256,
            "artifact_payload_sha256": learned_result.artifact_payload_sha256,
            "trial": learned_result.trial.to_dict(),
            "test_metrics": asdict(learned_metrics),
        },
        "bootstrap": asdict(gate),
        "passes": gate.passes,
    }
    content = _canonical_bytes(document)
    _write_immutable(output_path, content)
    _write_immutable(
        output_path.with_name("gate.sha256"),
        hashlib.sha256(content).hexdigest().encode("ascii"),
    )
    return document


def run_materialization_fold(
    prepared: PreparedSelectorData,
    learned_result: SelectorSearchResult,
    fold_id: int,
    output_root: Path,
    *,
    output_artifact_sha256: str,
) -> Path:
    if fold_id not in range(5):
        raise ValueError("selector materialization fold must be in [0, 5)")
    _assert_prepared_semantics(prepared, (learned_result,))
    if learned_result.trial.family != "learned":
        raise ValueError("fold materialization requires the learned winner")
    destination = output_root / output_artifact_sha256
    if destination.exists():
        manifest, _ = _verify_materialization_fold(output_root, output_artifact_sha256)
        if (
            manifest["fold_id"] != fold_id
            or manifest["prepared_semantics_sha256"] != prepared.semantics_sha256
            or manifest["learned_artifact_sha256"] != learned_result.artifact_sha256
        ):
            raise RuntimeError("immutable selector fold artifact differs")
        return destination
    output_root.mkdir(parents=True, exist_ok=True)
    partial_directory = output_root / (
        f".{output_artifact_sha256}.partial-{os.getpid()}-{time.time_ns()}"
    )
    partial_directory.mkdir()
    source_path = prepared.width_path(
        learned_result.trial.configuration.period_width_seconds
    )
    try:
        model, fit_user_count = _fit_materialization_fold_model(
            source_path,
            fold_id,
            learned_result.trial.configuration,
            partial_directory,
        )
        for name in ("fit_features.bin", "fit_outcomes.bin"):
            (partial_directory / name).unlink()
        score_rows, score_user_count = _write_materialization_fold_scores(
            source_path,
            fold_id,
            model,
            partial_directory / "scores.parquet",
        )
        model_bytes = pickle.dumps(model, protocol=5)
        manifest = {
            "version": FOLD_VERSION,
            "output_artifact_sha256": output_artifact_sha256,
            "fold_id": fold_id,
            "prepared_sha256": prepared.sha256,
            "prepared_semantics_sha256": prepared.semantics_sha256,
            "learned_artifact_sha256": learned_result.artifact_sha256,
            "selected_configuration": learned_result.trial.to_dict(),
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "relevance_threshold": model.relevance_threshold,
            "fit_user_count": fit_user_count,
            "score_user_count": score_user_count,
            "score_rows": score_rows,
            "scores": _file_identity(partial_directory / "scores.parquet"),
            "materialization_backend": "g4-disk-streaming-v1",
        }
        _write_immutable(partial_directory / "model.pkl", model_bytes)
        _write_immutable(
            partial_directory / "manifest.json", _canonical_bytes(manifest)
        )
        os.replace(partial_directory, destination)
        return destination
    except BaseException:
        if partial_directory.exists():
            _remove_tree(partial_directory)
        raise


def _fit_materialization_fold_model(
    source_path: Path,
    fold_id: int,
    configuration: SelectorConfiguration,
    scratch_directory: Path,
    *,
    batch_rows: int = 65_536,
) -> tuple[LearnedSelector, int]:
    import duckdb

    source = _sql_file(source_path)
    order = ", ".join(CANDIDATE_IDENTITY_COLUMNS)
    features_sql = ", ".join(FEATURE_NAMES)
    connection = duckdb.connect()
    try:
        row_count, user_count = connection.execute(
            f"SELECT count(*), count(DISTINCT query_uid) "
            f"FROM read_parquet('{source}') WHERE fold <> {fold_id}"
        ).fetchone()
        if row_count == 0:
            raise ValueError(f"fold {fold_id} has no fit rows")
        features = np.memmap(
            scratch_directory / "fit_features.bin",
            dtype=np.float64,
            mode="w+",
            shape=(int(row_count), len(FEATURE_NAMES)),
        )
        outcomes = np.memmap(
            scratch_directory / "fit_outcomes.bin",
            dtype=np.float64,
            mode="w+",
            shape=(int(row_count),),
        )
        reader = connection.execute(
            f"SELECT {features_sql}, relevance_outcome "
            f"FROM read_parquet('{source}') WHERE fold <> {fold_id} "
            f"ORDER BY {order}"
        ).fetch_record_batch(rows_per_batch=batch_rows)
        position = 0
        for batch in reader:
            columns = batch.to_pydict()
            end = position + len(batch)
            for feature_index, name in enumerate(FEATURE_NAMES):
                features[position:end, feature_index] = columns[name]
            outcomes[position:end] = columns["relevance_outcome"]
            position = end
        if position != row_count:
            raise RuntimeError("fold fit stream row count differs")
        features.flush()
        outcomes.flush()
        return (
            fit_learned_feature_matrix(features, outcomes, configuration),
            int(user_count),
        )
    finally:
        connection.close()


def _write_materialization_fold_scores(
    source_path: Path,
    fold_id: int,
    model: LearnedSelector,
    destination: Path,
    *,
    batch_rows: int = 65_536,
) -> tuple[int, int]:
    import duckdb

    source = _sql_file(source_path)
    order = ", ".join(CANDIDATE_IDENTITY_COLUMNS)
    selected = (
        *CANDIDATE_IDENTITY_COLUMNS,
        "query_next_item",
        "fold",
        "base_eligible",
        "candidate_like_count",
        *FEATURE_NAMES,
    )
    connection = duckdb.connect()
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        user_count = int(
            connection.execute(
                f"SELECT count(DISTINCT query_uid) "
                f"FROM read_parquet('{source}') WHERE fold = {fold_id}"
            ).fetchone()[0]
        )
        reader = connection.execute(
            f"SELECT {', '.join(selected)} FROM read_parquet('{source}') "
            f"WHERE fold = {fold_id} ORDER BY {order}"
        ).fetch_record_batch(rows_per_batch=batch_rows)
        for batch in reader:
            columns = batch.to_pydict()
            features = np.column_stack(
                [np.asarray(columns[name], dtype=np.float64) for name in FEATURE_NAMES]
            )
            eligibility_data = {
                name: np.asarray(columns[name])
                for name in (
                    "query_timestamp",
                    "candidate_timestamp",
                    "candidate_like_count",
                    "base_eligible",
                )
            }
            eligibility_data["features"] = features
            scores = model.score_matrix(
                features, _eligibility(eligibility_data, model.configuration)
            )
            table = pa.Table.from_pydict(
                {
                    name: columns[name]
                    for name in (
                        *CANDIDATE_IDENTITY_COLUMNS,
                        "query_next_item",
                        "fold",
                    )
                }
                | {"score": scores}
            )
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema)
            writer.write_table(table, row_group_size=len(table))
            rows += len(table)
        if writer is None:
            raise ValueError(f"fold {fold_id} has no score rows")
        return rows, user_count
    finally:
        if writer is not None:
            writer.close()
        connection.close()


def _sql_file(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def open_materialization_fold(
    output_root: Path, output_artifact_sha256: str
) -> tuple[Mapping[str, Any], pl.DataFrame]:
    manifest, score_path = _verify_materialization_fold(
        output_root, output_artifact_sha256
    )
    frame = pl.read_parquet(score_path)
    if len(frame) != manifest["score_rows"]:
        raise ValueError("selector fold row count differs")
    if set(frame["fold"].to_list()) != {manifest["fold_id"]}:
        raise ValueError("selector fold artifact contains another fold")
    return manifest, frame


def _verify_materialization_fold(
    output_root: Path, output_artifact_sha256: str
) -> tuple[Mapping[str, Any], Path]:
    path = output_root / output_artifact_sha256
    manifest = json.loads((path / "manifest.json").read_bytes())
    if (
        manifest.get("version") != FOLD_VERSION
        or manifest.get("output_artifact_sha256") != output_artifact_sha256
    ):
        raise ValueError("selector fold artifact identity differs")
    identity = manifest["scores"]
    actual = _file_identity(path / identity["file"])
    if any(actual[key] != identity[key] for key in actual):
        raise ValueError("selector fold score artifact differs")
    model = (path / "model.pkl").read_bytes()
    if hashlib.sha256(model).hexdigest() != manifest["model_sha256"]:
        raise ValueError("selector fold model differs")
    return manifest, path / identity["file"]


def finalize_materialization_folds(
    prepared: PreparedSelectorData,
    gate: Mapping[str, Any],
    deterministic_result: SelectorSearchResult,
    learned_result: SelectorSearchResult,
    fold_artifact_sha256s: Sequence[str],
    *,
    fold_root: Path = FOLD_ROOT,
    output_root: Path = DEFAULT_PERIOD_ARTIFACT_ROOT,
    provenance: Mapping[str, Any] | None = None,
    cost: Mapping[str, Any] | None = None,
) -> tuple[PeriodArtifactIdentity, PeriodArtifactIdentity]:
    if gate.get("passes") is not True:
        raise ValueError("selector quality gate did not pass")
    if (
        gate.get("deterministic", {}).get("artifact_sha256")
        != deterministic_result.artifact_sha256
        or gate.get("deterministic", {}).get("artifact_payload_sha256")
        != deterministic_result.artifact_payload_sha256
        or gate.get("learned", {}).get("artifact_sha256")
        != learned_result.artifact_sha256
        or gate.get("learned", {}).get("artifact_payload_sha256")
        != learned_result.artifact_payload_sha256
    ):
        raise ValueError("selector gate winners differ from materialization inputs")
    if len(fold_artifact_sha256s) != 5 or len(set(fold_artifact_sha256s)) != 5:
        raise ValueError("exactly five distinct fold artifacts are required")
    _assert_prepared_semantics(prepared, (deterministic_result, learned_result))
    manifests_and_paths = [
        _verify_materialization_fold(fold_root, sha256)
        for sha256 in fold_artifact_sha256s
    ]
    manifests = [item[0] for item in manifests_and_paths]
    if {int(manifest["fold_id"]) for manifest in manifests} != set(range(5)):
        raise ValueError("selector fold artifacts do not cover folds 0..4")
    for manifest in manifests:
        if (
            manifest["prepared_semantics_sha256"] != prepared.semantics_sha256
            or manifest["learned_artifact_sha256"] != learned_result.artifact_sha256
        ):
            raise ValueError("selector fold artifact provenance differs")
    expected_rows = prepared.manifest["widths"][
        str(learned_result.trial.configuration.period_width_seconds)
    ]["rows"]
    if sum(int(manifest["score_rows"]) for manifest in manifests) != expected_rows:
        raise ValueError("selector fold artifacts do not cover every candidate row")
    score_paths = [item[1] for item in manifests_and_paths]
    _verify_fold_score_coverage(
        prepared.width_path(learned_result.trial.configuration.period_width_seconds),
        score_paths,
        scratch_root=output_root,
    )
    shared_provenance = {
        "prepared_sha256": prepared.sha256,
        "prepared_semantics_sha256": prepared.semantics_sha256,
        "gate_sha256": hashlib.sha256(_canonical_bytes(gate)).hexdigest(),
        **dict(provenance or {}),
    }
    deterministic = write_period_artifact(
        _stream_deterministic_period_queries(
            prepared.query_path,
            prepared.width_path(
                deterministic_result.trial.configuration.period_width_seconds
            ),
            deterministic_result.trial.configuration,
            scratch_root=output_root,
        ),
        selector_kind="deterministic",
        selected_configuration=deterministic_result.trial.to_dict(),
        provenance=shared_provenance
        | {"materialization_backend": "g4-disk-streaming-v1"},
        cost=dict(cost or {}),
        output_root=output_root,
    )
    learned = write_period_artifact(
        _stream_learned_period_queries(
            prepared.query_path, score_paths, scratch_root=output_root
        ),
        selector_kind="learned",
        selected_configuration=learned_result.trial.to_dict(),
        provenance=shared_provenance
        | {
            "materialization_backend": "g4-disk-streaming-v1",
            "fold_artifact_sha256s": list(fold_artifact_sha256s),
            "fold_models": [
                {
                    "fold": manifest["fold_id"],
                    "model_sha256": manifest["model_sha256"],
                    "relevance_threshold": manifest["relevance_threshold"],
                }
                for manifest in sorted(manifests, key=lambda item: item["fold_id"])
            ],
        },
        cost=dict(cost or {}),
        output_root=output_root,
    )
    return deterministic, learned


def _verify_fold_score_coverage(
    prepared_path: Path,
    score_paths: Sequence[Path],
    *,
    scratch_root: Path,
) -> None:
    import duckdb

    scratch_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".g4-coverage-", dir=scratch_root))
    columns = ", ".join(CANDIDATE_IDENTITY_COLUMNS)
    prepared = _sql_file(prepared_path)
    scores = _sql_path_list(score_paths)
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit = '2GB'")
        connection.execute(f"SET temp_directory = '{_sql_file(temporary)}'")
        differences = connection.execute(
            f"""
            SELECT count(*) FROM (
                (SELECT {columns} FROM read_parquet('{prepared}')
                 EXCEPT ALL SELECT {columns} FROM read_parquet({scores}))
                UNION ALL
                (SELECT {columns} FROM read_parquet({scores})
                 EXCEPT ALL SELECT {columns} FROM read_parquet('{prepared}'))
            )
            """
        ).fetchone()[0]
        if differences:
            raise ValueError(
                "selector fold artifacts do not exactly cover prepared rows"
            )
    finally:
        connection.close()
        _remove_tree(temporary)


def _stream_deterministic_period_queries(
    query_path: Path,
    candidate_path: Path,
    configuration: SelectorConfiguration,
    *,
    scratch_root: Path,
) -> Iterator[ScoredQuery]:
    source = (
        f"SELECT *, {_deterministic_score_sql(configuration)} AS score "
        f"FROM read_parquet('{_sql_file(candidate_path)}')"
    )
    yield from _stream_period_queries(query_path, source, scratch_root=scratch_root)


def _stream_learned_period_queries(
    query_path: Path,
    score_paths: Sequence[Path],
    *,
    scratch_root: Path,
) -> Iterator[ScoredQuery]:
    source = f"SELECT * FROM read_parquet({_sql_path_list(score_paths)})"
    yield from _stream_period_queries(query_path, source, scratch_root=scratch_root)


def _deterministic_score_sql(configuration: SelectorConfiguration) -> str:
    eligible = (
        "base_eligible AND candidate_timestamp <= query_timestamp + "
        f"{configuration.lookahead_seconds} AND candidate_like_count >= "
        f"{configuration.minimum_liked_events}"
    )
    if configuration.family == "time":
        eligible += (
            " AND (1.0 - circular_time_similarity) * 84 * 3600 <= "
            f"{int(configuration.time_tolerance_seconds or 0)} + 1e-9"
        )
        value = "1.0"
    elif configuration.family == "content":
        value = "content_similarity"
    elif configuration.family == "frequency":
        value = f"{configuration.frequency_entity}_jaccard"
    else:
        raise ValueError("learned selector needs fold scores")
    return f"CASE WHEN {eligible} THEN {value} ELSE 0.0 END"


def _stream_period_queries(
    query_path: Path,
    candidate_source: str,
    *,
    scratch_root: Path,
    batch_rows: int = 65_536,
) -> Iterator[ScoredQuery]:
    import duckdb

    scratch_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".g4-stream-", dir=scratch_root))
    connection = duckdb.connect()
    current_key: tuple[int, int, int, int] | None = None
    current_next_item = 0
    current_fold = 0
    period_key: tuple[int, int] | None = None
    period_score = 0.0
    occurrences: list[ScoredOccurrence] = []
    periods: list[ScoredPeriod] = []

    def finish_period() -> None:
        nonlocal period_key, period_score, occurrences
        if period_key is not None and period_score > 0.0:
            periods.append(
                ScoredPeriod(
                    period_key[0], period_key[1], period_score, tuple(occurrences)
                )
            )
        period_key = None
        period_score = 0.0
        occurrences = []

    def finish_query() -> ScoredQuery | None:
        if current_key is None:
            return None
        finish_period()
        return ScoredQuery(
            current_key[0],
            current_key[1],
            current_key[2],
            current_key[3],
            current_next_item,
            current_fold,
            tuple(periods),
        )

    try:
        connection.execute("SET memory_limit = '2GB'")
        connection.execute(f"SET temp_directory = '{_sql_file(temporary)}'")
        reader = connection.execute(
            f"""
            SELECT q.uid, q.prefix_timestamp, q.prefix_item_id,
                   q.occurrence_position, q.next_item, q.fold,
                   c.period_start, c.period_end, c.score,
                   c.candidate_timestamp, c.candidate_item, c.candidate_position
            FROM read_parquet('{_sql_file(query_path)}') q
            LEFT JOIN ({candidate_source}) c
              ON q.uid = c.query_uid
             AND q.prefix_timestamp = c.query_timestamp
             AND q.prefix_item_id = c.query_item
             AND q.occurrence_position = c.query_position
            ORDER BY q.uid, q.prefix_timestamp, q.prefix_item_id,
                     q.occurrence_position, c.period_start, c.period_end,
                     c.candidate_timestamp, c.candidate_item, c.candidate_position
            """
        ).fetch_record_batch(rows_per_batch=batch_rows)
        for batch in reader:
            columns = batch.to_pydict()
            for index in range(len(batch)):
                key = (
                    int(columns["uid"][index]),
                    int(columns["prefix_timestamp"][index]),
                    int(columns["prefix_item_id"][index]),
                    int(columns["occurrence_position"][index]),
                )
                if key != current_key:
                    completed = finish_query()
                    if completed is not None:
                        yield completed
                    current_key = key
                    current_next_item = int(columns["next_item"][index])
                    current_fold = int(columns["fold"][index])
                    periods = []
                if columns["period_start"][index] is None:
                    continue
                next_period_key = (
                    int(columns["period_start"][index]),
                    int(columns["period_end"][index]),
                )
                if next_period_key != period_key:
                    finish_period()
                    period_key = next_period_key
                score = float(columns["score"][index])
                period_score = max(period_score, score)
                if score > 0.0:
                    occurrences.append(
                        ScoredOccurrence(
                            int(columns["candidate_timestamp"][index]),
                            int(columns["candidate_item"][index]),
                            int(columns["candidate_position"][index]),
                        )
                    )
        completed = finish_query()
        if completed is not None:
            yield completed
    finally:
        connection.close()
        _remove_tree(temporary)


def _sql_path_list(paths: Sequence[Path]) -> str:
    if not paths:
        raise ValueError("at least one score path is required")
    return "[" + ",".join(f"'{_sql_file(path)}'" for path in paths) + "]"


def load_search_result(
    output_root: Path,
    artifact_sha256: str,
    *,
    expected_payload_sha256: str | None = None,
) -> SelectorSearchResult:
    artifact_path = output_root / artifact_sha256
    artifact_content = (artifact_path / "artifact.json").read_bytes()
    payload_sha256 = hashlib.sha256(artifact_content).hexdigest()
    if (artifact_path / "artifact.sha256").read_text() != payload_sha256:
        raise ValueError("selector search artifact payload hash differs")
    if (
        expected_payload_sha256 is not None
        and payload_sha256 != expected_payload_sha256
    ):
        raise ValueError("selector search artifact differs from the frozen payload")
    artifact = json.loads(artifact_content)
    if artifact.get("version") != SEARCH_VERSION:
        raise ValueError("selector search artifact version differs")
    declared_output = artifact.get("output_artifact_sha256")
    if declared_output not in {None, artifact_sha256}:
        raise ValueError("selector search output slot differs")
    trial = selector_trial_from_job(artifact["trial"])
    if (
        artifact.get("sampler_seed") != trial.sampler_seed
        or artifact.get("classifier_seed") != 42
    ):
        raise ValueError("selector search random-seed provenance differs")
    result_document = json.loads((artifact_path / "result.json").read_bytes())
    if result_document.get("artifact_sha256") != artifact_sha256:
        raise ValueError("selector search result identity differs")
    immutable_result = {
        "version": SEARCH_VERSION,
        "trial": artifact["trial"],
        "validation_metrics": artifact["validation_metrics"],
        "relevance_threshold": artifact["relevance_threshold"],
        "artifact_sha256": artifact_sha256,
        "artifact_payload_sha256": payload_sha256,
        "prepared_sha256": artifact["prepared_sha256"],
        "prepared_semantics_sha256": artifact["prepared_semantics_sha256"],
    }
    if any(
        result_document.get(name) != value for name, value in immutable_result.items()
    ):
        raise ValueError("selector search result differs from its hashed artifact")
    wall_seconds = result_document.get("wall_seconds")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(wall_seconds)
        or wall_seconds < 0
    ):
        raise ValueError("selector search runtime evidence is invalid")
    metrics = SelectorMetrics(**artifact["validation_metrics"])
    return SelectorSearchResult(
        trial=trial,
        metrics=metrics,
        relevance_threshold=float(artifact["relevance_threshold"]),
        artifact_sha256=artifact_sha256,
        artifact_payload_sha256=payload_sha256,
        artifact_path=artifact_path,
        prepared_sha256=artifact["prepared_sha256"],
        prepared_semantics_sha256=artifact["prepared_semantics_sha256"],
        wall_seconds=float(wall_seconds),
    )


def load_gate_result(
    output_root: Path,
    artifact_sha256: str,
    *,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    artifact_path = output_root / artifact_sha256
    content = (artifact_path / "gate.json").read_bytes()
    payload_sha256 = hashlib.sha256(content).hexdigest()
    if payload_sha256 != expected_payload_sha256:
        raise ValueError("selector gate differs from the frozen payload")
    if (artifact_path / "gate.sha256").read_text() != payload_sha256:
        raise ValueError("selector gate payload hash differs")
    document = json.loads(content)
    if (
        document.get("version") != GATE_VERSION
        or document.get("output_artifact_sha256") != artifact_sha256
    ):
        raise ValueError("selector gate identity differs")
    bootstrap = document.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("selector gate bootstrap evidence is invalid")
    lower_95 = bootstrap.get("lower_95")
    if (
        not isinstance(lower_95, (int, float))
        or isinstance(lower_95, bool)
        or not math.isfinite(lower_95)
    ):
        raise ValueError("selector gate bootstrap evidence is invalid")
    recomputed_passes = lower_95 > 0.0
    if (
        bootstrap.get("passes") is not recomputed_passes
        or document.get("passes") is not recomputed_passes
    ):
        raise ValueError("selector gate decision differs from bootstrap evidence")
    for name in ("deterministic", "learned"):
        selected = document.get(name)
        if not isinstance(selected, dict):
            raise ValueError("selector gate winner evidence is invalid")
        for identity in ("artifact_sha256", "artifact_payload_sha256"):
            value = selected.get(identity)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("selector gate winner identity is invalid")
    return document


def run_crossfit_materialization(
    prepared: PreparedSelectorData,
    gate: Mapping[str, Any],
    search_results: Sequence[SelectorSearchResult],
    *,
    output_root: Path = DEFAULT_PERIOD_ARTIFACT_ROOT,
    provenance: Mapping[str, Any] | None = None,
    cost: Mapping[str, Any] | None = None,
) -> tuple[PeriodArtifactIdentity, PeriodArtifactIdentity]:
    if gate.get("passes") is not True:
        raise ValueError("selector quality gate did not pass")
    deterministic_result = _result_by_sha(
        search_results, gate["deterministic"]["artifact_sha256"]
    )
    learned_result = _result_by_sha(search_results, gate["learned"]["artifact_sha256"])
    if (
        gate["deterministic"].get("artifact_payload_sha256")
        != deterministic_result.artifact_payload_sha256
        or gate["learned"].get("artifact_payload_sha256")
        != learned_result.artifact_payload_sha256
    ):
        raise ValueError("selector gate payload identities differ")
    _assert_prepared_semantics(prepared, (deterministic_result, learned_result))
    if gate.get("prepared_semantics_sha256") != prepared.semantics_sha256:
        raise ValueError("selector gate was evaluated on different prepared semantics")
    deterministic_data = _load_partition(
        prepared.width_path(
            deterministic_result.trial.configuration.period_width_seconds
        ),
        None,
    )
    learned_data = _load_partition(
        prepared.width_path(learned_result.trial.configuration.period_width_seconds),
        None,
    )
    deterministic_scores = _deterministic_scores(
        deterministic_data, deterministic_result.trial.configuration
    )
    learned_scores = np.zeros(len(learned_data["relevance_outcome"]), dtype=np.float64)
    fold_models: list[dict[str, Any]] = []
    for scored_fold in range(5):
        fit_mask = learned_data["fold"] != scored_fold
        score_mask = ~fit_mask
        model = fit_learned_feature_matrix(
            learned_data["features"][fit_mask],
            learned_data["relevance_outcome"][fit_mask],
            learned_result.trial.configuration,
        )
        learned_scores[score_mask] = model.score_matrix(
            learned_data["features"][score_mask],
            _eligibility(learned_data, learned_result.trial.configuration)[score_mask],
        )
        model_bytes = pickle.dumps(model, protocol=5)
        fold_models.append(
            {
                "fold": scored_fold,
                "fit_users": sorted(
                    set(int(value) for value in learned_data["query_uid"][fit_mask])
                ),
                "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                "relevance_threshold": model.relevance_threshold,
            }
        )
    shared_provenance = {
        "prepared_sha256": prepared.sha256,
        "prepared_semantics_sha256": prepared.semantics_sha256,
        "gate_sha256": hashlib.sha256(_canonical_bytes(gate)).hexdigest(),
        **dict(provenance or {}),
    }
    queries = pl.read_parquet(prepared.query_path).sort(
        "uid", "prefix_timestamp", "prefix_item_id", "occurrence_position"
    )
    deterministic_identity = write_period_artifact(
        _scored_query_stream(
            queries,
            deterministic_data,
            deterministic_scores,
        ),
        selector_kind="deterministic",
        selected_configuration=deterministic_result.trial.to_dict(),
        provenance=shared_provenance,
        cost=dict(cost or {}),
        output_root=output_root,
    )
    learned_identity = write_period_artifact(
        _scored_query_stream(queries, learned_data, learned_scores),
        selector_kind="learned",
        selected_configuration=learned_result.trial.to_dict(),
        provenance=shared_provenance | {"fold_models": fold_models},
        cost=dict(cost or {}),
        output_root=output_root,
    )
    return deterministic_identity, learned_identity


def run_native500m_feasibility_sample(
    paths: SelectorInputPaths,
    bounds: ChronologicalBounds,
    percent: int,
    *,
    measurement_directory: Path,
    enforce_reference_fixture: bool = True,
) -> dict[str, Any]:
    _native500m_feasibility_sample(percent)
    if measurement_directory.exists() and any(measurement_directory.iterdir()):
        raise ValueError("feasibility measurement directory must be empty")
    with _hold_native_materialization_lock(NATIVE_MATERIALIZATION_LOCK_PATH):
        measurement_directory.mkdir(parents=True, exist_ok=True)
        if enforce_reference_fixture:
            _verify_reference_fixture()
        os.sched_setaffinity(0, set(range(16)))
        result_path = measurement_directory / "child_result.json"
        error_path = measurement_directory / "child_error.txt"
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=_spawn_callable(_native500m_feasibility_child),
            args=(
                paths,
                bounds,
                percent,
                measurement_directory,
                result_path,
                error_path,
            ),
            name=f"g4-native500m-feasibility-{percent}",
        )
        monitor = SupervisedProcessTreeRssMonitor(
            interval_seconds=0.1, logical_root=measurement_directory
        )
        started = time.perf_counter()
        monitor.start()
        try:
            process.start()
            monitor.set_root_pid(process.pid)
            process.join()
            monitor.sample()
        finally:
            monitor.stop()
            exitcode = process.exitcode
            process.close()
        wall_seconds = time.perf_counter() - started
        if exitcode not in {None, 0}:
            detail = error_path.read_text() if error_path.exists() else "no traceback"
            raise RuntimeError(f"native-500M feasibility child failed:\n{detail}")
        result = json.loads(result_path.read_bytes())
        logical_bytes = max(
            monitor.peak_logical_bytes,
            _logical_regular_file_bytes(measurement_directory),
        )
        evidence = {
            "version": "g4-native500m-feasibility-measurement-v1",
            "sample_percent": percent,
            "user_sample": _native500m_feasibility_sample(percent),
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": monitor.peak_bytes,
            "logical_bytes": logical_bytes,
            "materialization_workers": 16,
            "timed_load_samples": list(monitor.load_samples),
            "post_launch_contention": not monitor.load_valid,
            "supervision": {
                "sample_interval_seconds": 0.1,
                "began_before_child_creation": True,
                "ended_after_child_reap": True,
            },
            "selection_eligible": False,
            "full_gate_eligible": False,
            "result": result,
        }
        evidence_path = measurement_directory / "feasibility.json"
        while True:
            content = _canonical_bytes(evidence)
            complete_bytes = logical_bytes + len(content)
            if evidence["logical_bytes"] == complete_bytes:
                break
            evidence["logical_bytes"] = complete_bytes
        _write_immutable(evidence_path, content)
        return evidence


def _native500m_feasibility_child(
    paths: SelectorInputPaths,
    bounds: ChronologicalBounds,
    percent: int,
    measurement_directory: Path,
    result_path: Path,
    error_path: Path,
) -> None:
    try:
        os.sched_setaffinity(0, set(range(16)))
        prepared = prepare_selector_data(
            paths,
            bounds,
            measurement_directory / "prepared",
            provenance={"stage": "native500m_feasibility", "sample_percent": percent},
            workers=16,
            feasibility_percent=percent,
        )
        deterministic_trial, learned_trial = native500m_feasibility_trials()
        search_root = measurement_directory / "search"
        deterministic = run_search_trial(prepared, deterministic_trial, search_root)
        learned = run_search_trial(prepared, learned_trial, search_root)
        gate = {
            "version": "g4-native500m-feasibility-gate-v1",
            "passes": True,
            "prepared_semantics_sha256": prepared.semantics_sha256,
            "deterministic": {
                "artifact_sha256": deterministic.artifact_sha256,
                "artifact_payload_sha256": deterministic.artifact_payload_sha256,
            },
            "learned": {
                "artifact_sha256": learned.artifact_sha256,
                "artifact_payload_sha256": learned.artifact_payload_sha256,
            },
        }
        fold_sha256s = [
            hashlib.sha256(
                _canonical_bytes(
                    [
                        "g4-native500m-feasibility-fold-v1",
                        percent,
                        fold_id,
                        prepared.semantics_sha256,
                        learned.artifact_sha256,
                    ]
                )
            ).hexdigest()
            for fold_id in range(5)
        ]
        fold_root = measurement_directory / "folds"
        for fold_id, sha256 in enumerate(fold_sha256s):
            run_materialization_fold(
                prepared,
                learned,
                fold_id,
                fold_root,
                output_artifact_sha256=sha256,
            )
        deterministic_artifact, learned_artifact = finalize_materialization_folds(
            prepared,
            gate,
            deterministic,
            learned,
            fold_sha256s,
            fold_root=fold_root,
            output_root=measurement_directory / "period_artifacts",
            provenance={
                "stage": "native500m_feasibility",
                "sample_percent": percent,
                "selection_eligible": False,
            },
        )
        _write_immutable(
            result_path,
            _canonical_bytes(
                {
                    "prepared_sha256": prepared.sha256,
                    "fold_artifact_sha256s": fold_sha256s,
                    "deterministic_artifact_sha256": deterministic_artifact.sha256,
                    "learned_artifact_sha256": learned_artifact.sha256,
                    "query_count": learned_artifact.query_count,
                    "period_count": learned_artifact.period_count,
                    "occurrence_count": learned_artifact.occurrence_count,
                }
            ),
        )
    except BaseException:
        error_path.write_text(traceback.format_exc())
        raise


def _load_inconclusive_materialization_attempt(
    path: Path,
    *,
    gate_sha256: str,
    source_identities: Mapping[str, Any],
) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("prior materialization attempt is not a regular file")
    document = json.loads(path.read_bytes())
    if path.read_bytes() != _canonical_bytes(document):
        raise ValueError("prior materialization attempt is not canonical")
    runtime = document.get("runtime")
    limits = document.get("limits")
    if (
        document.get("version") != "g4-materialization-cost-v1"
        or document.get("attempt") != 1
        or document.get("decision") != "inconclusive"
        or document.get("passes") is not False
        or document.get("post_launch_contention") is not True
        or document.get("gate_sha256") != gate_sha256
        or document.get("source_identities") != source_identities
        or limits != MATERIALIZATION_COST_LIMITS
        or not isinstance(runtime, dict)
        or float(runtime.get("wall_seconds", -1)) <= limits["wall_seconds"]
        or int(runtime.get("peak_aggregate_rss_bytes", -1))
        > limits["peak_aggregate_rss_bytes"]
        or int(document.get("logical_output_scratch_bytes", -1))
        > limits["logical_output_scratch_bytes"]
    ):
        raise ValueError("prior materialization attempt is not inconclusive evidence")
    return {"path": str(path.resolve()), "sha256": _file_identity(path)["sha256"]}


def run_native_materialization(
    paths: SelectorInputPaths,
    bounds: ChronologicalBounds,
    gate: Mapping[str, Any],
    search_results: Sequence[SelectorSearchResult],
    fold_artifact_sha256s: Sequence[str],
    *,
    measurement_directory: Path,
    output_root: Path | None = None,
    provenance: Mapping[str, Any] | None = None,
    quiet_window_seconds: float = 600.0,
    load_sample_seconds: float = 60.0,
    enforce_reference_fixture: bool = True,
    attempt: int = 1,
    previous_attempt_path: Path | None = None,
) -> dict[str, Any]:
    if attempt not in {1, 2} or isinstance(attempt, bool):
        raise ValueError("materialization attempt must be 1 or 2")
    if (attempt == 1) != (previous_attempt_path is None):
        raise ValueError("attempt 2 requires exactly one prior inconclusive result")
    with _hold_native_materialization_lock(NATIVE_MATERIALIZATION_LOCK_PATH):
        return _run_native_materialization(
            paths,
            bounds,
            gate,
            search_results,
            fold_artifact_sha256s,
            measurement_directory=measurement_directory,
            output_root=output_root,
            provenance=provenance,
            quiet_window_seconds=quiet_window_seconds,
            load_sample_seconds=load_sample_seconds,
            enforce_reference_fixture=enforce_reference_fixture,
            attempt=attempt,
            previous_attempt_path=previous_attempt_path,
        )


def _run_native_materialization(
    paths: SelectorInputPaths,
    bounds: ChronologicalBounds,
    gate: Mapping[str, Any],
    search_results: Sequence[SelectorSearchResult],
    fold_artifact_sha256s: Sequence[str],
    *,
    measurement_directory: Path,
    output_root: Path | None = None,
    provenance: Mapping[str, Any] | None = None,
    quiet_window_seconds: float = 600.0,
    load_sample_seconds: float = 60.0,
    enforce_reference_fixture: bool = True,
    attempt: int = 1,
    previous_attempt_path: Path | None = None,
) -> dict[str, Any]:
    if len(fold_artifact_sha256s) != 5 or len(set(fold_artifact_sha256s)) != 5:
        raise ValueError("native materialization requires five frozen fold outputs")
    if measurement_directory.exists() and any(measurement_directory.iterdir()):
        raise ValueError("materialization measurement directory must be empty")
    measurement_directory.mkdir(parents=True, exist_ok=True)
    output_root = output_root or measurement_directory / "period_artifacts"
    if not output_root.resolve().is_relative_to(measurement_directory.resolve()):
        raise ValueError(
            "measured period artifacts must stay in the dedicated directory"
        )
    if enforce_reference_fixture:
        _verify_reference_fixture()
    os.sched_setaffinity(0, set(range(16)))
    expected_input_sha256 = _shared_search_input_sha256(search_results)
    source_identities = prewarm_and_verify_inputs(
        paths.paths, expected_sha256=expected_input_sha256
    )
    previous_attempt = (
        None
        if previous_attempt_path is None
        else _load_inconclusive_materialization_attempt(
            previous_attempt_path,
            gate_sha256=hashlib.sha256(_canonical_bytes(gate)).hexdigest(),
            source_identities=source_identities,
        )
    )
    load_samples = wait_for_quiet_materialization_window(
        duration_seconds=quiet_window_seconds,
        sample_seconds=load_sample_seconds,
    )
    for _ in range(2):
        if all_input_pages_resident(paths.paths):
            break
        source_identities = prewarm_and_verify_inputs(
            paths.paths, expected_sha256=expected_input_sha256
        )
    else:
        raise RuntimeError(
            "materialization inputs are not fully resident after prewarm"
        )
    measurement_id = hashlib.sha256(
        _canonical_bytes(
            {
                "revision": "g4-materialization-gate-v1",
                "gate_sha256": hashlib.sha256(_canonical_bytes(gate)).hexdigest(),
                "started_ns": time.time_ns(),
            }
        )
    ).hexdigest()
    child_result = measurement_directory / "child_result.json"
    child_error = measurement_directory / "child_error.txt"
    context = multiprocessing.get_context("spawn")
    monitor = SupervisedProcessTreeRssMonitor(logical_root=measurement_directory)
    started = time.perf_counter()
    monitor.start()
    process = context.Process(
        target=_spawn_callable(_native_materialization_child),
        args=(
            paths,
            bounds,
            gate,
            tuple(search_results),
            tuple(fold_artifact_sha256s),
            measurement_directory,
            output_root,
            dict(provenance or {}),
            measurement_id,
            child_result,
            child_error,
        ),
        name="g4-native-materialization",
    )
    try:
        process.start()
        monitor.set_root_pid(process.pid)
        process.join()
        monitor.sample()
    finally:
        monitor.stop()
        exitcode = process.exitcode
        process.close()
    wall_seconds = time.perf_counter() - started
    if exitcode not in {None, 0}:
        detail = (
            child_error.read_text() if child_error.exists() else "no child traceback"
        )
        raise RuntimeError(f"native materialization child failed:\n{detail}")
    result = json.loads(child_result.read_bytes())
    prepared = open_prepared_selector_data(
        measurement_directory / "prepared", result["prepared_sha256"]
    )
    learned = PeriodArtifactIdentity(
        sha256=result["learned_artifact_sha256"],
        path=output_root / result["learned_artifact_sha256"],
        query_count=result["learned_query_count"],
        period_count=result["learned_period_count"],
        occurrence_count=result["learned_occurrence_count"],
        logical_bytes=result["learned_logical_bytes"],
    )
    deterministic = PeriodArtifactIdentity(
        sha256=result["deterministic_artifact_sha256"],
        path=output_root / result["deterministic_artifact_sha256"],
        query_count=result["deterministic_query_count"],
        period_count=result["deterministic_period_count"],
        occurrence_count=result["deterministic_occurrence_count"],
        logical_bytes=result["deterministic_logical_bytes"],
    )
    logical_bytes = max(
        monitor.peak_logical_bytes,
        _logical_regular_file_bytes(measurement_directory),
    )
    evidence: dict[str, Any] = {
        "version": "g4-materialization-cost-v1",
        "measurement_id": measurement_id,
        "gate_sha256": hashlib.sha256(_canonical_bytes(gate)).hexdigest(),
        "prepared_sha256": prepared.sha256,
        "deterministic_artifact_sha256": deterministic.sha256,
        "learned_artifact_sha256": learned.sha256,
        "runtime": runtime_evidence(monitor, wall_seconds),
        "logical_output_scratch_bytes": logical_bytes,
        "quiet_window_load_samples": list(load_samples),
        "timed_load_samples": list(monitor.load_samples),
        "timed_load_valid": monitor.load_valid,
        "post_launch_contention": not monitor.load_valid,
        "attempt": attempt,
        "source_identities": source_identities,
        "materialization_workers": 16,
        "hardware": _reference_hardware(),
        "command": [sys.executable, *sys.argv],
        "supervision": {
            "sample_interval_seconds": 0.1,
            "began_before_child_creation": True,
            "ended_after_child_reap": True,
        },
        "limits": dict(MATERIALIZATION_COST_LIMITS),
    }
    if previous_attempt is not None:
        evidence["previous_attempt"] = previous_attempt
    evidence_path = measurement_directory / "materialization_cost.json"
    while True:
        evidence["decision"] = classify_native500m_materialization_cost(
            wall_seconds=wall_seconds,
            peak_rss_bytes=monitor.peak_bytes,
            logical_bytes=evidence["logical_output_scratch_bytes"],
            post_launch_contention=not monitor.load_valid,
            attempt=attempt,
            limits=evidence["limits"],
        )
        evidence["passes"] = evidence["decision"] == "pass"
        content = _canonical_bytes(evidence)
        complete_logical_bytes = logical_bytes + len(content)
        if evidence["logical_output_scratch_bytes"] == complete_logical_bytes:
            break
        evidence["logical_output_scratch_bytes"] = complete_logical_bytes
    _write_immutable(evidence_path, content)
    if evidence["passes"]:
        _promote_measured_artifacts(
            output_root,
            (deterministic.sha256, learned.sha256),
            DEFAULT_PERIOD_ARTIFACT_ROOT,
        )
    return evidence


def run_native_materialization_from_ledger(
    *,
    ledger_path: Path,
    semantics_paths: Mapping[str, Path],
    measurement_directory: Path,
    attempt: int = 1,
    previous_attempt_path: Path | None = None,
) -> dict[str, Any]:
    from experiments.g4_future_items.protocol.manifest import (
        load_ledger,
        verify_ledger_semantics,
    )

    ledger = load_ledger(ledger_path)
    if ledger["stage"] != "selector_materialization":
        raise ValueError("native materialization requires its five-fold ledger")
    verify_ledger_semantics(ledger, dict(semantics_paths))
    rows = ledger["rows"]
    gate_artifacts = {row["job"]["input_artifact_sha256"] for row in rows}
    gate_payloads = {row["job"]["input_payload_sha256"] for row in rows}
    if len(gate_artifacts) != 1 or len(gate_payloads) != 1:
        raise ValueError("materialization folds reference different selector gates")
    gate_artifact_sha256 = gate_artifacts.pop()
    gate = load_gate_result(
        GATE_ROOT,
        gate_artifact_sha256,
        expected_payload_sha256=gate_payloads.pop(),
    )
    if gate.get("passes") is not True:
        raise ValueError("selector quality gate is absent or did not pass")
    deterministic = load_search_result(
        SEARCH_ROOT,
        gate["deterministic"]["artifact_sha256"],
        expected_payload_sha256=gate["deterministic"]["artifact_payload_sha256"],
    )
    learned = load_search_result(
        SEARCH_ROOT,
        gate["learned"]["artifact_sha256"],
        expected_payload_sha256=gate["learned"]["artifact_payload_sha256"],
    )
    configuration_fields = (
        "family",
        "period_width_seconds",
        "lookahead_seconds",
        "minimum_liked_events",
        "time_tolerance_seconds",
        "frequency_entity",
        "max_leaf_nodes",
        "learning_rate",
        "l2_regularization",
    )
    expected_configuration = asdict(learned.trial.configuration)
    if any(
        {name: row["job"][name] for name in configuration_fields}
        != expected_configuration
        for row in rows
    ):
        raise ValueError("materialization ledger does not freeze the learned winner")
    prepared_sha256 = _shared_prepared_sha256((deterministic, learned))
    prepared = open_prepared_selector_data(PREPARED_ROOT, prepared_sha256)
    paths = _selector_input_paths(prepared)
    bounds = ChronologicalBounds(
        **{
            name: TimePartition(name, values["start"], values["end"])
            for name, values in prepared.manifest["bounds"].items()
        }
    )
    evidence = run_native_materialization(
        paths,
        bounds,
        gate,
        (deterministic, learned),
        [row["job"]["output_artifact_sha256"] for row in rows],
        measurement_directory=measurement_directory,
        provenance={
            "materialization_ledger_sha256": ledger["sha256"],
            "selector_gate_artifact_sha256": gate_artifact_sha256,
        },
        attempt=attempt,
        previous_attempt_path=previous_attempt_path,
    )
    if evidence.get("decision") == "inconclusive":
        raise RuntimeError(
            "native materialization wall time is inconclusive; one fresh quiet rerun is permitted"
        )
    if evidence["passes"] is not True:
        raise RuntimeError("native materialization cost gate did not pass")
    return evidence


def _selector_input_paths(prepared: PreparedSelectorData) -> SelectorInputPaths:
    from experiments.g4_future_items.protocol.manifest import (
        canonical_sha256,
        load_strict_json,
    )

    control = load_strict_json(CONTROL_SEMANTICS_MANIFEST_PATH)
    if canonical_sha256(control) != prepared.manifest["provenance"].get(
        "control_semantics_manifest_sha256"
    ):
        raise ValueError("prepared selector control semantics differ")
    data = control["data_identity"]
    raw_identity = prepared.manifest["inputs"]["raw_events"]
    raw_name = raw_identity["file"]
    if Path(raw_name).name != raw_name:
        raise ValueError("prepared raw-event filename is invalid")
    size = control["resolved_anchor_configuration"]["fixed"]["data"]["size"]
    paths = SelectorInputPaths(
        control_likes=Path(data["main"]["path"]),
        raw_events=PROJECT_ROOT / "generated/yambda_data/flat" / size / raw_name,
        item_id_remap=Path(data["remap"]["path"]),
        compact_embeddings=Path(data["content_embeddings"]["path"]),
    )
    paths.validate()
    names = (
        "control_likes",
        "raw_events",
        "item_id_remap",
        "compact_embeddings",
    )
    for name, path in zip(names, paths.paths):
        if _file_identity(path) != prepared.manifest["inputs"][name]:
            raise ValueError(f"prepared selector input identity differs: {name}")
    return paths


def _native_materialization_child(
    paths: SelectorInputPaths,
    bounds: ChronologicalBounds,
    gate: Mapping[str, Any],
    search_results: tuple[SelectorSearchResult, ...],
    fold_artifact_sha256s: tuple[str, ...],
    measurement_directory: Path,
    output_root: Path,
    provenance: Mapping[str, Any],
    measurement_id: str,
    result_path: Path,
    error_path: Path,
) -> None:
    try:
        os.sched_setaffinity(0, set(range(16)))
        prepared = prepare_selector_data(
            paths,
            bounds,
            measurement_directory / "prepared",
            provenance=provenance,
            workers=16,
        )
        deterministic_result = _result_by_sha(
            search_results, gate["deterministic"]["artifact_sha256"]
        )
        learned_result = _result_by_sha(
            search_results, gate["learned"]["artifact_sha256"]
        )
        fold_root = measurement_directory / "folds"
        for fold_id, artifact_sha256 in enumerate(fold_artifact_sha256s):
            run_materialization_fold(
                prepared,
                learned_result,
                fold_id,
                fold_root,
                output_artifact_sha256=artifact_sha256,
            )
        deterministic, learned = finalize_materialization_folds(
            prepared,
            gate,
            deterministic_result,
            learned_result,
            fold_artifact_sha256s,
            fold_root=fold_root,
            output_root=output_root,
            provenance=provenance,
            cost={"measurement_id": measurement_id},
        )
        from experiments.g4_future_items.protocol.materialization import PeriodArtifact

        PeriodArtifact.open(output_root, expected_sha256=deterministic.sha256)
        PeriodArtifact.open(output_root, expected_sha256=learned.sha256)
        result = {
            "prepared_sha256": prepared.sha256,
            "deterministic_artifact_sha256": deterministic.sha256,
            "deterministic_query_count": deterministic.query_count,
            "deterministic_period_count": deterministic.period_count,
            "deterministic_occurrence_count": deterministic.occurrence_count,
            "deterministic_logical_bytes": deterministic.logical_bytes,
            "learned_artifact_sha256": learned.sha256,
            "learned_query_count": learned.query_count,
            "learned_period_count": learned.period_count,
            "learned_occurrence_count": learned.occurrence_count,
            "learned_logical_bytes": learned.logical_bytes,
        }
        _write_immutable(result_path, _canonical_bytes(result))
    except BaseException:
        error_path.write_text(traceback.format_exc())
        raise


def _iter_prepared_user_documents(
    paths: SelectorInputPaths,
    bounds: ChronologicalBounds,
    workers: int,
    *,
    feasibility_percent: int | None = None,
) -> Iterator[
    tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, int]]
]:
    selected_user_ids = None
    if feasibility_percent is not None:
        _native500m_feasibility_sample(feasibility_percent)
        selected_user_ids = [
            int(uid)
            for uid in (
                pl.scan_parquet(paths.control_likes)
                .select("uid")
                .unique()
                .collect(engine="streaming")["uid"]
            )
            if native500m_feasibility_user_selected(int(uid), feasibility_percent)
        ]
    users = iter_selector_users(
        paths,
        start_timestamp=bounds.train.start,
        cutoff_timestamp=bounds.test.end,
        selected_user_ids=selected_user_ids,
    )
    arguments = ((user, bounds) for user in users)
    if workers == 1:
        for argument in arguments:
            yield _prepare_user_documents(argument)
        return
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        workers, initializer=_spawn_callable(_pin_materialization_worker)
    ) as pool:
        yield from pool.imap(
            _spawn_callable(_prepare_user_documents), arguments, chunksize=1
        )


def _native500m_feasibility_sample(percent: int | None) -> dict[str, int | str] | None:
    if percent is None:
        return None
    if percent not in FEASIBILITY_SAMPLE_PERCENTS:
        raise ValueError("feasibility sample percent must be 5 or 10")
    return {
        "revision": "g4-feasibility-v1",
        "seed": 42,
        "percent": percent,
        "modulus": 100 // percent,
        "remainder": 0,
    }


def _pin_materialization_worker() -> None:
    identity = multiprocessing.current_process()._identity
    worker_index = (identity[-1] - 1) % 16 if identity else 0
    os.sched_setaffinity(0, {worker_index})


def _spawn_callable(function):
    if function.__module__ != CANONICAL_MODULE:
        function.__module__ = CANONICAL_MODULE
    return function


def _prepare_user_documents(
    argument: tuple[SelectorUserEvents, ChronologicalBounds],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, int]]:
    user, bounds = argument
    indexed_likes = sorted(
        enumerate(user.likes), key=lambda pair: (pair[1].timestamp, pair[0])
    )
    query_documents = [
        {
            "uid": user.uid,
            "prefix_timestamp": like.timestamp,
            "prefix_item_id": like.item_id,
            "occurrence_position": position,
            "next_item": indexed_likes[position + 1][1].item_id,
            "fold": fold_for_user(user.uid),
        }
        for position, (_, like) in enumerate(indexed_likes[:-1])
    ]
    next_items = {
        position: indexed_likes[position + 1][1].item_id
        for position in range(len(indexed_likes) - 1)
    }
    documents_by_width: dict[int, list[dict[str, Any]]] = {}
    query_counts_by_width: dict[int, int] = {}
    for width in WIDTHS:
        neutral = SelectorConfiguration(
            family="content",
            period_width_seconds=width,
            lookahead_seconds=28 * DAY_SECONDS,
            minimum_liked_events=1,
        )
        examples = build_selector_examples(user.likes, user.listens, bounds, neutral)
        documents_by_width[width] = [
            _example_document(
                example,
                bounds,
                next_items[example.query.occurrence_ordinal],
            )
            for example in examples
        ]
        query_counts_by_width[width] = len({example.query for example in examples})
    return query_documents, documents_by_width, query_counts_by_width


class SupervisedProcessTreeRssMonitor:
    def __init__(
        self,
        interval_seconds: float = 0.1,
        load_interval_seconds: float = 60.0,
        logical_root: Path | None = None,
    ):
        self.interval_seconds = interval_seconds
        self.load_interval_seconds = load_interval_seconds
        self.logical_root = logical_root
        self.peak_bytes = 0
        self.peak_logical_bytes = 0
        self.load_samples: list[float] = []
        self.load_valid = True
        self._root_pid: int | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._next_load_sample = 0.0
        self._consecutive_overload = 0

    def start(self) -> None:
        self._thread.start()

    def set_root_pid(self, root_pid: int) -> None:
        with self._lock:
            self._root_pid = root_pid
            self._next_load_sample = time.monotonic()
        self.sample()

    def sample(self) -> None:
        with self._lock:
            root_pid = self._root_pid
            if root_pid is not None:
                self.peak_bytes = max(self.peak_bytes, _process_tree_rss(root_pid))
                if self.logical_root is not None:
                    self.peak_logical_bytes = max(
                        self.peak_logical_bytes,
                        _logical_regular_file_bytes(self.logical_root),
                    )
            now = time.monotonic()
            if root_pid is not None and now >= self._next_load_sample:
                load = float(os.getloadavg()[0])
                self.load_samples.append(load)
                self._consecutive_overload = (
                    self._consecutive_overload + 1 if load > 16.0 else 0
                )
                if self._consecutive_overload >= 2:
                    self.load_valid = False
                self._next_load_sample = now + self.load_interval_seconds

    def stop(self) -> None:
        self.sample()
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()


def prewarm_and_verify_inputs(
    paths: Sequence[Path],
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    names = (
        "control_likes",
        "raw_events",
        "item_id_remap",
        "compact_embeddings",
    )
    if len(paths) != len(names):
        raise ValueError("materialization prewarm requires the four selector inputs")
    identities: dict[str, dict[str, Any]] = {}
    for name, path in zip(names, paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"materialization input must be a regular file: {path}")
        identity = _file_identity(path)
        if expected_sha256 is not None and identity["sha256"] != expected_sha256[name]:
            raise ValueError(f"materialization input SHA-256 differs: {name}")
        identities[name] = identity | {"path": str(path.resolve())}
    return identities


def all_input_pages_resident(paths: Sequence[Path]) -> bool:
    return all(_all_pages_resident(path) for path in paths)


def _all_pages_resident(path: Path) -> bool:
    size = path.stat().st_size
    if size == 0:
        return True
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = (size + page_size - 1) // page_size
    libc = ctypes.CDLL(None, use_errno=True)
    with path.open("rb") as handle, mmap.mmap(
        handle.fileno(), length=0, access=mmap.ACCESS_COPY
    ) as mapping:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        vector = (ctypes.c_ubyte * page_count)()
        result = libc.mincore(
            ctypes.c_void_p(address), ctypes.c_size_t(size), ctypes.byref(vector)
        )
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), path)
        return all(value & 1 for value in vector)


def wait_for_quiet_materialization_window(
    *, duration_seconds: float = 600.0, sample_seconds: float = 60.0
) -> tuple[float, ...]:
    if duration_seconds < 0 or sample_seconds <= 0:
        raise ValueError("quiet materialization window durations are invalid")
    required = math.ceil(duration_seconds / sample_seconds)
    accepted: list[float] = []
    samples: list[float] = []
    consecutive_overload = 0
    while len(accepted) < required:
        time.sleep(sample_seconds)
        load = float(os.getloadavg()[0])
        samples.append(load)
        if _foreign_materialization_or_training_active():
            raise RuntimeError("another training or G4 materialization job is active")
        if load > 16.0:
            accepted.clear()
            consecutive_overload += 1
            if consecutive_overload >= 2:
                raise RuntimeError("host load exceeded 16 for two consecutive samples")
        else:
            consecutive_overload = 0
            accepted.append(load)
    return tuple(samples)


def _foreign_materialization_or_training_active() -> bool:
    current = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == current:
            continue
        try:
            command = [
                token.decode()
                for token in (entry / "cmdline").read_bytes().split(b"\0")
                if token
            ]
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeError):
            continue
        if "python -m dcn.main" in " ".join(command) or (
            _is_native_materialization_command(command)
        ):
            return True
    return False


def _is_native_materialization_command(command: Sequence[str]) -> bool:
    launcher = any(
        Path(token).name == "run_selectors.py" or token.endswith(".run_selectors")
        for token in command
    )
    return launcher and "native-materialization" in command


@contextmanager
def _hold_native_materialization_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another G4 native materialization holds the lock"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _verify_reference_fixture() -> None:
    expected_host = "a100-1.vla.yp-c.yandex.net"
    if socket.gethostname() != expected_host:
        raise RuntimeError(f"materialization gate requires host {expected_host}")
    if os.sys.version_info[:3] != (3, 12, 13):
        raise RuntimeError("materialization gate requires Python 3.12.13")
    affinity = os.sched_getaffinity(0)
    if not set(range(16)) <= affinity:
        raise RuntimeError("materialization gate requires CPUs 0-15")
    memory_bytes = _host_memory_bytes()
    if memory_bytes != 929_980_153_856:
        raise RuntimeError("materialization gate host RAM differs")
    hardware = _reference_hardware()
    if hardware["cpu_models"] != ["AMD EPYC 7702 64-Core Processor"]:
        raise RuntimeError("materialization gate CPU model differs")
    if hardware["cpu_socket_ids"] != ["0"]:
        raise RuntimeError("materialization gate CPU socket topology differs")
    expected_environment = {
        "POLARS_MAX_THREADS": "16",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for name, value in expected_environment.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"materialization gate requires {name}={value}")


def _host_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot determine host RAM")


def _reference_hardware() -> dict[str, Any]:
    models: set[str] = set()
    sockets: set[str] = set()
    processor: dict[str, str] = {}
    for line in [*Path("/proc/cpuinfo").read_text().splitlines(), ""]:
        if not line:
            if processor:
                models.add(processor.get("model name", ""))
                sockets.add(processor.get("physical id", ""))
                processor = {}
            continue
        name, separator, value = line.partition(":")
        if separator:
            processor[name.strip()] = value.strip()
    hardware = {
        "host": socket.gethostname(),
        "cpu_models": sorted(models),
        "cpu_socket_ids": sorted(sockets),
        "host_memory_bytes": _host_memory_bytes(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    return hardware


def _promote_measured_artifacts(
    measured_root: Path,
    artifact_sha256s: Sequence[str],
    destination_root: Path,
) -> None:
    from experiments.g4_future_items.protocol.materialization import PeriodArtifact

    destination_root.mkdir(parents=True, exist_ok=True)
    for artifact_sha256 in artifact_sha256s:
        source = measured_root / artifact_sha256
        PeriodArtifact.open(source, expected_sha256=artifact_sha256)
        destination = destination_root / artifact_sha256
        if destination.exists():
            PeriodArtifact.open(destination, expected_sha256=artifact_sha256)
            continue
        temporary = destination_root / (
            f".{artifact_sha256}.promoting-{os.getpid()}-{time.time_ns()}"
        )
        shutil.copytree(source, temporary)
        PeriodArtifact.open(temporary, expected_sha256=artifact_sha256)
        os.replace(temporary, destination)


def _shared_search_input_sha256(
    results: Sequence[SelectorSearchResult],
) -> dict[str, str]:
    inputs = []
    for result in results:
        artifact = json.loads((result.artifact_path / "artifact.json").read_bytes())
        inputs.append(artifact["prepared_input_sha256"])
    if not inputs or any(value != inputs[0] for value in inputs[1:]):
        raise ValueError("selector search artifacts have different source inputs")
    return dict(inputs[0])


def _example_document(
    example: SelectorExample,
    bounds: ChronologicalBounds,
    next_item: int,
) -> dict[str, Any]:
    partition = bounds.partition_at(example.query.timestamp)
    assert partition is not None
    document = {
        "partition": partition.name,
        "query_uid": example.query.uid,
        "query_timestamp": example.query.timestamp,
        "query_item": example.query.item_id,
        "query_position": example.query.occurrence_ordinal,
        "query_next_item": next_item,
        "fold": fold_for_user(example.query.uid),
        "candidate_timestamp": example.candidate.timestamp,
        "candidate_item": example.candidate.item_id,
        "candidate_position": example.candidate.occurrence_ordinal,
        "period_start": example.period_start,
        "period_end": example.period_end,
        "base_eligible": example.eligible,
        "relevance_outcome": example.relevance_outcome,
        "candidate_like_count": example.candidate_like_count,
    }
    document.update(dict(zip(FEATURE_NAMES, example.feature_vector())))
    return document


def _load_partition(path: Path, partition: str | None) -> dict[str, np.ndarray]:
    frame = pl.read_parquet(path)
    if partition is not None:
        frame = frame.filter(pl.col("partition") == partition)
    frame = frame.sort(
        "query_uid",
        "query_timestamp",
        "query_item",
        "query_position",
        "candidate_timestamp",
        "candidate_item",
        "candidate_position",
    )
    return {
        "query_uid": frame["query_uid"].to_numpy(),
        "query_timestamp": frame["query_timestamp"].to_numpy(),
        "query_item": frame["query_item"].to_numpy(),
        "query_position": frame["query_position"].to_numpy(),
        "query_next_item": frame["query_next_item"].to_numpy(),
        "fold": frame["fold"].to_numpy(),
        "candidate_timestamp": frame["candidate_timestamp"].to_numpy(),
        "candidate_item": frame["candidate_item"].to_numpy(),
        "candidate_position": frame["candidate_position"].to_numpy(),
        "period_start": frame["period_start"].to_numpy(),
        "period_end": frame["period_end"].to_numpy(),
        "base_eligible": frame["base_eligible"].to_numpy(),
        "relevance_outcome": frame["relevance_outcome"].to_numpy(),
        "candidate_like_count": frame["candidate_like_count"].to_numpy(),
        "features": np.column_stack([frame[name].to_numpy() for name in FEATURE_NAMES]),
    }


def _eligibility(
    data: Mapping[str, np.ndarray], configuration: SelectorConfiguration
) -> np.ndarray:
    eligible = (
        data["base_eligible"].astype(bool)
        & (
            data["candidate_timestamp"]
            <= data["query_timestamp"] + configuration.lookahead_seconds
        )
        & (data["candidate_like_count"] >= configuration.minimum_liked_events)
    )
    if configuration.family == "time":
        distance_seconds = (1.0 - data["features"][:, 0]) * 84 * 3_600
        eligible &= (
            distance_seconds <= int(configuration.time_tolerance_seconds or 0) + 1e-9
        )
    return eligible


def _deterministic_scores(
    data: Mapping[str, np.ndarray], configuration: SelectorConfiguration
) -> np.ndarray:
    eligible = _eligibility(data, configuration)
    scores = np.zeros(len(eligible), dtype=np.float64)
    feature_index = {
        "content": 1,
        "item": 2,
        "artist": 3,
        "album": 4,
    }
    if configuration.family == "time":
        scores[eligible] = 1.0
    elif configuration.family == "content":
        scores[eligible] = data["features"][eligible, feature_index["content"]]
    elif configuration.family == "frequency":
        scores[eligible] = data["features"][
            eligible, feature_index[configuration.frequency_entity]
        ]
    else:
        raise ValueError("learned selector needs its fitted model")
    return scores


def _evaluate_columns(
    data: Mapping[str, np.ndarray], scores: np.ndarray, threshold: float
) -> tuple[SelectorMetrics, dict[tuple[int, int, int, int], float]]:
    labels = data["relevance_outcome"] > threshold
    queries: dict[tuple[int, int, int, int], float] = {}
    position = 0
    while position < len(scores):
        key = tuple(
            int(data[name][position])
            for name in ("query_uid", "query_timestamp", "query_item", "query_position")
        )
        end = position + 1
        while end < len(scores) and all(
            int(data[name][end]) == value
            for name, value in zip(
                ("query_uid", "query_timestamp", "query_item", "query_position"), key
            )
        ):
            end += 1
        order = sorted(
            range(position, end),
            key=lambda index: (
                -float(scores[index]),
                int(data["period_start"][index]),
                int(data["candidate_item"][index]),
                int(data["candidate_position"][index]),
            ),
        )[:10]
        dcg = sum(
            int(labels[index]) / math.log2(rank + 2) for rank, index in enumerate(order)
        )
        positives = min(int(labels[position:end].sum()), 10)
        ideal = sum(1 / math.log2(rank + 2) for rank in range(positives))
        queries[key] = dcg / ideal if ideal else 0.0
        position = end
    by_user: dict[int, list[float]] = {}
    for key, value in queries.items():
        by_user.setdefault(key[0], []).append(value)
    user_ndcg = [float(np.mean(values)) for values in by_user.values()]
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    metrics = SelectorMetrics(
        user_balanced_ndcg_at_10=float(np.mean(user_ndcg)) if user_ndcg else 0.0,
        auroc=_auroc(labels, scores),
        query_count=len(queries),
        user_count=len(by_user),
        pair_count=len(labels),
        positive_count=positive_count,
        negative_count=negative_count,
        positive_rate=positive_count / len(labels) if len(labels) else 0.0,
    )
    return metrics, queries


def _bootstrap_user_differences(values: Mapping[int, Sequence[float]]) -> BootstrapGate:
    differences = np.asarray(
        [np.mean(values[uid]) for uid in sorted(values)], dtype=np.float64
    )
    generator = np.random.Generator(np.random.PCG64(42))
    replicates = generator.choice(
        differences, size=(10_000, len(differences)), replace=True
    ).mean(axis=1)
    replicates.sort()
    lower = float(replicates[249])
    upper = float(replicates[math.ceil(0.975 * 10_000) - 1])
    return BootstrapGate(
        user_count=len(differences),
        mean_difference=float(differences.mean()),
        lower_95=lower,
        upper_95=upper,
        passes=lower > 0.0,
    )


def _scored_query_stream(
    queries: pl.DataFrame,
    data: Mapping[str, np.ndarray],
    scores: np.ndarray,
) -> Iterator[ScoredQuery]:
    row_position = 0
    for query_row in queries.iter_rows(named=True):
        key = (
            int(query_row["uid"]),
            int(query_row["prefix_timestamp"]),
            int(query_row["prefix_item_id"]),
            int(query_row["occurrence_position"]),
        )
        while row_position < len(scores) and _data_key(data, row_position) < key:
            row_position += 1
        begin = row_position
        while row_position < len(scores) and _data_key(data, row_position) == key:
            row_position += 1
        periods: list[ScoredPeriod] = []
        period_position = begin
        while period_position < row_position:
            period_key = (
                int(data["period_start"][period_position]),
                int(data["period_end"][period_position]),
            )
            period_end = period_position + 1
            while (
                period_end < row_position
                and (
                    int(data["period_start"][period_end]),
                    int(data["period_end"][period_end]),
                )
                == period_key
            ):
                period_end += 1
            score = float(np.max(scores[period_position:period_end]))
            if score > 0.0:
                positive_indices = [
                    index
                    for index in range(period_position, period_end)
                    if float(scores[index]) > 0.0
                ]
                periods.append(
                    ScoredPeriod(
                        start=period_key[0],
                        end=period_key[1],
                        score=score,
                        occurrences=tuple(
                            ScoredOccurrence(
                                int(data["candidate_timestamp"][index]),
                                int(data["candidate_item"][index]),
                                int(data["candidate_position"][index]),
                            )
                            for index in positive_indices
                        ),
                    )
                )
            period_position = period_end
        yield ScoredQuery(
            uid=key[0],
            prefix_timestamp=key[1],
            prefix_item_id=key[2],
            occurrence_position=key[3],
            next_item=int(query_row["next_item"]),
            fold=int(query_row["fold"]),
            periods=tuple(periods),
        )


def _data_key(data: Mapping[str, np.ndarray], index: int) -> tuple[int, int, int, int]:
    return tuple(
        int(data[name][index])
        for name in ("query_uid", "query_timestamp", "query_item", "query_position")
    )


def _assert_common_universe(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> None:
    for name in (
        "query_uid",
        "query_timestamp",
        "query_item",
        "query_position",
        "candidate_timestamp",
        "candidate_item",
        "candidate_position",
        "relevance_outcome",
    ):
        if not np.array_equal(left[name], right[name]):
            raise ValueError(f"selector common universe differs at {name}")


def _load_learned_model(result: SelectorSearchResult) -> LearnedSelector:
    content = (result.artifact_path / "model.pkl").read_bytes()
    artifact = json.loads((result.artifact_path / "artifact.json").read_bytes())
    if hashlib.sha256(content).hexdigest() != artifact["model_sha256"]:
        raise ValueError("learned selector model hash differs")
    model = pickle.loads(content)
    if not isinstance(model, LearnedSelector):
        raise ValueError("learned selector artifact has the wrong type")
    return model


def _result_by_sha(
    results: Sequence[SelectorSearchResult], sha256: str
) -> SelectorSearchResult:
    matches = [result for result in results if result.artifact_sha256 == sha256]
    if len(matches) != 1:
        raise ValueError(f"selector artifact {sha256!r} is not unique")
    return matches[0]


def _search_result_document(result: SelectorSearchResult) -> dict[str, Any]:
    return {
        "version": SEARCH_VERSION,
        "trial": result.trial.to_dict(),
        "validation_metrics": asdict(result.metrics),
        "relevance_threshold": result.relevance_threshold,
        "artifact_sha256": result.artifact_sha256,
        "artifact_payload_sha256": result.artifact_payload_sha256,
        "prepared_sha256": result.prepared_sha256,
        "prepared_semantics_sha256": result.prepared_semantics_sha256,
        "wall_seconds": result.wall_seconds,
    }


def _assert_prepared_semantics(
    prepared: PreparedSelectorData, results: Sequence[SelectorSearchResult]
) -> None:
    mismatches = [
        result.artifact_sha256
        for result in results
        if result.prepared_semantics_sha256 != prepared.semantics_sha256
    ]
    if mismatches:
        raise ValueError(
            "selector search artifacts were evaluated on different prepared semantics: "
            + ", ".join(mismatches)
        )


def _verify_prepared(path: Path, expected_sha256: str) -> dict[str, Any]:
    content = (path / "manifest.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("prepared selector manifest hash differs")
    manifest = json.loads(content)
    if manifest.get("version") != PREPARED_VERSION:
        raise ValueError("prepared selector version differs")
    identities = [manifest["queries"], *manifest["widths"].values()]
    for identity in identities:
        actual = _file_identity(path / identity["file"])
        if any(actual[key] != identity[key] for key in actual):
            raise ValueError(f"prepared selector file {identity['file']} differs")
    return manifest


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "file": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable selector artifact differs: {path}")
        return
    path.write_bytes(content)


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path)


def _logical_regular_file_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_symlink():
                raise ValueError(f"symlinks are forbidden in measured output: {child}")
            if child.is_file():
                total += child.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2
        position = end
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _process_tree_rss(root_pid: int) -> int:
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().split()
            parents[int(entry.name)] = int(stat[3])
            rss[int(entry.name)] = (
                int((entry / "statm").read_text().split()[1]) * page_size
            )
        except (
            FileNotFoundError,
            ProcessLookupError,
            PermissionError,
            IndexError,
            ValueError,
        ):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rss.get(pid, 0) for pid in descendants)


def runtime_evidence(
    monitor: SupervisedProcessTreeRssMonitor, wall_seconds: float
) -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "python": os.sys.version,
        "wall_seconds": wall_seconds,
        "peak_aggregate_rss_bytes": monitor.peak_bytes,
        "self_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "POLARS_MAX_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }


@dataclass
class SelectorQueueExperiment(Experiment):
    compiled_job: Mapping[str, Any] | None = None
    ledger_path: Path = Path()

    @classmethod
    def from_environment(cls) -> SelectorQueueExperiment:
        from experiments.g4_future_items.launchers.compiled import (
            compiled_job_from_environment,
        )

        compiled, ledger_path = compiled_job_from_environment()
        return cls(
            run_name=compiled.job.get(
                "run_name", f"g4_{compiled.row_id.replace(':', '_')}_native50m"
            ),
            seed=42,
            compiled_job=compiled.to_dict(),
            ledger_path=ledger_path,
        )

    def create_dataset_source(self):
        raise NotImplementedError

    def create_counters(self):
        raise NotImplementedError

    def _create_model(self):
        raise NotImplementedError

    def create_criterion(self):
        raise NotImplementedError

    def create_optimizers(self):
        raise NotImplementedError

    def run(self) -> None:
        if self.compiled_job is None:
            raise RuntimeError("selector queue experiment has no compiled job")
        job = self.compiled_job["job"]
        stage = job["stage"]
        if stage in {"selector_search", "selector_search_boundary"}:
            prepared = open_prepared_selector_data(
                PREPARED_ROOT, job["input_artifact_sha256"]
            )
            trial = selector_trial_from_job(job)
            self._wait_for_training_release()
            run_search_trial(
                prepared,
                trial,
                SEARCH_ROOT,
                output_artifact_sha256=job["output_artifact_sha256"],
            )
            return
        if stage == "selector_gate":
            results = [
                load_search_result(
                    SEARCH_ROOT,
                    job[f"{name}_artifact_sha256"],
                    expected_payload_sha256=job[f"{name}_payload_sha256"],
                )
                for name in ("deterministic", "learned")
            ]
            prepared = open_prepared_selector_data(
                PREPARED_ROOT, _shared_prepared_sha256(results)
            )
            self._wait_for_training_release()
            destination = GATE_ROOT / job["output_artifact_sha256"] / "gate.json"
            run_gate(
                prepared,
                results,
                destination,
                output_artifact_sha256=job["output_artifact_sha256"],
            )
            return
        if stage == "selector_materialization":
            raise RuntimeError(
                "selector materialization must run through the native cost gate"
            )
        raise ValueError(f"unsupported selector stage {stage!r}")


def _shared_prepared_sha256(results: Sequence[SelectorSearchResult]) -> str:
    identities = {result.prepared_sha256 for result in results}
    if len(identities) != 1:
        raise ValueError("selector search artifacts name different prepared artifacts")
    return identities.pop()


def _parse_reference_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in result:
            raise ValueError("semantics references must be unique NAME=PATH values")
        result[name] = Path(raw_path).resolve()
    return result


def prepare_selector_data_from_control_semantics(
    *, control_semantics_path: Path, raw_events_path: Path, workers: int
) -> PreparedSelectorData:
    from experiments.g4_future_items.protocol.manifest import (
        canonical_sha256,
        load_strict_json,
    )

    semantics = load_strict_json(control_semantics_path)
    if semantics.get("kind") != "g4_control_semantics":
        raise ValueError("selector preparation requires control semantics")
    data = semantics["data_identity"]
    paths = SelectorInputPaths(
        control_likes=Path(data["main"]["path"]),
        raw_events=raw_events_path.resolve(),
        item_id_remap=Path(data["remap"]["path"]),
        compact_embeddings=Path(data["content_embeddings"]["path"]),
    )
    paths.validate()
    cutoff = int(data["split_cutoff_timestamp"])
    start = (
        pl.scan_parquet(paths.control_likes)
        .filter(pl.col("timestamp") < cutoff)
        .select(pl.col("timestamp").min())
        .collect(engine="streaming")
        .item()
    )
    if start is None:
        raise ValueError("control likes have no pre-evaluation rows")
    return prepare_selector_data(
        paths,
        ChronologicalBounds.from_interval(int(start), cutoff),
        PREPARED_ROOT,
        provenance={
            "control_semantics_manifest_sha256": canonical_sha256(semantics),
        },
        workers=workers,
    )


def compile_treatment_semantics_freeze() -> dict[str, Any]:
    from experiments.g4_future_items.protocol.manifest import (
        build_treatment_semantics_manifest,
        canonical_sha256,
        expected_control_source_paths,
        load_strict_json,
        source_manifest,
    )

    control = expected_control_source_paths()
    selector_paths = {
        "experiments/g4_future_items/configs/selectors.py",
        "experiments/g4_future_items/launchers/__init__.py",
        "experiments/g4_future_items/launchers/compiled.py",
        "experiments/g4_future_items/launchers/run_selectors.py",
        "experiments/g4_future_items/protocol/materialization.py",
        "experiments/g4_future_items/selectors.py",
    }
    treatment_paths = {
        "experiments/g4_future_items/configs/treatments.py",
        "experiments/g4_future_items/launchers/__init__.py",
        "experiments/g4_future_items/launchers/compiled.py",
        "experiments/g4_future_items/launchers/run_treatments.py",
        "experiments/g4_future_items/protocol/materialization.py",
        "experiments/g4_future_items/selectors.py",
        "experiments/g4_future_items/targets.py",
    }
    entrypoints = {
        "experiments/g4_future_items/launchers/run_control.py": control,
        "experiments/g4_future_items/launchers/run_selectors.py": sorted(
            set(control) | selector_paths
        ),
        "experiments/g4_future_items/launchers/run_treatments.py": sorted(
            set(control) | treatment_paths
        ),
    }
    source_paths = sorted({path for paths in entrypoints.values() for path in paths})
    selected = load_strict_json(SELECTED_CONTROL_MANIFEST_PATH)
    return build_treatment_semantics_manifest(
        selected_control_manifest_sha256=canonical_sha256(selected),
        entrypoint_source_paths=entrypoints,
        post_review_sources=source_manifest(PROJECT_ROOT, source_paths),
        schema_revisions=TREATMENT_SCHEMA_REVISIONS,
        fixture_paths=TREATMENT_FIXTURE_PATHS,
    )


def freeze_treatment_semantics(*, write: bool = False) -> dict[str, Any]:
    from experiments.g4_future_items.protocol.manifest import (
        canonical_bytes,
        canonical_sha256,
        write_frozen_manifest,
    )

    document = compile_treatment_semantics_freeze()
    destination = "absent"
    if TREATMENT_SEMANTICS_MANIFEST_PATH.exists():
        if TREATMENT_SEMANTICS_MANIFEST_PATH.read_bytes() != canonical_bytes(document):
            raise RuntimeError(
                "frozen treatment semantics manifest differs: "
                f"{TREATMENT_SEMANTICS_MANIFEST_PATH}"
            )
        destination = "matching"
    if write:
        write_frozen_manifest(TREATMENT_SEMANTICS_MANIFEST_PATH, document)
    return {
        "write": write,
        "treatment_semantics_manifest_sha256": canonical_sha256(document),
        "path": str(TREATMENT_SEMANTICS_MANIFEST_PATH.resolve()),
        "destination": destination,
        "source_count": len(document["source_paths"]),
        "changed_source_count": len(document["changed_paths"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("compile")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--control-semantics", required=True, type=Path)
    prepare.add_argument("--raw-events", required=True, type=Path)
    prepare.add_argument("--workers", default=16, type=int)
    native = subparsers.add_parser("native-materialization")
    native.add_argument("--ledger", required=True, type=Path)
    native.add_argument("--semantics", action="append", default=[])
    native.add_argument("--measurement-directory", required=True, type=Path)
    native.add_argument("--attempt", default=1, type=int)
    native.add_argument("--previous-attempt", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "compile":
        print(
            json.dumps(
                [trial.to_dict() for trial in compile_selector_search()], indent=2
            )
        )
        return
    if arguments.command == "prepare":
        prepared = prepare_selector_data_from_control_semantics(
            control_semantics_path=arguments.control_semantics.resolve(),
            raw_events_path=arguments.raw_events.resolve(),
            workers=arguments.workers,
        )
        print(prepared.sha256)
        return
    evidence = run_native_materialization_from_ledger(
        ledger_path=arguments.ledger.resolve(),
        semantics_paths=_parse_reference_paths(arguments.semantics),
        measurement_directory=arguments.measurement_directory.resolve(),
        attempt=arguments.attempt,
        previous_attempt_path=(
            None
            if arguments.previous_attempt is None
            else arguments.previous_attempt.resolve()
        ),
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


experiment = (
    SelectorQueueExperiment.from_environment()
    if os.environ.get("G4_COMPILED_JOB_B64") is not None
    and multiprocessing.current_process().name == "MainProcess"
    else None
)
