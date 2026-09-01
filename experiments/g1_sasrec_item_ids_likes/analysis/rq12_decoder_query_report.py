from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import cache
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

from experiments.g1_sasrec_item_ids_likes.analysis import (
    reporting,
    rq8_reinvestigation_report,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq8_reinvestigation_candidates import (
    Rq8Candidate,
    query_initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


_CONFIG = Path(__file__).parents[1] / "configs/rq8_reinvestigation_variant.py"
_METHODS = ("standard", "end_only", "interleaved")
_METRICS = (
    "recall@100",
    "ndcg@100",
    "recall@10",
    "ndcg@10",
    "coverage@100",
)
_TIMING_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_EPOCH_PATTERN = re.compile(
    rf"\bepoch (\d+) finished\b.*?"
    rf"timing\.train_epoch_time=({_TIMING_NUMBER}).*?"
    rf"timing\.val_inference_time=({_TIMING_NUMBER}).*?"
    rf"timing\.val_save_time=({_TIMING_NUMBER})"
)
_TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")
_DATASET_PATH_PATTERN = re.compile(r"\bPreparing yambda .* in (?P<path>\S+)$")
_SEQUENCE_CACHE_PATTERN = re.compile(
    r"\bLoaded cached user sequences from "
    r"(?P<path>\S+/sequences/(?P<split>train|val|true_metric_query)_[^\s/]+)$"
)
_DATASET_CONTENT_FILES = (
    "events.parquet",
    "events_remapped.parquet",
    "item_id_remap.parquet",
)
_DATA_FIELDS = (
    "dataset_size",
    "user_sample",
    "event_type_filter",
    "min_item_interactions_per_item",
    "drop_unmapped_items",
    "validation_interval_seconds",
    "day_range",
    "max_seq_len",
    "window",
)
_EVALUATOR_FIELDS = (
    "eval_ks",
    "eval_max_users",
    "eval_every_n_epochs",
    "early_stopping_metric",
    "early_stopping_metric_prefix",
    "selection_k",
    "evaluation_catalog",
    "exclude_seen_from_evaluation",
    "restore_best_weights",
)
_BASE_ARCHITECTURE_FIELDS = (
    "experiment_class",
    "mup_base_dim",
    "mup_delta_dim",
    "mup_base_ffn_dim",
    "mup_delta_ffn_dim",
    "model_dim",
    "item_embedding_dim",
    "bos",
    "timestamp_delta",
    "timestamp_combination",
    "timestamp_num_bins",
    "timestamp_bin_semantics_revision",
    "per_layer_item_embeddings",
)
_OBJECTIVE_FIELDS = (
    "negative_sampling",
    "num_in_batch_negatives",
    "logq_correction",
    "random_negative_fraction",
    "logq_alpha",
    "correct_positive_logq",
    "mask_false_negatives",
    "exclude_own_group_negatives",
    "dense_random_negative_scores",
)


class Rq12ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Rq12ReportBundle:
    reader_markdown: str
    evidence: dict[str, object]


@dataclass(frozen=True)
class EpochTiming:
    epoch: int
    train_seconds: float
    validation_seconds: float
    save_seconds: float


RecipeVerifier = Callable[[Path, Rq8Candidate], bool]


def collect_report_bundle(
    logs: Path,
    *,
    verify_recipe: RecipeVerifier | None = None,
) -> Rq12ReportBundle:
    verifier = _verify_recipe if verify_recipe is None else verify_recipe
    initial = query_initial_candidates()
    missing = [
        candidate.run_name
        for candidate in initial
        if not rq8_reinvestigation_report._artifact_complete(logs, candidate)
    ]
    if missing:
        raise Rq12ReportError(
            f"RQ12 query surface is incomplete: {len(initial) - len(missing)}/"
            f"{len(initial)} initial artifacts"
        )
    try:
        initial_runs = [
            rq8_reinvestigation_report._load_run(logs, candidate)
            for candidate in initial
        ]
        grouped = rq8_reinvestigation_report._group_initial_runs(initial_runs)
        surfaces = {
            method: rq8_reinvestigation_report._resolve_surface(
                logs, f"query {method}", grouped[("query", method)]
            )
            for method in _METHODS
        }
        selected = {
            method: rq8_reinvestigation_report._load_query_repeats(
                logs, surfaces[method].selected
            )
            for method in _METHODS
        }
    except rq8_reinvestigation_report.Rq8ReportError as error:
        raise Rq12ReportError(str(error)) from error

    all_required_runs = {
        method: tuple(
            {
                run.candidate.run_name: run
                for run in (*surfaces[method].runs, *selected[method][1:])
            }.values()
        )
        for method in _METHODS
    }
    for run in (
        run for method_runs in all_required_runs.values() for run in method_runs
    ):
        directory = logs / run.candidate.run_name
        if not verifier(directory, run.candidate):
            raise Rq12ReportError(
                f"{run.candidate.run_name}: protocol-incompatible artifact"
            )

    all_required_records = {
        method: tuple(
            _artifact_record(logs / run.candidate.run_name, run)
            for run in all_required_runs[method]
        )
        for method in _METHODS
    }
    records_by_name = {
        str(record["run_name"]): record
        for method_records in all_required_records.values()
        for record in method_records
    }
    artifact_records = {
        method: tuple(
            records_by_name[run.candidate.run_name] for run in selected[method]
        )
        for method in _METHODS
    }
    compatibility = _compatibility(all_required_records)
    anomalies = _timing_anomalies(all_required_records)
    total_required_cost = _total_required_cost(all_required_records, anomalies)
    evidence = _evidence(
        surfaces,
        selected,
        artifact_records,
        all_required_records,
        compatibility,
        anomalies,
        total_required_cost,
    )
    return Rq12ReportBundle(
        reader_markdown=_reader_markdown(
            selected, artifact_records, total_required_cost
        ),
        evidence=evidence,
    )


def write_report_bundle(
    bundle: Rq12ReportBundle,
    scratchpad: Path,
    evidence: Path,
) -> dict[str, Path]:
    paths = {
        "reader": scratchpad / "rq12_decoder_query_reader_500m.md",
        "evidence": evidence / "rq12_decoder_query_results.json",
    }
    _write_atomically(paths["reader"], bundle.reader_markdown)
    _write_atomically(
        paths["evidence"],
        json.dumps(bundle.evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return paths


def _verify_recipe(directory: Path, candidate: Rq8Candidate) -> bool:
    return verify_artifact.verify_config(
        directory,
        _CONFIG,
        [f"G1_RQ8_RUN={candidate.run_name}"],
    )


def _artifact_record(
    directory: Path,
    run: rq8_reinvestigation_report.Run,
) -> dict[str, object]:
    metadata = _load_json(directory / "training_metadata.json")
    final_metrics = _load_json(directory / "final_metrics.json")
    invariants = metadata["transfer_invariants"]
    if not isinstance(invariants, dict):
        raise Rq12ReportError(f"{directory.name}: missing transfer invariants")
    efficiency_inputs, efficiency_outputs = _efficiency(
        directory, metadata, run.candidate.query_method
    )
    dataset_identity = _dataset_identity(
        directory,
        datetime.fromisoformat(str(efficiency_inputs["prepared_stage_timestamp"])),
    )
    fingerprints = _fingerprints(metadata, final_metrics, dataset_identity)
    return {
        "run_name": run.candidate.run_name,
        "seed": run.candidate.seed,
        "deep_learning_rate": run.candidate.deep_lr,
        "config_recipe_verified": True,
        "sha256": {
            name: _file_sha256(directory / name)
            for name in ("training_metadata.json", "final_metrics.json", "sweep.log")
        },
        "dataset_identity": dataset_identity,
        "fingerprints": fingerprints,
        "validation_metrics": dict(run.validation_metrics),
        "full_user_metrics": {
            metric: float(final_metrics[metric]) for metric in _METRICS
        },
        "full_user_count": int(final_metrics["num_users"]),
        "best_epoch": run.best_epoch,
        "stopped_epoch": run.stopped_epoch,
        "restored_best_checkpoint": bool(invariants["restore_best_weights"]),
        "efficiency_inputs": efficiency_inputs,
        "efficiency_outputs": efficiency_outputs,
    }


def _efficiency(
    directory: Path,
    metadata: dict[str, Any],
    method: str,
) -> tuple[dict[str, object], dict[str, float]]:
    targets = _positive_int(metadata, "targets_per_epoch", directory.name)
    tokens = _positive_int(metadata, "tokens_per_epoch", directory.name)
    best_epoch = _positive_int(metadata, "best_epoch", directory.name)
    stopped_epoch = _positive_int(metadata, "stopped_epoch", directory.name)
    if method == "standard":
        examples = tokens - targets
        derivation = "input_tokens - next_item_targets"
    elif method == "end_only":
        difference = tokens - targets
        if difference % 2:
            raise Rq12ReportError(f"{directory.name}: non-integral example count")
        examples = difference // 2
        derivation = "(input_tokens - next_item_targets) / 2"
    elif method == "interleaved":
        if tokens % 2:
            raise Rq12ReportError(f"{directory.name}: non-integral event count")
        examples = tokens // 2 - targets
        derivation = "input_tokens / 2 - next_item_targets"
    else:
        raise Rq12ReportError(f"unknown query method {method!r}")
    if examples <= 0:
        raise Rq12ReportError(f"{directory.name}: invalid example count")

    timings, prepared, final = _load_timings(directory, stopped_epoch)
    steady = timings[1:]
    steady_train_seconds = sum(item.train_seconds for item in steady)
    if not steady or steady_train_seconds <= 0:
        raise Rq12ReportError(f"{directory.name}: no steady-state timing evidence")
    through_best = timings[:best_epoch]
    return (
        {
            "examples_per_epoch": examples,
            "examples_derivation": derivation,
            "next_item_targets_per_epoch": targets,
            "auxiliary_ntp_targets_per_epoch": 0,
            "input_tokens_per_epoch": tokens,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "prepared_stage_timestamp": prepared.isoformat(),
            "final_metrics_timestamp": final.isoformat(),
            "epoch_timings": [
                {
                    "epoch": item.epoch,
                    "train_seconds": item.train_seconds,
                    "validation_seconds": item.validation_seconds,
                    "save_seconds": item.save_seconds,
                }
                for item in timings
            ],
        },
        {
            "steady_state_targets_per_second": (
                targets * len(steady) / steady_train_seconds
            ),
            "time_through_selected_checkpoint_seconds": sum(
                item.train_seconds + item.validation_seconds + item.save_seconds
                for item in through_best
            ),
            "required_horizon_train_validation_seconds": sum(
                item.train_seconds + item.validation_seconds + item.save_seconds
                for item in timings
            ),
            "observed_end_to_end_wall_seconds": (final - prepared).total_seconds(),
        },
    )


def _load_timings(
    directory: Path,
    stopped_epoch: int,
) -> tuple[tuple[EpochTiming, ...], datetime, datetime]:
    try:
        lines = (directory / "sweep.log").read_text().splitlines()
    except OSError as error:
        raise Rq12ReportError(f"{directory.name}: cannot read sweep.log") from error
    timings: dict[int, EpochTiming] = {}
    prepared: list[datetime] = []
    final: list[datetime] = []
    for line in lines:
        if "Prepared stage '" in line:
            prepared.append(_timestamp(line, directory.name))
        if "Final metrics (" in line:
            final.append(_timestamp(line, directory.name))
        match = _EPOCH_PATTERN.search(line)
        if match is None:
            continue
        epoch = int(match.group(1))
        values = tuple(float(match.group(index)) for index in range(2, 5))
        if epoch in timings or any(
            not math.isfinite(value) or value < 0 for value in values
        ):
            raise Rq12ReportError(f"{directory.name}: malformed epoch timing evidence")
        timings[epoch] = EpochTiming(epoch, *values)
    if sorted(timings) != list(range(stopped_epoch)):
        raise Rq12ReportError(f"{directory.name}: incomplete epoch timing evidence")
    if len(prepared) != 1 or len(final) != 1 or final[0] < prepared[0]:
        raise Rq12ReportError(f"{directory.name}: malformed wall-time evidence")
    return (
        tuple(timings[index] for index in range(stopped_epoch)),
        prepared[0],
        final[0],
    )


def _timestamp(line: str, context: str) -> datetime:
    match = _TIMESTAMP_PATTERN.search(line)
    if match is None:
        raise Rq12ReportError(f"{context}: missing timestamp")
    return datetime.fromisoformat(match.group(1).replace(",", "."))


def _dataset_identity(directory: Path, run_start: datetime) -> dict[str, object]:
    try:
        lines = (directory / "sweep.log").read_text().splitlines()
    except OSError as error:
        raise Rq12ReportError(
            f"{directory.name}: cannot read dataset provenance"
        ) from error
    dataset_paths = {
        Path(match.group("path"))
        for line in lines
        if (match := _DATASET_PATH_PATTERN.search(line)) is not None
    }
    cache_paths: dict[str, set[Path]] = {
        "train": set(),
        "val": set(),
        "true_metric_query": set(),
    }
    for line in lines:
        match = _SEQUENCE_CACHE_PATTERN.search(line)
        if match is not None:
            cache_paths[match.group("split")].add(Path(match.group("path")))
    if len(dataset_paths) != 1 or any(
        len(paths) != 1 for paths in cache_paths.values()
    ):
        raise Rq12ReportError(
            f"{directory.name}: incomplete or ambiguous dataset/cache provenance"
        )
    dataset_path = dataset_paths.pop()
    caches = {split: paths.pop() for split, paths in cache_paths.items()}
    dataset_content = _content_sha256_manifest(
        dataset_path,
        tuple(dataset_path / name for name in _DATASET_CONTENT_FILES),
        run_start,
        directory.name,
    )
    cache_content = {}
    for path in caches.values():
        metadata = path / "metadata.json"
        buckets = tuple(sorted((path / "buckets").glob("*.parquet")))
        if not buckets:
            raise Rq12ReportError(
                f"{directory.name}: {path.name} has no bucket content"
            )
        cache_content[path.name] = _content_sha256_manifest(
            path,
            (metadata, *buckets),
            run_start,
            directory.name,
        )
    cache_parents = {path.parent.parent.name for path in caches.values()}
    if len(cache_parents) != 1:
        raise Rq12ReportError(f"{directory.name}: sequence cache parents disagree")
    identity = {
        "dataset_directory": dataset_path.name,
        "dataset_content_sha256": dataset_content,
        "sequence_cache_parent": cache_parents.pop(),
        "sequence_caches": {split: path.name for split, path in caches.items()},
        "sequence_cache_content_sha256": cache_content,
        "content_mtime_verified_before_run_start": True,
    }
    return {**identity, "manifest_sha256": _canonical_sha256(identity)}


def _content_sha256_manifest(
    root: Path,
    paths: tuple[Path, ...],
    run_start: datetime,
    context: str,
) -> dict[str, str]:
    manifest = {}
    for path in paths:
        try:
            modified = path.stat().st_mtime
        except OSError as error:
            raise Rq12ReportError(
                f"{context}: missing provenance file {path}"
            ) from error
        if modified > run_start.timestamp():
            raise Rq12ReportError(
                f"{context}: provenance file postdates run start: {path}"
            )
        manifest[path.relative_to(root).as_posix()] = _file_sha256(path)
    return manifest


def _fingerprints(
    metadata: dict[str, Any],
    final_metrics: dict[str, Any],
    dataset_identity: dict[str, object],
) -> dict[str, str]:
    invariants = metadata["transfer_invariants"]
    transformer = dict(invariants["transformer"])
    attention_window = transformer.pop("attention_window")
    payloads = {
        "data": {
            **{key: invariants[key] for key in _DATA_FIELDS},
            "dataset_identity": dataset_identity,
        },
        "evaluator": {
            **{key: invariants[key] for key in _EVALUATOR_FIELDS},
            "full_user_count": final_metrics["num_users"],
        },
        "base_architecture": {
            **{key: invariants[key] for key in _BASE_ARCHITECTURE_FIELDS},
            "transformer": transformer,
        },
        "common_objective": {
            **{key: invariants[key] for key in _OBJECTIVE_FIELDS},
            "training_semantics_revision": metadata["training_semantics_revision"],
        },
        "query_layout": {
            "cls_token_mode": invariants["cls_token_mode"],
            "attention_window": attention_window,
        },
        "full_transfer_invariants": invariants,
    }
    return {name: _canonical_sha256(payload) for name, payload in payloads.items()}


def _compatibility(
    records: dict[str, tuple[dict[str, object], ...]],
) -> dict[str, object]:
    data = _fingerprint_values(records, "data")
    evaluator = _fingerprint_values(records, "evaluator")
    architecture = _fingerprint_values(records, "base_architecture")
    common_objective = _fingerprint_values(records, "common_objective")
    if (
        len(data) != 1
        or len(evaluator) != 1
        or len(architecture) != 1
        or len(common_objective) != 1
    ):
        raise Rq12ReportError("RQ12 selected artifacts do not share one frozen surface")
    query_layouts = {
        method: sorted(
            {str(record["fingerprints"]["query_layout"]) for record in method_records}
        )
        for method, method_records in records.items()
    }
    if any(len(values) != 1 for values in query_layouts.values()):
        raise Rq12ReportError("RQ12 query-layout fingerprint varies within a method")
    workloads = {
        (
            int(record["efficiency_inputs"]["examples_per_epoch"]),
            int(record["efficiency_inputs"]["next_item_targets_per_epoch"]),
            int(record["efficiency_inputs"]["auxiliary_ntp_targets_per_epoch"]),
        )
        for method_records in records.values()
        for record in method_records
    }
    if len(workloads) != 1:
        raise Rq12ReportError("RQ12 artifacts do not share one workload")
    examples, next_item_targets, auxiliary_targets = workloads.pop()
    return {
        "config_recipe_verified": True,
        "canonicalization": "sha256(sorted compact JSON)",
        "data_fields": list(_DATA_FIELDS),
        "evaluator_fields": [*_EVALUATOR_FIELDS, "full_user_count"],
        "base_architecture_fields": [
            *_BASE_ARCHITECTURE_FIELDS,
            "transformer excluding attention_window",
        ],
        "common_objective_fields": [
            *_OBJECTIVE_FIELDS,
            "training_semantics_revision",
        ],
        "query_layout_fields": ["cls_token_mode", "attention_window"],
        "data_fingerprints": sorted(data),
        "evaluator_fingerprints": sorted(evaluator),
        "base_architecture_fingerprints": sorted(architecture),
        "common_objective_fingerprints": sorted(common_objective),
        "query_layout_fingerprints": {
            method: values[0] for method, values in query_layouts.items()
        },
        "workload": {
            "examples_per_epoch": examples,
            "next_item_targets_per_epoch": next_item_targets,
            "auxiliary_ntp_targets_per_epoch": auxiliary_targets,
        },
    }


def _fingerprint_values(
    records: dict[str, tuple[dict[str, object], ...]],
    name: str,
) -> set[str]:
    return {
        str(record["fingerprints"][name])
        for method_records in records.values()
        for record in method_records
    }


def _timing_anomalies(
    records: dict[str, tuple[dict[str, object], ...]],
) -> list[dict[str, object]]:
    anomalies = []
    for method, method_records in records.items():
        walls = sorted(
            float(record["efficiency_outputs"]["observed_end_to_end_wall_seconds"])
            for record in method_records
        )
        median = walls[len(walls) // 2]
        for record in method_records:
            value = float(
                record["efficiency_outputs"]["observed_end_to_end_wall_seconds"]
            )
            if median > 0 and value > 1.5 * median:
                anomalies.append(
                    {
                        "method": method,
                        "run_name": record["run_name"],
                        "metric": "observed_end_to_end_wall_seconds",
                        "value": value,
                        "method_median": median,
                        "rule": "value > 1.5 * method median",
                        "interpretation": (
                            "Observed wall time includes overhead outside logged "
                            "train/validation components and is not clean model-cost evidence."
                        ),
                    }
                )
    return anomalies


def _total_required_cost(
    records: dict[str, tuple[dict[str, object], ...]],
    anomalies: list[dict[str, object]],
) -> dict[str, object]:
    names = [
        str(record["run_name"])
        for method_records in records.values()
        for record in method_records
    ]
    if len(names) != len(set(names)):
        raise Rq12ReportError("RQ12 required-cost artifacts are not unique")
    anomalous_names = {str(item["run_name"]) for item in anomalies}

    def summarize(method_records: tuple[dict[str, object], ...]) -> dict[str, object]:
        return {
            "unique_artifact_count": len(method_records),
            "logged_train_validation_seconds": sum(
                float(
                    record["efficiency_outputs"][
                        "required_horizon_train_validation_seconds"
                    ]
                )
                for record in method_records
            ),
            "observed_end_to_end_wall_seconds": sum(
                float(record["efficiency_outputs"]["observed_end_to_end_wall_seconds"])
                for record in method_records
            ),
            "wall_anomaly_artifacts": [
                str(record["run_name"])
                for record in method_records
                if str(record["run_name"]) in anomalous_names
            ],
        }

    by_method = {
        method: summarize(method_records) for method, method_records in records.items()
    }
    return {
        "definition": (
            "All unique LR-tuning, deterministic LR-boundary, and selected-seed "
            "confirmation artifacts required to resolve RQ12. Logged cost sums "
            "train, validation-inference, and save components over each complete "
            "20-epoch horizon; observed wall spans Prepared stage through Final metrics."
        ),
        "unique_artifact_count": len(names),
        "logged_train_validation_seconds": sum(
            float(item["logged_train_validation_seconds"])
            for item in by_method.values()
        ),
        "observed_end_to_end_wall_seconds": sum(
            float(item["observed_end_to_end_wall_seconds"])
            for item in by_method.values()
        ),
        "wall_anomaly_artifacts": sorted(anomalous_names),
        "by_method": by_method,
    }


def _evidence(
    surfaces: dict[str, rq8_reinvestigation_report.SurfaceResolution],
    selected: dict[str, tuple[rq8_reinvestigation_report.Run, ...]],
    records: dict[str, tuple[dict[str, object], ...]],
    all_required_records: dict[str, tuple[dict[str, object], ...]],
    compatibility: dict[str, object],
    anomalies: list[dict[str, object]],
    total_required_cost: dict[str, object],
) -> dict[str, object]:
    return {
        "research_question": "RQ12 decoder-only query layout",
        "dataset_size": "500m",
        "source_study": "RQ8 query-token reinvestigation",
        "tuning_ledger": (
            "experiments/g1_sasrec_item_ids_likes/scratchpad/"
            "rq8_reinvestigation_tuning_500m.md"
        ),
        "timing_definition": {
            "steady_state_targets_per_second": (
                "next-item targets divided by logged train time over epochs 2-20; "
                "the first epoch is excluded"
            ),
            "time_through_selected_checkpoint_seconds": (
                "sum of logged train, validation inference, and validation save "
                "time through the validation-selected best epoch"
            ),
            "required_horizon_train_validation_seconds": (
                "sum of logged train, validation inference, and validation save "
                "time over the complete 20-epoch annealing horizon"
            ),
            "observed_end_to_end_wall_seconds": (
                "timestamp difference from Prepared stage to Final metrics; includes "
                "unattributed overhead and full-user evaluation"
            ),
        },
        "compatibility": compatibility,
        "timing_anomalies": anomalies,
        "total_required_cost": total_required_cost,
        "methods": [
            {
                "method": method,
                "selected_deep_learning_rate": (
                    surfaces[method].selected.candidate.deep_lr
                ),
                "tuning_artifacts": [
                    run.candidate.run_name for run in surfaces[method].runs
                ],
                "mean_full_user_metrics": _mean_quality(selected[method]),
                "mean_efficiency": _mean_efficiency(records[method]),
                "artifacts": list(records[method]),
                "all_required_artifacts": list(all_required_records[method]),
            }
            for method in _METHODS
        ],
    }


def _reader_markdown(
    selected: dict[str, tuple[rq8_reinvestigation_report.Run, ...]],
    records: dict[str, tuple[dict[str, object], ...]],
    total_required_cost: dict[str, object],
) -> str:
    quality = {method: _mean_quality(selected[method]) for method in _METHODS}
    efficiency = {method: _mean_efficiency(records[method]) for method in _METHODS}
    control = quality["standard"]
    labels = {
        "standard": "standard item-state",
        "end_only": "**end-only CLS**",
        "interleaved": "interleaved CLS",
    }
    lines = [
        "## RQ12 — Which decoder-only query-token layout works best?",
        "",
        "### Candidate-generation quality",
        "",
        "| query objective | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in _METHODS:
        cells = [labels[method]]
        for metric in _METRICS:
            cells.append(
                reporting.absolute(quality[method][metric])
                if method == "standard"
                else reporting.change_cell(
                    quality[method][metric], control[metric], metric
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Training efficiency",
        "",
        "| query objective | examples/epoch | next-item targets/epoch | auxiliary NTP targets/epoch | input tokens/epoch | best epochs (seeds 42 / 43 / 44) | mean steady-state targets/s (epochs 2–20 train only) | mean time through selected checkpoint (train+validation), s | mean full-horizon logged train+validation, s | all required artifacts logged train+validation, s | all required artifacts observed wall (Prepared stage → Final metrics), s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in _METHODS:
        values = efficiency[method]
        required = total_required_cost["by_method"][method]
        lines.append(
            "| "
            + " | ".join(
                [
                    labels[method],
                    str(values["examples_per_epoch"]),
                    str(values["next_item_targets_per_epoch"]),
                    str(values["auxiliary_ntp_targets_per_epoch"]),
                    str(values["input_tokens_per_epoch"]),
                    " / ".join(
                        str(epoch) for epoch in values["best_epochs_by_seed"].values()
                    ),
                    f"{values['steady_state_targets_per_second']:.3f}",
                    f"{values['time_through_selected_checkpoint_seconds']:.3f}",
                    f"{values['required_horizon_train_validation_seconds']:.3f}",
                    f"{required['logged_train_validation_seconds']:.3f}",
                    f"{required['observed_end_to_end_wall_seconds']:.3f}",
                ]
            )
            + " |"
        )
    lines.append(
        "| **all query objectives** | — | — | — | — | — | — | — | — | "
        f"{total_required_cost['logged_train_validation_seconds']:.3f} | "
        f"{total_required_cost['observed_end_to_end_wall_seconds']:.3f} |"
    )
    return "\n".join(lines) + "\n"


def _mean_quality(
    runs: tuple[rq8_reinvestigation_report.Run, ...],
) -> dict[str, float]:
    return {
        metric: sum(run.metrics[metric] for run in runs) / len(runs)
        for metric in _METRICS
    }


def _mean_efficiency(records: tuple[dict[str, object], ...]) -> dict[str, Any]:
    input_keys = (
        "examples_per_epoch",
        "next_item_targets_per_epoch",
        "auxiliary_ntp_targets_per_epoch",
        "input_tokens_per_epoch",
    )
    fixed = {}
    for key in input_keys:
        values = {int(record["efficiency_inputs"][key]) for record in records}
        if len(values) != 1:
            raise Rq12ReportError(f"RQ12 selected artifacts disagree on {key}")
        fixed[key] = values.pop()
    output_keys = (
        "steady_state_targets_per_second",
        "time_through_selected_checkpoint_seconds",
        "required_horizon_train_validation_seconds",
        "observed_end_to_end_wall_seconds",
    )
    return {
        **fixed,
        "best_epochs_by_seed": {
            str(record["seed"]): int(record["efficiency_inputs"]["best_epoch"])
            for record in sorted(records, key=lambda item: int(item["seed"]))
        },
        **{
            key: sum(float(record["efficiency_outputs"][key]) for record in records)
            / len(records)
            for key in output_keys
        },
    }


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@cache
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq12ReportError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq12ReportError(f"{path} must contain an object")
    return value


def _positive_int(mapping: dict[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Rq12ReportError(f"{context}: {key} must be a positive integer")
    return value


def _write_atomically(path: Path, content: str) -> None:
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
    arguments = parser.parse_args()
    bundle = collect_report_bundle(arguments.logs)
    for path in write_report_bundle(
        bundle, arguments.scratchpad, arguments.evidence
    ).values():
        print(path)


if __name__ == "__main__":
    main()
