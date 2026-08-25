from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION
from experiments.g1_sasrec_item_ids_likes.analysis import reporting
from experiments.g1_sasrec_item_ids_likes.analysis.rq8_reinvestigation_candidates import (
    Rq8Candidate,
    initial_candidates,
    make_boundary_candidate,
    make_confirmation_candidate,
)


_QUERY_METHODS = ("standard", "end_only", "interleaved")
_POSITION_METHODS = ("alibi", "rope_reverse_alibi")
_SEQUENCE_LENGTHS = (12, 25, 50, 100, 128, 200, 256, 512)
_CONFIRMATION_SEEDS = (42, 43, 44)
_METRICS = (
    "recall@100",
    "ndcg@100",
    "recall@10",
    "ndcg@10",
    "coverage@100",
)
_METRIC_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


class Rq8ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Run:
    candidate: Rq8Candidate
    best_epoch: int
    stopped_epoch: int
    validation_metrics: dict[str, float]
    metrics: dict[str, float]


@dataclass(frozen=True)
class Rq8ReportBundle:
    query_runs: dict[str, tuple[Run, ...]]
    sequence_runs: dict[str, dict[int, Run]]
    tuning_markdown: str
    reader_markdown: str
    evidence: dict[str, object]


@dataclass(frozen=True)
class SurfaceResolution:
    selected: Run
    runs: tuple[Run, ...]


def collect_report_bundle(logs: Path) -> Rq8ReportBundle:
    expected = initial_candidates()
    complete = [candidate for candidate in expected if _artifact_complete(logs, candidate)]
    if len(complete) != len(expected):
        missing = [candidate.run_name for candidate in expected if candidate not in complete]
        raise Rq8ReportError(
            f"RQ8 corrected surface is incomplete: {len(complete)}/{len(expected)}; "
            f"missing {', '.join(missing)}"
        )
    initial_runs = [_load_run(logs, candidate) for candidate in expected]
    grouped = _group_initial_runs(initial_runs)

    query_surfaces = {
        method: _resolve_surface(logs, f"query {method}", grouped[("query", method)])
        for method in _QUERY_METHODS
    }
    sequence_surfaces = {
        (position, length): _resolve_surface(
            logs,
            f"sequence {position} length {length}",
            grouped[("sequence", position, length)],
        )
        for position in _POSITION_METHODS
        for length in _SEQUENCE_LENGTHS
    }
    query_runs = {
        method: _load_query_repeats(logs, query_surfaces[method].selected)
        for method in _QUERY_METHODS
    }
    sequence_runs = {
        position: {
            length: sequence_surfaces[(position, length)].selected
            for length in _SEQUENCE_LENGTHS
        }
        for position in _POSITION_METHODS
    }
    tuning_runs = _all_tuning_runs(query_surfaces, query_runs, sequence_surfaces)
    evidence = _build_evidence(query_runs, sequence_runs, tuning_runs)
    return Rq8ReportBundle(
        query_runs=query_runs,
        sequence_runs=sequence_runs,
        tuning_markdown=_render_tuning(
            tuning_runs,
            {key: value.selected for key, value in query_surfaces.items()},
            {
                key: value.selected
                for key, value in sequence_surfaces.items()
            },
        ),
        reader_markdown=_render_reader(query_runs, sequence_runs),
        evidence=evidence,
    )


def write_report_bundle(
    bundle: Rq8ReportBundle,
    scratchpad: Path,
    evidence: Path,
) -> dict[str, Path]:
    paths = {
        "tuning": scratchpad / "rq8_reinvestigation_tuning_500m.md",
        "reader": scratchpad / "rq8_reinvestigation_reader_500m.md",
        "evidence": evidence / "rq8_reinvestigation_results.json",
    }
    _write_atomically(paths["tuning"], bundle.tuning_markdown)
    _write_atomically(paths["reader"], bundle.reader_markdown)
    _write_atomically(
        paths["evidence"],
        json.dumps(bundle.evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return paths


def _artifact_complete(logs: Path, candidate: Rq8Candidate) -> bool:
    directory = logs / candidate.run_name
    return all(
        (directory / name).is_file()
        for name in ("training_metadata.json", "final_metrics.json", "sweep.log")
    )


def _load_run(logs: Path, candidate: Rq8Candidate) -> Run:
    directory = logs / candidate.run_name
    if not _artifact_complete(logs, candidate):
        raise Rq8ReportError(f"{candidate.run_name}: artifact is incomplete")
    metadata = _load_json(directory / "training_metadata.json", candidate.run_name)
    metrics = _load_json(directory / "final_metrics.json", candidate.run_name)
    _validate_metadata(metadata, candidate)
    selected = _load_selection_metrics(directory, metadata)
    final_metrics = {
        metric: _required_bounded_metric(metrics, metric, candidate.run_name)
        for metric in _METRICS
    }
    _require_equal(metrics.get("num_users"), 37018, candidate.run_name, "num_users")
    return Run(
        candidate=candidate,
        best_epoch=_required_int(metadata, "best_epoch", candidate.run_name),
        stopped_epoch=_required_int(metadata, "stopped_epoch", candidate.run_name),
        validation_metrics=selected,
        metrics=final_metrics,
    )


def _validate_metadata(metadata: dict[str, Any], candidate: Rq8Candidate) -> None:
    context = candidate.run_name
    cls_mode = _cls_mode(candidate)
    top_level = {
        "training_semantics_revision": GENERATION_TRAINING_SEMANTICS_REVISION,
        "dataset_size": "500m",
        "seed": candidate.seed,
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 20,
        "stopped_epoch": 20,
        "early_stopped": False,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "batch_size": 1280,
        "physical_batch_size": 1280,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 1280,
        "val_batch_size": 8192,
        "num_workers": 4,
        "prefetch_factor": 4,
        "model_dim": 64,
        "item_embedding_dim": 64,
        "embedding_learning_rate": 0.064,
        "deep_learning_rate": candidate.deep_lr,
        "weight_decay": 0.0,
        "initializer_std": 0.02,
        "runtime_dtype": "torch.bfloat16",
        "runtime_compile": False,
        "gradient_clip_norm": None,
        "negative_sampling": "random",
        "cls_token": cls_mode != "none",
        "cls_token_mode": cls_mode,
        "lr_schedule_horizon_epochs": 20,
    }
    for key, expected in top_level.items():
        _require_equal(metadata.get(key), expected, context, key)
    best_epoch = _required_int(metadata, "best_epoch", context)
    if not 1 <= best_epoch <= 20:
        raise Rq8ReportError(f"{context}: best_epoch must be in [1, 20]")

    invariants = metadata.get("transfer_invariants")
    if not isinstance(invariants, dict):
        raise Rq8ReportError(f"{context}: transfer_invariants is absent")
    expected_invariants = {
        "experiment_class": "MuTransferGenerationExperiment",
        "mup_base_dim": 16,
        "mup_delta_dim": 32,
        "mup_base_ffn_dim": None,
        "mup_delta_ffn_dim": None,
        "dataset_size": "500m",
        "user_sample": None,
        "event_type_filter": "like",
        "min_item_interactions_per_item": 5,
        "drop_unmapped_items": True,
        "validation_interval_seconds": 604800,
        "day_range": {"start_day": 0, "end_day": 300},
        "batch_size": 1280,
        "physical_batch_size": 1280,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 1280,
        "model_dim": 64,
        "item_embedding_dim": 64,
        "max_seq_len": candidate.max_seq_len,
        "window": "next_item",
        "bos": False,
        "cls_token": cls_mode != "none",
        "cls_token_mode": cls_mode,
        "timestamp_delta": "bins",
        "timestamp_combination": "add",
        "timestamp_num_bins": 16,
        "timestamp_bin_semantics_revision": 2,
        "per_layer_item_embeddings": False,
        "negative_sampling": "random",
        "num_in_batch_negatives": 512,
        "logq_correction": "yi2019",
        "random_negative_fraction": 0.5,
        "logq_alpha": 0.01,
        "correct_positive_logq": False,
        "mask_false_negatives": False,
        "exclude_own_group_negatives": False,
        "dense_random_negative_scores": False,
        "eval_ks": [10, 50, 100],
        "eval_max_users": 20000,
        "eval_every_n_epochs": 1,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0,
        "early_stopping_metric": "recall@100",
        "early_stopping_metric_prefix": "epoch/val_true",
        "selection_k": 100,
        "evaluation_catalog": "all",
        "exclude_seen_from_evaluation": False,
        "restore_best_weights": True,
        "adaptive_schedule_early_stopping": False,
        "lr_schedule_horizon_epochs": 20,
        "lr_schedule": _expected_schedule(),
    }
    for key, expected in expected_invariants.items():
        _require_equal(invariants.get(key), expected, context, f"transfer_invariants.{key}")
    transformer = invariants.get("transformer")
    if not isinstance(transformer, dict):
        raise Rq8ReportError(f"{context}: transfer_invariants.transformer is absent")
    for key, expected in _expected_transformer(candidate).items():
        _require_equal(
            transformer.get(key),
            expected,
            context,
            f"transfer_invariants.transformer.{key}",
        )


def _expected_transformer(candidate: Rq8Candidate) -> dict[str, object]:
    if candidate.position_method == "learned_forward":
        position = {"alibi": False, "rope": None, "learned_positions": "forward"}
    elif candidate.position_method == "alibi":
        position = {"alibi": True, "rope": None, "learned_positions": None}
    else:
        position = {"alibi": True, "rope": "reverse", "learned_positions": None}
    return {
        "dim": 64,
        "num_layers": 2,
        "nhead": 2,
        "num_kv_heads": 1,
        "ffn_intermediate_dim": 171,
        "dropout": 0.1,
        "input_dropout": 0.1,
        "ffn_dropout": 0.1,
        "gated_ffn_dropout": False,
        "ffn": "swiglu",
        "norm": "layer",
        "norm_place": "pre",
        "input_norm": None,
        "final_norm": "layer",
        **position,
        "attention_window": (
            {
                "standard": 50,
                "end_only": 51,
                "interleaved": 100,
            }[candidate.query_method]
            if candidate.study == "query"
            else None
        ),
    }


def _expected_schedule() -> dict[str, object]:
    return {
        "shape": "linear",
        "warmup_fraction": 0.0,
        "min_lr_fraction": 0.0,
        "cycles": 1,
        "timescale_steps": None,
        "timescale_fraction": None,
        "power_exponent": -0.51,
        "power_transition_tokens": None,
        "optimizer_group_scope": "both",
    }


def _load_selection_metrics(directory: Path, metadata: dict[str, Any]) -> dict[str, float]:
    best_epoch = _required_int(metadata, "best_epoch", directory.name)
    epoch_index = best_epoch - 1
    values: set[tuple[float, float]] = set()
    for line in (directory / "sweep.log").read_text().splitlines():
        if re.search(rf"\bepoch {epoch_index} finished\b", line) is None:
            continue
        recall = re.search(rf"\bepoch/val_true\.recall@100=({_METRIC_NUMBER})\b", line)
        ndcg = re.search(rf"\bepoch/val_true\.ndcg@100=({_METRIC_NUMBER})\b", line)
        if recall is None or ndcg is None:
            continue
        pair = (float(recall.group(1)), float(ndcg.group(1)))
        if all(math.isfinite(value) and 0 <= value <= 1 for value in pair):
            values.add(pair)
    if len(values) != 1:
        raise Rq8ReportError(
            f"{directory.name}: best epoch has missing or conflicting validation metrics"
        )
    recall, ndcg = values.pop()
    return {"recall@100": recall, "ndcg@100": ndcg}


def _group_initial_runs(runs: list[Run]) -> dict[tuple[object, ...], list[Run]]:
    grouped: dict[tuple[object, ...], list[Run]] = {}
    for run in runs:
        candidate = run.candidate
        key = (
            ("query", candidate.query_method)
            if candidate.study == "query"
            else ("sequence", candidate.position_method, candidate.max_seq_len)
        )
        grouped.setdefault(key, []).append(run)
    return grouped


def _resolve_surface(
    logs: Path, label: str, initial: list[Run]
) -> SurfaceResolution:
    runs = list(initial)
    for _ in range(16):
        winner = _select(runs)
        rates = sorted(run.candidate.deep_lr for run in runs)
        if winner.candidate.deep_lr not in (rates[0], rates[-1]):
            return SurfaceResolution(selected=winner, runs=tuple(runs))
        side = "low" if winner.candidate.deep_lr == rates[0] else "high"
        previous_steps = [
            run.candidate.boundary_step or 0
            for run in runs
            if run.candidate.stage == "boundary"
            and run.candidate.boundary_side == side
        ]
        continuation = make_boundary_candidate(
            winner.candidate, side, max(previous_steps, default=0) + 1
        )
        if not _artifact_complete(logs, continuation):
            raise Rq8ReportError(
                f"RQ8 {label} selects boundary deep LR {winner.candidate.deep_lr:g}; "
                f"required continuation {continuation.deep_lr:g} is absent "
                f"({continuation.run_name})"
            )
        runs.append(_load_run(logs, continuation))
    raise Rq8ReportError(f"RQ8 {label} did not resolve after 16 boundary continuations")


def _select(runs: list[Run]) -> Run:
    ordered = sorted(
        runs,
        key=lambda run: (
            run.validation_metrics["recall@100"],
            run.validation_metrics["ndcg@100"],
        ),
        reverse=True,
    )
    if len(ordered) > 1 and ordered[0].validation_metrics == ordered[1].validation_metrics:
        raise Rq8ReportError(
            f"{ordered[0].candidate.surface_key}: exact validation tie after "
            "recall@100 and NDCG@100"
        )
    return ordered[0]


def _load_query_repeats(logs: Path, winner: Run) -> tuple[Run, ...]:
    repeats = [winner]
    for seed in _CONFIRMATION_SEEDS[1:]:
        candidate = make_confirmation_candidate(winner.candidate, seed)
        if not _artifact_complete(logs, candidate):
            raise Rq8ReportError(
                f"RQ8 query confirmation for {winner.candidate.query_method}, "
                f"seed {seed}, is absent ({candidate.run_name})"
            )
        repeats.append(_load_run(logs, candidate))
    return tuple(repeats)


def _all_tuning_runs(
    query_surfaces: dict[str, SurfaceResolution],
    query_runs: dict[str, tuple[Run, ...]],
    sequence_surfaces: dict[tuple[str, int], SurfaceResolution],
) -> dict[tuple[object, ...], tuple[Run, ...]]:
    result: dict[tuple[object, ...], tuple[Run, ...]] = {}
    for method in _QUERY_METHODS:
        base = list(query_surfaces[method].runs)
        base += list(query_runs[method][1:])
        result[("query", method)] = tuple(base)
    for key, surface in sequence_surfaces.items():
        result[("sequence", *key)] = surface.runs
    return result


def _render_reader(
    query_runs: dict[str, tuple[Run, ...]],
    sequence_runs: dict[str, dict[int, Run]],
) -> str:
    lines = [
        "## RQ8 — How do scaling and architecture choices affect metrics?",
        "",
        _query_table(query_runs),
        "",
        _sequence_table("alibi", sequence_runs["alibi"]),
        "",
        _sequence_table(
            "rope_reverse_alibi", sequence_runs["rope_reverse_alibi"]
        ),
    ]
    return "\n".join(lines) + "\n"


def _query_table(query_runs: dict[str, tuple[Run, ...]]) -> str:
    means = {method: _mean_metrics(runs) for method, runs in query_runs.items()}
    control = means["standard"]
    labels = {
        "standard": "standard item-state",
        "end_only": "**end-only CLS**",
        "interleaved": "interleaved CLS",
    }
    lines = [
        "| query objective | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in _QUERY_METHODS:
        metrics = means[method]
        cells = [labels[method]]
        for metric in _METRICS:
            cells.append(
                reporting.absolute(metrics[metric])
                if method == "standard"
                else reporting.change_cell(metrics[metric], control[metric], metric)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _sequence_table(position: str, selected: dict[int, Run]) -> str:
    if tuple(selected) != _SEQUENCE_LENGTHS:
        raise Rq8ReportError(f"RQ8 sequence {position} has malformed length order")
    reference = selected[128].metrics
    best_length = max(
        _SEQUENCE_LENGTHS,
        key=lambda length: selected[length].metrics["recall@100"],
    )
    label = {
        "alibi": "causal ALiBi retained history length",
        "rope_reverse_alibi": "reverse-RoPE + ALiBi retained history length",
    }[position]
    lines = [
        f"| {label} | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for length in _SEQUENCE_LENGTHS:
        metrics = selected[length].metrics
        cells = [str(length)]
        for metric in _METRICS:
            cells.append(
                reporting.absolute(metrics[metric])
                if length == 128
                else reporting.change_cell(metrics[metric], reference[metric], metric)
            )
        if length == best_length:
            cells = [f"**{cell}**" for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_tuning(
    grouped: dict[tuple[object, ...], tuple[Run, ...]],
    selected_query: dict[str, Run],
    selected_sequence: dict[tuple[str, int], Run],
) -> str:
    lines = [
        "# G1 RQ8 — query-token and full-causal sequence tuning on native Yambda-500M",
        "",
        (
            "Selection uses best-epoch validation recall@100, then same-epoch "
            "validation NDCG@100. Full-user metrics come from restored best checkpoints."
        ),
    ]
    for method in _QUERY_METHODS:
        lines += ["", f"### Query: {_query_label(method)}", ""]
        lines += _tuning_table(grouped[("query", method)], selected_query[method])
    for position in _POSITION_METHODS:
        for length in _SEQUENCE_LENGTHS:
            lines += [
                "",
                f"### Sequence: {_position_label(position)}, length {length}",
                "",
            ]
            lines += _tuning_table(
                grouped[("sequence", position, length)],
                selected_sequence[(position, length)],
            )
    return "\n".join(lines) + "\n"


def _tuning_table(runs: tuple[Run, ...], selected: Run) -> list[str]:
    lines = [
        "| seed | embedding LR | deep LR | batch size | best/stopped/horizon epoch | validation recall@100 | validation ndcg@100 | full-user recall@100 | full-user ndcg@100 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in sorted(
        runs, key=lambda item: (item.candidate.seed, item.candidate.deep_lr)
    ):
        values = (
            str(run.candidate.seed),
            f"{run.candidate.embedding_lr:.3f}",
            f"{run.candidate.deep_lr:.6g}",
            str(run.candidate.batch_size),
            f"{run.best_epoch}/{run.stopped_epoch}/20",
            f"{run.validation_metrics['recall@100']:.4f}",
            f"{run.validation_metrics['ndcg@100']:.4f}",
            reporting.absolute(run.metrics["recall@100"]),
            reporting.absolute(run.metrics["ndcg@100"]),
        )
        if run.candidate.seed == 42 and run.candidate.deep_lr == selected.candidate.deep_lr:
            values = tuple(f"**{value}**" for value in values)
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _build_evidence(
    query_runs: dict[str, tuple[Run, ...]],
    sequence_runs: dict[str, dict[int, Run]],
    tuning_runs: dict[tuple[object, ...], tuple[Run, ...]],
) -> dict[str, object]:
    all_runs = {
        run.candidate.run_name: run
        for runs in tuning_runs.values()
        for run in runs
    }
    return {
        "dataset_size": "500m",
        "initial_query_surface_runs": 9,
        "initial_sequence_surface_runs": 48,
        "query_confirmation_seeds": list(_CONFIRMATION_SEEDS),
        "protocol": {
            "embedding_learning_rate": 0.064,
            "batch_size": 1280,
            "linear_horizon_epochs": 20,
            "training_semantics_revision": GENERATION_TRAINING_SEMANTICS_REVISION,
            "sequence_attention_window": None,
            "sequence_attention": "full_causal",
            "sequence_protocol_revision": 2,
            "sequence_lengths": list(_SEQUENCE_LENGTHS),
            "sequence_position_methods": list(_POSITION_METHODS),
            "sequence_query_method": "standard",
        },
        "query_results": [
            {
                "method": method,
                "selected_deep_learning_rate": runs[0].candidate.deep_lr,
                "mean_full_user_metrics": _mean_metrics(runs),
                "artifacts": [run.candidate.run_name for run in runs],
            }
            for method, runs in query_runs.items()
        ],
        "sequence_results": [
            {
                "position_method": position,
                "max_seq_len": length,
                "selected_deep_learning_rate": run.candidate.deep_lr,
                "full_user_metrics": run.metrics,
                "artifact": run.candidate.run_name,
            }
            for position in _POSITION_METHODS
            for length, run in sequence_runs[position].items()
        ],
        "validated_artifacts": [
            {
                "run_name": name,
                "study": run.candidate.study,
                "seed": run.candidate.seed,
                "deep_learning_rate": run.candidate.deep_lr,
                "validation_metrics": run.validation_metrics,
                "full_user_metrics": run.metrics,
            }
            for name, run in sorted(all_runs.items())
        ],
    }


def _mean_metrics(runs: tuple[Run, ...]) -> dict[str, float]:
    return {
        metric: sum(run.metrics[metric] for run in runs) / len(runs)
        for metric in _METRICS
    }


def _cls_mode(candidate: Rq8Candidate) -> str:
    if candidate.study != "query" or candidate.query_method == "standard":
        return "none"
    return candidate.query_method


def _query_label(method: str) -> str:
    return {
        "standard": "standard",
        "end_only": "end-only CLS",
        "interleaved": "interleaved CLS",
    }[method]


def _position_label(method: str) -> str:
    return {
        "alibi": "causal ALiBi",
        "rope_reverse_alibi": "reverse-RoPE + ALiBi",
    }[method]


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq8ReportError(f"{context}: cannot read {path.name}") from error
    if not isinstance(value, dict):
        raise Rq8ReportError(f"{context}: {path.name} must contain an object")
    return value


def _required_int(mapping: dict[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise Rq8ReportError(f"{context}: {key} must be an integer")
    return value


def _required_bounded_metric(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise Rq8ReportError(f"{context}: {key} must be finite and in [0, 1]")
    return float(value)


def _require_equal(value: object, expected: object, context: str, key: str) -> None:
    if value != expected:
        raise Rq8ReportError(f"{context}: {key} expected {expected!r}, got {value!r}")


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
    for path in write_report_bundle(bundle, arguments.scratchpad, arguments.evidence).values():
        print(path)


if __name__ == "__main__":
    main()
