from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import polars as pl
import torch

from dcn.data.features import FeatureValues
from dcn.main import load_experiment
from dcn.models.cross_attention_retrieval import CrossAttentionRetrievalModel
from dcn.nn.sampled_softmax import OfflineInBatchSoftmax
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import (
    make_selected_cap_candidates,
    rq13_cap4_candidates,
)
from utils.global_config import config as global_config


REQUIRED_CHECKS = (
    "prefix_counts_and_latest_slices",
    "target_exclusion_no_leakage",
    "encoder_attention_mask",
    "gradient_flow",
    "candidate_only_loss",
    "learning_curves",
    "lr_boundary",
)
_RESEARCH_QUESTION = "RQ13 encoder-decoder prefix expansion"
_ORIGINAL_TREATMENTS = {
    "one_example",
    "truncated_8",
    "truncated_16",
    "required_8",
    "required_16",
}
_STAGE_ONE_TREATMENTS = {*_ORIGINAL_TREATMENTS, "truncated_4"}
_TRAIN_CACHE_PATTERN = re.compile(
    r"(?:\bLoaded cached user sequences from |\bBuilt \d+ user sequences at )"
    r"(?P<path>\S+/sequences/train_[^\s/]+)(?:$|\s)"
)
_ROW_FILTER_PATTERN = re.compile(
    r'col\("(?P<event_column>[^"]+)"\)\) == \(dyn int: '
    r'(?P<event_value>-?\d+)\).*col\("(?P<timestamp_column>[^"]+)"\)\) < '
    r"\(dyn int: (?P<timestamp_cutoff>-?\d+)\)"
)
_CONFIG_PATH = Path(
    "experiments/g1_sasrec_item_ids_likes/configs/rq13_rq14_query_variant.py"
)
_IMPLEMENTATION_FILES = (
    Path("dcn/config/generation.py"),
    Path("dcn/config/networks.py"),
    Path("dcn/config/query_retrieval.py"),
    Path("dcn/config/sequence.py"),
    Path("dcn/config/settings.py"),
    Path("dcn/data/sequence_dataset.py"),
    Path("dcn/models/cross_attention_retrieval.py"),
    Path("dcn/models/history_tokens.py"),
    Path("dcn/models/sequence_targets.py"),
    Path("dcn/models/two_tower.py"),
    Path("dcn/nn/sampled_softmax.py"),
    Path("dcn/nn/transformer.py"),
    Path(
        "experiments/g1_sasrec_item_ids_likes/analysis/" "rq13_rq14_query_candidates.py"
    ),
    Path(
        "experiments/g1_sasrec_item_ids_likes/analysis/rq13_prefix_expansion_audit.py"
    ),
    Path(
        "experiments/g1_sasrec_item_ids_likes/analysis/rq13_prefix_expansion_report.py"
    ),
    Path("experiments/g1_sasrec_item_ids_likes/analysis/rq13_prefix_cap_fit.py"),
    Path("experiments/g1_sasrec_item_ids_likes/configs/rq13_rq14_query_variant.py"),
    Path("experiments/g1_sasrec_item_ids_likes/configs/variant.py"),
)
_MEAN_REDUCTION_PROBE_LOSS = OfflineInBatchSoftmax(
    q=torch.full((128,), 1 / 128),
    num_in_batch_negatives=1,
    correction="none",
    mask_false_negatives=False,
    exclude_own_group=False,
)


class Rq13AuditError(RuntimeError):
    pass


def audit_prefix_caches(
    caches: Mapping[int, Path],
    required_caches: Mapping[int, Path] | None = None,
) -> dict[str, object]:
    if not {1, 8, 16}.issubset(caches) or any(
        isinstance(cap, bool) or not isinstance(cap, int) or cap < 1 for cap in caches
    ):
        raise Rq13AuditError(
            "prefix audit needs caps 1, 8, and 16 plus valid extensions"
        )
    if required_caches is not None and set(required_caches) != {8, 16}:
        raise Rq13AuditError("required-prefix audit needs exactly caps 8 and 16")
    cache_groups = {
        (
            "one_example"
            if cap == 1
            else f"truncated_{cap}" if cap <= 16 else f"selected_cap_{cap}"
        ): ("truncated", cap, path)
        for cap, path in sorted(caches.items())
    }
    cache_groups.update(
        (
            {}
            if required_caches is None
            else {
                "required_8": ("required", 8, required_caches[8]),
                "required_16": ("required", 16, required_caches[16]),
            }
        )
    )
    frames = {}
    metadata = {}
    for treatment, (rule, cap, path) in cache_groups.items():
        document = _load_json(path / "metadata.json")
        params = document.get("params")
        lengths = document.get("bucket_lengths")
        if not isinstance(params, dict) or not isinstance(lengths, list):
            raise Rq13AuditError(f"{path.name}: incomplete cache metadata")
        if (
            params.get("window") != "bounded_prefix"
            or params.get("prefix_length_rule") != rule
            or params.get("prefix_cap") != cap
            or params.get("max_seq_len") != 128
        ):
            raise Rq13AuditError(f"{path.name}: wrong bounded-prefix recipe")
        bucket_paths = sorted((path / "buckets").glob("*.parquet"))
        if not bucket_paths:
            raise Rq13AuditError(f"{path.name}: no cache buckets")
        frame = pl.concat(pl.read_parquet(bucket) for bucket in bucket_paths)
        if len(frame) != sum(lengths):
            raise Rq13AuditError(f"{path.name}: bucket count metadata is stale")
        frames[treatment] = frame
        metadata[treatment] = document

    columns = frames["one_example"].columns
    if any(frame.columns != columns for frame in frames.values()):
        raise Rq13AuditError("prefix caches have different columns")
    base_params = metadata["one_example"]["params"]
    user_column = base_params.get("user_column", "uid")
    item_columns = base_params.get("columns", ["compact_item_id"])
    item_column = item_columns[0]
    timestamp_column = base_params.get("timestamp_column", "timestamp")
    required_columns = {user_column, item_column, timestamp_column}
    if not required_columns.issubset(columns):
        raise Rq13AuditError("prefix caches lack user, item, or timestamp columns")

    invariant_params = {
        key: value
        for key, value in base_params.items()
        if key not in {"prefix_length_rule", "prefix_cap"}
    }
    for treatment, document in metadata.items():
        current = {
            key: value
            for key, value in document["params"].items()
            if key not in {"prefix_length_rule", "prefix_cap"}
        }
        if current != invariant_params:
            raise Rq13AuditError(f"{treatment}: cache source recipe differs")

    unique_users = {
        treatment: frame[user_column].n_unique() for treatment, frame in frames.items()
    }
    if len(set(unique_users.values())) != 1:
        raise Rq13AuditError("prefix caps contain different users")
    per_user = {
        treatment: frame.group_by(user_column).len().sort(user_column)
        for treatment, frame in frames.items()
    }
    if not per_user["one_example"]["len"].eq(1).all():
        raise Rq13AuditError("cap 1 does not emit exactly one example per user")
    joined = per_user["truncated_8"].join(
        per_user["truncated_16"], on=user_column, suffix="_16", validate="1:1"
    )
    if not (joined["len"] == joined["len_16"].clip(upper_bound=8)).all():
        raise Rq13AuditError("cap-8 per-user counts are not capped cap-16 counts")
    if (
        not frames["truncated_16"]
        .group_by(user_column, maintain_order=True)
        .head(8)
        .equals(frames["truncated_8"])
    ):
        raise Rq13AuditError("cap 8 is not the latest cap-16 slices")
    if (
        not frames["truncated_8"]
        .group_by(user_column, maintain_order=True)
        .head(1)
        .equals(frames["one_example"])
    ):
        raise Rq13AuditError("cap 1 is not the latest cap-8 slice")
    truncated = [
        (
            cap,
            (
                "one_example"
                if cap == 1
                else f"truncated_{cap}" if cap <= 16 else f"selected_cap_{cap}"
            ),
        )
        for cap in sorted(caches)
    ]
    nested = {}
    for (smaller_cap, smaller), (larger_cap, larger) in zip(
        truncated[:-1], truncated[1:], strict=True
    ):
        matches = (
            frames[larger]
            .group_by(user_column, maintain_order=True)
            .head(smaller_cap)
            .equals(frames[smaller])
        )
        if not matches:
            raise Rq13AuditError(
                f"cap {smaller_cap} is not the latest cap-{larger_cap} slice"
            )
        nested[f"{smaller_cap}_in_{larger_cap}"] = True

    source = _source_events(base_params, user_column, timestamp_column, item_columns)
    source_matches = {}
    for treatment, (rule, cap, _) in cache_groups.items():
        expected = _expected_prefixes(
            source,
            user_column=user_column,
            timestamp_column=timestamp_column,
            item_columns=item_columns,
            max_seq_len=base_params["max_seq_len"],
            min_seq_len=base_params["min_seq_len"],
            prefix_length_rule=rule,
            prefix_cap=cap,
        )
        if not _frames_equal_chunked(frames[treatment], expected):
            raise Rq13AuditError(
                f"{treatment}: cached prefixes differ from latest source histories"
            )
        source_matches[treatment] = True

    minimum_length = math.inf
    maximum_length = 0
    for frame in frames.values():
        item_lengths = frame[item_column].list.len()
        timestamp_lengths = frame[timestamp_column].list.len()
        if not item_lengths.equals(timestamp_lengths):
            raise Rq13AuditError("item and timestamp slice lengths differ")
        minimum_length = min(minimum_length, item_lengths.min())
        maximum_length = max(maximum_length, item_lengths.max())
        minimum_delta = frame.select(
            pl.col(timestamp_column).list.diff().list.min().fill_null(0).min()
        ).item()
        if minimum_delta < 0:
            raise Rq13AuditError("a prefix slice is not chronological")
    if minimum_length < 2 or maximum_length > 129:
        raise Rq13AuditError("prefix slices do not contain history plus one target")

    return {
        "passed": True,
        "cache_names": {
            treatment: path.name for treatment, (_, _, path) in cache_groups.items()
        },
        "cache_files_sha256": {
            treatment: _cache_files_sha256(path)
            for treatment, (_, _, path) in cache_groups.items()
        },
        "unique_users": unique_users["one_example"],
        "expanded_examples": {
            treatment: len(frame) for treatment, frame in frames.items()
        },
        "source_file_count": len(base_params["parquet_files"]),
        "source_files_manifest_sha256": _canonical_sha256(
            {
                str(Path(path).resolve()): _file_sha256(Path(path))
                for path in base_params["parquet_files"]
            }
        ),
        "source_history_matches": source_matches,
        "cap1_is_latest_cap8_slice": True,
        "cap8_is_latest_cap16_slice": True,
        "nested_latest_slices": nested,
        "sequence_length_range": [int(minimum_length), int(maximum_length)],
        "target_position": "final event of each cached sequence",
        "model_history_length_range": [
            int(minimum_length - 1),
            int(maximum_length - 1),
        ],
    }


def eligible_target_counts_from_cache(cache: Path) -> list[int]:
    document = _load_json(cache / "metadata.json")
    params = document.get("params")
    if not isinstance(params, dict):
        raise Rq13AuditError(f"{cache.name}: incomplete cache metadata")
    user_column = params.get("user_column", "uid")
    timestamp_column = params.get("timestamp_column", "timestamp")
    item_columns = params.get("columns", ["compact_item_id"])
    if (
        not isinstance(user_column, str)
        or not isinstance(timestamp_column, str)
        or not isinstance(item_columns, list)
        or not item_columns
        or not all(isinstance(column, str) for column in item_columns)
    ):
        raise Rq13AuditError(f"{cache.name}: invalid source columns")
    source = _source_events(params, user_column, timestamp_column, item_columns)
    counts = (
        source.group_by(user_column)
        .len()
        .select((pl.col("len") - 1).alias("eligible_targets"))
        .filter(pl.col("eligible_targets") > 0)
        .get_column("eligible_targets")
        .to_list()
    )
    if not counts:
        raise Rq13AuditError(f"{cache.name}: no eligible source histories")
    return counts


def _source_events(
    params: Mapping[str, object],
    user_column: str,
    timestamp_column: str,
    item_columns: list[str],
) -> pl.DataFrame:
    parquet_files = params.get("parquet_files")
    row_filter = params.get("row_filter")
    if not isinstance(parquet_files, list) or not parquet_files:
        raise Rq13AuditError("prefix cache has no source parquet provenance")
    if not isinstance(row_filter, str):
        raise Rq13AuditError("prefix cache has no source row-filter provenance")
    match = _ROW_FILTER_PATTERN.search(row_filter)
    if match is None or len(re.findall(r'col\("', row_filter)) != 2:
        raise Rq13AuditError("prefix cache row filter is not auditable")
    filter_timestamp = match.group("timestamp_column")
    if filter_timestamp != timestamp_column:
        raise Rq13AuditError("prefix cache filter uses a different timestamp column")
    event_column = match.group("event_column")
    needed = list(
        dict.fromkeys([user_column, timestamp_column, *item_columns, event_column])
    )
    return (
        pl.concat(pl.scan_parquet(Path(path)).select(needed) for path in parquet_files)
        .filter(
            (pl.col(event_column) == int(match.group("event_value")))
            & (pl.col(timestamp_column) < int(match.group("timestamp_cutoff")))
        )
        .select(
            pl.col(user_column),
            pl.col(timestamp_column).cast(pl.Int64),
            *[pl.col(column) for column in item_columns],
        )
        .collect(engine="streaming")
        .sort([user_column, timestamp_column], maintain_order=True)
    )


def _expected_prefixes(
    events: pl.DataFrame,
    *,
    user_column: str,
    timestamp_column: str,
    item_columns: list[str],
    max_seq_len: int,
    min_seq_len: int,
    prefix_length_rule: str,
    prefix_cap: int,
) -> pl.DataFrame:
    list_columns = [timestamp_column, *item_columns]
    grouped = events.group_by(user_column, maintain_order=True).agg(
        pl.col(timestamp_column),
        *[pl.col(column) for column in item_columns],
    )
    user_length = pl.col(timestamp_column).list.len().cast(pl.Int64)
    latest_target = user_length - 1
    if prefix_length_rule == "truncated":
        prefix_count = pl.min_horizontal(prefix_cap, latest_target)
    elif prefix_length_rule == "required":
        prefix_count = (
            pl.when(latest_target < max_seq_len)
            .then(1)
            .otherwise(pl.min_horizontal(prefix_cap, user_length - max_seq_len))
        )
    else:
        raise Rq13AuditError(f"unknown prefix rule {prefix_length_rule!r}")
    return (
        grouped.with_columns(
            user_length.alias("_user_length"),
            pl.int_ranges(0, prefix_count).alias("_rank"),
        )
        .explode("_rank")
        .drop_nulls("_rank")
        .with_columns((pl.col("_user_length") - 1 - pl.col("_rank")).alias("_target"))
        .with_columns(
            pl.max_horizontal(0, pl.col("_target") - max_seq_len).alias("_start")
        )
        .with_columns(
            pl.col(column).list.slice(
                pl.col("_start"), pl.col("_target") - pl.col("_start") + 1
            )
            for column in list_columns
        )
        .filter(pl.col(timestamp_column).list.len() >= min_seq_len)
        .drop("_user_length", "_rank", "_target", "_start")
        .select(user_column, *list_columns)
    )


def _frames_equal_chunked(
    left: pl.DataFrame, right: pl.DataFrame, chunk_rows: int = 10_000
) -> bool:
    if left.schema != right.schema or len(left) != len(right):
        return False
    return all(
        left.slice(offset, chunk_rows)
        .rechunk()
        .equals(right.slice(offset, chunk_rows).rechunk())
        for offset in range(0, len(left), chunk_rows)
    )


def run_model_correctness_probe(
    run_name: str = "g1_rq13_truncated_16_d0012_seed42_h20_r1_500m",
) -> dict[str, object]:
    experiment = _load_production_experiment(run_name)
    model = experiment.base_model
    targets = experiment.create_targets()
    criterion = experiment.create_criterion()
    model.eval()
    optimizer = experiment.create_optimizers()
    initial_readout_l1 = float(model.query_projection.weight.abs().sum())
    if initial_readout_l1 != 0:
        raise Rq13AuditError("the RQ13 μP readout probe is not zero-initialized")
    previous_cpu_attention = global_config.cpu_attention
    global_config.set_cpu_attention(True)
    try:
        history = _packed_batch([[1, 2, 3], [10, 11, 12]])
        future_changed = _packed_batch([[1, 2, 4], [10, 11, 12]])
        other_user_changed = _packed_batch([[1, 2, 3], [10, 11, 13]])
        encoded = _encoded_history(model, history)
        encoded_future_changed = _encoded_history(model, future_changed)
        encoded_other_user_changed = _encoded_history(model, other_user_changed)
        future_delta = float((encoded[0] - encoded_future_changed[0]).abs().max())
        other_user_delta = float(
            (encoded[:3] - encoded_other_user_changed[:3]).abs().max()
        )

        training = _packed_batch([[1, 2, 3], [10, 11, 12]])
        bootstrap_pairs = targets(model(training))
        bootstrap_loss = criterion(training)["loss"]
        bootstrap_loss.backward()
        first_readout_gradient = _gradient_l1(model.query_projection.parameters())
        first_encoder_gradient = _gradient_l1(model.memory_encoder.parameters())
        first_cross_gradient = _gradient_l1(
            model.decoder.cross_attention_layers.parameters()
        )
        optimizer.step()
        bootstrapped_readout_l1 = float(model.query_projection.weight.abs().sum())
        model.zero_grad(set_to_none=True)

        target_changed = _packed_batch(
            [[1, 2, 4], [10, 11, 13]],
            timestamps=[[0, 1, 99], [0, 1, 199]],
        )
        pairs = targets(model(training))
        changed_pairs = targets(model(target_changed))
        target_query_delta = float(
            (pairs.query_repr - changed_pairs.query_repr).abs().max()
        )
        query_l1 = float(pairs.query_repr.abs().sum())
        if pairs.positive_ids.tolist() != [3, 12]:
            raise Rq13AuditError("the model did not expose each final event as target")

        candidate_loss = _candidate_loss(pairs.query_repr, pairs.positive_repr)
        duplicated = _packed_batch([[1, 2, 3], [10, 11, 12], [1, 2, 3], [10, 11, 12]])
        duplicated_pairs = targets(model(duplicated))
        duplicated_loss = _candidate_loss(
            duplicated_pairs.query_repr, duplicated_pairs.positive_repr
        )
        loss_delta = float((candidate_loss - duplicated_loss).abs())

        model.zero_grad(set_to_none=True)
        pairs = targets(model(training))
        criterion(training)["loss"].backward()
        encoder_gradient = _gradient_l1(model.memory_encoder.parameters())
        cross_gradient = _gradient_l1(model.decoder.cross_attention_layers.parameters())
    finally:
        global_config.set_cpu_attention(previous_cpu_attention)

    encoder_is_bidirectional = all(
        not layer.is_causal for layer in model.memory_encoder.layers
    )
    checks = {
        "production_run_name": experiment.run_name,
        "production_experiment_class": type(experiment).__name__,
        "production_query_architecture": experiment.query_architecture,
        "production_window": experiment.window,
        "production_prefix_length_rule": experiment.prefix_length_rule,
        "production_prefix_cap": experiment.prefix_cap,
        "production_targets_class": type(targets).__name__,
        "production_criterion_class": type(criterion).__name__,
        "production_optimizer_class": type(optimizer).__name__,
        "encoder_is_bidirectional": encoder_is_bidirectional,
        "future_history_changes_earlier_state": future_delta > 1e-6,
        "other_user_history_max_delta": other_user_delta,
        "initial_readout_l1": initial_readout_l1,
        "bootstrap_readout_gradient_l1": first_readout_gradient,
        "bootstrap_memory_encoder_gradient_l1": first_encoder_gradient,
        "bootstrap_cross_attention_gradient_l1": first_cross_gradient,
        "bootstrapped_readout_l1": bootstrapped_readout_l1,
        "target_only_query_max_delta": target_query_delta,
        "changed_target_timestamps": [99, 199],
        "post_bootstrap_query_l1": query_l1,
        "candidate_targets": int(pairs.positive_ids.numel()),
        "candidate_targets_per_example": int(
            pairs.positive_ids.numel() / pairs.group_sizes.numel()
        ),
        "candidate_loss": float(candidate_loss),
        "duplicated_batch_loss": float(duplicated_loss),
        "duplicated_batch_loss_delta": loss_delta,
        "memory_encoder_gradient_l1": encoder_gradient,
        "cross_attention_gradient_l1": cross_gradient,
    }
    if (
        not encoder_is_bidirectional
        or future_delta <= 1e-6
        or other_user_delta > 1e-6
        or first_readout_gradient <= 0
        or first_encoder_gradient != 0
        or first_cross_gradient != 0
        or bootstrapped_readout_l1 <= 0
        or target_query_delta > 1e-6
        or query_l1 <= 0
        or checks["candidate_targets"] != 2
        or checks["candidate_targets_per_example"] != 1
        or loss_delta > 1e-6
        or encoder_gradient <= 0
        or cross_gradient <= 0
    ):
        raise Rq13AuditError("production-model correctness probe failed")
    return {"passed": True, **checks}


def _load_production_experiment(run_name: str) -> Any:
    controlled = {
        "G1_QUERY_RUN": run_name,
        "DCN_GPU_LOCK_SLOT": "rq13-correctness-audit",
    }
    cleared = (
        "G1_MAX_USERS",
        "G1_TRAIN_BATCH_SIZE",
        "G1_VAL_BATCH_SIZE",
    )
    previous = {key: os.environ.get(key) for key in (*controlled, *cleared)}
    try:
        os.environ.update(controlled)
        for key in cleared:
            os.environ.pop(key, None)
        experiment = load_experiment(_CONFIG_PATH)
        experiment.setup()
        experiment.__dict__["device"] = torch.device("cpu")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if (
        type(experiment).__name__ != "MuTransferCrossAttentionGenerationExperiment"
        or experiment.run_name != run_name
        or experiment.query_architecture != "encoder_decoder"
        or experiment.window != "bounded_prefix"
        or experiment.max_seq_len != 128
        or experiment.transformer.dim != 64
        or experiment.transformer.num_layers != 2
        or experiment.transformer.nhead != 2
        or experiment.transformer.num_kv_heads != 1
        or experiment.transformer.ffn != "swiglu"
        or experiment.transformer.ffn_intermediate_dim != 171
        or experiment.transformer.attention_window is not None
        or experiment.timestamp_delta != "bins"
        or experiment.timestamp_num_bins != 16
        or experiment.retrieval_decoder.num_layers != 1
        or experiment.retrieval_decoder.ffn != "swiglu"
        or experiment.retrieval_decoder.ffn_intermediate_dim != 128
        or experiment.negative_sampling != "random"
        or experiment.num_in_batch_negatives != 512
        or experiment.dense_random_negative_scores
    ):
        raise Rq13AuditError("loaded production RQ13 recipe differs from the audit")
    return experiment


def build_correctness_audit(logs: Path, results_path: Path) -> dict[str, object]:
    results = _load_json(results_path)
    validate_audit_generation_stage(results)
    surfaces = _audit_result_surfaces(results, logs)
    selected = results.get("selected")
    if not selected:
        selected = results.get("surface_winners")
    if not isinstance(selected, dict):
        raise Rq13AuditError("RQ13 selected treatments are absent")
    truncated_treatments = {
        1: "one_example",
        8: "truncated_8",
        16: "truncated_16",
    }
    if "truncated_4" in selected:
        truncated_treatments[4] = "truncated_4"
    fitted = [
        treatment for treatment in selected if treatment.startswith("selected_cap_")
    ]
    if len(fitted) > 1:
        raise Rq13AuditError("RQ13 has multiple fitted-cap treatments")
    if fitted:
        truncated_treatments[int(fitted[0].rsplit("_", 1)[1])] = fitted[0]
    caches = {
        cap: _selected_train_cache(logs, selected[treatment])
        for cap, treatment in truncated_treatments.items()
    }
    required_caches = {
        cap: _selected_train_cache(logs, selected[treatment])
        for cap, treatment in ((8, "required_8"), (16, "required_16"))
    }
    probe_selection = selected.get(fitted[0] if fitted else "truncated_16")
    if not isinstance(probe_selection, dict) or not isinstance(
        probe_selection.get("run_name"), str
    ):
        raise Rq13AuditError("selected truncated-16 run identity is absent")
    prefix = audit_prefix_caches(caches, required_caches)
    model = run_model_correctness_probe(probe_selection["run_name"])
    counts = surfaces["candidate_counts"]
    checks = {
        "prefix_counts_and_latest_slices": prefix,
        "target_exclusion_no_leakage": {
            "passed": model["target_only_query_max_delta"] <= 1e-6,
            "target_only_query_max_delta": model["target_only_query_max_delta"],
            "cached_target_position": prefix["target_position"],
        },
        "encoder_attention_mask": {
            "passed": bool(
                model["encoder_is_bidirectional"]
                and model["future_history_changes_earlier_state"]
                and model["other_user_history_max_delta"] <= 1e-6
            ),
            "encoder_is_bidirectional": model["encoder_is_bidirectional"],
            "future_history_changes_earlier_state": model[
                "future_history_changes_earlier_state"
            ],
            "other_user_history_max_delta": model["other_user_history_max_delta"],
            "production_run_name": model["production_run_name"],
            "production_experiment_class": model["production_experiment_class"],
            "production_query_architecture": model["production_query_architecture"],
            "production_window": model["production_window"],
            "production_prefix_length_rule": model["production_prefix_length_rule"],
            "production_prefix_cap": model["production_prefix_cap"],
            "production_targets_class": model["production_targets_class"],
            "production_criterion_class": model["production_criterion_class"],
            "production_optimizer_class": model["production_optimizer_class"],
        },
        "gradient_flow": {
            "passed": bool(
                model["initial_readout_l1"] == 0
                and model["bootstrap_readout_gradient_l1"] > 0
                and model["bootstrap_memory_encoder_gradient_l1"] == 0
                and model["bootstrap_cross_attention_gradient_l1"] == 0
                and model["bootstrapped_readout_l1"] > 0
                and model["memory_encoder_gradient_l1"] > 0
                and model["cross_attention_gradient_l1"] > 0
            ),
            "initial_readout_l1": model["initial_readout_l1"],
            "bootstrap_readout_gradient_l1": model["bootstrap_readout_gradient_l1"],
            "bootstrap_memory_encoder_gradient_l1": model[
                "bootstrap_memory_encoder_gradient_l1"
            ],
            "bootstrap_cross_attention_gradient_l1": model[
                "bootstrap_cross_attention_gradient_l1"
            ],
            "bootstrapped_readout_l1": model["bootstrapped_readout_l1"],
            "memory_encoder_gradient_l1": model["memory_encoder_gradient_l1"],
            "cross_attention_gradient_l1": model["cross_attention_gradient_l1"],
        },
        "candidate_only_loss": {
            "passed": bool(
                counts["all_candidate_targets_equal_examples"]
                and counts["all_ntp_targets_zero"]
                and model["candidate_targets_per_example"] == 1
                and model["duplicated_batch_loss_delta"] <= 1e-6
            ),
            **counts,
            "micro_candidate_targets": model["candidate_targets"],
            "micro_candidate_targets_per_example": model[
                "candidate_targets_per_example"
            ],
            "candidate_loss": model["candidate_loss"],
            "duplicated_batch_loss": model["duplicated_batch_loss"],
            "duplicated_batch_loss_delta": model["duplicated_batch_loss_delta"],
            "reduction": (
                "deterministic fixed-logit proxy for the production "
                "sampled-softmax mean reduction over candidate targets"
            ),
        },
        "learning_curves": surfaces["learning_curves"],
        "lr_boundary": surfaces["lr_boundary"],
    }
    failed = [
        name for name in REQUIRED_CHECKS if checks[name].get("passed") is not True
    ]
    if failed:
        raise Rq13AuditError("correctness audit failed: " + ", ".join(failed))
    return {
        "schema_version": 1,
        "research_question": _RESEARCH_QUESTION,
        "dataset_size": "500m",
        "status": "passed",
        "checks": checks,
        "run_artifacts": surfaces["run_artifacts"],
        "implementation_sha256": current_implementation_sha256(),
    }


def validate_correctness_audit(
    audit: Mapping[str, object],
    expected_run_artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    if (
        audit.get("schema_version") != 1
        or audit.get("research_question") != _RESEARCH_QUESTION
        or audit.get("dataset_size") != "500m"
        or audit.get("status") != "passed"
    ):
        raise Rq13AuditError("RQ13 correctness audit identity or status is invalid")
    checks = audit.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(REQUIRED_CHECKS):
        raise Rq13AuditError("RQ13 correctness audit has incomplete checks")
    for name in REQUIRED_CHECKS:
        check = checks[name]
        if not isinstance(check, Mapping) or check.get("passed") is not True:
            raise Rq13AuditError(f"RQ13 correctness check {name} did not pass")
    _validate_check_evidence(checks, set(expected_run_artifacts))
    if audit.get("run_artifacts") != dict(expected_run_artifacts):
        raise Rq13AuditError(
            "RQ13 correctness audit is stale for current run artifacts"
        )
    if audit.get("implementation_sha256") != current_implementation_sha256():
        raise Rq13AuditError(
            "RQ13 correctness audit is stale for current implementation"
        )
    return {
        "status": "passed",
        "schema_version": 1,
        "artifact_sha256": _canonical_sha256(audit),
    }


def validate_audit_generation_stage(results: Mapping[str, object]) -> str:
    if (
        results.get("research_question") != _RESEARCH_QUESTION
        or results.get("dataset_size") != "500m"
    ):
        raise Rq13AuditError("RQ13 correctness-audit result identity is invalid")
    cap_fit = results.get("cap_fit")
    if not isinstance(cap_fit, Mapping):
        if results.get("required_followups") == []:
            return "legacy_resolved"
        raise Rq13AuditError("RQ13 correctness-audit stage is absent")
    status = cap_fit.get("status")
    missing = results.get("missing_initial_artifacts")
    boundary = results.get("required_boundary_followups")
    required = results.get("required_followups")
    if status == "pending_cap4":
        expected = [candidate.run_name for candidate in rq13_cap4_candidates()]
        if missing != expected or boundary != [] or required != expected:
            raise Rq13AuditError("RQ13 pending cap-4 stage is not exact")
        return "original_surface"
    if status in {"stage_one_audit_required", "selected_cap_pending"}:
        cap = cap_fit.get("selected_cap")
        ceiling = cap_fit.get("practical_ceiling")
        practical = ceiling.get("selected") if isinstance(ceiling, Mapping) else None
        if (
            isinstance(cap, bool)
            or not isinstance(cap, int)
            or isinstance(practical, bool)
            or not isinstance(practical, int)
            or not 17 <= cap <= practical <= 32
        ):
            raise Rq13AuditError("RQ13 fitted cap exceeds its practical ceiling")
        try:
            expected = [
                candidate.run_name for candidate in make_selected_cap_candidates(cap)
            ]
        except ValueError as error:
            raise Rq13AuditError("RQ13 fitted cap is invalid") from error
        if boundary != []:
            raise Rq13AuditError("RQ13 stage one has unresolved earlier artifacts")
        if status == "stage_one_audit_required":
            if (
                missing != []
                or required != []
                or cap_fit.get("proposed_followups") != expected
            ):
                raise Rq13AuditError(
                    "RQ13 stage-one audit has invalid selected-cap followups"
                )
            return "stage_one"
        if missing != expected or required != expected:
            raise Rq13AuditError("RQ13 selected-cap followups are not exact")
        return "stage_one_bound"
    if status in {"not_requested", "resolved"} and required == []:
        return "resolved"
    raise Rq13AuditError("RQ13 correctness-audit stage is not auditable")


def validate_bound_stage_one_audit(
    results: Mapping[str, object], audit: Mapping[str, object]
) -> dict[str, object]:
    if validate_audit_generation_stage(results) != "stage_one_bound":
        raise Rq13AuditError("RQ13 selected-cap stage is not ready")
    expected_artifacts = _stage_one_run_artifacts(results)
    validated = validate_correctness_audit(audit, expected_artifacts)
    cap_fit = results["cap_fit"]
    assert isinstance(cap_fit, Mapping)
    bindings = cap_fit.get("input_bindings")
    bound = (
        bindings.get("stage_one_correctness_audit")
        if isinstance(bindings, Mapping)
        else None
    )
    if bound != validated:
        raise Rq13AuditError("RQ13 stage-one correctness-audit binding is stale")
    return validated


def _stage_one_run_artifacts(
    results: Mapping[str, object],
) -> dict[str, Mapping[str, str]]:
    treatments = results.get("treatments")
    if not isinstance(treatments, Mapping):
        raise Rq13AuditError("RQ13 stage-one treatments are absent")
    artifacts: dict[str, Mapping[str, str]] = {}
    populated = set()
    for treatment, record in treatments.items():
        rows = record.get("artifacts") if isinstance(record, Mapping) else None
        if not isinstance(rows, list):
            raise Rq13AuditError(f"{treatment}: malformed result surface")
        if rows:
            populated.add(treatment)
        for row in rows:
            run_name = row.get("run_name") if isinstance(row, Mapping) else None
            hashes = row.get("artifact_sha256") if isinstance(row, Mapping) else None
            if (
                not isinstance(run_name, str)
                or not isinstance(hashes, Mapping)
                or run_name in artifacts
            ):
                raise Rq13AuditError(f"{treatment}: malformed or repeated artifact")
            artifacts[run_name] = hashes
    if populated != _STAGE_ONE_TREATMENTS:
        raise Rq13AuditError("RQ13 stage-one artifact treatments are not exact")
    selected = results.get("surface_winners")
    if not isinstance(selected, Mapping) or set(selected) != _STAGE_ONE_TREATMENTS:
        raise Rq13AuditError("RQ13 stage-one winners are not exact")
    return artifacts


def _validate_check_evidence(
    checks: Mapping[str, object], expected_run_names: set[str]
) -> None:
    prefix = checks["prefix_counts_and_latest_slices"]
    leakage = checks["target_exclusion_no_leakage"]
    attention = checks["encoder_attention_mask"]
    gradient = checks["gradient_flow"]
    candidate = checks["candidate_only_loss"]
    curves = checks["learning_curves"]
    boundary = checks["lr_boundary"]
    assert isinstance(prefix, Mapping)
    assert isinstance(leakage, Mapping)
    assert isinstance(attention, Mapping)
    assert isinstance(gradient, Mapping)
    assert isinstance(candidate, Mapping)
    assert isinstance(curves, Mapping)
    assert isinstance(boundary, Mapping)
    surfaces = boundary.get("surfaces")
    if not isinstance(surfaces, Mapping):
        raise Rq13AuditError("RQ13 LR-boundary evidence is incomplete")
    treatments = set(surfaces)
    source_matches = prefix.get("source_history_matches")
    expanded = prefix.get("expanded_examples")
    cache_names = prefix.get("cache_names")
    cache_hashes = prefix.get("cache_files_sha256")
    if (
        not isinstance(source_matches, Mapping)
        or set(source_matches) != treatments
        or any(value is not True for value in source_matches.values())
        or not isinstance(expanded, Mapping)
        or set(expanded) != treatments
        or any(not isinstance(value, int) or value <= 0 for value in expanded.values())
        or not isinstance(cache_names, Mapping)
        or set(cache_names) != treatments
        or not isinstance(cache_hashes, Mapping)
        or set(cache_hashes) != treatments
        or prefix.get("cap1_is_latest_cap8_slice") is not True
        or prefix.get("cap8_is_latest_cap16_slice") is not True
        or not isinstance(prefix.get("source_file_count"), int)
        or prefix["source_file_count"] <= 0
        or not isinstance(prefix.get("source_files_manifest_sha256"), str)
    ):
        raise Rq13AuditError("RQ13 prefix-source evidence is incomplete")
    if (
        not _near_zero(leakage.get("target_only_query_max_delta"))
        or leakage.get("cached_target_position")
        != "final event of each cached sequence"
    ):
        raise Rq13AuditError("RQ13 target-exclusion evidence is inconsistent")
    if (
        attention.get("encoder_is_bidirectional") is not True
        or attention.get("future_history_changes_earlier_state") is not True
        or not _near_zero(attention.get("other_user_history_max_delta"))
        or attention.get("production_experiment_class")
        != "MuTransferCrossAttentionGenerationExperiment"
        or attention.get("production_query_architecture") != "encoder_decoder"
        or attention.get("production_window") != "bounded_prefix"
        or attention.get("production_prefix_length_rule") != "truncated"
        or not isinstance(attention.get("production_prefix_cap"), int)
        or attention["production_prefix_cap"] < 16
        or attention.get("production_targets_class") != "NextItemTargets"
        or attention.get("production_criterion_class") != "TwoTowerLoss"
        or attention.get("production_optimizer_class") != "Adam"
        or not isinstance(attention.get("production_run_name"), str)
    ):
        raise Rq13AuditError("RQ13 encoder-mask evidence is inconsistent")
    if (
        gradient.get("initial_readout_l1") != 0
        or not _positive(gradient.get("bootstrap_readout_gradient_l1"))
        or gradient.get("bootstrap_memory_encoder_gradient_l1") != 0
        or gradient.get("bootstrap_cross_attention_gradient_l1") != 0
        or not _positive(gradient.get("bootstrapped_readout_l1"))
        or not _positive(gradient.get("memory_encoder_gradient_l1"))
        or not _positive(gradient.get("cross_attention_gradient_l1"))
    ):
        raise Rq13AuditError("RQ13 gradient-flow evidence is inconsistent")
    if (
        candidate.get("all_candidate_targets_equal_examples") is not True
        or candidate.get("all_ntp_targets_zero") is not True
        or candidate.get("micro_candidate_targets_per_example") != 1
        or not _near_zero(candidate.get("duplicated_batch_loss_delta"))
        or "proxy" not in str(candidate.get("reduction"))
    ):
        raise Rq13AuditError("RQ13 candidate-loss evidence is inconsistent")
    run_curves = curves.get("runs")
    crossings = curves.get("selected_threshold_crossings")
    if (
        not isinstance(run_curves, Mapping)
        or set(run_curves) != expected_run_names
        or not isinstance(crossings, Mapping)
        or set(crossings)
        != {
            "truncated_8_vs_one_example",
            "truncated_16_vs_truncated_8",
        }
    ):
        raise Rq13AuditError("RQ13 learning-curve evidence is incomplete")
    for run_name, curve in run_curves.items():
        if (
            not isinstance(curve, list)
            or len(curve) != 20
            or [point.get("epoch") for point in curve if isinstance(point, Mapping)]
            != list(range(1, 21))
            or any(
                not isinstance(point, Mapping)
                or not _finite(point.get("recall@100"))
                or not _finite(point.get("ndcg@100"))
                for point in curve
            )
        ):
            raise Rq13AuditError(f"RQ13 curve {run_name} is incomplete")
    if (
        boundary.get("required_followups") != []
        or not isinstance(surfaces, Mapping)
        or set(surfaces) != treatments
        or any(
            not isinstance(rows, list) or len(rows) < 3 for rows in surfaces.values()
        )
    ):
        raise Rq13AuditError("RQ13 LR-boundary evidence is incomplete")
    surface_rows = [row for rows in surfaces.values() for row in rows]
    if (
        any(
            not isinstance(row, Mapping)
            or row.get("run_name") not in run_curves
            or not _positive(row.get("deep_learning_rate"))
            or row.get("best_epoch") not in range(1, 21)
            or not _finite(row.get("validation_recall@100"))
            or not math.isclose(
                row["validation_recall@100"],
                run_curves[row["run_name"]][row["best_epoch"] - 1]["recall@100"],
                rel_tol=0,
                abs_tol=1e-12,
            )
            for row in surface_rows
        )
        or {row["run_name"] for row in surface_rows} != expected_run_names
        or any(
            len({row["deep_learning_rate"] for row in rows}) != len(rows)
            for rows in surfaces.values()
        )
    ):
        raise Rq13AuditError("RQ13 LR rows disagree with saved curves")
    for key, (treatment, reference) in {
        "truncated_8_vs_one_example": ("truncated_8", "one_example"),
        "truncated_16_vs_truncated_8": ("truncated_16", "truncated_8"),
    }.items():
        crossing = crossings[key]
        if (
            not isinstance(crossing, Mapping)
            or crossing.get("first_matching_epoch") not in range(1, 21)
            or crossing.get("selected_epoch") not in range(1, 21)
            or crossing["first_matching_epoch"] > crossing["selected_epoch"]
            or not _finite(crossing.get("threshold_recall@100"))
            or not _finite(crossing.get("recall@100"))
            or not _finite(crossing.get("selected_recall@100"))
            or crossing["recall@100"] < crossing["threshold_recall@100"]
            or crossing["selected_recall@100"] < crossing["recall@100"]
        ):
            raise Rq13AuditError("RQ13 threshold crossing is inconsistent")
        treatment_rows = surfaces[treatment]
        reference_rows = surfaces[reference]
        selected_row = max(treatment_rows, key=lambda row: row["validation_recall@100"])
        reference_row = max(
            reference_rows, key=lambda row: row["validation_recall@100"]
        )
        if (
            sum(
                row["validation_recall@100"] == selected_row["validation_recall@100"]
                for row in treatment_rows
            )
            != 1
            or sum(
                row["validation_recall@100"] == reference_row["validation_recall@100"]
                for row in reference_rows
            )
            != 1
        ):
            raise Rq13AuditError("RQ13 selected LR is ambiguous")
        selected_curve = run_curves[selected_row["run_name"]]
        first = next(
            (
                point
                for point in selected_curve
                if point["recall@100"] >= reference_row["validation_recall@100"]
            ),
            None,
        )
        if (
            first is None
            or crossing["threshold_recall@100"]
            != reference_row["validation_recall@100"]
            or crossing["first_matching_epoch"] != first["epoch"]
            or crossing["recall@100"] != first["recall@100"]
            or crossing["selected_epoch"] != selected_row["best_epoch"]
            or crossing["selected_recall@100"] != selected_row["validation_recall@100"]
        ):
            raise Rq13AuditError("RQ13 crossing disagrees with selected curves")


def _near_zero(value: object) -> bool:
    return (
        isinstance(value, (int, float)) and math.isfinite(value) and abs(value) <= 1e-6
    )


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def load_correctness_audit(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return _load_json(path)


def current_implementation_sha256() -> dict[str, str]:
    return {str(path): _file_sha256(path) for path in _IMPLEMENTATION_FILES}


def write_correctness_audit(document: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o664)


def _audit_result_surfaces(
    results: Mapping[str, object], logs: Path
) -> dict[str, object]:
    stage = validate_audit_generation_stage(results)
    treatments = results.get("treatments")
    selected = results.get("selected")
    if not selected:
        selected = results.get("surface_winners")
    if not isinstance(treatments, dict) or not isinstance(selected, dict):
        raise Rq13AuditError("RQ13 treatments or selections are absent")
    expected_treatments = {
        "original_surface": _ORIGINAL_TREATMENTS,
        "stage_one": _STAGE_ONE_TREATMENTS,
        "stage_one_bound": _STAGE_ONE_TREATMENTS,
    }.get(stage)
    if expected_treatments is not None and set(selected) != expected_treatments:
        raise Rq13AuditError(f"RQ13 {stage} selections are not exact")
    run_artifacts = {}
    curves = {}
    lr_surfaces = {}
    candidate_counts = []
    for treatment, treatment_record in treatments.items():
        artifacts = (
            treatment_record.get("artifacts")
            if isinstance(treatment_record, dict)
            else None
        )
        if artifacts == []:
            continue
        if not isinstance(artifacts, list):
            raise Rq13AuditError(f"{treatment}: no result artifacts")
        surface = []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise Rq13AuditError(f"{treatment}: malformed result artifact")
            run_name = artifact.get("run_name")
            hashes = artifact.get("artifact_sha256")
            curve = artifact.get("validation_curve")
            if not isinstance(run_name, str) or not isinstance(hashes, dict):
                raise Rq13AuditError(f"{treatment}: artifact identity is absent")
            directory = logs / run_name
            actual_hashes = {
                name: _file_sha256(directory / name)
                for name in (
                    "training_metadata.json",
                    "final_metrics.json",
                    "sweep.log",
                )
            }
            if hashes != actual_hashes:
                raise Rq13AuditError(f"{run_name}: result artifact hash is stale")
            if not isinstance(curve, list) or len(curve) != 20:
                raise Rq13AuditError(f"{run_name}: incomplete validation curve")
            epochs = [point.get("epoch") for point in curve if isinstance(point, dict)]
            if epochs != list(range(1, 21)):
                raise Rq13AuditError(
                    f"{run_name}: validation epochs are not contiguous"
                )
            best = max(
                curve,
                key=lambda point: (
                    point["recall@100"],
                    point["ndcg@100"],
                    -point["epoch"],
                ),
            )
            if best["epoch"] != artifact.get("best_epoch"):
                raise Rq13AuditError(f"{run_name}: selected epoch disagrees with curve")
            run_artifacts[run_name] = hashes
            curves[run_name] = curve
            surface.append(
                {
                    "run_name": run_name,
                    "deep_learning_rate": artifact.get("deep_learning_rate"),
                    "best_epoch": artifact.get("best_epoch"),
                    "validation_recall@100": artifact.get("validation", {}).get(
                        "recall@100"
                    ),
                }
            )
            candidate_counts.append(
                (
                    artifact.get("expanded_examples_per_epoch"),
                    artifact.get("candidate_targets_per_epoch"),
                    artifact.get("ntp_targets_per_epoch"),
                )
            )
        lr_surfaces[treatment] = sorted(
            surface, key=lambda row: row["deep_learning_rate"]
        )
    if expected_treatments is not None and set(lr_surfaces) != expected_treatments:
        raise Rq13AuditError(f"RQ13 {stage} artifact surfaces are not exact")
    selected_crossings = _selected_threshold_crossings(selected, curves)
    for treatment, selection in selected.items():
        if not isinstance(selection, dict):
            raise Rq13AuditError(f"{treatment}: malformed selection")
        rates = [row["deep_learning_rate"] for row in lr_surfaces[treatment]]
        selected_rate = selection.get("deep_learning_rate")
        if selected_rate not in rates or selected_rate in {min(rates), max(rates)}:
            raise Rq13AuditError(f"{treatment}: LR boundary remains unresolved")
    return {
        "run_artifacts": run_artifacts,
        "candidate_counts": {
            "passed": True,
            "all_candidate_targets_equal_examples": all(
                examples == targets for examples, targets, _ in candidate_counts
            ),
            "all_ntp_targets_zero": all(ntp == 0 for _, _, ntp in candidate_counts),
        },
        "learning_curves": {
            "passed": True,
            "selected_threshold_crossings": selected_crossings,
            "runs": curves,
        },
        "lr_boundary": {
            "passed": True,
            "required_followups": [],
            "surfaces": lr_surfaces,
        },
    }


def _selected_threshold_crossings(
    selected: Mapping[str, object], curves: Mapping[str, list[dict[str, object]]]
) -> dict[str, object]:
    result = {}
    for treatment, reference in (
        ("truncated_8", "one_example"),
        ("truncated_16", "truncated_8"),
    ):
        run = selected.get(treatment)
        control = selected.get(reference)
        if not isinstance(run, dict) or not isinstance(control, dict):
            raise Rq13AuditError("selected truncated treatments are incomplete")
        threshold = control.get("validation", {}).get("recall@100")
        curve = curves.get(run.get("run_name"))
        if not isinstance(threshold, (int, float)) or curve is None:
            raise Rq13AuditError("selected threshold evidence is incomplete")
        crossing = next(
            (point for point in curve if point["recall@100"] >= threshold), None
        )
        if crossing is None:
            raise Rq13AuditError(f"{treatment}: never reaches {reference} quality")
        result[f"{treatment}_vs_{reference}"] = {
            "threshold_recall@100": threshold,
            "first_matching_epoch": crossing["epoch"],
            "recall@100": crossing["recall@100"],
            "selected_epoch": run.get("best_epoch"),
            "selected_recall@100": run.get("validation", {}).get("recall@100"),
        }
    return result


def _selected_train_cache(logs: Path, selection: object) -> Path:
    if not isinstance(selection, dict) or not isinstance(
        selection.get("run_name"), str
    ):
        raise Rq13AuditError("selected run identity is absent")
    log = logs / selection["run_name"] / "sweep.log"
    paths = {
        Path(match.group("path"))
        for line in log.read_text().splitlines()
        if (match := _TRAIN_CACHE_PATTERN.search(line)) is not None
    }
    if len(paths) != 1:
        raise Rq13AuditError(
            f"{selection['run_name']}: train cache provenance is incomplete"
        )
    return paths.pop()


def _encoded_history(
    model: CrossAttentionRetrievalModel, batch: dict[str, object]
) -> torch.Tensor:
    tokens = model.tokenizer(batch)
    return model.memory_encoder(
        tokens.embeddings, tokens.cumulative_lens, tokens.timestamps
    )


def _packed_batch(
    sequences: list[list[int]], timestamps: list[list[int]] | None = None
) -> dict[str, object]:
    item_ids = torch.tensor([item for sequence in sequences for item in sequence])
    timestamp_values = torch.tensor(
        [
            timestamp
            for sequence in (
                timestamps
                if timestamps is not None
                else [list(range(len(sequence))) for sequence in sequences]
            )
            for timestamp in sequence
        ],
        dtype=torch.int64,
    )
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.int64)
    cumulative_lens = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    return {
        "int_columns": {
            "compact_item_id": FeatureValues(
                item_ids,
                torch.arange(item_ids.numel() + 1, dtype=torch.int64),
            )
        },
        "float_columns": {},
        "timestamp": timestamp_values,
        "cumulative_lens": cumulative_lens,
    }


def _candidate_loss(query: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    positive_score = (query * positive).sum(dim=1)
    logits = torch.stack([positive_score, torch.zeros_like(positive_score)], dim=1)
    return _MEAN_REDUCTION_PROBE_LOSS.loss_from_logits(logits)


def _gradient_l1(parameters: Any) -> float:
    return float(
        sum(
            parameter.grad.abs().sum()
            for parameter in parameters
            if parameter.grad is not None
        )
    )


def _cache_files_sha256(path: Path) -> dict[str, str]:
    files = [path / "metadata.json", *sorted((path / "buckets").glob("*.parquet"))]
    return {str(file.relative_to(path)): _file_sha256(file) for file in files}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise Rq13AuditError(f"cannot hash {path}") from error
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq13AuditError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq13AuditError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the saved RQ13 correctness audit"
    )
    parser.add_argument("--logs", type=Path, default=Path("generated/logs"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "experiments/g1_sasrec_item_ids_likes/evidence/rq13_prefix_expansion_results.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/g1_sasrec_item_ids_likes/evidence/rq13_prefix_expansion_correctness.json"
        ),
    )
    arguments = parser.parse_args()
    write_correctness_audit(
        build_correctness_audit(arguments.logs, arguments.results), arguments.output
    )


if __name__ == "__main__":
    main()
