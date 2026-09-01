from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import json
import os
from pathlib import Path
import re
from statistics import median
from time import perf_counter
from types import MethodType
from typing import Any

import torch

from dcn.eval.true_metric import evaluate_true_ndcg, prepare_ranking
from dcn.models import BosTokenizer, ItemTokenizer, TimestampDeltaTokenizer
from dcn.config.settings import RuntimeConfig
from experiments.g6_rqkmeans_history.analysis.preflight import (
    maximum_history_batch,
    preflight_contract_metadata,
    record_probe,
    verify_workload_identity,
)
from experiments.g6_rqkmeans_history.configs.rq0 import build_semantic_treatment
from experiments.g6_rqkmeans_history.protocol.manifest import CONTROL_BATCHES
from neuralrec.data.transforms import move_to_device
from neuralrec.run.train import TrainRunner


EVENTS_PER_SEQUENCE = 101
TOKENS_PER_EVENT = 6
VALIDATION_BATCHES = (128, 256, 512, 1024, 2048, 4096, 8192)


def _mode() -> str:
    value = os.environ.get("G6_RQ0_PREFLIGHT_MODE", "")
    if value not in {"training", "validation", "overhead"}:
        raise ValueError("G6_RQ0_PREFLIGHT_MODE is invalid")
    return value


def _training_batch() -> int:
    raw = os.environ.get("G6_RQ0_PREFLIGHT_BATCH", "")
    if not raw.isdigit() or int(raw) not in CONTROL_BATCHES:
        raise ValueError("G6_RQ0_PREFLIGHT_BATCH is not an approved batch")
    return int(raw)


def _attempt() -> str:
    value = os.environ.get("G6_RQ0_PREFLIGHT_ATTEMPT", "")
    if re.fullmatch(r"[a-z0-9_]+", value) is None:
        raise ValueError("G6_RQ0_PREFLIGHT_ATTEMPT is invalid")
    return value


def _evidence_path() -> Path:
    raw = os.environ.get("G6_RQ0_PREFLIGHT_EVIDENCE", "")
    if not raw:
        raise ValueError("G6_RQ0_PREFLIGHT_EVIDENCE is required")
    return Path(raw).resolve()


def _device_evidence(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        raise RuntimeError("G6 RQ0 preflight requires CUDA")
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    if name != "NVIDIA A100-SXM4-80GB" or capability != (8, 0):
        raise RuntimeError("G6 RQ0 preflight requires an A100-SXM4-80GB")
    return {"device_name": name, "compute_capability": list(capability)}


def _measure(
    operation: Callable[[], Any], device: torch.device
) -> tuple[bool, float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    started = perf_counter()
    try:
        operation()
        torch.cuda.synchronize(device)
        fits = True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.synchronize(device)
        fits = False
    elapsed = perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    torch.cuda.empty_cache()
    return fits, elapsed, (peak - baseline) / 1024**3


def _training_probe(owner: Any, runner: TrainRunner) -> dict[str, object]:
    batch = maximum_history_batch(
        batch_size=batch_size,
        events_per_sequence=EVENTS_PER_SEQUENCE,
        item_id_column=owner.item_id_column,
    )
    fits, elapsed, incremental_peak = _measure(
        lambda: runner.train_epoch(0, [batch]), runner.device
    )
    return {
        **_device_evidence(runner.device),
        "batch_size": batch_size,
        "events_per_sequence": EVENTS_PER_SEQUENCE,
        "physical_tokens": batch_size * (EVENTS_PER_SEQUENCE * TOKENS_PER_EVENT + 2),
        "fits": fits,
        "step_seconds": elapsed,
        "incremental_peak_memory_gb": incremental_peak,
    }


def _validation_probe(owner: Any, runner: TrainRunner) -> dict[str, object]:
    results = []
    for candidate in VALIDATION_BATCHES:
        batch = maximum_history_batch(
            batch_size=candidate,
            events_per_sequence=EVENTS_PER_SEQUENCE,
            item_id_column=owner.item_id_column,
        )

        def encode() -> None:
            moved = move_to_device(batch, runner.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                owner.base_model.encode_cutoff_queries(moved)

        fits, elapsed, incremental_peak = _measure(encode, runner.device)
        results.append(
            {
                "batch_size": candidate,
                "physical_tokens": candidate
                * (EVENTS_PER_SEQUENCE * TOKENS_PER_EVENT + 2),
                "fits": fits,
                "seconds": elapsed,
                "incremental_peak_memory_gb": incremental_peak,
            }
        )
        if not fits:
            break
    return {**_device_evidence(runner.device), "probes": results}


def _timed_tokenizer(
    tokenizer: torch.nn.Module, batch: dict[str, Any], device: torch.device
) -> dict[str, float | int]:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            tokenizer(batch)
        torch.cuda.synchronize(device)
        durations = []
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        output = None
        for _ in range(10):
            started = perf_counter()
            output = tokenizer(batch)
            torch.cuda.synchronize(device)
            durations.append(perf_counter() - started)
        assert output is not None
        tokens = output.embeddings.shape[0]
        peak = torch.cuda.max_memory_allocated(device)
    return {
        "median_milliseconds": 1000 * median(durations),
        "incremental_peak_memory_gb": (peak - baseline) / 1024**3,
        "output_tokens": tokens,
    }


def _metric_timing(
    arguments: dict[str, Any], device: torch.device
) -> tuple[dict[str, float], float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    durations = []
    result = {}
    for _ in range(3):
        started = perf_counter()
        result = evaluate_true_ndcg(**arguments)
        torch.cuda.synchronize(device)
        durations.append(perf_counter() - started)
    peak = torch.cuda.max_memory_allocated(device)
    return result, median(durations), (peak - baseline) / 1024**3


def _overhead_probe(owner: Any, runner: TrainRunner) -> dict[str, object]:
    device = runner.device
    batch = move_to_device(
        maximum_history_batch(
            batch_size=128,
            events_per_sequence=EVENTS_PER_SEQUENCE,
            item_id_column=owner.item_id_column,
        ),
        device,
    )
    semantic_tokenizer = owner.base_model.tokenizer
    baseline_tokenizer = owner._with_bos(
        TimestampDeltaTokenizer(
            ItemTokenizer(
                owner.item_embedding,
                item_id_column=owner.item_id_column,
                projection=owner.create_input_projection(),
            ),
            kind="bins",
            combination="add",
            num_bins=32,
        )
    ).to(device)
    baseline_tokens = _timed_tokenizer(baseline_tokenizer, batch, device)
    semantic_tokens = _timed_tokenizer(semantic_tokenizer, batch, device)

    codes = owner.semantic_codes.codes[:, :4].to(device)
    item_count = codes.shape[0]
    user_count = 3414
    generator = torch.Generator(device=device).manual_seed(42)
    item_repr = torch.randn(item_count, 64, generator=generator, device=device)
    query_repr = torch.randn(user_count, 64, generator=generator, device=device)
    item_ids = owner.semantic_codes.item_ids.to(device)
    query_user_ids = torch.arange(user_count, device=device)
    item_id_list = owner.semantic_codes.item_ids.tolist()
    relevance = {
        user_id: {item_id_list[(user_id * 7919) % item_count]}
        for user_id in range(user_count)
    }
    prepared = prepare_ranking(
        query_user_ids,
        item_ids,
        relevance,
        {},
        device=device,
        user_chunk=256,
        exclude_seen=False,
    )
    common = {
        "query_repr": query_repr,
        "query_user_ids": query_user_ids,
        "item_repr": item_repr,
        "item_ids": item_ids,
        "relevance": relevance,
        "train_seen": {},
        "ks": (10, 50, 100),
        "device": device,
        "prepared": prepared,
        "exclude_seen": False,
    }
    baseline_metrics, baseline_seconds, baseline_peak = _metric_timing(common, device)
    semantic_metrics, semantic_seconds, semantic_peak = _metric_timing(
        {**common, "item_semantic_codes": codes}, device
    )
    common_names = set(baseline_metrics)
    metrics_equal = all(
        baseline_metrics[name] == semantic_metrics[name] for name in common_names
    )
    return {
        **_device_evidence(device),
        "tokenization": {
            "batch_size": 128,
            "events_per_sequence": EVENTS_PER_SEQUENCE,
            "baseline": baseline_tokens,
            "semantic": semantic_tokens,
            "latency_ratio": semantic_tokens["median_milliseconds"]
            / baseline_tokens["median_milliseconds"],
        },
        "sid_full_catalog_metrics": {
            "users": user_count,
            "catalog_items": item_count,
            "levels": 4,
            "baseline_seconds": baseline_seconds,
            "semantic_seconds": semantic_seconds,
            "latency_ratio": semantic_seconds / baseline_seconds,
            "baseline_incremental_peak_memory_gb": baseline_peak,
            "semantic_incremental_peak_memory_gb": semantic_peak,
            "base_metrics_equal": metrics_equal,
        },
    }


mode = _mode()
batch_size = _training_batch() if mode == "training" else 128
attempt = _attempt()
evidence_path = _evidence_path()
experiment = build_semantic_treatment(
    "interleaved_item_sid_tokens",
    backbone="best_g1",
    batch_size=batch_size,
    validation_batch_size=8192,
    embedding_learning_rate=0.01,
    deep_learning_rate=0.01,
    num_levels=4,
    num_codes=512,
    representation_width=128,
    run_name=f"g6_rq0_preflight_{attempt}_{mode}_{batch_size}",
)
experiment.runtime = RuntimeConfig(
    dtype=torch.bfloat16,
    compile=False,
    gradient_clip_norm=None,
)


def prebuild_runner_data(owner: Any) -> None:
    if mode == "training":
        _ = owner._offline_item_probabilities


def prebuild_runner_components(owner: Any) -> None:
    verify_workload_identity(
        dataset_cache_key=owner.dataset_key,
        semantic_codes=owner.semantic_codes,
        semantic_codebooks=owner.semantic_codebooks,
    )
    _ = owner.base_model


def create_runner(owner: Any) -> TrainRunner:
    model = owner.create_training_model() if mode == "training" else owner.base_model
    runner = TrainRunner(model=model, optimizer=owner.create_optimizers())

    def run_probe(current: TrainRunner) -> None:
        if mode == "training":
            result = _training_probe(owner, current)
            group, name = "training", str(batch_size)
        elif mode == "validation":
            result = _validation_probe(owner, current)
            group, name = "validation", "worst_layout"
        else:
            result = _overhead_probe(owner, current)
            group, name = "overhead", "worst_layout"
        root = Path(owner.base_path) / "logs" / owner.run_name
        root.mkdir(parents=True, exist_ok=True)
        artifact = root / "preflight.json"
        if artifact.exists():
            raise FileExistsError(f"preflight artifact already exists: {artifact}")
        artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        record_probe(
            evidence_path,
            group,
            name,
            result,
            metadata=preflight_contract_metadata(),
        )

    runner.train = MethodType(run_probe, runner)
    return runner


def finish(owner: Any, runner: TrainRunner) -> None:
    return None


experiment.prebuild_runner_data = MethodType(prebuild_runner_data, experiment)
experiment.prebuild_runner_components = MethodType(
    prebuild_runner_components, experiment
)
experiment.create_runner = MethodType(create_runner, experiment)
experiment.finish = MethodType(finish, experiment)
