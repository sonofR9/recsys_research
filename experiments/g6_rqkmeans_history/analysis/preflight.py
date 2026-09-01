from __future__ import annotations

import json
from dataclasses import dataclass
import hashlib
import inspect
import math
from pathlib import Path
from typing import Any

import mup
import torch

from dcn.data.features import FeatureValues
from dcn.semantic import ResidualCodebooks, SemanticCodes
from experiments.g6_rqkmeans_history.protocol.manifest import CONTROL_BATCHES
from utils.locks import hold

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_EVIDENCE_PATH = (
    _PROJECT_ROOT / "experiments/g6_rqkmeans_history/evidence/rq0_preflight_v5.json"
)
_DEVICE_NAME = "NVIDIA A100-SXM4-80GB"
_VALIDATION_BATCHES = (128, 256, 512, 1024, 2048, 4096, 8192)
_IMPLEMENTATION_FILES = (
    "dcn/config/generation.py",
    "dcn/config/semantic_history.py",
    "dcn/eval/true_metric.py",
    "dcn/models/history_tokens.py",
    "dcn/models/sequence_retrieval.py",
    "dcn/nn/densenet.py",
    "dcn/nn/sampled_softmax.py",
    "dcn/nn/semantic_embedding.py",
    "dcn/nn/transformer.py",
    "experiments/g6_rqkmeans_history/analysis/preflight.py",
    "experiments/g6_rqkmeans_history/configs/rq0.py",
    "experiments/g6_rqkmeans_history/launchers/run_preflight.py",
)
_WORKLOAD = {
    "dataset_size": "native-50m",
    "dataset_cache_key": "0fb0a01c70e1",
    "catalog_items": 33148,
    "catalog_item_ids_sha256": (
        "aef5e631abfeb0f952d0a91ae19e5aa0364524cf605b806b13c86b460966af40"
    ),
    "backbone": "best_g1",
    "representation": "interleaved_item_sid_tokens",
    "representation_width": 128,
    "base_levels": 4,
    "codes_per_base_level": 512,
    "collision_suffix_levels": 1,
    "semantic_codes_sha256": (
        "8f0744c3a462167dbab8d03d76f1b35945dc54258ea4a3f7bd01b30d7f764b70"
    ),
    "semantic_codebooks_sha256": (
        "1612834ba127fc7981c7d0b94ea0047aeceb8ffb4e06157f920a98728b8613f0"
    ),
    "tokens_per_event": 6,
    "events_per_sequence": 101,
    "bos_tokens": 1,
    "end_cls_tokens": 1,
    "model_dim": 64,
    "transformer_layers": 4,
    "ffn": "swiglu",
    "ffn_width": 192,
    "negative_count": 2048,
    "dtype": "bfloat16",
    "compile": False,
    "optimizer": "MuAdam",
    "torch_version": torch.__version__,
    "embedding_learning_rate": 0.01,
    "deep_learning_rate": 0.01,
}


@dataclass(frozen=True)
class PreflightEvidence:
    feasible_training_batches: tuple[int, ...]
    validation_batch_size: int


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative_path in _IMPLEMENTATION_FILES:
        digest.update(relative_path.encode())
        digest.update((_PROJECT_ROOT / relative_path).read_bytes())
    optimizer_path = inspect.getsourcefile(mup.MuAdam)
    if optimizer_path is None:
        raise RuntimeError("cannot identify MuAdam implementation")
    digest.update("mup.MuAdam".encode())
    digest.update(Path(optimizer_path).read_bytes())
    return digest.hexdigest()


def preflight_contract_metadata() -> dict[str, object]:
    return {
        "schema_version": 2,
        "workload": _WORKLOAD,
        "implementation_sha256": _implementation_sha256(),
    }


def _nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            digest.update(key.encode())
            _update_digest(digest, value[key])
    else:
        digest.update(repr(value).encode())


def _content_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def verify_workload_identity(
    *,
    dataset_cache_key: str,
    semantic_codes: SemanticCodes,
    semantic_codebooks: ResidualCodebooks,
) -> None:
    observed = {
        "dataset_cache_key": dataset_cache_key,
        "catalog_items": len(semantic_codes.item_ids),
        "catalog_item_ids_sha256": _content_sha256(semantic_codes.item_ids),
        "semantic_codes_sha256": _content_sha256(
            {
                "item_ids": semantic_codes.item_ids,
                "codes": semantic_codes.codes,
                "codes_per_level": list(semantic_codes.codes_per_level),
            }
        ),
        "semantic_codebooks_sha256": _content_sha256(semantic_codebooks.centroids),
    }
    if any(_WORKLOAD[name] != value for name, value in observed.items()):
        raise ValueError("G6 RQ0 runtime workload identity changed")


def load_preflight_evidence(
    path: Path = PREFLIGHT_EVIDENCE_PATH,
) -> PreflightEvidence:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read G6 RQ0 preflight evidence {path}") from error
    for name, value in preflight_contract_metadata().items():
        if document.get(name) != value:
            raise ValueError(f"G6 RQ0 preflight {name} changed")

    training = document.get("training")
    if not isinstance(training, dict) or set(training) != {
        str(batch_size) for batch_size in CONTROL_BATCHES
    }:
        raise ValueError("G6 RQ0 preflight requires every approved training batch")
    measured_feasible = []
    for batch_size in CONTROL_BATCHES:
        row = training[str(batch_size)]
        if not isinstance(row, dict):
            raise ValueError("G6 RQ0 training probe is invalid")
        expected = {
            "batch_size": batch_size,
            "device_name": _DEVICE_NAME,
            "compute_capability": [8, 0],
            "events_per_sequence": 101,
            "physical_tokens": batch_size * 608,
        }
        if any(row.get(name) != value for name, value in expected.items()):
            raise ValueError("G6 RQ0 training probe contract changed")
        if not isinstance(row.get("fits"), bool):
            raise ValueError("G6 RQ0 training fit result is invalid")
        for metric in ("step_seconds", "incremental_peak_memory_gb"):
            if not isinstance(row.get(metric), (int, float)) or row[metric] < 0:
                raise ValueError("G6 RQ0 training probe measurement is invalid")
        if row["fits"]:
            measured_feasible.append(batch_size)

    selection = document.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("G6 RQ0 preflight selection is absent")
    feasible = selection.get("feasible_training_batches")
    if feasible != measured_feasible or not feasible:
        raise ValueError("G6 RQ0 feasible training batches do not match probes")

    validation = document.get("validation", {}).get("worst_layout", {})
    if validation.get("device_name") != _DEVICE_NAME or validation.get(
        "compute_capability"
    ) != [8, 0]:
        raise ValueError("G6 RQ0 validation probe device changed")
    validation_rows = validation.get("probes")
    if not isinstance(validation_rows, list) or not validation_rows:
        raise ValueError("G6 RQ0 validation probes are incomplete")
    measured_validation_batches = [
        row.get("batch_size") for row in validation_rows if isinstance(row, dict)
    ]
    expected_validation_batches = list(
        _VALIDATION_BATCHES[: len(measured_validation_batches)]
    )
    if measured_validation_batches != expected_validation_batches or (
        len(validation_rows) < len(_VALIDATION_BATCHES)
        and validation_rows[-1].get("fits") is not False
    ):
        raise ValueError("G6 RQ0 validation probes are incomplete")
    for row, batch_size in zip(
        validation_rows, expected_validation_batches, strict=True
    ):
        if (
            not isinstance(row, dict)
            or row.get("batch_size") != batch_size
            or row.get("physical_tokens") != batch_size * 608
            or not isinstance(row.get("fits"), bool)
        ):
            raise ValueError("G6 RQ0 validation probe contract changed")
        for metric in ("seconds", "incremental_peak_memory_gb"):
            if not isinstance(row.get(metric), (int, float)) or row[metric] < 0:
                raise ValueError("G6 RQ0 validation probe measurement is invalid")
    selected_validation_batch = selection.get("validation_batch_size")
    passing_validation = [row["batch_size"] for row in validation_rows if row["fits"]]
    if not passing_validation or selected_validation_batch != max(passing_validation):
        raise ValueError("G6 RQ0 validation selection does not match probes")
    overhead = document.get("overhead", {}).get("worst_layout", {})
    if overhead.get("device_name") != _DEVICE_NAME or overhead.get(
        "compute_capability"
    ) != [8, 0]:
        raise ValueError("G6 RQ0 overhead probe device changed")
    tokenization = overhead.get("tokenization", {})
    if (
        tokenization.get("batch_size") != 128
        or tokenization.get("events_per_sequence") != 101
        or tokenization.get("baseline", {}).get("output_tokens") != 13056
        or tokenization.get("semantic", {}).get("output_tokens") != 77696
    ):
        raise ValueError("G6 RQ0 tokenization probe contract changed")
    for row in (tokenization.get("baseline", {}), tokenization.get("semantic", {})):
        for metric_name in ("median_milliseconds", "incremental_peak_memory_gb"):
            if not _nonnegative_number(row.get(metric_name)):
                raise ValueError("G6 RQ0 tokenization measurement is invalid")
    if not _nonnegative_number(tokenization.get("latency_ratio")):
        raise ValueError("G6 RQ0 tokenization ratio is invalid")

    metric = overhead.get("sid_full_catalog_metrics", {})
    if (
        metric.get("base_metrics_equal") is not True
        or metric.get("users") != 3414
        or metric.get("catalog_items") != 33148
        or metric.get("levels") != 4
    ):
        raise ValueError("G6 RQ0 SID metric probe changed base metrics")
    for metric_name in (
        "baseline_seconds",
        "semantic_seconds",
        "latency_ratio",
        "baseline_incremental_peak_memory_gb",
        "semantic_incremental_peak_memory_gb",
    ):
        if not _nonnegative_number(metric.get(metric_name)):
            raise ValueError("G6 RQ0 SID metric measurement is invalid")
    return PreflightEvidence(tuple(feasible), selected_validation_batch)


def promote_preflight(
    source: Path,
    destination: Path = PREFLIGHT_EVIDENCE_PATH,
) -> None:
    load_preflight_evidence(source)
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with hold(destination.with_suffix(".lock"), "G6 RQ0 preflight promotion"):
        if destination.exists():
            if destination.read_bytes() == payload:
                return
            raise FileExistsError(f"preflight evidence already exists: {destination}")
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    load_preflight_evidence(destination)


def maximum_history_batch(
    *,
    batch_size: int,
    events_per_sequence: int,
    item_id_column: str,
) -> dict[str, Any]:
    if batch_size < 1 or events_per_sequence < 1:
        raise ValueError("batch and sequence sizes must be positive")
    total_events = batch_size * events_per_sequence
    item_ids = torch.ones(total_events, dtype=torch.int64)
    return {
        "int_columns": {
            item_id_column: FeatureValues(
                item_ids,
                torch.arange(total_events + 1, dtype=torch.int64),
            )
        },
        "float_columns": {},
        "timestamp": torch.arange(events_per_sequence, dtype=torch.int64).repeat(
            batch_size
        ),
        "cumulative_lens": torch.arange(
            0,
            total_events + 1,
            events_per_sequence,
            dtype=torch.int64,
        ),
    }


def record_probe(
    destination: Path,
    group: str,
    name: str,
    result: dict[str, object],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with hold(destination.with_suffix(".lock"), "G6 RQ0 preflight evidence"):
        document = json.loads(destination.read_text()) if destination.exists() else {}
        if metadata is not None:
            for key, value in metadata.items():
                if key in document and document[key] != value:
                    raise ValueError(f"G6 RQ0 preflight {key} changed")
                document[key] = value
        probes = document.setdefault(group, {})
        if name in probes and probes[name] != result:
            raise ValueError(f"G6 RQ0 preflight {group}/{name} result changed")
        probes[name] = result
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)


def finalize_preflight(destination: Path = PREFLIGHT_EVIDENCE_PATH) -> None:
    with hold(destination.with_suffix(".lock"), "G6 RQ0 preflight evidence"):
        document = json.loads(destination.read_text())
        training = document.get("training", {})
        feasible = [
            batch_size
            for batch_size in CONTROL_BATCHES
            if training.get(str(batch_size), {}).get("fits") is True
        ]
        validation = document.get("validation", {}).get("worst_layout", {})
        passing_validation = [
            row["batch_size"]
            for row in validation.get("probes", [])
            if row.get("fits") is True
        ]
        if not feasible or not passing_validation:
            raise ValueError("G6 RQ0 preflight has no feasible batch")
        selection = {
            "feasible_training_batches": feasible,
            "validation_batch_size": max(passing_validation),
        }
        if "selection" in document and document["selection"] != selection:
            raise ValueError("G6 RQ0 preflight selection changed")
        document["selection"] = selection
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)
    load_preflight_evidence(destination)
