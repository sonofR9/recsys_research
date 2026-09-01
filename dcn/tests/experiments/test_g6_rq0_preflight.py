import json

import pytest
import torch

from experiments.g6_rqkmeans_history.analysis.preflight import (
    load_preflight_evidence,
    maximum_history_batch,
    preflight_contract_metadata,
    promote_preflight,
    record_probe,
)
from experiments.g6_rqkmeans_history.protocol.manifest import CONTROL_BATCHES


def test_maximum_history_batch_has_one_dense_item_per_event() -> None:
    batch = maximum_history_batch(
        batch_size=3,
        events_per_sequence=5,
        item_id_column="compact_item_id",
    )

    assert batch["cumulative_lens"].tolist() == [0, 5, 10, 15]
    assert batch["timestamp"].tolist() == list(range(5)) * 3
    items = batch["int_columns"]["compact_item_id"]
    assert items.dense().tolist() == [1] * 15
    assert items.offsets.tolist() == list(range(16))


def test_record_probe_preserves_other_probe_results(tmp_path) -> None:
    destination = tmp_path / "preflight.json"

    record_probe(destination, "training", "128", {"fits": True})
    record_probe(destination, "training", "256", {"fits": False})
    record_probe(destination, "validation", "all", {"safe_batch": 1024})

    assert json.loads(destination.read_text()) == {
        "training": {"128": {"fits": True}, "256": {"fits": False}},
        "validation": {"all": {"safe_batch": 1024}},
    }

    with pytest.raises(ValueError, match="result changed"):
        record_probe(destination, "training", "128", {"fits": False})


def test_maximum_history_batch_rejects_invalid_sizes() -> None:
    for batch_size, events_per_sequence in ((0, 2), (2, 0)):
        try:
            maximum_history_batch(
                batch_size=batch_size,
                events_per_sequence=events_per_sequence,
                item_id_column="item",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid maximum-history shape was accepted")


def test_load_preflight_evidence_returns_only_measured_feasible_batches(
    tmp_path,
) -> None:
    training = {
        str(batch_size): {
            "batch_size": batch_size,
            "fits": batch_size <= 512,
            "device_name": "NVIDIA A100-SXM4-80GB",
            "compute_capability": [8, 0],
            "events_per_sequence": 101,
            "physical_tokens": batch_size * 608,
            "step_seconds": 1.0,
            "incremental_peak_memory_gb": 1.0,
        }
        for batch_size in CONTROL_BATCHES
    }
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            {
                **preflight_contract_metadata(),
                "training": training,
                "validation": {
                    "worst_layout": {
                        "device_name": "NVIDIA A100-SXM4-80GB",
                        "compute_capability": [8, 0],
                        "probes": [
                            {
                                "batch_size": batch_size,
                                "physical_tokens": batch_size * 608,
                                "fits": batch_size <= 1024,
                                "seconds": 1.0,
                                "incremental_peak_memory_gb": 1.0,
                            }
                            for batch_size in (
                                128,
                                256,
                                512,
                                1024,
                                2048,
                                4096,
                                8192,
                            )
                        ],
                    }
                },
                "overhead": {
                    "worst_layout": {
                        "device_name": "NVIDIA A100-SXM4-80GB",
                        "compute_capability": [8, 0],
                        "tokenization": {
                            "batch_size": 128,
                            "events_per_sequence": 101,
                            "baseline": {
                                "output_tokens": 13056,
                                "median_milliseconds": 1.0,
                                "incremental_peak_memory_gb": 1.0,
                            },
                            "semantic": {
                                "output_tokens": 77696,
                                "median_milliseconds": 1.0,
                                "incremental_peak_memory_gb": 1.0,
                            },
                            "latency_ratio": 1.0,
                        },
                        "sid_full_catalog_metrics": {
                            "base_metrics_equal": True,
                            "users": 3414,
                            "catalog_items": 33148,
                            "levels": 4,
                            "baseline_seconds": 1.0,
                            "semantic_seconds": 1.0,
                            "latency_ratio": 1.0,
                            "baseline_incremental_peak_memory_gb": 1.0,
                            "semantic_incremental_peak_memory_gb": 1.0,
                        },
                    }
                },
                "selection": {
                    "feasible_training_batches": [128, 256, 512],
                    "validation_batch_size": 1024,
                },
            }
        )
    )

    evidence = load_preflight_evidence(path)

    assert evidence.feasible_training_batches == (128, 256, 512)
    assert evidence.validation_batch_size == 1024

    promoted = tmp_path / "canonical.json"
    promote_preflight(path, promoted)
    assert promoted.read_bytes() == path.read_bytes()
    promote_preflight(path, promoted)
    promoted.write_text("{}")
    with pytest.raises(FileExistsError, match="already exists"):
        promote_preflight(path, promoted)
