from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import (
    rq8_reinvestigation_candidates,
    rq8_reinvestigation_report,
)


METRICS = {
    "recall@100": 0.130,
    "ndcg@100": 0.050,
    "recall@10": 0.027,
    "ndcg@10": 0.021,
    "coverage@100": 0.600,
    "num_users": 37018,
}


def _transformer(candidate: rq8_reinvestigation_candidates.Rq8Candidate) -> dict:
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


def _write_artifact(
    logs: Path,
    candidate: rq8_reinvestigation_candidates.Rq8Candidate,
    *,
    validation_recall: float,
    validation_ndcg: float | None = None,
    final_recall: float | None = None,
) -> None:
    directory = logs / candidate.run_name
    directory.mkdir(parents=True, exist_ok=True)
    cls_mode = (
        candidate.query_method
        if candidate.study == "query" and candidate.query_method != "standard"
        else "none"
    )
    invariants = {
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
        "transformer": _transformer(candidate),
        "lr_schedule": {
            "shape": "linear",
            "warmup_fraction": 0.0,
            "min_lr_fraction": 0.0,
            "cycles": 1,
            "timescale_steps": None,
            "timescale_fraction": None,
            "power_exponent": -0.51,
            "power_transition_tokens": None,
            "optimizer_group_scope": "both",
        },
    }
    metadata = {
        "training_semantics_revision": 2,
        "dataset_size": "500m",
        "seed": candidate.seed,
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 20,
        "best_epoch": 2,
        "stopped_epoch": 20,
        "early_stopped": False,
        "best_epoch_at_cap": False,
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
        "transfer_invariants": invariants,
    }
    ndcg = validation_recall / 2 if validation_ndcg is None else validation_ndcg
    metrics = dict(METRICS)
    metrics["recall@100"] = (
        validation_recall if final_recall is None else final_recall
    )
    metrics["ndcg@100"] = ndcg
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text(json.dumps(metrics))
    (directory / "sweep.log").write_text(
        "epoch 0 finished epoch/val_true.recall@100=0.100000 "
        "epoch/val_true.ndcg@100=0.040000\n"
        f"epoch 1 finished epoch/val_true.recall@100={validation_recall:.6f} "
        f"epoch/val_true.ndcg@100={ndcg:.6f}\n"
    )


def _query_candidates() -> tuple[rq8_reinvestigation_candidates.Rq8Candidate, ...]:
    return tuple(
        candidate
        for candidate in rq8_reinvestigation_candidates.initial_candidates()
        if candidate.study == "query"
    )


def _sequence_candidates() -> tuple[rq8_reinvestigation_candidates.Rq8Candidate, ...]:
    return tuple(
        candidate
        for candidate in rq8_reinvestigation_candidates.initial_candidates()
        if candidate.study == "sequence"
    )


def _complete_initial(logs: Path, *, boundary_standard: bool = False) -> None:
    for candidate in rq8_reinvestigation_candidates.initial_candidates():
        score = 0.130 - abs(candidate.deep_lr - 0.012)
        if (
            boundary_standard
            and candidate.study == "query"
            and candidate.query_method == "standard"
        ):
            score = candidate.deep_lr
        final_recall = (
            0.120 + candidate.max_seq_len / 100000
            if candidate.study == "sequence"
            else None
        )
        _write_artifact(
            logs,
            candidate,
            validation_recall=score,
            final_recall=final_recall,
        )


def _write_query_confirmations(logs: Path) -> None:
    for method in ("standard", "end_only", "interleaved"):
        winner = next(
            candidate
            for candidate in _query_candidates()
            if candidate.query_method == method
            and candidate.deep_lr == 0.012
        )
        for seed in (43, 44):
            _write_artifact(
                logs,
                rq8_reinvestigation_candidates.make_confirmation_candidate(
                    winner, seed
                ),
                validation_recall=0.130 + (seed - 42) / 1000,
                final_recall=0.130 + (seed - 42) / 1000,
            )


def test_report_fails_closed_on_an_incomplete_initial_surface(tmp_path: Path) -> None:
    candidate = _query_candidates()[0]
    _write_artifact(tmp_path, candidate, validation_recall=0.13)

    with pytest.raises(
        rq8_reinvestigation_report.Rq8ReportError,
        match="corrected surface is incomplete: 1/57",
    ):
        rq8_reinvestigation_report.collect_report_bundle(tmp_path)


def test_report_requires_the_next_lr_when_a_winner_is_on_a_boundary(
    tmp_path: Path,
) -> None:
    _complete_initial(tmp_path, boundary_standard=True)

    with pytest.raises(
        rq8_reinvestigation_report.Rq8ReportError,
        match=r"query standard.*0\.024.*0\.048",
    ):
        rq8_reinvestigation_report.collect_report_bundle(tmp_path)


def test_boundary_continuation_is_retained_in_the_tuning_evidence(
    tmp_path: Path,
) -> None:
    _complete_initial(tmp_path, boundary_standard=True)
    standard = next(
        candidate
        for candidate in _query_candidates()
        if candidate.query_method == "standard"
        and candidate.deep_lr == 0.024
    )
    continuation = rq8_reinvestigation_candidates.make_boundary_candidate(
        standard, "high", 1
    )
    _write_artifact(tmp_path, continuation, validation_recall=0.020)
    for method in ("standard", "end_only", "interleaved"):
        winner = standard if method == "standard" else next(
            candidate
            for candidate in _query_candidates()
            if candidate.query_method == method
            and candidate.deep_lr == 0.012
        )
        for seed in (43, 44):
            _write_artifact(
                tmp_path,
                rq8_reinvestigation_candidates.make_confirmation_candidate(
                    winner, seed
                ),
                validation_recall=0.13,
            )

    bundle = rq8_reinvestigation_report.collect_report_bundle(tmp_path)

    names = {
        record["run_name"] for record in bundle.evidence["validated_artifacts"]
    }
    assert continuation.run_name in names
    assert "| 42 | 0.064 | 0.048 |" in bundle.tuning_markdown


def test_report_requires_both_frozen_query_confirmation_seeds(tmp_path: Path) -> None:
    _complete_initial(tmp_path)

    with pytest.raises(
        rq8_reinvestigation_report.Rq8ReportError,
        match="query confirmation.*seed 43",
    ):
        rq8_reinvestigation_report.collect_report_bundle(tmp_path)


def test_report_validates_native_500m_protocol_metadata(tmp_path: Path) -> None:
    _complete_initial(tmp_path)
    candidate = _query_candidates()[0]
    metadata_path = tmp_path / candidate.run_name / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dataset_size"] = "50m"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(
        rq8_reinvestigation_report.Rq8ReportError,
        match=r"expected '500m', got '50m'",
    ):
        rq8_reinvestigation_report.collect_report_bundle(tmp_path)


def test_report_requires_full_causal_sequence_metadata(tmp_path: Path) -> None:
    _complete_initial(tmp_path)
    candidate = _sequence_candidates()[0]
    metadata_path = tmp_path / candidate.run_name / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["transfer_invariants"]["transformer"]["attention_window"] = 50
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(
        rq8_reinvestigation_report.Rq8ReportError,
        match=r"attention_window expected None, got 50",
    ):
        rq8_reinvestigation_report.collect_report_bundle(tmp_path)


def test_report_requires_sequence_boundary_continuation(tmp_path: Path) -> None:
    _complete_initial(tmp_path)
    candidates = [
        candidate
        for candidate in _sequence_candidates()
        if candidate.position_method == "alibi" and candidate.max_seq_len == 12
    ]
    for candidate in candidates:
        _write_artifact(tmp_path, candidate, validation_recall=candidate.deep_lr)

    with pytest.raises(
        rq8_reinvestigation_report.Rq8ReportError,
        match=r"sequence alibi length 12.*0\.024.*0\.048.*sequence_fullcausal",
    ):
        rq8_reinvestigation_report.collect_report_bundle(tmp_path)


def test_report_fails_closed_on_an_exact_validation_tie(tmp_path: Path) -> None:
    _complete_initial(tmp_path)
    standard = {
        candidate.deep_lr: candidate
        for candidate in _query_candidates()
        if candidate.query_method == "standard"
    }
    _write_artifact(tmp_path, standard[0.006], validation_recall=0.130)

    with pytest.raises(
        rq8_reinvestigation_report.Rq8ReportError,
        match="exact validation tie",
    ):
        rq8_reinvestigation_report.collect_report_bundle(tmp_path)


@pytest.mark.parametrize(("source", "value"), [("validation", -0.1), ("final", 1.2)])
def test_report_rejects_out_of_range_metrics(
    tmp_path: Path, source: str, value: float
) -> None:
    _complete_initial(tmp_path)
    candidate = _query_candidates()[0]
    if source == "validation":
        _write_artifact(tmp_path, candidate, validation_recall=value)
    else:
        metrics_path = tmp_path / candidate.run_name / "final_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["recall@100"] = value
        metrics_path.write_text(json.dumps(metrics))

    with pytest.raises(rq8_reinvestigation_report.Rq8ReportError):
        rq8_reinvestigation_report.collect_report_bundle(tmp_path)


def test_complete_report_contains_query_and_two_corrected_sequence_tables(
    tmp_path: Path,
) -> None:
    _complete_initial(tmp_path)
    _write_query_confirmations(tmp_path)

    bundle = rq8_reinvestigation_report.collect_report_bundle(tmp_path)

    assert bundle.reader_markdown.count("| query objective |") == 1
    assert bundle.reader_markdown.count("retained history length") == 2
    assert bundle.reader_markdown.count("| 128 |") == 2
    assert len(re.findall(r"(?m)^\| (?:\*\*)?12(?:\*\*)? \|", bundle.reader_markdown)) == 2
    assert bundle.reader_markdown.count("| **512** |") == 2
    assert "### Sequence: causal ALiBi, length 12" in bundle.tuning_markdown
    assert "### Sequence: reverse-RoPE + ALiBi, length 512" in bundle.tuning_markdown
    assert "end-only CLS" in bundle.reader_markdown
    assert "interleaved CLS" in bundle.reader_markdown
    assert "| standard item-state |" in bundle.reader_markdown
    assert "| **end-only CLS** |" in bundle.reader_markdown
    assert "FFN" not in bundle.reader_markdown
    assert "deep LR" not in bundle.reader_markdown
    assert "g1_rq8_" not in bundle.reader_markdown
    assert "50M" not in bundle.reader_markdown
    assert bundle.evidence["dataset_size"] == "500m"
    assert bundle.evidence["initial_query_surface_runs"] == 9
    assert bundle.evidence["initial_sequence_surface_runs"] == 48
    assert bundle.evidence["query_confirmation_seeds"] == [42, 43, 44]
    assert len(bundle.evidence["sequence_results"]) == 16
    assert bundle.evidence["protocol"]["sequence_attention_window"] is None
    assert bundle.evidence["protocol"]["sequence_protocol_revision"] == 2
    assert all(
        "sequence_fullcausal" in result["artifact"]
        for result in bundle.evidence["sequence_results"]
    )
    assert bundle.tuning_markdown.count("### Query:") == 3
    assert bundle.tuning_markdown.count("### Sequence:") == 16

    alibi = bundle.reader_markdown.index("| causal ALiBi retained history length |")
    reverse = bundle.reader_markdown.index(
        "| reverse-RoPE + ALiBi retained history length |"
    )
    assert alibi < reverse
    alibi_table = bundle.reader_markdown[alibi:reverse]
    rows = [
        int(line.split("|")[1].strip().strip("*"))
        for line in alibi_table.splitlines()
        if re.match(r"^\| \*?\*?\d+", line)
    ]
    assert rows == [12, 25, 50, 100, 128, 200, 256, 512]
    reference_row = next(
        line for line in alibi_table.splitlines() if line.startswith("| 128 |")
    )
    assert "0.121" in reference_row
    assert "(0.121)" not in reference_row


def test_written_report_bundle_contains_only_corrected_sequence_results(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _complete_initial(logs)
    _write_query_confirmations(logs)

    paths = rq8_reinvestigation_report.write_report_bundle(
        rq8_reinvestigation_report.collect_report_bundle(logs),
        tmp_path / "scratchpad",
        tmp_path / "evidence",
    )

    assert set(paths) == {"tuning", "reader", "evidence"}
    for path in paths.values():
        content = path.read_text()
        assert re.search(r"g1_rq8_sequence_(?!fullcausal)", content) is None
    assert "sequence_results" in paths["evidence"].read_text()
    assert paths["reader"].read_text().count("retained history length") == 2


def test_report_does_not_read_legacy_fixed_window_artifacts(tmp_path: Path) -> None:
    _complete_initial(tmp_path)
    _write_query_confirmations(tmp_path)
    legacy = tmp_path / "g1_rq8_sequence_alibi_s512_legacy_r1_500m"
    legacy.mkdir()
    (legacy / "training_metadata.json").write_text("{}")
    (legacy / "final_metrics.json").write_text(
        json.dumps({"recall@100": 0.999, "ndcg@100": 0.999})
    )
    (legacy / "sweep.log").write_text("recall@100=0.999")

    bundle = rq8_reinvestigation_report.collect_report_bundle(tmp_path)

    serialized = json.dumps(bundle.evidence)
    assert "0.999" not in bundle.reader_markdown
    assert "0.999" not in serialized
    assert "g1_rq8_sequence_alibi" not in serialized


def test_query_rate_selection_uses_validation_but_renders_full_user_metrics(
    tmp_path: Path,
) -> None:
    _complete_initial(tmp_path)
    standard = {
        candidate.deep_lr: candidate
        for candidate in _query_candidates()
        if candidate.query_method == "standard"
    }
    _write_artifact(
        tmp_path,
        standard[0.006],
        validation_recall=0.131,
        final_recall=0.190,
    )
    _write_artifact(
        tmp_path,
        standard[0.012],
        validation_recall=0.140,
        final_recall=0.150,
    )
    for method in ("standard", "end_only", "interleaved"):
        winner = standard[0.012] if method == "standard" else next(
            candidate
            for candidate in _query_candidates()
            if candidate.query_method == method
            and candidate.deep_lr == 0.012
        )
        for seed in (43, 44):
            _write_artifact(
                tmp_path,
                rq8_reinvestigation_candidates.make_confirmation_candidate(
                    winner, seed
                ),
                validation_recall=0.14,
                final_recall=0.15,
            )

    bundle = rq8_reinvestigation_report.collect_report_bundle(tmp_path)

    standard_row = next(
        line
        for line in bundle.reader_markdown.splitlines()
        if line.startswith("| standard item-state |")
    )
    assert "0.150" in standard_row
    assert "0.190" not in standard_row
