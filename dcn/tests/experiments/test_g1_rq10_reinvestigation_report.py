from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import collect
from experiments.g1_sasrec_item_ids_likes.analysis.rq10_reinvestigation_candidates import (
    Rq10Candidate,
    initial_candidates,
    selected_width_lr_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq10_reinvestigation_report import (
    Rq10ReportError,
    Run,
    build_report_bundle,
    collect_report_bundle,
    render_readme_section,
    write_report_bundle,
)


def _run(
    candidate: Rq10Candidate,
    validation_recall: float,
    validation_ndcg: float,
) -> Run:
    return Run(
        candidate=candidate,
        best_epoch=10,
        stopped_epoch=20,
        validation_recall=validation_recall,
        validation_ndcg=validation_ndcg,
        params_total=10_000_000,
        median_train_epoch_seconds=12.5,
        metrics={
            "recall@100": validation_recall,
            "ndcg@100": validation_ndcg,
            "recall@10": validation_recall / 4,
            "ndcg@10": validation_ndcg / 2,
            "coverage@100": 0.6,
        },
    )


def _resolved_runs() -> list[Run]:
    runs = []
    for candidate in initial_candidates():
        width_bonus = (
            {8: -0.002, 16: 0.001, 32: -0.001}[candidate.feature_width]
            if candidate.family == "gemma_ple"
            else {
                None: 0.0,
                16: 0.0,
                32: 0.001,
                64: -0.001,
            }[candidate.feature_width]
        )
        rate_bonus = 0.002 if candidate.deep_lr == 0.012 else 0.0
        runs.append(_run(candidate, 0.13 + width_bonus + rate_bonus, 0.05))
    for candidate in selected_width_lr_candidates(
        concat_feature_width=32, gemma_feature_width=16
    ):
        runs.append(_run(candidate, 0.129, 0.049))
    return runs


def test_bundle_selects_validation_recall_then_same_epoch_ndcg() -> None:
    runs = _resolved_runs()
    control = [run for run in runs if run.candidate.family == "input_output_only"]
    control[0] = _run(control[0].candidate, 0.14, 0.050)
    control[1] = _run(control[1].candidate, 0.14, 0.051)
    runs = [
        run for run in runs if run.candidate.family != "input_output_only"
    ] + control

    bundle = build_report_bundle(runs)

    selected = bundle.evidence["selected"]["input_output_only"]
    assert selected["deep_lr"] == 0.012
    assert bundle.evidence["claims_status"] == "ready"
    assert "Earlier valid two-layer comparison" in bundle.reader_markdown
    assert "Four-layer reinvestigation" in bundle.reader_markdown


def test_boundary_winner_keeps_claims_pending() -> None:
    runs = _resolved_runs()
    runs = [
        (
            _run(run.candidate, run.candidate.deep_lr, 0.05)
            if run.candidate.family == "direct_add"
            else run
        )
        for run in runs
    ]

    bundle = build_report_bundle(runs)

    assert bundle.evidence["claims_status"] == "pending"
    assert any(
        "direct_add" in name and "d0p048" in name
        for name in bundle.evidence["required_followups"]
    )
    assert (
        "Direct full-width addition |"
        not in bundle.reader_markdown.split("### Four-layer reinvestigation", 1)[1]
    )


def test_acceptance_needs_one_non_inferior_added_feature_not_every_family() -> None:
    runs = _resolved_runs()
    adjusted = []
    for run in runs:
        if run.candidate.family in {"direct_add", "concat_residual"}:
            adjusted.append(
                Run(
                    **{
                        **run.__dict__,
                        "metrics": {
                            **run.metrics,
                            "recall@100": 0.10,
                            "ndcg@100": 0.04,
                        },
                    }
                )
            )
        else:
            adjusted.append(run)

    bundle = build_report_bundle(adjusted)

    assert bundle.evidence["acceptance_status"] == "accepted"
    assert bundle.evidence["selected_added_feature"]["family"] == "gemma_ple"


def test_ready_report_handles_no_accepted_added_feature() -> None:
    runs = _resolved_runs()
    adjusted = []
    for run in runs:
        if run.candidate.family == "input_output_only":
            adjusted.append(run)
        else:
            adjusted.append(
                Run(
                    **{
                        **run.__dict__,
                        "metrics": {
                            **run.metrics,
                            "recall@100": 0.10,
                            "ndcg@100": 0.04,
                        },
                    }
                )
            )

    bundle = build_report_bundle(adjusted)
    report = render_readme_section(bundle)

    assert bundle.evidence["acceptance_status"] == "not_met"
    assert bundle.evidence["selected_added_feature"] is None
    assert "Keep the input/output-only control" in report


def test_reader_reports_costs_and_formats_historical_context() -> None:
    bundle = build_report_bundle(_resolved_runs())

    assert "| parameters (M) | median epoch (s) |" in bundle.reader_markdown
    assert "Method details:" not in bundle.reader_markdown
    assert "Method details:" in render_readme_section(bundle)
    historical = bundle.reader_markdown.split("### Four-layer reinvestigation", 1)[0]
    assert "| runs |" not in historical
    assert "0.140 | 0.053" in historical


def _write_artifact(
    logs: Path,
    candidate: Rq10Candidate,
    *,
    horizon: bool,
    attention_window: int = 50,
) -> None:
    directory = logs / candidate.run_name
    directory.mkdir(parents=True)
    metadata = {
        "training_semantics_revision": 2,
        "dataset_size": "500m",
        "seed": 42,
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 20,
        "best_epoch": 2,
        "stopped_epoch": 20,
        "early_stopped": False,
        "lr_horizon_complete": horizon,
        "selection_resolved": horizon,
        "batch_size": 1280,
        "effective_batch_size": 1280,
        "model_dim": 64,
        "item_embedding_dim": 64,
        "initializer_std": 0.02,
        "embedding_learning_rate": 0.064,
        "deep_learning_rate": candidate.deep_lr,
        "weight_decay": 0.0,
        "gradient_clip_norm": None,
        "runtime_dtype": "torch.bfloat16",
        "negative_sampling": "random",
        "lr_schedule_horizon_epochs": 20,
        "transfer_invariants": {
            "experiment_class": "MuTransferGenerationExperiment",
            "dataset_size": "500m",
            "batch_size": 1280,
            "physical_batch_size": 1280,
            "effective_batch_size": 1280,
            "gradient_accumulation_steps": 1,
            "model_dim": 64,
            "item_embedding_dim": 64,
            "per_layer_item_features": (
                "none" if candidate.family == "input_output_only" else candidate.family
            ),
            "per_layer_item_feature_dim": (
                candidate.feature_width
                if candidate.family in {"concat_residual", "gemma_ple"}
                else None
            ),
            "negative_sampling": "random",
            "num_in_batch_negatives": 512,
            "dense_random_negative_scores": True,
            "mask_false_negatives": False,
            "exclude_own_group_negatives": False,
            "correct_positive_logq": False,
            "random_negative_fraction": 0.5,
            "logq_alpha": 0.01,
            "logq_correction": "yi2019",
            "max_seq_len": 128,
            "bos": False,
            "cls_token": False,
            "cls_token_mode": "none",
            "timestamp_delta": "bins",
            "timestamp_combination": "add",
            "timestamp_num_bins": 16,
            "timestamp_bin_semantics_revision": 2,
            "event_type_filter": "like",
            "window": "next_item",
            "drop_unmapped_items": True,
            "min_item_interactions_per_item": 5,
            "day_range": {"start_day": 0, "end_day": 300},
            "validation_interval_seconds": 604800,
            "user_sample": None,
            "evaluation_catalog": "all",
            "exclude_seen_from_evaluation": False,
            "eval_ks": [10, 50, 100],
            "eval_max_users": 20000,
            "eval_every_n_epochs": 1,
            "restore_best_weights": True,
            "selection_k": 100,
            "early_stopping_metric": "recall@100",
            "early_stopping_metric_prefix": "epoch/val_true",
            "mup_base_dim": 16,
            "mup_delta_dim": 32,
            "mup_base_ffn_dim": None,
            "mup_delta_ffn_dim": None,
            "transformer": {
                "alibi": False,
                "attention_window": attention_window,
                "dim": 64,
                "dropout": 0.1,
                "ffn": "swiglu",
                "ffn_dropout": 0.1,
                "ffn_intermediate_dim": 171,
                "final_norm": "layer",
                "gated_ffn_dropout": False,
                "input_dropout": 0.1,
                "input_norm": None,
                "learned_positions": "forward",
                "nhead": 2,
                "norm": "layer",
                "norm_place": "pre",
                "num_kv_heads": 1,
                "num_layers": 4,
                "rope": None,
            },
            "lr_schedule": {
                "cycles": 1,
                "min_lr_fraction": 0.0,
                "optimizer_group_scope": "both",
                "power_exponent": -0.51,
                "power_transition_tokens": None,
                "shape": "linear",
                "timescale_fraction": None,
                "timescale_steps": None,
                "warmup_fraction": 0.0,
            },
            "lr_schedule_horizon_epochs": 20,
        },
    }
    metrics = {
        "recall@100": 0.13,
        "ndcg@100": 0.05,
        "recall@10": 0.03,
        "ndcg@10": 0.02,
        "coverage@100": 0.6,
        "num_users": 37018,
    }
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text(json.dumps(metrics))
    (directory / "sweep.log").write_text(
        "".join(
            f"epoch {epoch} finished epoch/val_true.recall@100=0.131 "
            "epoch/val_true.ndcg@100=0.051 timing.train_epoch_time=12.5 "
            "resources.params_total=10000000.0000\n"
            for epoch in range(20)
        )
    )


def test_collector_rejects_a_horizon_incomplete_artifact(tmp_path: Path) -> None:
    _write_artifact(tmp_path, initial_candidates()[0], horizon=False)

    with pytest.raises(Rq10ReportError, match="horizon-complete"):
        collect_report_bundle(tmp_path)


def test_collector_rejects_a_confounded_frozen_recipe(tmp_path: Path) -> None:
    _write_artifact(
        tmp_path,
        initial_candidates()[0],
        horizon=True,
        attention_window=49,
    )

    with pytest.raises(Rq10ReportError, match="transfer_invariants.transformer"):
        collect_report_bundle(tmp_path)


def test_writer_emits_tuning_reader_and_evidence(tmp_path: Path) -> None:
    bundle = build_report_bundle(_resolved_runs())

    paths = write_report_bundle(bundle, tmp_path / "scratchpad", tmp_path / "evidence")

    assert set(paths) == {"tuning", "reader", "evidence"}
    assert all(path.is_file() for path in paths.values())
    assert json.loads(paths["evidence"].read_text())["claims_status"] == "ready"


def test_collect_uses_dedicated_rq10_reader_instead_of_manifest_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dedicated = "### Earlier valid two-layer comparison\n\n| dedicated |\n| --- |"
    monkeypatch.setattr(
        collect,
        "_load_rq10_report_bundle",
        lambda: SimpleNamespace(reader_markdown=dedicated),
    )
    monkeypatch.setattr(
        collect,
        "select_architecture_report_runs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("obsolete")),
    )

    section = collect._render_current_component_question("500m", 10, [], [])

    assert dedicated in section


def test_focused_writer_updates_scratchpad_and_readme_without_losing_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = build_report_bundle(_resolved_runs())
    scratchpad = tmp_path / "scratchpad"
    scratchpad.mkdir()
    report = (
        "# report\n\n## RQ9 — keep\n\n| old9 |\n\n"
        "## RQ10 — stale\n\n| stale |\n\n"
        "## RQ11 — keep\n\n| old11 |\n"
    )
    (scratchpad / "research_questions_500m.md").write_text(report)
    readme = tmp_path / "README.md"
    readme.write_text(report)
    monkeypatch.setattr(collect, "_load_rq10_report_bundle", lambda: bundle)
    monkeypatch.setattr(collect, "READER_REPORT", readme)
    monkeypatch.setattr(collect, "EXPERIMENT", tmp_path)

    collect.write_rq10_reports(scratchpad)

    assert "Earlier valid two-layer comparison" in readme.read_text()
    assert "Conclusion: Select" in readme.read_text()
    assert "## RQ9 — keep" in readme.read_text()
    assert "## RQ11 — keep" in readme.read_text()
