import json
from pathlib import Path

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import collect, rq4_activation_depth


def _write_artifact(
    generated: Path,
    spec: rq4_activation_depth.RunSpec,
    recall: float,
    ndcg: float,
    validation_recall: float | None = None,
) -> None:
    directory = generated / "logs" / spec.name
    directory.mkdir(parents=True, exist_ok=True)
    transformer = {
        "ffn": spec.family.lower(),
        "ffn_intermediate_dim": spec.width,
        "ffn_dropout": 0.1,
        "num_layers": spec.layers,
    }
    if spec.gated:
        transformer["gated_ffn_dropout"] = True
    (directory / "training_metadata.json").write_text(
        json.dumps(
            {
                "embedding_learning_rate": 0.064,
                "deep_learning_rate": spec.deep_lr,
                "batch_size": 1280,
                "best_epoch": 12,
                "stopped_epoch": 20,
                "lr_schedule_horizon_epochs": 20,
                "selection_resolved": True,
                "transfer_invariants": {
                    "item_embedding_dim": 64,
                    "model_dim": 64,
                    "mup_base_dim": 16,
                    "mup_delta_dim": 32,
                    "restore_best_weights": True,
                    "transformer": transformer,
                },
            }
        )
    )
    (directory / "final_metrics.json").write_text(
        json.dumps(
            {
                "recall@100": recall,
                "ndcg@100": ndcg,
                "num_users": 37018,
            }
        )
    )
    selected_validation_recall = (
        recall if validation_recall is None else validation_recall
    )
    (directory / "sweep.log").write_text(
        "epoch 0 finished epoch/val_true.recall@100="
        f"{selected_validation_recall - 0.01:.6f}\n"
        "epoch 1 finished epoch/val_true.recall@100="
        f"{selected_validation_recall:.6f}\n"
    )


def _complete_surface(generated: Path) -> None:
    family_offset = {
        "ReLU": 0.000,
        "GELU": 0.002,
        "SiLU": 0.001,
        "ReGLU": 0.004,
        "GEGLU": 0.005,
        "SwiGLU": 0.006,
    }
    for spec in rq4_activation_depth.expected_specs():
        optimum = 0.024 if spec.family in {"ReLU", "SiLU"} else 0.012
        score = 0.13 + family_offset[spec.family] + spec.layers / 1000
        score -= abs(spec.deep_lr - optimum)
        _write_artifact(generated, spec, score, score / 2)


def test_loader_pins_exact_compatible_surface_and_ignores_historical_glob(
    tmp_path: Path,
) -> None:
    _complete_surface(tmp_path)
    incompatible = (
        tmp_path
        / "logs/g1_rqtune_rqfinal_ffn_gelu171_e0p128_d0p012_b1280_cap40_ts2_r3_500m"
    )
    incompatible.mkdir(parents=True)
    (incompatible / "training_metadata.json").write_text("{}")
    (incompatible / "final_metrics.json").write_text(
        json.dumps({"recall@100": 1.0, "ndcg@100": 1.0})
    )

    runs = rq4_activation_depth.load_runs(tmp_path)

    assert len(runs) == 42
    assert all("e0p128" not in run.name for run in runs)
    reused = [run.name for run in runs if run.reused]
    assert reused == list(rq4_activation_depth.PINNED_GELU_DEPTH2)


def test_reader_tables_compare_gating_pairs_and_depth_without_rates(
    tmp_path: Path,
) -> None:
    _complete_surface(tmp_path)
    runs = rq4_activation_depth.load_runs(tmp_path)

    paired = rq4_activation_depth.paired_table(runs)
    depth = rq4_activation_depth.depth_table(runs)

    assert "| activation | plain FFN recall@100 | gated FFN recall@100 |" in paired
    assert "| ReLU → ReGLU |" in paired
    assert "| GELU → GEGLU |" in paired
    assert "| SiLU → SwiGLU |" in paired
    assert "| layers | GELU recall@100 | SwiGLU recall@100 |" in depth
    assert "| 2 |" in depth
    assert "| 4 |" in depth
    assert "| 8 |" in depth
    assert "LR" not in paired
    assert "LR" not in depth


def test_tuning_ledger_contains_every_run_and_bolds_each_selected_rate(
    tmp_path: Path,
) -> None:
    _complete_surface(tmp_path)

    ledger = rq4_activation_depth.tuning_report(
        rq4_activation_depth.load_runs(tmp_path)
    )

    assert ledger.startswith("# G1 RQ4 — FFN activation, gating, and depth tuning")
    assert ledger.count("### ") == 10
    assert ledger.count("**0.064**") == 10
    assert (
        sum(
            line.startswith("| 0.064 |") or line.startswith("| **0.064** |")
            for line in ledger.splitlines()
        )
        == 42
    )
    assert "artifact" not in ledger.lower()
    assert "rqffnact" not in ledger
    assert "validation recall@100" in ledger


def test_selection_uses_validation_recall_but_reader_reports_selected_final_metrics(
    tmp_path: Path,
) -> None:
    _complete_surface(tmp_path)
    specs = {
        spec.deep_lr: spec
        for spec in rq4_activation_depth.expected_specs()
        if spec.family == "GELU" and spec.layers == 2
    }
    _write_artifact(
        tmp_path,
        specs[0.012],
        recall=0.150,
        ndcg=0.060,
        validation_recall=0.130,
    )
    _write_artifact(
        tmp_path,
        specs[0.024],
        recall=0.140,
        ndcg=0.055,
        validation_recall=0.140,
    )

    runs = rq4_activation_depth.load_runs(tmp_path)
    selected = rq4_activation_depth.selected_runs(runs)[("GELU", 2)]
    reader = rq4_activation_depth.depth_table(runs)
    ledger = rq4_activation_depth.tuning_report(runs)

    assert selected.deep_lr == 0.024
    assert selected.validation_recall == 0.140
    assert selected.metrics["recall@100"] == 0.140
    assert "| 2 | 0.140 |" in reader
    assert "| **0.064** | **0.024** |" in ledger
    assert "**0.140** | **0.140** |" in ledger


def test_boundary_validation_optimum_is_rejected(tmp_path: Path) -> None:
    _complete_surface(tmp_path)
    boundary = next(
        spec
        for spec in rq4_activation_depth.expected_specs()
        if spec.family == "GELU" and spec.layers == 2 and spec.deep_lr == 0.096
    )
    _write_artifact(
        tmp_path,
        boundary,
        recall=0.150,
        ndcg=0.060,
        validation_recall=0.200,
    )

    with pytest.raises(ValueError, match="GELU.*2.*0.096.*boundary"):
        rq4_activation_depth.reader_tables(rq4_activation_depth.load_runs(tmp_path))


def test_partial_surface_is_rejected(tmp_path: Path) -> None:
    spec = rq4_activation_depth.expected_specs()[0]
    _write_artifact(tmp_path, spec, 0.13, 0.05)

    with pytest.raises(ValueError, match="incomplete"):
        rq4_activation_depth.load_runs(tmp_path)


def test_compact_rq4_uses_only_corrected_activation_depth_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collect.rq4_activation_depth, "load_runs", lambda _: [object()])
    monkeypatch.setattr(
        collect.rq4_activation_depth, "reader_tables", lambda _: "new reader tables"
    )

    tables = collect._rq4_tables("500m", [])

    assert tables == "new reader tables"
