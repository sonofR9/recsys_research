from __future__ import annotations

import json
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from experiments.g6_rqkmeans_history.native500m.analysis import report as report_module

from experiments.g6_rqkmeans_history.native500m.analysis.report import (
    assemble_report_evidence,
    render_compact_report,
    render_tuning_report,
    write_reports,
)


METRICS = ("recall@100", "ndcg@100", "mrr@100", "coverage@100")


def _collected(
    root: Path,
    digit: str,
    metrics: dict[str, float],
    *,
    semantic: bool,
    backbone: str | None = None,
    stage: str | None = None,
    job_id: str | None = None,
) -> Path:
    directory = root / f"run-{digit}"
    directory.mkdir()
    job_id = job_id or f"job:{digit}"
    parameters = {
        "backbone": backbone or ("best_g1" if semantic else "original_g1"),
        "embedding_learning_rate": 0.02,
        "deep_learning_rate": 0.01,
        "seed": 42,
    }
    if semantic:
        parameters |= {
            "representation": "learned_sid_event",
            "levels": 3,
            "shared_codes": 512,
            "representation_width": 128,
            "collision_policy": "suffix",
            "sid_initialization": "random",
        }
    resolved_stage = stage or ("rq0_surface" if semantic else "controls")
    contract = {
        "manifest_logical_sha256": digit * 64,
        "job_logical_sha256": digit * 64,
        "job": {"job_id": job_id, "stage": resolved_stage, "parameters": parameters},
    }
    metadata = {"best_epoch": 20, "epochs_trained": 26}
    final = {
        "artifacts": (
            {
                "semantic_id_diagnostics": {
                    "identifier_collision_rate": 0.01,
                    "collided_item_fraction": 0.02,
                    "unique_base_tuples": 118,
                    "collision_bucket_size_p50": 1.0,
                    "collision_bucket_size_p95": 2.0,
                    "collision_bucket_size_p99": 2.0,
                    "collision_bucket_size_max": 2,
                    "p95_occupied_load": [2.0, 3.0, 4.0],
                    "p95_to_mean_occupied_load": [1.5, 1.7, 1.8],
                    "occupied_codes": [100, 110, 120],
                    "dead_code_fraction": [0.1, 0.05, 0.02],
                    "intra_code_cosine_similarity": [0.2, 0.1, 0.05],
                    "reconstruction_mse_by_depth": [0.4, 0.2, 0.1],
                },
                "slice_diagnostics": {
                    row["slice"]: {
                        key: value for key, value in row.items() if key != "slice"
                    }
                    for row in _slice_diagnostics()
                },
            }
            if semantic
            else {}
        )
    }
    payloads = {
        "g6_native500m_job.json": json.dumps(contract).encode(),
        "training_metadata.json": json.dumps(metadata).encode(),
        "final_metrics.json": json.dumps(metrics).encode(),
        "ranking_evidence.pt": b"ranking",
        "top100_item_evidence.pt": b"topk",
        "final_evaluation.json": json.dumps(final).encode(),
    }
    artifacts = {}
    for name, content in payloads.items():
        path = directory / name
        path.write_bytes(content)
        artifacts[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    body = {
        "schema": "g6-native500m-collected-run/v1",
        "stage": resolved_stage,
        "job_id": job_id,
        "run_name": f"run-{digit}",
        "manifest_logical_sha256": digit * 64,
        "manifest_physical_sha256": digit * 64,
        "job_logical_sha256": digit * 64,
        "batch_id": digit * 32,
        "best_epoch": 20,
        "trained_epochs": 26,
        "metrics": metrics,
        "artifacts": artifacts,
    }
    document = {
        **body,
        "evidence_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    path = root / f"collected-{digit}.json"
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return path


def _ledger(root: Path, index: int, collected_path: Path) -> Path:
    collected = json.loads(collected_path.read_text())
    contract_path = Path(collected["artifacts"]["g6_native500m_job.json"]["path"])
    base = json.loads(contract_path.read_text())["job"]["parameters"]
    artifacts = collected["artifacts"]
    if index == 0:
        stage = "rq0_surface"
        specifications = [("representation", "learned_sid_event", 42)]
    elif index == 1:
        stage = "rq1_confirmation"
        specifications = [
            ("sid_initialization", initialization, seed)
            for initialization in ("random", "content_pca")
            for seed in (42, 43, 44, 45)
        ]
    else:
        stage = "rq2_rq3_confirmation"
        specifications = [
            ("confirmation_arm", arm, seed)
            for arm in ("rq0_anchor", "suffix", "none")
            for seed in (42, 43, 44)
        ]
    rows = []
    for order, (field, value, seed) in enumerate(specifications):
        parameters = dict(base) | {"seed": seed}
        if field == "sid_initialization":
            parameters[field] = value
        if field == "confirmation_arm":
            parameters["collision_policy"] = "none" if value == "none" else "suffix"
        job_id = collected["job_id"] if order == 0 else f"{stage}:{value}:{seed}"
        rows.append(
            {
                "job_id": job_id,
                "job_logical_sha256": (
                    collected["job_logical_sha256"]
                    if order == 0
                    else hashlib.sha256(job_id.encode()).hexdigest()
                ),
                "run_name": (
                    collected["run_name"] if order == 0 else job_id.replace(":", "_")
                ),
                "manifest_order": order,
                "best_epoch": 20,
                "training_horizon": 26,
                "restored_checkpoint_sha256": "a" * 64,
                "parameters": parameters,
                "validation_metrics": {
                    "recall@100": collected["metrics"]["recall@100"],
                    "ndcg@100": collected["metrics"]["ndcg@100"],
                },
                "convergence": {
                    "first_epoch_at_95_percent": 10.0,
                    "normalized_recall_auc": 0.8,
                },
                "artifacts": artifacts,
            }
        )
    selected = (
        {"learned_sid_event": rows[0]["job_id"]}
        if index == 0
        else (
            {
                "random": rows[0]["job_id"],
                "content_pca": rows[4]["job_id"],
            }
            if index == 1
            else {
                "rq0_anchor": rows[0]["job_id"],
                "suffix": rows[3]["job_id"],
                "none": rows[6]["job_id"],
            }
        )
    )
    body = {
        "schema": "g6-native500m-stage-selection/v1",
        "stage": stage,
        "manifest_logical_sha256": collected["manifest_logical_sha256"],
        "manifest_physical_sha256": collected["manifest_physical_sha256"],
        "batch_id": "c" * 32,
        "recall_relative_dispersion": 0.01685,
        "selection_group_field": specifications[0][0],
        "selected_job_ids": selected,
        "candidates": rows,
    }
    document = {
        **body,
        "selection_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    path = root / f"selection-rq{index}.json"
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return path


def _bridge_ledger(root: Path, collected_path: Path) -> Path:
    collected = json.loads(collected_path.read_text())
    contract = json.loads(
        Path(collected["artifacts"]["g6_native500m_job.json"]["path"]).read_text()
    )
    candidate = {
        "job_id": collected["job_id"],
        "job_logical_sha256": collected["job_logical_sha256"],
        "run_name": collected["run_name"],
        "manifest_order": 0,
        "best_epoch": collected["best_epoch"],
        "training_horizon": 26,
        "restored_checkpoint_sha256": "a" * 64,
        "parameters": contract["job"]["parameters"],
        "validation_metrics": {
            "recall@100": collected["metrics"]["recall@100"],
            "ndcg@100": collected["metrics"]["ndcg@100"],
        },
        "convergence": {
            "first_epoch_at_95_percent": 10.0,
            "normalized_recall_auc": 0.8,
        },
        "artifacts": collected["artifacts"],
    }
    body = {
        "schema": "g6-native500m-stage-selection/v1",
        "stage": "terminal_bridge",
        "manifest_logical_sha256": collected["manifest_logical_sha256"],
        "manifest_physical_sha256": collected["manifest_physical_sha256"],
        "batch_id": "d" * 32,
        "recall_relative_dispersion": 0.01685,
        "selection_group_field": "representation",
        "selected_job_ids": {"learned_sid_event": collected["job_id"]},
        "candidates": [candidate],
    }
    document = {
        **body,
        "selection_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    path = root / "selection-terminal-bridge.json"
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return path


def _metrics(recall: float, ndcg: float) -> dict[str, float]:
    return {
        "recall@100": recall,
        "ndcg@100": ndcg,
        "mrr@100": recall / 3,
        "coverage@100": recall * 2,
    }


def _semantic_metrics(recall: float, ndcg: float) -> dict[str, float]:
    metrics = _metrics(recall, ndcg)
    for cutoff in (10, 50, 100):
        metrics[f"sid_exact_recall@{cutoff}"] = recall / 2
        metrics[f"sid_prefix_recall@{cutoff}_l1"] = recall * 1.5
        metrics[f"sid_prefix_recall@{cutoff}_l2"] = recall
        metrics[f"sid_prefix_recall@{cutoff}_l3"] = recall / 2
    return metrics


def _slice_diagnostics() -> list[dict[str, object]]:
    names = (
        "frequency_low",
        "frequency_middle",
        "frequency_high",
        "history_has_collided_base_sid",
        "history_has_no_collided_base_sid",
        "target_bucket_size_1",
        "target_bucket_size_2",
        "target_bucket_size_3_to_4",
        "target_bucket_size_5_plus",
    )
    return [
        {
            "slice": name,
            "control": {"recall@100": 0.1},
            "semantic": {"recall@100": 0.11},
            "num_users": 10,
            "num_targets": 12,
        }
        for name in names
    ]


def _evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    transitive_rq1_reuse: bool = False,
) -> dict[str, object]:
    def load_fixture_ledger(binding: object) -> dict[str, object]:
        assert isinstance(binding, dict)
        return json.loads(Path(binding["path"]).read_text())

    monkeypatch.setattr(report_module, "_load_selection_ledger", load_fixture_ledger)
    original = _collected(tmp_path, "1", _metrics(0.100, 0.040), semantic=False)
    best = _collected(tmp_path, "2", _metrics(0.120, 0.048), semantic=False)
    titles = (
        "How should SIDs describe history?",
        "Does content initialization beat random initialization?",
        "Which collision-resolved tokenizer should be used?",
        "Should collision resolution be removed?",
    )
    rqs = []
    treatments = []
    for index, title in enumerate(titles):
        treatment_metrics = _semantic_metrics(
            0.130 + index * 0.002, 0.051 + index * 0.001
        )
        stage = (
            "rq0_surface"
            if index == 0
            else "rq1_confirmation" if index == 1 else "rq2_rq3_confirmation"
        )
        treatment = _collected(
            tmp_path,
            str(index + 3),
            treatment_metrics,
            semantic=True,
            stage=stage,
            job_id=(
                f"{stage}:rq0_anchor:42" if index >= 2 else f"{stage}:selected:{index}"
            ),
        )
        treatments.append(treatment)
        ledger = _ledger(tmp_path, index, treatment)
        if index == 1 and transitive_rq1_reuse:
            selection = json.loads(ledger.read_text())
            selected = selection["candidates"][0]
            selected["reused_from"] = {
                "selection_sha256": "a" * 64,
                "job_id": "rq1_surface:random:42",
                "job_logical_sha256": "b" * 64,
            }
            body = {
                name: value
                for name, value in selection.items()
                if name != "selection_sha256"
            }
            selection["selection_sha256"] = hashlib.sha256(
                json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            ledger.write_text(
                json.dumps(selection, sort_keys=True, separators=(",", ":"))
            )
        rqs.append(
            {
                "index": index,
                "title": title,
                "candidates": [
                    {"method": f"RQ{index} selected SID method", "path": str(treatment)}
                ],
                "selection_paths": [str(ledger)],
            }
        )
    bridge = _collected(
        tmp_path,
        "7",
        _semantic_metrics(0.125, 0.049),
        semantic=True,
        backbone="original_g1",
        stage="terminal_bridge",
        job_id="terminal_bridge:learned_sid_event:00",
    )
    bridge_ledger = _bridge_ledger(tmp_path, bridge)
    assembly = {
        "schema": "g6-native500m-report-assembly/v1",
        "original": {"method": "Original G1/SASRec item IDs", "path": str(original)},
        "best_g1": {"method": "Best-G1 item IDs", "path": str(best)},
        "rqs": rqs,
        "terminal": {
            "method": "Best-G1 plus terminal SID bundle",
            "path": str(treatments[-1]),
        },
        "terminal_bridge": {
            "method": "Original-G1 plus terminal SID bundle",
            "path": str(bridge),
        },
        "terminal_bridge_selection_paths": [
            {
                "path": str(bridge_ledger),
                "manifest_path": "unused-by-render-fixture",
                "logs_root": "unused-by-render-fixture",
                "queue_state_directory": "unused-by-render-fixture",
            }
        ],
    }
    for question in assembly["rqs"]:
        question["selection_paths"] = [
            {
                "path": path,
                "manifest_path": "unused-by-render-fixture",
                "logs_root": "unused-by-render-fixture",
                "queue_state_directory": "unused-by-render-fixture",
            }
            for path in question["selection_paths"]
        ]
    return assemble_report_evidence(assembly)


def test_report_resolves_transitive_exact_reuse_to_ultimate_artifact_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path, monkeypatch, transitive_rq1_reuse=True)

    selected = next(row for row in evidence["rqs"][1]["rows"] if row.get("selected"))
    assert selected["job_id"] == "rq1_confirmation:selected:1"


def _section(report: str, heading: str) -> str:
    return report.split(heading, 1)[1].split("\n## ", 1)[0]


def test_report_has_consecutive_rqs_and_separate_reference_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = render_compact_report(_evidence(tmp_path, monkeypatch))

    positions = [report.index(f"## RQ{index}:") for index in range(4)]
    assert positions == sorted(positions)
    for index in range(4):
        section = _section(report, f"## RQ{index}:")
        assert "### Original-G1/SASRec comparison" in section
        assert "### Best-G1 local comparison" in section
        primary = section.split("### Original-G1/SASRec comparison", 1)[1]
        assert primary.index("Original G1/SASRec item IDs") < primary.index(
            f"RQ{index} selected SID method"
        )


def test_quality_tables_keep_recall_and_delta_columns_adjacent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = render_compact_report(_evidence(tmp_path, monkeypatch))

    assert "| Method | Recall@100 | Delta Recall@100 | NDCG@100 |" in report
    assert "RQ0 selected SID method" in report
    assert "0.130" in report
    assert "+30.000%" in report
    assert "+8.333%" in report
    for forbidden in ("latency", "throughput", "parameters", "memory", "MACs"):
        assert forbidden not in report


def test_reader_reports_all_sid_depths_intrinsics_and_eligible_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = render_compact_report(_evidence(tmp_path, monkeypatch))

    assert "### SID retrieval diagnostics" in report
    assert "Exact@10 / @50 / @100" in report
    assert "Prefix L1 @10 / @50 / @100" in report
    assert "Prefix L2 @10 / @50 / @100" in report
    assert "### Intrinsic SID diagnostics" in report
    assert "ICR" in report
    assert "Occupied p95 by level" in report
    assert "Intra-code cosine by level" in report
    assert "Reconstruction residual by depth" in report
    assert "Bucket p50 / p95 / p99 / max" in report
    assert "### Eligible target-frequency and collision slices" in report
    assert "frequency_low" in report
    assert "target_bucket_size_5_plus" in report


def test_aggregate_uses_unrounded_component_arithmetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = render_compact_report(_evidence(tmp_path, monkeypatch))
    aggregate = _section(report, "## Aggregated improvement")

    assert (
        "| Recall@100 | 0.100 | 0.136 | +0.036 | +0.020 | +0.025 | +0.045 | -0.009 | negative |"
        in aggregate
    )
    assert (
        "| NDCG@100 | 0.040 | 0.054 | +0.014 | +0.008 | +0.009 | +0.017 | -0.003 | negative |"
        in aggregate
    )


def test_report_writer_emits_rq0_and_rq1_rq3_reader_and_tuning_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path, monkeypatch)
    paths = write_reports(evidence, tmp_path)

    assert {path.name for path in paths} == {
        "rq0_reader_native500m.md",
        "rq0_tuning_native500m.md",
        "rq1_rq3_reader_native500m.md",
        "rq1_rq3_tuning_native500m.md",
    }
    assert "## RQ0:" in (tmp_path / "rq0_reader_native500m.md").read_text()
    assert "## RQ1:" in (tmp_path / "rq1_rq3_reader_native500m.md").read_text()
    tuning = render_tuning_report(evidence)
    assert "Embedding LR" in tuning
    assert "Deep LR" in tuning
    assert "Horizon" in tuning
    assert "Best epoch" in tuning
    assert (
        "| Method | Recall@100 | NDCG@100 | Embedding LR | Deep LR | Horizon | Best epoch |"
        in tuning
    )


def test_report_rejects_unbound_or_ambiguous_selected_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path, monkeypatch)
    unbound = deepcopy(evidence)
    del unbound["rqs"][0]["rows"][2]["evidence_identity"]
    with pytest.raises(ValueError, match="artifact-backed reassembly"):
        render_compact_report(unbound)

    ambiguous = deepcopy(evidence)
    duplicate = deepcopy(ambiguous["rqs"][0]["rows"][2])
    duplicate["method"] = "another selected treatment"
    ambiguous["rqs"][0]["rows"].append(duplicate)
    with pytest.raises(ValueError, match="artifact-backed reassembly"):
        render_compact_report(ambiguous)


def test_report_rejects_foreign_terminal_bridge_with_same_job_id_and_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path, monkeypatch)
    assembly = deepcopy(evidence["assembly"])
    unselected = _collected(
        tmp_path,
        "8",
        _semantic_metrics(0.140, 0.060),
        semantic=True,
        backbone="original_g1",
        stage="terminal_bridge",
        job_id="terminal_bridge:learned_sid_event:00",
    )
    assembly["terminal_bridge"] = {
        "method": "Unselected compatible bridge",
        "path": str(unselected),
    }

    with pytest.raises(ValueError, match="terminal bridge differs"):
        assemble_report_evidence(assembly)
