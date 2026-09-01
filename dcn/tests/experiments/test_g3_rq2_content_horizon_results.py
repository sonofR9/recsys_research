import json
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.analysis import (
    rq2_content_horizon_results as results,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_id_boundary_results import (
    RQ2_ID_BOUNDARY_EVIDENCE_PATH,
    load_rq2_id_boundary_evidence,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_next_stage_results import (
    RQ2_NEXT_STAGE_EVIDENCE_PATH,
    load_rq2_next_stage_evidence,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_content_horizon_ledger import (
    RQ2_CONTENT_HORIZON_LEDGER_PATH,
    compile_rq2_content_horizon_ledger,
)


ROOT = Path(__file__).resolve().parents[3]
_BATCH_ID = "content-horizon-batch"


def _metrics(recall: float, ndcg: float) -> dict[str, float]:
    values = {
        f"{name}@{cutoff}": 0.01
        for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
        for cutoff in (10, 50, 100)
    }
    values["recall@100"] = recall
    values["ndcg@100"] = ndcg
    values["num_users"] = 3414.0
    return values


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _fixture(root: Path, monkeypatch, *, mutation: str | None = None):
    ledger = compile_rq2_content_horizon_ledger(ROOT)
    ledger_path = root / RQ2_CONTENT_HORIZON_LEDGER_PATH
    _write_json(ledger_path, ledger.to_dict())
    next_evidence = load_rq2_next_stage_evidence(ROOT / RQ2_NEXT_STAGE_EVIDENCE_PATH)
    id_evidence = load_rq2_id_boundary_evidence(
        ROOT / RQ2_ID_BOUNDARY_EVIDENCE_PATH
    )
    _write_json(root / ledger.resolved_next_stage_evidence.path, next_evidence)
    _write_json(root / ledger.id_boundary_evidence.path, id_evidence)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    context_path.parent.mkdir(parents=True)
    context_path.write_bytes(b"context")
    runner_path = (
        root
        / "experiments/g3_pretrained_item_embeddings/launchers/"
        "run_rq2_content_horizon.py"
    )
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("runner")
    recalls = (0.08, 0.09, 0.085)
    ndcgs = (0.03, 0.031, 0.032)
    metrics_by_run = {}
    job_ids = []
    for index, row in enumerate(ledger.rows):
        run_directory = root / "generated/logs" / row.run_name
        run_directory.mkdir(parents=True)
        compiled = decode_control_job(encode_control_job(ledger, row.id), ledger)
        contract = compiled.to_dict() | {
            "ledger_path": str(ledger_path),
            "ledger_sha256": ledger.sha256,
        }
        if mutation == "contract" and index == 0:
            contract["row_id"] = "changed"
        _write_json(run_directory / "g3_rq2_content_horizon_job.json", contract)
        horizon = row.horizon_epochs
        metadata = {
            "batch_size": 512,
            "seed": 42,
            "embedding_learning_rate": row.embedding_learning_rate,
            "deep_learning_rate": row.deep_learning_rate,
            "lr_schedule_horizon_epochs": horizon,
            "epochs_trained": horizon,
            "stopped_epoch": horizon,
            "lr_horizon_complete": True,
            "selection_resolved": True,
            "early_stopped": False,
            "g3_dataset_size": "native-50m",
            "g3_protocol_sha256": results.APPROVED_PROTOCOL_SHA256,
            "training_semantics_revision": 2,
            "g3_representation": results._EXPECTED_REPRESENTATION,
            "transfer_invariants": results._TRANSFER_INVARIANTS
            | {"lr_schedule_horizon_epochs": horizon},
            "best_epoch": (
                horizon
                if mutation == "boundary_winner" and index == 2
                else min(horizon, 10 + index)
            ),
            "lr_group_traces": {
                "embedding": [1.0] * (horizon - 1) + [0.0],
                "deep": [1.0] * (horizon - 1) + [0.0],
            },
        }
        if mutation == "representation" and index == 0:
            metadata["g3_representation"] = (
                results._EXPECTED_REPRESENTATION | {"history_hidden_dim": 64}
            )
        if mutation == "training" and index == 0:
            metadata["epochs_trained"] = horizon - 1
        _write_json(run_directory / "training_metadata.json", metadata)
        metrics = _metrics(recalls[index], ndcgs[index])
        if mutation == "boundary_winner" and index == 2:
            metrics["recall@100"] = 0.2
        metrics_by_run[row.run_name] = metrics
        stored_metrics = dict(metrics)
        if mutation == "metrics" and index == 0:
            stored_metrics["recall@100"] += 0.01
        _write_json(run_directory / "final_metrics.json", stored_metrics)
        (run_directory / "ranking_evidence.pt").write_bytes(b"ranking")
        _write_json(run_directory / "top_item_rankings.json", {"items": []})
        _write_json(
            run_directory / "g3_training_diagnostics.json",
            {"nonfinite_count": 1 if mutation == "diagnostics" and index == 0 else 0},
        )
        (run_directory / "sweep.log").write_text("complete")
        job_id = f"job-{index}"
        job_ids.append(job_id)
        environment = [
            f"{results.JOB_ENVIRONMENT}={encode_control_job(ledger, row.id)}",
            f"{results.LEDGER_ENVIRONMENT}={ledger_path}",
            "WANDB_MODE=offline",
        ]
        if mutation == "environment" and index == 0:
            environment.append("EXTRA=1")
        queue_script = runner_path
        if mutation == "runner" and index == 0:
            queue_script = root / "other/run_rq2_content_horizon.py"
            queue_script.parent.mkdir(parents=True)
            queue_script.write_text("different runner")
        _write_json(
            root
            / "generated/training-queue-service/completed"
            / f"{job_id}.json",
            {
                "id": job_id,
                "batch_id": _BATCH_ID,
                "data_group": "g3-native50m-likes",
                "submitted_at": 1.0,
                "dispatched_at": 2.0,
                "finished_at": 12.0 + index,
                "environment": environment,
                "exit_code": 0,
                "run": row.run_name,
                "script": str(queue_script),
            },
        )
    if mutation == "batch":
        job_ids.append("extra")
    _write_json(
        root
        / "generated/training-queue-service/batches"
        / f"{_BATCH_ID}.json",
        {
            "id": _BATCH_ID,
            "jobs": job_ids,
            "sealed": True,
            "submitted_at": 0.0,
            "sealed_at": 1.0,
        },
    )
    unique_calls = []
    window_calls = []
    monkeypatch.setattr(
        results,
        "load_rq2_content_horizon_ledger",
        lambda *args, **kwargs: ledger,
    )
    monkeypatch.setattr(
        results,
        "verify_rq2_content_horizon_inputs",
        lambda *args, **kwargs: root / "features.parquet",
    )
    monkeypatch.setattr(
        results,
        "verify_rq2_next_stage_evidence",
        lambda *args, **kwargs: next_evidence,
    )
    monkeypatch.setattr(
        results,
        "verify_rq2_id_boundary_evidence",
        lambda *args, **kwargs: id_evidence,
    )
    monkeypatch.setattr(
        results,
        "_recompute_metrics",
        lambda _context, ranking, _rankings: metrics_by_run[ranking.parent.name],
    )
    monkeypatch.setattr(
        results,
        "verify_unique_completed_run",
        lambda *args, **kwargs: unique_calls.append(kwargs),
    )
    monkeypatch.setattr(
        results,
        "verify_artifacts_in_job_window",
        lambda *args, **kwargs: window_calls.append((args, kwargs)),
    )
    return ledger, unique_calls, window_calls


def _candidate(
    row_id: str,
    recall: float,
    ndcg: float,
    seconds: float,
    order: int,
    *,
    embedding_learning_rate: float = 0.1,
    deep_learning_rate: float = 0.02,
    horizon_epochs: int = 25,
    best_epoch: int = 20,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "capacity": 32,
        "metrics": {"recall@100": recall, "ndcg@100": ndcg},
        "queue_wall_seconds": seconds,
        "manifest_order": order,
        "embedding_learning_rate": embedding_learning_rate,
        "deep_learning_rate": deep_learning_rate,
        "horizon_epochs": horizon_epochs,
        "best_epoch": best_epoch,
    }


def test_content_horizon_selection_uses_the_approved_tie_break_order() -> None:
    candidates = (
        _candidate("a", 0.1, 0.04, 8.0, 0),
        _candidate("b", 0.1, 0.05, 9.0, 1),
        _candidate("c", 0.1, 0.05, 7.0, 2),
        _candidate("d", 0.1, 0.05, 7.0, 3),
    )

    assert results.select_rq2_content_candidate(candidates)["row_id"] == "c"


def test_content_horizon_boundary_actions_follow_the_approved_plan() -> None:
    selected = _candidate(
        "boundary",
        0.1,
        0.04,
        8.0,
        0,
        embedding_learning_rate=results.APPROVED_PROTOCOL.embedding_lr_bounds[0],
        deep_learning_rate=0.03,
        horizon_epochs=40,
        best_epoch=40,
    )

    decision = results.assess_content_horizon_boundaries(selected)

    assert decision["capacity"] == {
        "selected": 32,
        "status": "resolved_user_approved",
        "additional_lower_capacity_authorized": False,
    }
    assert decision["extension_required"] is True
    assert decision["required_actions"] == [
        {
            "action": "three_joint_outward_lr_probes",
            "optimizer_group": "embedding_learning_rate",
            "direction": "lower",
        },
        {"action": "horizon_extension", "horizon_epochs": 60},
    ]


def test_collector_authenticates_three_runs_and_exposes_rq3_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    ledger, unique_calls, window_calls = _fixture(tmp_path, monkeypatch)

    evidence = results.build_rq2_content_horizon_evidence(
        tmp_path,
        batch_id=_BATCH_ID,
    )

    assert evidence["content_horizon_ledger"]["logical_sha256"] == ledger.sha256
    assert len(evidence["tuning_ledger"]) == 3
    assert evidence["horizon_probe_selection"]["selected"]["row_id"] == (
        "rq2_content_concat:11"
    )
    assert evidence["final_content_selection"]["selected"]["row_id"] == (
        "rq2_content_concat:11"
    )
    assert evidence["final_content_selection"]["status"] == "resolved"
    assert evidence["final_content_selection"]["provisional_selected"] is None
    assert evidence["final_content_selection"]["boundary_decision"][
        "extension_required"
    ] is False
    assert evidence["final_rq2_comparison"]["id_only_densenet"]["row_id"] == (
        "rq2_id_only_densenet:12"
    )
    assert len(evidence["rq3_inputs"]["reusable_width_32_content_rows"]) == 6
    assert len(unique_calls) == len(window_calls) == 3
    assert all(
        run["diagnostic_nonfinite_count"] == 0
        for run in evidence["tuning_ledger"]
    )
    assert all(
        run["metric_provenance"]["recomputed_from_ranking_evidence"] is True
        for run in evidence["tuning_ledger"]
    )


def test_boundary_winner_is_not_exported_to_rq3(tmp_path: Path, monkeypatch) -> None:
    _fixture(tmp_path, monkeypatch, mutation="boundary_winner")

    evidence = results.build_rq2_content_horizon_evidence(
        tmp_path,
        batch_id=_BATCH_ID,
    )

    selection = evidence["final_content_selection"]
    assert selection["status"] == "pending_boundary_followup"
    assert selection["selected"] is None
    assert selection["provisional_selected"]["row_id"] == "rq2_content_concat:12"
    assert selection["boundary_decision"]["extension_required"] is True
    assert evidence["final_rq2_comparison"] is None
    assert evidence["rq3_inputs"] is None


@pytest.mark.parametrize(
    "mutation",
    (
        "batch",
        "environment",
        "runner",
        "contract",
        "representation",
        "training",
        "metrics",
        "diagnostics",
    ),
)
def test_collector_rejects_mutated_post_run_provenance(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    _fixture(tmp_path, monkeypatch, mutation=mutation)

    with pytest.raises(ValueError):
        results.build_rq2_content_horizon_evidence(
            tmp_path,
            batch_id=_BATCH_ID,
        )


def test_content_horizon_evidence_is_hash_validated_and_immutable(
    tmp_path: Path, monkeypatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    evidence = results.build_rq2_content_horizon_evidence(
        tmp_path,
        batch_id=_BATCH_ID,
    )
    path = tmp_path / "result.json"

    results.persist_rq2_content_horizon_evidence(path, evidence, root=tmp_path)
    results.persist_rq2_content_horizon_evidence(path, evidence, root=tmp_path)
    assert results.load_rq2_content_horizon_evidence(path, root=tmp_path) == evidence
    changed = dict(evidence)
    changed["kind"] = "changed"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="identity or hash"):
        results.load_rq2_content_horizon_evidence(path, root=tmp_path)


def test_persistence_rejects_forged_rehashed_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    evidence = results.build_rq2_content_horizon_evidence(
        tmp_path,
        batch_id=_BATCH_ID,
    )
    forged = json.loads(json.dumps(evidence))
    forged["final_content_selection"]["selected"]["row_id"] = "forged"
    payload = {name: value for name, value in forged.items() if name != "sha256"}
    forged["sha256"] = results._canonical_sha256(payload)

    with pytest.raises(ValueError, match="differs from bound artifacts"):
        results.persist_rq2_content_horizon_evidence(
            tmp_path / "forged.json",
            forged,
            root=tmp_path,
        )
