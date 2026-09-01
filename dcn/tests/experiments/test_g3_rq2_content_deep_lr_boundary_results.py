import json
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.analysis import (
    rq2_content_deep_lr_boundary_results as results,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_capacity_results import (
    RQ2_CAPACITY_EVIDENCE_PATH,
    load_rq2_capacity_evidence,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_content_horizon_results import (
    RQ2_CONTENT_HORIZON_EVIDENCE_PATH,
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
from experiments.g3_pretrained_item_embeddings.protocol.rq2_content_deep_lr_boundary_ledger import (
    RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_PATH,
    compile_rq2_content_deep_lr_boundary_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq2_content_horizon_ledger import (
    RQ2_CONTENT_HORIZON_LEDGER_PATH,
    load_rq2_content_horizon_ledger,
)


ROOT = Path(__file__).resolve().parents[3]
_BATCH_ID = "content-deep-lr-boundary-batch"


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


def _fixture(
    root: Path,
    monkeypatch,
    *,
    winner: str = "middle",
    mutation: str | None = None,
):
    ledger = compile_rq2_content_deep_lr_boundary_ledger(ROOT)
    ledger_path = root / RQ2_CONTENT_DEEP_LR_BOUNDARY_LEDGER_PATH
    _write_json(ledger_path, ledger.to_dict())
    horizon_evidence = json.loads(
        (ROOT / RQ2_CONTENT_HORIZON_EVIDENCE_PATH).read_text()
    )
    source_metadata_fact = horizon_evidence["final_content_selection"][
        "provisional_selected"
    ]["artifacts"]["training_metadata"]
    source_metadata_path = ROOT / source_metadata_fact["path"]
    source_metadata = json.loads(source_metadata_path.read_text())
    normalized_schedule = [
        value / source_metadata["embedding_learning_rate"]
        for value in source_metadata["lr_group_traces"]["embedding"]
    ]
    copied_source_metadata = root / source_metadata_fact["path"]
    copied_source_metadata.parent.mkdir(parents=True, exist_ok=True)
    copied_source_metadata.write_bytes(source_metadata_path.read_bytes())
    horizon_ledger = load_rq2_content_horizon_ledger(
        ROOT / RQ2_CONTENT_HORIZON_LEDGER_PATH,
        root=ROOT,
    )
    next_evidence = load_rq2_next_stage_evidence(ROOT / RQ2_NEXT_STAGE_EVIDENCE_PATH)
    capacity_evidence = load_rq2_capacity_evidence(ROOT / RQ2_CAPACITY_EVIDENCE_PATH)
    id_evidence = load_rq2_id_boundary_evidence(
        ROOT / RQ2_ID_BOUNDARY_EVIDENCE_PATH
    )
    for path, value in (
        (root / ledger.content_horizon_evidence.path, horizon_evidence),
        (root / ledger.resolved_next_stage_evidence.path, next_evidence),
        (root / RQ2_CAPACITY_EVIDENCE_PATH, capacity_evidence),
        (root / ledger.id_boundary_evidence.path, id_evidence),
    ):
        _write_json(path, value)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    context_path.parent.mkdir(parents=True)
    context_path.write_bytes(b"context")
    runner_path = (
        root
        / "experiments/g3_pretrained_item_embeddings/launchers/"
        "run_rq2_content_deep_lr_boundary.py"
    )
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("runner")
    recalls = {
        "source": (0.08, 0.08, 0.08),
        "middle": (0.08, 0.1, 0.09),
        "smallest": (0.08, 0.09, 0.1),
    }[winner]
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
        _write_json(
            run_directory / "g3_rq2_content_deep_lr_boundary_job.json",
            contract,
        )
        metadata = {
            "batch_size": 512,
            "seed": 42,
            "embedding_learning_rate": row.embedding_learning_rate,
            "deep_learning_rate": row.deep_learning_rate,
            "lr_schedule_horizon_epochs": 40,
            "epochs_trained": 40,
            "stopped_epoch": 40,
            "lr_horizon_complete": True,
            "selection_resolved": True,
            "early_stopped": False,
            "g3_dataset_size": "native-50m",
            "g3_protocol_sha256": results.APPROVED_PROTOCOL_SHA256,
            "training_semantics_revision": 2,
            "g3_representation": results.EXPECTED_REPRESENTATION,
            "transfer_invariants": source_metadata["transfer_invariants"],
            "best_epoch": 10 + index,
            "lr_group_traces": {
                "embedding": [
                    row.embedding_learning_rate * factor
                    for factor in normalized_schedule
                ],
                "deep": [
                    row.deep_learning_rate * factor
                    for factor in normalized_schedule
                ],
            },
        }
        if mutation == "representation" and index == 0:
            metadata["g3_representation"] = results.EXPECTED_REPRESENTATION | {
                "history_hidden_dim": 64
            }
        if mutation == "dataset" and index == 0:
            metadata["g3_dataset_size"] = "native-500m"
        if mutation == "training" and index == 0:
            metadata["deep_learning_rate"] = 0.1
        if mutation == "invariants" and index == 0:
            metadata["transfer_invariants"] = dict(
                source_metadata["transfer_invariants"]
            ) | {"batch_size": 128}
        if mutation == "schedule" and index == 0:
            metadata["lr_group_traces"]["deep"] = [1.0] * 40
        if mutation == "schedule_shape" and index == 0:
            metadata["lr_group_traces"] = {
                name: [value * 0.5 for value in trace]
                for name, trace in metadata["lr_group_traces"].items()
            }
        _write_json(run_directory / "training_metadata.json", metadata)
        metrics = _metrics(recalls[index], 0.03 + index * 0.001)
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
            queue_script = root / "other/run_rq2_content_deep_lr_boundary.py"
            queue_script.parent.mkdir(parents=True)
            queue_script.write_text("different")
        _write_json(
            root / "generated/training-queue-service/completed" / f"{job_id}.json",
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
        root / "generated/training-queue-service/batches" / f"{_BATCH_ID}.json",
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
        "load_rq2_content_deep_lr_boundary_ledger",
        lambda *args, **kwargs: ledger,
    )
    monkeypatch.setattr(
        results,
        "verify_rq2_content_deep_lr_boundary_inputs",
        lambda *args, **kwargs: root / "features.parquet",
    )
    monkeypatch.setattr(
        results,
        "load_bound_rq2_content_horizon_ancestry",
        lambda *args, **kwargs: (horizon_evidence, horizon_ledger),
    )
    monkeypatch.setattr(
        results,
        "load_rq2_next_stage_evidence",
        lambda *args, **kwargs: next_evidence,
    )
    monkeypatch.setattr(
        results,
        "load_rq2_capacity_evidence",
        lambda *args, **kwargs: capacity_evidence,
    )
    monkeypatch.setattr(
        results,
        "load_rq2_id_boundary_evidence",
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
    wall: float,
    order: int,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "metrics": {"recall@100": recall, "ndcg@100": ndcg},
        "queue_wall_seconds": wall,
        "combined_manifest_order": order,
    }


def test_boundary_selection_uses_the_approved_tie_break_order() -> None:
    candidates = (
        _candidate("a", 0.1, 0.04, 8.0, 0),
        _candidate("b", 0.1, 0.05, 9.0, 1),
        _candidate("c", 0.1, 0.05, 7.0, 2),
        _candidate("d", 0.1, 0.05, 7.0, 3),
    )

    assert results.select_rq2_content_boundary_candidate(candidates)["row_id"] == "c"


def test_nonboundary_outward_winner_freezes_rq2_and_exposes_rq3(
    tmp_path: Path, monkeypatch
) -> None:
    ledger, unique_calls, window_calls = _fixture(tmp_path, monkeypatch)

    evidence = results.build_rq2_content_deep_lr_boundary_evidence(
        tmp_path,
        batch_id=_BATCH_ID,
    )

    selection = evidence["final_content_selection"]
    assert selection["status"] == "resolved"
    assert selection["selected"]["row_id"] == "rq2_content_concat:17"
    assert selection["provisional_selected"] is None
    assert selection["boundary_decision"]["required_actions"] == []
    assert evidence["final_rq2_comparison"]["id_only_densenet"]["row_id"] == (
        "rq2_id_only_densenet:12"
    )
    assert evidence["rq3_inputs"]["selected_content_input"]["row_id"] == (
        "rq2_content_concat:17"
    )
    assert len(evidence["rq3_inputs"]["reusable_width_32_content_rows"]) == 9
    assert len(evidence["all_tuning_ledger"]) == 30
    assert len({row["row_id"] for row in evidence["all_tuning_ledger"]}) == 30
    assert evidence["content_deep_lr_boundary_ledger"]["logical_sha256"] == (
        ledger.sha256
    )
    assert len(unique_calls) == len(window_calls) == 3


def test_authenticated_source_can_remain_the_frozen_winner(
    tmp_path: Path, monkeypatch
) -> None:
    _fixture(tmp_path, monkeypatch, winner="source")

    evidence = results.build_rq2_content_deep_lr_boundary_evidence(
        tmp_path,
        batch_id=_BATCH_ID,
    )

    selection = evidence["final_content_selection"]
    assert selection["status"] == "resolved"
    assert selection["selected"]["row_id"] == "rq2_content_concat:12"
    assert selection["boundary_decision"]["outward_probe_won"] is False
    assert evidence["rq3_inputs"]["selected_content_input"]["row_id"] == (
        "rq2_content_concat:12"
    )


def test_smallest_outward_lr_winner_requires_renewed_user_approval(
    tmp_path: Path, monkeypatch
) -> None:
    _fixture(tmp_path, monkeypatch, winner="smallest")

    evidence = results.build_rq2_content_deep_lr_boundary_evidence(
        tmp_path,
        batch_id=_BATCH_ID,
    )

    selection = evidence["final_content_selection"]
    assert selection["status"] == "pending_renewed_user_approval"
    assert selection["selected"] is None
    assert selection["provisional_selected"]["row_id"] == "rq2_content_concat:18"
    assert selection["boundary_decision"] == {
        "status": "pending_renewed_user_approval",
        "outward_probe_won": True,
        "selected_is_smallest_tested_deep_lr": True,
        "smallest_tested_deep_learning_rate": pytest.approx(
            0.0081084848 / (2 * 2**0.5)
        ),
        "additional_runs_authorized": False,
        "required_actions": [
            {
                "action": "renewed_user_approval",
                "reason": "outward_winner_on_new_lower_deep_lr_boundary",
            }
        ],
    }
    assert evidence["final_rq2_comparison"] is None
    assert evidence["rq3_inputs"] is None


@pytest.mark.parametrize(
    "mutation",
    (
        "batch",
        "environment",
        "runner",
        "contract",
        "dataset",
        "representation",
        "training",
        "invariants",
        "schedule",
        "schedule_shape",
        "metrics",
        "diagnostics",
    ),
)
def test_collector_rejects_mutated_post_run_provenance(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    _fixture(tmp_path, monkeypatch, mutation=mutation)

    with pytest.raises(ValueError):
        results.build_rq2_content_deep_lr_boundary_evidence(
            tmp_path,
            batch_id=_BATCH_ID,
        )


def test_boundary_evidence_persistence_authenticates_bound_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    evidence = results.build_rq2_content_deep_lr_boundary_evidence(
        tmp_path,
        batch_id=_BATCH_ID,
    )
    path = tmp_path / "result.json"

    results.persist_rq2_content_deep_lr_boundary_evidence(
        path,
        evidence,
        root=tmp_path,
    )
    assert (
        results.load_rq2_content_deep_lr_boundary_evidence(path, root=tmp_path)
        == evidence
    )
    forged = json.loads(json.dumps(evidence))
    forged["final_content_selection"]["selected"]["row_id"] = "forged"
    payload = {name: value for name, value in forged.items() if name != "sha256"}
    forged["sha256"] = results.canonical_sha256(payload)

    with pytest.raises(ValueError, match="differs from bound artifacts"):
        results.persist_rq2_content_deep_lr_boundary_evidence(
            tmp_path / "forged.json",
            forged,
            root=tmp_path,
        )
