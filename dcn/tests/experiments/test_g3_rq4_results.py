from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dcn.tests.experiments.test_g3_rq4_initial_infrastructure import (
    _persisted_ledger,
)
from experiments.g3_pretrained_item_embeddings.analysis import rq4_results as results
from experiments.g3_pretrained_item_embeddings.analysis.rq4_results import (
    _authoritative_ranking_metrics,
    assess_rq4_capacity_extension_boundaries,
    assess_rq4_family_boundaries,
    resolve_rq4_metadata_selection,
    select_rq4_capacity_winners,
    select_rq4_capacity_extension_winners,
    select_rq4_family_winners,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4 import (
    RQ4_METADATA_FAMILIES,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_initial_ledger import (
    RQ4_INITIAL_ARTIFACT_CONTRACTS,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    G3_CPU_THREAD_ENVIRONMENT,
    encode_control_job,
)


_FEATURE_IDENTITY = {
    "manifest_sha256": "a" * 64,
    "feature_data_sha256": "b" * 64,
    "frequency_terciles": {"num_catalog_items": 3, "slices": {}},
    "training_count_reference": {"sha256": "c" * 64},
    "slice_membership_reference": {"sha256": "d" * 64},
}


def test_rq4_uses_snapshot_coverage_and_ranking_evidence_for_other_metrics() -> None:
    reported = {
        f"{name}@{cutoff}": 0.1
        for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
        for cutoff in (10, 50, 100)
    } | {"num_users": 3.0}
    recomputed = dict(reported) | {"coverage@10": 0.08, "coverage@100": 0.12}

    metrics, coverage_deltas = _authoritative_ranking_metrics(
        reported, recomputed, row_label="row"
    )

    assert metrics["recall@100"] == 0.1
    assert metrics["coverage@10"] == 0.08
    assert metrics["coverage@100"] == 0.12
    assert coverage_deltas["coverage@10"] == pytest.approx(0.02)
    assert coverage_deltas["coverage@100"] == pytest.approx(-0.02)
    changed = dict(recomputed) | {"ndcg@100": 0.1001}
    with pytest.raises(ValueError, match="non-coverage"):
        _authoritative_ranking_metrics(reported, changed, row_label="row")


class _Row:
    def __init__(self, run: dict[str, object]) -> None:
        self.id = str(run["row_id"])
        self.family_id = str(run["family_id"])
        self._document = {
            "id": self.id,
            "family_id": run["family_id"],
            "representation": {"metadata_dim": run["metadata_dim"]},
            "training": {
                "embedding_learning_rate": run["embedding_learning_rate"],
                "deep_learning_rate": run["deep_learning_rate"],
                "horizon_epochs": run["horizon_epochs"],
            },
        }

    def to_dict(self) -> dict[str, object]:
        return self._document


def _ledger(runs: list[dict[str, object]], sha: str):
    ledger = SimpleNamespace(sha256=sha, rows=tuple(_Row(run) for run in runs))
    for run, row in zip(runs, ledger.rows, strict=True):
        run.update(
            {
                "ledger_sha256": sha,
                "job": row.to_dict(),
                "metric_provenance": {"recomputed_from_ranking_evidence": True},
                "feature_identity": deepcopy(_FEATURE_IDENTITY),
                "queue_job": {"job_id": run["row_id"]},
                "artifacts": {
                    contract.name: {"sha256": contract.name}
                    for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS
                },
                "efficiency": {"parameter_count": 1_000_000},
            }
        )
    return ledger


def _extra_surface(
    metadata_runs: list[dict[str, object]], initial, horizon
) -> tuple[dict[str, object], list[dict[str, object]]]:
    family = "rq4_artist_album"
    source_rows = [
        row.to_dict()
        for ledger in (initial, horizon)
        for row in ledger.rows
        if row.family_id == family
    ]
    rows = [
        {
            "id": f"rq4_extra_item_id:{index + 1:02d}",
            "family_id": "rq4_extra_item_id",
            "run_name": f"extra-{index + 1:02d}",
            "representation": {
                "catalog": "id_frozen_content",
                "history_hidden_dim": 128,
                "extra_item_id_dim": 8,
                "matched_metadata_family": family,
                "matched_metadata_dim": source["representation"]["metadata_dim"],
                "parameter_mismatch_fraction": 0.005,
            },
            "training": dict(source["training"])
            | {
                "batch_size": 512,
                "seed": 42,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
        }
        for index, source in enumerate(source_rows)
    ]
    payload = {
        "schema_version": 1,
        "kind": "g3_rq4_parameter_matched_extra_item_id",
        "protocol_sha256": results.APPROVED_PROTOCOL_SHA256,
        "selected_metadata_family": family,
        "selected_metadata_dim": 32,
        "maximum_parameter_mismatch_fraction": 0.01,
        "artifact_contracts": [
            contract.to_dict() for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS
        ],
        "rows": rows,
    }
    import hashlib

    sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    ledger = payload | {"sha256": sha}
    metadata_by_coordinate = {
        (
            run["metadata_dim"],
            run["embedding_learning_rate"],
            run["deep_learning_rate"],
            run["horizon_epochs"],
        ): run
        for run in metadata_runs
        if run["family_id"] == family
    }
    extra_runs = []
    for index, row in enumerate(rows):
        representation = row["representation"]
        training = row["training"]
        source = metadata_by_coordinate[
            (
                representation["matched_metadata_dim"],
                training["embedding_learning_rate"],
                training["deep_learning_rate"],
                training["horizon_epochs"],
            )
        ]
        run = _run(
            "rq4_extra_item_id",
            index + 1,
            capacity=representation["matched_metadata_dim"],
            horizon=training["horizon_epochs"],
            embedding_rate=training["embedding_learning_rate"],
            deep_rate=training["deep_learning_rate"],
            recall=0.09 + index / 1000,
            tail_recall=0.03 + index / 1000,
        )
        run.update(
            {
                "run_name": row["run_name"],
                "ledger_sha256": sha,
                "job": row,
                "metric_provenance": {"recomputed_from_ranking_evidence": True},
                "feature_identity": deepcopy(_FEATURE_IDENTITY),
                "queue_job": {"job_id": row["id"]},
                "artifacts": {
                    contract.name: {"sha256": contract.name}
                    for contract in RQ4_INITIAL_ARTIFACT_CONTRACTS
                },
                "efficiency": {
                    "parameter_count": source["efficiency"]["parameter_count"]
                    + 5_000
                },
            }
        )
        extra_runs.append(run)
    return ledger, extra_runs


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _collection_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutation: str | None = None,
):
    ledger_path, ledger, _, _ = _persisted_ledger(root, monkeypatch)
    runner = (
        root
        / "experiments/g3_pretrained_item_embeddings/launchers/run_rq4_initial.py"
    )
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("runner")
    context = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    context.parent.mkdir(parents=True, exist_ok=True)
    context.write_bytes(b"context")
    metrics_by_run = {}
    job_ids = []
    for index, row in enumerate(ledger.rows):
        job = row.to_dict()
        representation = job["representation"]
        training = job["training"]
        directory = root / "generated/logs" / row.run_name
        directory.mkdir(parents=True)
        compiled = results.decode_control_job(encode_control_job(ledger, row.id), ledger)
        contract = compiled.to_dict() | {
            "ledger_path": str(ledger_path),
            "ledger_sha256": ledger.sha256,
        }
        if mutation == "contract" and index == 0:
            contract["row_id"] = "changed"
        _write_json(directory / "g3_rq4_initial_job.json", contract)
        horizon = int(training["horizon_epochs"])
        _write_json(
            directory / "training_metadata.json",
            {
                "batch_size": 512,
                "seed": 42,
                "embedding_learning_rate": training["embedding_learning_rate"],
                "deep_learning_rate": training["deep_learning_rate"],
                "lr_schedule_horizon_epochs": horizon,
                "num_epochs": horizon,
                "max_epochs": horizon,
                "epochs_trained": horizon,
                "stopped_epoch": horizon,
                "lr_horizon_complete": True,
                "selection_resolved": True,
                "early_stopped": False,
                "g3_dataset_size": "native-50m",
                "g3_protocol_sha256": results.APPROVED_PROTOCOL_SHA256,
                "g3_representation": {
                    "catalog_representation": representation["catalog"],
                    "content_gate": "fixed",
                    "extra_item_id_dim": None,
                    "gate_hidden_dim": None,
                    "history_hidden_dim": representation["history_hidden_dim"],
                    "history_representation": "id_content",
                    "metadata": representation["metadata"],
                    "metadata_dim": representation["metadata_dim"],
                },
                "best_epoch": 20,
                "best_epoch_at_cap": False,
                "lr_group_traces": {
                    "embedding": [1.0] * (horizon - 1) + [0.0],
                    "deep": [1.0] * (horizon - 1) + [0.0],
                },
            },
        )
        metrics = {
            f"{name}@{cutoff}": 0.01
            for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
            for cutoff in (10, 50, 100)
        } | {"num_users": 3.0}
        metrics_by_run[row.run_name] = metrics
        stored = dict(metrics)
        if mutation == "metrics" and index == 0:
            stored["recall@100"] = 0.02
        _write_json(directory / "final_metrics.json", stored)
        (directory / "ranking_evidence.pt").write_bytes(b"ranking")
        _write_json(directory / "top_item_rankings.json", {})
        _write_json(directory / "g3_training_diagnostics.json", {})
        (directory / "sweep.log").write_text("done")
        job_id = f"job-{index:02d}"
        job_ids.append(job_id)
        _write_json(
            root / "generated/training-queue-service/completed" / f"{job_id}.json",
            {
                "id": job_id,
                "batch_id": "batch",
                "data_group": "g3-native50m-likes",
                "submitted_at": 1.0,
                "dispatched_at": 2.0,
                "finished_at": 12.0,
                "environment": [
                    f"G3_RQ4_INITIAL_JOB_B64={encode_control_job(ledger, row.id)}",
                    f"G3_RQ4_INITIAL_LEDGER_PATH={ledger_path}",
                    "WANDB_MODE=offline",
                    *G3_CPU_THREAD_ENVIRONMENT,
                ],
                "exit_code": 0,
                "run": row.run_name,
                "script": str(runner),
            },
        )
    _write_json(
        root / "generated/training-queue-service/batches/batch.json",
        {
            "id": "batch",
            "jobs": job_ids,
            "sealed": True,
            "submitted_at": 0.0,
            "sealed_at": 1.0,
        },
    )
    monkeypatch.setattr(results, "load_training_item_counts", lambda path: {0: 0})
    monkeypatch.setattr(
        results,
        "_recompute_metrics",
        lambda context_path, ranking_path, rankings_path: metrics_by_run[
            ranking_path.parent.name
        ],
    )
    monkeypatch.setattr(results, "_validate_training_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(
        results,
        "_ranking_slices",
        lambda **kwargs: _run("slice", 1, capacity=0)["slices"],
    )
    monkeypatch.setattr(
        results,
        "_efficiency",
        lambda **kwargs: {
            "queue_wall_seconds": kwargs["queue_wall_seconds"],
            "parameter_count": 1_000_000,
        },
    )
    monkeypatch.setattr(results, "verify_unique_completed_run", lambda *a, **k: None)
    monkeypatch.setattr(results, "verify_artifacts_in_job_window", lambda *a, **k: None)
    return ledger_path, ledger


def _run(
    family: str,
    index: int,
    *,
    capacity: int,
    horizon: int = 25,
    embedding_rate: float = 0.14,
    deep_rate: float = 0.03,
    recall: float = 0.10,
    tail_recall: float = 0.04,
) -> dict[str, object]:
    return {
        "row_id": f"{family}:{index:02d}",
        "family_id": family,
        "metadata_dim": capacity,
        "embedding_learning_rate": embedding_rate,
        "deep_learning_rate": deep_rate,
        "horizon_epochs": horizon,
        "best_epoch": min(20, horizon),
        "queue_wall_seconds": 100.0 + index,
        "metrics": {"recall@100": recall, "ndcg@100": recall / 3},
        "slices": {
            name: {
                "recall@100": tail_recall if name == "tail" else recall,
                "num_users": 100 + offset,
                "num_targets": 200 + offset,
                "item_membership_sha256": str(offset) * 64,
            }
            for offset, name in enumerate(("head", "mid", "tail"), start=1)
        },
    }


def _capacity_runs() -> list[dict[str, object]]:
    runs = []
    for family_offset, family in enumerate(RQ4_METADATA_FAMILIES):
        index = 1
        for capacity in (16, 32, 64):
            for rate_offset, (embedding, deep) in enumerate(
                ((0.057, 0.019), (0.147, 0.032), (0.178, 0.0104))
            ):
                runs.append(
                    _run(
                        family,
                        index,
                        capacity=capacity,
                        embedding_rate=embedding,
                        deep_rate=deep,
                        recall=(
                            0.09
                            + family_offset * 0.001
                            + (0.003 if capacity == 32 else 0.0)
                            + rate_offset * 0.0001
                        ),
                    )
                )
                index += 1
    return runs


def test_width256_selection_requires_both_capacity_renewals_but_no_lr_extension() -> None:
    source = []
    boundary = []
    coordinates = {
        "rq4_artist": (
            (0.2514903402185926, 0.00737546917452711),
            (0.3556610499429575, 0.005215244267740468),
            (0.5029806804371852, 0.003687734587263555),
        ),
        "rq4_album": (
            (0.04068087164982432, 0.01852175330591617),
            (0.028765720208170354, 0.01852175330591617),
            (0.02034043582491216, 0.01852175330591617),
        ),
    }
    for family in coordinates:
        index = 1
        for capacity in (16, 32, 64, 128):
            for embedding, deep in (
                (0.05753144041634071, 0.010430488535480936),
                (0.1474458978470563, 0.01852175330591617),
                (0.17783052497147875, 0.032433939334700325),
            ):
                source.append(
                    _run(
                        family,
                        index,
                        capacity=capacity,
                        embedding_rate=embedding,
                        deep_rate=deep,
                        recall=0.09,
                    )
                )
                index += 1
        for offset, (embedding, deep) in enumerate(coordinates[family], start=1):
            boundary.append(
                _run(
                    family,
                    offset + 20,
                    capacity=256,
                    embedding_rate=embedding,
                    deep_rate=deep,
                    recall=0.11 if offset == 2 else 0.10,
                )
            )

    selections, capacity_approval, lr_approval = (
        results.select_rq4_single_metadata_width256_boundaries(source, boundary)
    )

    assert capacity_approval == ["rq4_artist", "rq4_album"]
    assert lr_approval == []
    assert {
        family: selection["selected"]["metadata_dim"]
        for family, selection in selections.items()
    } == {"rq4_artist": 256, "rq4_album": 256}
    assert all(
        selection["boundary_decision"]["embedding_learning_rate"]["direction"]
        is None
        and selection["boundary_decision"]["deep_learning_rate"]["direction"]
        is None
        for selection in selections.values()
    )


def _full_runs() -> list[dict[str, object]]:
    runs = _capacity_runs()
    for family_offset, family in enumerate(RQ4_METADATA_FAMILIES):
        for offset, (horizon, embedding, deep) in enumerate(
            (
                (15, 0.047, 0.041),
                (25, 0.124, 0.024),
                (40, 0.304, 0.0145),
            ),
            start=10,
        ):
            runs.append(
                _run(
                    family,
                    offset,
                    capacity=32,
                    horizon=horizon,
                    embedding_rate=embedding,
                    deep_rate=deep,
                    recall=0.095 + family_offset * 0.002 + offset * 0.0001,
                    tail_recall=0.05 + family_offset * 0.002,
                )
            )
    return runs


def test_capacity_selection_requires_equal_nine_opportunity_surfaces() -> None:
    runs = _capacity_runs()
    ledger = _ledger(runs, "1" * 64)

    selected = select_rq4_capacity_winners(runs, ledger=ledger)

    assert set(selected) == set(RQ4_METADATA_FAMILIES)
    assert {row["metadata_dim"] for row in selected.values()} == {32}
    with pytest.raises(ValueError, match="authenticated ledger rows"):
        select_rq4_capacity_winners(runs[:-1], ledger=ledger)


def test_capacity_extension_combines_all_rows_and_exposes_actual_boundaries() -> None:
    initial_runs = _capacity_runs()
    extension_runs = []
    for family_offset, family in enumerate(RQ4_METADATA_FAMILIES):
        for index, (embedding, deep) in enumerate(
            ((0.057, 0.019), (0.147, 0.032), (0.178, 0.0104)), start=1
        ):
            run = _run(
                family,
                index,
                capacity=128,
                embedding_rate=embedding,
                deep_rate=deep,
                recall=0.092 + family_offset * 0.001 + index * 0.0001,
            )
            run["row_id"] = f"{family}_capacity_extension:{index:02d}"
            extension_runs.append(run)
    extension_runs[1]["metrics"]["recall@100"] = 0.2
    initial = _ledger(initial_runs, "1" * 64)
    extension = _ledger(extension_runs, "2" * 64)

    selected = select_rq4_capacity_extension_winners(
        initial_runs,
        extension_runs,
        initial_ledger=initial,
        extension_ledger=extension,
    )
    boundaries = assess_rq4_capacity_extension_boundaries(
        selected, [*initial_runs, *extension_runs]
    )

    assert selected["rq4_artist"]["metadata_dim"] == 128
    assert boundaries["rq4_artist"]["capacity"] == {
        "selected": 128,
        "tested_values": [16, 32, 64, 128],
        "direction": "upper",
        "renewed_approval_required": True,
    }
    assert boundaries["rq4_artist"]["embedding_learning_rate"]["direction"] is None
    assert boundaries["rq4_album"]["capacity"]["direction"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    ((None, None), ("contract", "job contract"), ("metrics", "ranking evidence")),
)
def test_stage_collection_authenticates_queue_contracts_artifacts_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str | None,
    message: str | None,
) -> None:
    ledger_path, ledger = _collection_fixture(
        tmp_path, monkeypatch, mutation=mutation
    )

    if message is not None:
        with pytest.raises(ValueError, match=message):
            results.collect_authenticated_rq4_stage_runs(
                tmp_path,
                ledger=ledger,
                ledger_path=ledger_path,
                batch_id="batch",
            )
        return

    runs = results.collect_authenticated_rq4_stage_runs(
        tmp_path,
        ledger=ledger,
        ledger_path=ledger_path,
        batch_id="batch",
    )

    assert len(runs) == 27
    assert all(run["ledger_sha256"] == ledger.sha256 for run in runs)
    assert all(
        run["metric_provenance"]["recomputed_from_ranking_evidence"] is True
        for run in runs
    )
    assert all(run["feature_identity"] == ledger.feature_identity for run in runs)

    runs[0]["efficiency"]["parameter_count"] = 2_000_000
    with pytest.raises(ValueError, match="efficiency changed"):
        results._reauthenticate_result_files(
            tmp_path,
            runs[0],
            identity=results._metadata_identity(tmp_path, ledger),
        )


def test_full_selection_requires_exactly_twelve_opportunities_per_family() -> None:
    runs = _full_runs()
    initial = _ledger(runs[:27], "1" * 64)
    horizon = _ledger(runs[27:], "2" * 64)

    selected = select_rq4_family_winners(
        runs, initial_ledger=initial, horizon_ledger=horizon
    )

    assert set(selected) == set(RQ4_METADATA_FAMILIES)
    assert all(str(row["row_id"]).endswith(":12") for row in selected.values())
    duplicate = deepcopy(runs)
    duplicate[-1]["row_id"] = duplicate[-2]["row_id"]
    with pytest.raises(ValueError, match="authenticated ledger row"):
        select_rq4_family_winners(
            duplicate, initial_ledger=initial, horizon_ledger=horizon
        )


def test_predecessor_is_bound_to_exact_rq3_selection_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, ledger, _, _ = _persisted_ledger(tmp_path, monkeypatch)
    selection = json.loads(
        (tmp_path / ledger.rq3_final_evidence.path).read_text()
    )
    selected = selection["selected"]
    contract = json.loads(
        (tmp_path / selected["artifacts"]["job_contract"]["path"]).read_text()
    )
    run = {
        "row_id": selected["row_id"],
        "family_id": selected["family_id"],
        "run_name": selected["run_name"],
        "job": contract["job"],
        "artifacts": selected["artifacts"],
        "queue_job": selected["queue_job"] | {"job_id": "rq3-job"},
        "metric_provenance": {
            "recomputed_from_ranking_evidence": True,
            "ranking_context": selection["ranking_context"],
        },
    }
    monkeypatch.setattr(results, "_reauthenticate_result_files", lambda *a, **k: None)

    results._validate_predecessor_run(
        tmp_path,
        run,
        ledger=ledger,
        identity=SimpleNamespace(),
    )

    changed = deepcopy(run)
    changed["artifacts"]["ranking_evidence"] = dict(
        changed["artifacts"]["ranking_evidence"]
    )
    changed["artifacts"]["ranking_evidence"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact RQ3 selection"):
        results._validate_predecessor_run(
            tmp_path,
            changed,
            ledger=ledger,
            identity=SimpleNamespace(),
        )


def test_boundaries_enforce_rates_horizon_endpoint_and_capacity() -> None:
    runs = _full_runs()
    initial = _ledger(runs[:27], "1" * 64)
    horizon = _ledger(runs[27:], "2" * 64)
    selected = select_rq4_family_winners(
        runs, initial_ledger=initial, horizon_ledger=horizon
    )
    selected["rq4_artist"] = next(
        row
        for row in runs
        if row["family_id"] == "rq4_artist" and row["row_id"].endswith(":07")
    )
    selected["rq4_album"] = next(
        row
        for row in runs
        if row["family_id"] == "rq4_album" and row["row_id"].endswith(":11")
    )
    selected["rq4_artist_album"] = next(
        row
        for row in runs
        if row["family_id"] == "rq4_artist_album"
        and row["row_id"].endswith(":12")
    )
    selected["rq4_artist_album"]["best_epoch"] = 40

    boundaries = assess_rq4_family_boundaries(selected, runs)

    assert boundaries["rq4_artist"]["capacity"]["direction"] == "upper"
    assert boundaries["rq4_artist"]["capacity"]["extension_capacity"] == 128
    assert boundaries["rq4_album"]["embedding_learning_rate"]["direction"] is None
    assert boundaries["rq4_artist_album"]["horizon"]["extend_to_epochs"] == 60
    assert boundaries["rq4_artist_album"]["extension_required"] is True

    selected["rq4_artist"] = next(
        row
        for row in runs
        if row["family_id"] == "rq4_artist" and row["row_id"].endswith(":10")
    )
    tested = assess_rq4_family_boundaries(selected, runs)["rq4_artist"]
    assert tested["embedding_learning_rate"]["direction"] == "lower"
    assert tested["deep_learning_rate"]["direction"] == "upper"
    assert tested["embedding_learning_rate"]["tested_interval"] == [0.047, 0.304]


def test_metadata_selection_requires_aggregate_noninferiority_and_tail_advantage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _full_runs()
    initial = _ledger(runs[:27], "1" * 64)
    horizon = _ledger(runs[27:], "2" * 64)
    winners = select_rq4_family_winners(
        runs, initial_ledger=initial, horizon_ledger=horizon
    )
    monkeypatch.setattr(
        results,
        "_metadata_identity",
        lambda root, ledger: SimpleNamespace(
            manifest_sha256=_FEATURE_IDENTITY["manifest_sha256"],
            feature_data_sha256=_FEATURE_IDENTITY["feature_data_sha256"],
            frequency_terciles=_FEATURE_IDENTITY["frequency_terciles"],
            training_count_reference=_FEATURE_IDENTITY["training_count_reference"],
            slice_membership_reference=_FEATURE_IDENTITY[
                "slice_membership_reference"
            ],
        ),
    )
    monkeypatch.setattr(results, "_reauthenticate_result_files", lambda *a, **k: None)
    monkeypatch.setattr(results, "_validate_predecessor_run", lambda *a, **k: None)
    monkeypatch.setattr(
        results,
        "assess_rq4_family_boundaries",
        lambda *a, **k: {
            family: {"extension_required": False}
            for family in RQ4_METADATA_FAMILIES
        },
    )
    predecessor = _run(
        "predecessor",
        1,
        capacity=0,
        recall=0.10,
        tail_recall=0.04,
    )
    predecessor["feature_identity"] = deepcopy(_FEATURE_IDENTITY)
    metadata = winners["rq4_artist_album"]
    metadata["metrics"]["recall@100"] = 0.101
    metadata["slices"]["tail"]["recall@100"] = 0.052
    extra_ledger, extra_runs = _extra_surface(runs, initial, horizon)
    monkeypatch.setattr(
        results,
        "_recompiled_extra_id_documents",
        lambda *a, **k: deepcopy(extra_ledger["rows"]),
    )
    extra_surface = SimpleNamespace()
    extra_runs[-1]["metrics"]["recall@100"] = 0.101
    extra_runs[-1]["slices"]["tail"]["recall@100"] = 0.045

    decision = resolve_rq4_metadata_selection(
        root=tmp_path,
        predecessor=predecessor,
        metadata_runs=runs,
        extra_id_runs=extra_runs,
        initial_ledger=initial,
        horizon_ledger=horizon,
        extra_id_ledger=extra_ledger,
        extra_id_surface=extra_surface,
        recall_relative_dispersion=0.02,
    )

    assert decision["selected_metadata_family"] == "rq4_artist_album"
    assert decision["aggregate_noninferior"] is True
    assert decision["tail_beats_predecessor"] is True
    assert decision["tail_beats_extra_item_id"] is True
    assert decision["metadata_promoted"] is True

    altered_ledger = deepcopy(extra_ledger)
    altered_ledger["rows"][0]["representation"]["catalog"] = "learned_id"
    altered_payload = {
        name: value for name, value in altered_ledger.items() if name != "sha256"
    }
    import hashlib

    altered_ledger["sha256"] = hashlib.sha256(
        json.dumps(
            altered_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    altered_runs = deepcopy(extra_runs)
    altered_runs[0]["job"] = altered_ledger["rows"][0]
    altered_runs[0]["ledger_sha256"] = altered_ledger["sha256"]
    with pytest.raises(ValueError, match="exact matched surface"):
        resolve_rq4_metadata_selection(
            root=tmp_path,
            predecessor=predecessor,
            metadata_runs=runs,
            extra_id_runs=altered_runs,
            initial_ledger=initial,
            horizon_ledger=horizon,
            extra_id_ledger=altered_ledger,
            extra_id_surface=extra_surface,
            recall_relative_dispersion=0.02,
        )

    duplicated_metadata = [*runs[:-1], runs[0]]
    with pytest.raises(ValueError, match="authenticated ledger row"):
        resolve_rq4_metadata_selection(
            root=tmp_path,
            predecessor=predecessor,
            metadata_runs=duplicated_metadata,
            extra_id_runs=extra_runs,
            initial_ledger=initial,
            horizon_ledger=horizon,
            extra_id_ledger=extra_ledger,
            extra_id_surface=extra_surface,
            recall_relative_dispersion=0.02,
        )

    miskeyed_metadata = deepcopy(runs)
    miskeyed_metadata[0]["family_id"] = "rq4_album"
    with pytest.raises(ValueError, match="authenticated ledger row"):
        resolve_rq4_metadata_selection(
            root=tmp_path,
            predecessor=predecessor,
            metadata_runs=miskeyed_metadata,
            extra_id_runs=extra_runs,
            initial_ledger=initial,
            horizon_ledger=horizon,
            extra_id_ledger=extra_ledger,
            extra_id_surface=extra_surface,
            recall_relative_dispersion=0.02,
        )

    with pytest.raises(ValueError, match="all twelve"):
        resolve_rq4_metadata_selection(
            root=tmp_path,
            predecessor=predecessor,
            metadata_runs=runs,
            extra_id_runs=extra_runs[:-1],
            initial_ledger=initial,
            horizon_ledger=horizon,
            extra_id_ledger=extra_ledger,
            extra_id_surface=extra_surface,
            recall_relative_dispersion=0.02,
        )

    extra_runs[0]["efficiency"]["parameter_count"] = 1_010_000
    with pytest.raises(ValueError, match="below one percent"):
        resolve_rq4_metadata_selection(
            root=tmp_path,
            predecessor=predecessor,
            metadata_runs=runs,
            extra_id_runs=extra_runs,
            initial_ledger=initial,
            horizon_ledger=horizon,
            extra_id_ledger=extra_ledger,
            extra_id_surface=extra_surface,
            recall_relative_dispersion=0.02,
        )
    extra_runs[0]["efficiency"]["parameter_count"] = 1_005_000

    extra_runs[-1]["slices"]["tail"]["recall@100"] = 0.06
    rejected = resolve_rq4_metadata_selection(
        root=tmp_path,
        predecessor=predecessor,
        metadata_runs=runs,
        extra_id_runs=extra_runs,
        initial_ledger=initial,
        horizon_ledger=horizon,
        extra_id_ledger=extra_ledger,
        extra_id_surface=extra_surface,
        recall_relative_dispersion=0.02,
    )
    assert rejected["metadata_promoted"] is False

    monkeypatch.setattr(
        results,
        "assess_rq4_family_boundaries",
        lambda *a, **k: {
            family: {"extension_required": family == "rq4_artist"}
            for family in RQ4_METADATA_FAMILIES
        },
    )
    with pytest.raises(ValueError, match="unresolved boundaries"):
        resolve_rq4_metadata_selection(
            root=tmp_path,
            predecessor=predecessor,
            metadata_runs=runs,
            extra_id_runs=extra_runs,
            initial_ledger=initial,
            horizon_ledger=horizon,
            extra_id_ledger=extra_ledger,
            extra_id_surface=extra_surface,
            recall_relative_dispersion=0.02,
        )
