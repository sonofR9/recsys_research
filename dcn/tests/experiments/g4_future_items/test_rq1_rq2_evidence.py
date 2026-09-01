from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from dcn.eval.ranking_evidence import RankingEvidence, write_ranking_evidence
from experiments.g4_future_items.protocol.manifest import (
    canonical_bytes,
    canonical_sha256,
    load_strict_json,
)
from experiments.g4_future_items.report.rq1_rq2_evidence import (
    load_verified_tuning_ledger,
    verify_ranking_artifacts,
    verify_rq1_rq2_evidence,
)


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = (
    ROOT
    / "experiments/g4_future_items/evidence/rq1_rq2_evaluation_native50m.json"
)


def test_frozen_rq1_rq2_evidence_is_provenance_closed() -> None:
    document = verify_rq1_rq2_evidence(ARTIFACT, repo_root=ROOT)

    assert document["schema_version"] == 3
    assert document["kind"] == "g4_rq1_rq2_evaluation_native50m"
    assert document["dataset_size"] == "native-50m"
    assert document["protocol"]["ranking_identity"]["evaluation_users"] == 3414

    selections = document["selection_provenance"]
    assert selections["control_next_item"]["candidate_count"] == 24
    assert selections["rq1_24h"]["candidate_count"] == 12
    assert selections["rq2_next10"]["candidate_count"] == 12
    assert selections["control_next_item"]["winner"]["row_id"] == (
        "control_tuning:16"
    )
    assert selections["rq1_24h"]["winner"]["row_id"] == "rq1_tuning:12"
    assert selections["rq2_next10"]["winner"]["row_id"] == "rq2_tuning:12"
    control_stages = selections["control_next_item"]["stages"]
    assert control_stages["base"]["boundary_decision"]["rate_directions"] == {
        "deep_learning_rate": None,
        "embedding_learning_rate": "upper",
    }
    assert control_stages["base"]["boundary_decision"]["requires_extension"]
    assert not control_stages["boundary_round_1"]["boundary_decision"][
        "requires_extension"
    ]
    for selection in selections.values():
        assert selection["final_boundary_decision"]["requires_extension"] is False
        batch_ids = {
            stage["queue_batch"]["batch_id"]
            for stage in selection["stages"].values()
        }
        for candidate in selection["candidates"]:
            assert set(candidate["artifacts"]) == {
                "job_contract",
                "sweep_log",
                "training_metadata",
            }
            assert candidate["queue_job"]["batch_id"] in batch_ids

    calibration = document["calibration"]
    assert calibration["seeds"] == list(range(42, 52))
    assert len(calibration["sources"]) == 10
    assert len(calibration["queue_batches"]) == 2
    assert calibration["configuration"]["batch_size"] == 512
    assert calibration["configuration"]["dataset_size"] == "50m"
    for source in calibration["sources"]:
        assert set(source["artifacts"]) == {
            "final_metrics",
            "sweep_log",
            "training_metadata",
        }
        assert "queue_job" in source

    context = document["protocol"]["ranking_context"]
    assert context["num_users"] == 3414
    assert context["num_relevant_items"] == 19592
    for selected in document["selected_runs"].values():
        assert selected["ranking_semantics"]["context_payload_sha256"] == context[
            "payload_sha256"
        ]

    expected_digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert ARTIFACT.with_suffix(".sha256").read_text() == expected_digest


def test_rq1_rq2_evidence_rejects_a_changed_sidecar(tmp_path: Path) -> None:
    artifact = tmp_path / ARTIFACT.name
    artifact.write_bytes(ARTIFACT.read_bytes())
    artifact.with_suffix(".sha256").write_text("0" * 64)

    try:
        verify_rq1_rq2_evidence(artifact, repo_root=ROOT)
    except ValueError as error:
        assert "sidecar" in str(error)
    else:
        raise AssertionError("a changed evidence digest was accepted")


def _reseal(document: dict, path: Path) -> None:
    unsigned = dict(document)
    unsigned.pop("sha256", None)
    document["sha256"] = canonical_sha256(unsigned)
    path.write_bytes(canonical_bytes(document))


def test_verified_ledger_rejects_a_resealed_candidate_change(tmp_path: Path) -> None:
    source = ROOT / "experiments/g4_future_items/protocol/ledgers/rq1_tuning.json"
    ledger = load_strict_json(source)
    ledger["rows"][1], ledger["rows"][2] = ledger["rows"][2], ledger["rows"][1]
    changed = tmp_path / source.name
    _reseal(ledger, changed)

    references = {
        name: ROOT / "experiments/g4_future_items/protocol" / filename
        for name, filename in (
            ("control_semantics_manifest_sha256", "control_semantics_manifest.json"),
            ("selected_control_manifest_sha256", "selected_control_manifest.json"),
            ("treatment_semantics_manifest_sha256", "treatment_semantics_manifest.json"),
        )
    }
    try:
        load_verified_tuning_ledger(changed, reference_paths=references)
    except ValueError as error:
        assert "seeded compilation" in str(error)
    else:
        raise AssertionError("a resealed tuning ledger was accepted")


def test_verified_ledger_rejects_changed_boundary_predecessor(
    tmp_path: Path,
) -> None:
    source = (
        ROOT
        / "experiments/g4_future_items/protocol/ledgers/control_tuning_boundary_r1.json"
    )
    ledger = load_strict_json(source)
    ledger["predecessor_evidence"]["ledgers"][0]["sha256"] = "0" * 64
    changed = tmp_path / source.name
    _reseal(ledger, changed)

    references = {
        "control_semantics_manifest_sha256": ROOT
        / "experiments/g4_future_items/protocol/control_semantics_manifest.json"
    }
    try:
        load_verified_tuning_ledger(changed, reference_paths=references)
    except ValueError as error:
        assert "predecessor" in str(error)
    else:
        raise AssertionError("changed boundary predecessor evidence was accepted")


def test_verified_ledger_rejects_resealed_source_drift(tmp_path: Path) -> None:
    protocol = ROOT / "experiments/g4_future_items/protocol"
    semantics = load_strict_json(protocol / "control_semantics_manifest.json")
    first_source = semantics["source_paths"][0]
    semantics["sources"][first_source] = "0" * 64
    semantics_path = tmp_path / "control_semantics_manifest.json"
    semantics_path.write_bytes(canonical_bytes(semantics))

    ledger = load_strict_json(protocol / "ledgers/control_tuning.json")
    ledger["control_semantics_manifest_sha256"] = canonical_sha256(semantics)
    ledger_path = tmp_path / "control_tuning.json"
    _reseal(ledger, ledger_path)

    try:
        load_verified_tuning_ledger(
            ledger_path,
            reference_paths={
                "control_semantics_manifest_sha256": semantics_path,
            },
        )
    except ValueError as error:
        assert "source" in str(error)
    else:
        raise AssertionError("resealed source drift was accepted")


def _ranking_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    context = tmp_path / "context.pt"
    ranking = tmp_path / "ranking_evidence.pt"
    write_ranking_evidence(
        RankingEvidence(
            user_ids=torch.tensor([10]),
            history_item_ids=torch.tensor([11]),
            history_offsets=torch.tensor([0, 1]),
            relevant_item_ids=torch.tensor([12]),
            relevance_offsets=torch.tensor([0, 1]),
            relevant_train_frequencies=torch.tensor([2]),
            relevant_ranks=torch.tensor([2]),
            max_k=3,
        ),
        context_path=context,
        ranking_path=ranking,
    )
    top = tmp_path / "top_item_rankings.json"
    top.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "catalog_sha256": "catalog",
                "catalog_size": 3,
                "exclude_seen": False,
                "max_k": 3,
                "rankings": [{"user_id": 10, "item_ids": [11, 12, 13]}],
            }
        )
    )
    return context, ranking, top


def test_ranking_artifacts_reject_a_missing_context(tmp_path: Path) -> None:
    context, ranking, top = _ranking_fixture(tmp_path)
    context.unlink()

    try:
        verify_ranking_artifacts(
            context_path=context,
            ranking_path=ranking,
            top_item_rankings_path=top,
            relevance_by_user={10: {12}},
        )
    except ValueError as error:
        assert "ranking evidence" in str(error)
    else:
        raise AssertionError("a missing ranking context was accepted")


def test_ranking_artifacts_reject_a_different_context(tmp_path: Path) -> None:
    context, ranking, top = _ranking_fixture(tmp_path)
    other_context = tmp_path / "other_context.pt"
    other_ranking = tmp_path / "other_ranking.pt"
    write_ranking_evidence(
        RankingEvidence(
            user_ids=torch.tensor([20]),
            history_item_ids=torch.tensor([13]),
            history_offsets=torch.tensor([0, 1]),
            relevant_item_ids=torch.tensor([12]),
            relevance_offsets=torch.tensor([0, 1]),
            relevant_train_frequencies=torch.tensor([0]),
            relevant_ranks=torch.tensor([0]),
            max_k=3,
        ),
        context_path=other_context,
        ranking_path=other_ranking,
    )

    try:
        verify_ranking_artifacts(
            context_path=other_context,
            ranking_path=ranking,
            top_item_rankings_path=top,
            relevance_by_user={10: {12}},
        )
    except ValueError as error:
        assert "different evaluation context" in str(error)
    else:
        raise AssertionError("a ranking payload accepted a different context")


def test_ranking_artifacts_reject_changed_ranks(tmp_path: Path) -> None:
    context, ranking, top = _ranking_fixture(tmp_path)
    payload = torch.load(ranking, map_location="cpu", weights_only=True)
    payload["relevant_ranks"] = torch.tensor([1])
    torch.save(payload, ranking)

    try:
        verify_ranking_artifacts(
            context_path=context,
            ranking_path=ranking,
            top_item_rankings_path=top,
            relevance_by_user={10: {12}},
        )
    except ValueError as error:
        assert "ranks" in str(error)
    else:
        raise AssertionError("changed ranking ranks were accepted")
