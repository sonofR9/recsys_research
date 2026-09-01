import copy
import json
from pathlib import Path

import pytest

from experiments.g3_pretrained_item_embeddings.analysis.rq2_id_boundary_results import (
    APPROVED_RQ2_ID_BOUNDARY_EVIDENCE_SHA256,
    RQ2_ID_BOUNDARY_BATCH_ID,
    RQ2_ID_BOUNDARY_EVIDENCE_PATH,
    load_rq2_id_boundary_evidence,
    select_rq2_candidate,
    verify_rq2_id_boundary_evidence,
)


ROOT = Path(__file__).resolve().parents[3]


def test_id_boundary_selection_uses_approved_tie_break_order() -> None:
    candidates = [
        {
            "row_id": "first",
            "manifest_order": 0,
            "queue_wall_seconds": 12.0,
            "metrics": {"recall@100": 0.1, "ndcg@100": 0.04},
        },
        {
            "row_id": "second",
            "manifest_order": 1,
            "queue_wall_seconds": 11.0,
            "metrics": {"recall@100": 0.1, "ndcg@100": 0.05},
        },
    ]

    assert select_rq2_candidate(candidates)["row_id"] == "second"


def test_materialized_id_boundary_evidence_resolves_to_predecessor_winner() -> None:
    evidence = verify_rq2_id_boundary_evidence(
        ROOT / RQ2_ID_BOUNDARY_EVIDENCE_PATH,
        root=ROOT,
    )

    assert evidence["sha256"] == APPROVED_RQ2_ID_BOUNDARY_EVIDENCE_SHA256
    assert evidence["queue_batch"]["batch_id"] == RQ2_ID_BOUNDARY_BATCH_ID
    assert evidence["queue_batch"]["batch_id"] == (
        "55c225b9687a4f098e858bef235e4366"
    )
    assert [run["row_id"] for run in evidence["boundary_runs"]] == [
        "rq2_id_only_densenet:13",
        "rq2_id_only_densenet:14",
        "rq2_id_only_densenet:15",
    ]
    assert [run["metrics"]["recall@100"] for run in evidence["boundary_runs"]] == [
        0.0848355468753444,
        0.0857631866845541,
        0.08495562995955214,
    ]
    assert [run["best_epoch"] for run in evidence["boundary_runs"]] == [24, 23, 15]
    outward = evidence["outward_probe_selection"]["selected"]
    assert outward["row_id"] == "rq2_id_only_densenet:14"
    assert outward["deep_learning_rate"] == 0.0040542424
    final = evidence["final_selection"]
    assert final["selected"]["row_id"] == "rq2_id_only_densenet:12"
    assert final["selected"]["metrics"]["recall@100"] == 0.09074562121371973
    assert final["selected"]["metrics"]["ndcg@100"] == 0.031100697732330106
    assert final["selected"]["best_epoch"] == 29
    assert final["boundary_decision"] == {
        "status": "resolved",
        "outward_winner_on_new_boundary": False,
        "additional_runs_authorized": False,
        "next_action": "none",
    }
    assert evidence["content_capacity_status"] == {
        "status": "deferred_pending_user_approval",
        "changed_by_this_evidence": False,
    }


def test_id_boundary_evidence_is_closed_and_hash_validated(tmp_path: Path) -> None:
    evidence = load_rq2_id_boundary_evidence(ROOT / RQ2_ID_BOUNDARY_EVIDENCE_PATH)
    changed = copy.deepcopy(evidence)
    changed["final_selection"]["selected"]["metrics"]["recall@100"] = 1.0
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="hash differs"):
        load_rq2_id_boundary_evidence(path)
