import hashlib

import pytest

from experiments.g6_rqkmeans_history.analysis.finalize_native50m import (
    _authenticate_surface_row,
    choose_terminal,
)
from experiments.g6_rqkmeans_history.protocol.rq1_manifest import rq1_search_manifest


def _metrics(recall: float, ndcg: float) -> dict[str, float]:
    return {
        "recall@100": recall,
        "ndcg@100": ndcg,
        "mrr@100": 0.04,
        "coverage@100": 0.2,
    }


def test_terminal_retains_rq0_without_beyond_band_improvement() -> None:
    assert choose_terminal(
        rq0=_metrics(0.130, 0.052),
        suffix=_metrics(0.131, 0.053),
        none=_metrics(0.131, 0.054),
        suffix_is_rq0=False,
    ) == "rq0"


def test_terminal_promotes_none_only_with_recall_gain_and_ndcg_safety() -> None:
    assert choose_terminal(
        rq0=_metrics(0.130, 0.052),
        suffix=_metrics(0.131, 0.053),
        none=_metrics(0.134, 0.051),
        suffix_is_rq0=False,
    ) == "none"


def test_rq2_falls_back_to_rq0_when_selected_suffix_regresses() -> None:
    assert choose_terminal(
        rq0=_metrics(0.130, 0.052),
        suffix=_metrics(0.127, 0.060),
        none=_metrics(0.128, 0.061),
        suffix_is_rq0=False,
    ) == "rq0"


def test_terminal_requires_strict_recall_gain_and_safe_ndcg() -> None:
    assert choose_terminal(
        rq0=_metrics(0.130, 0.052),
        suffix=_metrics(0.132, 0.060),
        none=_metrics(0.131, 0.061),
        suffix_is_rq0=False,
    ) == "rq0"
    assert choose_terminal(
        rq0=_metrics(0.130, 0.052),
        suffix=_metrics(0.133, 0.049),
        none=_metrics(0.130, 0.051),
        suffix_is_rq0=False,
    ) == "rq0"


def test_surface_authentication_rejects_changed_artifact(tmp_path) -> None:
    job = rq1_search_manifest().jobs[0]
    directory = tmp_path / job.physical_run_name
    directory.mkdir(parents=True)
    names = {
        "job_contract": "g6_rq0_job.json" if job.reused else "g6_rq1_job.json",
        "final_metrics": "final_metrics.json",
        "training_metadata": "training_metadata.json",
        "ranking_evidence": "ranking_evidence.pt",
        "sid_diagnostics": "semantic_id_diagnostics.json",
        "sweep_log": "sweep.log",
    }
    digests = {}
    for key, filename in names.items():
        path = directory / filename
        path.write_bytes(key.encode())
        digests[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {
        "job_id": job.id,
        "run_name": job.physical_run_name,
        "artifact_sha256": digests,
    }
    _authenticate_surface_row(row, tmp_path)
    (directory / "final_metrics.json").write_text("changed")
    with pytest.raises(ValueError, match="frozen final_metrics changed"):
        _authenticate_surface_row(row, tmp_path)
