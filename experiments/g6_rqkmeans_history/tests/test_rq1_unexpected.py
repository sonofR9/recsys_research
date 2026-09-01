import json
from pathlib import Path

import pytest
import torch

from experiments.g6_rqkmeans_history.analysis.rq1_unexpected import (
    collect_rq1_unexpected_evidence,
    gradient_probe,
    projection_reconstruction,
)


ROOT = Path(__file__).resolve().parents[1]


def test_projection_reconstruction_measures_the_requested_lossy_width() -> None:
    centroids = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])).unsqueeze(0)

    rows = projection_reconstruction(centroids, output_dim=2)

    assert len(rows) == 1
    assert rows[0]["input_width"] == 4
    assert rows[0]["output_width"] == 2
    assert 0 < rows[0]["retained_variance_fraction"] < 1
    assert rows[0]["centered_reconstruction_mse"] > 0


def test_gradient_probe_reaches_learned_codes_and_not_frozen_centroids() -> None:
    probe = gradient_probe()

    assert probe["learned_base_gradient_nonzero"] is True
    assert probe["learned_base_rows_with_gradient"] == 4
    assert probe["frozen_centroid_trainable_parameters"] == 0
    assert (
        probe["frozen_centroid_sha256_before"] == probe["frozen_centroid_sha256_after"]
    )
    assert probe["frozen_centroid_unchanged"] is True


def test_committed_unexpected_evidence_is_complete() -> None:
    evidence = json.loads((ROOT / "evidence/rq1_unexpected_native50m.json").read_text())

    assert evidence["schema"] == "g6-rq1-unexpected/v1"
    assert evidence["dataset_size"] == "native-50m"
    assert evidence["source_artifacts"]["authenticated_surface_rq1_runs"] == 32
    assert evidence["source_artifacts"]["authenticated_confirmation_rq1_runs"] == 8
    assert evidence["initialization_identity"]["all_checks_passed"] is True
    assert evidence["initialization_identity"]["paired_seeds"] == [43, 44, 45]
    assert len(evidence["projection_reconstruction"]["levels"]) == 4
    assert (
        evidence["duplicated_frozen_centroid_information"][
            "deterministic_linear_projection_of_frozen_view"
        ]
        is True
    )
    assert len(evidence["lr_warm_start"]["paired_curves"]) == 16
    assert evidence["lr_warm_start"]["monotonic_erasure_supported"] is False
    assert evidence["gradient_probe"]["learned_base_gradient_nonzero"] is True


def test_committed_projection_matches_a_fresh_read_when_cache_is_available() -> None:
    cache = ROOT.parents[1] / (
        "generated/preprocessed/dataset/0fb0a01c70e1/semantic/"
        "kmeans_4x512_602bc598e0/codebooks.pt"
    )
    if not cache.is_file():
        pytest.skip("raw G6 codebook cache is not present")
    evidence = json.loads((ROOT / "evidence/rq1_unexpected_native50m.json").read_text())
    codebooks = torch.load(cache, map_location="cpu", weights_only=True)

    assert (
        projection_reconstruction(codebooks, output_dim=32)
        == evidence["projection_reconstruction"]["levels"]
    )


def test_committed_evidence_matches_a_fresh_collection_when_raw_runs_exist() -> None:
    logs = ROOT.parents[1] / "generated/logs"
    evidence = json.loads((ROOT / "evidence/rq1_unexpected_native50m.json").read_text())
    surface = json.loads((ROOT / "evidence/rq1_rq3_surface_native50m.json").read_text())
    confirmation = json.loads(
        (ROOT / "evidence/rq1_rq3_confirmation_native50m.json").read_text()
    )
    run_rows = list(surface["rq1"]["rows"])
    run_rows.extend(
        row
        for mode in ("random", "content_pca")
        for row in confirmation["rq1"][mode]["rows"]
    )
    required = [
        logs / row["run_name"] / filename
        for row in run_rows
        for filename in ("sweep.log", "training_metadata.json")
    ]
    codebook_path = ROOT.parents[1] / evidence["source_artifacts"]["codebooks"]["path"]
    required.extend([codebook_path, codebook_path.with_name("codes.pt")])
    if not all(path.is_file() for path in required):
        pytest.skip("complete raw G6 RQ1 evidence is not present")

    assert collect_rq1_unexpected_evidence() == evidence
