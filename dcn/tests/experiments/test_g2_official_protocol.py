import json
import math
from pathlib import Path
import subprocess

import numpy as np
import polars as pl
import pytest

from experiments.g2_esasrec.official import check_protocol, protocol, provenance


RECTOOLS_PYTHON = Path("/home/sashanovak/envs/esasrec/bin/python")
EXPECTED_RECTOOLS_SOURCES = {
    "rectools_init",
    "rectools_columns",
    "rectools_dataset",
    "rectools_identifiers",
    "rectools_interactions",
    "rectools_utils_array_set_ops",
    "rectools_utils_indexing",
    "rectools_utils_misc",
    "rectools_model_base",
    "rectools_item_net",
    "rectools_transformer_base",
    "rectools_data_preparator",
    "rectools_transformer_constants",
    "rectools_negative_sampler",
    "rectools_net_blocks",
    "rectools_ligr",
    "rectools_lightning",
    "rectools_torch_backbone",
    "rectools_similarity",
    "rectools_sasrec",
    "rectools_rank_init",
    "rectools_rank_base",
    "rectools_rank_torch",
}


def _split() -> protocol.Split:
    return protocol.Split(
        train=pl.DataFrame(
            {
                "uid": [1, 1, 2, 3],
                "compact_item_id": [2, 3, 2, 4],
                "timestamp": [1, 2, 1, 1],
            }
        ),
        validation=pl.DataFrame(
            {
                "uid": [1, 1, 2, 4],
                "compact_item_id": [4, 99, 3, 2],
                "timestamp": [10, 11, 10, 10],
            }
        ),
        cutoff=10,
        catalog=np.array([2, 3, 4]),
    )


def test_protocol_scores_only_users_with_history_and_catalog_relevance() -> None:
    split = _split()

    relevant = protocol.relevance(split)
    histories = protocol.query_histories(split, max_seq_len=1)
    users = protocol.evaluable_users(histories, relevant)
    metrics = protocol.score_rankings(
        {1: [3, 4, 2]}, relevant, users, (1, 3), split.catalog_size
    )

    assert relevant == {1: {4}, 2: {3}, 4: {2}}
    assert histories == {1: [3]}
    assert users == [1]
    assert metrics["recall@1"] == 0
    assert metrics["recall@3"] == 1
    assert metrics["ndcg@3"] == pytest.approx(1 / math.log2(3))
    assert metrics["mrr@3"] == 0.5
    assert metrics["coverage@3"] == 1
    assert metrics["num_users"] == 1


def test_official_protocol_accepts_only_native_50m(tmp_path: Path) -> None:
    assert protocol.dataset_dir(tmp_path) == (
        tmp_path / "datasets" / "yambda" / "50m_like_core5_knownitems"
    )


def test_candidate_catalog_evidence_requires_the_complete_mapped_catalog() -> None:
    split = _split()

    evidence = protocol.candidate_catalog_evidence(split, np.array([4, 2, 3]))

    assert evidence == {
        "catalog_size": 3,
        "model_candidate_catalog_size": 3,
        "train_catalog_size": 3,
        "mapped_items_absent_from_training": 0,
        "candidate_catalog_sha256": (
            "b93e20304c02fe6f25cae506ae46e17dcef640e1cf5473118db8c2d554a9443e"
        ),
    }
    with pytest.raises(ValueError, match="candidate catalog differs"):
        protocol.candidate_catalog_evidence(split, np.array([2, 3]))


def test_source_manifest_records_content_hashes(tmp_path: Path) -> None:
    source = tmp_path / "ligr.py"
    source.write_text("local oracle fixture\n")

    manifest = provenance.source_manifest({"ligr": source})

    assert manifest == {
        "ligr": {
            "path": str(source.resolve()),
            "sha256": "cbe6b7cc2002180f15ab09f57dc919da224b8b9ecc4936b209139bb8eb48cdaf",
        }
    }


def test_official_provenance_enumerates_every_rectools_execution_source() -> None:
    assert set(provenance.RECTOOLS_SOURCE_MODULES) == EXPECTED_RECTOOLS_SOURCES
    assert set(provenance.RECTOOLS_SOURCE_SHA256) == EXPECTED_RECTOOLS_SOURCES


@pytest.mark.skipif(not RECTOOLS_PYTHON.is_file(), reason="RecTools environment absent")
def test_official_rectools_source_hashes_match_the_pinned_019_installation() -> None:
    script = """
import json
from experiments.g2_esasrec.official import run_official
print(json.dumps(run_official._source_provenance()))
"""
    completed = subprocess.run(
        [str(RECTOOLS_PYTHON), "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    manifest = json.loads(completed.stdout)

    assert set(manifest) == EXPECTED_RECTOOLS_SOURCES | {
        "catalog_data",
        "runner",
        "protocol",
        "provenance",
    }
    assert {
        name: manifest[name]["sha256"] for name in EXPECTED_RECTOOLS_SOURCES
    } == provenance.RECTOOLS_SOURCE_SHA256


def test_protocol_check_pins_the_rectools_source_contract() -> None:
    assert check_protocol.EXPECTED["rectools_source_contract_sha256"] == (
        provenance.rectools_source_contract_sha256()
    )


@pytest.mark.skipif(not RECTOOLS_PYTHON.is_file(), reason="RecTools environment absent")
def test_official_environment_versions_match_the_pinned_contract() -> None:
    script = """
import importlib.metadata
import json
import platform
from experiments.g2_esasrec.official import provenance
print(json.dumps({
    "python": platform.python_version(),
    "packages": {
        name: importlib.metadata.version(name)
        for name in provenance.OFFICIAL_PACKAGE_VERSIONS
    },
}))
"""
    completed = subprocess.run(
        [str(RECTOOLS_PYTHON), "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )

    assert json.loads(completed.stdout) == {
        "python": provenance.OFFICIAL_PYTHON_VERSION,
        "packages": provenance.OFFICIAL_PACKAGE_VERSIONS,
    }
