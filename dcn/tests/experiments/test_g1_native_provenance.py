from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis import native_500m_provenance


def _selection() -> dict:
    digest = "a" * 64
    return {
        "source_digest": digest,
        "source_id": digest[:12],
        "source_artifacts": 42,
        "winner_run": (
            "g1_transfer_batchscale_b1280_e0p001_d0p002_cap60_ts2_r2_50m"
        ),
        "embedding_lr": "0.001",
        "deep_lr": "0.002",
    }


def _target(tmp_path: Path) -> Path:
    directory = (
        tmp_path
        / "g1_transfer_selected_native50_aaaaaaaaaaaa_e0p001_d0p002_"
        "cap40_ts2_r2_500m"
    )
    directory.mkdir()
    (directory / "training_metadata.json").write_text(
        json.dumps({"selection_resolved": True})
    )
    (directory / "final_metrics.json").write_text(
        json.dumps({"recall@100": 0.12})
    )
    return directory


def test_manifest_binds_full_selector_and_exact_evidence_bytes(
    tmp_path: Path,
) -> None:
    directory = _target(tmp_path)
    manifest_path = tmp_path / "provenance.json"
    selection = _selection()
    native_500m_provenance.write_manifest(
        directory,
        selection,
        manifest_path=manifest_path,
    )

    manifest = native_500m_provenance.validate(
        directory,
        selection=selection,
        manifest_path=manifest_path,
    )

    assert manifest["selector"]["source_digest"] == "a" * 64
    assert manifest["target"]["run"] == directory.name

    (directory / "final_metrics.json").write_text(
        json.dumps({"recall@100": 0.13})
    )
    with pytest.raises(ValueError, match="hashes"):
        native_500m_provenance.validate(
            directory,
            selection=selection,
            manifest_path=manifest_path,
        )


def test_manifest_rejects_copied_or_renamed_target(tmp_path: Path) -> None:
    directory = _target(tmp_path)
    manifest_path = tmp_path / "provenance.json"
    selection = _selection()
    native_500m_provenance.write_manifest(
        directory,
        selection,
        manifest_path=manifest_path,
    )
    copied = tmp_path / directory.name.replace("cap40", "cap60")
    shutil.copytree(directory, copied)

    with pytest.raises(ValueError, match="target"):
        native_500m_provenance.validate(
            copied,
            selection=selection,
            manifest_path=manifest_path,
        )
