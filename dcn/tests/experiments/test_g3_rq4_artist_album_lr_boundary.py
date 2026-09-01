from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.g3_pretrained_item_embeddings.launchers import (
    rq4_artist_album_lr_boundary,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import (
    PROJECT_ROOT,
    decode_control_job,
    encode_control_job,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4_artist_album_lr_boundary_ledger import (
    RQ4_ARTIST_ALBUM_LR_BOUNDARY_LEDGER_PATH,
    RQ4_CAPACITY_EXTENSION_SELECTION_SHA256,
    compile_rq4_artist_album_lr_boundary_ledger,
    load_rq4_artist_album_lr_boundary_ledger,
    persist_rq4_artist_album_lr_boundary_ledger,
    validate_rq4_artist_album_lr_boundary_ledger_document,
)


def test_artist_album_lr_boundary_has_exact_three_joint_outward_probes() -> None:
    ledger = compile_rq4_artist_album_lr_boundary_ledger(PROJECT_ROOT)
    factors = (math.sqrt(2.0), 2.0, 2.0 * math.sqrt(2.0))

    assert len(ledger.rows) == 3
    assert (
        ledger.capacity_extension_selection.logical_sha256
        == RQ4_CAPACITY_EXTENSION_SELECTION_SHA256
    )
    for row, factor in zip(ledger.rows, factors, strict=True):
        document = row.to_dict()
        assert document["family_id"] == "rq4_artist_album"
        assert document["representation"]["metadata_dim"] == 64
        assert document["training"]["horizon_epochs"] == 25
        assert row.embedding_learning_rate == pytest.approx(
            0.17783052497147875 * factor
        )
        assert row.deep_learning_rate == pytest.approx(
            0.010430488535480936 / factor
        )


def test_artist_album_lr_boundary_is_canonical_immutable_and_type_exact(
    tmp_path: Path,
) -> None:
    ledger = compile_rq4_artist_album_lr_boundary_ledger(PROJECT_ROOT)
    canonical = PROJECT_ROOT / RQ4_ARTIST_ALBUM_LR_BOUNDARY_LEDGER_PATH
    persist_rq4_artist_album_lr_boundary_ledger(
        canonical, ledger, root=PROJECT_ROOT
    )
    loaded = load_rq4_artist_album_lr_boundary_ledger(
        canonical,
        root=PROJECT_ROOT,
        expected_ledger_sha256=ledger.sha256,
    )
    assert loaded == ledger
    with pytest.raises(ValueError, match="canonical path"):
        persist_rq4_artist_album_lr_boundary_ledger(
            tmp_path / "copy.json", ledger, root=PROJECT_ROOT
        )
    drifted = deepcopy(ledger.to_dict())
    drifted["rows"][0]["training"]["horizon_epochs"] = 25.0
    with pytest.raises(ValueError, match="ledger changed"):
        validate_rq4_artist_album_lr_boundary_ledger_document(
            drifted,
            root=PROJECT_ROOT,
            expected_ledger_sha256=ledger.sha256,
        )


def test_artist_album_lr_boundary_queue_has_exactly_three_jobs(
    tmp_path: Path,
) -> None:
    ledger = compile_rq4_artist_album_lr_boundary_ledger(PROJECT_ROOT)
    path = PROJECT_ROOT / RQ4_ARTIST_ALBUM_LR_BOUNDARY_LEDGER_PATH
    commands = rq4_artist_album_lr_boundary.compile_queue_surface(
        ledger_path=path,
        ledger=ledger,
        state_dir=tmp_path / "queue",
    )

    assert len(commands) == 6
    assert sum("enqueue-run" in command for command in commands) == 3
    assert {
        command[command.index("--run") + 1] for command in commands[2:5]
    } == {row.run_name for row in ledger.rows}


def test_artist_album_lr_boundary_builder_uses_exact_ledger_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = compile_rq4_artist_album_lr_boundary_ledger(PROJECT_ROOT)
    compiled = decode_control_job(
        encode_control_job(ledger, ledger.rows[0].id), ledger
    )
    captured = {}

    def fake_builder(**arguments):
        captured.update(arguments)
        return SimpleNamespace(base_path="unused")

    monkeypatch.setattr(
        rq4_artist_album_lr_boundary, "build_g3_experiment", fake_builder
    )
    rq4_artist_album_lr_boundary.build_training_experiment(
        compiled,
        ledger=ledger,
        feature_data_path=PROJECT_ROOT / "features.parquet",
    )

    training = compiled.job["training"]
    assert captured["embedding_learning_rate"] == training[
        "embedding_learning_rate"
    ]
    assert captured["deep_learning_rate"] == training["deep_learning_rate"]
    assert captured["lr_schedule_horizon_epochs"] == 25
    assert captured["representation"].metadata == ("artist", "album")
    assert captured["representation"].metadata_dim == 64
