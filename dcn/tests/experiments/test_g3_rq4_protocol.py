from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from experiments.g3_pretrained_item_embeddings.launchers.rq4 import (
    build_rq4_training_experiment,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4 import (
    RQ4_METADATA_FAMILIES,
    TransferredHorizonRate,
    compile_rq4_capacity_surface,
    compile_rq4_extra_id_surface,
    compile_rq4_horizon_followup,
)
import experiments.g3_pretrained_item_embeddings.protocol.rq4 as rq4_protocol


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(value) + "\n")


def _logical_document(path: Path, payload: dict[str, object]) -> tuple[Path, str]:
    payload["sha256"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    _write_json(path, payload)
    return path, str(payload["sha256"])


def _fact(
    root: Path, path: Path, *, logical_sha256: str | None = None
) -> dict[str, object]:
    value = {
        "path": str(path.relative_to(root)),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }
    if logical_sha256 is not None:
        value["logical_sha256"] = logical_sha256
    return value


def _distribution(count: int = 1) -> dict[str, object]:
    return {
        "count": count,
        "nonfinite_count": 0,
        "mean": 1.0,
        "standard_deviation": 0.0,
        "minimum": 1.0,
        "maximum": 1.0,
    }


_CATALOG_DIAGNOSTIC_FACTORS = {
    "learned_id": ((), ("weight",), None),
    "frozen_content": (
        ("catalog_content_table", "catalog_projection"),
        (),
        False,
    ),
    "trainable_content": (
        ("catalog_content_table", "catalog_projection"),
        ("content.embedding.weight",),
        True,
    ),
    "id_frozen_content": (
        ("catalog_content_table", "catalog_item_table", "catalog_projection"),
        ("item_embedding.weight",),
        False,
    ),
    "id_trainable_content": (
        ("catalog_content_table", "catalog_item_table", "catalog_projection"),
        ("content.embedding.weight", "item_embedding.weight"),
        True,
    ),
}


def _diagnostic_epoch(
    epoch: int, *, catalog_representation: str = "id_frozen_content"
) -> dict[str, object]:
    extra_components, table_parameters, content_trainable = (
        _CATALOG_DIAGNOSTIC_FACTORS[catalog_representation]
    )
    row_gradients = {
        name: _distribution()
        for name in (
            "all_row_exposure_weighted_norm",
            "conditional_on_active_row_norm",
            "active_row_count",
            "active_row_fraction",
        )
    }
    pretrained_content = {"available": False}
    if content_trainable is not None:
        pretrained_content = {
            "available": True,
            "trainable": content_trainable,
            "drift_l2": {
                name: _distribution(3 if name == "global" else 1)
                for name in ("global", "tail", "mid", "head")
            },
            "cosine_to_initial": {
                name: _distribution(3 if name == "global" else 1)
                for name in ("global", "tail", "mid", "head")
            },
        }
    return {
        "epoch": epoch,
        "training": {
            name: {
                "num_examples": 1,
                "query_norm": _distribution(),
                "positive_logit": _distribution(),
                "negative_logit": _distribution(),
            }
            for name in ("global", "tail", "mid", "head")
        },
        "component_gradient_norms": {
            name: _distribution()
            for name in (
                "catalog_encoder",
                "history_encoder",
                "sequence_model",
                *extra_components,
            )
        },
        "catalog_representation_norm": {
            name: _distribution(3 if name == "global" else 1)
            for name in ("global", "tail", "mid", "head")
        },
        "pretrained_content": pretrained_content,
        "catalog_table_gradient_norms": {
            parameter: {
                scope: row_gradients for scope in ("global", "tail", "mid", "head")
            }
            for parameter in table_parameters
        },
    }


def _integer_reference(values: tuple[int, ...]) -> dict[str, object]:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return {
        "length": len(values),
        "encoding": "canonical-json-integers",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _predecessors(
    root: Path,
    monkeypatch,
    *,
    catalog_representation: str = "id_frozen_content",
) -> tuple[Path, str, Path, str]:
    feature_directory = root / "features"
    feature_directory.mkdir()
    pl.DataFrame(
        {
            "compact_item_id": [1, 2, 3],
            "training_count": [3, 2, 1],
            "artist_compact_ids": [[1, 2], [2], []],
            "album_compact_ids": [[1], [2, 3], []],
        }
    ).write_parquet(feature_directory / "item_features.parquet")
    pl.DataFrame({"uid": [10, 20], "training_history_length": [4, 2]}).write_parquet(
        feature_directory / "training_user_histories.parquet"
    )
    pl.DataFrame({"raw_artist_id": [100, 200]}).write_parquet(
        feature_directory / "artist_vocab.parquet"
    )
    pl.DataFrame({"raw_album_id": [300, 400, 500]}).write_parquet(
        feature_directory / "album_vocab.parquet"
    )
    for name in ("events.parquet", "remap.parquet", "materializer.py"):
        (feature_directory / name).write_bytes(name.encode())
    roles = {
        "events_source": feature_directory / "events.parquet",
        "compact_remap": feature_directory / "remap.parquet",
        "materialization_implementation": feature_directory / "materializer.py",
        "item_features": feature_directory / "item_features.parquet",
        "training_user_histories": feature_directory
        / "training_user_histories.parquet",
        "artist_vocab": feature_directory / "artist_vocab.parquet",
        "album_vocab": feature_directory / "album_vocab.parquet",
    }
    manifest = {
        "schema_version": 1,
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "artifacts": [
            {"role": role, **_fact(root, path)} for role, path in sorted(roles.items())
        ],
        "metadata": {
            "dataset_size": "native-50m",
            "validation_interval_seconds": 604800,
            "num_items": 3,
            "training_rows": 6,
            "training_users": 2,
            "artist_vocab_size": 2,
            "album_vocab_size": 3,
            "artist_unknown_rate": 1 / 3,
            "album_unknown_rate": 1 / 3,
            "artist_max_cardinality": 2,
            "album_max_cardinality": 2,
        },
    }
    manifest_path = root / "feature_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_logical = hashlib.sha256(_canonical(manifest).encode()).hexdigest()
    monkeypatch.setattr(
        rq4_protocol, "NATIVE50_FEATURE_MANIFEST_PATH", "feature_manifest.json"
    )
    monkeypatch.setattr(
        rq4_protocol, "NATIVE50_FEATURE_MANIFEST_SHA256", manifest_logical
    )
    monkeypatch.setattr(
        rq4_protocol, "NATIVE50_FEATURE_MANIFEST_FILE_SHA256", _sha(manifest_path)
    )
    manifest_reference = _fact(root, manifest_path, logical_sha256=manifest_logical)

    rq2_payload = {
        "schema_version": 1,
        "kind": "g3_rq2_content_selection_for_rq3",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "selected_family_id": "rq2_content_concat",
        "selected_history_hidden_dim": 64,
        "selection_resolved": True,
        "feature_manifest": manifest_reference,
        "rows": [{"source_id": "rq2_content_concat:01"}],
    }
    rq2_path, rq2_sha = _logical_document(root / "rq2_selection.json", rq2_payload)

    run_directory = root / "runs/rq3_winner"
    metadata_path = run_directory / "training_metadata.json"
    metrics_path = run_directory / "final_metrics.json"
    diagnostics_path = run_directory / "g3_training_diagnostics.json"
    contract_path = run_directory / "job.json"
    ranking_path = run_directory / "ranking_evidence.pt"
    rankings_path = run_directory / "top_item_rankings.json"
    sweep_path = run_directory / "sweep.log"
    queue_path = root / "rq3_queue/completed/rq3-job.json"
    context_path = root / "ranking_context.pt"
    family_id = next(
        family
        for family, representation in rq4_protocol.RQ3_CATALOG_REPRESENTATIONS.items()
        if representation == catalog_representation
    )
    selected = {
        "row_id": f"{family_id}:04",
        "family_id": family_id,
        "run_name": "rq3_winner",
        "history_hidden_dim": 64,
        "catalog_representation": catalog_representation,
        "embedding_learning_rate": 0.12,
        "deep_learning_rate": 0.03,
        "horizon_epochs": 25,
    }
    _write_json(
        metadata_path,
        {
            "g3_dataset_size": "native-50m",
            "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "training_semantics_revision": 2,
            "g3_representation": {
                "history_representation": "id_content",
                "catalog_representation": catalog_representation,
                "history_hidden_dim": 64,
                "content_gate": "fixed",
                "gate_hidden_dim": None,
                "metadata": [],
                "metadata_dim": None,
                "extra_item_id_dim": None,
            },
        },
    )
    _write_json(metrics_path, {"recall@100": 0.11, "ndcg@100": 0.04})
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_path.write_bytes(b"ranking")
    _write_json(rankings_path, {"rankings": []})
    sweep_path.write_text("training")
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(queue_path, {"id": "rq3-job"})
    context_path.write_bytes(b"context")
    _write_json(
        diagnostics_path,
        {
            "schema_version": 2,
            "frequency_terciles": {
                "num_catalog_items": 3,
                "slices": {
                    "tail": {"num_items": 1, "training_interactions": 1},
                    "mid": {"num_items": 1, "training_interactions": 2},
                    "head": {"num_items": 1, "training_interactions": 3},
                },
            },
            "training_count_reference": _integer_reference((0, 3, 2, 1)),
            "slice_membership_reference": _integer_reference((-1, 2, 1, 0)),
            "content_drift_reference": (
                {"available": False}
                if _CATALOG_DIAGNOSTIC_FACTORS[catalog_representation][2] is None
                else {
                    "available": True,
                    "shape": [3, 128],
                    "dtype": "torch.float32",
                    "sha256": "a" * 64,
                }
            ),
            "epochs": [
                _diagnostic_epoch(
                    epoch, catalog_representation=catalog_representation
                )
                for epoch in range(25)
            ],
        },
    )
    source_job = {
        "id": selected["row_id"],
        "family_id": selected["family_id"],
        "phase": "rq3_catalog_output",
        "stage": "rq3_post_boundary_output_search",
        "role": "search",
        "run_name": selected["run_name"],
        "reused_from": None,
        "source_ledger": None,
        "representation": {
            "id": selected["family_id"],
            "history_representation": "id_content",
            "history_hidden_dim": selected["history_hidden_dim"],
            "catalog_representation": selected["catalog_representation"],
        },
        "dataset": {
            "size": "native-50m",
            "source": "likes",
            "event_limit": 50_000_000,
            "sampling": "none",
            "minimum_user_interactions": 5,
            "validation_interval_seconds": 604800,
            "candidate_catalog": "full",
            "exclude_seen": False,
        },
        "training": {
            "batch_size": 512,
            "seed": 42,
            "embedding_learning_rate": selected["embedding_learning_rate"],
            "deep_learning_rate": selected["deep_learning_rate"],
            "horizon_epochs": selected["horizon_epochs"],
            "validate_every_epoch": True,
            "restore_best_validation_epoch": True,
        },
    }
    source_ledger_path = root / "rq3_source_ledger.json"
    source_ledger_sha = "b" * 64
    _write_json(source_ledger_path, {"row": source_job, "sha256": source_ledger_sha})
    _write_json(
        contract_path,
        {
            "row_id": selected["row_id"],
            "job": source_job,
            "ledger_path": str(source_ledger_path),
            "ledger_sha256": source_ledger_sha,
        },
    )
    monkeypatch.setattr(
        rq4_protocol,
        "_load_rq3_source_ledger",
        lambda path: SimpleNamespace(
            sha256=json.loads(path.read_text())["sha256"],
            final_rq2_evidence_sha256=rq2_sha,
            logical_rows=(
                SimpleNamespace(
                    id=json.loads(path.read_text())["row"]["id"],
                    to_dict=lambda: json.loads(path.read_text())["row"],
                ),
            ),
        ),
    )
    artifacts = {
        "job_contract": _fact(root, contract_path),
        "training_metadata": _fact(root, metadata_path),
        "final_metrics": _fact(root, metrics_path),
        "ranking_evidence": _fact(root, ranking_path),
        "top_item_rankings": _fact(root, rankings_path),
        "training_diagnostics": _fact(root, diagnostics_path),
        "sweep_log": _fact(root, sweep_path),
    }
    rq3_payload = {
        "schema_version": 1,
        "kind": "g3_rq3_catalog_selection_for_rq4",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "selection_resolved": True,
        "source_rq2_selection": _fact(root, rq2_path, logical_sha256=rq2_sha),
        "feature_manifest": manifest_reference,
        "ranking_context": _fact(root, context_path),
        "selected": {
            **selected,
            "queue_job": _fact(root, queue_path),
            "artifacts": artifacts,
        },
    }
    rq3_path, rq3_sha = _logical_document(root / "rq3_selection.json", rq3_payload)
    return rq2_path, rq2_sha, rq3_path, rq3_sha


def _capacity_surface(root: Path, monkeypatch):
    rq2_path, rq2_sha, rq3_path, rq3_sha = _predecessors(root, monkeypatch)
    return compile_rq4_capacity_surface(
        root=root,
        rq2_selection_path=rq2_path,
        expected_rq2_selection_sha256=rq2_sha,
        rq3_selection_path=rq3_path,
        expected_rq3_selection_sha256=rq3_sha,
    )


def _rates() -> tuple[TransferredHorizonRate, ...]:
    return (
        TransferredHorizonRate(15, 0.10, 0.02),
        TransferredHorizonRate(25, 0.12, 0.03),
        TransferredHorizonRate(40, 0.14, 0.04),
    )


def _horizon_surface(root: Path, monkeypatch):
    capacity = _capacity_surface(root, monkeypatch)
    followup = compile_rq4_horizon_followup(
        capacity,
        selected_capacities={family: 32 for family in RQ4_METADATA_FAMILIES},
        transferred_horizon_rates={
            family: _rates() for family in RQ4_METADATA_FAMILIES
        },
    )
    return capacity, followup


def _metadata_winner_payload(capacity, followup) -> dict[str, object]:
    row = followup.rows_by_family["rq4_artist_album"][1]
    return {
        "schema_version": 1,
        "kind": "g3_rq4_metadata_winner_for_extra_id",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "selection_resolved": True,
        "rq2_selection_sha256": capacity.predecessor.rq2_selection_sha256,
        "rq3_selection_sha256": capacity.predecessor.rq3_selection_sha256,
        "initial_ledger_sha256": "1" * 64,
        "horizon_ledger_sha256": "2" * 64,
        "selected_family_id": "rq4_artist_album",
        "selected_metadata_dim": 32,
        "selected": {
            "row_id": row.id,
            "family_id": row.family_id,
            "metadata_dim": row.metadata_dim,
            "embedding_learning_rate": row.embedding_learning_rate,
            "deep_learning_rate": row.deep_learning_rate,
            "horizon_epochs": row.horizon_epochs,
            "ledger_sha256": "2" * 64,
            "job": {
                "id": row.id,
                "family_id": row.family_id,
                "representation": {
                    "metadata": list(row.metadata),
                    "metadata_dim": row.metadata_dim,
                },
                "training": {
                    "embedding_learning_rate": row.embedding_learning_rate,
                    "deep_learning_rate": row.deep_learning_rate,
                    "horizon_epochs": row.horizon_epochs,
                },
            },
            "artifacts": {"final_metrics": {"sha256": "3" * 64}},
            "metric_provenance": {"recomputed_from_ranking_evidence": True},
        },
        "family_boundaries": {
            family: {"extension_required": False}
            for family in RQ4_METADATA_FAMILIES
        },
    }


def _extra_surface(root: Path, monkeypatch):
    capacity, followup = _horizon_surface(root, monkeypatch)
    winner_payload = _metadata_winner_payload(capacity, followup)
    winner_path, winner_sha = _logical_document(
        root / "rq4_winner.json", winner_payload
    )
    extra = compile_rq4_extra_id_surface(
        root=root,
        capacity_surface=capacity,
        horizon_followup=followup,
        winner_selection_path=winner_path,
        expected_winner_selection_sha256=winner_sha,
    )
    return capacity, followup, extra


def _rewrite_rq3_diagnostics(
    root: Path,
    rq3_path: Path,
    mutation,
) -> str:
    rq3 = json.loads(rq3_path.read_text())
    reference = rq3["selected"]["artifacts"]["training_diagnostics"]
    diagnostics_path = root / reference["path"]
    diagnostics = json.loads(diagnostics_path.read_text())
    mutation(diagnostics)
    _write_json(diagnostics_path, diagnostics)
    rq3["selected"]["artifacts"]["training_diagnostics"] = _fact(root, diagnostics_path)
    rq3.pop("sha256")
    rq3["sha256"] = hashlib.sha256(_canonical(rq3).encode()).hexdigest()
    _write_json(rq3_path, rq3)
    return str(rq3["sha256"])


def test_capacity_surface_is_equal_budget_and_binds_training_only_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    surface = _capacity_surface(tmp_path, monkeypatch)

    assert tuple(surface.rows_by_family) == RQ4_METADATA_FAMILIES
    assert surface.predecessor.history_hidden_dim == 64
    assert surface.predecessor.catalog_representation == "id_frozen_content"
    assert surface.metadata_identity.training_count_sha256
    assert surface.metadata_identity.artist_membership_sha256
    assert surface.metadata_identity.album_membership_sha256
    assert surface.extra_id_rows == ()
    signatures = []
    for family, rows in surface.rows_by_family.items():
        assert len(rows) == 9
        assert [row.metadata_dim for row in rows] == [16] * 3 + [32] * 3 + [64] * 3
        expected_metadata = {
            "rq4_artist": ("artist",),
            "rq4_album": ("album",),
            "rq4_artist_album": ("artist", "album"),
        }[family]
        assert all(row.metadata == expected_metadata for row in rows)
        signatures.append(
            tuple(
                (
                    row.embedding_learning_rate,
                    row.deep_learning_rate,
                    row.horizon_epochs,
                    row.metadata_dim,
                )
                for row in rows
            )
        )
    assert signatures[0] == signatures[1] == signatures[2]


@pytest.mark.parametrize("catalog_representation", _CATALOG_DIAGNOSTIC_FACTORS)
def test_capacity_surface_accepts_each_exact_rq3_catalog_diagnostic_contract(
    tmp_path: Path, monkeypatch, catalog_representation: str
) -> None:
    rq2_path, rq2_sha, rq3_path, rq3_sha = _predecessors(
        tmp_path,
        monkeypatch,
        catalog_representation=catalog_representation,
    )

    surface = compile_rq4_capacity_surface(
        root=tmp_path,
        rq2_selection_path=rq2_path,
        expected_rq2_selection_sha256=rq2_sha,
        rq3_selection_path=rq3_path,
        expected_rq3_selection_sha256=rq3_sha,
    )

    assert surface.predecessor.catalog_representation == catalog_representation


def test_horizon_followup_preserves_three_opportunities_for_every_family(
    tmp_path: Path, monkeypatch
) -> None:
    capacity = _capacity_surface(tmp_path, monkeypatch)
    selected = {
        "rq4_artist": 16,
        "rq4_album": 32,
        "rq4_artist_album": 64,
    }
    followup = compile_rq4_horizon_followup(
        capacity,
        selected_capacities=selected,
        transferred_horizon_rates={
            family: _rates() for family in RQ4_METADATA_FAMILIES
        },
    )

    for family, rows in followup.rows_by_family.items():
        assert len(rows) == 3
        assert [row.horizon_epochs for row in rows] == [15, 25, 40]
        assert all(row.metadata_dim == selected[family] for row in rows)
        assert len((*capacity.rows_by_family[family], *rows)) == 12


def test_launcher_builds_only_an_authenticated_rq4_row(
    tmp_path: Path, monkeypatch
) -> None:
    surface = _capacity_surface(tmp_path, monkeypatch)
    row = surface.rows_by_family["rq4_artist_album"][0]

    experiment = build_rq4_training_experiment(surface, row, root=tmp_path)

    assert experiment.run_name == row.run_name
    assert experiment.representation.history_representation == "id_content"
    assert experiment.representation.catalog_representation == "id_frozen_content"
    assert experiment.representation.metadata == ("artist", "album")
    assert experiment.representation.metadata_dim == 16


def test_launcher_rechecks_frozen_predecessor_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    surface = _capacity_surface(tmp_path, monkeypatch)
    row = surface.rows_by_family["rq4_artist"][0]
    rq3_path = tmp_path / surface.predecessor.rq3_selection_path
    rq3_path.write_text(rq3_path.read_text() + " ")

    with pytest.raises(ValueError, match="predecessor"):
        build_rq4_training_experiment(surface, row, root=tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda diagnostics: diagnostics.update(epochs=[]),
        lambda diagnostics: diagnostics.update(training_count_reference={}),
        lambda diagnostics: diagnostics["training_count_reference"].update(
            sha256="0" * 64
        ),
        lambda diagnostics: diagnostics["epochs"][0]["component_gradient_norms"][
            "catalog_encoder"
        ].update(count=0),
    ],
    ids=(
        "empty_epochs",
        "empty_reference",
        "tampered_reference",
        "empty_epoch_payload",
    ),
)
def test_capacity_surface_reuses_substantive_rq3_diagnostics_authentication(
    tmp_path: Path, monkeypatch, mutation
) -> None:
    rq2_path, rq2_sha, rq3_path, _ = _predecessors(tmp_path, monkeypatch)
    rq3_sha = _rewrite_rq3_diagnostics(tmp_path, rq3_path, mutation)

    with pytest.raises(ValueError, match="diagnostics"):
        compile_rq4_capacity_surface(
            root=tmp_path,
            rq2_selection_path=rq2_path,
            expected_rq2_selection_sha256=rq2_sha,
            rq3_selection_path=rq3_path,
            expected_rq3_selection_sha256=rq3_sha,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda diagnostics: diagnostics.update(
            content_drift_reference={"available": False}
        ),
        lambda diagnostics: diagnostics["epochs"][0][
            "component_gradient_norms"
        ].pop("catalog_item_table"),
        lambda diagnostics: diagnostics["epochs"][0]["pretrained_content"].update(
            trainable=True
        ),
        lambda diagnostics: diagnostics["epochs"][0][
            "catalog_table_gradient_norms"
        ].update(
            weight=diagnostics["epochs"][0]["catalog_table_gradient_norms"][
                "item_embedding.weight"
            ]
        ),
    ],
    ids=("drift_reference", "component_set", "content_state", "table_parameters"),
)
def test_capacity_surface_rejects_rehashed_catalog_semantic_mutations(
    tmp_path: Path, monkeypatch, mutation
) -> None:
    rq2_path, rq2_sha, rq3_path, _ = _predecessors(tmp_path, monkeypatch)
    rq3_sha = _rewrite_rq3_diagnostics(tmp_path, rq3_path, mutation)

    with pytest.raises(ValueError, match="diagnostics"):
        compile_rq4_capacity_surface(
            root=tmp_path,
            rq2_selection_path=rq2_path,
            expected_rq2_selection_sha256=rq2_sha,
            rq3_selection_path=rq3_path,
            expected_rq3_selection_sha256=rq3_sha,
        )


@pytest.mark.parametrize(
    "catalog_representation",
    (
        "frozen_content",
        "trainable_content",
        "id_frozen_content",
        "id_trainable_content",
    ),
)
def test_capacity_surface_rejects_unavailable_content_for_content_catalogs(
    tmp_path: Path, monkeypatch, catalog_representation: str
) -> None:
    rq2_path, rq2_sha, rq3_path, _ = _predecessors(
        tmp_path,
        monkeypatch,
        catalog_representation=catalog_representation,
    )
    rq3_sha = _rewrite_rq3_diagnostics(
        tmp_path,
        rq3_path,
        lambda diagnostics: diagnostics["epochs"][0].update(
            pretrained_content={"available": False}
        ),
    )

    with pytest.raises(ValueError, match="diagnostics"):
        compile_rq4_capacity_surface(
            root=tmp_path,
            rq2_selection_path=rq2_path,
            expected_rq2_selection_sha256=rq2_sha,
            rq3_selection_path=rq3_path,
            expected_rq3_selection_sha256=rq3_sha,
        )


@pytest.mark.parametrize("stage", ["capacity", "horizon", "extra_id"])
def test_launcher_recompiles_the_exact_staged_surface(
    tmp_path: Path, monkeypatch, stage: str
) -> None:
    if stage == "capacity":
        surface = _capacity_surface(tmp_path, monkeypatch)
        family = "rq4_artist"
        original = surface.rows_by_family[family][0]
        row = replace(original, run_name="tampered-capacity")
        surface.rows_by_family[family] = (row, *surface.rows_by_family[family][1:])
    elif stage == "horizon":
        _, surface = _horizon_surface(tmp_path, monkeypatch)
        family = "rq4_album"
        original = surface.rows_by_family[family][0]
        row = replace(original, deep_learning_rate=0.031)
        surface.rows_by_family[family] = (row, *surface.rows_by_family[family][1:])
    else:
        _, _, surface = _extra_surface(tmp_path, monkeypatch)
        original = surface.rows[0]
        row = replace(original, extra_item_id_dim=original.extra_item_id_dim + 1)
        surface = replace(surface, rows=(row, *surface.rows[1:]))

    with pytest.raises(ValueError, match="staged surface"):
        build_rq4_training_experiment(surface, row, root=tmp_path)


def test_extra_id_launcher_rechecks_winner_bytes(tmp_path: Path, monkeypatch) -> None:
    _, _, surface = _extra_surface(tmp_path, monkeypatch)
    winner_path = tmp_path / surface.winner_selection_path
    winner_path.write_text(winner_path.read_text() + " ")

    with pytest.raises(ValueError, match="winner"):
        build_rq4_training_experiment(surface, surface.rows[0], root=tmp_path)


@pytest.mark.parametrize("target", ["feature", "rq2", "rq3_artifact"])
def test_capacity_surface_fails_closed_on_predecessor_or_metadata_drift(
    tmp_path: Path, monkeypatch, target: str
) -> None:
    rq2_path, rq2_sha, rq3_path, rq3_sha = _predecessors(tmp_path, monkeypatch)
    if target == "feature":
        (tmp_path / "features/artist_vocab.parquet").write_bytes(b"changed")
    elif target == "rq2":
        rq2_path.write_text(rq2_path.read_text() + " ")
    else:
        rq3 = json.loads(rq3_path.read_text())
        metadata_path = (
            tmp_path / rq3["selected"]["artifacts"]["training_metadata"]["path"]
        )
        metadata_path.write_text(metadata_path.read_text() + " ")

    with pytest.raises(ValueError):
        compile_rq4_capacity_surface(
            root=tmp_path,
            rq2_selection_path=rq2_path,
            expected_rq2_selection_sha256=rq2_sha,
            rq3_selection_path=rq3_path,
            expected_rq3_selection_sha256=rq3_sha,
        )


def test_extra_id_surface_is_deferred_until_a_hash_bound_metadata_winner(
    tmp_path: Path, monkeypatch
) -> None:
    capacity = _capacity_surface(tmp_path, monkeypatch)
    followup = compile_rq4_horizon_followup(
        capacity,
        selected_capacities={family: 32 for family in RQ4_METADATA_FAMILIES},
        transferred_horizon_rates={
            family: _rates() for family in RQ4_METADATA_FAMILIES
        },
    )
    winner_payload = _metadata_winner_payload(capacity, followup)
    winner_path, winner_sha = _logical_document(
        tmp_path / "rq4_winner.json", winner_payload
    )

    extra = compile_rq4_extra_id_surface(
        root=tmp_path,
        capacity_surface=capacity,
        horizon_followup=followup,
        winner_selection_path=winner_path,
        expected_winner_selection_sha256=winner_sha,
    )

    assert len(extra.rows) == 12
    assert all(row.matched_metadata_family == "rq4_artist_album" for row in extra.rows)
    assert [row.matched_metadata_dim for row in extra.rows[:9]] == [16] * 3 + [
        32
    ] * 3 + [64] * 3
    assert all(row.extra_item_id_dim > 0 for row in extra.rows)
    assert all(row.parameter_mismatch_fraction < 0.01 for row in extra.rows)
    assert all(math.isfinite(row.parameter_mismatch_fraction) for row in extra.rows)

    tampered = _metadata_winner_payload(capacity, followup)
    tampered["selected"]["job"]["training"]["deep_learning_rate"] = 0.9
    tampered_path, tampered_sha = _logical_document(
        tmp_path / "rq4_winner_tampered.json", tampered
    )
    with pytest.raises(ValueError, match="exact selected metadata winner"):
        compile_rq4_extra_id_surface(
            root=tmp_path,
            capacity_surface=capacity,
            horizon_followup=followup,
            winner_selection_path=tampered_path,
            expected_winner_selection_sha256=tampered_sha,
        )
