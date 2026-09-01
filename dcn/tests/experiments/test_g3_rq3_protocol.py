import hashlib
import json
import math
from pathlib import Path

import pytest
import polars as pl

from experiments.g3_pretrained_item_embeddings.launchers.rq3 import (
    build_rq3_training_experiment,
)
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    RQ3_OUTPUT_FAMILY_IDS,
    compile_rq3_output_surface,
)
import experiments.g3_pretrained_item_embeddings.protocol.rq3 as rq3_protocol


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(value) + "\n")


def _fact(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _distribution(count: int = 1) -> dict[str, object]:
    return {
        "count": count,
        "nonfinite_count": 0,
        "mean": 1.0,
        "standard_deviation": 0.0,
        "minimum": 1.0,
        "maximum": 1.0,
    }


def _diagnostic_epoch(epoch: int, *, schema_version: int) -> dict[str, object]:
    entry = {
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
            for name in ("catalog_encoder", "history_encoder", "sequence_model")
        },
        "catalog_representation_norm": {
            name: _distribution(3 if name == "global" else 1)
            for name in ("global", "tail", "mid", "head")
        },
        "pretrained_content": {"available": False},
    }
    if schema_version == 2:
        row_gradients = {
            name: _distribution()
            for name in (
                "all_row_exposure_weighted_norm",
                "conditional_on_active_row_norm",
                "active_row_count",
                "active_row_fraction",
            )
        }
        entry["catalog_table_gradient_norms"] = {
            "weight": {
                scope: row_gradients for scope in ("global", "tail", "mid", "head")
            }
        }
    return entry


def _integer_reference(values: tuple[int, ...]) -> dict[str, object]:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return {
        "length": len(values),
        "encoding": "canonical-json-integers",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _feature_fixture(root: Path, monkeypatch) -> dict[str, object]:
    directory = root / "features"
    directory.mkdir()
    pl.DataFrame(
        {
            "compact_item_id": [1, 2, 3],
            "training_count": [1, 2, 3],
            "artist_compact_ids": [[1], [1], [1]],
            "album_compact_ids": [[1], [1], [1]],
        }
    ).write_parquet(directory / "item_features.parquet")
    pl.DataFrame({"uid": [1], "training_history_length": [3]}).write_parquet(
        directory / "training_user_histories.parquet"
    )
    pl.DataFrame({"raw_artist_id": [10]}).write_parquet(
        directory / "artist_vocab.parquet"
    )
    pl.DataFrame({"raw_album_id": [20]}).write_parquet(
        directory / "album_vocab.parquet"
    )
    for name in ("events.parquet", "remap.parquet", "materializer.py"):
        (directory / name).write_bytes(name.encode())
    roles = {
        "events_source": directory / "events.parquet",
        "compact_remap": directory / "remap.parquet",
        "materialization_implementation": directory / "materializer.py",
        "item_features": directory / "item_features.parquet",
        "training_user_histories": directory / "training_user_histories.parquet",
        "artist_vocab": directory / "artist_vocab.parquet",
        "album_vocab": directory / "album_vocab.parquet",
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
            "training_users": 1,
            "artist_vocab_size": 1,
            "album_vocab_size": 1,
            "artist_unknown_rate": 0.0,
            "album_unknown_rate": 0.0,
            "artist_max_cardinality": 1,
            "album_max_cardinality": 1,
        },
    }
    manifest_path = root / "feature_manifest.json"
    _write_json(manifest_path, manifest)
    logical = hashlib.sha256(_canonical(manifest).encode()).hexdigest()
    monkeypatch.setattr(
        rq3_protocol, "NATIVE50_FEATURE_MANIFEST_PATH", "feature_manifest.json"
    )
    monkeypatch.setattr(rq3_protocol, "NATIVE50_FEATURE_MANIFEST_SHA256", logical)
    monkeypatch.setattr(
        rq3_protocol, "NATIVE50_FEATURE_MANIFEST_FILE_SHA256", _sha(manifest_path)
    )
    return _fact(root, manifest_path) | {"logical_sha256": logical}


def _selection_fixture(
    root: Path,
    monkeypatch,
    *,
    dataset_mutation: tuple[str, object] | None = None,
    invariant_mutation: tuple[str, object] | None = None,
    indirect_feature_binding: bool = False,
    diagnostics_schema_version: int = 1,
) -> tuple[Path, str]:
    monkeypatch.setitem(
        rq3_protocol._RQ2_REUSE_SOURCE_VALIDATORS,
        "test_rq2_evidence",
        lambda *args, **kwargs: None,
    )
    feature_manifest = _feature_fixture(root, monkeypatch)
    ledger_path = root / "evidence/rq2_ledger.json"
    evidence_path = root / "evidence/rq2_results.json"
    rows = []
    evidence_runs = []
    coordinates = (
        ("rq2_content_concat:01", "rq2_a", 0.11, 0.021, 25),
        ("rq2_content_concat:02", "rq2_b", 0.09, 0.028, 15),
    )
    for row_id, run_name, embedding_rate, deep_rate, horizon in coordinates:
        rows.append(
            {
                "id": row_id,
                "family_id": "rq2_content_concat",
                "run_name": run_name,
                "dataset": {
                    "candidate_catalog": "full",
                    "event_limit": 50_000_000,
                    "exclude_seen": False,
                    "minimum_user_interactions": 5,
                    "sampling": "none",
                    "size": "native-50m",
                    "source": "likes",
                    "validation_interval_seconds": 604800,
                },
                "representation": {
                    "id": "rq2_content_concat",
                    "history": "learned_item_id_plus_frozen_content",
                    "catalog": "learned_item_id",
                    "history_hidden_dim": 64,
                    "content_trainable": False,
                    "content_width": 128,
                    "separate_history_catalog_tables": True,
                },
                "training": {
                    "batch_size": 512,
                    "seed": 42,
                    "embedding_learning_rate": embedding_rate,
                    "deep_learning_rate": deep_rate,
                    "horizon_epochs": horizon,
                    "validate_every_epoch": True,
                    "restore_best_validation_epoch": True,
                },
            }
        )
    if dataset_mutation is not None:
        rows[0]["dataset"][dataset_mutation[0]] = dataset_mutation[1]
    ledger_payload = {
        "schema_version": 1,
        "kind": "test_rq2_ledger",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "inputs": {
            "feature_manifest": {
                "kind": "native50m_features",
                "path": feature_manifest["path"],
                "sha256": feature_manifest["logical_sha256"],
            }
        },
        "rows": rows,
    }
    if indirect_feature_binding:
        predecessor_path = root / "evidence/rq2_predecessor_ledger.json"
        predecessor = {
            "schema_version": 1,
            "kind": "test_rq2_predecessor_ledger",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": ledger_payload["inputs"],
            "rows": [],
        }
        predecessor["sha256"] = hashlib.sha256(
            _canonical(predecessor).encode()
        ).hexdigest()
        _write_json(predecessor_path, predecessor)
        ledger_payload["inputs"] = {
            "preselection_ledger": {
                "kind": "test_rq2_predecessor_ledger",
                "path": str(predecessor_path.relative_to(root)),
                "sha256": predecessor["sha256"],
            }
        }
    ledger_payload["sha256"] = hashlib.sha256(
        _canonical(ledger_payload).encode()
    ).hexdigest()
    _write_json(ledger_path, ledger_payload)

    selection_rows = []
    for row in rows:
        run_directory = root / "runs" / row["run_name"]
        contract_path = run_directory / "job.json"
        metadata_path = run_directory / "training_metadata.json"
        metrics_path = run_directory / "final_metrics.json"
        _write_json(
            contract_path,
            {
                "row_id": row["id"],
                "job": row,
                "ledger_path": str(ledger_path),
                "ledger_sha256": ledger_payload["sha256"],
            },
        )
        training = row["training"]
        representation = row["representation"]
        _write_json(
            metadata_path,
            {
                "batch_size": 512,
                "seed": 42,
                "embedding_learning_rate": training["embedding_learning_rate"],
                "deep_learning_rate": training["deep_learning_rate"],
                "lr_schedule_horizon_epochs": training["horizon_epochs"],
                "epochs_trained": training["horizon_epochs"],
                "stopped_epoch": training["horizon_epochs"],
                "lr_horizon_complete": True,
                "selection_resolved": True,
                "g3_dataset_size": "native-50m",
                "g3_protocol_sha256": APPROVED_PROTOCOL_SHA256,
                "training_semantics_revision": 2,
                "g3_representation": {
                    "history_representation": "id_content",
                    "catalog_representation": "learned_id",
                    "history_hidden_dim": representation["history_hidden_dim"],
                    "content_gate": "fixed",
                    "gate_hidden_dim": None,
                    "metadata": [],
                    "metadata_dim": None,
                    "extra_item_id_dim": None,
                },
                "transfer_invariants": {
                    "dataset_size": "50m",
                    "user_sample": None,
                    "event_type_filter": "like",
                    "min_item_interactions_per_item": 5,
                    "drop_unmapped_items": True,
                    "validation_interval_seconds": 604800,
                    "day_range": {"start_day": 0, "end_day": 300},
                    "window": "next_item",
                    "evaluation_catalog": "all",
                    "exclude_seen_from_evaluation": False,
                    "eval_ks": [10, 50, 100],
                    "selection_k": 100,
                    "eval_max_users": 20000,
                    "eval_every_n_epochs": 1,
                    "negative_sampling": "random",
                    "num_in_batch_negatives": 512,
                    "logq_correction": "yi2019",
                    "random_negative_fraction": 0.5,
                    "logq_alpha": 0.01,
                    "correct_positive_logq": False,
                    "mask_false_negatives": False,
                    "exclude_own_group_negatives": False,
                    "dense_random_negative_scores": True,
                    "restore_best_weights": True,
                    "adaptive_schedule_early_stopping": False,
                    "lr_schedule_horizon_epochs": training["horizon_epochs"],
                    "lr_schedule": {
                        "cycles": 1,
                        "min_lr_fraction": 0.0,
                        "optimizer_group_scope": "both",
                        "power_exponent": -0.51,
                        "power_transition_tokens": None,
                        "shape": "linear",
                        "timescale_fraction": None,
                        "timescale_steps": None,
                        "warmup_fraction": 0.0,
                    },
                },
            },
        )
        if invariant_mutation is not None and row is rows[0]:
            metadata = json.loads(metadata_path.read_text())
            metadata["transfer_invariants"][invariant_mutation[0]] = invariant_mutation[
                1
            ]
            _write_json(metadata_path, metadata)
        _write_json(metrics_path, {"recall@100": 0.1, "ndcg@100": 0.03})
        diagnostics_path = run_directory / "g3_training_diagnostics.json"
        diagnostics = {
            "schema_version": 1,
            "frequency_terciles": {
                "num_catalog_items": 3,
                "slices": {
                    "tail": {"num_items": 1, "training_interactions": 1},
                    "mid": {"num_items": 1, "training_interactions": 2},
                    "head": {"num_items": 1, "training_interactions": 3},
                },
            },
            "content_drift_reference": {"available": False},
            "epochs": [
                _diagnostic_epoch(epoch, schema_version=diagnostics_schema_version)
                for epoch in range(training["horizon_epochs"])
            ],
        }
        diagnostics["schema_version"] = diagnostics_schema_version
        if diagnostics_schema_version == 2:
            diagnostics["training_count_reference"] = _integer_reference((0, 1, 2, 3))
            diagnostics["slice_membership_reference"] = _integer_reference(
                (-1, 0, 1, 2)
            )
        _write_json(diagnostics_path, diagnostics)
        artifacts = {
            "job_contract": _fact(root, contract_path),
            "training_metadata": _fact(root, metadata_path),
            "final_metrics": _fact(root, metrics_path),
            "training_diagnostics": _fact(root, diagnostics_path),
        }
        evidence_runs.append(
            {
                "row_id": row["id"],
                "run_name": row["run_name"],
                "family_id": row["family_id"],
                "capacity": representation["history_hidden_dim"],
                "embedding_learning_rate": training["embedding_learning_rate"],
                "deep_learning_rate": training["deep_learning_rate"],
                "horizon_epochs": training["horizon_epochs"],
                "selection_resolved": True,
                "artifacts": artifacts,
            }
        )
        selection_rows.append(
            {
                "source_id": row["id"],
                "source_ledger_row_id": row["id"],
                "source_ledger": None,
                "source_evidence": None,
                "run_name": row["run_name"],
                "family_id": row["family_id"],
                "history_hidden_dim": representation["history_hidden_dim"],
                "embedding_learning_rate": training["embedding_learning_rate"],
                "deep_learning_rate": training["deep_learning_rate"],
                "horizon_epochs": training["horizon_epochs"],
                "artifacts": artifacts,
            }
        )
    evidence_payload = {
        "schema_version": 1,
        "kind": "test_rq2_evidence",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "tuning_ledger": evidence_runs,
    }
    evidence_payload["sha256"] = hashlib.sha256(
        _canonical(evidence_payload).encode()
    ).hexdigest()
    _write_json(evidence_path, evidence_payload)
    ledger_reference = _fact(root, ledger_path) | {
        "logical_sha256": ledger_payload["sha256"]
    }
    evidence_reference = _fact(root, evidence_path) | {
        "logical_sha256": evidence_payload["sha256"]
    }
    for row in selection_rows:
        row["source_ledger"] = ledger_reference
        row["source_evidence"] = evidence_reference
    selection = {
        "schema_version": 1,
        "kind": "g3_rq2_content_selection_for_rq3",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "selected_family_id": "rq2_content_concat",
        "selected_history_hidden_dim": 64,
        "selection_resolved": True,
        "feature_manifest": feature_manifest,
        "source_evidence": [evidence_reference],
        "rows": selection_rows,
    }
    selection["sha256"] = hashlib.sha256(_canonical(selection).encode()).hexdigest()
    selection_path = root / "selection.json"
    _write_json(selection_path, selection)
    return selection_path, str(selection["sha256"])


def _load_selection(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _rewrite_selection(path: Path, document: dict[str, object]) -> str:
    document.pop("sha256", None)
    document["sha256"] = hashlib.sha256(_canonical(document).encode()).hexdigest()
    _write_json(path, document)
    return str(document["sha256"])


def test_rq3_surface_authenticates_reuse_and_preserves_selected_width(
    tmp_path: Path, monkeypatch
) -> None:
    selection_path, selection_sha256 = _selection_fixture(tmp_path, monkeypatch)

    compiled = compile_rq3_output_surface(
        root=tmp_path,
        selection_path=selection_path,
        expected_selection_sha256=selection_sha256,
    )

    assert compiled.selected_history_hidden_dim == 64
    assert compiled.feature_data_sha256
    assert compiled.training_count_reference["sha256"]
    assert compiled.slice_membership_reference["sha256"]
    assert all(
        row.authenticated_source is None
        or (
            row.authenticated_source.training_count_sha256
            == compiled.training_count_reference["sha256"]
            and row.authenticated_source.slice_membership_sha256
            == compiled.slice_membership_reference["sha256"]
            and row.authenticated_source.diagnostics_epoch_count == row.horizon_epochs
        )
        for row in compiled.rows_by_family["rq3_output_learned"]
    )
    assert tuple(compiled.rows_by_family) == RQ3_OUTPUT_FAMILY_IDS
    signatures = [
        tuple(
            (
                row.embedding_learning_rate,
                row.deep_learning_rate,
                row.horizon_epochs,
                row.history_hidden_dim,
            )
            for row in compiled.rows_by_family[family]
        )
        for family in RQ3_OUTPUT_FAMILY_IDS
    ]
    assert all(signature == signatures[0] for signature in signatures[1:])
    assert (
        sum(
            row.reused_from is not None
            for row in compiled.rows_by_family["rq3_output_learned"]
        )
        == 2
    )
    assert all(
        row.reused_from is None
        for family in RQ3_OUTPUT_FAMILY_IDS[1:]
        for row in compiled.rows_by_family[family]
    )

    with pytest.raises(ValueError, match="must not launch duplicate"):
        build_rq3_training_experiment(
            compiled,
            compiled.rows_by_family["rq3_output_learned"][0],
            root=tmp_path,
        )


def test_rq3_surface_authenticates_feature_manifest_through_ledger_ancestry(
    tmp_path: Path, monkeypatch
) -> None:
    selection_path, selection_sha256 = _selection_fixture(
        tmp_path, monkeypatch, indirect_feature_binding=True
    )

    compiled = compile_rq3_output_surface(
        root=tmp_path,
        selection_path=selection_path,
        expected_selection_sha256=selection_sha256,
    )

    assert compiled.feature_manifest_sha256 == (
        rq3_protocol.NATIVE50_FEATURE_MANIFEST_SHA256
    )


@pytest.mark.parametrize("mutation", ["id_only", "fabricated_rate", "artifact"])
def test_rq3_surface_rejects_unauthenticated_or_incompatible_reuse(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    selection_path, _ = _selection_fixture(tmp_path, monkeypatch)
    document = _load_selection(selection_path)
    rows = document["rows"]
    assert isinstance(rows, list)
    first = rows[0]
    if mutation == "id_only":
        first["family_id"] = "rq2_id_only_densenet"
    elif mutation == "fabricated_rate":
        first["embedding_learning_rate"] = 0.2
    else:
        first["artifacts"]["training_metadata"]["sha256"] = "0" * 64
    selection_sha256 = _rewrite_selection(selection_path, document)

    with pytest.raises(ValueError):
        compile_rq3_output_surface(
            root=tmp_path,
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
        )


def test_rq3_surface_rejects_unbound_selection_and_rehashed_source(
    tmp_path: Path, monkeypatch
) -> None:
    selection_path, selection_sha256 = _selection_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="bound resolved content result"):
        compile_rq3_output_surface(
            root=tmp_path,
            selection_path=selection_path,
            expected_selection_sha256="0" * 64,
        )

    document = _load_selection(selection_path)
    rows = document["rows"]
    source = rows[0]["source_evidence"]
    source_path = tmp_path / source["path"]
    source_document = json.loads(source_path.read_text())
    source_document["tuning_ledger"][0]["run_name"] = "fabricated"
    _write_json(source_path, source_document)
    source["sha256"] = _sha(source_path)
    source["size_bytes"] = source_path.stat().st_size
    document["source_evidence"] = [source]
    selection_sha256 = _rewrite_selection(selection_path, document)
    with pytest.raises(ValueError, match="logical hash changed"):
        compile_rq3_output_surface(
            root=tmp_path,
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
        )


def test_compiled_rq3_row_builds_the_bound_training_experiment(
    tmp_path: Path, monkeypatch
) -> None:
    selection_path, selection_sha256 = _selection_fixture(tmp_path, monkeypatch)
    surface = compile_rq3_output_surface(
        root=tmp_path,
        selection_path=selection_path,
        expected_selection_sha256=selection_sha256,
    )
    row = surface.rows_by_family["rq3_output_learned_trainable_content"][0]

    experiment = build_rq3_training_experiment(
        surface,
        row,
        root=tmp_path,
    )

    assert experiment.run_name == row.run_name
    assert experiment.representation.history_representation == "id_content"
    assert experiment.representation.history_hidden_dim == 64
    assert experiment.representation.catalog_representation == ("id_trainable_content")
    assert experiment.embedding_learning_rate == row.embedding_learning_rate
    assert experiment.deep_learning_rate == row.deep_learning_rate
    assert math.isclose(
        experiment.lr_schedule_horizon_epochs,
        row.horizon_epochs,
    )


@pytest.mark.parametrize(
    ("axis", "fixture_arguments"),
    [
        ("dataset", {"dataset_mutation": ("exclude_seen", True)}),
        ("loss", {"invariant_mutation": ("negative_sampling", "online_logq")}),
    ],
)
def test_rq3_reuse_rejects_dataset_or_loss_semantic_mutation(
    tmp_path: Path, monkeypatch, axis: str, fixture_arguments: dict[str, object]
) -> None:
    selection_path, selection_sha256 = _selection_fixture(
        tmp_path, monkeypatch, **fixture_arguments
    )
    with pytest.raises(ValueError, match=axis):
        compile_rq3_output_surface(
            root=tmp_path,
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
        )


def test_rq3_surface_rejects_feature_bytes_changed_after_selection(
    tmp_path: Path, monkeypatch
) -> None:
    selection_path, selection_sha256 = _selection_fixture(tmp_path, monkeypatch)
    selection = _load_selection(selection_path)
    manifest = json.loads(
        (tmp_path / selection["feature_manifest"]["path"]).read_text()
    )
    item_features = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "item_features"
    )
    (tmp_path / item_features["path"]).write_bytes(b"changed")

    with pytest.raises(ValueError, match="item_features"):
        compile_rq3_output_surface(
            root=tmp_path,
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
        )


def test_rq3_surface_rejects_rehashed_diagnostics_with_different_slices(
    tmp_path: Path, monkeypatch
) -> None:
    selection_path, _ = _selection_fixture(tmp_path, monkeypatch)
    selection = _load_selection(selection_path)
    first = selection["rows"][0]
    diagnostics_path = tmp_path / first["artifacts"]["training_diagnostics"]["path"]
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["frequency_terciles"]["slices"]["tail"]["training_interactions"] = 2
    _write_json(diagnostics_path, diagnostics)
    diagnostics_fact = _fact(tmp_path, diagnostics_path)
    first["artifacts"]["training_diagnostics"] = diagnostics_fact

    evidence_reference = first["source_evidence"]
    evidence_path = tmp_path / evidence_reference["path"]
    evidence = json.loads(evidence_path.read_text())
    evidence["tuning_ledger"][0]["artifacts"]["training_diagnostics"] = diagnostics_fact
    evidence.pop("sha256")
    evidence["sha256"] = hashlib.sha256(_canonical(evidence).encode()).hexdigest()
    _write_json(evidence_path, evidence)
    rebound_evidence = _fact(tmp_path, evidence_path) | {
        "logical_sha256": evidence["sha256"]
    }
    selection["source_evidence"] = [rebound_evidence]
    for row in selection["rows"]:
        row["source_evidence"] = rebound_evidence
    selection_sha256 = _rewrite_selection(selection_path, selection)

    with pytest.raises(ValueError, match="training-frequency slices"):
        compile_rq3_output_surface(
            root=tmp_path,
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
        )


def test_rq3_surface_rejects_rehashed_selection_missing_approved_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    selection_path, _ = _selection_fixture(tmp_path, monkeypatch)
    selection = _load_selection(selection_path)
    selection["rows"].pop()
    selection_sha256 = _rewrite_selection(selection_path, selection)

    with pytest.raises(ValueError, match="complete approved reusable"):
        compile_rq3_output_surface(
            root=tmp_path,
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
        )


@pytest.mark.parametrize("mutation", ["empty", "missing_common_field", "wrong_epoch"])
def test_rq3_surface_rejects_noncomparable_reused_diagnostics(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    selection_path, _ = _selection_fixture(tmp_path, monkeypatch)
    selection = _load_selection(selection_path)
    first = selection["rows"][0]
    diagnostics_path = tmp_path / first["artifacts"]["training_diagnostics"]["path"]
    diagnostics = json.loads(diagnostics_path.read_text())
    if mutation == "empty":
        diagnostics["epochs"] = []
    elif mutation == "missing_common_field":
        diagnostics["epochs"][0].pop("training")
    else:
        diagnostics["epochs"][0]["epoch"] = 1
    _write_json(diagnostics_path, diagnostics)
    diagnostics_fact = _fact(tmp_path, diagnostics_path)
    first["artifacts"]["training_diagnostics"] = diagnostics_fact

    evidence_reference = first["source_evidence"]
    evidence_path = tmp_path / evidence_reference["path"]
    evidence = json.loads(evidence_path.read_text())
    evidence["tuning_ledger"][0]["artifacts"]["training_diagnostics"] = diagnostics_fact
    evidence.pop("sha256")
    evidence["sha256"] = hashlib.sha256(_canonical(evidence).encode()).hexdigest()
    _write_json(evidence_path, evidence)
    rebound = _fact(tmp_path, evidence_path) | {"logical_sha256": evidence["sha256"]}
    selection["source_evidence"] = [rebound]
    for row in selection["rows"]:
        row["source_evidence"] = rebound
    selection_sha256 = _rewrite_selection(selection_path, selection)

    with pytest.raises(ValueError, match="diagnostics"):
        compile_rq3_output_surface(
            root=tmp_path,
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
        )


@pytest.mark.parametrize("schema_version", [1, 2])
@pytest.mark.parametrize(
    "mutation",
    ["training_scope", "catalog_scope", "component_gradients", "pretrained_content"],
)
def test_rq3_surface_rejects_substantively_empty_nested_diagnostics(
    tmp_path: Path, monkeypatch, schema_version: int, mutation: str
) -> None:
    selection_path, _ = _selection_fixture(
        tmp_path,
        monkeypatch,
        diagnostics_schema_version=schema_version,
    )
    selection = _load_selection(selection_path)
    first = selection["rows"][0]
    diagnostics_path = tmp_path / first["artifacts"]["training_diagnostics"]["path"]
    diagnostics = json.loads(diagnostics_path.read_text())
    epoch = diagnostics["epochs"][0]
    if mutation == "training_scope":
        epoch["training"]["global"] = {}
    elif mutation == "catalog_scope":
        epoch["catalog_representation_norm"]["global"] = {}
    elif mutation == "component_gradients":
        epoch["component_gradient_norms"] = {}
    else:
        epoch["pretrained_content"] = {}
    _rebind_first_diagnostics(tmp_path, selection, diagnostics_path, diagnostics)
    selection_sha256 = _rewrite_selection(selection_path, selection)

    with pytest.raises(ValueError, match="diagnostics"):
        compile_rq3_output_surface(
            root=tmp_path,
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
        )


def test_rq3_surface_rejects_empty_schema_v2_catalog_table_gradients(
    tmp_path: Path, monkeypatch
) -> None:
    selection_path, _ = _selection_fixture(
        tmp_path,
        monkeypatch,
        diagnostics_schema_version=2,
    )
    selection = _load_selection(selection_path)
    first = selection["rows"][0]
    diagnostics_path = tmp_path / first["artifacts"]["training_diagnostics"]["path"]
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["epochs"][0]["catalog_table_gradient_norms"] = {}
    _rebind_first_diagnostics(tmp_path, selection, diagnostics_path, diagnostics)
    selection_sha256 = _rewrite_selection(selection_path, selection)

    with pytest.raises(ValueError, match="diagnostics"):
        compile_rq3_output_surface(
            root=tmp_path,
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
        )


def test_rq3_surface_accepts_complete_schema_v2_nested_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    selection_path, selection_sha256 = _selection_fixture(
        tmp_path,
        monkeypatch,
        diagnostics_schema_version=2,
    )

    surface = compile_rq3_output_surface(
        root=tmp_path,
        selection_path=selection_path,
        expected_selection_sha256=selection_sha256,
    )

    reused = [
        row.authenticated_source
        for row in surface.rows_by_family["rq3_output_learned"]
        if row.authenticated_source is not None
    ]
    assert reused
    assert all(source.diagnostics_schema_version == 2 for source in reused)


@pytest.mark.parametrize(
    ("schema_version", "mutation"),
    [
        (1, "content_drift_reference"),
        (1, "component_gradient_keys"),
        (1, "pretrained_content"),
        (2, "catalog_table_parameters"),
        (1, "catalog_table_schema"),
    ],
)
def test_rq3_surface_rejects_rehashed_reuse_semantic_mutations(
    tmp_path: Path,
    monkeypatch,
    schema_version: int,
    mutation: str,
) -> None:
    selection_path, _ = _selection_fixture(
        tmp_path,
        monkeypatch,
        diagnostics_schema_version=schema_version,
    )
    selection = _load_selection(selection_path)
    first = selection["rows"][0]
    diagnostics_path = tmp_path / first["artifacts"]["training_diagnostics"]["path"]
    diagnostics = json.loads(diagnostics_path.read_text())
    epoch = diagnostics["epochs"][0]
    if mutation == "content_drift_reference":
        diagnostics["content_drift_reference"] = {"available": True}
    elif mutation == "component_gradient_keys":
        epoch["component_gradient_norms"]["content_encoder"] = _distribution()
    elif mutation == "pretrained_content":
        epoch["pretrained_content"] = {
            "available": True,
            "trainable": False,
            "drift_l2": {
                scope: _distribution(3 if scope == "global" else 1)
                for scope in ("global", "tail", "mid", "head")
            },
            "cosine_to_initial": {
                scope: _distribution(3 if scope == "global" else 1)
                for scope in ("global", "tail", "mid", "head")
            },
        }
    elif mutation == "catalog_table_parameters":
        epoch["catalog_table_gradient_norms"]["bias"] = (
            epoch["catalog_table_gradient_norms"]["weight"]
        )
    else:
        epoch["catalog_table_gradient_norms"] = {
            "weight": {
                scope: {
                    name: _distribution()
                    for name in (
                        "all_row_exposure_weighted_norm",
                        "conditional_on_active_row_norm",
                        "active_row_count",
                        "active_row_fraction",
                    )
                }
                for scope in ("global", "tail", "mid", "head")
            }
        }
    _rebind_first_diagnostics(tmp_path, selection, diagnostics_path, diagnostics)
    selection_sha256 = _rewrite_selection(selection_path, selection)

    with pytest.raises(ValueError, match="diagnostics"):
        compile_rq3_output_surface(
            root=tmp_path,
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
        )


def _rebind_first_diagnostics(
    root: Path,
    selection: dict[str, object],
    diagnostics_path: Path,
    diagnostics: dict[str, object],
) -> None:
    _write_json(diagnostics_path, diagnostics)
    diagnostics_fact = _fact(root, diagnostics_path)
    first = selection["rows"][0]
    first["artifacts"]["training_diagnostics"] = diagnostics_fact
    evidence_reference = first["source_evidence"]
    evidence_path = root / evidence_reference["path"]
    evidence = json.loads(evidence_path.read_text())
    evidence["tuning_ledger"][0]["artifacts"]["training_diagnostics"] = diagnostics_fact
    evidence.pop("sha256")
    evidence["sha256"] = hashlib.sha256(_canonical(evidence).encode()).hexdigest()
    _write_json(evidence_path, evidence)
    rebound = _fact(root, evidence_path) | {"logical_sha256": evidence["sha256"]}
    selection["source_evidence"] = [rebound]
    for row in selection["rows"]:
        row["source_evidence"] = rebound
