from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import polars as pl
import pytest
import torch

import dcn.config.generation as generation_module
from dcn.config.generation import GenerationExperiment
import experiments.g4_future_items.launchers.freeze_control as freeze_control_module
import experiments.g4_future_items.launchers.run_selectors as run_selectors_module
import experiments.g4_future_items.protocol.manifest as manifest_module
import experiments.g4_future_items.configs.control as control_module
from experiments.g4_future_items.configs.control import (
    G4GenerationExperiment,
    _ranking_snapshot_document,
    build_anchor_control,
    build_control,
    control_runtime_projection,
)
from experiments.g4_future_items.configs.treatments import build_treatment
from experiments.g4_future_items.configs.selectors import compile_selector_search
from experiments.g4_future_items.launchers.compiled import (
    build_training_experiment,
    decode_job,
    encode_job,
)
from experiments.g4_future_items.launchers.freeze_control import (
    compile_control_freeze,
    freeze_control,
)
from experiments.g4_future_items.protocol.manifest import (
    APPROVED_CONTROL_MANIFEST_SHA256,
    MATERIALIZATION_COST_LIMITS,
    CompiledJob,
    build_control_semantics_manifest,
    build_runtime_compatibility_evidence,
    build_selected_control_manifest,
    build_treatment_compatibility_manifest,
    build_treatment_semantics_manifest,
    canonical_bytes,
    canonical_sha256,
    compile_control_tuning_ledger,
    compile_recommender_boundary_ledger,
    compile_selector_gate_ledger,
    compile_selector_boundary_ledger,
    compile_selector_materialization_ledger,
    compile_selector_search_ledger,
    compile_treatment_tuning_ledger,
    expected_control_source_paths,
    load_control_manifest,
    load_strict_json,
    load_ledger,
    resolve_control_data_identity,
    source_manifest,
    resolve_ledger_row,
    validate_control_round_trip,
    verify_ledger_semantics,
    write_frozen_ledger,
    write_frozen_manifest,
    _validate_control_semantics_document,
    _validate_selected_control_document,
    _verify_materialization_artifacts,
    _compile_recommender_boundary_ledger,
    _compile_selector_boundary_ledger,
)
from experiments.g4_future_items.protocol.materialization import write_period_artifact
from utils.global_config import config as global_config


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
from experiments.g4_future_items.targets import FutureEventIndex


def _materialization_evidence() -> dict[str, object]:
    return {
        "version": "g4-materialization-cost-v1",
        "measurement_id": SHA_E,
        "passes": True,
        "deterministic_artifact_sha256": SHA_D,
        "learned_artifact_sha256": SHA_C,
        "runtime": {"wall_seconds": 1.0, "peak_aggregate_rss_bytes": 1},
        "logical_output_scratch_bytes": 1,
        "timed_load_valid": True,
        "limits": dict(MATERIALIZATION_COST_LIMITS),
    }


def _control_semantics_fixture(tmp_path: Path) -> dict[str, object]:
    main = tmp_path / "events_remapped.parquet"
    remap = tmp_path / "item_id_remap.parquet"
    content = tmp_path / "embeddings_compact.parquet"
    pl.DataFrame({"timestamp": [1_700_604_800], "compact_item_id": [1]}).write_parquet(
        main
    )
    pl.DataFrame({"compact_id": [1]}).write_parquet(remap)
    pl.DataFrame({"compact_id": [1], "embedding": [[1.0]]}).write_parquet(content)
    artifacts = SimpleNamespace(
        main_parquet=main,
        item_id_column="compact_item_id",
        precomputed_embeddings={"compact_item_id": content},
    )
    dataset_key = hashlib.sha1(str(main.resolve()).encode()).hexdigest()[:12]
    experiment = SimpleNamespace(
        artifacts=artifacts,
        dataset_key=dataset_key,
        validation_cutoff_timestamp=1_700_000_000,
    )
    root = Path(__file__).resolve().parents[4]
    paths = expected_control_source_paths()
    return build_control_semantics_manifest(
        source_paths=paths,
        sources=source_manifest(root, paths),
        data_identity=resolve_control_data_identity(experiment),
        training_semantics_revisions={
            "generation": 2,
            "negative_sampling": 1,
            "timestamp_bins": 1,
        },
    )


def _control_experiment_fixture(tmp_path: Path) -> SimpleNamespace:
    main = tmp_path / "events_remapped.parquet"
    remap = tmp_path / "item_id_remap.parquet"
    content = tmp_path / "embeddings_compact.parquet"
    pl.DataFrame({"timestamp": [1_700_604_800], "compact_item_id": [1]}).write_parquet(
        main
    )
    pl.DataFrame({"compact_id": [1]}).write_parquet(remap)
    pl.DataFrame({"compact_id": [1], "embedding": [[1.0]]}).write_parquet(content)
    artifacts = SimpleNamespace(
        main_parquet=main,
        item_id_column="compact_item_id",
        precomputed_embeddings={"compact_item_id": content},
    )
    return SimpleNamespace(
        artifacts=artifacts,
        dataset_key=hashlib.sha1(str(main.resolve()).encode()).hexdigest()[:12],
        validation_cutoff_timestamp=1_700_000_000,
    )


def _completed_control_run(
    tmp_path: Path,
    *,
    control_semantics_manifest_sha256: str,
    winner_row_id: str = "control_tuning:03",
) -> tuple[Path, list[Path]]:
    ledger = compile_control_tuning_ledger(control_semantics_manifest_sha256)
    ledger_path = tmp_path / "control_tuning.json"
    write_frozen_ledger(ledger_path, ledger)
    run_directories = []
    for row in ledger["rows"]:
        job = row["job"]
        run_directory = tmp_path / job["run_name"]
        run_directory.mkdir()
        (run_directory / "g4_job.json").write_text(
            json.dumps(
                {
                    "ledger_sha256": ledger["sha256"],
                    "row_id": row["id"],
                    "job": job,
                    "ledger_path": str(ledger_path.resolve()),
                    "ledger_stage": ledger["stage"],
                }
            )
        )
        horizon = job["lr_schedule_horizon_epochs"]
        (run_directory / "training_metadata.json").write_text(
            json.dumps(
                {
                    "best_epoch": 2,
                    "epochs_trained": horizon,
                    "lr_schedule_horizon_epochs": horizon,
                    "num_epochs": horizon,
                    "max_epochs": horizon,
                    "batch_size": 512,
                    "embedding_learning_rate": job["embedding_learning_rate"],
                    "deep_learning_rate": job["deep_learning_rate"],
                    "lr_horizon_complete": True,
                    "selection_resolved": True,
                }
            )
        )
        (run_directory / "sweep.log").write_text(
            "epoch 1 finished epoch/val.loss=1.0 "
            f"epoch/val_true.recall@100={0.2 if row['id'] == winner_row_id else 0.1}\n"
        )
        run_directories.append(run_directory)
    return ledger_path, run_directories


def _selector_search_artifacts(
    tmp_path: Path,
    ledger: dict[str, object],
    *,
    winner_row_id: str = "selector_search:learned:06",
) -> Path:
    from experiments.g4_future_items.configs.selectors import selector_trial_from_job

    search_root = tmp_path / "selector-search"
    metrics = {
        "user_balanced_ndcg_at_10": 0.1,
        "auroc": 0.5,
        "query_count": 10,
        "user_count": 5,
        "pair_count": 20,
        "positive_count": 10,
        "negative_count": 10,
        "positive_rate": 0.5,
    }
    for row in ledger["rows"]:
        job = row["job"]
        trial = selector_trial_from_job(job)
        validation_metrics = dict(metrics)
        if row["id"] == winner_row_id:
            validation_metrics["user_balanced_ndcg_at_10"] = 0.2
        artifact = {
            "version": "g4-selector-search-v1",
            "trial": trial.to_dict(),
            "sampler_seed": trial.sampler_seed,
            "classifier_seed": 42,
            "prepared_sha256": SHA_C,
            "prepared_semantics_sha256": SHA_D,
            "prepared_input_sha256": {},
            "relevance_threshold": 0.5,
            "output_artifact_sha256": job["output_artifact_sha256"],
            "validation_metrics": validation_metrics,
        }
        content = canonical_bytes(artifact)
        payload_sha256 = hashlib.sha256(content).hexdigest()
        destination = search_root / job["output_artifact_sha256"]
        destination.mkdir(parents=True)
        (destination / "artifact.json").write_bytes(content)
        (destination / "artifact.sha256").write_text(payload_sha256)
        result = {
            "version": "g4-selector-search-v1",
            "trial": trial.to_dict(),
            "validation_metrics": validation_metrics,
            "relevance_threshold": 0.5,
            "artifact_sha256": job["output_artifact_sha256"],
            "artifact_payload_sha256": payload_sha256,
            "prepared_sha256": SHA_C,
            "prepared_semantics_sha256": SHA_D,
            "wall_seconds": 1.0,
        }
        (destination / "result.json").write_bytes(canonical_bytes(result))
    return search_root


def test_control_freeze_derives_documents_and_dry_run_does_not_write(
    tmp_path: Path,
) -> None:
    experiment = _control_experiment_fixture(tmp_path)
    semantics_path = tmp_path / "protocol" / "control_semantics_manifest.json"
    ledger_path = tmp_path / "protocol" / "ledgers" / "control_tuning.json"

    compiled = compile_control_freeze(experiment)
    preview = freeze_control(
        semantics_path=semantics_path,
        ledger_path=ledger_path,
        experiment=experiment,
        write=False,
    )

    assert compiled == compile_control_freeze(experiment)
    assert compiled.semantics["source_paths"] == expected_control_source_paths()
    assert compiled.semantics["training_semantics_revisions"] == {
        "generation": 2,
        "negative_sampling": 2,
        "timestamp_bins": 2,
    }
    assert compiled.ledger["control_semantics_manifest_sha256"] == canonical_sha256(
        compiled.semantics
    )
    assert preview["write"] is False
    assert preview["destinations"] == {
        "control_semantics_manifest": "absent",
        "control_tuning_ledger": "absent",
    }
    assert not semantics_path.exists()
    assert not ledger_path.exists()


def test_control_freeze_initializes_the_runtime_base_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor = build_anchor_control()
    anchor.base_path = tmp_path / "generated"
    monkeypatch.setattr(
        freeze_control_module, "build_anchor_control", lambda: anchor
    )

    compile_control_freeze(_control_experiment_fixture(tmp_path))

    assert global_config.base_path == Path(anchor.base_path)


def test_control_freeze_writes_both_documents_immutably_after_preflight(
    tmp_path: Path,
) -> None:
    experiment = _control_experiment_fixture(tmp_path)
    semantics_path = tmp_path / "protocol" / "control_semantics_manifest.json"
    ledger_path = tmp_path / "protocol" / "ledgers" / "control_tuning.json"

    result = freeze_control(
        semantics_path=semantics_path,
        ledger_path=ledger_path,
        experiment=experiment,
        write=True,
    )

    assert result["write"] is True
    assert load_strict_json(semantics_path)["kind"] == "g4_control_semantics"
    assert load_ledger(ledger_path)["stage"] == "control_tuning"
    assert semantics_path.stat().st_mode & 0o222 == 0
    assert ledger_path.stat().st_mode & 0o222 == 0

    different = tmp_path / "different"
    different.mkdir()
    other_experiment = _control_experiment_fixture(different)
    ledger_path.unlink()
    with pytest.raises(RuntimeError, match="frozen manifest differs"):
        freeze_control(
            semantics_path=semantics_path,
            ledger_path=ledger_path,
            experiment=other_experiment,
            write=True,
        )
    assert not ledger_path.exists()


def test_treatment_freeze_owns_reviewed_closures_revisions_and_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "treatment-semantics.json"
    monkeypatch.setattr(
        run_selectors_module, "TREATMENT_SEMANTICS_MANIFEST_PATH", destination
    )

    document = run_selectors_module.compile_treatment_semantics_freeze()
    preview = run_selectors_module.freeze_treatment_semantics(write=False)

    assert document["kind"] == "g4_treatment_semantics"
    assert document["schema_revisions"] == (
        run_selectors_module.TREATMENT_SCHEMA_REVISIONS
    )
    assert set(document["fixtures"]) == set(
        run_selectors_module.TREATMENT_FIXTURE_PATHS
    )
    assert (
        "experiments/g4_future_items/launchers/compiled.py"
        in document["entrypoint_source_paths"][
            "experiments/g4_future_items/launchers/run_selectors.py"
        ]
    )
    assert "experiments/g4_future_items/protocol/metrics.py" not in document[
        "source_paths"
    ]
    assert preview["destination"] == "absent"
    assert preview["write"] is False
    assert not destination.exists()


def test_approved_control_manifest_is_canonical_and_round_trips() -> None:
    manifest = load_control_manifest()

    assert canonical_sha256(manifest) == APPROVED_CONTROL_MANIFEST_SHA256
    assert validate_control_round_trip(build_anchor_control()) == manifest


def test_control_entry_has_no_transitive_g4_import_outside_approved_closure() -> None:
    root = Path(__file__).resolve().parents[4]
    allowed = set(expected_control_source_paths())
    expected_g4 = {
        "experiments/g4_future_items/__init__.py",
        "experiments/g4_future_items/configs/__init__.py",
        "experiments/g4_future_items/configs/control.py",
        "experiments/g4_future_items/launchers/freeze_control.py",
        "experiments/g4_future_items/launchers/run_control.py",
        "experiments/g4_future_items/protocol/__init__.py",
        "experiments/g4_future_items/protocol/manifest.py",
        "experiments/g4_future_items/protocol/control_manifest.json",
        "experiments/g4_future_items/protocol/manifest_contract.md",
    }
    assert {
        path for path in allowed if path.startswith("experiments/g4_")
    } == expected_g4
    assert not any(path.startswith("generated/") for path in allowed)
    assert all((root / path).is_file() for path in allowed)
    for relative in (
        "experiments/g4_future_items/configs/control.py",
        "experiments/g4_future_items/launchers/freeze_control.py",
        "experiments/g4_future_items/launchers/run_control.py",
        "experiments/g4_future_items/protocol/manifest.py",
    ):
        tree = ast.parse((root / relative).read_text())
        imported = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        } | {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        local_imports = {
            f"{name.replace('.', '/')}.py"
            for name in imported
            if name.startswith("experiments.g4_future_items")
        }
        assert local_imports <= allowed


def test_control_semantics_validation_rejects_reduced_source_closure(
    tmp_path: Path,
) -> None:
    document = _control_semantics_fixture(tmp_path)
    removed = document["source_paths"].pop()
    document["sources"].pop(removed)

    with pytest.raises(ValueError, match="source closure differs"):
        _validate_control_semantics_document(document)


def test_control_builder_projects_every_approved_value() -> None:
    experiment = build_control(
        run_name="g4_control_unit",
        batch_size=512,
        embedding_learning_rate=0.004,
        deep_learning_rate=0.003,
        lr_schedule_horizon_epochs=15,
        seed=47,
    )
    projection = control_runtime_projection(experiment)

    assert projection["fixed"] == load_control_manifest()["fixed"]
    assert projection["selected"] == {
        "batch_size": 512,
        "embedding_learning_rate": 0.004,
        "deep_learning_rate": 0.003,
        "lr_schedule_horizon_epochs": 15,
    }
    assert experiment.seed == 47
    assert experiment.final_ranking_evidence_group == "g4-native50m"

    with pytest.raises(ValueError, match="must be 512"):
        build_control(
            run_name="g4_control_bad_batch",
            batch_size=1024,
            embedding_learning_rate=0.004,
            deep_learning_rate=0.003,
            lr_schedule_horizon_epochs=15,
        )


def test_ranking_snapshot_document_records_catalog_and_rankings() -> None:
    prepared = type("Prepared", (), {"item_id_list": [1, 2, 3]})()
    document = _ranking_snapshot_document(
        prepared=prepared,
        rankings={7: (3, 2, 1)},
        exclude_seen=False,
        max_k=3,
    )
    assert document["catalog_size"] == 3


def test_final_report_writes_the_snapshot_from_its_single_ranking_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = SimpleNamespace(item_id_list=[1, 2, 3])
    calls = []
    evidence = object()

    def score_with_evidence_and_rankings(*, max_users):
        calls.append(max_users)
        return {"recall@1": 1.0}, evidence, {7: (3, 2, 1)}

    metric = SimpleNamespace(
        _prepared={None: prepared},
        score_with_evidence_and_rankings=score_with_evidence_and_rankings,
    )
    experiment = object.__new__(G4GenerationExperiment)
    experiment.__dict__.update(
        true_metric=metric,
        run_name="trace",
        exclude_seen_from_evaluation=False,
        restore_best_weights=False,
        final_ranking_evidence_group="g4-test",
    )
    written = []
    monkeypatch.setattr(
        generation_module,
        "write_ranking_evidence",
        lambda value, **kwargs: written.append((value, kwargs)),
    )
    monkeypatch.setattr(torch, "topk", lambda *args, **kwargs: pytest.fail("topk"))
    monkeypatch.setattr(
        control_module, "global_config", SimpleNamespace(logs_path=tmp_path)
    )
    monkeypatch.setattr(
        generation_module, "global_config", SimpleNamespace(logs_path=tmp_path)
    )

    GenerationExperiment._report_final_metrics(experiment, None)

    assert calls == [None]
    assert written[0][0] is evidence
    snapshot = json.loads((tmp_path / "trace/top_item_rankings.json").read_text())
    assert snapshot["rankings"] == [{"user_id": 7, "item_ids": [3, 2, 1]}]


def test_control_tuning_ledger_is_deterministic_closed_and_anchored(
    tmp_path: Path,
) -> None:
    first = compile_control_tuning_ledger(SHA_A)
    second = compile_control_tuning_ledger(SHA_A)

    assert first == second
    assert len(first["rows"]) == 20
    assert {row["job"]["dataloader"]["batch_size"] for row in first["rows"]} == {
        512
    }
    assert first["rows"][0]["job"]["embedding_learning_rate"] == pytest.approx(
        0.16590060219780284
    )
    assert first["rows"][0]["job"]["deep_learning_rate"] == pytest.approx(
        0.02879154157702692
    )
    assert first["rows"][0]["job"]["lr_schedule_horizon_epochs"] == 20
    assert {
        row["job"]["lr_schedule_horizon_epochs"] for row in first["rows"]
    } <= {5, 10, 15, 20, 25, 30}
    assert {row["id"] for row in first["rows"]} == {
        f"control_tuning:{trial:02d}" for trial in range(1, 21)
    }

    path = tmp_path / "control.json"
    write_frozen_ledger(path, first)
    assert load_ledger(path) == first
    assert resolve_ledger_row(path, "control_tuning:01") == first["rows"][0]

    changed = json.loads(path.read_text())
    changed["rows"][1]["job"]["seed"] = 43
    path.chmod(0o644)
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="ledger hash"):
        load_ledger(path)


def test_ledger_validation_rejects_self_rehashed_incomplete_topology(
    tmp_path: Path,
) -> None:
    ledger = compile_control_tuning_ledger(SHA_A)
    forged = {key: value for key, value in ledger.items() if key != "sha256"}
    forged["rows"] = [json.loads(json.dumps(ledger["rows"][0]))]
    forged["rows"][0]["id"] = "control_tuning:999"
    forged["rows"][0]["job"]["protocol"]["trial_id"] = 999
    forged["sha256"] = canonical_sha256(forged)
    path = tmp_path / "forged-control.json"
    path.write_text(json.dumps(forged))

    with pytest.raises(ValueError, match="row topology"):
        load_ledger(path)


def test_ledger_validation_rejects_unapproved_boundary_metadata(
    tmp_path: Path,
) -> None:
    ledger_path, run_directories = _completed_control_run(
        tmp_path,
        control_semantics_manifest_sha256=SHA_A,
        winner_row_id="control_tuning:01",
    )
    boundary = compile_recommender_boundary_ledger(
        predecessor_ledger_paths=[ledger_path],
        candidate_run_directories=run_directories,
        boundary_round=1,
        control_semantics_manifest_sha256=SHA_A,
    )

    forged = {key: value for key, value in boundary.items() if key != "sha256"}
    forged["rate_bounds"] = {
        "embedding_learning_rate": [0.00005, 0.256],
        "deep_learning_rate": [0.0001, 0.128],
    }
    forged["sha256"] = canonical_sha256(forged)
    path = tmp_path / "forged-rates.json"
    path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="approved transition"):
        load_ledger(path)

    forged = json.loads(json.dumps(boundary))
    forged["horizon_values"] = [49]
    for row in forged["rows"]:
        row["job"]["lr_schedule_horizon_epochs"] = 49
    unsigned = {key: value for key, value in forged.items() if key != "sha256"}
    forged["sha256"] = canonical_sha256(unsigned)
    path = tmp_path / "forged-horizon.json"
    path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="approved transition"):
        load_ledger(path)


def test_ledger_validation_rejects_self_rehashed_seeded_coordinate_changes(
    tmp_path: Path,
) -> None:
    ledgers = [
        compile_control_tuning_ledger(SHA_A),
        compile_treatment_tuning_ledger(
            objective_id="rq1_24h",
            selected_control_manifest_sha256=SHA_A,
            treatment_semantics_manifest_sha256=SHA_B,
            batch_size=512,
            embedding_learning_rate=0.02,
            deep_learning_rate=0.01,
            lr_schedule_horizon_epochs=20,
        ),
        compile_selector_search_ledger(
            treatment_semantics_manifest_sha256=SHA_A,
            input_artifact_sha256=SHA_B,
        ),
    ]
    for index, ledger in enumerate(ledgers):
        forged = json.loads(json.dumps(ledger))
        if forged["stage"] == "selector_search":
            job = forged["rows"][-1]["job"]
            job["learning_rate"] = 0.15
            from experiments.g4_future_items.protocol.manifest import (
                _selector_output_identity,
            )

            job["output_artifact_sha256"] = _selector_output_identity(job)
        else:
            forged["rows"][1]["job"]["deep_learning_rate"] = 0.011
        unsigned = {key: value for key, value in forged.items() if key != "sha256"}
        forged["sha256"] = canonical_sha256(unsigned)
        path = tmp_path / f"forged-base-{index}.json"
        path.write_text(json.dumps(forged))
        with pytest.raises(ValueError, match="exact seeded compilation"):
            load_ledger(path)


def test_boundary_ledgers_bind_entering_row_and_exact_seeded_coordinates(
    tmp_path: Path,
) -> None:
    ledger_path, run_directories = _completed_control_run(
        tmp_path,
        control_semantics_manifest_sha256=SHA_A,
        winner_row_id="control_tuning:01",
    )
    boundary = compile_recommender_boundary_ledger(
        predecessor_ledger_paths=[ledger_path],
        candidate_run_directories=run_directories,
        boundary_round=1,
        control_semantics_manifest_sha256=SHA_A,
    )
    forged = json.loads(json.dumps(boundary))
    forged["rows"][0]["job"]["embedding_learning_rate"] *= 0.9
    forged["rows"][0]["job"]["deep_learning_rate"] = 0.02
    unsigned = {key: value for key, value in forged.items() if key != "sha256"}
    forged["sha256"] = canonical_sha256(unsigned)
    path = tmp_path / "forged-recommender-boundary.json"
    path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="exact seeded compilation"):
        load_ledger(path)


def test_boundary_production_reconstructs_winner_and_rejects_fabricated_entry(
    tmp_path: Path,
) -> None:
    ledger_path, run_directories = _completed_control_run(
        tmp_path,
        control_semantics_manifest_sha256=SHA_A,
        winner_row_id="control_tuning:01",
    )
    boundary = compile_recommender_boundary_ledger(
        predecessor_ledger_paths=[ledger_path],
        candidate_run_directories=run_directories,
        boundary_round=1,
        control_semantics_manifest_sha256=SHA_A,
    )
    boundary_path = tmp_path / "bound-boundary.json"
    write_frozen_ledger(boundary_path, boundary)
    assert boundary["entering_row"] == resolve_ledger_row(
        ledger_path, "control_tuning:01"
    )

    fabricated = json.loads(json.dumps(boundary["entering_row"]))
    fabricated["job"]["embedding_learning_rate"] = 0.0001
    forged = _compile_recommender_boundary_ledger(
        entering_row=fabricated,
        boundary_round=1,
        control_semantics_manifest_sha256=SHA_A,
    )
    unsigned = {key: value for key, value in forged.items() if key != "sha256"}
    unsigned["predecessor_evidence"] = boundary["predecessor_evidence"]
    forged = unsigned | {"sha256": canonical_sha256(unsigned)}
    forged_path = tmp_path / "fabricated-boundary.json"
    forged_path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="exact seeded compilation"):
        load_ledger(forged_path)


def test_selector_boundary_binds_hashed_results_and_rejects_fabricated_entry(
    tmp_path: Path,
) -> None:
    base = compile_selector_search_ledger(
        treatment_semantics_manifest_sha256=SHA_A,
        input_artifact_sha256=SHA_B,
    )
    ledger_path = tmp_path / "selector-search.json"
    write_frozen_ledger(ledger_path, base)
    search_root = _selector_search_artifacts(tmp_path, base)
    boundary = compile_selector_boundary_ledger(
        predecessor_ledger_paths=[ledger_path],
        search_root=search_root,
        treatment_semantics_manifest_sha256=SHA_A,
        boundary_round=1,
    )
    boundary_path = tmp_path / "selector-boundary.json"
    write_frozen_ledger(boundary_path, boundary)
    assert boundary["entering_row"]["id"] == "selector_search:learned:06"

    fabricated = json.loads(json.dumps(boundary["entering_row"]))
    fabricated["job"]["learning_rate"] = 0.01
    forged = _compile_selector_boundary_ledger(
        entering_row=fabricated,
        treatment_semantics_manifest_sha256=SHA_A,
        boundary_round=1,
    )
    unsigned = {key: value for key, value in forged.items() if key != "sha256"}
    unsigned["predecessor_evidence"] = boundary["predecessor_evidence"]
    forged = unsigned | {"sha256": canonical_sha256(unsigned)}
    forged_path = tmp_path / "fabricated-selector-boundary.json"
    forged_path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="exact seeded compilation"):
        load_ledger(forged_path)


def test_treatment_ledgers_fix_batch_and_tune_rates_and_horizon() -> None:
    selected = {
        "selected_control_manifest_sha256": SHA_A,
        "batch_size": 512,
        "embedding_learning_rate": 0.02,
        "deep_learning_rate": 0.01,
        "lr_schedule_horizon_epochs": 20,
    }
    rq1 = compile_treatment_tuning_ledger(
        objective_id="rq1_24h",
        treatment_semantics_manifest_sha256=SHA_B,
        **selected,
    )
    rq3 = compile_treatment_tuning_ledger(
        objective_id="rq3_learned_hard",
        treatment_semantics_manifest_sha256=SHA_B,
        selector_artifact_sha256=SHA_C,
        materialization_cost_evidence=_materialization_evidence(),
        **selected,
    )

    assert {row["job"]["dataloader"]["batch_size"] for row in rq1["rows"]} == {512}
    assert {
        row["job"]["lr_schedule_horizon_epochs"] for row in rq1["rows"]
    } <= {5, 10, 15, 20, 25, 30}
    assert rq1["rows"][0]["job"]["lr_schedule_horizon_epochs"] == 20
    assert rq1["rows"][0]["job"]["objective"] == {
        "id": "rq1_24h",
        "window_seconds": 86400,
    }
    assert rq1["rows"][0]["job"]["loss"] == {
        "valid_positive_mask_mode": "next_24h_unique"
    }
    assert rq3["rows"][0]["job"]["objective"] == {
        "id": "rq3_learned_hard",
        "selector_artifact_sha256": SHA_C,
        "period_count": 1,
    }
    assert rq3["materialization_cost_evidence_sha256"] == canonical_sha256(
        _materialization_evidence()
    )
    assert (
        build_training_experiment(
            CompiledJob.from_row(ledger_sha256=rq3["sha256"], row=rq3["rows"][0])
        ).objective_id
        == "rq3_learned_hard"
    )

    with pytest.raises(ValueError, match="objective"):
        compile_treatment_tuning_ledger(
            objective_id="not-approved",  # type: ignore[arg-type]
            treatment_semantics_manifest_sha256=SHA_B,
            **selected,
        )
    with pytest.raises(ValueError, match="materialization evidence"):
        compile_treatment_tuning_ledger(
            objective_id="rq3_learned_hard",
            treatment_semantics_manifest_sha256=SHA_B,
            selector_artifact_sha256=SHA_C,
            **selected,
        )
    forged_limits = _materialization_evidence() | {
        "runtime": {
            "wall_seconds": 99 * 60 * 60,
            "peak_aggregate_rss_bytes": 900 * 2**30,
        },
        "logical_output_scratch_bytes": 900 * 2**30,
        "limits": {
            "wall_seconds": 100 * 60 * 60,
            "peak_aggregate_rss_bytes": 1000 * 2**30,
            "logical_output_scratch_bytes": 1000 * 2**30,
        },
    }
    with pytest.raises(ValueError, match="limits differ"):
        compile_treatment_tuning_ledger(
            objective_id="rq3_learned_hard",
            treatment_semantics_manifest_sha256=SHA_B,
            selector_artifact_sha256=SHA_C,
            materialization_cost_evidence=forged_limits,
            **selected,
        )


def test_materialization_artifacts_are_bound_to_measurement(tmp_path: Path) -> None:
    deterministic = write_period_artifact(
        [],
        selector_kind="deterministic",
        selected_configuration={},
        provenance={},
        cost={"measurement_id": SHA_E},
        output_root=tmp_path,
    )
    learned = write_period_artifact(
        [],
        selector_kind="learned",
        selected_configuration={},
        provenance={},
        cost={"measurement_id": SHA_E},
        output_root=tmp_path,
    )
    evidence = _materialization_evidence() | {
        "deterministic_artifact_sha256": deterministic.sha256,
        "learned_artifact_sha256": learned.sha256,
    }

    _verify_materialization_artifacts(evidence, tmp_path)
    with pytest.raises(ValueError, match="measurement identity"):
        _verify_materialization_artifacts(
            evidence | {"measurement_id": SHA_A}, tmp_path
        )


@pytest.mark.parametrize(
    ("objective", "mask", "window_seconds", "event_lookahead"),
    [
        (
            {"id": "rq1_24h", "window_seconds": 86400},
            "next_24h_unique",
            86400,
            None,
        ),
        (
            {"id": "rq2_next10", "event_lookahead": 10},
            "next_10_unique",
            None,
            10,
        ),
    ],
)
def test_treatment_config_constructs_the_exact_epoch_aware_target(
    objective: dict[str, object],
    mask: str,
    window_seconds: int | None,
    event_lookahead: int | None,
) -> None:
    experiment = build_treatment(
        objective=objective,
        valid_positive_mask_mode=mask,
        run_name=f"g4_{objective['id']}_unit",
        batch_size=512,
        embedding_learning_rate=0.02,
        deep_learning_rate=0.01,
        lr_schedule_horizon_epochs=20,
        seed=42,
    )
    experiment.__dict__["validation_cutoff_timestamp"] = 1_700_000_000
    experiment.__dict__["future_event_index"] = FutureEventIndex.from_columns(
        [], [], []
    )
    targets = experiment.create_targets()

    assert targets.objective_id == objective["id"]
    assert targets.window_seconds == window_seconds
    assert targets.event_lookahead == event_lookahead
    assert targets.training_seed == 42
    assert targets.training_cutoff_timestamp == 1_700_000_000
    assert experiment.emit_user_column is True


def test_compiled_payload_is_bound_to_one_exact_ledger_row(tmp_path: Path) -> None:
    ledger = compile_control_tuning_ledger(SHA_A)
    path = tmp_path / "control.json"
    write_frozen_ledger(path, ledger)
    compiled = CompiledJob.from_row(
        ledger_sha256=ledger["sha256"],
        row=resolve_ledger_row(path, "control_tuning:02"),
    )

    decoded = decode_job(encode_job(compiled), path)
    experiment = build_training_experiment(decoded)

    assert decoded == compiled
    assert experiment.run_name == compiled.job["run_name"]
    assert experiment.seed == 42
    assert experiment.dataloader.batch_size == compiled.job["dataloader"]["batch_size"]

    tampered = replace(
        compiled,
        job=compiled.job | {"embedding_learning_rate": 0.125},
    )
    with pytest.raises(ValueError, match="ledger row"):
        decode_job(encode_job(tampered), path)


def test_semantics_and_selection_manifests_are_closed_and_immutable(
    tmp_path: Path,
) -> None:
    semantics = _control_semantics_fixture(tmp_path)
    semantics_path = tmp_path / "semantics.json"
    write_frozen_manifest(semantics_path, semantics)
    semantics_sha = canonical_sha256(semantics)
    ledger_path, run_directories = _completed_control_run(
        tmp_path, control_semantics_manifest_sha256=semantics_sha
    )
    selected = build_selected_control_manifest(
        control_semantics_manifest_sha256=semantics_sha,
        ledger_paths=[ledger_path],
        run_directories=run_directories,
    )
    selected_path = tmp_path / "selected.json"
    write_frozen_manifest(selected_path, selected)

    assert selected["seed_42_configuration_sha256"] == canonical_sha256(
        selected["seed_42_configuration"]
    )
    assert selected["selection"]["best_epoch"] == 2
    assert selected["selection"]["epochs_trained"] == 10
    selected_run_directory = next(
        path
        for path in run_directories
        if path.name == selected["selection"]["run_name"]
    )
    metadata_path = selected_run_directory / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["epochs_trained"] = 19
    metadata["lr_horizon_complete"] = False
    metadata["selection_resolved"] = False
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not usable"):
        build_selected_control_manifest(
            control_semantics_manifest_sha256=semantics_sha,
            ledger_paths=[ledger_path],
            run_directories=run_directories,
        )
    with pytest.raises(RuntimeError, match="frozen manifest differs"):
        write_frozen_manifest(selected_path, selected | {"version": 3})


def test_selected_control_rejects_incomplete_or_caller_chosen_evidence(
    tmp_path: Path,
) -> None:
    ledger_path, run_directories = _completed_control_run(
        tmp_path,
        control_semantics_manifest_sha256=SHA_A,
    )
    with pytest.raises(ValueError, match="candidate set is incomplete"):
        build_selected_control_manifest(
            control_semantics_manifest_sha256=SHA_A,
            ledger_paths=[ledger_path],
            run_directories=run_directories[:-1],
        )

    selected = build_selected_control_manifest(
        control_semantics_manifest_sha256=SHA_A,
        ledger_paths=[ledger_path],
        run_directories=run_directories,
    )
    forged = json.loads(json.dumps(selected))
    forged["selection"]["row_id"] = "control_tuning:01"
    forged["selection"]["validation_recall_at_100"] = 99.0
    forged["selection"]["canonical_parameters"][
        "lr_schedule_horizon_epochs"
    ] = 49
    with pytest.raises(ValueError, match="evidence|reproduce|configuration"):
        _validate_selected_control_document(forged)


def test_selected_control_requires_a_triggered_boundary_ledger(tmp_path: Path) -> None:
    ledger_path, run_directories = _completed_control_run(
        tmp_path,
        control_semantics_manifest_sha256=SHA_A,
        winner_row_id="control_tuning:01",
    )

    with pytest.raises(ValueError, match="boundary candidate set is incomplete"):
        build_selected_control_manifest(
            control_semantics_manifest_sha256=SHA_A,
            ledger_paths=[ledger_path],
            run_directories=run_directories,
        )


def test_recommender_boundary_ledger_freezes_every_nontriggered_axis() -> None:
    base = compile_treatment_tuning_ledger(
        objective_id="rq2_next10",
        selected_control_manifest_sha256=SHA_A,
        treatment_semantics_manifest_sha256=SHA_B,
        batch_size=512,
        embedding_learning_rate=0.0001,
        deep_learning_rate=0.02,
        lr_schedule_horizon_epochs=20,
    )
    entering = base["rows"][0]
    boundary = _compile_recommender_boundary_ledger(
        entering_row=entering,
        selected_control_manifest_sha256=SHA_A,
        treatment_semantics_manifest_sha256=SHA_B,
        boundary_round=1,
    )

    assert len(boundary["rows"]) == 4
    assert boundary["rate_bounds"]["embedding_learning_rate"] == [0.000025, 0.256]
    assert boundary["horizon_values"] == [20]
    for row in boundary["rows"]:
        job = row["job"]
        assert job["protocol"]["stage"] == "rq2_tuning_boundary"
        assert job["protocol"]["boundary_round"] == 1
        assert job["dataloader"] == entering["job"]["dataloader"]
        assert job["objective"] == entering["job"]["objective"]
        assert job["loss"] == entering["job"]["loss"]
        assert job["deep_learning_rate"] == entering["job"]["deep_learning_rate"]
        assert job["lr_schedule_horizon_epochs"] == 20
    compiled = CompiledJob.from_row(
        ledger_sha256=boundary["sha256"], row=boundary["rows"][0]
    )
    assert build_training_experiment(compiled).objective_id == "rq2_next10"


@pytest.mark.parametrize(
    ("base_horizon", "round_one", "round_one_winner", "round_two"),
    [
        (5, list(range(2, 31)), 3, list(range(1, 31))),
        (30, list(range(5, 41)), 38, list(range(5, 51))),
    ],
)
def test_recommender_horizon_boundary_uses_the_exact_two_round_surface(
    base_horizon: int,
    round_one: list[int],
    round_one_winner: int,
    round_two: list[int],
) -> None:
    base = compile_control_tuning_ledger(SHA_A)
    entering = json.loads(json.dumps(base["rows"][0]))
    entering["job"]["embedding_learning_rate"] = 0.0001
    entering["job"]["deep_learning_rate"] = 0.01
    entering["job"]["lr_schedule_horizon_epochs"] = base_horizon
    first = _compile_recommender_boundary_ledger(
        entering_row=entering,
        control_semantics_manifest_sha256=SHA_A,
        boundary_round=1,
    )

    assert first["horizon_values"] == round_one
    assert {
        row["job"]["lr_schedule_horizon_epochs"] for row in first["rows"]
    } <= set(round_one)

    second_entering = json.loads(json.dumps(first["rows"][0]))
    second_entering["job"]["lr_schedule_horizon_epochs"] = round_one_winner
    second = _compile_recommender_boundary_ledger(
        entering_row=second_entering,
        control_semantics_manifest_sha256=SHA_A,
        boundary_round=2,
        entering_rate_bounds=first["rate_bounds"],
        entering_horizon_values=first["horizon_values"],
    )

    assert second["horizon_values"] == round_two
    assert {
        row["job"]["lr_schedule_horizon_epochs"] for row in second["rows"]
    } <= set(round_two)


def test_recommender_horizon_does_not_retrigger_at_an_old_endpoint() -> None:
    base = compile_control_tuning_ledger(SHA_A)
    entering = json.loads(json.dumps(base["rows"][0]))
    entering["job"]["embedding_learning_rate"] = 0.0001
    entering["job"]["deep_learning_rate"] = 0.01
    entering["job"]["lr_schedule_horizon_epochs"] = 5
    first = _compile_recommender_boundary_ledger(
        entering_row=entering,
        control_semantics_manifest_sha256=SHA_A,
        boundary_round=1,
    )
    second_entering = json.loads(json.dumps(first["rows"][0]))
    second_entering["job"]["embedding_learning_rate"] = 0.000025
    second_entering["job"]["lr_schedule_horizon_epochs"] = 5

    second = _compile_recommender_boundary_ledger(
        entering_row=second_entering,
        control_semantics_manifest_sha256=SHA_A,
        boundary_round=2,
        entering_rate_bounds=first["rate_bounds"],
        entering_horizon_values=first["horizon_values"],
    )

    assert second["horizon_values"] == [5]


def _selector_trials() -> list[dict[str, object]]:
    result = []
    for trial in compile_selector_search(seed=42):
        row = trial.to_dict()
        row.pop("stage")
        row.pop("run_name")
        result.append(row)
    return result


def test_selector_ledgers_close_search_gate_and_crossfit_materialization() -> None:
    search = compile_selector_search_ledger(
        trials=_selector_trials(),
        treatment_semantics_manifest_sha256=SHA_A,
        input_artifact_sha256=SHA_B,
    )
    assert search == compile_selector_search_ledger(
        treatment_semantics_manifest_sha256=SHA_A,
        input_artifact_sha256=SHA_B,
    )
    gate = compile_selector_gate_ledger(
        treatment_semantics_manifest_sha256=SHA_A,
        deterministic_artifact_sha256=SHA_C,
        deterministic_payload_sha256=SHA_A,
        learned_artifact_sha256=SHA_D,
        learned_payload_sha256=SHA_B,
    )
    materialization = compile_selector_materialization_ledger(
        treatment_semantics_manifest_sha256=SHA_A,
        selected_configuration=_selector_trials()[-1],
        selector_gate_artifact_sha256=SHA_E,
        selector_gate_payload_sha256=SHA_C,
    )

    assert len(search["rows"]) == 48
    assert {row["job"]["family"] for row in search["rows"]} == {
        "time",
        "content",
        "frequency",
        "learned",
    }
    assert len(gate["rows"]) == 1
    assert gate["rows"][0]["job"]["stage"] == "selector_gate"
    assert gate["rows"][0]["job"]["input_artifact_sha256"] is None
    assert gate["rows"][0]["job"]["deterministic_artifact_sha256"] == (SHA_C)
    assert gate["rows"][0]["job"]["deterministic_payload_sha256"] == SHA_A
    assert gate["rows"][0]["job"]["learned_artifact_sha256"] == SHA_D
    assert [row["job"]["fold_id"] for row in materialization["rows"]] == list(range(5))
    assert all(
        row["job"]["input_artifact_sha256"] == SHA_E for row in materialization["rows"]
    )
    assert all(
        row["job"]["input_payload_sha256"] == SHA_C for row in materialization["rows"]
    )


def test_selector_boundary_resamples_only_learned_rate() -> None:
    selected = _selector_trials()[-1] | {"learning_rate": 0.01}
    search = compile_selector_search_ledger(
        trials=[
            (
                trial
                if trial["family"] != "learned" or trial["trial_id"] != 12
                else selected
            )
            for trial in _selector_trials()
        ],
        treatment_semantics_manifest_sha256=SHA_A,
        input_artifact_sha256=SHA_B,
    )
    entering = search["rows"][-1]
    boundary = _compile_selector_boundary_ledger(
        entering_row=entering,
        treatment_semantics_manifest_sha256=SHA_A,
        boundary_round=1,
    )

    assert boundary["rate_bounds"] == {"learning_rate": [0.0025, 0.2]}
    for row in boundary["rows"]:
        job = row["job"]
        assert job["stage"] == "selector_search_boundary"
        assert job["boundary_round"] == 1
        assert job["family"] == "learned"
        assert job["l2_regularization"] == entering["job"]["l2_regularization"]
        assert job["period_width_seconds"] == entering["job"]["period_width_seconds"]


def test_launch_verifier_requires_every_referenced_semantics_hash(
    tmp_path: Path,
) -> None:
    semantics = _control_semantics_fixture(tmp_path)
    semantics_path = tmp_path / "semantics.json"
    write_frozen_manifest(semantics_path, semantics)
    ledger = compile_control_tuning_ledger(canonical_sha256(semantics))

    verify_ledger_semantics(
        ledger,
        {"control_semantics_manifest_sha256": semantics_path},
    )
    with pytest.raises(ValueError, match="semantics manifest hash"):
        verify_ledger_semantics(
            ledger,
            {"control_semantics_manifest_sha256": tmp_path / "missing.json"},
        )


def test_launch_verifier_rejects_current_source_and_data_tampering(
    tmp_path: Path,
) -> None:
    semantics = _control_semantics_fixture(tmp_path)
    source_tampered = json.loads(json.dumps(semantics))
    source_name = source_tampered["source_paths"][0]
    source_tampered["sources"][source_name] = SHA_A
    source_path = tmp_path / "source-tampered.json"
    write_frozen_manifest(source_path, source_tampered)
    source_ledger = compile_control_tuning_ledger(canonical_sha256(source_tampered))
    with pytest.raises(ValueError, match="current control source hashes"):
        verify_ledger_semantics(
            source_ledger,
            {"control_semantics_manifest_sha256": source_path},
        )

    data_tampered = json.loads(json.dumps(semantics))
    data_tampered["data_identity"]["main"]["sha256"] = SHA_A
    data_path = tmp_path / "data-tampered.json"
    write_frozen_manifest(data_path, data_tampered)
    data_ledger = compile_control_tuning_ledger(canonical_sha256(data_tampered))
    with pytest.raises(ValueError, match="current control data identity"):
        verify_ledger_semantics(
            data_ledger,
            {"control_semantics_manifest_sha256": data_path},
        )


def test_launch_verifier_resolves_the_transitive_treatment_chain(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[4]
    control = _control_semantics_fixture(tmp_path)
    control_path = tmp_path / "control-semantics.json"
    write_frozen_manifest(control_path, control)
    ledger_path, run_directories = _completed_control_run(
        tmp_path,
        control_semantics_manifest_sha256=canonical_sha256(control),
    )
    selected = build_selected_control_manifest(
        control_semantics_manifest_sha256=canonical_sha256(control),
        ledger_paths=[ledger_path],
        run_directories=run_directories,
    )
    selected_path = tmp_path / "selected-control.json"
    write_frozen_manifest(selected_path, selected)
    preimplementation = json.loads(
        (
            root
            / "experiments/g4_future_items/protocol/preimplementation_source_manifest.json"
        ).read_text()
    )["paths"]
    source_paths = sorted(set(expected_control_source_paths()) | set(preimplementation))
    entrypoints = {
        entrypoint: source_paths
        for entrypoint in (
            "experiments/g4_future_items/launchers/run_control.py",
            "experiments/g4_future_items/launchers/run_selectors.py",
            "experiments/g4_future_items/launchers/run_treatments.py",
        )
    }
    treatment = build_treatment_semantics_manifest(
        selected_control_manifest_sha256=canonical_sha256(selected),
        entrypoint_source_paths=entrypoints,
        post_review_sources=source_manifest(root, source_paths),
        schema_revisions={"artifact": 1, "target_rng": "g4-target-v1"},
        fixture_paths={
            "control_manifest": (
                "experiments/g4_future_items/protocol/control_manifest.json"
            )
        },
    )
    treatment_path = tmp_path / "treatment-semantics.json"
    write_frozen_manifest(treatment_path, treatment)
    selected_parameters = selected["selection"]["canonical_parameters"]
    ledger = compile_treatment_tuning_ledger(
        objective_id="rq1_24h",
        selected_control_manifest_sha256=canonical_sha256(selected),
        treatment_semantics_manifest_sha256=canonical_sha256(treatment),
        batch_size=selected_parameters["batch_size"],
        embedding_learning_rate=selected_parameters["embedding_learning_rate"],
        deep_learning_rate=selected_parameters["deep_learning_rate"],
        lr_schedule_horizon_epochs=selected_parameters[
            "lr_schedule_horizon_epochs"
        ],
    )
    references = {
        "control_semantics_manifest_sha256": control_path,
        "selected_control_manifest_sha256": selected_path,
        "treatment_semantics_manifest_sha256": treatment_path,
    }

    verify_ledger_semantics(ledger, references)
    wrong_anchor = compile_treatment_tuning_ledger(
        objective_id="rq1_24h",
        selected_control_manifest_sha256=canonical_sha256(selected),
        treatment_semantics_manifest_sha256=canonical_sha256(treatment),
        batch_size=512,
        embedding_learning_rate=0.01,
        deep_learning_rate=0.01,
        lr_schedule_horizon_epochs=20,
    )
    with pytest.raises(ValueError, match="treatment anchor"):
        verify_ledger_semantics(wrong_anchor, references)
    with pytest.raises(ValueError, match="control_semantics_manifest_sha256"):
        verify_ledger_semantics(
            ledger,
            {
                key: value
                for key, value in references.items()
                if key != "control_semantics_manifest_sha256"
            },
        )

    fixture_tampered = json.loads(json.dumps(treatment))
    fixture_tampered["fixtures"]["control_manifest"]["sha256"] = SHA_A
    fixture_path = tmp_path / "fixture-tampered-treatment.json"
    write_frozen_manifest(fixture_path, fixture_tampered)
    fixture_ledger = compile_treatment_tuning_ledger(
        objective_id="rq1_24h",
        selected_control_manifest_sha256=canonical_sha256(selected),
        treatment_semantics_manifest_sha256=canonical_sha256(fixture_tampered),
        batch_size=512,
        embedding_learning_rate=0.16590060219780284,
        deep_learning_rate=0.02879154157702692,
        lr_schedule_horizon_epochs=20,
    )
    with pytest.raises(ValueError, match="current treatment fixture"):
        verify_ledger_semantics(
            fixture_ledger,
            references | {"treatment_semantics_manifest_sha256": fixture_path},
        )


def test_launch_verifier_accepts_only_proven_treatment_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[4]
    protocol = root / "experiments/g4_future_items/protocol"
    bridge_path = protocol / "treatment_semantics_compatibility_v10.json"
    evidence_path = (
        root
        / "experiments/g4_future_items/evidence/runtime_equivalence_native50m_v9.json"
    )
    bridge = load_strict_json(bridge_path)
    evidence = load_strict_json(evidence_path)
    changes = bridge["source_changes"]
    assert evidence == build_runtime_compatibility_evidence(
        predecessor_treatment_sha256=bridge["historical_lineage"][
            "treatment_semantics"
        ]["sha256"],
        source_changes=changes,
    )
    assert bridge == build_treatment_compatibility_manifest(
        predecessor_treatment_path=protocol / "treatment_semantics_manifest.json",
        selected_control_path=protocol / "selected_control_manifest.json",
        control_semantics_path=protocol / "control_semantics_manifest.json",
        compatibility_evidence_path=evidence_path,
        approved_source_changes=[change["path"] for change in changes],
    )
    original_load_ledger = manifest_module.load_ledger

    def replaced_ledger(path: Path) -> dict[str, object]:
        document = original_load_ledger(path)
        if path.name != "rq1_tuning.json":
            return document
        replacement = json.loads(json.dumps(document))
        replacement["rows"][0]["job"]["embedding_learning_rate"] = 0.125
        replacement["sha256"] = canonical_sha256(
            {key: value for key, value in replacement.items() if key != "sha256"}
        )
        return replacement

    monkeypatch.setattr(manifest_module, "load_ledger", replaced_ledger)
    with pytest.raises(ValueError, match="frozen compatibility ledger identity"):
        build_runtime_compatibility_evidence(
            predecessor_treatment_sha256=bridge["historical_lineage"][
                "treatment_semantics"
            ]["sha256"],
            source_changes=changes,
        )
    monkeypatch.setattr(manifest_module, "load_ledger", original_load_ledger)
    selector_ledger = compile_selector_search_ledger(
        treatment_semantics_manifest_sha256=canonical_sha256(bridge),
        input_artifact_sha256=SHA_A,
    )
    verify_ledger_semantics(
        selector_ledger,
        {"treatment_semantics_manifest_sha256": bridge_path},
    )
    historical_control = load_ledger(protocol / "ledgers/control_tuning.json")
    verify_ledger_semantics(
        historical_control,
        {
            "control_semantics_manifest_sha256": (
                protocol / "control_semantics_manifest.json"
            )
        },
        compatibility_path=bridge_path,
    )
    selected = load_strict_json(protocol / "selected_control_manifest.json")
    parameters = selected["selection"]["canonical_parameters"]
    treatment_ledger = compile_treatment_tuning_ledger(
        objective_id="rq1_24h",
        selected_control_manifest_sha256=canonical_sha256(selected),
        treatment_semantics_manifest_sha256=canonical_sha256(bridge),
        batch_size=parameters["batch_size"],
        embedding_learning_rate=parameters["embedding_learning_rate"],
        deep_learning_rate=parameters["deep_learning_rate"],
        lr_schedule_horizon_epochs=parameters["lr_schedule_horizon_epochs"],
    )
    verify_ledger_semantics(
        treatment_ledger,
        {
            "control_semantics_manifest_sha256": (
                protocol / "control_semantics_manifest.json"
            ),
            "selected_control_manifest_sha256": (
                protocol / "selected_control_manifest.json"
            ),
            "treatment_semantics_manifest_sha256": bridge_path,
        },
    )
    original_identity = manifest_module._current_control_data_identity
    monkeypatch.setattr(
        manifest_module,
        "_current_control_data_identity",
        lambda identity: original_identity(identity) | {"dataset_key": "changed"},
    )
    with pytest.raises(ValueError, match="current control data identity"):
        verify_ledger_semantics(
            selector_ledger,
            {"treatment_semantics_manifest_sha256": bridge_path},
        )
    monkeypatch.setattr(
        manifest_module, "_current_control_data_identity", original_identity
    )
    tampered = json.loads(json.dumps(evidence))
    tampered["recommender_rows"][0]["job_sha256"] = SHA_A
    with pytest.raises(ValueError, match="does not reproduce"):
        manifest_module._validate_compatibility_evidence(
            tampered,
            predecessor_treatment_sha256=bridge["historical_lineage"][
                "treatment_semantics"
            ]["sha256"],
            source_changes=changes,
        )


def test_compatibility_closure_rejects_an_unapproved_new_local_import(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    entrypoint = package / "entrypoint.py"
    entrypoint.write_text("from package import added\n")
    added = package / "added.py"
    added.write_text("VALUE = 1\n")

    closures = manifest_module.derive_current_entrypoint_source_paths(
        {"package/entrypoint.py": ["package/__init__.py", "package/entrypoint.py"]},
        project_root=tmp_path,
    )
    additions = [
        {
            "path": "package/added.py",
            "after_sha256": hashlib.sha256(added.read_bytes()).hexdigest(),
        }
    ]

    assert closures["package/entrypoint.py"] == [
        "package/__init__.py",
        "package/added.py",
        "package/entrypoint.py",
    ]
    with pytest.raises(ValueError, match="source additions are not approved"):
        manifest_module._validate_compatibility_source_additions(additions)


def test_treatment_semantics_derives_exact_changed_paths(tmp_path: Path) -> None:
    entrypoints = sorted(
        {
            "experiments/g4_future_items/launchers/run_control.py",
            "experiments/g4_future_items/launchers/run_selectors.py",
            "experiments/g4_future_items/launchers/run_treatments.py",
        }
    )
    shared = "dcn/shared.py"
    for relative in [*entrypoints, shared]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"identity = {relative!r}\n")
    fixture = tmp_path / "fixtures/target.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("{}\n")
    source_paths = sorted([*entrypoints, shared])
    before_path = tmp_path / "preimplementation.json"
    before_path.write_text(
        json.dumps(
            {
                "canonicalization": "cpython-3.12-json-v1",
                "paths": {path: SHA_A for path in source_paths},
            },
            sort_keys=True,
        )
    )
    closures = {entrypoint: sorted([entrypoint, shared]) for entrypoint in entrypoints}
    document = build_treatment_semantics_manifest(
        selected_control_manifest_sha256=SHA_B,
        entrypoint_source_paths=closures,
        post_review_sources=source_manifest(tmp_path, source_paths),
        schema_revisions={"artifact": 1, "target_rng": "g4-target-v1"},
        fixture_paths={"target": "fixtures/target.json"},
        control_source_paths=source_paths,
        project_root=tmp_path,
        preimplementation_path=before_path,
    )

    assert document["source_paths"] == source_paths
    assert [change["path"] for change in document["changed_paths"]] == source_paths
    assert (
        document["fixtures"]["target"]["sha256"]
        == hashlib.sha256(fixture.read_bytes()).hexdigest()
    )


def test_compiled_cli_dry_run_emits_persistent_queue_submission(tmp_path: Path) -> None:
    semantics = _control_semantics_fixture(tmp_path)
    semantics_path = tmp_path / "semantics.json"
    write_frozen_manifest(semantics_path, semantics)
    ledger = compile_control_tuning_ledger(canonical_sha256(semantics))
    ledger_path = tmp_path / "ledger.json"
    write_frozen_ledger(ledger_path, ledger)

    result = subprocess.run(
        [
            "python",
            "-m",
            "experiments.g4_future_items.launchers.compiled",
            "submit",
            str(ledger_path),
            "control_tuning:01",
            "--semantics",
            f"control_semantics_manifest_sha256={semantics_path}",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[4],
        env=os.environ
        | {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert "new-batch" in result.stdout
    assert "enqueue-run" in result.stdout
    assert "run_control.py" in result.stdout
    assert "G4_COMPILED_JOB_B64=" in result.stdout
    assert "WANDB_MODE=offline" in result.stdout
