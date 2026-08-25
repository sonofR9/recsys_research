import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.g2_esasrec.analysis.evidence import (
    METRICS,
    OFFICIAL_LOCAL_SOURCE_SHA256,
    aggregate_artifacts,
    control_band_artifacts,
    empirical_bands,
    load_verified_artifact,
    select_best,
    select_control_with_fit_gate,
)
from experiments.g2_esasrec.analysis.fit_evidence import (
    FIT_DEVICE_COMPUTE_CAPABILITY,
    FIT_DEVICE_NAME,
    fit_device_evidence,
    load_fit_evidence,
    run_one_step_fit_probe,
    write_fit_evidence,
)
from experiments.g2_esasrec.analysis.report import (
    render_compact_report,
    render_tuning_ledger,
)
from experiments.g2_esasrec.configs.local import COMPONENT_METHODS
from experiments.g2_esasrec.launchers.compiled import persisted_job_contract
from experiments.g2_esasrec.official.provenance import RECTOOLS_SOURCE_SHA256
from experiments.g2_esasrec.protocol.manifest import CompiledJob, approved_manifest


def _write_artifact(
    logs,
    job,
    *,
    recall,
    ndcg,
    coverage=0.2,
    resolved=True,
    batch_size=128,
    parameters_override=None,
):
    directory = logs / job.run_name
    directory.mkdir(parents=True)
    metrics = {metric: 0.1 for metric in METRICS} | {"num_users": 3414.0}
    metrics.update({"recall@100": recall, "ndcg@100": ndcg, "coverage@100": coverage})
    (directory / "final_metrics.json").write_text(json.dumps(metrics))
    (directory / "cost_metrics.json").write_text(
        json.dumps(
            {
                "params_total": 1_000_000,
                "params_deep": 500_000,
                "training_seconds": 12.0,
                "wall_seconds": 13.0,
                "median_train_epoch_seconds": 1.0,
                "peak_memory_gb": 2.0,
                "targets_per_second": 1000.0,
                "best_epoch": 4,
            }
        )
    )
    parameters = parameters_override
    if parameters is None:
        parameters = {
            "batch_size": batch_size,
            "embedding_learning_rate": 0.001,
            "deep_learning_rate": 0.001,
        }
        if job.stage == "official":
            parameters = {"rectools_version": "0.19.0"}
        elif job.stage == "control_repeats":
            parameters |= {
                "selected_control": True,
                "selected_control_job_id": "control_tuning:control_trial_01",
            }
        elif job.stage == "component_tuning":
            parameters["selected_control_job_id"] = "control_tuning:control_trial_01"
            if job.method.startswith(("ligr_", "matched_standard_")):
                parameters["ligr_multiplier"] = 4
            if job.method.endswith("_gbce"):
                parameters["gbce_t"] = 0.75
            parameters.update(job.forced_parameters)
    parameters = dict(parameters)
    if (
        job.stage == "component_tuning"
        and (job.method.startswith("matched_standard_") or job.method == "ligr_gbce")
        and "source_job_id" not in parameters
    ):
        ligr_source = next(
            candidate
            for candidate in approved_manifest().jobs_for_stage("component_tuning")
            if candidate.method == "ligr_sampled_softmax" and candidate.trial == 1
        )
        parameters["source_job_id"] = ligr_source.id
        if not (logs / ligr_source.run_name).exists():
            _write_artifact(logs, ligr_source, recall=0.1, ndcg=0.1)
    selected_control_id = parameters.get("selected_control_job_id")
    if isinstance(selected_control_id, str):
        selected_control = next(
            candidate
            for candidate in approved_manifest().jobs
            if candidate.id == selected_control_id
        )
        if not (logs / selected_control.run_name).exists():
            _write_artifact(
                logs,
                selected_control,
                recall=0.1,
                ndcg=0.1,
                batch_size=batch_size,
            )
    compiled = CompiledJob(job, parameters)
    contract = persisted_job_contract(compiled)
    if job.stage == "official":
        metadata = {
            "dataset_size": "native-50m",
            "seed": job.seed,
            "early_stopped": resolved,
            "best_epoch_at_cap": False,
            "best_epoch": 4,
            "epochs_trained": 14,
            "max_epochs": 100,
            "patience": 10,
            "wall_seconds": 13.0,
            "implementation": "RecTools SASRecModel with LiGRLayers",
            "hyperparameters": {
                "n_blocks": 2,
                "n_heads": 4,
                "n_factors": 256,
                "dropout_rate": 0.2,
                "session_max_len": 100,
                "n_negatives": 256,
                "lr": 0.001,
                "batch_size": 128,
            },
            "protocol": {
                "cutoff": 25394930,
                "catalog_size": 33148,
                "model_candidate_catalog_size": 33148,
                "train_catalog_size": 33112,
                "mapped_items_absent_from_training": 36,
                "candidate_catalog_sha256": (
                    "fa5acc91da974d077fb8c870ea4d4fc776efebd2ea374d8c3b0d23977ea1c831"
                ),
                "candidate_catalog_source": "full mapped pre-split catalog",
                "train_events": 614244,
                "model_train_events_after_session_truncation": 362723,
                "validation_events": 20398,
                "evaluable_users": 3414,
                "eval_ks": [10, 50, 100],
                "selection_metric": "recall@100",
                "exclude_seen": False,
            },
            "provenance": {
                "environment": {
                    "python": "3.12.13",
                    "packages": {
                        "RecTools": "0.19.0",
                        "torch": "2.7.1",
                        "pytorch-lightning": "2.5.2",
                        "numpy": "1.26.4",
                        "pandas": "2.2.3",
                        "polars": "1.43.2",
                    },
                },
                "sources": {
                    name: {"sha256": sha256, "path": f"/{name}.py"}
                    for name, sha256 in (
                        RECTOOLS_SOURCE_SHA256 | OFFICIAL_LOCAL_SOURCE_SHA256
                    ).items()
                },
            },
        }
    else:
        training = contract["local_implementation"]["training"]
        metadata = {
            "dataset_size": "50m",
            "seed": job.seed,
            "selection_resolved": resolved,
            "best_epoch": 4,
            "best_epoch_at_cap": False,
            "epochs_trained": 14,
            "max_epochs": training["max_epochs"],
            "initializer_std": training["initializer_std"],
            "weight_decay": training["weight_decay"],
            "runtime_dtype": training["runtime_dtype"],
            "runtime_compile": training["runtime_compile"],
            "embedding_learning_rate": parameters["embedding_learning_rate"],
            "deep_learning_rate": parameters["deep_learning_rate"],
            "batch_size": parameters["batch_size"],
            "transfer_invariants": contract["local_implementation"][
                "transfer_invariants"
            ],
        }
        if training["layer_family"] != "g1_control":
            metadata["g2_recipe"] = {
                name: training[name] for name in ("layer_family", "loss_kind", "gbce_t")
            }
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "g2_job.json").write_text(
        json.dumps(
            {
                **contract,
            }
        )
    )
    return directory


def test_artifact_verifier_requires_native_50m_resolved_metrics_and_contract(tmp_path):
    job = approved_manifest().jobs_for_stage("component_tuning")[1]
    directory = _write_artifact(tmp_path, job, recall=0.2, ndcg=0.1)

    artifact = load_verified_artifact(job, tmp_path)
    assert artifact.metrics["recall@100"] == 0.2

    metadata_path = directory / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["selection_resolved"] = False
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="selection-resolved"):
        load_verified_artifact(job, tmp_path)


def test_artifact_verifier_pins_denominator_ranges_and_local_implementation(
    tmp_path,
):
    job = approved_manifest().jobs_for_stage("component_tuning")[1]
    directory = _write_artifact(tmp_path, job, recall=0.2, ndcg=0.1)

    metrics_path = directory / "final_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["num_users"] = 3413
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="3,414"):
        load_verified_artifact(job, tmp_path)

    metrics["num_users"] = 3414
    metrics["recall@100"] = 1.01
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_verified_artifact(job, tmp_path)

    metrics["recall@100"] = 0.2
    metrics_path.write_text(json.dumps(metrics))
    contract_path = directory / "g2_job.json"
    contract = json.loads(contract_path.read_text())
    source = next(iter(contract["local_implementation"]["sources"]))
    contract["local_implementation"]["sources"][source] = "0" * 64
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="local implementation"):
        load_verified_artifact(job, tmp_path)


def test_artifact_verifier_rejects_an_omitted_local_source(tmp_path):
    job = approved_manifest().jobs_for_stage("component_tuning")[1]
    directory = _write_artifact(tmp_path, job, recall=0.2, ndcg=0.1)
    contract_path = directory / "g2_job.json"
    contract = json.loads(contract_path.read_text())
    contract["local_implementation"]["sources"].popitem()
    contract_path.write_text(json.dumps(contract))

    with pytest.raises(ValueError, match="local implementation"):
        load_verified_artifact(job, tmp_path)


def test_artifact_verifier_pins_the_executed_g2_recipe(tmp_path):
    job = approved_manifest().jobs_for_stage("component_tuning")[1]
    directory = _write_artifact(tmp_path, job, recall=0.2, ndcg=0.1)
    metadata_path = directory / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["g2_recipe"]["loss_kind"] = "gbce"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="executed G2 recipe"):
        load_verified_artifact(job, tmp_path)


def test_artifact_verifier_requires_the_selected_batch_to_propagate(tmp_path):
    job = approved_manifest().jobs_for_stage("component_tuning")[1]
    selected = approved_manifest().jobs_for_stage("control_tuning")[1]
    _write_artifact(tmp_path, selected, recall=0.1, ndcg=0.1, batch_size=128)
    _write_artifact(tmp_path, job, recall=0.2, ndcg=0.1, batch_size=256)

    with pytest.raises(ValueError, match="batch_size was not propagated"):
        load_verified_artifact(job, tmp_path)


def test_artifact_verifier_pins_official_version_sources_recipe_and_protocol(
    tmp_path,
):
    job = approved_manifest().jobs_for_stage("official")[0]
    directory = _write_artifact(tmp_path, job, recall=0.2, ndcg=0.1)

    load_verified_artifact(job, tmp_path)
    metadata_path = directory / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["hyperparameters"]["n_negatives"] = 255
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="official hyperparameters"):
        load_verified_artifact(job, tmp_path)

    metadata["hyperparameters"]["n_negatives"] = 256
    metadata["protocol"]["evaluable_users"] = 3413
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="official protocol"):
        load_verified_artifact(job, tmp_path)

    metadata["protocol"]["evaluable_users"] = 3414
    metadata["provenance"]["sources"]["rectools_negative_sampler"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="official RecTools source hashes"):
        load_verified_artifact(job, tmp_path)

    metadata["provenance"]["sources"]["rectools_negative_sampler"]["sha256"] = (
        RECTOOLS_SOURCE_SHA256["rectools_negative_sampler"]
    )
    metadata["provenance"]["sources"]["catalog_data"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="local official source hashes"):
        load_verified_artifact(job, tmp_path)


def test_artifact_verifier_rejects_an_unexpected_official_source(tmp_path):
    job = approved_manifest().jobs_for_stage("official")[0]
    directory = _write_artifact(tmp_path, job, recall=0.2, ndcg=0.1)
    metadata_path = directory / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["provenance"]["sources"]["unexpected_source"] = {
        "sha256": "0" * 64,
        "path": "/unexpected.py",
    }
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="official source set changed"):
        load_verified_artifact(job, tmp_path)


def test_artifact_verifier_pins_the_complete_official_environment(tmp_path):
    job = approved_manifest().jobs_for_stage("official")[0]
    directory = _write_artifact(tmp_path, job, recall=0.2, ndcg=0.1)
    metadata_path = directory / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())

    metadata["provenance"]["environment"]["python"] = "3.12.12"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="official Python version changed"):
        load_verified_artifact(job, tmp_path)

    metadata["provenance"]["environment"]["python"] = "3.12.13"
    metadata["provenance"]["environment"]["packages"]["torch"] = "2.7.0"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="official package versions changed"):
        load_verified_artifact(job, tmp_path)


def test_pinned_local_official_hashes_match_the_runner_sources() -> None:
    project_root = Path(__file__).resolve().parents[3]
    official = project_root / "experiments/g2_esasrec/official"
    paths = {
        "catalog_data": official / "catalog_data.py",
        "runner": official / "run_official.py",
        "protocol": official / "protocol.py",
        "provenance": official / "provenance.py",
    }

    observed = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }

    assert observed == OFFICIAL_LOCAL_SOURCE_SHA256


def test_unresolved_cap_prescribes_only_a_new_approved_extension(tmp_path):
    job = approved_manifest().jobs_for_stage("component_tuning")[1]
    directory = _write_artifact(tmp_path, job, recall=0.2, ndcg=0.1, resolved=False)
    metadata_path = directory / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(best_epoch=100, epochs_trained=100, max_epochs=100)
    metadata_path.write_text(json.dumps(metadata))

    expected_name = f"{job.run_name}_cap150"
    with pytest.raises(ValueError, match=expected_name) as failure:
        load_verified_artifact(job, tmp_path)
    assert "preserve raw artifact" in str(failure.value)
    assert "request approval" in str(failure.value)
    assert not (tmp_path / expected_name).exists()


def test_empirical_bands_require_exact_control_seeds_and_round_up(tmp_path):
    jobs = approved_manifest().jobs_for_stage("control_repeats")
    artifacts = []
    for index, job in enumerate(jobs):
        _write_artifact(
            tmp_path,
            job,
            recall=0.2 + index * 0.0001,
            ndcg=0.1 + index * 0.00001,
        )
        artifacts.append(load_verified_artifact(job, tmp_path))

    bands = empirical_bands(artifacts)
    expected = math.sqrt(sum((index - 4.5) ** 2 for index in range(10)) / 9) * 0.0001
    assert bands["recall@100"].sample_standard_deviation == pytest.approx(expected)
    assert bands["recall@100"].reader_threshold == 0.0004

    with pytest.raises(ValueError, match="seeds 42 through 51"):
        empirical_bands(artifacts[:-1])


def test_empirical_bands_reuse_the_exact_selected_seed_42_control(tmp_path):
    selected_job = approved_manifest().jobs_for_stage("control_tuning")[1]
    _write_artifact(
        tmp_path,
        selected_job,
        recall=0.2,
        ndcg=0.1,
        batch_size=128,
    )
    selected = load_verified_artifact(selected_job, tmp_path)
    repeats = []
    for index, job in enumerate(
        approved_manifest().jobs_for_stage("control_repeats")[1:], start=1
    ):
        _write_artifact(
            tmp_path,
            job,
            recall=0.2 + index * 0.0001,
            ndcg=0.1 + index * 0.00001,
            batch_size=128,
        )
        repeats.append(load_verified_artifact(job, tmp_path))

    rows = control_band_artifacts(selected, repeats)
    assert [row.job.seed for row in rows] == list(range(42, 52))
    assert empirical_bands(rows)["recall@100"].reader_threshold == 0.0004

    changed = replace(
        repeats[0],
        parameters={**repeats[0].parameters, "embedding_learning_rate": 0.002},
    )
    with pytest.raises(ValueError, match="exact selected control configuration"):
        empirical_bands([selected, changed, *repeats[1:]])


def test_selector_uses_recall_band_then_ndcg_then_cost(tmp_path):
    jobs = approved_manifest().jobs_for_stage("control_tuning")[:3]
    values = [(0.2000, 0.100, 12.0), (0.2002, 0.110, 14.0), (0.2001, 0.110, 10.0)]
    artifacts = []
    for job, (recall, ndcg, wall) in zip(jobs, values, strict=True):
        directory = _write_artifact(tmp_path, job, recall=recall, ndcg=ndcg)
        costs = json.loads((directory / "cost_metrics.json").read_text())
        costs["wall_seconds"] = wall
        (directory / "cost_metrics.json").write_text(json.dumps(costs))
        artifacts.append(load_verified_artifact(job, tmp_path))

    winner = select_best(
        artifacts,
        metric_bands={"recall@100": 0.0003, "ndcg@100": 0.001},
    )
    assert winner.job == jobs[2]


def test_selector_does_not_treat_lower_ndcg_as_a_cost_tie(tmp_path):
    jobs = approved_manifest().jobs_for_stage("control_tuning")[:2]
    first = _write_artifact(tmp_path, jobs[0], recall=0.2, ndcg=0.11)
    second = _write_artifact(tmp_path, jobs[1], recall=0.2002, ndcg=0.10)
    costs = json.loads((first / "cost_metrics.json").read_text())
    costs["wall_seconds"] = 100.0
    (first / "cost_metrics.json").write_text(json.dumps(costs))
    costs = json.loads((second / "cost_metrics.json").read_text())
    costs["wall_seconds"] = 1.0
    (second / "cost_metrics.json").write_text(json.dumps(costs))

    artifacts = [load_verified_artifact(job, tmp_path) for job in jobs]
    winner = select_best(artifacts, metric_bands={"recall@100": 0.0003})

    assert winner.job == jobs[0]


def test_control_selection_requires_persisted_max_ligr_fit_evidence(tmp_path):
    jobs = approved_manifest().jobs_for_stage("control_tuning")[:2]
    artifacts = []
    for job, batch, recall in zip(jobs, (512, 1024), (0.2, 0.3), strict=True):
        _write_artifact(tmp_path, job, recall=recall, ndcg=0.1, batch_size=batch)
        artifacts.append(load_verified_artifact(job, tmp_path))
    fit_path = tmp_path / "fit.json"
    probe_rows = []
    for batch_size in (128, 256, 512, 1024, 1280):
        fits = batch_size != 1024
        probe_path = tmp_path / f"probe-b{batch_size}.json"
        probe_path.write_text(
            json.dumps(
                {
                    "manifest_sha256": approved_manifest().sha256,
                    "dataset_size": "native-50m-diagnostic-2000-users-by-id",
                    "ligr_multiplier": 6,
                    "ffn_width": 1536,
                    "loss_kind": "gbce",
                    "gbce_t": 0.75,
                    "optimizer_steps": 1,
                    "device_name": FIT_DEVICE_NAME,
                    "device_compute_capability": list(FIT_DEVICE_COMPUTE_CAPABILITY),
                    "batch_size": batch_size,
                    "fits": fits,
                }
            )
        )
        probe_rows.append(
            {"batch_size": batch_size, "fits": fits, "artifact": probe_path.name}
        )
    fit_path.write_text(
        json.dumps(
            {
                "manifest_sha256": approved_manifest().sha256,
                "dataset_size": "native-50m-diagnostic-2000-users-by-id",
                "ligr_multiplier": 6,
                "ffn_width": 1536,
                "loss_kind": "gbce",
                "gbce_t": 0.75,
                "optimizer_steps": 1,
                "device_name": FIT_DEVICE_NAME,
                "device_compute_capability": list(FIT_DEVICE_COMPUTE_CAPABILITY),
                "probes": probe_rows,
            }
        )
    )

    winner = select_control_with_fit_gate(artifacts, load_fit_evidence(fit_path))

    assert winner.parameters["batch_size"] == 512

    document = json.loads(fit_path.read_text())
    document["probes"].pop()
    fit_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="every approved control batch"):
        load_fit_evidence(fit_path)


def test_one_step_fit_probe_runs_once_and_persists_max_ligr_contract(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: FIT_DEVICE_NAME)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: FIT_DEVICE_COMPUTE_CAPABILITY,
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []
    probes = []
    for batch_size in (128, 256, 512, 1024, 1280):
        probes.append(
            run_one_step_fit_probe(
                lambda batch_size=batch_size: calls.append(batch_size),
                batch_size=batch_size,
                destination=tmp_path / f"probe-{batch_size}.json",
                device=torch.device("cuda"),
            )
        )
    aggregate = tmp_path / "fit.json"
    write_fit_evidence(probes, aggregate)

    assert calls == [128, 256, 512, 1024, 1280]
    assert load_fit_evidence(aggregate).eligible_batches == {
        128,
        256,
        512,
        1024,
        1280,
    }


def test_fit_probe_rejects_non_a100_device_identity(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda _device: "NVIDIA H100 80GB HBM3"
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))

    with pytest.raises(RuntimeError, match="A100-SXM4-80GB"):
        fit_device_evidence(torch.device("cuda"))

    with pytest.raises(RuntimeError, match="A100-SXM4-80GB"):
        fit_device_evidence(torch.device("cpu"))


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("aggregate", "device_name", None),
        ("aggregate", "device_name", "NVIDIA H100 80GB HBM3"),
        ("artifact", "device_compute_capability", [9, 0]),
    ],
)
def test_fit_evidence_rejects_absent_wrong_or_stale_device_provenance(
    tmp_path, target, field, value
):
    probe_rows = []
    common = {
        "manifest_sha256": approved_manifest().sha256,
        "dataset_size": "native-50m-diagnostic-2000-users-by-id",
        "ligr_multiplier": 6,
        "ffn_width": 1536,
        "loss_kind": "gbce",
        "gbce_t": 0.75,
        "optimizer_steps": 1,
        "device_name": FIT_DEVICE_NAME,
        "device_compute_capability": list(FIT_DEVICE_COMPUTE_CAPABILITY),
    }
    for batch_size in (128, 256, 512, 1024, 1280):
        artifact = tmp_path / f"probe-{batch_size}.json"
        artifact.write_text(
            json.dumps(common | {"batch_size": batch_size, "fits": True})
        )
        probe_rows.append(
            {"batch_size": batch_size, "fits": True, "artifact": artifact.name}
        )
    aggregate = tmp_path / "fit.json"
    aggregate.write_text(json.dumps(common | {"probes": probe_rows}))
    path = aggregate if target == "aggregate" else tmp_path / "probe-128.json"
    document = json.loads(path.read_text())
    if value is None:
        document.pop(field)
    else:
        document[field] = value
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="device"):
        load_fit_evidence(aggregate)


def test_official_seed_aggregate_is_one_reader_row(tmp_path):
    jobs = approved_manifest().jobs_for_stage("official")
    artifacts = []
    for index, job in enumerate(jobs):
        directory = _write_artifact(tmp_path, job, recall=0.2 + index * 0.01, ndcg=0.1)
        metadata = json.loads((directory / "training_metadata.json").read_text())
        metadata["dataset_size"] = "native-50m"
        metadata["early_stopped"] = True
        metadata["best_epoch_at_cap"] = False
        metadata["wall_seconds"] = 12.0 + index
        (directory / "training_metadata.json").write_text(json.dumps(metadata))
        (directory / "cost_metrics.json").unlink()
        artifacts.append(load_verified_artifact(job, tmp_path))

    aggregate = aggregate_artifacts(artifacts, run_name="official 3-seed mean")
    compact = render_compact_report({"RQ1": [aggregate]}, reference=aggregate)

    assert aggregate.metrics["recall@100"] == pytest.approx(0.21)
    assert compact.count("official rectools") == 1


def test_reports_are_compact_and_tuning_ledger_marks_each_method_winner(tmp_path):
    manifest = approved_manifest()
    component_jobs = manifest.jobs_for_stage("component_tuning")
    jobs = [
        next(job for job in component_jobs if job.method == method and job.trial == 1)
        for method in COMPONENT_METHODS
    ]
    ligr_source = jobs[COMPONENT_METHODS.index("ligr_sampled_softmax")]
    _write_artifact(tmp_path, ligr_source, recall=0.24, ndcg=0.1)
    for index, job in enumerate(jobs):
        if job != ligr_source:
            _write_artifact(tmp_path, job, recall=0.2 + index * 0.01, ndcg=0.1)
    artifacts = [load_verified_artifact(job, tmp_path) for job in jobs]

    ledger = render_tuning_ledger(artifacts)
    compact = render_compact_report(
        {"RQ2": artifacts},
        reference=artifacts[0],
        metric_bands={
            "recall@100": 0.005,
            "ndcg@100": 0.005,
            "coverage@100": 0.005,
        },
    )

    assert "## RQ2" in ledger
    assert "**0.210**" in ledger
    assert "embedding LR" in ledger
    assert compact.startswith("# G2 eSASRec on native Yambda-50M\n\n## RQ2\n")
    assert "implementation details" not in compact.lower()
    assert '<span style="color: green">+5.000% (0.210)</span>' in compact
    assert "| fixed factor |" not in compact
    assert "| variant | layer | loss |" not in compact
    assert compact.count("| recall@100 | ndcg@100 | coverage@100 |") == 7
    rq2 = compact.split("## RQ2\n", 1)[1]
    tables = rq2.strip().split("\n\n")
    assert len(tables) == 7
    for table in tables:
        table_lines = table.splitlines()
        assert len(table_lines) == 4
        assert "%" not in table_lines[2]
        assert "%" in table_lines[3]
    for first_column in (
        "loss (standard block, FFN width 256)",
        "loss (parameter-matched SASRec, FFN width 1792)",
        "loss (LiGR, FFN width 1024)",
        "FFN width (standard block, sampled softmax)",
        "FFN width (standard block, gBCE)",
        "block (sampled softmax, parameter-matched)",
        "block (gBCE, parameter-matched)",
    ):
        assert f"| {first_column} | recall@100 |" in compact
    assert "| sampled softmax | 0.200 | 0.100 | 0.200 |" in compact
    assert '| **gBCE** | <span style="color: green">+5.000% (0.210)</span> |' in compact
    assert "| 256 | 0.200 | 0.100 | 0.200 |" in compact
    assert "| parameter-matched SASRec | 0.220 | 0.100 | 0.200 |" in compact
    assert "+5.000% (0.210)" in compact
    assert "params" in compact
    assert "peak GB" in compact

    mixed_job = manifest.jobs_for_stage("mixed_tuning")[0]
    mixed = replace(
        artifacts[0],
        job=mixed_job,
        metrics=artifacts[1].metrics,
        parameters={
            **artifacts[0].parameters,
            "ligr_multiplier": 4,
            "uniform_fraction": 0.6,
            "logq_correction": "none",
        },
    )
    mixed_report = render_compact_report({"RQ3": [mixed]}, reference=artifacts[0])
    assert "| LiGR | sampled softmax | 1024 | — | 0.6 | none |" in mixed_report
    assert "0.210" in mixed_report


def test_tuning_ledger_marks_the_band_aware_ndcg_winner_on_a_recall_tie(tmp_path):
    jobs = approved_manifest().jobs_for_stage("control_tuning")[:2]
    first = _write_artifact(tmp_path, jobs[0], recall=0.2, ndcg=0.11)
    second = _write_artifact(tmp_path, jobs[1], recall=0.2002, ndcg=0.10)
    first_costs = json.loads((first / "cost_metrics.json").read_text())
    first_costs["wall_seconds"] = 100.0
    (first / "cost_metrics.json").write_text(json.dumps(first_costs))
    second_costs = json.loads((second / "cost_metrics.json").read_text())
    second_costs["wall_seconds"] = 1.0
    (second / "cost_metrics.json").write_text(json.dumps(second_costs))
    artifacts = [load_verified_artifact(job, tmp_path) for job in jobs]

    ledger = render_tuning_ledger(
        artifacts,
        metric_bands={"recall@100": 0.0003},
    )
    lines = ledger.splitlines()
    first_line = next(line for line in lines if jobs[0].run_name in line)
    second_line = next(line for line in lines if jobs[1].run_name in line)

    assert "**0.200**" in first_line
    assert "**" not in second_line


def test_compact_report_uses_rq_specific_references_and_three_decimals(tmp_path):
    jobs = approved_manifest().jobs_for_stage("component_tuning")[:2]
    first = _write_artifact(tmp_path, jobs[0], recall=0.2, ndcg=0.1)
    second = _write_artifact(tmp_path, jobs[1], recall=0.21, ndcg=0.11)
    artifacts = [load_verified_artifact(job, tmp_path) for job in jobs]

    report = render_compact_report(
        {"RQ1": artifacts, "RQ3": artifacts},
        reference=artifacts[0],
        references={"RQ3": artifacts[1]},
    )

    rq3 = report.split("## RQ3", maxsplit=1)[1]
    assert "+0.000% (0.210)" in rq3
    assert "0.21000" not in report
    assert first.is_dir() and second.is_dir()


def _lineage_jobs():
    manifest = approved_manifest()
    control = manifest.jobs_for_stage("control_tuning")[1]
    ligr = next(
        job
        for job in manifest.jobs_for_stage("component_tuning")
        if job.method == "ligr_sampled_softmax" and job.trial == 1
    )
    mixed = next(
        job for job in manifest.jobs_for_stage("mixed_tuning") if job.trial == 2
    )
    return control, ligr, mixed


def _write_ligr_lineage(logs, *, multiplier=6, deep_learning_rate=0.02):
    control, ligr, _ = _lineage_jobs()
    _write_artifact(logs, control, recall=0.1, ndcg=0.1, batch_size=512)
    parameters = {
        "batch_size": 512,
        "embedding_learning_rate": 0.01,
        "deep_learning_rate": deep_learning_rate,
        "selected_control_job_id": control.id,
        "ligr_multiplier": multiplier,
    }
    _write_artifact(
        logs,
        ligr,
        recall=0.2,
        ndcg=0.1,
        parameters_override=parameters,
    )
    return control, ligr, parameters


def test_verified_capacity_component_rejects_a_multiplier_forged_from_its_ligr_source(
    tmp_path,
):
    control, ligr, _ = _write_ligr_lineage(tmp_path, multiplier=6)
    job = next(
        job
        for job in approved_manifest().jobs_for_stage("component_tuning")
        if job.method == "ligr_gbce" and job.trial == 1
    )
    parameters = {
        "batch_size": 512,
        "embedding_learning_rate": 0.03,
        "deep_learning_rate": 0.04,
        "selected_control_job_id": control.id,
        "source_job_id": ligr.id,
        "ligr_multiplier": 4,
        "gbce_t": 0.6,
    }
    _write_artifact(
        tmp_path,
        job,
        recall=0.2,
        ndcg=0.1,
        parameters_override=parameters,
    )

    with pytest.raises(ValueError, match="ligr_multiplier"):
        load_verified_artifact(job, tmp_path)


def test_verified_control_repeat_rejects_forged_selected_rates(tmp_path):
    control = approved_manifest().jobs_for_stage("control_tuning")[1]
    _write_artifact(tmp_path, control, recall=0.1, ndcg=0.1, batch_size=512)
    repeat = next(
        job
        for job in approved_manifest().jobs_for_stage("control_repeats")
        if job.seed == 43
    )
    parameters = {
        "batch_size": 512,
        "embedding_learning_rate": 0.002,
        "deep_learning_rate": 0.001,
        "selected_control": True,
        "selected_control_job_id": control.id,
    }
    _write_artifact(
        tmp_path,
        repeat,
        recall=0.1,
        ndcg=0.1,
        parameters_override=parameters,
    )

    with pytest.raises(ValueError, match="embedding_learning_rate"):
        load_verified_artifact(repeat, tmp_path)


@pytest.mark.parametrize(
    ("name", "forged"),
    (
        ("batch_size", 256),
        ("embedding_learning_rate", 0.03),
        ("deep_learning_rate", 0.04),
        ("ligr_multiplier", 4),
    ),
)
def test_verified_mixed_job_rejects_forged_ligr_lineage(tmp_path, name, forged):
    control, ligr, source = _write_ligr_lineage(tmp_path)
    _, _, mixed = _lineage_jobs()
    parameters = source | {
        "selected_control_job_id": control.id,
        "source_job_id": ligr.id,
        "uniform_fraction": 0.5,
        "logq_correction": "yi2019",
        name: forged,
    }
    _write_artifact(
        tmp_path,
        mixed,
        recall=0.2,
        ndcg=0.1,
        parameters_override=parameters,
    )

    with pytest.raises(ValueError, match="not propagated"):
        load_verified_artifact(mixed, tmp_path)


@pytest.mark.parametrize(
    ("name", "forged"),
    (("embedding_learning_rate", 0.03), ("deep_learning_rate", 0.3)),
)
def test_verified_lr_boundary_rejects_forged_source_lineage(tmp_path, name, forged):
    control, ligr, source = _write_ligr_lineage(tmp_path, deep_learning_rate=0.128)
    boundary = next(
        job
        for job in approved_manifest().jobs_for_stage("lr_boundary")
        if job.method == "ligr_sampled_softmax" and job.trial == 0
    )
    parameters = source | {
        "builder": "component",
        "method": "ligr_sampled_softmax",
        "source_job_id": ligr.id,
        "deep_learning_rate": 0.384,
        name: forged,
    }
    _write_artifact(
        tmp_path,
        boundary,
        recall=0.2,
        ndcg=0.1,
        parameters_override=parameters,
    )

    with pytest.raises(ValueError, match="prerequisite .* changed|not propagated"):
        load_verified_artifact(boundary, tmp_path)


def test_verified_confirmation_requires_the_exact_selected_sampler_lineage(tmp_path):
    control, ligr, source = _write_ligr_lineage(tmp_path)
    _, _, mixed = _lineage_jobs()
    mixed_parameters = source | {
        "selected_control_job_id": control.id,
        "source_job_id": ligr.id,
        "uniform_fraction": 0.5,
        "logq_correction": "yi2019",
    }
    _write_artifact(
        tmp_path,
        mixed,
        recall=0.2,
        ndcg=0.1,
        parameters_override=mixed_parameters,
    )
    valid_confirmation = next(
        job
        for job in approved_manifest().jobs_for_stage("reversal_confirmation")
        if job.trial == 0 and job.seed == 43
    )
    valid_parameters = mixed_parameters | {
        "builder": "mixed_sampler",
        "source_job_id": mixed.id,
    }
    _write_artifact(
        tmp_path,
        valid_confirmation,
        recall=0.2,
        ndcg=0.1,
        parameters_override=valid_parameters,
    )
    assert load_verified_artifact(valid_confirmation, tmp_path).parameters == (
        valid_parameters
    )
    forged_confirmation = next(
        job
        for job in approved_manifest().jobs_for_stage("reversal_confirmation")
        if job.trial == 0 and job.seed == 44
    )
    forged = valid_parameters | {
        "uniform_fraction": 0.7,
    }
    _write_artifact(
        tmp_path,
        forged_confirmation,
        recall=0.2,
        ndcg=0.1,
        parameters_override=forged,
    )

    with pytest.raises(ValueError, match="uniform_fraction"):
        load_verified_artifact(forged_confirmation, tmp_path)
