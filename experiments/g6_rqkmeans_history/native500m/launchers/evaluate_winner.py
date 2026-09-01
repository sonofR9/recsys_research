from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from dcn.eval.ranking_evidence import load_ranking_evidence, write_ranking_evidence
from dcn.eval.true_metric import evaluate_true_ndcg, prepare_ranking
from dcn.semantic.artifacts import load_item_embeddings
from experiments.g6_rqkmeans_history.native500m.analysis.collect import (
    _ranking_context_sha256,
    recompute_metrics,
    recompute_sid_diagnostics,
    recompute_slice_diagnostics,
    reuse_source_is_eligible,
)
from experiments.g6_rqkmeans_history.native500m.analysis.topk_evidence import (
    TopKContext,
    tensor_sha256,
    write_topk_evidence,
)
from experiments.g6_rqkmeans_history.native500m.launchers.materialize import (
    load_selection_binding,
)
from experiments.g6_rqkmeans_history.native500m.launchers.queue import (
    QueueJob,
    canonical_bytes,
    load_queue_manifest,
    persist_immutable_bytes,
)
from experiments.g6_rqkmeans_history.native500m.launchers.runtime import (
    build_experiment,
)


@dataclass(frozen=True)
class _EvaluationManifest:
    stage: str
    logical_sha256: str


def evaluate_selected_winner(
    *,
    manifest_path: Path,
    selection_path: Path,
    job_id: str,
    ranking_context_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
    control_ranking_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_queue_manifest(manifest_path)
    _, selection = load_selection_binding(
        selection_path,
        source_manifest_path=manifest_path,
        logs_root=logs_root,
        queue_state_directory=queue_state_directory,
    )
    if (
        selection["stage"] != manifest.stage
        or selection["manifest_logical_sha256"] != manifest.logical_sha256
        or job_id not in selection["selected_job_ids"].values()
    ):
        raise ValueError("winner is not selected by this manifest evidence")
    matches = [job for job in manifest.jobs if job.job_id == job_id]
    if len(matches) != 1:
        raise ValueError("selected winner is absent from the stage manifest")
    job, evaluation_manifest, evaluation_selection = _evaluation_source(
        matches[0], selection
    )
    experiment = build_experiment(job)
    run_directory = Path(experiment.base_path) / "logs" / job.run_name
    if logs_root.resolve() != (Path(experiment.base_path) / "logs").resolve():
        raise ValueError("winner evaluator logs root differs from the experiment")
    metadata = _read_json(run_directory / "training_metadata.json")
    checkpoint_path, checkpoint_sha256 = _checkpoint(metadata, experiment.base_path)
    existing_final = run_directory / "final_evaluation.json"
    if existing_final.is_file():
        return _existing_final_evaluation(
            existing_final,
            run_directory=run_directory,
            manifest=evaluation_manifest,
            job=job,
            selection=evaluation_selection,
            checkpoint_sha256=checkpoint_sha256,
            ranking_context_path=ranking_context_path,
        )
    experiment.setup()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    experiment.base_model.load_state_dict(state, strict=True)
    experiment.base_model.to(experiment.device)
    callback = experiment.true_metric
    callback.train_item_frequencies = experiment.training_item_frequencies
    callback.prepare()
    with (
        torch.inference_mode(),
        torch.autocast(
            experiment.device.type,
            dtype=experiment.runtime.dtype,
            enabled=experiment.runtime.dtype != torch.float32,
        ),
    ):
        item_ids, item_vectors = callback._encode_catalog()
        query_user_ids, query_vectors = callback._encode_queries()
    prepared = prepare_ranking(
        query_user_ids,
        item_ids,
        callback.relevance,
        callback.train_seen,
        device=experiment.device,
        user_chunk=callback.user_chunk,
        max_users=None,
        seed=callback.seed,
        exclude_seen=False,
    )
    semantic_codes = _catalog_semantic_codes(experiment, item_ids)
    scored, ranks = evaluate_true_ndcg(
        query_vectors.float(),
        query_user_ids,
        item_vectors.float(),
        item_ids,
        callback.relevance,
        callback.train_seen,
        (10, 50, 100),
        device=experiment.device,
        user_chunk=callback.user_chunk,
        prepared=prepared,
        exclude_seen=False,
        item_semantic_codes=semantic_codes,
        return_relevant_ranks=True,
    )
    recommendations = _top100(
        query_vectors.float(), item_vectors.float(), item_ids, prepared
    )
    user_ids = torch.tensor([user.user_id for user in prepared.evaluable])
    ranking = callback._ranking_evidence(prepared, ranks)
    ranking_path = run_directory / "ranking_evidence.pt"
    ranking_context_sha256 = write_ranking_evidence(
        ranking,
        context_path=ranking_context_path,
        ranking_path=ranking_path,
    )
    evaluator = {
        "dataset": "yambda-500m",
        "split": "final-seven-days",
        "cutoffs": [10, 50, 100],
        "full_catalog": True,
        "exclude_seen": False,
        "all_eligible_users": True,
    }
    evaluator_sha256 = hashlib.sha256(canonical_bytes(evaluator)).hexdigest()
    catalog = item_ids.detach().cpu().to(torch.int64).contiguous()
    codes = (
        torch.empty((catalog.shape[0], 0), dtype=torch.int64)
        if semantic_codes is None
        else semantic_codes.detach().cpu().to(torch.int64).contiguous()
    )
    context = TopKContext(
        dataset="yambda-500m",
        split="final-seven-days",
        ranking_context_sha256=ranking_context_sha256,
        ordered_catalog_sha256=tensor_sha256(catalog),
        checkpoint_sha256=checkpoint_sha256,
        evaluator_configuration_sha256=evaluator_sha256,
        stage=evaluation_manifest.stage,
        job_id=job.job_id,
        job_logical_sha256=job.logical_sha256,
        manifest_logical_sha256=evaluation_manifest.logical_sha256,
        semantic_codes_sha256=tensor_sha256(codes),
    )
    topk = write_topk_evidence(
        run_directory / "top100_item_evidence.pt",
        context=context,
        user_ids=user_ids,
        recommended_item_ids=recommendations,
        ordered_catalog_item_ids=catalog,
    )
    metrics = recompute_metrics(
        context_path=ranking_context_path,
        ranking_path=ranking_path,
        topk=topk,
        semantic_codes=None if semantic_codes is None else codes,
    )
    _verify_rank_metric_agreement(scored, metrics)
    _write_json(run_directory / "final_metrics.json", metrics)
    artifacts: dict[str, object] = {}
    if semantic_codes is not None:
        item_embedding_ids, content = load_item_embeddings(
            Path(experiment.artifacts.precomputed_embeddings[experiment.item_id_column])
        )
        if not torch.equal(item_embedding_ids, experiment.semantic_codes.item_ids):
            raise ValueError("semantic diagnostics content order differs")
        diagnostics = recompute_sid_diagnostics(
            semantic_codes=experiment.semantic_codes.codes[
                :, : experiment.semantic.num_levels
            ],
            normalized_content_vectors=content,
            codebook_centroids=experiment.semantic_codebooks.centroids,
            codes_per_level=tuple(
                experiment.semantic_codes.codes_per_level[
                    : experiment.semantic.num_levels
                ]
            ),
        )
        _write_json(run_directory / "semantic_id_diagnostics.json", diagnostics)
        artifacts["semantic_id_diagnostics"] = diagnostics
        if control_ranking_path is not None:
            control = load_ranking_evidence(ranking_context_path, control_ranking_path)
            slices = recompute_slice_diagnostics(
                control=control,
                treatment=ranking,
                ordered_catalog_item_ids=catalog,
                semantic_codes=codes,
            )
            _write_json(run_directory / "slice_diagnostics.json", slices)
            artifacts["slice_diagnostics"] = slices
    final = {
        "schema": "g6-native500m-final-evaluation/v1",
        "manifest_logical_sha256": evaluation_manifest.logical_sha256,
        "job_logical_sha256": job.logical_sha256,
        "selection_sha256": evaluation_selection["selection_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "ranking_context_sha256": ranking_context_sha256,
        "ordered_catalog_sha256": tensor_sha256(catalog),
        "catalog_size": catalog.shape[0],
        "evaluator_configuration": evaluator,
        "evaluator_configuration_sha256": evaluator_sha256,
        "topk_logical_sha256": topk.logical_sha256,
        "topk_physical_sha256": topk.physical_sha256,
        "artifacts": artifacts,
    }
    _write_json(run_directory / "final_evaluation.json", final)
    return final


def _evaluation_source(
    job: QueueJob, selection: dict[str, Any]
) -> tuple[QueueJob, _EvaluationManifest, dict[str, Any]]:
    declarations = job.payload["exact_reuse"]
    if not declarations:
        return (
            job,
            _EvaluationManifest(
                selection["stage"], selection["manifest_logical_sha256"]
            ),
            selection,
        )
    if not isinstance(declarations, list) or len(declarations) != 1:
        raise ValueError("selected exact reuse must name one source")
    declaration = declarations[0]
    rows = [row for row in selection["candidates"] if row.get("job_id") == job.job_id]
    if len(rows) != 1:
        raise ValueError("selected exact-reuse candidate is absent")
    row = rows[0]
    reused = row.get("reused_from")
    artifacts = row.get("artifacts")
    contract_identity = (
        artifacts.get("g6_native500m_job.json") if isinstance(artifacts, dict) else None
    )
    if (
        not isinstance(reused, dict)
        or reused.get("job_id") != declaration.get("source_job_id")
        or reused.get("job_logical_sha256") != declaration.get("source_contract_sha256")
        or not isinstance(contract_identity, dict)
    ):
        raise ValueError("selected exact-reuse source identity differs")
    contract_path = Path(contract_identity["path"])
    if contract_path.is_symlink() or not contract_path.is_file():
        raise ValueError("selected exact-reuse source contract differs")
    content = contract_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != contract_identity.get("sha256"):
        raise ValueError("selected exact-reuse source contract differs")
    contract = json.loads(content)
    source_selection_path = Path(declaration["source_selection_path"])
    source_selection_content = source_selection_path.read_bytes()
    if hashlib.sha256(source_selection_content).hexdigest() != declaration.get(
        "source_selection_physical_sha256"
    ):
        raise ValueError("selected exact-reuse source selection differs")
    source_selection = json.loads(source_selection_content)
    source_selection_body = {
        name: value
        for name, value in source_selection.items()
        if name != "selection_sha256"
    }
    if hashlib.sha256(canonical_bytes(source_selection_body)).hexdigest() != (
        source_selection.get("selection_sha256")
    ):
        raise ValueError("selected exact-reuse source selection differs")
    source_job = contract.get("job")
    if (
        not isinstance(source_job, dict)
        or hashlib.sha256(canonical_bytes(source_job)).hexdigest()
        != contract.get("job_logical_sha256")
        or contract.get("job_logical_sha256")
        != declaration.get("source_contract_sha256")
        or contract.get("manifest_logical_sha256")
        != source_selection.get("manifest_logical_sha256")
        or contract.get("manifest_physical_sha256")
        != source_selection.get("manifest_physical_sha256")
        or source_selection.get("stage") != declaration.get("source_selection_stage")
        or source_selection.get("selection_sha256")
        != declaration.get("source_selection_sha256")
        or not reuse_source_is_eligible(
            job,
            declaration.get("source_job_id"),
            source_selection.get("selected_job_ids"),
            source_selection_stage=source_selection.get("stage"),
        )
    ):
        raise ValueError("selected exact-reuse source contract differs")
    parameters = source_job.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("selected exact-reuse source parameters differ")
    resolved = QueueJob(
        job_id=source_job["job_id"],
        run_name=parameters["run_name"],
        runner=parameters["runner"],
        config_logical_sha256=parameters["config_logical_sha256"],
        data_group=parameters["data_group"],
        logical_sha256=contract["job_logical_sha256"],
        payload=source_job,
        environment=dict(parameters.get("environment", {})),
    )
    return (
        resolved,
        _EvaluationManifest(
            source_selection["stage"], source_selection["manifest_logical_sha256"]
        ),
        source_selection,
    )


def _verify_rank_metric_agreement(
    scored: dict[str, float], recomputed: dict[str, float]
) -> None:
    rank_metric_prefixes = ("recall@", "capped_recall@", "ndcg@", "mrr@")
    names = [
        name
        for name in scored
        if name == "num_users" or name.startswith(rank_metric_prefixes)
    ]
    for name in names:
        actual = recomputed.get(name)
        expected = scored[name]
        if (
            actual is None
            or not math.isfinite(actual)
            or not math.isfinite(expected)
            or abs(actual - expected) > 1e-10
        ):
            raise ValueError(
                f"winner evaluator disagrees on {name}: "
                f"recomputed={actual!r}, scored={expected!r}"
            )


def _existing_final_evaluation(
    path: Path,
    *,
    run_directory: Path,
    manifest: _EvaluationManifest,
    job: QueueJob,
    selection: dict[str, Any],
    checkpoint_sha256: str,
    ranking_context_path: Path,
) -> dict[str, Any]:
    final = _read_json(path)
    topk = run_directory / "top100_item_evidence.pt"
    if (
        final.get("schema") != "g6-native500m-final-evaluation/v1"
        or final.get("manifest_logical_sha256") != manifest.logical_sha256
        or final.get("job_logical_sha256") != job.logical_sha256
        or final.get("selection_sha256") != selection.get("selection_sha256")
        or final.get("checkpoint_sha256") != checkpoint_sha256
        or final.get("ranking_context_sha256")
        != _ranking_context_sha256(ranking_context_path)
        or not topk.is_file()
        or hashlib.sha256(topk.read_bytes()).hexdigest()
        != final.get("topk_physical_sha256")
    ):
        raise ValueError("existing winner evaluation identity differs")
    return final


def _top100(
    query_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    item_ids: torch.Tensor,
    prepared: object,
) -> torch.Tensor:
    rows = []
    for query_rows in prepared.query_rows_by_chunk:
        indices = torch.topk(
            query_vectors[query_rows] @ item_vectors.T, k=100, dim=1
        ).indices
        rows.append(item_ids[indices].detach().cpu())
    return torch.cat(rows).to(torch.int64)


def _catalog_semantic_codes(
    experiment: object, item_ids: torch.Tensor
) -> torch.Tensor | None:
    if not hasattr(experiment, "semantic_codes"):
        return None
    semantic = experiment.semantic_codes
    positions = {
        int(item_id): index for index, item_id in enumerate(semantic.item_ids.tolist())
    }
    try:
        rows = torch.tensor([positions[int(item_id)] for item_id in item_ids.tolist()])
    except KeyError as error:
        raise ValueError(
            "ranked catalog contains an item without semantic codes"
        ) from error
    return semantic.codes[rows, : experiment.semantic.num_levels].to(experiment.device)


def _checkpoint(metadata: dict[str, Any], base_path: str | Path) -> tuple[Path, str]:
    artifact = metadata.get("best_model_artifact")
    if not isinstance(artifact, dict) or set(artifact) != {"schema", "path", "sha256"}:
        raise ValueError("best-model artifact binding is missing")
    relative = artifact["path"]
    sha256 = artifact["sha256"]
    if not isinstance(relative, str) or not isinstance(sha256, str):
        raise ValueError("best-model artifact binding differs")
    path = Path(base_path) / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if artifact["schema"] != "g6-best-model-state/v1" or digest != sha256:
        raise ValueError("best-model artifact binding differs")
    return path, digest


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    persist_immutable_bytes(path, canonical_bytes(value), label="winner evidence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("job_id")
    parser.add_argument("ranking_context", type=Path)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--queue-state", type=Path, required=True)
    parser.add_argument("--control-ranking", type=Path)
    arguments = parser.parse_args()
    result = evaluate_selected_winner(
        manifest_path=arguments.manifest,
        selection_path=arguments.selection,
        job_id=arguments.job_id,
        ranking_context_path=arguments.ranking_context,
        logs_root=arguments.logs_root,
        queue_state_directory=arguments.queue_state,
        control_ranking_path=arguments.control_ranking,
    )
    print(result["topk_physical_sha256"])


if __name__ == "__main__":
    main()
