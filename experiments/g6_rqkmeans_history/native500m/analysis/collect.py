from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch

from dcn.eval.ranking_evidence import RankingEvidence, load_ranking_evidence
from dcn.semantic import ResidualCodebooks, SemanticCodes, semantic_id_diagnostics
from experiments.g6_rqkmeans_history.analysis.rq0_slices import (
    bucket_size_comparison,
    slice_comparison,
)
from experiments.g6_rqkmeans_history.native500m.analysis.topk_evidence import (
    TopKContext,
    TopKEvidence,
    load_topk_evidence,
    tensor_sha256,
)
from experiments.g6_rqkmeans_history.native500m.launchers.queue import (
    build_batch_specification,
    canonical_bytes,
    persist_immutable_bytes,
    QueueJob,
    QueueManifest,
)
from experiments.g6_rqkmeans_history.native500m.protocol.selection import (
    Candidate,
    MetricValues,
    select_by_quality,
)
from experiments.g6_rqkmeans_history.native500m.protocol.contracts import (
    job_id_has_coordinate,
)


_CUTOFFS = (10, 50, 100)
NATIVE500M_RELATIVE_DISPERSIONS = {
    "recall@10": 0.03152,
    "recall@50": 0.02116,
    "recall@100": 0.01685,
    "ndcg@10": 0.02680,
    "ndcg@50": 0.02272,
    "ndcg@100": 0.01966,
    "mrr@10": 0.02393,
    "mrr@50": 0.02157,
    "mrr@100": 0.02085,
    "capped_recall@10": 0.02955,
    "capped_recall@50": 0.02107,
    "capped_recall@100": 0.01683,
    "coverage@10": 0.16765,
    "coverage@50": 0.15102,
    "coverage@100": 0.13429,
}
RECALL_RELATIVE_DISPERSION = NATIVE500M_RELATIVE_DISPERSIONS["recall@100"]


def recompute_sid_diagnostics(
    *,
    semantic_codes: torch.Tensor | None,
    normalized_content_vectors: torch.Tensor,
    codebook_centroids: torch.Tensor,
    codes_per_level: tuple[int, ...],
) -> dict[str, Any]:
    codes = semantic_codes.detach().to(device="cpu", dtype=torch.int64).contiguous()
    embeddings = normalized_content_vectors.detach().to(device="cpu").contiguous()
    centroids = codebook_centroids.detach().to(device="cpu").contiguous()
    if codes.ndim != 2 or len(codes_per_level) != codes.shape[1]:
        raise ValueError("semantic code widths do not match the base levels")
    if centroids.ndim != 3 or centroids.shape[0] != codes.shape[1]:
        raise ValueError("codebooks do not match the semantic code levels")
    if len(set(codes_per_level)) != 1 or centroids.shape[1] != codes_per_level[0]:
        raise ValueError("native-500M diagnostics require one shared codebook size")
    semantic = SemanticCodes(
        item_ids=torch.arange(codes.shape[0]),
        codes=codes,
        codes_per_level=codes_per_level,
    )
    diagnostics = semantic_id_diagnostics(
        semantic,
        embeddings,
        num_base_levels=codes.shape[1],
        codebooks=ResidualCodebooks(centroids),
    )
    return asdict(diagnostics)


def recompute_slice_diagnostics(
    *,
    control: RankingEvidence,
    treatment: RankingEvidence,
    ordered_catalog_item_ids: torch.Tensor,
    semantic_codes: torch.Tensor,
) -> dict[str, dict[str, object]]:
    catalog = ordered_catalog_item_ids.detach().to(torch.int64).cpu().contiguous()
    codes = semantic_codes.detach().to(torch.int64).cpu().contiguous()
    if catalog.ndim != 1 or codes.ndim != 2 or codes.shape[0] != catalog.shape[0]:
        raise ValueError("slice semantic codes do not align with the ordered catalog")
    widths = tuple(int(codes[:, level].max()) + 1 for level in range(codes.shape[1]))
    semantic = SemanticCodes(catalog, codes, widths)
    eligible = slice_comparison(
        control,
        treatment,
        semantic_codes=semantic,
        semantic_base_levels=codes.shape[1],
        control_run_name="authenticated-control",
        semantic_run_name="authenticated-treatment",
    )["slices"]
    buckets = bucket_size_comparison(
        control,
        treatment,
        semantic_codes=semantic,
        semantic_base_levels=codes.shape[1],
    )["slices"]
    return {
        **eligible,
        **{f"target_{name}": value for name, value in buckets.items()},
    }


def recompute_metrics(
    *,
    context_path: Path,
    ranking_path: Path,
    topk: TopKEvidence,
    semantic_codes: torch.Tensor | None = None,
) -> dict[str, float]:
    ranking = load_ranking_evidence(context_path, ranking_path)
    if topk.context.ranking_context_sha256 != _ranking_context_sha256(context_path):
        raise ValueError("top-K evidence references a different compact context")
    if ranking.max_k != 100 or not torch.equal(ranking.user_ids, topk.user_ids):
        raise ValueError("top-K and compact ranking populations differ")
    catalog = topk.ordered_catalog_item_ids.tolist()
    positions = {item_id: index for index, item_id in enumerate(catalog)}
    if len(positions) != len(catalog):
        raise ValueError("ordered catalog contains duplicate item IDs")
    relevant_by_user = _ragged(
        ranking.relevant_item_ids.tolist(), ranking.relevance_offsets.tolist()
    )
    recomputed_ranks = []
    for relevant, recommended in zip(
        relevant_by_user, topk.recommended_item_ids.tolist(), strict=True
    ):
        rank_by_item = {item_id: rank for rank, item_id in enumerate(recommended, 1)}
        recomputed_ranks.extend(rank_by_item.get(item_id, 0) for item_id in relevant)
    if recomputed_ranks != ranking.relevant_ranks.tolist():
        raise ValueError("top-K recommendations disagree with compact ranks")
    metrics = _ranking_metrics(
        relevant_by_user,
        recomputed_ranks,
        topk.recommended_item_ids,
        len(catalog),
    )
    if semantic_codes is not None:
        codes = semantic_codes.detach().to(device="cpu", dtype=torch.int64).contiguous()
        if codes.ndim != 2 or codes.shape[0] != len(catalog) or codes.shape[1] < 1:
            raise ValueError("semantic codes do not align with the ordered catalog")
        if tensor_sha256(codes) != topk.context.semantic_codes_sha256:
            raise ValueError("semantic codes SHA-256 differs from top-K context")
        metrics.update(
            _semantic_metrics(
                relevant_by_user,
                topk.recommended_item_ids,
                positions,
                codes,
            )
        )
    metrics["num_users"] = float(len(relevant_by_user))
    return metrics


def metrics_agree(
    saved: Mapping[str, object],
    recomputed: Mapping[str, float],
    *,
    tolerance: float,
) -> None:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("metric tolerance must be finite and non-negative")
    missing = set(recomputed) - set(saved)
    if missing:
        raise ValueError(f"saved metrics are missing {sorted(missing)}")
    for name, expected in recomputed.items():
        actual = saved[name]
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isfinite(actual)
            or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance)
        ):
            raise ValueError(
                f"saved {name}={actual!r} differs from recomputed {expected!r}"
            )


def collect_final_run(
    *,
    manifest: QueueManifest,
    job_id: str,
    logs_root: Path,
    queue_state_directory: Path,
    batch_id: str,
    selection_sha256: str,
    ranking_context_path: Path,
    semantic_codes: torch.Tensor,
    metric_tolerance: float,
    output_path: Path,
) -> dict[str, Any]:
    job = _manifest_job(manifest, job_id)
    directory = logs_root / job.run_name
    contract_path = directory / "g6_native500m_job.json"
    metadata_path = directory / "training_metadata.json"
    metrics_path = directory / "final_metrics.json"
    ranking_path = directory / "ranking_evidence.pt"
    topk_path = directory / "top100_item_evidence.pt"
    final_evaluation_path = directory / "final_evaluation.json"
    required = (
        contract_path,
        metadata_path,
        metrics_path,
        ranking_path,
        topk_path,
        final_evaluation_path,
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"{job.run_name}: missing final artifacts {missing}")
    contract = _read_json(contract_path)
    expected_contract = _run_contract(manifest, job)
    if contract != expected_contract:
        raise ValueError(f"{job.run_name}: run contract differs")
    metadata = _read_json(metadata_path)
    _validate_training_metadata(metadata, job, require_final_evidence=False)
    _verify_queue_success(
        queue_state_directory,
        batch_ids=_normalize_batch_ids(batch_id),
        manifest=manifest,
        job=job,
    )
    checkpoint_sha256, _ = _best_model_artifact(metadata, directory)
    final_evaluation = _read_json(final_evaluation_path)
    if (
        final_evaluation.get("schema") != "g6-native500m-final-evaluation/v1"
        or final_evaluation.get("manifest_logical_sha256") != manifest.logical_sha256
        or final_evaluation.get("job_logical_sha256") != job.logical_sha256
        or final_evaluation.get("checkpoint_sha256") != checkpoint_sha256
        or final_evaluation.get("selection_sha256") != selection_sha256
        or final_evaluation.get("ranking_context_sha256")
        != _ranking_context_sha256(ranking_context_path)
    ):
        raise ValueError(f"{job.run_name}: final evaluation identity differs")
    catalog_size = final_evaluation.get("catalog_size")
    if (
        not isinstance(catalog_size, int)
        or isinstance(catalog_size, bool)
        or catalog_size < 100
    ):
        raise ValueError(f"{job.run_name}: final catalog size is invalid")
    codes = (
        torch.empty((catalog_size, 0), dtype=torch.int64)
        if semantic_codes is None
        else semantic_codes
    )
    expected_context = TopKContext(
        dataset="yambda-500m",
        split="final-seven-days",
        ranking_context_sha256=_ranking_context_sha256(ranking_context_path),
        ordered_catalog_sha256=final_evaluation["ordered_catalog_sha256"],
        checkpoint_sha256=checkpoint_sha256,
        evaluator_configuration_sha256=final_evaluation[
            "evaluator_configuration_sha256"
        ],
        stage=manifest.stage,
        job_id=job.job_id,
        job_logical_sha256=job.logical_sha256,
        manifest_logical_sha256=manifest.logical_sha256,
        semantic_codes_sha256=tensor_sha256(codes),
    )
    topk = load_topk_evidence(
        topk_path,
        expected_context=expected_context,
        expected_physical_sha256=final_evaluation["topk_physical_sha256"],
    )
    if topk.logical_sha256 != final_evaluation["topk_logical_sha256"]:
        raise ValueError(f"{job.run_name}: top-K logical identity differs")
    recomputed = recompute_metrics(
        context_path=ranking_context_path,
        ranking_path=ranking_path,
        topk=topk,
        semantic_codes=semantic_codes,
    )
    saved = _read_json(metrics_path)
    metrics_agree(saved, recomputed, tolerance=metric_tolerance)
    body = {
        "schema": "g6-native500m-collected-run/v1",
        "stage": manifest.stage,
        "job_id": job.job_id,
        "run_name": job.run_name,
        "manifest_logical_sha256": manifest.logical_sha256,
        "manifest_physical_sha256": manifest.physical_sha256,
        "job_logical_sha256": job.logical_sha256,
        "batch_id": batch_id,
        "best_epoch": metadata["best_epoch"],
        "trained_epochs": metadata["epochs_trained"],
        "metrics": recomputed,
        "artifacts": {
            path.name: {"path": str(path.resolve()), "sha256": _file_sha256(path)}
            for path in required
        },
    }
    document = {
        **body,
        "evidence_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest(),
    }
    _write_immutable_json(output_path, document)
    return document


def collect_stage_candidates(
    *,
    manifest: QueueManifest,
    logs_root: Path,
    queue_state_directory: Path,
    batch_id: str | tuple[str, ...] | list[str],
    output_path: Path | None,
    recall_relative_dispersion: float = RECALL_RELATIVE_DISPERSION,
) -> dict[str, Any]:
    if recall_relative_dispersion != RECALL_RELATIVE_DISPERSION:
        raise ValueError("native-500M Recall@100 dispersion must be exactly 0.01685")
    batch_ids = _normalize_batch_ids(batch_id)
    rows = []
    selectable = []
    for order, job in enumerate(manifest.jobs):
        if job.payload["exact_reuse"]:
            row = _replayed_candidate(job, order)
            rows.append(row)
            metrics = row["validation_metrics"]
            selectable.append(
                Candidate(
                    job.job_id,
                    MetricValues(metrics["recall@100"], metrics["ndcg@100"]),
                    order,
                )
            )
            continue
        directory = logs_root / job.run_name
        contract_path = directory / "g6_native500m_job.json"
        metadata_path = directory / "training_metadata.json"
        history_path = directory / "validation_history.json"
        for path in (contract_path, metadata_path, history_path):
            if not path.is_file():
                raise ValueError(
                    f"{job.run_name}: missing candidate artifact {path.name}"
                )
        if _read_json(contract_path) != _run_contract(manifest, job):
            raise ValueError(f"{job.run_name}: run contract differs")
        metadata = _read_json(metadata_path)
        _validate_training_metadata(metadata, job, require_final_evidence=False)
        checkpoint_sha256, checkpoint_path = _best_model_artifact(metadata, directory)
        history = _read_json(history_path)
        metrics = _validation_selection_metrics(history, metadata, job)
        _verify_queue_success(
            queue_state_directory,
            batch_ids=batch_ids,
            manifest=manifest,
            job=job,
        )
        row = {
            "job_id": job.job_id,
            "job_logical_sha256": job.logical_sha256,
            "run_name": job.run_name,
            "manifest_order": order,
            "best_epoch": metadata["best_epoch"],
            "training_horizon": 26,
            "restored_checkpoint_sha256": checkpoint_sha256,
            "parameters": job.payload["parameters"],
            "validation_metrics": metrics,
            "convergence": _validation_convergence(history, metrics["recall@100"]),
            "artifacts": {
                path.name: {
                    "path": str(path.resolve()),
                    "sha256": _file_sha256(path),
                }
                for path in (contract_path, metadata_path, history_path)
            }
            | {
                checkpoint_path.name: {
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_sha256,
                }
            },
        }
        rows.append(row)
        selectable.append(
            Candidate(
                job.job_id,
                MetricValues(metrics["recall@100"], metrics["ndcg@100"]),
                order,
            )
        )
    group_field = _selection_group_field(manifest.stage)
    grouped: dict[str, list[Candidate]] = {}
    for candidate, job in zip(selectable, manifest.jobs, strict=True):
        group = candidate_selection_group(manifest.stage, job, group_field)
        if group is None:
            continue
        grouped.setdefault(group, []).append(candidate)
    if manifest.stage.endswith("confirmation"):
        selected_job_ids = {
            group: next(
                job.job_id
                for job in manifest.jobs
                if job.seed == 42
                and candidate_selection_group(manifest.stage, job, group_field) == group
            )
            for group in grouped
        }
    else:
        selected_job_ids = {
            group: select_by_quality(
                candidates, recall_relative_dispersion=recall_relative_dispersion
            ).identifier
            for group, candidates in grouped.items()
        }
    body = {
        "schema": "g6-native500m-stage-selection/v1",
        "stage": manifest.stage,
        "manifest_logical_sha256": manifest.logical_sha256,
        "manifest_physical_sha256": manifest.physical_sha256,
        "batch_id": batch_ids[0] if len(batch_ids) == 1 else list(batch_ids),
        "recall_relative_dispersion": recall_relative_dispersion,
        "selection_group_field": group_field,
        "selected_job_ids": selected_job_ids,
        "candidates": rows,
    }
    document = {
        **body,
        "selection_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest(),
    }
    if output_path is not None:
        _write_immutable_json(output_path, document)
    return document


def candidate_selection_group(
    stage: str, job: QueueJob, group_field: str
) -> str | None:
    if stage in {"rq2_rq3_surface", "rq2_rq3_refinement"} and (
        job_id_has_coordinate(job.job_id, "random", 99)
        or job_id_has_coordinate(job.job_id, "rq0_anchor", 99)
    ):
        return None
    if stage == "rq2_rq3_confirmation" and "rq0_anchor" in job.job_id:
        return "rq0_anchor"
    return str(job.payload["parameters"].get(group_field, "all"))


def _replayed_candidate(job: QueueJob, order: int) -> dict[str, Any]:
    environment = job.environment
    declarations = job.payload["exact_reuse"]
    if not isinstance(declarations, list) or len(declarations) != 1:
        raise ValueError(f"{job.job_id}: exact reuse must name one source")
    declaration = declarations[0]
    path_value = declaration.get("source_selection_path") or environment.get(
        "G6_NATIVE500M_REUSE_SELECTION_PATH"
    )
    physical_sha256 = declaration.get(
        "source_selection_physical_sha256"
    ) or environment.get("G6_NATIVE500M_REUSE_SELECTION_PHYSICAL_SHA256")
    if not isinstance(path_value, str) or not isinstance(physical_sha256, str):
        raise ValueError(f"{job.job_id}: exact reuse selection binding is missing")
    path = Path(path_value).resolve(strict=True)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != physical_sha256:
        raise ValueError(
            f"{job.job_id}: exact reuse selection physical identity differs"
        )
    selection = _read_json(path)
    body = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if (
        selection.get("selection_sha256")
        != hashlib.sha256(canonical_bytes(body)).hexdigest()
    ):
        raise ValueError(
            f"{job.job_id}: exact reuse selection logical identity differs"
        )
    source_selection = job.payload.get("source_selection")
    expected_selection_sha256 = declaration.get("source_selection_sha256") or (
        source_selection.get("selection_sha256")
        if isinstance(source_selection, dict)
        else None
    )
    expected_stage = declaration.get("source_selection_stage") or (
        source_selection.get("stage") if isinstance(source_selection, dict) else None
    )
    if (
        selection["selection_sha256"] != expected_selection_sha256
        or selection.get("stage") != expected_stage
        or not reuse_source_is_eligible(
            job,
            declaration["source_job_id"],
            selection.get("selected_job_ids"),
            source_selection_stage=selection.get("stage"),
        )
    ):
        raise ValueError(f"{job.job_id}: exact reuse selection binding differs")
    candidates = selection.get("candidates")
    matches = (
        [row for row in candidates if row.get("job_id") == declaration["source_job_id"]]
        if isinstance(candidates, list)
        else []
    )
    if len(matches) != 1:
        raise ValueError(f"{job.job_id}: exact reuse source candidate is absent")
    source = matches[0]
    if source.get("job_logical_sha256") != declaration["source_contract_sha256"]:
        raise ValueError(f"{job.job_id}: exact reuse source contract differs")
    source_parameters = source.get("parameters")
    target_parameters = job.payload["parameters"]
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"{job.job_id}: exact reuse source artifacts are absent")
    for identity in artifacts.values():
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
            raise ValueError(f"{job.job_id}: exact reuse artifact identity is invalid")
        artifact_binding = Path(identity["path"])
        if artifact_binding.is_symlink() or not artifact_binding.is_file():
            raise ValueError(f"{job.job_id}: exact reuse source artifact differs")
        artifact_path = artifact_binding.resolve(strict=True)
        if _file_sha256(artifact_path) != identity["sha256"]:
            raise ValueError(f"{job.job_id}: exact reuse source artifact differs")
    if not isinstance(source_parameters, dict) or any(
        source_parameters.get(field) != target_parameters.get(field)
        for field in declaration["fields"]
    ):
        raise ValueError(f"{job.job_id}: exact reuse fields differ")
    return {
        "job_id": job.job_id,
        "job_logical_sha256": job.logical_sha256,
        "run_name": source["run_name"],
        "manifest_order": order,
        "best_epoch": source["best_epoch"],
        "training_horizon": source["training_horizon"],
        "restored_checkpoint_sha256": source["restored_checkpoint_sha256"],
        "parameters": target_parameters,
        "validation_metrics": source["validation_metrics"],
        "convergence": source.get("convergence"),
        "artifacts": artifacts,
        "reused_from": {
            "selection_sha256": selection["selection_sha256"],
            "job_id": source["job_id"],
            "job_logical_sha256": source["job_logical_sha256"],
        },
    }


def reuse_source_is_eligible(
    job: QueueJob,
    source_job_id: object,
    selected_job_ids: object,
    *,
    source_selection_stage: object = None,
) -> bool:
    if not isinstance(source_job_id, str) or not isinstance(selected_job_ids, dict):
        return False
    if source_job_id in selected_job_ids.values():
        return True
    if (
        job.payload.get("stage") in {"rq2_rq3_refinement", "rq2_rq3_confirmation"}
        and ":rq0_anchor:" in job.job_id
    ):
        return True
    return (
        job.payload.get("stage") == "rq2_rq3_confirmation"
        and job.seed in {43, 44}
        and source_selection_stage == "rq1_confirmation"
    )


def _validation_convergence(
    history: Mapping[str, Any], selected_recall: float
) -> dict[str, float]:
    epochs = history.get("epochs")
    if not isinstance(epochs, list) or len(epochs) != 26 or selected_recall <= 0:
        raise ValueError("validation convergence history is invalid")
    recalls = [float(row["recall@100"]) for row in epochs]
    threshold = 0.95 * selected_recall
    first = next(
        (index for index, value in enumerate(recalls, 1) if value >= threshold), 26
    )
    return {
        "first_epoch_at_95_percent": float(first),
        "normalized_recall_auc": sum(
            min(value / selected_recall, 1.0) for value in recalls
        )
        / len(recalls),
    }


def _selection_group_field(stage: str) -> str:
    if stage.startswith("controls"):
        return "backbone"
    if stage.startswith("rq0"):
        return "representation"
    if stage.startswith("rq1"):
        return "sid_initialization"
    if stage.startswith("rq2_rq3"):
        return "collision_policy"
    return "backbone"


def _ranking_metrics(
    relevant_by_user: list[list[int]],
    flattened_ranks: list[int],
    recommendations: torch.Tensor,
    catalog_size: int,
) -> dict[str, float]:
    ranks_by_user: list[list[int]] = []
    offset = 0
    for relevant in relevant_by_user:
        ranks_by_user.append(flattened_ranks[offset : offset + len(relevant)])
        offset += len(relevant)
    metrics: dict[str, float] = {}
    for cutoff in _CUTOFFS:
        recall_sum = capped_sum = ndcg_sum = mrr_sum = 0.0
        discounts = [1.0 / math.log2(rank + 1) for rank in range(1, cutoff + 1)]
        ideal = [0.0]
        for discount in discounts:
            ideal.append(ideal[-1] + discount)
        for ranks in ranks_by_user:
            hits = [rank for rank in ranks if 0 < rank <= cutoff]
            recall_sum += len(hits) / len(ranks)
            capped_sum += len(hits) / min(len(ranks), cutoff)
            ndcg_sum += (
                sum(1.0 / math.log2(rank + 1) for rank in hits)
                / ideal[min(len(ranks), cutoff)]
            )
            mrr_sum += 0.0 if not hits else 1.0 / min(hits)
        users = len(ranks_by_user)
        metrics[f"recall@{cutoff}"] = recall_sum / users
        metrics[f"capped_recall@{cutoff}"] = capped_sum / users
        metrics[f"ndcg@{cutoff}"] = ndcg_sum / users
        metrics[f"mrr@{cutoff}"] = mrr_sum / users
        metrics[f"coverage@{cutoff}"] = (
            len(set(recommendations[:, :cutoff].flatten().tolist())) / catalog_size
        )
    return metrics


def _semantic_metrics(
    relevant_by_user: list[list[int]],
    recommendations: torch.Tensor,
    positions: dict[int, int],
    semantic_codes: torch.Tensor,
) -> dict[str, float]:
    for relevant in relevant_by_user:
        if any(item_id not in positions for item_id in relevant):
            raise ValueError("relevant item is absent from the ordered catalog")
    metrics: dict[str, float] = {}
    recommendation_rows = recommendations.tolist()
    for cutoff in _CUTOFFS:
        depth_sums = [0.0] * semantic_codes.shape[1]
        for relevant, recommended in zip(
            relevant_by_user, recommendation_rows, strict=True
        ):
            relevant_codes = semantic_codes[
                torch.tensor([positions[item_id] for item_id in relevant])
            ]
            ranked_codes = semantic_codes[
                torch.tensor([positions[item_id] for item_id in recommended[:cutoff]])
            ]
            for depth in range(1, semantic_codes.shape[1] + 1):
                hits = 0
                for target in relevant_codes[:, :depth]:
                    if bool((ranked_codes[:, :depth] == target).all(1).any()):
                        hits += 1
                depth_sums[depth - 1] += hits / len(relevant)
        users = len(relevant_by_user)
        for depth, total in enumerate(depth_sums, 1):
            metrics[f"sid_prefix_recall@{cutoff}_l{depth}"] = total / users
        metrics[f"sid_exact_recall@{cutoff}"] = depth_sums[-1] / users
    return metrics


def _ragged(values: list[int], offsets: list[int]) -> list[list[int]]:
    rows = [values[start:end] for start, end in zip(offsets, offsets[1:])]
    if not rows or any(not row for row in rows):
        raise ValueError("ranking evidence has an empty relevance row")
    return rows


def _ranking_context_sha256(path: Path) -> str:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"cannot read ranking context: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("ranking context payload is invalid")
    digest = hashlib.sha256()
    scalars = {
        name: value
        for name, value in payload.items()
        if not isinstance(value, torch.Tensor)
    }
    digest.update(json.dumps(scalars, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(payload):
        value = payload[name]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _manifest_job(manifest: QueueManifest, job_id: str) -> QueueJob:
    matches = [job for job in manifest.jobs if job.job_id == job_id]
    if len(matches) != 1:
        raise ValueError(f"unknown manifest job {job_id!r}")
    return matches[0]


def _run_contract(manifest: QueueManifest, job: QueueJob) -> dict[str, Any]:
    return {
        "schema": "g6-native500m-run-contract/v1",
        "manifest_logical_sha256": manifest.logical_sha256,
        "manifest_physical_sha256": manifest.physical_sha256,
        "job_logical_sha256": job.logical_sha256,
        "config_logical_sha256": job.config_logical_sha256,
        "job": job.payload,
    }


def _validate_training_metadata(
    metadata: dict[str, Any],
    job: QueueJob,
    *,
    require_final_evidence: bool = True,
) -> None:
    expected = {
        "dataset_size": "500m",
        "batch_size": 512,
        "physical_batch_size": 512,
        "effective_batch_size": 512,
        "gradient_accumulation_steps": 1,
        "num_epochs": 26,
        "max_epochs": 26,
        "epochs_trained": 26,
        "stopped_epoch": 26,
        "early_stopped": False,
    }
    errors = [
        name
        for name, value in expected.items()
        if type(metadata.get(name)) is not type(value) or metadata.get(name) != value
    ]
    best_epoch = metadata.get("best_epoch")
    if (
        not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or not 1 <= best_epoch <= 26
    ):
        errors.append("best_epoch")
    invariants = metadata.get("transfer_invariants")
    if (
        not isinstance(invariants, dict)
        or invariants.get("restore_best_weights") is not True
    ):
        errors.append("restore_best_weights")
    schedule = (
        None if not isinstance(invariants, dict) else invariants.get("lr_schedule")
    )
    expected_schedule = job.payload["schedule"]
    if (
        not isinstance(schedule, dict)
        or (expected_schedule == "constant" and schedule.get("shape") != "constant")
        or (
            expected_schedule == "annealed"
            and (
                schedule.get("shape") == "constant"
                or metadata.get("lr_horizon_complete") is not True
            )
        )
    ):
        errors.append("schedule")
    if require_final_evidence:
        for name in ("ordered_catalog_sha256", "evaluator_configuration_sha256"):
            value = metadata.get(name)
            if not isinstance(value, str) or len(value) != 64:
                errors.append(name)
    if errors:
        raise ValueError(
            f"{job.run_name}: invalid final execution {sorted(set(errors))}"
        )


def _normalize_batch_ids(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    batch_ids = (value,) if isinstance(value, str) else tuple(value)
    if (
        not batch_ids
        or len(set(batch_ids)) != len(batch_ids)
        or any(not re.fullmatch(r"[0-9a-f]{32}", batch_id) for batch_id in batch_ids)
    ):
        raise ValueError("queue batch IDs are invalid")
    return tuple(sorted(batch_ids))


def _verify_queue_success(
    state: Path,
    *,
    batch_ids: tuple[str, ...],
    manifest: QueueManifest,
    job: QueueJob,
) -> None:
    successful = []
    manifest_by_run = {
        candidate.run_name: candidate
        for candidate in manifest.jobs
        if not candidate.payload["exact_reuse"]
    }
    for batch_id in batch_ids:
        batch = _read_json(state / "batches" / f"{batch_id}.json")
        records = []
        for service_job_id in batch.get("jobs", []):
            matches = [
                path
                for directory in ("completed", "failed")
                if (path := state / directory / f"{service_job_id}.json").is_file()
            ]
            if len(matches) != 1:
                raise ValueError("queue batch job is not terminal")
            records.append(_read_json(matches[0]))
        included = frozenset(
            manifest_by_run[record["run"]].job_id
            for record in records
            if record.get("run") in manifest_by_run
        )
        if len(included) != len(records):
            raise ValueError("queue batch contains a foreign job")
        specification = build_batch_specification(manifest, included_job_ids=included)
        if (
            batch.get("sealed") is not True
            or batch.get("atomic_submission") is not True
            or batch.get("specification_sha256") != specification.sha256
        ):
            raise ValueError("queue batch is not sealed")
        expected_rows = [
            row for row in specification.document["jobs"] if row["run"] == job.run_name
        ]
        completed = [
            record
            for record in records
            if record.get("run") == job.run_name and record.get("exit_code") == 0
        ]
        if len(expected_rows) == 1 and len(completed) == 1:
            expected = expected_rows[0]
            if any(
                completed[0].get(name) != expected[name]
                for name in ("script", "run", "data_group", "environment")
            ):
                raise ValueError(f"{job.run_name}: queue payload differs")
            successful.append(completed[0])
    if len(successful) != 1:
        raise ValueError(f"{job.run_name}: successful queue job is absent")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validation_selection_metrics(
    history: dict[str, Any], metadata: dict[str, Any], job: QueueJob
) -> dict[str, float]:
    if (
        history.get("schema") != "g6-native500m-validation-history/v1"
        or history.get("job_id") != job.job_id
        or history.get("job_logical_sha256") != job.logical_sha256
        or history.get("config_logical_sha256") != job.config_logical_sha256
        or history.get("selection_metric") != "recall@100"
        or history.get("best_epoch") != metadata.get("best_epoch")
    ):
        raise ValueError(f"{job.run_name}: validation history identity differs")
    epochs = history.get("epochs")
    if not isinstance(epochs, list) or [row.get("epoch") for row in epochs] != list(
        range(1, 27)
    ):
        raise ValueError(f"{job.run_name}: validation history is incomplete")
    recalls = [row.get("recall@100") for row in epochs]
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in recalls
    ):
        raise ValueError(f"{job.run_name}: invalid validation recall@100")
    recomputed_best_epoch = recalls.index(max(recalls)) + 1
    if metadata["best_epoch"] != recomputed_best_epoch:
        raise ValueError(
            f"{job.run_name}: best epoch disagrees with validation history"
        )
    selected = epochs[recomputed_best_epoch - 1]
    metrics = {}
    for name in ("recall@100", "ndcg@100"):
        value = selected.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{job.run_name}: invalid validation {name}")
        metrics[name] = float(value)
    return metrics


def _best_model_artifact(
    metadata: dict[str, Any], run_directory: Path
) -> tuple[str, Path]:
    artifact = metadata.get("best_model_artifact")
    if not isinstance(artifact, dict) or set(artifact) != {"schema", "path", "sha256"}:
        raise ValueError(f"{run_directory.name}: best model artifact is missing")
    relative = artifact["path"]
    sha256 = artifact["sha256"]
    if (
        artifact["schema"] != "g6-best-model-state/v1"
        or not isinstance(relative, str)
        or not isinstance(sha256, str)
        or len(sha256) != 64
    ):
        raise ValueError(f"{run_directory.name}: best model artifact identity differs")
    expected = run_directory / "best_model_state.pt"
    candidates = (Path(relative), run_directory.parent.parent / relative)
    matches = [path.resolve() for path in candidates if path.is_file()]
    if expected.resolve() not in matches or _file_sha256(expected) != sha256:
        raise ValueError(f"{run_directory.name}: best model artifact hash differs")
    return sha256, expected.resolve()


def _write_immutable_json(path: Path, document: dict[str, Any]) -> None:
    persist_immutable_bytes(path, canonical_bytes(document), label="evidence")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
