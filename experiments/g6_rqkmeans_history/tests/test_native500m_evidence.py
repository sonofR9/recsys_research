from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from dcn.eval.ranking_evidence import RankingEvidence, write_ranking_evidence
from experiments.g6_rqkmeans_history.native500m.analysis.collect import (
    metrics_agree,
    recompute_metrics,
    recompute_sid_diagnostics,
    recompute_slice_diagnostics,
)
from experiments.g6_rqkmeans_history.native500m.analysis.topk_evidence import (
    TopKContext,
    load_topk_evidence,
    tensor_sha256,
    write_topk_evidence,
)
from experiments.g6_rqkmeans_history.native500m.launchers.evaluate_winner import (
    _evaluation_source,
    _verify_rank_metric_agreement,
)
from experiments.g6_rqkmeans_history.native500m.launchers.queue import (
    QueueJob,
    canonical_bytes,
)


def _sha(digit: str) -> str:
    return digit * 64


def _ranking() -> RankingEvidence:
    return RankingEvidence(
        user_ids=torch.tensor([7, 8]),
        history_item_ids=torch.tensor([91, 92, 93]),
        history_offsets=torch.tensor([0, 2, 3]),
        relevant_item_ids=torch.tensor([0, 1, 2]),
        relevance_offsets=torch.tensor([0, 2, 3]),
        relevant_train_frequencies=torch.tensor([10, 5, 1]),
        relevant_ranks=torch.tensor([1, 0, 2]),
        max_k=100,
    )


def _recommendations() -> torch.Tensor:
    first = torch.tensor([0, *range(3, 102)])
    second = torch.tensor([3, 2, *range(4, 102)])
    return torch.stack((first, second))


def _semantic_codes() -> torch.Tensor:
    codes = torch.arange(120).repeat(2, 1).t()
    codes[0] = torch.tensor([1, 1])
    codes[1] = torch.tensor([1, 2])
    codes[2] = torch.tensor([2, 1])
    codes[3] = torch.tensor([2, 2])
    return codes


def _write_pair(tmp_path: Path) -> tuple[Path, Path, Path, TopKContext]:
    ranking = _ranking()
    context_path = tmp_path / "context.pt"
    ranking_path = tmp_path / "ranking.pt"
    ranking_context_sha256 = write_ranking_evidence(
        ranking, context_path=context_path, ranking_path=ranking_path
    )
    catalog = torch.arange(120)
    context = TopKContext(
        dataset="yambda-500m",
        split="final-seven-days",
        ranking_context_sha256=ranking_context_sha256,
        ordered_catalog_sha256=tensor_sha256(catalog),
        checkpoint_sha256=_sha("1"),
        evaluator_configuration_sha256=_sha("2"),
        stage="rq0_final",
        job_id="rq0:final",
        job_logical_sha256=_sha("3"),
        manifest_logical_sha256=_sha("4"),
        semantic_codes_sha256=tensor_sha256(_semantic_codes()),
    )
    topk_path = tmp_path / "topk.pt"
    write_topk_evidence(
        topk_path,
        context=context,
        user_ids=ranking.user_ids,
        recommended_item_ids=_recommendations(),
        ordered_catalog_item_ids=catalog,
    )
    return context_path, ranking_path, topk_path, context


def test_top100_recomputes_concrete_coverage_and_sid_metrics(tmp_path: Path) -> None:
    context_path, ranking_path, topk_path, context = _write_pair(tmp_path)
    evidence = load_topk_evidence(topk_path, expected_context=context)

    metrics = recompute_metrics(
        context_path=context_path,
        ranking_path=ranking_path,
        topk=evidence,
        semantic_codes=_semantic_codes(),
    )

    assert metrics["recall@100"] == pytest.approx(0.75)
    assert metrics["capped_recall@100"] == pytest.approx(0.75)
    assert metrics["mrr@100"] == pytest.approx(0.75)
    assert metrics["coverage@100"] == pytest.approx(101 / 120)
    assert metrics["sid_exact_recall@100"] == pytest.approx(0.75)
    assert metrics["sid_prefix_recall@100_l1"] == pytest.approx(1.0)
    assert metrics["sid_prefix_recall@100_l2"] == pytest.approx(0.75)
    assert metrics["num_users"] == 2.0


def test_recomputes_intrinsic_sid_and_eligible_slice_diagnostics() -> None:
    codes = _semantic_codes()
    codes[1] = codes[0]
    embeddings = torch.nn.functional.normalize(
        torch.arange(120 * 3, dtype=torch.float64).reshape(120, 3) + 1,
        dim=1,
    )
    centroids = torch.zeros((2, 120, 3), dtype=torch.float64)
    centroids[0, :, 0] = torch.arange(120)
    centroids[1, :, 1] = torch.arange(120)

    diagnostics = recompute_sid_diagnostics(
        semantic_codes=codes,
        normalized_content_vectors=embeddings,
        codebook_centroids=centroids,
        codes_per_level=(120, 120),
    )
    slices = recompute_slice_diagnostics(
        control=_ranking(),
        treatment=_ranking(),
        ordered_catalog_item_ids=torch.arange(120),
        semantic_codes=codes,
    )

    assert diagnostics["identifier_collision_rate"] == pytest.approx(1 / 120)
    assert diagnostics["collided_item_fraction"] == pytest.approx(2 / 120)
    assert diagnostics["unique_base_tuples"] == 119
    assert len(diagnostics["p95_occupied_load"]) == 2
    assert len(diagnostics["intra_code_cosine_similarity"]) == 2
    assert len(diagnostics["reconstruction_mse_by_depth"]) == 2
    assert set(slices) == {
        "frequency_low",
        "frequency_middle",
        "frequency_high",
        "history_has_collided_base_sid",
        "history_has_no_collided_base_sid",
        "target_bucket_size_1",
        "target_bucket_size_2",
        "target_bucket_size_3_to_4",
        "target_bucket_size_5_plus",
    }


def test_top100_rejects_context_or_payload_drift(tmp_path: Path) -> None:
    _, _, topk_path, context = _write_pair(tmp_path)

    changed = TopKContext(**{**context.to_dict(), "checkpoint_sha256": _sha("9")})
    with pytest.raises(ValueError, match="context"):
        load_topk_evidence(topk_path, expected_context=changed)

    payload = torch.load(topk_path, weights_only=True)
    payload["recommended_item_ids"][0, 0] = 119
    topk_path.chmod(0o644)
    torch.save(payload, topk_path)
    with pytest.raises(ValueError, match="logical SHA-256"):
        load_topk_evidence(topk_path, expected_context=context)


def test_recomputation_rejects_rank_and_saved_metric_disagreement(
    tmp_path: Path,
) -> None:
    context_path, ranking_path, topk_path, context = _write_pair(tmp_path)
    evidence = load_topk_evidence(topk_path, expected_context=context)
    changed = evidence.recommended_item_ids.clone()
    changed[1, 0], changed[1, 1] = changed[1, 1].clone(), changed[1, 0].clone()
    changed_path = tmp_path / "changed.pt"
    write_topk_evidence(
        changed_path,
        context=context,
        user_ids=evidence.user_ids,
        recommended_item_ids=changed,
        ordered_catalog_item_ids=evidence.ordered_catalog_item_ids,
    )

    with pytest.raises(ValueError, match="compact ranks"):
        recompute_metrics(
            context_path=context_path,
            ranking_path=ranking_path,
            topk=load_topk_evidence(changed_path, expected_context=context),
            semantic_codes=_semantic_codes(),
        )

    recomputed = recompute_metrics(
        context_path=context_path,
        ranking_path=ranking_path,
        topk=evidence,
        semantic_codes=_semantic_codes(),
    )
    saved = {**recomputed, "recall@100": recomputed["recall@100"] + 1e-5}
    with pytest.raises(ValueError, match="recall@100"):
        metrics_agree(saved, recomputed, tolerance=1e-8)


def test_top100_write_is_immutable(tmp_path: Path) -> None:
    _, _, topk_path, context = _write_pair(tmp_path)
    existing = load_topk_evidence(topk_path, expected_context=context)
    changed = existing.recommended_item_ids.clone()
    changed[0, 0] = 119

    with pytest.raises(RuntimeError, match="immutable"):
        write_topk_evidence(
            topk_path,
            context=context,
            user_ids=existing.user_ids,
            recommended_item_ids=changed,
            ordered_catalog_item_ids=existing.ordered_catalog_item_ids,
        )


def test_winner_crosscheck_uses_rank_metrics_not_second_topk_diagnostics() -> None:
    scored = {
        "recall@100": 0.25,
        "ndcg@100": 0.2,
        "coverage@100": 0.5,
        "sid_exact_recall@100": 0.4,
        "num_users": 10.0,
    }
    recomputed = {
        **scored,
        "coverage@100": 0.6,
        "sid_exact_recall@100": 0.3,
    }

    _verify_rank_metric_agreement(scored, recomputed)

    recomputed["recall@100"] = 0.24
    with pytest.raises(ValueError, match="recall@100"):
        _verify_rank_metric_agreement(scored, recomputed)

    for invalid in (float("nan"), float("inf"), float("-inf")):
        recomputed["recall@100"] = invalid
        with pytest.raises(ValueError, match="recall@100"):
            _verify_rank_metric_agreement(scored, recomputed)

        recomputed["recall@100"] = scored["recall@100"]
        invalid_scored = {**scored, "recall@100": invalid}
        with pytest.raises(ValueError, match="recall@100"):
            _verify_rank_metric_agreement(invalid_scored, recomputed)


def test_exact_reuse_evaluation_resolves_authenticated_selected_source(
    tmp_path: Path,
) -> None:
    source_job = {
        "schema": "g6-native500m-job/v1",
        "job_id": "source:job",
        "stage": "source_stage",
        "parameters": {
            "run_name": "source_run",
            "runner": "source.py",
            "config_logical_sha256": _sha("3"),
            "data_group": "source-data",
            "environment": {"G6_NATIVE500M_SOURCE_SHA256": _sha("4")},
        },
    }
    source_job_sha256 = hashlib.sha256(canonical_bytes(source_job)).hexdigest()
    source_selection_body = {
        "stage": "source_stage",
        "manifest_logical_sha256": _sha("5"),
        "manifest_physical_sha256": _sha("6"),
        "selected_job_ids": {"all": "source:job"},
    }
    source_selection = {
        **source_selection_body,
        "selection_sha256": hashlib.sha256(
            canonical_bytes(source_selection_body)
        ).hexdigest(),
    }
    source_selection_path = tmp_path / "source-selection.json"
    source_selection_path.write_bytes(canonical_bytes(source_selection))
    source_contract = {
        "manifest_logical_sha256": _sha("5"),
        "manifest_physical_sha256": _sha("6"),
        "job_logical_sha256": source_job_sha256,
        "job": source_job,
    }
    source_contract_path = tmp_path / "g6_native500m_job.json"
    source_contract_path.write_bytes(canonical_bytes(source_contract))
    declaration = {
        "source_job_id": "source:job",
        "source_contract_sha256": source_job_sha256,
        "source_selection_stage": "source_stage",
        "source_selection_sha256": source_selection["selection_sha256"],
        "source_selection_physical_sha256": hashlib.sha256(
            source_selection_path.read_bytes()
        ).hexdigest(),
        "source_selection_path": str(source_selection_path),
    }
    target = QueueJob(
        job_id="target:job",
        run_name="target_run",
        runner="target.py",
        config_logical_sha256=_sha("8"),
        data_group="target-data",
        logical_sha256=_sha("9"),
        payload={"exact_reuse": [declaration]},
        environment={},
    )
    target_selection = {
        "stage": "target_stage",
        "manifest_logical_sha256": _sha("a"),
        "candidates": [
            {
                "job_id": "target:job",
                "reused_from": {
                    "job_id": "source:job",
                    "job_logical_sha256": source_job_sha256,
                },
                "artifacts": {
                    "g6_native500m_job.json": {
                        "path": str(source_contract_path),
                        "sha256": hashlib.sha256(
                            source_contract_path.read_bytes()
                        ).hexdigest(),
                    }
                },
            }
        ],
    }

    resolved, manifest, selection = _evaluation_source(target, target_selection)

    assert resolved.job_id == "source:job"
    assert resolved.run_name == "source_run"
    assert resolved.logical_sha256 == source_job_sha256
    assert manifest.stage == "source_stage"
    assert manifest.logical_sha256 == _sha("5")
    assert selection == source_selection

    source_selection["selected_job_ids"] = {"all": "other:job"}
    changed_body = {
        name: value
        for name, value in source_selection.items()
        if name != "selection_sha256"
    }
    source_selection["selection_sha256"] = hashlib.sha256(
        canonical_bytes(changed_body)
    ).hexdigest()
    declaration["source_selection_sha256"] = source_selection["selection_sha256"]
    source_selection_path.write_bytes(canonical_bytes(source_selection))
    declaration["source_selection_physical_sha256"] = hashlib.sha256(
        source_selection_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="source contract"):
        _evaluation_source(target, target_selection)

    comparator = replace(
        target,
        job_id="rq2_rq3_confirmation:rq0_anchor:42",
        payload={
            "stage": "rq2_rq3_confirmation",
            "exact_reuse": [declaration],
        },
    )
    comparator_selection = {
        **target_selection,
        "candidates": [
            {
                **target_selection["candidates"][0],
                "job_id": comparator.job_id,
            }
        ],
    }

    resolved, _, _ = _evaluation_source(comparator, comparator_selection)

    assert resolved.job_id == "source:job"
