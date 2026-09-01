from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import tempfile

from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_candidates import (
    TREATMENTS,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_lesion_candidates import (
    Rq14LesionDiagnosticCandidate,
    diagnostic_candidates,
)
from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


class Rq14LesionEvidenceError(RuntimeError):
    pass


_EXPERIMENT = Path(__file__).parents[1]
_CONFIG = _EXPERIMENT / "configs/rq14_pretrained_lesion_variant.py"
_METRICS = ("recall@10", "ndcg@10", "recall@100", "ndcg@100", "coverage@100")
_EXPECTED_EFFECTS = {
    "distinct_vs_shared_cls_only",
    "distinct_vs_shared_history",
    "history_vs_cls_only_shared",
    "history_vs_cls_only_distinct",
}
_IMPLEMENTATION_FILES = (
    Path("dcn/config/query_retrieval_training.py"),
    Path("dcn/eval/callback.py"),
    Path("dcn/models/cross_attention_retrieval.py"),
    Path("dcn/models/history_tokens.py"),
    Path(
        "experiments/g1_sasrec_item_ids_likes/analysis/rq14_pretrained_lesion_candidates.py"
    ),
    Path(
        "experiments/g1_sasrec_item_ids_likes/analysis/rq14_pretrained_lesion_evidence.py"
    ),
    Path(
        "experiments/g1_sasrec_item_ids_likes/configs/rq14_pretrained_lesion_variant.py"
    ),
    Path(
        "experiments/g1_sasrec_item_ids_likes/launchers/architecture/rq14_pretrained_lesions_500m.sh"
    ),
)


def validate_diagnostic_artifact(
    candidate: Rq14LesionDiagnosticCandidate,
    metadata: Mapping[str, object],
    final_metrics: Mapping[str, object],
    diagnostic: Mapping[str, object],
    *,
    expected_checkpoint_sha256: str,
) -> dict[str, object]:
    expected_metadata = {
        "dataset_size": "500m",
        "seed": 42,
        "effective_batch_size": 1280,
        "embedding_learning_rate": 0.00025,
        "deep_learning_rate": 0.00075,
        "stopped_epoch": 20,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "training_method": "pretrained_finetune",
        "ntp_targets_per_epoch": 0,
        "auxiliary_ntp_weight": 0.0,
        "query_slots_shared": candidate.query_slots_shared,
        "include_history_memory": candidate.include_history_memory,
        "num_query_slots": 4,
        "diagnostic_protocol": "rq14_selected_cell_inference_lesions_v1",
        "diagnostic_lesions": list(candidate.lesions),
        "diagnostic_full_user_evaluation": True,
        "diagnostic_after_best_checkpoint_restore": True,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise Rq14LesionEvidenceError(
                f"{candidate.run_name}: training invariant {key} is absent or changed"
            )
    expected_protocol = {
        "seed": 42,
        "effective_batch_size": 1280,
        "embedding_learning_rate": 0.00025,
        "deep_learning_rate": 0.00075,
        "horizon_epochs": 20,
        "lr_schedule": "linear",
        "lr_schedule_optimizer_group_scope": "both",
        "training_method": "pretrained_finetune",
        "auxiliary_ntp_weight": 0.0,
        "source_checkpoint_sha256": expected_checkpoint_sha256,
        "best_checkpoint_restored": True,
    }
    treatment = diagnostic.get("treatment")
    protocol = diagnostic.get("training_protocol")
    if (
        diagnostic.get("schema_version") != 1
        or diagnostic.get("run_name") != candidate.run_name
        or diagnostic.get("dataset_size") != "500m"
        or treatment
        != {
            "query_slots_shared": candidate.query_slots_shared,
            "include_history_memory": candidate.include_history_memory,
        }
        or protocol != expected_protocol
        or not _sha256(diagnostic.get("best_model_state_sha256"))
    ):
        raise Rq14LesionEvidenceError(
            f"{candidate.run_name}: diagnostic training binding is absent or changed"
        )
    normal = _metrics(diagnostic.get("normal_metrics"), candidate.run_name)
    if normal != _metrics(final_metrics, candidate.run_name):
        raise Rq14LesionEvidenceError(
            f"{candidate.run_name}: normal diagnostics differ from final metrics"
        )
    lesions = diagnostic.get("lesions")
    if not isinstance(lesions, Mapping) or set(lesions) != set(candidate.lesions):
        raise Rq14LesionEvidenceError(f"{candidate.run_name}: lesion set is incomplete")
    validated_lesions = {}
    for name in candidate.lesions:
        record = lesions[name]
        if not isinstance(record, Mapping):
            raise Rq14LesionEvidenceError(
                f"{candidate.run_name}: invalid {name} lesion"
            )
        lesion_metrics = _metrics(record.get("metrics"), f"{candidate.run_name}:{name}")
        query_change = _query_change(
            record.get("query_change"), f"{candidate.run_name}:{name}"
        )
        validated_lesions[name] = {
            "metrics": lesion_metrics,
            "query_change": query_change,
            "effect": classify_lesion_effect(normal, lesion_metrics, query_change),
        }
    return {
        "run_name": candidate.run_name,
        "treatment": candidate.treatment,
        "normal_metrics": normal,
        "best_model_state_sha256": diagnostic["best_model_state_sha256"],
        "lesions": validated_lesions,
    }


def classify_lesion_effect(
    normal_metrics: Mapping[str, object],
    lesion_metrics: Mapping[str, object],
    query_change: Mapping[str, object],
) -> dict[str, object]:
    normal = _metrics(normal_metrics, "normal")
    lesion = _metrics(lesion_metrics, "lesion")
    change = _query_change(query_change, "lesion")
    fraction = float(change["changed_user_fraction"])
    mean_l2 = float(change["mean_l2_change"])
    max_l2 = float(change["max_l2_change"])
    if fraction == 0 and mean_l2 == 0 and max_l2 == 0:
        state_use = "states_ignored"
    elif fraction > 0 and mean_l2 > 0 and max_l2 > 0:
        state_use = "states_used"
    else:
        raise Rq14LesionEvidenceError(
            "query-change diagnostics are internally inconsistent"
        )
    recall_delta = lesion["recall@100"] - normal["recall@100"]
    ndcg_delta = lesion["ndcg@100"] - normal["ndcg@100"]
    if abs(recall_delta) <= 0.003 and abs(ndcg_delta) <= 0.001:
        recommendation_effect = "within_noise_or_redundant"
    elif recall_delta < -0.003 or ndcg_delta < -0.001:
        recommendation_effect = "resolved_degradation_after_removal"
    else:
        recommendation_effect = "resolved_change_after_removal"
    return {
        "state_use": state_use,
        "recommendation_effect": recommendation_effect,
        "lesion_minus_normal_recall@100": recall_delta,
        "lesion_minus_normal_ndcg@100": ndcg_delta,
    }


def validate_selected_rerun_compatibility(
    source: Mapping[str, object], normal_metrics: Mapping[str, object], context: str
) -> dict[str, float]:
    source_metrics = source.get("full_user_metrics")
    if not isinstance(source_metrics, Mapping):
        raise Rq14LesionEvidenceError(f"{context}: selected source metrics are absent")
    normal = _metrics(normal_metrics, context)
    deltas = {}
    for name, band in (("recall@100", 0.003), ("ndcg@100", 0.001)):
        value = source_metrics.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise Rq14LesionEvidenceError(
                f"{context}: selected source {name} is absent or invalid"
            )
        delta = normal[name] - float(value)
        if abs(delta) > band:
            raise Rq14LesionEvidenceError(
                f"{context}: diagnostic rerun differs from the selected source beyond the band"
            )
        deltas[f"diagnostic_minus_source_{name}"] = delta
    return deltas


def collect_lesion_evidence(logs: Path, rq14_results_path: Path) -> dict[str, object]:
    rq14_results = _load_json(rq14_results_path)
    if (
        rq14_results.get("research_question")
        != "RQ14 pretrained decoder-decoder query memory"
        or rq14_results.get("dataset_size") != "500m"
        or rq14_results.get("claims_status")
        != "unexpected_result_requires_investigation"
        or rq14_results.get("required_followups") != []
    ):
        raise Rq14LesionEvidenceError(
            "pretrained RQ14 source result is not investigation-ready"
        )
    checkpoint_sha256 = rq14_results.get("checkpoint_sha256")
    if not _sha256(checkpoint_sha256):
        raise Rq14LesionEvidenceError(
            "pretrained RQ14 source checkpoint binding is invalid"
        )
    selected = rq14_results.get("selected")
    unexpected = rq14_results.get("unexpected_effects")
    if (
        not isinstance(selected, Mapping)
        or set(selected) != set(TREATMENTS)
        or not isinstance(unexpected, Mapping)
        or set(unexpected) != _EXPECTED_EFFECTS
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or abs(value) > 0.003
            for value in unexpected.values()
        )
    ):
        raise Rq14LesionEvidenceError("pretrained RQ14 selected effects are incomplete")

    runs = {}
    artifacts = {}
    for candidate in diagnostic_candidates():
        source = selected[candidate.treatment]
        if (
            not isinstance(source, Mapping)
            or source.get("deep_lr") != 0.00075
            or source.get("embedding_lr") != 0.00025
            or source.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise Rq14LesionEvidenceError(
                f"{candidate.treatment}: selected-cell source changed"
            )
        directory = logs / candidate.run_name
        metadata = _load_json(directory / "training_metadata.json")
        final_metrics = _load_json(directory / "final_metrics.json")
        diagnostic = _load_json(directory / "rq14_lesion_diagnostics.json")
        initialization = metadata.get("first_stage_initialization")
        if not isinstance(initialization, Mapping):
            raise Rq14LesionEvidenceError(
                f"{candidate.run_name}: initialization is absent"
            )
        checkpoint_path = initialization.get("checkpoint_path")
        if initialization.get(
            "checkpoint_sha256"
        ) != checkpoint_sha256 or not isinstance(checkpoint_path, str):
            raise Rq14LesionEvidenceError(
                f"{candidate.run_name}: source checkpoint changed"
            )
        assignments = [
            f"G1_RQ14_LESION_RUN={candidate.run_name}",
            f"G1_RQ15_SOURCE_RUN={Path(checkpoint_path).parent.name}",
            f"G1_RQ15_FIRST_STAGE_CHECKPOINT={checkpoint_path}",
        ]
        if not verify_artifact.verify_config(directory, _CONFIG, assignments):
            raise Rq14LesionEvidenceError(
                f"{candidate.run_name}: recipe verification failed"
            )
        record = validate_diagnostic_artifact(
            candidate,
            metadata,
            final_metrics,
            diagnostic,
            expected_checkpoint_sha256=str(checkpoint_sha256),
        )
        rerun_deltas = validate_selected_rerun_compatibility(
            source, record["normal_metrics"], candidate.run_name
        )
        artifact_sha256 = {
            name: _file_sha256(directory / name)
            for name in (
                "training_metadata.json",
                "final_metrics.json",
                "sweep.log",
                "rq14_lesion_diagnostics.json",
            )
        }
        record.update(
            {
                "source_selected_run_name": source.get("run_name"),
                "selected_rerun_compatibility": rerun_deltas,
                "artifact_sha256": artifact_sha256,
            }
        )
        runs[candidate.treatment] = record
        artifacts[candidate.run_name] = artifact_sha256
    return {
        "schema_version": 1,
        "research_question": "RQ14 pretrained decoder-decoder query-memory lesions",
        "dataset_size": "500m",
        "status": "passed",
        "claims_status": "diagnostics_complete_claims_not_published",
        "source_rq14_results_sha256": _file_sha256(rq14_results_path),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_unexpected_effects": dict(unexpected),
        "implementation_sha256": {
            str(path): _file_sha256(path) for path in _IMPLEMENTATION_FILES
        },
        "run_artifacts": artifacts,
        "runs": runs,
    }


def build_lesion_explanation(evidence: Mapping[str, object]) -> dict[str, object]:
    runs = evidence.get("runs")
    if (
        evidence.get("status") != "passed"
        or evidence.get("claims_status") != "diagnostics_complete_claims_not_published"
        or not isinstance(runs, Mapping)
        or set(runs) != set(TREATMENTS)
    ):
        raise Rq14LesionEvidenceError("lesion evidence is incomplete")
    findings = {}
    counts = {
        "states_used": 0,
        "states_ignored": 0,
        "within_noise_or_redundant": 0,
        "resolved_degradation_after_removal": 0,
        "resolved_change_after_removal": 0,
    }
    for candidate in diagnostic_candidates():
        run = runs[candidate.treatment]
        if not isinstance(run, Mapping):
            raise Rq14LesionEvidenceError("lesion run is invalid")
        normal = _metrics(run.get("normal_metrics"), candidate.treatment)
        lesions = run.get("lesions")
        if not isinstance(lesions, Mapping) or set(lesions) != set(candidate.lesions):
            raise Rq14LesionEvidenceError("lesion explanation set is incomplete")
        treatment_findings = {}
        for name in candidate.lesions:
            lesion = lesions[name]
            if not isinstance(lesion, Mapping):
                raise Rq14LesionEvidenceError("lesion explanation row is invalid")
            effect = classify_lesion_effect(
                normal,
                lesion.get("metrics", {}),
                lesion.get("query_change", {}),
            )
            counts[str(effect["state_use"])] += 1
            counts[str(effect["recommendation_effect"])] += 1
            treatment_findings[name] = effect
        findings[candidate.treatment] = treatment_findings
    return {
        "schema_version": 1,
        "research_question": "RQ14 pretrained decoder-decoder query-memory lesions",
        "status": "passed",
        "claims_status": "diagnostics_complete_claims_not_published",
        "evidence_sha256": _canonical_sha256(evidence),
        "findings": findings,
        "summary": counts,
        "interpretation_rule": (
            "A nonzero query-representation change shows that the removed state is used. "
            "When Recall@100 and NDCG@100 changes remain inside their native-500M bands, "
            "the marginal recommendation benefit is within noise or redundant. Exact zero "
            "query change identifies an ignored state. Within-band treatment deltas are not gains."
        ),
    }


def _metrics(value: object, context: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise Rq14LesionEvidenceError(f"{context}: metrics are absent")
    result = {}
    for name in _METRICS:
        metric = value.get(name)
        if (
            not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(metric)
        ):
            raise Rq14LesionEvidenceError(f"{context}: {name} is absent or invalid")
        result[name] = float(metric)
    return result


def _query_change(value: object, context: str) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        raise Rq14LesionEvidenceError(f"{context}: query-change diagnostics are absent")
    result: dict[str, float | int] = {}
    num_users = value.get("num_users")
    if not isinstance(num_users, int) or isinstance(num_users, bool) or num_users < 1:
        raise Rq14LesionEvidenceError(f"{context}: query user count is invalid")
    result["num_users"] = num_users
    for name in (
        "changed_user_fraction",
        "mean_l2_change",
        "max_l2_change",
        "mean_relative_l2_change",
        "mean_cosine_distance",
    ):
        metric = value.get(name)
        if (
            not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(metric)
            or metric < 0
        ):
            raise Rq14LesionEvidenceError(f"{context}: {name} is invalid")
        result[name] = float(metric)
    if float(result["changed_user_fraction"]) > 1:
        raise Rq14LesionEvidenceError(f"{context}: changed-user fraction is invalid")
    return result


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq14LesionEvidenceError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq14LesionEvidenceError(f"{path}: expected JSON object")
    return value


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--rq14-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--explanation", type=Path, required=True)
    arguments = parser.parse_args()
    evidence = collect_lesion_evidence(arguments.logs, arguments.rq14_results)
    explanation = build_lesion_explanation(evidence)
    _write_json(arguments.output, evidence)
    _write_json(arguments.explanation, explanation)


if __name__ == "__main__":
    main()
