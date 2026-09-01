from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile

from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_candidates import (
    TREATMENTS,
    launch_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_report import (
    current_implementation_sha256,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_query_memory_audit import (
    run_query_memory_model_probe,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_audit import (
    run_model_correctness_probe,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import (
    source_candidates,
)


class Rq14PretrainedAuditError(RuntimeError):
    pass


_EXPERIMENT = Path(__file__).parents[1]
_CONFIG = _EXPERIMENT / "configs/rq14_pretrained_query_variant.py"
def build_correctness_audit(
    logs: Path,
    results_path: Path,
    *,
    query_probe: Mapping[str, object] | None = None,
    checkpoint_probe: Mapping[str, object] | None = None,
    recipe_probe: Mapping[str, object] | None = None,
) -> dict[str, object]:
    results = _load_json(results_path)
    artifacts, checkpoint_sha256 = _validate_results_and_artifacts(logs, results)
    query = dict(query_probe or run_query_memory_model_probe())
    checkpoint = dict(
        checkpoint_probe
        or run_model_correctness_probe()["checkpoint_copy_identity"]
    )
    recipes = dict(recipe_probe or _production_recipe_probe())
    checks = {
        "treatment_recipes": {
            "passed": recipes.get("passed") is True,
            **recipes,
        },
        "exact_checkpoint_load_scope": {
            "passed": all(
                checkpoint.get(key) is True
                for key in (
                    "copied_item_embedding",
                    "copied_memory_encoder",
                    "copied_tokenizer",
                    "preserved_decoder",
                    "preserved_decoder_query",
                    "preserved_query_projection",
                    "preserved_query_slots",
                )
            ),
            **checkpoint,
        },
        "candidate_only_loss_and_target_exclusion": {
            "passed": bool(
                query.get("target_only_query_max_delta", 1.0) <= 1e-6
                and query.get("candidate_targets_per_example") == 1
                and query.get("positive_ids") == [3, 11]
                and query.get("changed_positive_ids") == [4, 12]
                and recipes.get("all_candidate_only") is True
            ),
        },
        "memory_ordering_and_gradients": {
            "passed": bool(
                query.get("slot_order_preserved") is True
                and query.get("history_precedes_slots") is True
                and query.get("cls_only_memory_lengths") == [4, 4]
                and query.get("history_memory_lengths") == [6, 5]
                and _positive_list(query.get("distinct_slot_gradient_l1"), 4)
                and _positive_list(query.get("shared_slot_gradient_l1"), 1)
                and _positive(query.get("history_embedding_gradient_l1"))
                and _positive(query.get("decoder_gradient_l1"))
            ),
        },
        "horizon_and_artifact_binding": {
            "passed": len(artifacts) >= 12 and _sha256(checkpoint_sha256),
            "run_count": len(artifacts),
        },
    }
    failed = [name for name, record in checks.items() if record["passed"] is not True]
    if failed:
        raise Rq14PretrainedAuditError("correctness audit failed: " + ", ".join(failed))
    return {
        "schema_version": 1,
        "research_question": "RQ14 pretrained decoder-decoder query memory",
        "dataset_size": "500m",
        "status": "passed",
        "checks": checks,
        "checkpoint_sha256": checkpoint_sha256,
        "run_artifacts": artifacts,
        "implementation_sha256": current_implementation_sha256(),
    }


def _validate_results_and_artifacts(
    logs: Path, results: Mapping[str, object]
) -> tuple[dict[str, dict[str, str]], str]:
    if (
        results.get("research_question")
        != "RQ14 pretrained decoder-decoder query memory"
        or results.get("dataset_size") != "500m"
        or results.get("required_followups") != []
    ):
        raise Rq14PretrainedAuditError("pretrained RQ14 results are unresolved")
    treatments = results.get("treatments")
    if not isinstance(treatments, Mapping) or set(treatments) != set(TREATMENTS):
        raise Rq14PretrainedAuditError("pretrained RQ14 treatments are incomplete")
    artifacts = {}
    checkpoint_hashes = set()
    for treatment in TREATMENTS:
        treatment_record = treatments[treatment]
        records = (
            treatment_record.get("artifacts")
            if isinstance(treatment_record, Mapping)
            else None
        )
        if not isinstance(records, list) or len(records) < 3:
            raise Rq14PretrainedAuditError("pretrained RQ14 surface is incomplete")
        for record in records:
            if not isinstance(record, Mapping):
                raise Rq14PretrainedAuditError("invalid pretrained RQ14 artifact")
            run_name = record.get("run_name")
            expected = record.get("artifact_sha256")
            checkpoint = record.get("checkpoint_sha256")
            if not isinstance(run_name, str) or not isinstance(expected, Mapping):
                raise Rq14PretrainedAuditError("invalid artifact binding")
            actual = {
                name: _file_sha256(logs / run_name / name) for name in expected
            }
            if actual != dict(expected):
                raise Rq14PretrainedAuditError(f"{run_name}: artifact hash mismatch")
            artifacts[run_name] = actual
            checkpoint_hashes.add(checkpoint)
            metadata = _load_json(logs / run_name / "training_metadata.json")
            if (
                metadata.get("stopped_epoch") != 20
                or metadata.get("lr_horizon_complete") is not True
                or metadata.get("training_method") != "pretrained_finetune"
                or metadata.get("ntp_targets_per_epoch") != 0
                or metadata.get("auxiliary_ntp_weight") != 0
            ):
                raise Rq14PretrainedAuditError(f"{run_name}: wrong training semantics")
    if len(checkpoint_hashes) != 1:
        raise Rq14PretrainedAuditError("runs do not share one exact checkpoint")
    checkpoint = next(iter(checkpoint_hashes))
    if not _sha256(checkpoint):
        raise Rq14PretrainedAuditError("invalid selected checkpoint hash")
    return artifacts, checkpoint


def _production_recipe_probe() -> dict[str, object]:
    previous = {
        key: os.environ.get(key)
        for key in (
            "G1_RQ14_PRETRAINED_RUN",
            "G1_RQ15_SOURCE_RUN",
            "G1_RQ15_FIRST_STAGE_CHECKPOINT",
        )
    }
    source = source_candidates()[1]
    checkpoint = "/tmp/rq14-preflight-selected-source.pt"
    treatments = set()
    all_candidate_only = True
    try:
        for candidate in launch_candidates():
            os.environ["G1_RQ14_PRETRAINED_RUN"] = candidate.run_name
            os.environ["G1_RQ15_SOURCE_RUN"] = source.run_name
            os.environ["G1_RQ15_FIRST_STAGE_CHECKPOINT"] = checkpoint
            experiment = runpy.run_path(str(_CONFIG))["experiment"]
            treatments.add(candidate.treatment)
            all_candidate_only &= bool(
                experiment.training_method == "pretrained_finetune"
                and experiment.auxiliary_ntp_weight == 0
                and experiment.first_stage_checkpoint == Path(checkpoint)
                and experiment.query_slots_shared is candidate.query_slots_shared
                and experiment.include_history_memory
                is candidate.include_history_memory
                and experiment.embedding_learning_rate == 0.00025
                and experiment.deep_learning_rate == candidate.deep_lr
                and experiment.num_epochs == 20
                and experiment.lr_schedule.shape == "linear"
                and experiment.lr_schedule.optimizer_group_scope == "both"
                and experiment.restore_best_weights is True
            )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {
        "passed": treatments == set(TREATMENTS) - {"distinct_cls_only"} and all_candidate_only,
        "new_recipe_count": len(launch_candidates()),
        "new_treatments": sorted(treatments),
        "all_candidate_only": all_candidate_only,
    }


def write_correctness_audit(document: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _positive_list(value: object, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(_positive(item) for item in value)


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq14PretrainedAuditError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq14PretrainedAuditError(f"{path}: expected JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_correctness_audit(
        build_correctness_audit(args.logs, args.results), args.output
    )


if __name__ == "__main__":
    main()
