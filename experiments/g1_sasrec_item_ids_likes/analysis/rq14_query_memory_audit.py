from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import tempfile
from typing import Any

import torch
from torch import nn

from dcn.data.features import FeatureValues
from dcn.models.cross_attention_retrieval import CrossAttentionRetrievalModel
from dcn.models.history_tokens import EndQuerySlots, ItemTokenizer, TokenizedHistory
from dcn.models.sequence_targets import NextItemTargets
from dcn.nn.types import ModuleWithDim
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import (
    rq14_initial_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_query_memory_report import (
    _validation_best_epoch,
    current_implementation_sha256,
)


_RESEARCH_QUESTION = "RQ14 decoder-decoder query memory"
_TREATMENTS = {
    "shared_cls_only",
    "distinct_cls_only",
    "shared_history",
    "distinct_history",
}
_REQUIRED_CHECKS = {
    "artifact_and_recipe_integrity",
    "query_slot_identity_and_order",
    "memory_content_and_lengths",
    "target_exclusion_and_candidate_only_loss",
    "gradient_flow_to_every_slot_and_history",
    "learning_curves_and_lr_boundaries",
}
_CONFIG = Path("experiments/g1_sasrec_item_ids_likes/configs/rq13_rq14_query_variant.py")
_DIM = 8


class Rq14AuditError(RuntimeError):
    pass


class _IdentityMemoryEncoder(ModuleWithDim):
    @property
    def out_dim(self) -> int:
        return _DIM

    def forward(
        self,
        embeddings: torch.Tensor,
        cumulative_lens: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return embeddings


class _MemoryMeanDecoder(ModuleWithDim):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.ones(_DIM))

    @property
    def out_dim(self) -> int:
        return _DIM

    def forward(
        self,
        query: torch.Tensor,
        query_cumulative_lens: torch.Tensor,
        memory: torch.Tensor,
        memory_cumulative_lens: torch.Tensor,
    ) -> torch.Tensor:
        means = [
            memory[start:end].mean(dim=0)
            for start, end in zip(
                memory_cumulative_lens[:-1], memory_cumulative_lens[1:]
            )
        ]
        return query + torch.stack(means) + self.offset


def build_correctness_audit(
    logs: Path,
    results_path: Path,
    *,
    model_probe: Callable[[], dict[str, object]] | None = None,
    implementation_hash: object | None = None,
) -> dict[str, object]:
    results = _load_json(results_path)
    _validate_resolved_results(results)
    run_artifacts = _result_run_artifacts(results)
    actual_artifacts, production_candidate_only = _actual_run_artifacts(
        logs, run_artifacts
    )
    probe = run_query_memory_model_probe() if model_probe is None else model_probe()
    implementation = (
        current_implementation_sha256()
        if implementation_hash is None
        else implementation_hash
    )
    expected_initial = {candidate.run_name for candidate in rq14_initial_candidates()}
    curves = _learning_curve_check(results, expected_initial)
    checks = {
        "artifact_and_recipe_integrity": {
            "passed": bool(
                actual_artifacts == run_artifacts
                and probe.get("production_recipes_match") is True
            ),
            "run_count": len(run_artifacts),
            "initial_run_count": len(expected_initial.intersection(run_artifacts)),
            "production_recipes_match": probe.get("production_recipes_match"),
        },
        "query_slot_identity_and_order": {
            "passed": bool(
                probe.get("shared_parameter_rows") == 1
                and probe.get("distinct_parameter_rows") == 4
                and probe.get("slot_order_preserved") is True
            ),
            "shared_parameter_rows": probe.get("shared_parameter_rows"),
            "distinct_parameter_rows": probe.get("distinct_parameter_rows"),
            "slot_order_preserved": probe.get("slot_order_preserved"),
        },
        "memory_content_and_lengths": {
            "passed": bool(
                probe.get("cls_only_memory_lengths") == [4, 4]
                and probe.get("history_memory_lengths") == [6, 5]
                and probe.get("history_precedes_slots") is True
            ),
            "cls_only_memory_lengths": probe.get("cls_only_memory_lengths"),
            "history_memory_lengths": probe.get("history_memory_lengths"),
            "history_precedes_slots": probe.get("history_precedes_slots"),
        },
        "target_exclusion_and_candidate_only_loss": {
            "passed": bool(
                _finite_nonnegative(probe.get("target_only_query_max_delta"))
                and float(probe["target_only_query_max_delta"]) <= 1e-6
                and probe.get("candidate_targets") == 2
                and probe.get("candidate_targets_per_example") == 1
                and probe.get("positive_ids") == [3, 11]
                and probe.get("changed_positive_ids") == [4, 12]
                and probe.get("production_target_class") == "NextItemTargets"
                and production_candidate_only
            ),
            "target_only_query_max_delta": probe.get("target_only_query_max_delta"),
            "candidate_targets": probe.get("candidate_targets"),
            "candidate_targets_per_example": probe.get(
                "candidate_targets_per_example"
            ),
            "positive_ids": probe.get("positive_ids"),
            "changed_positive_ids": probe.get("changed_positive_ids"),
            "production_target_class": probe.get("production_target_class"),
            "production_candidate_only_metadata": production_candidate_only,
        },
        "gradient_flow_to_every_slot_and_history": {
            "passed": bool(
                _all_positive(probe.get("distinct_slot_gradient_l1"), expected=4)
                and _all_positive(probe.get("shared_slot_gradient_l1"), expected=1)
                and _positive(probe.get("history_embedding_gradient_l1"))
                and _positive(probe.get("decoder_gradient_l1"))
            ),
            "distinct_slot_gradient_l1": probe.get("distinct_slot_gradient_l1"),
            "shared_slot_gradient_l1": probe.get("shared_slot_gradient_l1"),
            "history_embedding_gradient_l1": probe.get(
                "history_embedding_gradient_l1"
            ),
            "decoder_gradient_l1": probe.get("decoder_gradient_l1"),
        },
        "learning_curves_and_lr_boundaries": curves,
    }
    failed = [name for name, check in checks.items() if check["passed"] is not True]
    if failed:
        raise Rq14AuditError("correctness audit failed: " + ", ".join(failed))
    return {
        "schema_version": 1,
        "research_question": _RESEARCH_QUESTION,
        "dataset_size": "500m",
        "status": "passed",
        "checks": checks,
        "run_artifacts": run_artifacts,
        "implementation_sha256": implementation,
    }


def validate_correctness_audit(
    audit: Mapping[str, object],
    expected_run_artifacts: Mapping[str, Mapping[str, str]],
    *,
    implementation_hash: object | None = None,
) -> dict[str, object]:
    checks = audit.get("checks")
    implementation = (
        current_implementation_sha256()
        if implementation_hash is None
        else implementation_hash
    )
    if (
        audit.get("schema_version") != 1
        or audit.get("research_question") != _RESEARCH_QUESTION
        or audit.get("dataset_size") != "500m"
        or audit.get("status") != "passed"
        or not isinstance(checks, Mapping)
        or set(checks) != _REQUIRED_CHECKS
        or any(
            not isinstance(checks[name], Mapping)
            or checks[name].get("passed") is not True
            for name in _REQUIRED_CHECKS
        )
        or audit.get("run_artifacts") != dict(expected_run_artifacts)
        or audit.get("implementation_sha256") != implementation
    ):
        raise Rq14AuditError("RQ14 correctness audit is incomplete, stale, or failed")
    return {
        "status": "passed",
        "schema_version": 1,
        "artifact_sha256": _canonical_sha256(audit),
    }


def run_query_memory_model_probe() -> dict[str, object]:
    production = _production_recipe_probe()
    shared_slots = EndQuerySlots(_DIM, num_slots=4, shared=True)
    distinct_slots = EndQuerySlots(_DIM, num_slots=4, shared=False)
    tokens = _tokens([[1, 2], [10]])
    slotted = distinct_slots(tokens)
    hidden = torch.arange(slotted.embeddings.numel(), dtype=torch.float32).reshape_as(
        slotted.embeddings
    )
    cls_memory = distinct_slots.extract_memory(
        hidden, slotted, include_history=False
    )
    history_memory = distinct_slots.extract_memory(
        hidden, slotted, include_history=True
    )
    history_precedes_slots = bool(
        torch.equal(history_memory.embeddings, hidden)
        and history_memory.is_query.tolist()
        == [False, False, True, True, True, True, False, True, True, True, True]
    )

    shared_model = _model(shared_slots, include_history=False)
    distinct_model = _model(distinct_slots, include_history=True)
    batch = _packed_batch([[1, 2, 3], [10, 11]])
    changed_target = _packed_batch([[1, 2, 4], [10, 12]])
    distinct_output = distinct_model(batch)
    changed_output = distinct_model(changed_target)
    pairs = NextItemTargets()(distinct_output)
    changed_pairs = NextItemTargets()(changed_output)
    target_delta = float(
        (pairs.query_repr - changed_pairs.query_repr).abs().max()
    )
    pairs.query_repr.square().mean().backward()
    distinct_gradients = distinct_slots.embeddings.grad
    if distinct_gradients is None:
        distinct_gradient_l1: list[float] = []
    else:
        distinct_gradient_l1 = distinct_gradients.abs().sum(dim=1).tolist()
    history_gradient = float(
        distinct_model.item_embedding.weight.grad[[1, 2, 10]].abs().sum()
    )
    decoder_gradient = _gradient_l1(distinct_model.decoder.parameters())

    shared_pairs = NextItemTargets()(shared_model(batch))
    shared_pairs.query_repr.square().mean().backward()
    shared_gradients = shared_slots.embeddings.grad
    shared_gradient_l1 = (
        [] if shared_gradients is None else shared_gradients.abs().sum(dim=1).tolist()
    )
    return {
        "production_recipes_match": production,
        "shared_parameter_rows": shared_slots.embeddings.shape[0],
        "distinct_parameter_rows": distinct_slots.embeddings.shape[0],
        "slot_order_preserved": cls_memory.embeddings.tolist()
        == hidden[slotted.is_query].tolist(),
        "cls_only_memory_lengths": cls_memory.cumulative_lens.diff().tolist(),
        "history_memory_lengths": history_memory.cumulative_lens.diff().tolist(),
        "history_precedes_slots": history_precedes_slots,
        "target_only_query_max_delta": target_delta,
        "candidate_targets": int(pairs.positive_ids.numel()),
        "candidate_targets_per_example": int(
            pairs.positive_ids.numel() / pairs.group_sizes.numel()
        ),
        "positive_ids": pairs.positive_ids.tolist(),
        "changed_positive_ids": changed_pairs.positive_ids.tolist(),
        "production_target_class": "NextItemTargets",
        "distinct_slot_gradient_l1": distinct_gradient_l1,
        "shared_slot_gradient_l1": shared_gradient_l1,
        "history_embedding_gradient_l1": history_gradient,
        "decoder_gradient_l1": decoder_gradient,
    }


def write_correctness_audit(document: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _production_recipe_probe() -> bool:
    previous = os.environ.get("G1_QUERY_RUN")
    try:
        for candidate in rq14_initial_candidates():
            if candidate.deep_lr != 0.012:
                continue
            os.environ["G1_QUERY_RUN"] = candidate.run_name
            experiment = runpy.run_path(str(_CONFIG))["experiment"]
            if (
                experiment.query_architecture != "decoder_decoder"
                or experiment.query_slots_shared
                is not candidate.treatment.startswith("shared_")
                or experiment.include_history_memory
                is not candidate.treatment.endswith("_history")
                or experiment.num_query_slots != 4
                or experiment.size != "500m"
                or experiment.seed != 42
                or experiment.num_epochs != 20
                or experiment.max_seq_len != 128
                or experiment.window != "bounded_prefix"
                or experiment.prefix_length_rule != "truncated"
                or experiment.prefix_cap != 1
                or experiment.transformer.dim != 64
                or experiment.transformer.num_layers != 2
                or experiment.transformer.attention_window != 54
                or experiment.retrieval_decoder.num_layers != 1
                or experiment.retrieval_decoder.ffn != "swiglu"
                or experiment.retrieval_decoder.ffn_intermediate_dim != 128
                or experiment.embedding_learning_rate != 0.064
                or experiment.deep_learning_rate != 0.012
                or experiment.dataloader.effective_batch_size != 1280
                or experiment.lr_schedule.shape != "linear"
                or experiment.lr_schedule_horizon_epochs != 20
                or type(experiment.create_targets()).__name__ != "NextItemTargets"
            ):
                return False
    finally:
        if previous is None:
            os.environ.pop("G1_QUERY_RUN", None)
        else:
            os.environ["G1_QUERY_RUN"] = previous
    return True


def _model(
    query_slots: EndQuerySlots, *, include_history: bool
) -> CrossAttentionRetrievalModel:
    item_embedding = nn.Embedding(64, _DIM)
    return CrossAttentionRetrievalModel(
        tokenizer=ItemTokenizer(item_embedding, "compact_item_id"),
        memory_encoder=_IdentityMemoryEncoder(),
        decoder=_MemoryMeanDecoder(),
        item_embedding=item_embedding,
        item_id_column="compact_item_id",
        query_slots=query_slots,
        include_history_memory=include_history,
    )


def _tokens(sequences: list[list[int]]) -> TokenizedHistory:
    values = torch.tensor([item for sequence in sequences for item in sequence])
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    cumulative = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
    return TokenizedHistory(
        embeddings=values[:, None].float().expand(-1, _DIM).clone(),
        cumulative_lens=cumulative,
        is_target=torch.ones(values.shape[0], dtype=torch.bool),
        item_ids=values,
        timestamps=torch.arange(values.shape[0]),
        token_ids=values,
    )


def _packed_batch(sequences: list[list[int]]) -> dict[str, object]:
    item_ids = torch.tensor([item for sequence in sequences for item in sequence])
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    cumulative = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
    return {
        "int_columns": {
            "compact_item_id": FeatureValues(
                item_ids,
                torch.arange(item_ids.numel() + 1, dtype=torch.long),
            )
        },
        "float_columns": {},
        "timestamp": torch.arange(item_ids.numel(), dtype=torch.long),
        "cumulative_lens": cumulative,
    }


def _validate_resolved_results(results: Mapping[str, object]) -> None:
    selected = results.get("selected")
    if (
        results.get("research_question") != _RESEARCH_QUESTION
        or results.get("dataset_size") != "500m"
        or results.get("missing_initial_artifacts") != []
        or results.get("required_boundary_followups") != []
        or results.get("required_followups") != []
        or not isinstance(selected, Mapping)
        or set(selected) != _TREATMENTS
    ):
        raise Rq14AuditError("RQ14 results are not a resolved four-treatment surface")


def _result_run_artifacts(
    results: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    treatments = results.get("treatments")
    if not isinstance(treatments, Mapping) or set(treatments) != _TREATMENTS:
        raise Rq14AuditError("RQ14 treatment evidence is incomplete")
    run_artifacts = {}
    for treatment in _TREATMENTS:
        treatment_record = treatments[treatment]
        artifacts = (
            treatment_record.get("artifacts")
            if isinstance(treatment_record, Mapping)
            else None
        )
        if not isinstance(artifacts, list) or len(artifacts) < 3:
            raise Rq14AuditError(f"{treatment}: fewer than three LR artifacts")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise Rq14AuditError(f"{treatment}: invalid artifact record")
            run_name = artifact.get("run_name")
            hashes = artifact.get("artifact_sha256")
            if (
                not isinstance(run_name, str)
                or not isinstance(hashes, Mapping)
                or set(hashes)
                != {"training_metadata.json", "final_metrics.json", "sweep.log"}
                or not all(isinstance(value, str) for value in hashes.values())
                or run_name in run_artifacts
            ):
                raise Rq14AuditError(f"{treatment}: invalid artifact identity")
            run_artifacts[run_name] = dict(hashes)
    expected_initial = {candidate.run_name for candidate in rq14_initial_candidates()}
    if not expected_initial.issubset(run_artifacts):
        raise Rq14AuditError("RQ14 audit lacks the exact 12-run initial grid")
    return run_artifacts


def _actual_run_artifacts(
    logs: Path, expected: Mapping[str, Mapping[str, str]]
) -> tuple[dict[str, dict[str, str]], bool]:
    actual = {}
    candidate_only = True
    for run_name in expected:
        directory = logs / run_name
        hashes = {}
        for name in ("training_metadata.json", "final_metrics.json", "sweep.log"):
            path = directory / name
            if not path.is_file():
                raise Rq14AuditError(f"{run_name}: missing {name}")
            hashes[name] = _file_sha256(path)
        if hashes != expected[run_name]:
            raise Rq14AuditError(f"{run_name}: artifact hash differs from report")
        metadata = _load_json(directory / "training_metadata.json")
        invariants = metadata.get("transfer_invariants")
        candidate_only = bool(
            candidate_only
            and metadata.get("query_architecture") == "decoder_decoder"
            and isinstance(invariants, Mapping)
            and invariants.get("query_architecture") == "decoder_decoder"
            and metadata.get("ntp_targets_per_epoch") == 0
            and invariants.get("ntp_targets_per_epoch") == 0
            and metadata.get("candidate_targets_per_epoch")
            == metadata.get("expanded_examples_per_epoch")
            and invariants.get("candidate_targets_per_epoch")
            == invariants.get("expanded_examples_per_epoch")
            == metadata.get("candidate_targets_per_epoch")
        )
        actual[run_name] = hashes
    return actual, candidate_only


def _learning_curve_check(
    results: Mapping[str, object], expected_initial: set[str]
) -> dict[str, object]:
    treatments = results["treatments"]
    curves = {}
    stages = {}
    for treatment in _TREATMENTS:
        for artifact in treatments[treatment]["artifacts"]:
            run_name = artifact["run_name"]
            curve = artifact.get("validation_curve")
            if (
                not isinstance(curve, list)
                or len(curve) != 20
                or [point.get("epoch") for point in curve if isinstance(point, Mapping)]
                != list(range(1, 21))
                or any(
                    not isinstance(point, Mapping)
                    or not all(
                        _unit_metric(point.get(metric))
                        for metric in ("recall@100", "ndcg@100")
                    )
                    for point in curve
                )
            ):
                raise Rq14AuditError(f"{run_name}: incomplete 20-epoch curve")
            curves[run_name] = curve
            stages[run_name] = artifact.get("stage")
            best_epoch = artifact.get("best_epoch")
            expected_epoch = _validation_best_epoch(
                (
                    point["epoch"],
                    point["recall@100"],
                    point["ndcg@100"],
                )
                for point in curve
            )
            if best_epoch != expected_epoch:
                raise Rq14AuditError(
                    f"{run_name}: best epoch is not the validation-curve winner"
                )
    return {
        "passed": expected_initial.issubset(curves),
        "complete_horizon_curves": len(curves),
        "initial_grid_complete": expected_initial.issubset(curves),
        "artifact_stages": stages,
        "curves": curves,
    }


def _all_positive(value: object, *, expected: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected
        and all(_positive(item) for item in value)
    )


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _unit_metric(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _gradient_l1(parameters: Any) -> float:
    return float(
        sum(
            parameter.grad.abs().sum()
            for parameter in parameters
            if parameter.grad is not None
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq14AuditError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq14AuditError(f"{path}: expected a JSON object")
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
