from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import replace
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

from dcn.config.networks import build_causal_transformer, build_transformer_decoder
from dcn.config.settings import TRANSFORMER
from dcn.data.features import FeatureValues
from dcn.models.cross_attention_retrieval import CrossAttentionRetrievalModel
from dcn.models.cross_attention_training import (
    AuxiliaryNtpCrossAttentionRetrievalModel,
    CandidateAuxiliaryNtpLoss,
    load_first_stage_checkpoint,
    save_first_stage_checkpoint,
)
from dcn.models.history_tokens import EndQuerySlots, ItemTokenizer, TokenizedHistory
from dcn.models.sequence_retrieval import SequenceRetrievalModel
from dcn.models.sequence_targets import NextItemTargets
from dcn.nn.sampled_softmax import InBatchSampledSoftmaxLoss
from dcn.nn.types import ModuleWithDim
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import (
    RQ15_SOURCE_CHECKPOINT_NAME,
    candidate_by_run,
    initial_candidates,
    source_candidate_by_run,
    source_candidates,
    source_checkpoint_metadata,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_report import (
    collect_report_bundle,
    current_implementation_sha256,
)
from neuralrec.utils import LOSS_DENOMINATOR
from utils.global_config import config as global_config


REQUIRED_CHECKS = {
    "target_leakage",
    "attention_masks",
    "gradient_flow",
    "separate_loss_normalization_and_counts",
    "checkpoint_copy_identity",
    "config_code_and_artifact_hashes",
}

_RESEARCH_QUESTION = "RQ15 decoder-decoder training method"
_EXPERIMENT = Path(__file__).parents[1]
_TREATMENT_CONFIG = _EXPERIMENT / "configs/rq15_decoder_training_variant.py"
_SOURCE_CONFIG = _EXPERIMENT / "configs/rq15_rq8_checkpoint_variant.py"
_ARTIFACT_NAMES = {"training_metadata.json", "final_metrics.json", "sweep.log"}
_COPIED_MODULES = ["item_embedding", "memory_encoder", "tokenizer"]
_NEW_MODULES = ["decoder", "decoder_query", "query_projection", "query_slots"]
_DIM = 8
_NUM_ITEMS = 32


class Rq15AuditError(RuntimeError):
    pass


class _LinearSequence(ModuleWithDim):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(_DIM, _DIM)

    @property
    def out_dim(self) -> int:
        return _DIM

    def forward(
        self,
        embeddings: torch.Tensor,
        cumulative_lens: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.linear(embeddings)


class _MemoryMeanDecoder(ModuleWithDim):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(_DIM, _DIM)

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
        return self.linear(query + torch.stack(means))


class _MeanSquaredPairLoss(InBatchSampledSoftmaxLoss):
    def __init__(self) -> None:
        super().__init__(num_in_batch_negatives=0)

    def _log_q(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(ids, dtype=torch.float32)

    def _q_total(self) -> torch.Tensor:
        return torch.tensor(1.0)

    def logits(
        self,
        query_repr: torch.Tensor,
        positive_item_repr: torch.Tensor,
        positive_item_ids: torch.Tensor,
        group_sizes: torch.Tensor,
        negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positive_item_ids, group_sizes, negatives
        squared = (query_repr - positive_item_repr).square().mean(dim=1)
        return torch.stack([-squared, torch.zeros_like(squared)], dim=1)

    def loss_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return -logits[:, 0].mean()


def build_correctness_audit(
    logs: Path,
    results_path: Path,
    *,
    model_probe: Callable[[], dict[str, dict[str, object]]] | None = None,
    recipe_probe: Callable[[], dict[str, object]] | None = None,
    authoritative_results_probe: Callable[[], Mapping[str, object]] | None = None,
    implementation_hash: object | None = None,
) -> dict[str, object]:
    results = _load_json(results_path)
    authoritative = (
        _collect_authoritative_results(logs)
        if authoritative_results_probe is None
        else authoritative_results_probe()
    )
    _validate_resolved_results(results, authoritative)
    expected_artifacts = _result_run_artifacts(results)
    artifact_probe = _audit_run_artifacts(logs, results, expected_artifacts)
    model = run_model_correctness_probe() if model_probe is None else model_probe()
    recipes = (
        _production_recipe_probe(logs, results)
        if recipe_probe is None
        else recipe_probe()
    )
    implementation = (
        current_implementation_sha256()
        if implementation_hash is None
        else implementation_hash
    )

    target = _probe_section(model, "target_leakage")
    attention = _probe_section(model, "attention_masks")
    gradients = _probe_section(model, "gradient_flow")
    losses = _probe_section(model, "separate_loss_normalization_and_counts")
    checkpoint = _probe_section(model, "checkpoint_copy_identity")
    checks = {
        "target_leakage": _checked(
            {
                **target,
                "passed": bool(
                    _at_most(target, "target_only_candidate_query_max_delta", 1e-6)
                    and _at_most(target, "target_only_ntp_query_max_delta", 1e-6)
                    and target.get("candidate_positive_ids") == [4, 12]
                    and target.get("changed_candidate_positive_ids") == [5, 13]
                    and target.get("ntp_positive_ids") == [2, 3, 4, 11, 12]
                ),
            }
        ),
        "attention_masks": _checked(
            {
                **attention,
                "passed": bool(
                    attention.get("history_is_causal") is True
                    and _at_most(
                        attention, "later_history_to_earlier_state_max_delta", 1e-5
                    )
                    and _positive(attention.get("later_history_to_later_state_l1"))
                    and _at_most(attention, "other_user_query_max_delta", 1e-5)
                    and _at_most(
                        attention, "query_slot_to_history_max_delta", 1e-5
                    )
                    and _at_most(
                        attention, "cross_attention_other_user_max_delta", 1e-5
                    )
                ),
            }
        ),
        "gradient_flow": _checked(
            {
                **gradients,
                "passed": bool(
                    _positive(gradients.get("candidate_memory_encoder_gradient_l1"))
                    and _positive(gradients.get("candidate_decoder_gradient_l1"))
                    and _all_positive(
                        gradients.get("candidate_slot_gradient_l1"), expected=4
                    )
                    and _positive(
                        gradients.get(
                            "auxiliary_first_stage_projection_gradient_l1"
                        )
                    )
                    and _positive(
                        gradients.get("auxiliary_memory_encoder_gradient_l1")
                    )
                ),
            }
        ),
        "separate_loss_normalization_and_counts": _checked(
            {
                **losses,
                "production_metadata_identity": artifact_probe[
                    "loss_normalization_and_counts"
                ],
                "passed": bool(
                    losses.get("candidate_targets") == 2
                    and losses.get("auxiliary_ntp_targets") == 5
                    and losses.get("candidate_accumulation_denominator") == 2
                    and losses.get("auxiliary_accumulation_denominator") == 5
                    and _at_most(losses, "combined_loss_delta", 1e-6)
                    and _at_most(
                        losses, "duplicated_batch_candidate_loss_delta", 1e-6
                    )
                    and _at_most(
                        losses, "duplicated_batch_auxiliary_loss_delta", 1e-6
                    )
                    and artifact_probe["loss_normalization_and_counts"] is True
                ),
            }
        ),
        "checkpoint_copy_identity": _checked(
            {
                **checkpoint,
                "production_checkpoint_identity": artifact_probe[
                    "checkpoint_copy_identity"
                ],
                "selected_checkpoint_sha256": artifact_probe[
                    "selected_checkpoint_sha256"
                ],
                "pretrained_run_count": artifact_probe["pretrained_run_count"],
                "passed": bool(
                    all(checkpoint.get(name) is True for name in (
                        "copied_item_embedding",
                        "copied_memory_encoder",
                        "copied_tokenizer",
                        "preserved_decoder",
                        "preserved_decoder_query",
                        "preserved_query_projection",
                        "preserved_query_slots",
                    ))
                    and artifact_probe["checkpoint_copy_identity"] is True
                ),
            }
        ),
        "config_code_and_artifact_hashes": _checked(
            {
                "passed": bool(
                    artifact_probe["artifacts_match"] is True
                    and artifact_probe["run_recipe_identity"] is True
                    and recipes.get("production_recipes_match") is True
                    and isinstance(implementation, Mapping)
                    and bool(implementation)
                    and all(_is_sha256(value) for value in implementation.values())
                ),
                "artifacts_match": artifact_probe["artifacts_match"],
                "run_recipe_identity": artifact_probe["run_recipe_identity"],
                "run_count": len(expected_artifacts),
                "production_recipe_probe": recipes,
                "implementation_sha256": implementation,
                "result_binding_sha256": _canonical_sha256(
                    _results_binding(results)
                ),
            }
        ),
    }
    failed = [name for name, check in checks.items() if check["passed"] is not True]
    if failed:
        raise Rq15AuditError("correctness audit failed: " + ", ".join(failed))
    return {
        "schema_version": 1,
        "research_question": _RESEARCH_QUESTION,
        "dataset_size": "500m",
        "status": "passed",
        "checks": checks,
        "run_artifacts": expected_artifacts,
        "implementation_sha256": implementation,
        "result_binding": _results_binding(results),
    }


def validate_correctness_audit(
    audit: Mapping[str, object],
    expected_run_artifacts: Mapping[str, Mapping[str, str]],
    expected_result_binding: Mapping[str, object],
    *,
    implementation_hash: object | None = None,
) -> dict[str, object]:
    implementation = (
        current_implementation_sha256()
        if implementation_hash is None
        else implementation_hash
    )
    checks = audit.get("checks")
    valid_checks = isinstance(checks, Mapping) and set(checks) == REQUIRED_CHECKS
    if valid_checks:
        valid_checks = all(
            isinstance(checks[name], Mapping)
            and checks[name].get("passed") is True
            and _check_digest_valid(checks[name])
            for name in REQUIRED_CHECKS
        )
    if not (
        audit.get("schema_version") == 1
        and audit.get("research_question") == _RESEARCH_QUESTION
        and audit.get("dataset_size") == "500m"
        and audit.get("status") == "passed"
        and valid_checks
        and audit.get("run_artifacts") == dict(expected_run_artifacts)
        and audit.get("implementation_sha256") == implementation
        and audit.get("result_binding") == dict(expected_result_binding)
    ):
        raise Rq15AuditError("RQ15 correctness audit is incomplete, stale, or failed")
    return {
        "status": "passed",
        "schema_version": 1,
        "artifact_sha256": _canonical_sha256(audit),
    }


def run_model_correctness_probe() -> dict[str, dict[str, object]]:
    target = _target_leakage_probe()
    attention = _attention_mask_probe()
    gradients, losses = _gradient_and_loss_probe()
    checkpoint = _checkpoint_copy_probe()
    return {
        "target_leakage": target,
        "attention_masks": attention,
        "gradient_flow": gradients,
        "separate_loss_normalization_and_counts": losses,
        "checkpoint_copy_identity": checkpoint,
    }


def write_correctness_audit(document: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _target_leakage_probe() -> dict[str, object]:
    torch.manual_seed(7)
    model = _auxiliary_model().eval()
    batch = _packed_batch([[1, 2, 3, 4], [10, 11, 12]])
    changed = _packed_batch([[1, 2, 3, 5], [10, 11, 13]])
    candidate_output, ntp_output = model.forward_training_tasks(batch)
    changed_candidate, changed_ntp = model.forward_training_tasks(changed)
    candidate_pairs = NextItemTargets()(candidate_output)
    changed_candidate_pairs = NextItemTargets()(changed_candidate)
    ntp_pairs = NextItemTargets()(ntp_output)
    changed_ntp_pairs = NextItemTargets()(changed_ntp)
    return {
        "target_only_candidate_query_max_delta": _max_delta(
            candidate_pairs.query_repr, changed_candidate_pairs.query_repr
        ),
        "target_only_ntp_query_max_delta": _max_delta(
            ntp_pairs.query_repr, changed_ntp_pairs.query_repr
        ),
        "candidate_positive_ids": candidate_pairs.positive_ids.tolist(),
        "changed_candidate_positive_ids": changed_candidate_pairs.positive_ids.tolist(),
        "ntp_positive_ids": ntp_pairs.positive_ids.tolist(),
    }


def _attention_mask_probe() -> dict[str, object]:
    transformer = replace(
        TRANSFORMER,
        dim=_DIM,
        num_layers=2,
        nhead=2,
        num_kv_heads=1,
        ffn_intermediate_dim=16,
        dropout=0.0,
        input_dropout=0.0,
        ffn_dropout=0.0,
        alibi=False,
        learned_positions=None,
        input_norm=None,
        final_norm=None,
        attention_window=None,
    )
    previous = global_config.cpu_attention
    global_config.set_cpu_attention(True)
    try:
        torch.manual_seed(11)
        encoder = build_causal_transformer(transformer, max_seq_len=8).eval()
        torch.manual_seed(12)
        values = torch.randn(5, _DIM)
        cumulative_lens = torch.tensor([0, 3, 5], dtype=torch.long)
        later_changed = values.clone()
        later_changed[2] += 10
        other_user_changed = values.clone()
        other_user_changed[3:] += 10
        original_hidden = encoder(values, cumulative_lens)
        later_hidden = encoder(later_changed, cumulative_lens)
        other_hidden = encoder(other_user_changed, cumulative_lens)
        tokens = TokenizedHistory(
            embeddings=values,
            cumulative_lens=cumulative_lens,
            is_target=torch.ones(5, dtype=torch.bool),
            item_ids=torch.arange(5),
            timestamps=torch.arange(5),
            token_ids=torch.arange(5),
        )
        slots = EndQuerySlots(_DIM, num_slots=4, shared=False)
        slotted = slots(tokens)
        changed_slots = slotted.embeddings.clone()
        changed_slots[slotted.is_query] += 10
        slotted_hidden = encoder(slotted.embeddings, slotted.cumulative_lens)
        changed_slot_hidden = encoder(changed_slots, slotted.cumulative_lens)

        torch.manual_seed(13)
        decoder = build_transformer_decoder(
            replace(transformer, num_layers=1), max_seq_len=1
        ).eval()
        query = torch.randn(2, _DIM)
        query_lens = torch.tensor([0, 1, 2], dtype=torch.long)
        memory = torch.randn(7, _DIM)
        memory_lens = torch.tensor([0, 4, 7], dtype=torch.long)
        changed_memory = memory.clone()
        changed_memory[4:] += 10
        decoded = decoder(query, query_lens, memory, memory_lens)
        decoded_changed = decoder(query, query_lens, changed_memory, memory_lens)
    finally:
        global_config.set_cpu_attention(previous)
    return {
        "history_is_causal": all(layer.is_causal for layer in encoder.layers),
        "later_history_to_earlier_state_max_delta": _max_delta(
            original_hidden[:2], later_hidden[:2]
        ),
        "later_history_to_later_state_l1": float(
            (original_hidden[2] - later_hidden[2]).abs().sum()
        ),
        "other_user_query_max_delta": _max_delta(
            original_hidden[:3], other_hidden[:3]
        ),
        "query_slot_to_history_max_delta": _max_delta(
            slotted_hidden[~slotted.is_query],
            changed_slot_hidden[~slotted.is_query],
        ),
        "cross_attention_other_user_max_delta": _max_delta(
            decoded[:1], decoded_changed[:1]
        ),
    }


def _gradient_and_loss_probe() -> tuple[dict[str, object], dict[str, object]]:
    batch = _packed_batch([[1, 2, 3, 4], [10, 11, 12]])
    duplicated = _packed_batch(
        [[1, 2, 3, 4], [10, 11, 12], [1, 2, 3, 4], [10, 11, 12]]
    )
    torch.manual_seed(17)
    candidate_model = _target_model()
    candidate_pairs = NextItemTargets()(candidate_model(batch))
    candidate_pairs.query_repr.square().mean().backward()
    assert candidate_model.query_slots is not None
    slots = candidate_model.query_slots.embeddings.grad
    candidate_slot_gradients = (
        [] if slots is None else slots.abs().sum(dim=1).tolist()
    )

    torch.manual_seed(19)
    auxiliary_model = _auxiliary_model()
    criterion = _criterion(auxiliary_model)
    output = criterion(batch)
    duplicated_output = criterion(duplicated)
    output["auxiliary_ntp_loss"].backward()
    specification = criterion.accumulation_spec(batch)
    gradients = {
        "candidate_memory_encoder_gradient_l1": _gradient_l1(
            candidate_model.memory_encoder.parameters()
        ),
        "candidate_decoder_gradient_l1": _gradient_l1(
            candidate_model.decoder.parameters()
        ),
        "candidate_slot_gradient_l1": candidate_slot_gradients,
        "auxiliary_first_stage_projection_gradient_l1": _gradient_l1(
            auxiliary_model.first_stage_query_projection.parameters()
        ),
        "auxiliary_memory_encoder_gradient_l1": _gradient_l1(
            auxiliary_model.memory_encoder.parameters()
        ),
    }
    losses = {
        "candidate_targets": output["candidate_targets"],
        "auxiliary_ntp_targets": output["auxiliary_ntp_targets"],
        "candidate_accumulation_denominator": specification["candidate_loss"][1],
        "auxiliary_accumulation_denominator": specification[
            "auxiliary_ntp_loss"
        ][1],
        "combined_loss_delta": float(
            (
                output["loss"]
                - output["candidate_loss"]
                - output["auxiliary_ntp_loss"]
            ).abs()
        ),
        "duplicated_batch_candidate_loss_delta": float(
            (output["candidate_loss"] - duplicated_output["candidate_loss"]).abs()
        ),
        "duplicated_batch_auxiliary_loss_delta": float(
            (
                output["auxiliary_ntp_loss"]
                - duplicated_output["auxiliary_ntp_loss"]
            ).abs()
        ),
    }
    return gradients, losses


def _checkpoint_copy_probe() -> dict[str, object]:
    torch.manual_seed(23)
    source = _source_model()
    torch.manual_seed(29)
    target = _target_model()
    assert target.query_slots is not None
    preserved = {
        "decoder": _cloned_state(target.decoder),
        "decoder_query": target.decoder_query.detach().clone(),
        "query_projection": _cloned_state(target.query_projection),
        "query_slots": target.query_slots.embeddings.detach().clone(),
    }
    metadata = {"dataset_size": "500m", "source_recipe_run_name": "probe"}
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "first_stage.pt"
        save_first_stage_checkpoint(
            source,
            checkpoint,
            metadata=metadata,
            history_position_count=128,
        )
        load_first_stage_checkpoint(
            target,
            checkpoint,
            expected_metadata=metadata,
            history_position_count=128,
        )
    return {
        "copied_item_embedding": _states_equal(
            target.item_embedding.state_dict(), source.item_embedding.state_dict()
        ),
        "copied_memory_encoder": _states_equal(
            target.memory_encoder.state_dict(), source.sequence_model.state_dict()
        ),
        "copied_tokenizer": _states_equal(
            target.tokenizer.state_dict(), source.tokenizer.state_dict()
        ),
        "preserved_decoder": _states_equal(
            target.decoder.state_dict(), preserved["decoder"]
        ),
        "preserved_decoder_query": bool(
            torch.equal(target.decoder_query, preserved["decoder_query"])
        ),
        "preserved_query_projection": _states_equal(
            target.query_projection.state_dict(), preserved["query_projection"]
        ),
        "preserved_query_slots": bool(
            torch.equal(target.query_slots.embeddings, preserved["query_slots"])
        ),
    }


def _audit_run_artifacts(
    logs: Path,
    results: Mapping[str, object],
    expected: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    actual: dict[str, dict[str, str]] = {}
    role_by_run = _treatment_roles(results)
    selected = _selected_checkpoint(results)
    selected_run = str(selected["run_name"])
    selected_candidate = source_candidate_by_run(selected_run)
    selected_checkpoint = selected_candidate.checkpoint_path(logs).resolve()
    selected_sha256 = selected.get("checkpoint_sha256")
    if not _is_sha256(selected_sha256) or not selected_checkpoint.is_file():
        raise Rq15AuditError("selected RQ15 checkpoint identity is absent")
    if _file_sha256(selected_checkpoint) != selected_sha256:
        raise Rq15AuditError("selected RQ15 checkpoint hash differs from results")

    loss_identity = True
    checkpoint_identity = True
    recipe_identity = True
    pretrained_count = 0
    for run_name, hashes in expected.items():
        directory = logs / run_name
        actual_hashes = {}
        for name in hashes:
            path = directory / name
            if not path.is_file():
                raise Rq15AuditError(f"{run_name}: missing {name}")
            actual_hashes[name] = _file_sha256(path)
        if actual_hashes != dict(hashes):
            raise Rq15AuditError(f"{run_name}: artifact hash differs from results")
        actual[run_name] = actual_hashes

        method = role_by_run.get(run_name)
        if method is None:
            try:
                source = source_candidate_by_run(run_name)
            except ValueError:
                recipe_identity = False
                continue
            metadata = _load_json(directory / "training_metadata.json")
            recipe_identity = bool(
                recipe_identity
                and metadata.get("dataset_size") == "500m"
                and metadata.get("seed") == 42
                and metadata.get("num_epochs") == 20
                and metadata.get("embedding_learning_rate") == source.embedding_lr
                and metadata.get("deep_learning_rate") == source.deep_lr
                and metadata.get("batch_size") == 1280
            )
            continue
        metadata = _load_json(directory / "training_metadata.json")
        invariants = metadata.get("transfer_invariants")
        if not isinstance(invariants, Mapping):
            raise Rq15AuditError(f"{run_name}: transfer invariants are absent")
        candidate_targets = metadata.get("candidate_targets_per_epoch")
        examples = metadata.get("expanded_examples_per_epoch")
        ntp_targets = metadata.get("ntp_targets_per_epoch")
        expected_ntp = method == "auxiliary_ntp"
        common = bool(
            metadata.get("query_architecture") == "decoder_decoder"
            and metadata.get("query_slots_shared") is False
            and metadata.get("include_history_memory") is False
            and metadata.get("num_query_slots") == 4
            and candidate_targets == examples
            and isinstance(candidate_targets, int)
            and not isinstance(candidate_targets, bool)
            and candidate_targets > 0
            and isinstance(ntp_targets, int)
            and not isinstance(ntp_targets, bool)
            and (ntp_targets > 0) is expected_ntp
            and metadata.get("targets_per_epoch") == candidate_targets + ntp_targets
        )
        current_rq15 = run_name.startswith("g1_rq15_")
        if current_rq15:
            candidate = candidate_by_run(run_name)
            recipe_identity = bool(
                recipe_identity
                and candidate.training_method == method
                and metadata.get("dataset_size") == candidate.dataset_size
                and metadata.get("seed") == candidate.seed
                and metadata.get("num_epochs") == candidate.horizon_epochs
                and metadata.get("embedding_learning_rate")
                == candidate.embedding_lr
                and metadata.get("deep_learning_rate") == candidate.deep_lr
                and metadata.get("batch_size") == candidate.batch_size
                and metadata.get("auxiliary_ntp_weight")
                == (
                    candidate.auxiliary_ntp_weight
                    if method == "auxiliary_ntp"
                    else 0.0
                )
            )
        elif method == "scratch_candidate_only":
            recipe_identity = bool(
                recipe_identity
                and metadata.get("embedding_learning_rate") == 0.064
                and metadata.get("deep_learning_rate") == 0.0015
            )
        loss_identity = bool(
            loss_identity
            and common
            and (
                not current_rq15
                or (
                    metadata.get("training_method") == method
                    and metadata.get("loss_normalization")
                    == "candidate_and_ntp_separately_mean_normalized"
                    and invariants.get("training_method") == method
                    and invariants.get("loss_normalization")
                    == "candidate_and_ntp_separately_mean_normalized"
                    and invariants.get("candidate_targets_per_epoch")
                    == candidate_targets
                    and invariants.get("ntp_targets_per_epoch") == ntp_targets
                )
            )
        )
        initialization = metadata.get("first_stage_initialization")
        if method == "pretrained_finetune":
            pretrained_count += 1
            expected_initialization = {
                "schema_version": 1,
                "checkpoint_path": str(selected_checkpoint),
                "checkpoint_sha256": selected_sha256,
                "source_metadata": source_checkpoint_metadata(selected_candidate),
                "history_position_count": 128,
                "copied_modules": _COPIED_MODULES,
                "newly_initialized_modules": _NEW_MODULES,
            }
            checkpoint_identity = bool(
                checkpoint_identity
                and initialization == expected_initialization
                and invariants.get("first_stage_initialization")
                == expected_initialization
            )
        elif current_rq15:
            checkpoint_identity = bool(
                checkpoint_identity
                and initialization == "scratch"
                and invariants.get("first_stage_initialization") == "scratch"
            )
    return {
        "artifacts_match": actual == dict(expected),
        "run_recipe_identity": recipe_identity,
        "loss_normalization_and_counts": loss_identity,
        "checkpoint_copy_identity": checkpoint_identity and pretrained_count > 0,
        "selected_checkpoint_sha256": selected_sha256,
        "pretrained_run_count": pretrained_count,
    }


def _production_recipe_probe(
    logs: Path, results: Mapping[str, object]
) -> dict[str, object]:
    selected = _selected_checkpoint(results)
    source_run = str(selected["run_name"])
    checkpoint = source_candidate_by_run(source_run).checkpoint_path(logs).resolve()
    previous = {
        name: os.environ.get(name)
        for name in (
            "G1_RQ15_RUN",
            "G1_RQ15_SOURCE_RUN",
            "G1_RQ15_FIRST_STAGE_CHECKPOINT",
            "G1_DATASET_SIZE",
            "G1_MAX_EPOCHS",
            "G1_VARIANT",
            "G1_RQ8_RUN",
        )
    }
    records = {}
    try:
        for method in (
            "scratch_candidate_only",
            "pretrained_finetune",
            "auxiliary_ntp",
        ):
            candidate = next(
                item
                for item in initial_candidates()
                if item.training_method == method
            )
            os.environ["G1_RQ15_RUN"] = candidate.run_name
            if method == "pretrained_finetune":
                os.environ["G1_RQ15_SOURCE_RUN"] = source_run
                os.environ["G1_RQ15_FIRST_STAGE_CHECKPOINT"] = str(checkpoint)
            experiment = runpy.run_path(str(_TREATMENT_CONFIG))["experiment"]
            records[method] = bool(
                experiment.training_method == method
                and experiment.query_architecture == "decoder_decoder"
                and experiment.query_slots_shared is False
                and experiment.include_history_memory is False
                and experiment.num_query_slots == 4
                and experiment.max_seq_len == 128
                and experiment.prefix_cap == 1
                and experiment.size == "500m"
                and type(experiment.create_targets()).__name__ == "NextItemTargets"
                and experiment.auxiliary_ntp_weight
                == (1.0 if method == "auxiliary_ntp" else 0.0)
            )
        os.environ["G1_RQ15_SOURCE_RUN"] = source_run
        source_experiment = runpy.run_path(str(_SOURCE_CONFIG))["experiment"]
        source_match = bool(
            source_experiment.size == "500m"
            and source_experiment.max_seq_len == 128
            and source_experiment.checkpoint_export_history_positions == 128
            and source_experiment.checkpoint_export_filename
            == RQ15_SOURCE_CHECKPOINT_NAME
            and source_experiment.checkpoint_export_metadata
            == source_checkpoint_metadata(source_candidate_by_run(source_run))
            and type(source_experiment.create_targets()).__name__ == "NextItemTargets"
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return {
        "production_recipes_match": all(records.values()) and source_match,
        "treatment_recipes": records,
        "source_recipe": source_match,
    }


def _collect_authoritative_results(logs: Path) -> Mapping[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        missing = Path(directory) / "missing.json"
        return collect_report_bundle(
            logs,
            correctness_evidence=missing,
            explanation_evidence=missing,
        ).evidence


def _validate_resolved_results(
    results: Mapping[str, object], authoritative: Mapping[str, object]
) -> None:
    artifact_audit = results.get("artifact_audit")
    winners = results.get("surface_winners")
    if (
        results.get("schema_version") != 1
        or results.get("research_question") != _RESEARCH_QUESTION
        or results.get("dataset_size") != "500m"
        or results.get("missing_artifacts") != []
        or results.get("required_followups") != []
        or not isinstance(artifact_audit, Mapping)
        or artifact_audit.get("status") != "passed"
        or not isinstance(winners, Mapping)
        or set(winners)
        != {"scratch_candidate_only", "pretrained_finetune", "auxiliary_ntp"}
    ):
        raise Rq15AuditError("RQ15 results are not a resolved three-method surface")
    if (
        _results_binding(results) != _results_binding(authoritative)
        or _result_run_artifacts(results) != _result_run_artifacts(authoritative)
    ):
        raise Rq15AuditError("RQ15 results differ from the authoritative report replay")
    _validate_initial_manifest(results)


def _results_binding(results: Mapping[str, object]) -> dict[str, object]:
    checkpoint = results.get("checkpoint_pretraining")
    winners = results.get("surface_winners")
    if not isinstance(checkpoint, Mapping) or not isinstance(winners, Mapping):
        raise Rq15AuditError("RQ15 result selection binding is absent")
    winner_names = {}
    for method in (
        "scratch_candidate_only",
        "pretrained_finetune",
        "auxiliary_ntp",
    ):
        winner = winners.get(method)
        run_name = winner.get("run_name") if isinstance(winner, Mapping) else None
        if not isinstance(run_name, str):
            raise Rq15AuditError(f"RQ15 {method} winner identity is absent")
        winner_names[method] = run_name
    checkpoint_run_name = checkpoint.get("run_name")
    if not isinstance(checkpoint_run_name, str):
        raise Rq15AuditError("RQ15 checkpoint winner identity is absent")
    return {
        "missing_artifacts": results.get("missing_artifacts"),
        "required_followups": results.get("required_followups"),
        "checkpoint_pretraining_run_name": checkpoint_run_name,
        "surface_winner_run_names": winner_names,
    }


def _validate_initial_manifest(results: Mapping[str, object]) -> None:
    roles = _treatment_roles(results)
    expected_by_method = {
        method: {
            candidate.run_name
            for candidate in initial_candidates()
            if candidate.training_method == method
        }
        for method in (
            "scratch_candidate_only",
            "pretrained_finetune",
            "auxiliary_ntp",
        )
    }
    actual_by_method = {
        method: {run_name for run_name, role in roles.items() if role == method}
        for method in expected_by_method
    }
    scratch_center = next(
        candidate.run_name
        for candidate in initial_candidates()
        if candidate.training_method == "scratch_candidate_only"
        and candidate.embedding_lr == 0.064
        and candidate.deep_lr == 0.0015
    )
    missing = {
        method: expected - actual_by_method[method]
        for method, expected in expected_by_method.items()
    }
    if (
        missing["scratch_candidate_only"] == {scratch_center}
        and any(
            not run_name.startswith("g1_rq15_")
            for run_name in actual_by_method["scratch_candidate_only"]
        )
    ):
        missing["scratch_candidate_only"] = set()
    source_surface = results.get("checkpoint_pretraining_surface")
    source_names = {
        record.get("run_name")
        for record in source_surface
        if isinstance(record, Mapping)
    } if isinstance(source_surface, list) else set()
    expected_sources = {candidate.run_name for candidate in source_candidates()}
    if any(missing.values()) or not expected_sources.issubset(source_names):
        raise Rq15AuditError("RQ15 initial or checkpoint surface is incomplete")


def _result_run_artifacts(
    results: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    artifact_audit = results["artifact_audit"]
    assert isinstance(artifact_audit, Mapping)
    raw = artifact_audit.get("run_artifacts")
    if not isinstance(raw, Mapping) or not raw:
        raise Rq15AuditError("RQ15 result artifact audit is empty")
    expected: dict[str, dict[str, str]] = {}
    for run_name, hashes in raw.items():
        if (
            not isinstance(run_name, str)
            or not isinstance(hashes, Mapping)
            or not _ARTIFACT_NAMES.issubset(hashes)
            or set(hashes) - (_ARTIFACT_NAMES | {RQ15_SOURCE_CHECKPOINT_NAME})
            or not all(_is_sha256(value) for value in hashes.values())
        ):
            raise Rq15AuditError("RQ15 result artifact identity is invalid")
        expected[run_name] = dict(hashes)

    records: dict[str, dict[str, str]] = {}
    source_surface = results.get("checkpoint_pretraining_surface")
    treatments = results.get("treatments")
    if not isinstance(source_surface, list) or not isinstance(treatments, Mapping):
        raise Rq15AuditError("RQ15 result run ledger is incomplete")
    ledgers: list[object] = list(source_surface)
    for method in (
        "scratch_candidate_only",
        "pretrained_finetune",
        "auxiliary_ntp",
    ):
        treatment = treatments.get(method)
        artifacts = treatment.get("artifacts") if isinstance(treatment, Mapping) else None
        if not isinstance(artifacts, list) or not artifacts:
            raise Rq15AuditError(f"RQ15 {method} artifact ledger is empty")
        ledgers.extend(artifacts)
    for record in ledgers:
        if not isinstance(record, Mapping):
            raise Rq15AuditError("RQ15 artifact record is invalid")
        run_name = record.get("run_name")
        hashes = record.get("artifact_sha256")
        if not isinstance(run_name, str) or not isinstance(hashes, Mapping):
            raise Rq15AuditError("RQ15 artifact record identity is invalid")
        normalized = dict(hashes)
        if run_name in records and records[run_name] != normalized:
            raise Rq15AuditError("RQ15 duplicated artifact record disagrees")
        records[run_name] = normalized
    if records != expected:
        raise Rq15AuditError("RQ15 result ledgers disagree with artifact audit")
    return expected


def _treatment_roles(results: Mapping[str, object]) -> dict[str, str]:
    treatments = results.get("treatments")
    assert isinstance(treatments, Mapping)
    result = {}
    for method in (
        "scratch_candidate_only",
        "pretrained_finetune",
        "auxiliary_ntp",
    ):
        record = treatments[method]
        assert isinstance(record, Mapping)
        artifacts = record["artifacts"]
        assert isinstance(artifacts, list)
        for artifact in artifacts:
            assert isinstance(artifact, Mapping)
            run_name = artifact.get("run_name")
            if not isinstance(run_name, str):
                raise Rq15AuditError("RQ15 treatment run name is invalid")
            existing = result.get(run_name)
            if existing is not None and existing != method:
                raise Rq15AuditError("RQ15 run appears under two methods")
            result[run_name] = method
    return result


def _selected_checkpoint(results: Mapping[str, object]) -> Mapping[str, object]:
    selected = results.get("checkpoint_pretraining")
    if not isinstance(selected, Mapping) or not isinstance(
        selected.get("run_name"), str
    ):
        raise Rq15AuditError("RQ15 selected checkpoint record is absent")
    return selected


def _auxiliary_model() -> AuxiliaryNtpCrossAttentionRetrievalModel:
    item_embedding = nn.Embedding(_NUM_ITEMS, _DIM)
    return AuxiliaryNtpCrossAttentionRetrievalModel(
        tokenizer=ItemTokenizer(item_embedding, "compact_item_id"),
        memory_encoder=_LinearSequence(),
        decoder=_MemoryMeanDecoder(),
        item_embedding=item_embedding,
        item_id_column="compact_item_id",
        query_projection=nn.Linear(_DIM, _DIM, bias=False),
        query_slots=EndQuerySlots(_DIM, num_slots=4, shared=False),
        include_history_memory=False,
        first_stage_query_projection=nn.Linear(_DIM, _DIM, bias=False),
    )


def _target_model() -> CrossAttentionRetrievalModel:
    item_embedding = nn.Embedding(_NUM_ITEMS, _DIM)
    return CrossAttentionRetrievalModel(
        tokenizer=ItemTokenizer(item_embedding, "compact_item_id"),
        memory_encoder=_LinearSequence(),
        decoder=_MemoryMeanDecoder(),
        item_embedding=item_embedding,
        item_id_column="compact_item_id",
        query_projection=nn.Linear(_DIM, _DIM, bias=False),
        query_slots=EndQuerySlots(_DIM, num_slots=4, shared=False),
        include_history_memory=False,
    )


def _source_model() -> SequenceRetrievalModel:
    item_embedding = nn.Embedding(_NUM_ITEMS, _DIM)
    return SequenceRetrievalModel(
        tokenizer=ItemTokenizer(item_embedding, "compact_item_id"),
        sequence_model=_LinearSequence(),
        item_embedding=item_embedding,
        item_id_column="compact_item_id",
        query_projection=nn.Linear(_DIM, _DIM, bias=False),
    )


def _criterion(
    model: AuxiliaryNtpCrossAttentionRetrievalModel,
) -> CandidateAuxiliaryNtpLoss:
    return CandidateAuxiliaryNtpLoss(
        model,
        candidate_loss=_MeanSquaredPairLoss(),
        auxiliary_ntp_loss=_MeanSquaredPairLoss(),
        auxiliary_ntp_weight=1.0,
    )


def _packed_batch(sequences: list[list[int]]) -> dict[str, object]:
    item_ids = torch.tensor(
        [item for sequence in sequences for item in sequence], dtype=torch.long
    )
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    cumulative_lens = torch.cat(
        [torch.zeros(1, dtype=torch.long), lengths.cumsum(0)]
    )
    return {
        "int_columns": {
            "compact_item_id": FeatureValues(
                item_ids,
                torch.arange(item_ids.numel() + 1, dtype=torch.long),
            )
        },
        "float_columns": {},
        "timestamp": torch.arange(item_ids.numel(), dtype=torch.long),
        "cumulative_lens": cumulative_lens,
    }


def _checked(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "artifact_sha256": _canonical_sha256(payload)}


def _check_digest_valid(check: Mapping[str, object]) -> bool:
    digest = check.get("artifact_sha256")
    payload = {key: value for key, value in check.items() if key != "artifact_sha256"}
    return digest == _canonical_sha256(payload)


def _probe_section(
    probe: Mapping[str, object], name: str
) -> dict[str, object]:
    section = probe.get(name)
    if not isinstance(section, Mapping):
        raise Rq15AuditError(f"RQ15 model probe lacks {name}")
    return dict(section)


def _at_most(mapping: Mapping[str, object], key: str, limit: float) -> bool:
    value = mapping.get(key)
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= float(value) <= limit
    )


def _positive(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _all_positive(value: object, *, expected: int) -> bool:
    return isinstance(value, list) and len(value) == expected and all(
        _positive(item) for item in value
    )


def _max_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max())


def _gradient_l1(parameters: Any) -> float:
    return float(
        sum(
            parameter.grad.abs().sum()
            for parameter in parameters
            if parameter.grad is not None
        )
    )


def _cloned_state(module: nn.Module | None) -> dict[str, torch.Tensor]:
    if module is None:
        return {}
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def _states_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _is_sha256(value: object) -> bool:
    return bool(
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
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq15AuditError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise Rq15AuditError(f"{path}: expected a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the saved RQ15 correctness audit")
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_correctness_audit(
        build_correctness_audit(arguments.logs, arguments.results),
        arguments.output,
    )


if __name__ == "__main__":
    main()
