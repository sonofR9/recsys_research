from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from dcn.nn.sampled_softmax import InBatchSampledSoftmaxLoss
from neuralrec.utils import LOSS_DENOMINATOR

from .cross_attention_retrieval import CrossAttentionRetrievalModel
from .sequence_retrieval import SequenceRetrievalModel
from .sequence_targets import NextItemTargets


class FirstStageCheckpointError(ValueError):
    pass


def first_stage_initialization_manifest(
    path: Path,
    *,
    source_metadata: Mapping[str, object],
    history_position_count: int,
) -> dict[str, object]:
    if not path.is_file():
        raise FirstStageCheckpointError(
            f"first-stage checkpoint does not exist: {path}"
        )
    if history_position_count < 1:
        raise ValueError("history position count must be positive")
    return {
        "schema_version": 1,
        "checkpoint_path": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_metadata": dict(source_metadata),
        "history_position_count": history_position_count,
        "copied_modules": ["item_embedding", "memory_encoder", "tokenizer"],
        "newly_initialized_modules": [
            "decoder",
            "decoder_query",
            "query_projection",
            "query_slots",
        ],
    }


class AuxiliaryNtpCrossAttentionRetrievalModel(CrossAttentionRetrievalModel):
    def __init__(self, *args: Any, first_stage_query_projection: nn.Linear, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if (
            first_stage_query_projection.in_features != self.memory_encoder.out_dim
            or first_stage_query_projection.out_features
            != self.item_embedding.embedding_dim
        ):
            raise ValueError("first-stage projection must map history to item width")
        if self.query_slots is None:
            raise ValueError("auxiliary NTP requires end query slots")
        self.first_stage_query_projection = first_stage_query_projection

    def forward_training_tasks(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        tokens = self.tokenizer(batch)
        history, target_ids, target_timestamps = self._training_history(tokens)
        assert self.query_slots is not None
        encoded = self.query_slots(history)
        hidden = self.memory_encoder(
            encoded.embeddings,
            encoded.cumulative_lens,
            encoded.timestamps,
        )
        memory = self.query_slots.extract_memory(
            hidden,
            encoded,
            include_history=self.include_history_memory,
        )
        candidate_queries = self._decode_encoded_memory(
            memory.embeddings,
            memory.cumulative_lens,
            target_ids.shape[0],
        )
        candidate_output = self._candidate_output(
            candidate_queries, target_ids, target_timestamps
        )

        history_hidden = hidden[~encoded.is_query]
        projected_history = self.first_stage_query_projection(history_hidden)
        target_positions = tokens.cumulative_lens[1:] - 1
        history_positions = torch.ones(
            tokens.item_ids.shape[0], dtype=torch.bool, device=tokens.item_ids.device
        )
        history_positions[target_positions] = False
        ntp_queries = projected_history.new_zeros(
            tokens.item_ids.shape[0], projected_history.shape[1]
        )
        ntp_queries[history_positions] = projected_history
        ntp_output = {
            "query_repr": ntp_queries,
            "item_repr": self.item_embedding(tokens.item_ids),
            "item_ids": tokens.item_ids,
            "lengths": tokens.cumulative_lens.diff(),
            "is_target": tokens.is_target,
            "is_query": history_positions,
            "timestamps": tokens.timestamps,
        }
        return candidate_output, ntp_output

    def _decode_encoded_memory(
        self,
        memory: torch.Tensor,
        memory_cumulative_lens: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        query = self.decoder_query.unsqueeze(0).expand(batch_size, -1)
        query_cumulative_lens = torch.arange(
            batch_size + 1,
            dtype=memory_cumulative_lens.dtype,
            device=memory_cumulative_lens.device,
        )
        decoded = self.decoder(
            query,
            query_cumulative_lens,
            memory,
            memory_cumulative_lens,
        )
        decoded = self.query_multiplier * decoded
        return decoded if self.query_projection is None else self.query_projection(decoded)

    def _candidate_output(
        self,
        queries: torch.Tensor,
        target_ids: torch.Tensor,
        target_timestamps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = queries.shape[0]
        return {
            "query_repr": torch.stack(
                [queries, torch.zeros_like(queries)], dim=1
            ).flatten(0, 1),
            "item_repr": self.item_embedding(
                torch.stack([torch.zeros_like(target_ids), target_ids], dim=1).flatten()
            ),
            "item_ids": torch.stack(
                [torch.zeros_like(target_ids), target_ids], dim=1
            ).flatten(),
            "lengths": torch.full(
                (batch_size,), 2, dtype=torch.long, device=queries.device
            ),
            "is_target": torch.tensor(
                [False, True], dtype=torch.bool, device=queries.device
            ).repeat(batch_size),
            "is_query": torch.tensor(
                [True, False], dtype=torch.bool, device=queries.device
            ).repeat(batch_size),
            "timestamps": target_timestamps.repeat_interleave(2),
        }


class CandidateAuxiliaryNtpLoss(nn.Module):
    def __init__(
        self,
        model: AuxiliaryNtpCrossAttentionRetrievalModel,
        *,
        candidate_loss: InBatchSampledSoftmaxLoss,
        auxiliary_ntp_loss: InBatchSampledSoftmaxLoss,
        auxiliary_ntp_weight: float,
    ) -> None:
        super().__init__()
        if not math.isfinite(auxiliary_ntp_weight) or auxiliary_ntp_weight <= 0:
            raise ValueError("auxiliary NTP weight must be finite and positive")
        self.model = model
        self.candidate_loss = candidate_loss
        self.auxiliary_ntp_loss = auxiliary_ntp_loss
        self.auxiliary_ntp_weight = float(auxiliary_ntp_weight)
        self.targets = NextItemTargets()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | int]:
        candidate_output, ntp_output = self.model.forward_training_tasks(batch)
        candidate = self._task(self.candidate_loss, candidate_output)
        auxiliary = self._task(self.auxiliary_ntp_loss, ntp_output)
        candidate_count = int(candidate[LOSS_DENOMINATOR])
        auxiliary_count = int(auxiliary[LOSS_DENOMINATOR])
        return {
            "loss": candidate["loss"]
            + self.auxiliary_ntp_weight * auxiliary["loss"],
            "candidate_loss": candidate["loss"],
            "auxiliary_ntp_loss": auxiliary["loss"],
            "candidate_hit_rate": candidate["hit_rate"],
            "auxiliary_ntp_hit_rate": auxiliary["hit_rate"],
            "candidate_targets": candidate_count,
            "auxiliary_ntp_targets": auxiliary_count,
            LOSS_DENOMINATOR: candidate_count,
        }

    def accumulation_spec(
        self, batch: dict[str, Any]
    ) -> dict[str, tuple[float, int]]:
        lengths = batch["cumulative_lens"].diff()
        return {
            "candidate_loss": (1.0, int(lengths.shape[0])),
            "auxiliary_ntp_loss": (
                self.auxiliary_ntp_weight,
                int((lengths - 1).sum()),
            ),
        }

    def _task(
        self,
        loss: InBatchSampledSoftmaxLoss,
        output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | int]:
        pairs = self.targets(output)
        if pairs.query_repr.shape[0] == 0:
            zero = (output["query_repr"].sum() + output["item_repr"].sum()) * 0.0
            return {"loss": zero, "hit_rate": zero.detach(), LOSS_DENOMINATOR: 0}
        logits = loss.logits(
            pairs.query_repr,
            pairs.positive_repr,
            pairs.positive_ids,
            pairs.group_sizes,
        )
        return {
            "loss": loss.loss_from_logits(logits),
            "hit_rate": (logits.detach().argmax(dim=1) == 0).float().mean(),
            LOSS_DENOMINATOR: int(pairs.query_repr.shape[0]),
        }


def save_first_stage_checkpoint(
    model: SequenceRetrievalModel,
    path: Path,
    *,
    metadata: Mapping[str, object],
    history_position_count: int,
) -> None:
    if history_position_count < 1:
        raise ValueError("history position count must be positive")
    document = {
        "schema_version": 1,
        "metadata": dict(metadata),
        "history_position_count": history_position_count,
        "model_dim": model.sequence_model.out_dim,
        "item_embedding_dim": model.item_embedding.embedding_dim,
        "catalog_size": model.item_embedding.num_embeddings,
        "item_id_column": model.item_id_column,
        "tokenizer": model.tokenizer.state_dict(),
        "memory_encoder": model.sequence_model.state_dict(),
        "item_embedding": model.item_embedding.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(document, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_first_stage_checkpoint(
    model: CrossAttentionRetrievalModel,
    path: Path,
    *,
    expected_metadata: Mapping[str, object],
    history_position_count: int,
) -> dict[str, object]:
    if not path.is_file():
        raise FirstStageCheckpointError(f"first-stage checkpoint does not exist: {path}")
    document = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise FirstStageCheckpointError("unsupported first-stage checkpoint schema")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise FirstStageCheckpointError("first-stage checkpoint metadata is missing")
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise FirstStageCheckpointError(
                f"first-stage checkpoint {key} is {metadata.get(key)!r}, expected {expected!r}"
            )
    expected_header = {
        "history_position_count": history_position_count,
        "model_dim": model.memory_encoder.out_dim,
        "item_embedding_dim": model.item_embedding.embedding_dim,
        "catalog_size": model.item_embedding.num_embeddings,
        "item_id_column": model.item_id_column,
    }
    for key, expected in expected_header.items():
        if document.get(key) != expected:
            raise FirstStageCheckpointError(
                f"first-stage checkpoint {key} is incompatible"
            )

    tokenizer_state = _tensor_mapping(document.get("tokenizer"), "tokenizer")
    item_state = _tensor_mapping(document.get("item_embedding"), "item_embedding")
    memory_state = _tensor_mapping(document.get("memory_encoder"), "memory_encoder")
    _require_exact_shapes(model.tokenizer.state_dict(), tokenizer_state, "tokenizer")
    _require_exact_shapes(model.item_embedding.state_dict(), item_state, "item_embedding")
    adapted_memory = _adapt_memory_state(
        model.memory_encoder.state_dict(), memory_state, history_position_count
    )
    model.tokenizer.load_state_dict(tokenizer_state, strict=True)
    model.item_embedding.load_state_dict(item_state, strict=True)
    model.memory_encoder.load_state_dict(adapted_memory, strict=True)
    return first_stage_initialization_manifest(
        path,
        source_metadata=expected_metadata,
        history_position_count=history_position_count,
    )


def _tensor_mapping(value: object, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    ):
        raise FirstStageCheckpointError(f"invalid {name} state")
    return value


def _require_exact_shapes(
    target: Mapping[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    name: str,
) -> None:
    if set(target) != set(source):
        raise FirstStageCheckpointError(f"{name} state keys are incompatible")
    wrong = [key for key in target if target[key].shape != source[key].shape]
    if wrong:
        raise FirstStageCheckpointError(f"{name} state shapes are incompatible: {wrong}")


def _adapt_memory_state(
    target: Mapping[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    history_position_count: int,
) -> dict[str, torch.Tensor]:
    if set(target) != set(source):
        raise FirstStageCheckpointError("memory encoder state keys are incompatible")
    result = {}
    for key, target_tensor in target.items():
        source_tensor = source[key]
        if target_tensor.shape == source_tensor.shape:
            result[key] = source_tensor
            continue
        if not key.endswith("position_embeddings.weight"):
            raise FirstStageCheckpointError(
                f"memory encoder state shape is incompatible: {key}"
            )
        if (
            target_tensor.ndim != 2
            or source_tensor.ndim != 2
            or target_tensor.shape[1] != source_tensor.shape[1]
            or target_tensor.shape[0] < history_position_count
            or source_tensor.shape[0] < history_position_count
        ):
            raise FirstStageCheckpointError(
                f"memory encoder position state is incompatible: {key}"
            )
        adapted = target_tensor.clone()
        adapted[:history_position_count] = source_tensor[:history_position_count]
        result[key] = adapted
    return result
