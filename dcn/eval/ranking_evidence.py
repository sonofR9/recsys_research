from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from utils.locks import hold


_SCHEMA_VERSION = 1


@dataclass(frozen=True, eq=False)
class RankingEvidence:
    user_ids: torch.Tensor
    history_item_ids: torch.Tensor
    history_offsets: torch.Tensor
    relevant_item_ids: torch.Tensor
    relevance_offsets: torch.Tensor
    relevant_train_frequencies: torch.Tensor
    relevant_ranks: torch.Tensor
    max_k: int

    def __post_init__(self) -> None:
        tensors = (
            self.user_ids,
            self.history_item_ids,
            self.history_offsets,
            self.relevant_item_ids,
            self.relevance_offsets,
            self.relevant_train_frequencies,
            self.relevant_ranks,
        )
        if any(tensor.ndim != 1 for tensor in tensors):
            raise ValueError("ranking evidence tensors must be one-dimensional")
        if self.max_k < 1:
            raise ValueError("ranking evidence max_k must be positive")
        if self.history_offsets.shape[0] != self.user_ids.shape[0] + 1:
            raise ValueError("history offsets must delimit every user")
        if self.relevance_offsets.shape[0] != self.user_ids.shape[0] + 1:
            raise ValueError("relevance offsets must delimit every user")
        _validate_offsets(
            self.history_offsets, self.history_item_ids.shape[0], "history"
        )
        _validate_offsets(
            self.relevance_offsets, self.relevant_item_ids.shape[0], "relevance"
        )
        relevant_count = self.relevant_item_ids.shape[0]
        if self.relevant_train_frequencies.shape[0] != relevant_count:
            raise ValueError("target frequencies must align with relevant items")
        if self.relevant_ranks.shape[0] != relevant_count:
            raise ValueError("ranks must align with relevant items")
        if bool((self.relevant_train_frequencies < 0).any()):
            raise ValueError("relevant item train frequencies must be nonnegative")
        if bool(((self.relevant_ranks < 0) | (self.relevant_ranks > self.max_k)).any()):
            raise ValueError("relevant ranks must be zero or within max_k")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RankingEvidence) or self.max_k != other.max_k:
            return False
        return all(
            torch.equal(getattr(self, name), getattr(other, name))
            for name in (
                "user_ids",
                "history_item_ids",
                "history_offsets",
                "relevant_item_ids",
                "relevance_offsets",
                "relevant_train_frequencies",
                "relevant_ranks",
            )
        )


def _validate_offsets(offsets: torch.Tensor, size: int, name: str) -> None:
    if offsets.shape[0] < 1 or int(offsets[0]) != 0 or int(offsets[-1]) != size:
        raise ValueError(f"{name} offsets do not delimit their values")
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError(f"{name} offsets must be nondecreasing")


def _cpu_int64(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.int64).contiguous()


def _context_payload(evidence: RankingEvidence) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "user_ids": _cpu_int64(evidence.user_ids),
        "history_item_ids": _cpu_int64(evidence.history_item_ids),
        "history_offsets": _cpu_int64(evidence.history_offsets),
        "relevant_item_ids": _cpu_int64(evidence.relevant_item_ids),
        "relevance_offsets": _cpu_int64(evidence.relevance_offsets),
        "relevant_train_frequencies": _cpu_int64(evidence.relevant_train_frequencies),
    }


def _ranking_payload(
    evidence: RankingEvidence, context_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "context_sha256": context_sha256,
        "max_k": evidence.max_k,
        "relevant_ranks": _cpu_int64(evidence.relevant_ranks),
    }


def _payload_digest(payload: dict[str, object]) -> str:
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


def _load_payload(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"cannot read ranking evidence: {path}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported ranking evidence schema: {path}")
    return payload


def _write_immutable(path: Path, payload: dict[str, object], kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with hold(path.with_suffix(path.suffix + ".lock"), f"ranking {kind}"):
        if path.exists():
            existing = _load_payload(path)
            if _payload_digest(existing) != _payload_digest(payload):
                raise RuntimeError(f"ranking {kind} changed: {path}")
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)


def write_ranking_evidence(
    evidence: RankingEvidence, *, context_path: Path, ranking_path: Path
) -> str:
    context = _context_payload(evidence)
    context_sha256 = _payload_digest(context)
    _write_immutable(context_path, context, "context")
    _write_immutable(
        ranking_path,
        _ranking_payload(evidence, context_sha256),
        "ranking",
    )
    return context_sha256


def load_ranking_evidence(context_path: Path, ranking_path: Path) -> RankingEvidence:
    context = _load_payload(context_path)
    ranking = _load_payload(ranking_path)
    context_sha256 = _payload_digest(context)
    if ranking.get("context_sha256") != context_sha256:
        raise ValueError("ranking evidence references a different evaluation context")
    required_context = (
        "user_ids",
        "history_item_ids",
        "history_offsets",
        "relevant_item_ids",
        "relevance_offsets",
        "relevant_train_frequencies",
    )
    if any(
        not isinstance(context.get(name), torch.Tensor) for name in required_context
    ):
        raise ValueError("ranking evaluation context is incomplete")
    if not isinstance(ranking.get("relevant_ranks"), torch.Tensor):
        raise ValueError("ranking evidence is incomplete")
    max_k = ranking.get("max_k")
    if not isinstance(max_k, int) or isinstance(max_k, bool):
        raise ValueError("ranking evidence max_k is invalid")
    return RankingEvidence(
        **{name: context[name] for name in required_context},
        relevant_ranks=ranking["relevant_ranks"],
        max_k=max_k,
    )
