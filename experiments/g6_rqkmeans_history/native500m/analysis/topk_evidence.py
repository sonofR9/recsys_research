from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any

import torch

from experiments.g6_rqkmeans_history.native500m.launchers.queue import (
    persist_immutable_bytes,
)


_SCHEMA = "g6-native500m-topk/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class TopKContext:
    dataset: str
    split: str
    ranking_context_sha256: str
    ordered_catalog_sha256: str
    checkpoint_sha256: str
    evaluator_configuration_sha256: str
    stage: str
    job_id: str
    job_logical_sha256: str
    manifest_logical_sha256: str
    semantic_codes_sha256: str

    def __post_init__(self) -> None:
        if self.dataset != "yambda-500m" or self.split != "final-seven-days":
            raise ValueError("top-K evidence dataset or split differs")
        if not self.stage or not self.job_id:
            raise ValueError("top-K evidence has no stage or job identity")
        for name in (
            "ranking_context_sha256",
            "ordered_catalog_sha256",
            "checkpoint_sha256",
            "evaluator_configuration_sha256",
            "job_logical_sha256",
            "manifest_logical_sha256",
            "semantic_codes_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"top-K evidence {name} is invalid")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TopKEvidence:
    path: Path
    context: TopKContext
    user_ids: torch.Tensor
    recommended_item_ids: torch.Tensor
    ordered_catalog_item_ids: torch.Tensor
    logical_sha256: str
    physical_sha256: str


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = _int64(value)
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def write_topk_evidence(
    path: Path,
    *,
    context: TopKContext,
    user_ids: torch.Tensor,
    recommended_item_ids: torch.Tensor,
    ordered_catalog_item_ids: torch.Tensor,
) -> TopKEvidence:
    payload = _payload(
        context,
        user_ids=user_ids,
        recommended_item_ids=recommended_item_ids,
        ordered_catalog_item_ids=ordered_catalog_item_ids,
    )
    _validate_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_topk_evidence(path, expected_context=context)
        if existing.logical_sha256 != payload["logical_sha256"]:
            raise RuntimeError(f"immutable top-K evidence differs: {path}")
        return existing
    stream = io.BytesIO()
    torch.save(payload, stream)
    persist_immutable_bytes(path, stream.getvalue(), label="top-K evidence")
    return load_topk_evidence(path, expected_context=context)


def load_topk_evidence(
    path: Path,
    *,
    expected_context: TopKContext | None = None,
    expected_physical_sha256: str | None = None,
) -> TopKEvidence:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("top-K evidence must be a regular file")
    physical_sha256 = _file_sha256(resolved)
    if (
        expected_physical_sha256 is not None
        and physical_sha256 != expected_physical_sha256
    ):
        raise ValueError("top-K evidence physical SHA-256 differs")
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"cannot read top-K evidence: {resolved}") from error
    _validate_payload(payload)
    context = TopKContext(**payload["context"])
    if expected_context is not None and context != expected_context:
        raise ValueError("top-K evidence context differs")
    return TopKEvidence(
        path=resolved,
        context=context,
        user_ids=payload["user_ids"],
        recommended_item_ids=payload["recommended_item_ids"],
        ordered_catalog_item_ids=payload["ordered_catalog_item_ids"],
        logical_sha256=payload["logical_sha256"],
        physical_sha256=physical_sha256,
    )


def _payload(
    context: TopKContext,
    *,
    user_ids: torch.Tensor,
    recommended_item_ids: torch.Tensor,
    ordered_catalog_item_ids: torch.Tensor,
) -> dict[str, Any]:
    body = {
        "schema": _SCHEMA,
        "context": context.to_dict(),
        "user_ids": _int64(user_ids),
        "recommended_item_ids": _int64(recommended_item_ids),
        "ordered_catalog_item_ids": _int64(ordered_catalog_item_ids),
    }
    return {**body, "logical_sha256": _payload_sha256(body)}


def _validate_payload(payload: object) -> None:
    expected = {
        "schema",
        "context",
        "user_ids",
        "recommended_item_ids",
        "ordered_catalog_item_ids",
        "logical_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("top-K evidence schema differs")
    body = {key: value for key, value in payload.items() if key != "logical_sha256"}
    if (
        payload["schema"] != _SCHEMA
        or not _SHA256.fullmatch(str(payload["logical_sha256"]))
        or _payload_sha256(body) != payload["logical_sha256"]
    ):
        raise ValueError("top-K evidence logical SHA-256 differs")
    context = payload["context"]
    if not isinstance(context, dict):
        raise ValueError("top-K evidence context is invalid")
    parsed_context = TopKContext(**context)
    users = payload["user_ids"]
    rankings = payload["recommended_item_ids"]
    catalog = payload["ordered_catalog_item_ids"]
    if any(not isinstance(value, torch.Tensor) for value in (users, rankings, catalog)):
        raise ValueError("top-K evidence tensors are missing")
    if users.ndim != 1 or catalog.ndim != 1 or rankings.ndim != 2:
        raise ValueError("top-K evidence tensor ranks differ")
    if rankings.shape != (users.shape[0], 100):
        raise ValueError("top-K evidence must contain exactly 100 items per user")
    if len(set(users.tolist())) != users.shape[0]:
        raise ValueError("top-K evidence user IDs are not unique")
    catalog_values = catalog.tolist()
    if len(set(catalog_values)) != catalog.shape[0]:
        raise ValueError("top-K evidence catalog IDs are not unique")
    catalog_set = set(catalog_values)
    for row in rankings.tolist():
        if len(set(row)) != 100 or not set(row) <= catalog_set:
            raise ValueError("top-K evidence rankings are invalid")
    if tensor_sha256(catalog) != parsed_context.ordered_catalog_sha256:
        raise ValueError("top-K evidence ordered catalog differs")


def _payload_sha256(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    scalars = {
        key: value for key, value in payload.items() if not isinstance(value, torch.Tensor)
    }
    digest.update(
        json.dumps(
            scalars,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    )
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, torch.Tensor):
            digest.update(key.encode())
            digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def _int64(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.int64).contiguous()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
