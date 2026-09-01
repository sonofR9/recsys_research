from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .constants import APPROVED_PROTOCOL_SHA256


RQ4_WIDTH256_SELECTION_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq4_single_metadata_width256_boundary_selection_native50m.json"
)
RQ4_WIDTH256_SELECTION_SHA256 = (
    "f2d972eb6f22637a47c11ae4326adce52c9401b7356b81f479afa8b8ac03d966"
)
RQ4_FOLLOWUP_GATE_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq4_post_width256_followup_gate.json"
)


def compile_rq4_followup_gate(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    selection_path = root / RQ4_WIDTH256_SELECTION_PATH
    if _has_symlink_component(root, selection_path):
        raise ValueError("RQ4 width-256 selection must not be a symlink")
    selection = _load_json(selection_path)
    if selection_path.read_bytes() != (_canonical_json(selection) + "\n").encode():
        raise ValueError("RQ4 width-256 selection must use exact canonical bytes")
    payload = {name: value for name, value in selection.items() if name != "sha256"}
    if (
        selection.get("sha256") != RQ4_WIDTH256_SELECTION_SHA256
        or _canonical_sha256(payload) != RQ4_WIDTH256_SELECTION_SHA256
        or selection.get("renewed_capacity_approval_required")
        != ["rq4_artist", "rq4_album"]
        or selection.get("renewed_learning_rate_approval_required") != []
    ):
        raise ValueError("RQ4 width-256 selection cannot authorize a follow-up")
    gate_payload = {
        "schema_version": 1,
        "kind": "g3_rq4_post_width256_followup_gate",
        "protocol_sha256": APPROVED_PROTOCOL_SHA256,
        "input": {
            "path": RQ4_WIDTH256_SELECTION_PATH,
            "size_bytes": selection_path.stat().st_size,
            "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            "logical_sha256": RQ4_WIDTH256_SELECTION_SHA256,
        },
        "approval_state": {
            "further_capacity_width_approved": False,
            "horizon_materialization_approved": False,
        },
        "blockers": [
            "rq4_artist_capacity_upper_boundary",
            "rq4_album_capacity_upper_boundary",
            "renewed_user_approval_required",
        ],
    }
    return gate_payload | {"sha256": _canonical_sha256(gate_payload)}


def persist_rq4_followup_gate(
    path: Path, document: dict[str, object], *, root: Path
) -> Path:
    root = root.resolve(strict=True)
    canonical = root / RQ4_FOLLOWUP_GATE_PATH
    if (
        _has_symlink_component(root, path)
        or path.resolve(strict=False) != canonical.resolve(strict=False)
    ):
        raise ValueError("RQ4 follow-up gate must use its canonical project path")
    if document != compile_rq4_followup_gate(root):
        raise ValueError("RQ4 follow-up gate differs from authenticated evidence")
    content = (_canonical_json(document) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable RQ4 follow-up gate differs: {path}")
    return path


def require_rq4_horizon_materialization_approval(root: Path) -> None:
    _require_approval(root, "horizon_materialization_approved", "horizon materialization")


def require_rq4_further_capacity_width_approval(root: Path) -> None:
    _require_approval(root, "further_capacity_width_approved", "further capacity width")


def _require_approval(root: Path, field: str, action: str) -> None:
    root = root.resolve(strict=True)
    gate_path = root / RQ4_FOLLOWUP_GATE_PATH
    if not gate_path.exists():
        raise ValueError("RQ4 materialization requires its canonical approval gate")
    if _has_symlink_component(root, gate_path):
        raise ValueError("RQ4 approval gate must not be a symlink")
    gate = _load_json(gate_path)
    expected = compile_rq4_followup_gate(root)
    expected_bytes = (_canonical_json(expected) + "\n").encode()
    if gate != expected or gate_path.read_bytes() != expected_bytes:
        raise ValueError("RQ4 follow-up gate changed from its exact canonical bytes")
    approval = gate.get("approval_state")
    if (
        not isinstance(approval, dict)
        or approval.get(field) is not True
    ):
        raise ValueError(
            f"RQ4 {action} requires renewed user approval after unresolved "
            "width-256 capacity boundaries"
        )


def _load_json(path: Path) -> dict[str, object]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError(f"duplicate JSON key {key!r}: {path}")
            document[key] = value
        return document

    document = json.loads(
        path.read_text(), object_pairs_hook=reject_duplicate_keys
    )
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _has_symlink_component(root: Path, path: Path) -> bool:
    try:
        relative = path.absolute().relative_to(root)
    except ValueError:
        return True
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            return True
    return False
