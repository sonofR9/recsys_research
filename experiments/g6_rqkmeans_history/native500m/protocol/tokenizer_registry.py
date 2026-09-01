from __future__ import annotations

import hashlib
import fcntl
import json
from pathlib import Path
import re
from typing import Mapping

from dcn.config.semantic import KMEANS_MATERIALIZATION_REVISION
from dcn.semantic import ResidualCodebooks, SemanticCodes

from .contracts import canonical_sha256
from .design import DATASET_SIZE, SHARED_CODEBOOK_SIZES, TOKENIZER_LEVELS


SCHEMA = "g6-native500m-tokenizer-registry/v1"
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "evidence/native500m/tokenizer_registry_shared_base_v2_r4.json"
)
ENVIRONMENT_FIELDS = {
    "G6_NATIVE500M_TOKENIZER_BINDING_REVISION",
    "G6_NATIVE500M_TOKENIZER_REGISTRY_SHA256",
    "G6_NATIVE500M_TOKENIZER_BASE_CACHE_KEY",
    "G6_NATIVE500M_TOKENIZER_FIT_SHA256",
    "G6_NATIVE500M_TOKENIZER_CODES_SHA256",
    "G6_NATIVE500M_TOKENIZER_MATERIALIZATION_SHA256",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def load_registry(path: Path, *, source_sha256: str) -> dict[str, object]:
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("tokenizer registry is not valid JSON") from error
    expected = {
        "schema",
        "dataset_size",
        "materialization_revision",
        "source_identity_sha256",
        "tokenizers",
        "sha256",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError("tokenizer registry schema differs")
    body = {name: value for name, value in document.items() if name != "sha256"}
    if (
        document["schema"] != SCHEMA
        or document["dataset_size"] != DATASET_SIZE
        or document["materialization_revision"] != KMEANS_MATERIALIZATION_REVISION
        or document["source_identity_sha256"] != source_sha256
        or document["sha256"] != canonical_sha256(body)
        or not _SHA256.fullmatch(str(document["sha256"]))
    ):
        raise ValueError("tokenizer registry identity differs")
    rows = document["tokenizers"]
    if not isinstance(rows, list) or len(rows) != len(TOKENIZER_LEVELS) * len(
        SHARED_CODEBOOK_SIZES
    ):
        raise ValueError("tokenizer registry coverage differs")
    coordinates: set[tuple[int, int]] = set()
    for row in rows:
        _validate_row(row)
        coordinates.add((row["levels"], row["shared_codes"]))
    expected_coordinates = {
        (levels, shared_codes)
        for levels in TOKENIZER_LEVELS
        for shared_codes in SHARED_CODEBOOK_SIZES
    }
    if coordinates != expected_coordinates:
        raise ValueError("tokenizer registry coordinates differ")
    return document


def binding_environment(
    registry: Mapping[str, object],
    *,
    levels: int,
    shared_codes: int,
    collision_policy: str,
) -> dict[str, str]:
    rows = registry.get("tokenizers")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)  # type: ignore[union-attr]
        and row.get("levels") == levels
        and row.get("shared_codes") == shared_codes
    ]
    if len(matches) != 1:
        raise ValueError("tokenizer registry coordinate is absent")
    row = matches[0]
    policies = row["policies"]
    if not isinstance(policies, dict) or collision_policy not in policies:
        raise ValueError("tokenizer registry collision policy is absent")
    policy = policies[collision_policy]
    if not isinstance(policy, dict):
        raise ValueError("tokenizer registry policy binding is invalid")
    return {
        "G6_NATIVE500M_TOKENIZER_BINDING_REVISION": KMEANS_MATERIALIZATION_REVISION,
        "G6_NATIVE500M_TOKENIZER_REGISTRY_SHA256": str(registry["sha256"]),
        "G6_NATIVE500M_TOKENIZER_BASE_CACHE_KEY": str(row["base_cache_key"]),
        "G6_NATIVE500M_TOKENIZER_FIT_SHA256": str(row["fit_materialization"]["sha256"]),
        "G6_NATIVE500M_TOKENIZER_CODES_SHA256": str(policy["codes_sha256"]),
        "G6_NATIVE500M_TOKENIZER_MATERIALIZATION_SHA256": str(
            policy["materialization_sha256"]
        ),
    }


def verify_binding(
    *, experiment: object, environment: Mapping[str, str], registry_sha256: str
) -> None:
    supplied = ENVIRONMENT_FIELDS & environment.keys()
    if not supplied:
        return
    if supplied != ENVIRONMENT_FIELDS:
        raise RuntimeError("native-500M tokenizer binding is incomplete")
    stage = experiment.semantic_stage
    with stage.materialization_lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        expected = {
            "G6_NATIVE500M_TOKENIZER_BINDING_REVISION": KMEANS_MATERIALIZATION_REVISION,
            "G6_NATIVE500M_TOKENIZER_REGISTRY_SHA256": registry_sha256,
            "G6_NATIVE500M_TOKENIZER_BASE_CACHE_KEY": experiment.semantic.base_cache_key,
            "G6_NATIVE500M_TOKENIZER_FIT_SHA256": _file_sha256(
                stage.fit_materialization_marker_path
            ),
            "G6_NATIVE500M_TOKENIZER_CODES_SHA256": _file_sha256(stage.codes_path),
            "G6_NATIVE500M_TOKENIZER_MATERIALIZATION_SHA256": _file_sha256(
                stage.materialization_marker_path
            ),
        }
        if dict((name, environment[name]) for name in ENVIRONMENT_FIELDS) != expected:
            raise RuntimeError("native-500M tokenizer artifact identity differs")
        if not stage.cache_complete:
            raise RuntimeError("native-500M tokenizer materialization is incomplete")
        semantic_codes = SemanticCodes.load(stage.codes_path)
        semantic_codebooks = ResidualCodebooks.load(stage.codebooks_path)
        if (
            _file_sha256(stage.codes_path)
            != expected["G6_NATIVE500M_TOKENIZER_CODES_SHA256"]
        ):
            raise RuntimeError("native-500M tokenizer changed while loading")
        experiment.__dict__["semantic_codes"] = semantic_codes
        experiment.__dict__["semantic_codebooks"] = semantic_codebooks


def _validate_row(value: object) -> None:
    expected = {
        "levels",
        "shared_codes",
        "base_cache_key",
        "fit_materialization",
        "policies",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("tokenizer registry row schema differs")
    if not isinstance(value["base_cache_key"], str) or not value["base_cache_key"]:
        raise ValueError("tokenizer registry base cache key is invalid")
    _validate_artifact(value["fit_materialization"])
    policies = value["policies"]
    if not isinstance(policies, dict) or set(policies) != {"suffix", "none"}:
        raise ValueError("tokenizer registry policies differ")
    for policy in policies.values():
        if not isinstance(policy, dict) or set(policy) != {
            "codes_path",
            "codes_sha256",
            "materialization_path",
            "materialization_sha256",
        }:
            raise ValueError("tokenizer registry policy schema differs")
        for name in ("codes_sha256", "materialization_sha256"):
            if not _SHA256.fullmatch(str(policy[name])):
                raise ValueError("tokenizer registry artifact SHA-256 is invalid")


def _validate_artifact(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256"}
        or not isinstance(value["path"], str)
        or not value["path"]
        or not _SHA256.fullmatch(str(value["sha256"]))
    ):
        raise ValueError("tokenizer registry fit artifact is invalid")


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError("native-500M tokenizer artifact is absent") from error
