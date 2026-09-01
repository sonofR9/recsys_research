import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dcn.semantic import ResidualCodebooks, SemanticCodes
from experiments.g6_rqkmeans_history.native500m.launchers import (
    pre_materialize_tokenizers,
)

from experiments.g6_rqkmeans_history.native500m.protocol.contracts import (
    canonical_sha256,
)
from experiments.g6_rqkmeans_history.native500m.protocol.tokenizer_registry import (
    ENVIRONMENT_FIELDS,
    SCHEMA,
    binding_environment,
    load_registry,
    verify_binding,
)


def _registry(path: Path, source_sha256: str) -> dict[str, object]:
    rows = []
    for levels in (3, 4):
        for shared_codes in (512, 2048, 8192):
            rows.append(
                {
                    "levels": levels,
                    "shared_codes": shared_codes,
                    "base_cache_key": f"kmeans_{levels}x{shared_codes}_test",
                    "fit_materialization": {
                        "path": f"fit-{levels}-{shared_codes}.json",
                        "sha256": "1" * 64,
                    },
                    "policies": {
                        policy: {
                            "codes_path": f"{policy}-codes.pt",
                            "codes_sha256": digit * 64,
                            "materialization_path": f"{policy}-marker.json",
                            "materialization_sha256": (digit.upper().lower()) * 64,
                        }
                        for policy, digit in (("suffix", "2"), ("none", "3"))
                    },
                }
            )
    body = {
        "schema": SCHEMA,
        "dataset_size": "native-500m",
        "materialization_revision": "shared-base-v2",
        "source_identity_sha256": source_sha256,
        "tokenizers": rows,
    }
    document = {**body, "sha256": canonical_sha256(body)}
    path.write_text(json.dumps(document))
    return document


def test_registry_binds_each_policy_to_one_shared_fit(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    source_sha256 = "a" * 64
    expected = _registry(path, source_sha256)

    registry = load_registry(path, source_sha256=source_sha256)
    suffix = binding_environment(
        registry, levels=4, shared_codes=8192, collision_policy="suffix"
    )
    none = binding_environment(
        registry, levels=4, shared_codes=8192, collision_policy="none"
    )

    assert registry == expected
    assert (
        suffix["G6_NATIVE500M_TOKENIZER_FIT_SHA256"]
        == none["G6_NATIVE500M_TOKENIZER_FIT_SHA256"]
    )
    assert (
        suffix["G6_NATIVE500M_TOKENIZER_CODES_SHA256"]
        != none["G6_NATIVE500M_TOKENIZER_CODES_SHA256"]
    )
    assert set(suffix) == ENVIRONMENT_FIELDS


def test_runtime_binding_rejects_changed_artifacts(tmp_path: Path) -> None:
    fit = tmp_path / "fit.json"
    codes = tmp_path / "codes.pt"
    codebooks = tmp_path / "codebooks.pt"
    marker = tmp_path / "marker.json"
    for path, content in ((fit, b"fit"), (marker, b"marker")):
        path.write_bytes(content)
    SemanticCodes(
        item_ids=torch.tensor([1, 2]),
        codes=torch.tensor([[0], [1]]),
        codes_per_level=(2,),
    ).save(codes)
    ResidualCodebooks(torch.randn(1, 2, 3)).save(codebooks)
    environment = {
        "G6_NATIVE500M_TOKENIZER_BINDING_REVISION": "shared-base-v2",
        "G6_NATIVE500M_TOKENIZER_REGISTRY_SHA256": "4" * 64,
        "G6_NATIVE500M_TOKENIZER_BASE_CACHE_KEY": "kmeans_4x8192_test",
        "G6_NATIVE500M_TOKENIZER_FIT_SHA256": hashlib.sha256(
            fit.read_bytes()
        ).hexdigest(),
        "G6_NATIVE500M_TOKENIZER_CODES_SHA256": hashlib.sha256(
            codes.read_bytes()
        ).hexdigest(),
        "G6_NATIVE500M_TOKENIZER_MATERIALIZATION_SHA256": hashlib.sha256(
            marker.read_bytes()
        ).hexdigest(),
    }
    experiment = SimpleNamespace(
        semantic=SimpleNamespace(base_cache_key="kmeans_4x8192_test"),
        semantic_stage=SimpleNamespace(
            fit_materialization_marker_path=fit,
            codes_path=codes,
            codebooks_path=codebooks,
            materialization_marker_path=marker,
            materialization_lock_path=tmp_path / "materialization.lock",
            cache_complete=True,
        ),
    )

    verify_binding(
        experiment=experiment, environment=environment, registry_sha256="4" * 64
    )
    assert "semantic_codes" in experiment.__dict__
    assert "semantic_codebooks" in experiment.__dict__
    codes.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="artifact identity"):
        verify_binding(
            experiment=experiment,
            environment=environment,
            registry_sha256="4" * 64,
        )


def test_runtime_binding_rejects_partial_environment() -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        verify_binding(
            experiment=SimpleNamespace(),
            environment={"G6_NATIVE500M_TOKENIZER_BINDING_REVISION": "shared-base-v2"},
            registry_sha256="4" * 64,
        )


def test_registry_is_not_sealed_after_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        pre_materialize_tokenizers,
        "source_identity_sha256",
        lambda: next(identities),
    )
    monkeypatch.setattr(pre_materialize_tokenizers, "TOKENIZER_LEVELS", ())
    monkeypatch.setattr(pre_materialize_tokenizers, "SHARED_CODEBOOK_SIZES", ())

    with pytest.raises(RuntimeError, match="changed during"):
        pre_materialize_tokenizers.materialize_registry(tmp_path / "registry.json")
