from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from dcn.config.semantic import KMEANS_MATERIALIZATION_REVISION
from dcn.semantic import SemanticCodes
from experiments.g6_rqkmeans_history.native500m.configs.runtime import (
    build_collision_pair,
)
from experiments.g6_rqkmeans_history.native500m.launchers.queue import (
    PROJECT_ROOT,
    canonical_bytes,
    persist_immutable_bytes,
    source_identity_sha256,
)
from experiments.g6_rqkmeans_history.native500m.protocol.contracts import (
    canonical_sha256,
)
from experiments.g6_rqkmeans_history.native500m.protocol.design import (
    DATASET_SIZE,
    SHARED_CODEBOOK_SIZES,
    TOKENIZER_LEVELS,
)
from experiments.g6_rqkmeans_history.native500m.protocol.tokenizer_registry import (
    SCHEMA,
    load_registry,
)
from utils.global_config import config as global_config


DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments/g6_rqkmeans_history/evidence/native500m/"
    "tokenizer_registry_shared_base_v2_r4.json"
)


def materialize_registry(output: Path = DEFAULT_OUTPUT) -> Path:
    source_sha256 = source_identity_sha256()
    rows = []
    for levels in TOKENIZER_LEVELS:
        for shared_codes in SHARED_CODEBOOK_SIZES:
            suffix, none = build_collision_pair(
                backbone="best_g1",
                representation="learned_sid_tokens",
                embedding_learning_rate=0.032,
                deep_learning_rate=0.016,
                num_levels=levels,
                num_codes=shared_codes,
                suffix_run_name=f"g6_tokenizer_l{levels}_c{shared_codes}_suffix",
                no_suffix_run_name=f"g6_tokenizer_l{levels}_c{shared_codes}_none",
            )
            global_config.initialize(Path(suffix.base_path))
            suffix.semantic_stage.run()
            none.semantic_stage.run()
            suffix_codes = SemanticCodes.load(suffix.semantic_stage.codes_path)
            none_codes = SemanticCodes.load(none.semantic_stage.codes_path)
            if (
                not suffix.semantic_stage.cache_complete
                or not none.semantic_stage.cache_complete
            ):
                raise RuntimeError("tokenizer pre-materialization is incomplete")
            if not suffix_codes.item_ids.equal(
                none_codes.item_ids
            ) or not suffix_codes.codes[:, :levels].equal(none_codes.codes):
                raise RuntimeError(
                    "collision policies do not share exact base assignments"
                )
            rows.append(_registry_row(suffix, none))
    if source_identity_sha256() != source_sha256:
        raise RuntimeError("source identity changed during tokenizer materialization")
    body = {
        "schema": SCHEMA,
        "dataset_size": DATASET_SIZE,
        "materialization_revision": KMEANS_MATERIALIZATION_REVISION,
        "source_identity_sha256": source_sha256,
        "tokenizers": rows,
    }
    document = {**body, "sha256": canonical_sha256(body)}
    path = persist_immutable_bytes(
        output, canonical_bytes(document), label="tokenizer registry"
    )
    load_registry(path, source_sha256=body["source_identity_sha256"])
    return path


def _registry_row(suffix: object, none: object) -> dict[str, object]:
    stage = suffix.semantic_stage
    return {
        "levels": suffix.semantic.num_levels,
        "shared_codes": suffix.semantic.num_codes,
        "base_cache_key": suffix.semantic.base_cache_key,
        "fit_materialization": _artifact(stage.fit_materialization_marker_path),
        "policies": {
            "suffix": _policy_artifacts(suffix.semantic_stage),
            "none": _policy_artifacts(none.semantic_stage),
        },
    }


def _policy_artifacts(stage: object) -> dict[str, str]:
    return {
        "codes_path": _relative(stage.codes_path),
        "codes_sha256": _sha256(stage.codes_path),
        "materialization_path": _relative(stage.materialization_marker_path),
        "materialization_sha256": _sha256(stage.materialization_marker_path),
    }


def _artifact(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256": _sha256(path)}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(materialize_registry(arguments.output))


if __name__ == "__main__":
    main()
