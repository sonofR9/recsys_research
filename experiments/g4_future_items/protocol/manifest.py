from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import optuna


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTROL_MANIFEST_PATH = Path(__file__).with_name("control_manifest.json")
PREIMPLEMENTATION_SOURCE_MANIFEST_PATH = Path(__file__).with_name(
    "preimplementation_source_manifest.json"
)
APPROVED_CONTROL_MANIFEST_SHA256 = (
    "ceccb6d6e73d082dea9502fa64f1e2af88c3460788dffdb4656e5aaf6aebd459"
)
LEDGER_VERSION = 2
CONTROL_BATCH_SIZE = 512
EMBEDDING_LR_BOUNDS = (0.0001, 0.256)
DEEP_LR_BOUNDS = (0.0001, 0.128)
EMBEDDING_LR_SELECTION_BOUNDS = (
    EMBEDDING_LR_BOUNDS[0] / 16,
    EMBEDDING_LR_BOUNDS[1] * 16,
)
DEEP_LR_SELECTION_BOUNDS = (DEEP_LR_BOUNDS[0] / 16, DEEP_LR_BOUNDS[1] * 16)
BASE_HORIZON_VALUES = (5, 10, 15, 20, 25, 30)
LOWER_ROUND_ONE_HORIZON_VALUES = tuple(range(2, 31))
UPPER_ROUND_ONE_HORIZON_VALUES = tuple(range(5, 41))
LOWER_ROUND_TWO_HORIZON_VALUES = tuple(range(1, 31))
UPPER_ROUND_TWO_HORIZON_VALUES = tuple(range(5, 51))
_CONTROL_SOURCE_PATHS = (
    "data/__init__.py",
    "data/counters/__init__.py",
    "data/counters/config.py",
    "data/counters/counter.py",
    "data/preprocessing.py",
    "data/split_by_day.py",
    "data/utils.py",
    "dcn/__init__.py",
    "dcn/config/__init__.py",
    "dcn/config/experiment.py",
    "dcn/config/generation.py",
    "dcn/config/networks.py",
    "dcn/config/query_retrieval.py",
    "dcn/config/ranking.py",
    "dcn/config/retrieval.py",
    "dcn/config/sasrec.py",
    "dcn/config/semantic.py",
    "dcn/config/semantic_history.py",
    "dcn/config/sequence.py",
    "dcn/config/settings.py",
    "dcn/config/yambda.py",
    "dcn/config/yambda_base.py",
    "dcn/data/__init__.py",
    "dcn/data/dataset.py",
    "dcn/data/dataset_manager.py",
    "dcn/data/features.py",
    "dcn/data/packed.py",
    "dcn/data/sequence_dataset.py",
    "dcn/datasets/__init__.py",
    "dcn/datasets/base.py",
    "dcn/datasets/remap.py",
    "dcn/datasets/yambda.py",
    "dcn/eval/__init__.py",
    "dcn/eval/base.py",
    "dcn/eval/callback.py",
    "dcn/eval/generation.py",
    "dcn/eval/pairwise.py",
    "dcn/eval/ranking_evidence.py",
    "dcn/eval/ranking_metrics.py",
    "dcn/eval/true_metric.py",
    "dcn/main.py",
    "dcn/models/__init__.py",
    "dcn/models/criterions.py",
    "dcn/models/cross_attention_retrieval.py",
    "dcn/models/history_tokens.py",
    "dcn/models/loss_wrapper.py",
    "dcn/models/multi_head_network.py",
    "dcn/models/semantic_constraint.py",
    "dcn/models/sequence_retrieval.py",
    "dcn/models/sequence_targets.py",
    "dcn/models/token_generation.py",
    "dcn/models/two_tower.py",
    "dcn/nn/__init__.py",
    "dcn/nn/crossnet.py",
    "dcn/nn/dcnv2.py",
    "dcn/nn/densenet.py",
    "dcn/nn/esasrec.py",
    "dcn/nn/ffn.py",
    "dcn/nn/history_encoder.py",
    "dcn/nn/layer_item_features.py",
    "dcn/nn/layer_registry.py",
    "dcn/nn/multi_task_embedding.py",
    "dcn/nn/ple.py",
    "dcn/nn/precomputed_embeddings.py",
    "dcn/nn/resnet.py",
    "dcn/nn/sampled_softmax.py",
    "dcn/nn/semantic_embedding.py",
    "dcn/nn/transformer.py",
    "dcn/nn/types.py",
    "dcn/semantic/__init__.py",
    "dcn/semantic/artifacts.py",
    "dcn/semantic/codes.py",
    "dcn/semantic/diagnostics.py",
    "dcn/semantic/residual_kmeans.py",
    "dcn/semantic/rq_vae.py",
    "dcn/semantic/trie.py",
    "dcn/training/__init__.py",
    "dcn/training/combined_optimizer.py",
    "dcn/training/epoch_trainer.py",
    "dcn/training/optimizer_groups.py",
    "dcn/training/pretrain_callback.py",
    "dcn/training/trainer.py",
    "dcn/training_metadata.py",
    "experiments/g4_future_items/__init__.py",
    "experiments/g4_future_items/configs/__init__.py",
    "experiments/g4_future_items/configs/control.py",
    "experiments/g4_future_items/launchers/freeze_control.py",
    "experiments/g4_future_items/launchers/run_control.py",
    "experiments/g4_future_items/protocol/__init__.py",
    "experiments/g4_future_items/protocol/control_manifest.json",
    "experiments/g4_future_items/protocol/manifest.py",
    "experiments/g4_future_items/protocol/manifest_contract.md",
    "neuralrec/__init__.py",
    "neuralrec/data/__init__.py",
    "neuralrec/data/dataloader.py",
    "neuralrec/data/transforms.py",
    "neuralrec/nn/__init__.py",
    "neuralrec/nn/autocast.py",
    "neuralrec/nn/metrics/__init__.py",
    "neuralrec/nn/metrics/base.py",
    "neuralrec/nn/metrics/classification.py",
    "neuralrec/nn/metrics/recall.py",
    "neuralrec/nn/metrics/regression.py",
    "neuralrec/run/__init__.py",
    "neuralrec/run/callbacks/__init__.py",
    "neuralrec/run/callbacks/base.py",
    "neuralrec/run/callbacks/best_weights.py",
    "neuralrec/run/callbacks/checkpoint.py",
    "neuralrec/run/callbacks/clipping.py",
    "neuralrec/run/callbacks/early_stopping.py",
    "neuralrec/run/callbacks/logging.py",
    "neuralrec/run/callbacks/lr_schedule.py",
    "neuralrec/run/callbacks/resources.py",
    "neuralrec/run/callbacks/tensorboard.py",
    "neuralrec/run/callbacks/validation.py",
    "neuralrec/run/callbacks/wandb.py",
    "neuralrec/run/train.py",
    "neuralrec/utils/__init__.py",
    "neuralrec/utils/stateful.py",
    "neuralrec/utils/utils.py",
    "utils/__init__.py",
    "utils/global_config.py",
    "utils/locks.py",
    "utils/report_file_facts.py",
)
MATERIALIZATION_COST_LIMITS = {
    "wall_seconds": 12 * 60 * 60,
    "peak_aggregate_rss_bytes": 250 * 2**30,
    "logical_output_scratch_bytes": 250 * 2**30,
}
ObjectiveId = Literal[
    "rq1_24h",
    "rq2_next10",
    "rq3_deterministic_hard",
    "rq3_learned_hard",
    "rq3_learned_proportional",
]


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")


def load_strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON document {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON document {path} must be an object")
    return value


def _same_json_type(value: Any, template: Any) -> bool:
    if isinstance(template, bool):
        return isinstance(value, bool)
    if isinstance(template, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(template, float):
        return isinstance(value, float)
    if template is None:
        return value is None
    return type(value) is type(template)


def _validate_closed_tree(value: Any, template: Any, path: str = "") -> None:
    if isinstance(template, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{path or '/'} must be an object")
        if value.keys() != template.keys():
            missing = sorted(template.keys() - value.keys())
            unknown = sorted(value.keys() - template.keys())
            raise ValueError(
                f"{path or '/'} has missing keys {missing} and unknown keys {unknown}"
            )
        for key in template:
            _validate_closed_tree(value[key], template[key], f"{path}/{key}")
        return
    if isinstance(template, list):
        if not isinstance(value, list) or len(value) != len(template):
            raise ValueError(f"{path or '/'} has the wrong list shape")
        for index, (item, expected) in enumerate(zip(value, template)):
            _validate_closed_tree(item, expected, f"{path}/{index}")
        return
    if not _same_json_type(value, template):
        raise ValueError(f"{path or '/'} has the wrong JSON value type")


def load_control_manifest(path: Path = CONTROL_MANIFEST_PATH) -> dict[str, Any]:
    document = load_strict_json(path)
    approved = (
        load_strict_json(CONTROL_MANIFEST_PATH)
        if path != CONTROL_MANIFEST_PATH
        else document
    )
    _validate_closed_tree(document, approved)
    digest = canonical_sha256(document)
    if digest != APPROVED_CONTROL_MANIFEST_SHA256:
        raise ValueError(
            f"control manifest hash {digest} does not match approved "
            f"{APPROVED_CONTROL_MANIFEST_SHA256}"
        )
    return document


def validate_control_round_trip(experiment: Any) -> dict[str, Any]:
    from experiments.g4_future_items.configs.control import control_runtime_projection

    manifest = load_control_manifest()
    projection = control_runtime_projection(experiment)
    anchor = manifest["anchor"]
    expected_selected = {
        "batch_size": anchor["batch_size"],
        "embedding_learning_rate": anchor["embedding_learning_rate"],
        "deep_learning_rate": anchor["deep_learning_rate"],
        "lr_schedule_horizon_epochs": anchor["lr_schedule_horizon_epochs"],
    }
    if projection != {"fixed": manifest["fixed"], "selected": expected_selected}:
        raise ValueError("control runtime does not round-trip through the manifest")
    return manifest


def source_manifest(root: Path, relative_paths: list[str]) -> dict[str, str]:
    if relative_paths != sorted(set(relative_paths)):
        raise ValueError("source paths must be unique and sorted")
    result: dict[str, str] = {}
    root = root.resolve()
    for relative in relative_paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                f"source path is absent or outside the project: {relative}"
            )
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"data identity path must be a regular file: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _file_sha256(path),
    }


def _mapped_catalog_sha256(remap_path: Path) -> str:
    import polars as pl

    frame = pl.read_parquet(remap_path, columns=["compact_id"])
    values = sorted(
        {
            int(value)
            for value in frame["compact_id"].to_list()
            if value is not None and int(value) > 0
        }
    )
    return canonical_sha256(values)


def next_item_target_fixture_sha256() -> str:
    import torch

    from dcn.models.sequence_targets import NextItemTargets

    item_ids = torch.tensor([11, 12, 13, 21, 22], dtype=torch.long)
    values = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    pairs = NextItemTargets()(
        {
            "lengths": torch.tensor([3, 2], dtype=torch.long),
            "item_ids": item_ids,
            "query_repr": values,
            "item_repr": values + 10,
            "is_target": torch.tensor([True, True, True, True, True]),
            "is_query": torch.tensor([True, False, True, True, True]),
        }
    )
    document = {
        "query_repr": pairs.query_repr.tolist(),
        "positive_repr": pairs.positive_repr.tolist(),
        "positive_ids": pairs.positive_ids.tolist(),
        "group_sizes": pairs.group_sizes.tolist(),
    }
    return canonical_sha256(document)


def resolve_control_data_identity(experiment: Any) -> dict[str, Any]:
    artifacts = experiment.artifacts
    main_path = Path(artifacts.main_parquet).resolve()
    remap_path = main_path.with_name("item_id_remap.parquet")
    try:
        content_path = Path(
            artifacts.precomputed_embeddings[artifacts.item_id_column]
        ).resolve()
    except (AttributeError, KeyError) as error:
        raise ValueError("control content-embedding artifact is unresolved") from error
    result = {
        "dataset_key": experiment.dataset_key,
        "main": _file_identity(main_path),
        "remap": _file_identity(remap_path),
        "content_embeddings": _file_identity(content_path),
        "split_cutoff_timestamp": experiment.validation_cutoff_timestamp,
        "mapped_catalog_sha256": _mapped_catalog_sha256(remap_path),
        "next_item_target_fixture_sha256": next_item_target_fixture_sha256(),
    }
    _validate_data_identity(result)
    return result


def _current_control_data_identity(frozen: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    main_path = Path(frozen["main"]["path"]).resolve()
    remap_path = Path(frozen["remap"]["path"]).resolve()
    content_path = Path(frozen["content_embeddings"]["path"]).resolve()
    fixed_data = load_control_manifest()["fixed"]["data"]
    maximum = (
        pl.scan_parquet(main_path)
        .select(pl.col("timestamp").max().alias("maximum"))
        .collect()["maximum"][0]
    )
    if maximum is None:
        raise ValueError("control main parquet has no timestamp rows")
    result = {
        "dataset_key": hashlib.sha1(str(main_path).encode()).hexdigest()[:12],
        "main": _file_identity(main_path),
        "remap": _file_identity(remap_path),
        "content_embeddings": _file_identity(content_path),
        "split_cutoff_timestamp": int(maximum)
        - fixed_data["validation_interval_seconds"],
        "mapped_catalog_sha256": _mapped_catalog_sha256(remap_path),
        "next_item_target_fixture_sha256": next_item_target_fixture_sha256(),
    }
    _validate_data_identity(result)
    return result


def expected_control_source_paths() -> list[str]:
    paths = list(_CONTROL_SOURCE_PATHS)
    if paths != sorted(set(paths)):
        raise RuntimeError("G4 control source closure is not canonical")
    return paths


_FILE_IDENTITY_KEYS = {"path", "size", "mtime_ns", "sha256"}
_DATA_IDENTITY_KEYS = {
    "dataset_key",
    "main",
    "remap",
    "content_embeddings",
    "split_cutoff_timestamp",
    "mapped_catalog_sha256",
    "next_item_target_fixture_sha256",
}


def _validate_data_identity(data: dict[str, Any]) -> None:
    if not isinstance(data, dict) or set(data) != _DATA_IDENTITY_KEYS:
        raise ValueError("data identity has missing or unknown fields")
    if not isinstance(data["dataset_key"], str) or not data["dataset_key"]:
        raise ValueError("dataset_key must be a nonempty string")
    for name in ("mapped_catalog_sha256", "next_item_target_fixture_sha256"):
        _validate_sha256_text(name, data[name])
    cutoff = data["split_cutoff_timestamp"]
    if isinstance(cutoff, bool) or not isinstance(cutoff, int):
        raise ValueError("split cutoff timestamp must be an integer")
    for name in ("main", "remap", "content_embeddings"):
        identity = data[name]
        if not isinstance(identity, dict) or set(identity) != _FILE_IDENTITY_KEYS:
            raise ValueError(f"{name} file identity has missing or unknown fields")
        if not isinstance(identity["path"], str) or not identity["path"]:
            raise ValueError(f"{name} path is invalid")
        for numeric in ("size", "mtime_ns"):
            if (
                isinstance(identity[numeric], bool)
                or not isinstance(identity[numeric], int)
                or identity[numeric] < 0
            ):
                raise ValueError(f"{name} {numeric} is invalid")
        _validate_sha256_text(f"{name} sha256", identity["sha256"])


def build_control_semantics_manifest(
    *,
    source_paths: list[str],
    sources: dict[str, str],
    data_identity: dict[str, Any],
    training_semantics_revisions: dict[str, int],
) -> dict[str, Any]:
    expected_paths = expected_control_source_paths()
    if source_paths != expected_paths or list(sources) != expected_paths:
        raise ValueError("control sources must exactly follow the approved allowlist")
    for path, digest in sources.items():
        _validate_sha256_text(f"source hash for {path}", digest)
    if source_manifest(PROJECT_ROOT, source_paths) != sources:
        raise ValueError("control source hashes differ from current sources")
    _validate_data_identity(data_identity)
    if _current_control_data_identity(data_identity) != data_identity:
        raise ValueError("control data identity differs from current data")
    if not training_semantics_revisions or any(
        not isinstance(name, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        for name, value in training_semantics_revisions.items()
    ):
        raise ValueError("training semantic revisions are invalid")
    from experiments.g4_future_items.configs.control import (
        build_anchor_control,
        control_runtime_projection,
    )

    resolved = control_runtime_projection(build_anchor_control())
    return {
        "version": 1,
        "kind": "g4_control_semantics",
        "control_manifest_sha256": APPROVED_CONTROL_MANIFEST_SHA256,
        "source_paths": source_paths,
        "sources": sources,
        "data_identity": data_identity,
        "training_semantics_revisions": dict(
            sorted(training_semantics_revisions.items())
        ),
        "resolved_anchor_configuration": resolved,
    }


_G4_ENTRYPOINTS = {
    "experiments/g4_future_items/launchers/run_control.py",
    "experiments/g4_future_items/launchers/run_selectors.py",
    "experiments/g4_future_items/launchers/run_treatments.py",
}

_G4_COMPATIBILITY_SCOPE = "frozen-g4-native50m-runtime-projections-v9"
_G4_COMPATIBILITY_EVIDENCE_VERSION = "g4-runtime-equivalence-v9"
_G4_COMPATIBILITY_LEDGER_PATHS = (
    "experiments/g4_future_items/protocol/ledgers/control_tuning.json",
    "experiments/g4_future_items/protocol/ledgers/control_tuning_boundary_r1.json",
    "experiments/g4_future_items/protocol/ledgers/rq1_tuning.json",
    "experiments/g4_future_items/protocol/ledgers/rq2_tuning.json",
)
_G4_COMPATIBILITY_LEDGER_IDENTITIES = {
    _G4_COMPATIBILITY_LEDGER_PATHS[0]: {
        "stage": "control_tuning",
        "row_count": 20,
        "ledger_sha256": (
            "a6aa821a9344321edf70353a8479cb7ae985d57b2e7bbfc27c513e9582db0e99"
        ),
        "canonical_sha256": (
            "5ab52bb3e84f1470e0852316bb1e103d3deaf0a047945325b45b2626bda37517"
        ),
    },
    _G4_COMPATIBILITY_LEDGER_PATHS[1]: {
        "stage": "control_tuning_boundary",
        "row_count": 4,
        "ledger_sha256": (
            "54b02f193fb96cdd05d0fc3266dae97f7db4dd6e126c99bb571f423d6334cd92"
        ),
        "canonical_sha256": (
            "b85d9ab38cbe9d41b54d7710ddae08ae99cbbeeae75fc1f7b0bc0eaaad2358cc"
        ),
    },
    _G4_COMPATIBILITY_LEDGER_PATHS[2]: {
        "stage": "rq1_tuning",
        "row_count": 12,
        "ledger_sha256": (
            "152c6bad2ab79b850b1e782b9ff0622d843341b60f54f6fe568d1c197e76a63c"
        ),
        "canonical_sha256": (
            "01481d5a93a722728d30e1aa5c32208b62a820a812e58411f8a1d19f3f786647"
        ),
    },
    _G4_COMPATIBILITY_LEDGER_PATHS[3]: {
        "stage": "rq2_tuning",
        "row_count": 12,
        "ledger_sha256": (
            "4842d7c373e5261574dd728bff188dd60ad222def021fc765182d61950c6e4eb"
        ),
        "canonical_sha256": (
            "b235a6db78e3cff0f8f1835e475f2645b5b61faa67a62581f8eded8cfd4bdece"
        ),
    },
}
_G4_COMPATIBILITY_CLOSURE_ONLY_PATHS = {
    "dcn/config/generation.py",
    "dcn/config/semantic.py",
    "dcn/config/semantic_history.py",
    "dcn/eval/callback.py",
    "dcn/eval/true_metric.py",
    "dcn/models/history_tokens.py",
    "dcn/models/sequence_retrieval.py",
    "dcn/nn/__init__.py",
    "dcn/nn/precomputed_embeddings.py",
    "dcn/nn/sampled_softmax.py",
    "dcn/semantic/__init__.py",
    "dcn/semantic/residual_kmeans.py",
    "experiments/g4_future_items/configs/control.py",
    "experiments/g4_future_items/launchers/run_selectors.py",
}
_G4_COMPATIBILITY_CLOSURE_ONLY_AFTER_SHA256 = {
    "dcn/config/generation.py": (
        "1db6abc44312ba8000750a707a5294c79283701ac69912c12b9467243b9e532c"
    ),
    "dcn/config/semantic.py": (
        "528f586d57dac3e41cc228768abbbdfd6a8480266b365c0bd0083409c224ff6f"
    ),
    "dcn/config/semantic_history.py": (
        "25d8bd66e2a90bcba9743324aa9739a19aa82bf944cae6858d8cf83624f18b52"
    ),
    "dcn/eval/callback.py": (
        "c5454167a4849f51c7033b5948c2443a2b6791d21f83eeba69547514ed6d0cf9"
    ),
    "dcn/eval/true_metric.py": (
        "ec358597797989512c944e2c6f3d644a8527f0a0ff05e4cbf45c48c1fe0b0889"
    ),
    "dcn/models/history_tokens.py": (
        "7b336d4a1a19dde9bc184b01001d792377e21f8caf9dd22fc3667845b6e7fc18"
    ),
    "dcn/models/sequence_retrieval.py": (
        "bd58c143a488a97cb9123c0bc31af84a9cdd174efc58bd76aaf4c33ec2008f71"
    ),
    "dcn/nn/__init__.py": (
        "f08cb0b14a25203af26d5b2971f0acb93d6b6d78c6f915b66fed0e3665829e84"
    ),
    "dcn/nn/precomputed_embeddings.py": (
        "65cd7a0c865786f17ba61b0eb4ceaff168cada3a9e556fd76b55a2f6322ff7c1"
    ),
    "dcn/nn/sampled_softmax.py": (
        "5b10cebe4e49b7d52d4a4b3f4d1fdde8a6946c5a868377ec1ec25075aa07defb"
    ),
    "dcn/semantic/__init__.py": (
        "0135abe0329eb19bc9bb5ab49c1d1a8ffd253b55e32eb5b6cf2e5c446ef9640d"
    ),
    "dcn/semantic/residual_kmeans.py": (
        "365774bbc199f3775372ca6edc11e39ebcf0a9dde3befc6ade2979c4c7935844"
    ),
    "experiments/g4_future_items/configs/control.py": (
        "cdd9d9d130843d94c344b00644df5ac647b76e9a70896e6c5e1e640e6799f4ca"
    ),
    "experiments/g4_future_items/launchers/run_selectors.py": (
        "38be93005ffc010cb1534fcf998ed0ebfdac65a92838e875c48125eb13873f9e"
    ),
}
_G4_COMPATIBILITY_OPTIMIZATION_PATHS = {
    "experiments/g4_future_items/selectors.py",
}
_G4_COMPATIBILITY_OPTIMIZATION_AFTER_SHA256 = {
    "experiments/g4_future_items/selectors.py": (
        "6d7aa41e2bd89f772cc5a57f4e1de462e667f804b5cf86a51b276b18c860487b"
    ),
}
_G4_COMPATIBILITY_PROTOCOL_PATHS = {
    "experiments/g4_future_items/protocol/manifest.py",
    "experiments/g4_future_items/protocol/manifest_contract.md",
}
_G4_COMPATIBILITY_SOURCE_ADDITIONS = {
    "dcn/nn/item_feature_encoders.py": (
        "c38afd54d584441b140fb1d3c2d0b8de95c011ea8d76e8f2da76abd02dc70790"
    ),
    "experiments/g4_future_items/report/__init__.py": (
        "c11848af237aee92649d8df1ff87052092a80a540e5aeb83bc5f905fa554c972"
    ),
    "experiments/g4_future_items/report/artifacts.py": (
        "86ce579c838909a936cf05b2855eb6cc61cb7b36ad477d8b26ae52afebcf8441"
    ),
    "experiments/g4_future_items/report/evaluation.py": (
        "be0656901b1356330be66dfd90e8bc24eaf85bb54ced3b852010fcec7e46e630"
    ),
    "experiments/g4_future_items/report/selection.py": (
        "202b00ce4d1a03d4e074797da8952f17314bf3ac3d6495bbe7d28b57bb7158a9"
    ),
    "experiments/g4_future_items/report/slices.py": (
        "d357a0fbe457962779cf0df75196f10ea88c872ada53b3b2f971af8cf69c792e"
    ),
    "utils/training_queue/gpu_check.py": (
        "c284a77b7688a69eae25f5b814f310dbc91178fc22dcabdf53269ad02f07351e"
    ),
}


def _validate_compatibility_source_additions(
    source_additions: list[dict[str, str]],
) -> None:
    expected = [
        {"path": path, "after_sha256": sha256}
        for path, sha256 in sorted(_G4_COMPATIBILITY_SOURCE_ADDITIONS.items())
    ]
    if source_additions != expected:
        raise ValueError("compatibility source additions are not approved")


def load_preimplementation_source_manifest(
    path: Path = PREIMPLEMENTATION_SOURCE_MANIFEST_PATH,
) -> dict[str, str | None]:
    document = load_strict_json(path)
    if (
        set(document) != {"canonicalization", "paths"}
        or document["canonicalization"] != "cpython-3.12-json-v1"
    ):
        raise ValueError("preimplementation source manifest schema differs")
    paths = document["paths"]
    if not isinstance(paths, dict) or list(paths) != sorted(paths):
        raise ValueError("preimplementation source paths are not canonical")
    for relative, digest in paths.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("preimplementation source path is invalid")
        if digest is not None:
            _validate_sha256_text(f"preimplementation hash for {relative}", digest)
    return cast(dict[str, str | None], paths)


def build_treatment_semantics_manifest(
    *,
    selected_control_manifest_sha256: str,
    entrypoint_source_paths: dict[str, list[str]],
    post_review_sources: dict[str, str],
    schema_revisions: dict[str, str | int],
    fixture_paths: dict[str, str],
    control_source_paths: list[str] | None = None,
    project_root: Path = PROJECT_ROOT,
    preimplementation_path: Path = PREIMPLEMENTATION_SOURCE_MANIFEST_PATH,
) -> dict[str, Any]:
    _validate_sha256_text(
        "selected_control_manifest_sha256", selected_control_manifest_sha256
    )
    if set(entrypoint_source_paths) != _G4_ENTRYPOINTS:
        raise ValueError("treatment entrypoint source maps are incomplete")
    for entrypoint, paths in entrypoint_source_paths.items():
        if (
            not isinstance(paths, list)
            or not all(isinstance(path, str) for path in paths)
            or paths != sorted(set(paths))
            or entrypoint not in paths
        ):
            raise ValueError(f"source closure for {entrypoint} is not canonical")
    source_paths = sorted(
        {path for paths in entrypoint_source_paths.values() for path in paths}
    )
    if list(post_review_sources) != source_paths:
        raise ValueError("post-review sources differ from the imported closure")
    for relative, digest in post_review_sources.items():
        _validate_sha256_text(f"post-review hash for {relative}", digest)
    actual_sources = source_manifest(project_root, source_paths)
    if post_review_sources != actual_sources:
        raise ValueError("post-review source hashes differ from current sources")

    before = load_preimplementation_source_manifest(preimplementation_path)
    approved_control = (
        expected_control_source_paths()
        if control_source_paths is None
        else control_source_paths
    )
    if approved_control != sorted(set(approved_control)):
        raise ValueError("control source closure is not canonical")
    if not set(approved_control).issubset(source_paths):
        raise ValueError("treatment closure does not contain the control closure")
    unknown = set(source_paths) - set(approved_control) - set(before)
    if unknown:
        raise ValueError(
            f"treatment closure contains unapproved paths: {sorted(unknown)}"
        )
    changed_paths = [
        {
            "path": relative,
            "before_sha256": before[relative],
            "after_sha256": post_review_sources[relative],
        }
        for relative in sorted(set(before) & set(source_paths))
        if before[relative] != post_review_sources[relative]
    ]
    if not schema_revisions or list(schema_revisions) != sorted(schema_revisions):
        raise ValueError("schema revisions must be a nonempty sorted map")
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not isinstance(value, (str, int))
        for name, value in schema_revisions.items()
    ):
        raise ValueError("schema revisions are invalid")
    if not fixture_paths or list(fixture_paths) != sorted(fixture_paths):
        raise ValueError("fixture paths must be a nonempty sorted map")
    fixtures = {}
    for name, relative in fixture_paths.items():
        if not isinstance(name, str) or not name or not isinstance(relative, str):
            raise ValueError(f"fixture {name!r} path is invalid")
        path = (project_root / relative).resolve()
        if not path.is_relative_to(project_root.resolve()) or not path.is_file():
            raise ValueError(f"fixture {name!r} path is invalid")
        fixtures[name] = {"path": relative, "sha256": _file_sha256(path)}
    preimplementation_document = load_strict_json(preimplementation_path)
    return {
        "version": 1,
        "kind": "g4_treatment_semantics",
        "selected_control_manifest_sha256": selected_control_manifest_sha256,
        "preimplementation_source_manifest_sha256": canonical_sha256(
            preimplementation_document
        ),
        "source_paths": source_paths,
        "sources": post_review_sources,
        "entrypoint_source_paths": entrypoint_source_paths,
        "changed_paths": changed_paths,
        "schema_revisions": schema_revisions,
        "fixtures": fixtures,
    }


def _project_manifest_identity(path: Path, project_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    root = project_root.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("historical semantics manifest path is invalid")
    document = load_strict_json(resolved)
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": canonical_sha256(document),
    }


def _local_module_paths(module: str, project_root: Path) -> list[Path]:
    base = project_root / module.replace(".", "/")
    paths = []
    module_path = base.with_suffix(".py")
    package_path = base / "__init__.py"
    if module_path.is_file():
        paths.append(module_path)
    if package_path.is_file():
        paths.append(package_path)
    return paths


def _parent_package_paths(path: Path, project_root: Path) -> list[Path]:
    relative = path.relative_to(project_root)
    parents = []
    parent = relative.parent
    while parent.parts:
        candidate = project_root / parent / "__init__.py"
        if candidate.is_file():
            parents.append(candidate)
        parent = parent.parent
    return parents


def derive_current_entrypoint_source_paths(
    historical_entrypoints: dict[str, list[str]],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, list[str]]:
    result = {}
    root = project_root.resolve()
    for entrypoint, historical_paths in historical_entrypoints.items():
        closure = set(historical_paths)
        pending = [
            root / relative for relative in historical_paths if relative.endswith(".py")
        ]
        parsed: set[str] = set()
        while pending:
            path = pending.pop()
            relative = path.relative_to(root).as_posix()
            if relative in parsed:
                continue
            if not path.is_file():
                raise ValueError(f"current source closure path is missing: {relative}")
            parsed.add(relative)
            closure.add(relative)
            for parent in _parent_package_paths(path, root):
                parent_relative = parent.relative_to(root).as_posix()
                closure.add(parent_relative)
                if parent_relative not in parsed:
                    pending.append(parent)
            tree = ast.parse(path.read_text(), filename=str(path))
            module_parts = list(path.relative_to(root).with_suffix("").parts)
            package_parts = module_parts[:-1]
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        keep = len(package_parts) - node.level + 1
                        base = package_parts[: max(0, keep)]
                        if node.module:
                            base = [*base, *node.module.split(".")]
                    else:
                        base = node.module.split(".") if node.module else []
                    if base:
                        modules.append(".".join(base))
                    modules.extend(
                        ".".join([*base, alias.name])
                        for alias in node.names
                        if alias.name != "*"
                    )
                for module in modules:
                    for imported in _local_module_paths(module, root):
                        imported_relative = imported.relative_to(root).as_posix()
                        closure.add(imported_relative)
                        if imported_relative not in parsed:
                            pending.append(imported)
        result[entrypoint] = sorted(closure)
    return result


def build_treatment_compatibility_manifest(
    *,
    predecessor_treatment_path: Path,
    selected_control_path: Path,
    control_semantics_path: Path,
    compatibility_evidence_path: Path,
    approved_source_changes: list[str],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    predecessor = load_strict_json(predecessor_treatment_path)
    selected = load_strict_json(selected_control_path)
    control = load_strict_json(control_semantics_path)
    _validate_treatment_semantics_document(predecessor)
    _validate_selected_control_document(selected)
    _validate_control_semantics_document(control)
    predecessor_identity = _project_manifest_identity(
        predecessor_treatment_path, project_root
    )
    selected_identity = _project_manifest_identity(selected_control_path, project_root)
    control_identity = _project_manifest_identity(control_semantics_path, project_root)
    if predecessor.get("version") != 1:
        raise ValueError("compatibility predecessor must be treatment semantics v1")
    if (
        predecessor["selected_control_manifest_sha256"] != selected_identity["sha256"]
        or selected["control_semantics_manifest_sha256"] != control_identity["sha256"]
    ):
        raise ValueError("historical semantics lineage differs")
    if approved_source_changes != sorted(set(approved_source_changes)):
        raise ValueError("approved compatibility source changes are not canonical")
    current_entrypoints = derive_current_entrypoint_source_paths(
        predecessor["entrypoint_source_paths"], project_root=project_root
    )
    current_paths = sorted(
        {path for paths in current_entrypoints.values() for path in paths}
    )
    current_sources = source_manifest(project_root, current_paths)
    source_changes = [
        {
            "path": relative,
            "before_sha256": predecessor["sources"][relative],
            "after_sha256": current_sources[relative],
        }
        for relative in predecessor["source_paths"]
        if predecessor["sources"][relative] != current_sources[relative]
    ]
    if [change["path"] for change in source_changes] != approved_source_changes:
        raise ValueError(
            "current source changes differ from the approved compatibility set"
        )
    source_additions = [
        {"path": relative, "after_sha256": current_sources[relative]}
        for relative in sorted(set(current_paths) - set(predecessor["source_paths"]))
    ]
    _validate_compatibility_source_additions(source_additions)
    evidence_identity = _project_manifest_identity(
        compatibility_evidence_path, project_root
    )
    evidence = load_strict_json(compatibility_evidence_path)
    _validate_compatibility_evidence(
        evidence,
        predecessor_treatment_sha256=predecessor_identity["sha256"],
        source_changes=source_changes,
        project_root=project_root,
    )
    return {
        "version": 2,
        "kind": "g4_treatment_semantics",
        "selected_control_manifest_sha256": selected_identity["sha256"],
        "historical_lineage": {
            "control_semantics": control_identity,
            "selected_control": selected_identity,
            "treatment_semantics": predecessor_identity,
        },
        "source_paths": current_paths,
        "sources": current_sources,
        "entrypoint_source_paths": current_entrypoints,
        "source_changes": source_changes,
        "source_additions": source_additions,
        "schema_revisions": predecessor["schema_revisions"],
        "fixtures": predecessor["fixtures"],
        "compatibility": {
            "scope": _G4_COMPATIBILITY_SCOPE,
            "approved_source_changes": approved_source_changes,
            "evidence": evidence_identity,
        },
    }


def _control_candidate_set(
    *,
    control_semantics_manifest_sha256: str,
    ledger_paths: list[Path],
    run_directories: list[Path],
) -> tuple[Any, Path, Path, list[Path], list[Path]]:
    from experiments.g4_future_items.report.artifacts import read_recommender_trial
    from experiments.g4_future_items.report.selection import select_recommender_trial

    resolved_ledgers = [path.resolve() for path in ledger_paths]
    resolved_runs = [path.resolve() for path in run_directories]
    if len(set(resolved_ledgers)) != len(resolved_ledgers) or len(
        set(resolved_runs)
    ) != len(resolved_runs):
        raise ValueError("selected control candidate paths must be unique")
    ledgers = [load_ledger(path) for path in resolved_ledgers]
    if not ledgers or ledgers[0]["stage"] != "control_tuning":
        raise ValueError("selected control requires one base control ledger first")
    if any(
        ledger["control_semantics_manifest_sha256"] != control_semantics_manifest_sha256
        for ledger in ledgers
    ):
        raise ValueError("selected control references different control semantics")
    contracts: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for run_directory in resolved_runs:
        contract = load_strict_json(run_directory / "g4_job.json")
        if set(contract) != {
            "ledger_sha256",
            "row_id",
            "job",
            "ledger_path",
            "ledger_stage",
        }:
            raise ValueError("selected control job contract schema differs")
        key = (contract["ledger_sha256"], contract["row_id"])
        if key in contracts:
            raise ValueError("selected control contains duplicate run evidence")
        contracts[key] = (run_directory, contract)
    trials: dict[str, tuple[Any, Path, Path, dict[str, Any]]] = {}
    for ledger_path, ledger in zip(resolved_ledgers, ledgers):
        for row in ledger["rows"]:
            key = (ledger["sha256"], row["id"])
            candidate = contracts.pop(key, None)
            if candidate is None:
                raise ValueError("selected control candidate set is incomplete")
            run_directory, contract = candidate
            if (
                Path(contract["ledger_path"]).resolve() != ledger_path
                or contract["ledger_stage"] != ledger["stage"]
                or contract["job"] != row["job"]
            ):
                raise ValueError("selected control job differs from its frozen row")
            trial = read_recommender_trial(run_directory)
            if not trial.usable:
                raise ValueError("selected control candidate is not usable")
            trials[row["id"]] = (trial, ledger_path, run_directory, row)
    if contracts:
        raise ValueError("selected control candidate set contains extra runs")

    base_trials = [trials[row["id"]][0] for row in ledgers[0]["rows"]]
    winner = select_recommender_trial(base_trials, objective="control")
    cumulative = [winner]
    entering_row = trials[winner.row_id][3]
    prior_rate_bounds: dict[str, list[float]] | None = None
    prior_horizons: list[int] | None = None
    consumed = 1
    for boundary_round in (1, 2):
        try:
            predecessor_paths = resolved_ledgers[:consumed]
            predecessor_hashes = {ledger["sha256"] for ledger in ledgers[:consumed]}
            predecessor_runs = [
                path
                for path in resolved_runs
                if load_strict_json(path / "g4_job.json")["ledger_sha256"]
                in predecessor_hashes
            ]
            expected = compile_recommender_boundary_ledger(
                predecessor_ledger_paths=predecessor_paths,
                candidate_run_directories=predecessor_runs,
                control_semantics_manifest_sha256=(control_semantics_manifest_sha256),
                boundary_round=boundary_round,
            )
        except ValueError as error:
            if "does not trigger" not in str(error):
                raise
            break
        if consumed >= len(ledgers) or ledgers[consumed] != expected:
            raise ValueError("selected control boundary candidate set is incomplete")
        round_trials = [trials[row["id"]][0] for row in expected["rows"]]
        cumulative.extend(round_trials)
        winner = select_recommender_trial(cumulative, objective="control")
        entering_row = trials[winner.row_id][3]
        prior_rate_bounds = expected["rate_bounds"]
        prior_horizons = expected["horizon_values"]
        consumed += 1
    if consumed != len(ledgers):
        raise ValueError("selected control contains an unapproved boundary ledger")
    if consumed == 3:
        job = entering_row["job"]
        unresolved_rate = any(
            _boundary_side(job[name], tuple(prior_rate_bounds[name])) is not None
            for name in ("embedding_learning_rate", "deep_learning_rate")
        )
        horizon = job["lr_schedule_horizon_epochs"]
        unresolved_horizon = (
            tuple(prior_horizons) == LOWER_ROUND_TWO_HORIZON_VALUES
            and horizon in {1, 2, 3}
        ) or (
            tuple(prior_horizons) == UPPER_ROUND_TWO_HORIZON_VALUES
            and horizon in {46, 47, 48, 49, 50}
        )
        if unresolved_rate or unresolved_horizon:
            raise ValueError("selected control surface remains unresolved")
    selected = trials[winner.row_id]
    return selected[0], selected[1], selected[2], resolved_ledgers, resolved_runs


def build_selected_control_manifest(
    *,
    control_semantics_manifest_sha256: str,
    ledger_paths: list[Path],
    run_directories: list[Path],
) -> dict[str, Any]:
    _validate_sha256_text(
        "control_semantics_manifest_sha256", control_semantics_manifest_sha256
    )
    trial, ledger_path, run_directory, ledger_paths, run_directories = (
        _control_candidate_set(
            control_semantics_manifest_sha256=control_semantics_manifest_sha256,
            ledger_paths=ledger_paths,
            run_directories=run_directories,
        )
    )
    ledger = load_ledger(ledger_path)
    contract_path = run_directory / "g4_job.json"
    metadata_path = run_directory / "training_metadata.json"
    log_path = run_directory / "sweep.log"
    contract = load_strict_json(contract_path)
    if set(contract) != {
        "ledger_sha256",
        "row_id",
        "job",
        "ledger_path",
        "ledger_stage",
    }:
        raise ValueError("selected control job contract schema differs")
    if (
        contract["ledger_sha256"] != ledger["sha256"]
        or Path(contract["ledger_path"]).resolve() != ledger_path
        or contract["ledger_stage"] != ledger["stage"]
    ):
        raise ValueError("selected control job contract differs from its ledger")
    row = resolve_ledger_row(ledger_path, contract["row_id"])
    if contract["job"] != row["job"]:
        raise ValueError("selected control job differs from its frozen ledger row")
    metadata = load_strict_json(metadata_path)
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= trial.horizon_epochs:
        raise ValueError("selected control best epoch differs")
    parameters = trial.parameters
    batch_size = parameters["batch_size"]
    embedding_learning_rate = parameters["embedding_learning_rate"]
    deep_learning_rate = parameters["deep_learning_rate"]
    lr_schedule_horizon_epochs = trial.horizon_epochs
    _finite_rate(
        "embedding_learning_rate",
        embedding_learning_rate,
        EMBEDDING_LR_SELECTION_BOUNDS,
    )
    _finite_rate("deep_learning_rate", deep_learning_rate, DEEP_LR_SELECTION_BOUNDS)
    _approved_selected_horizon(lr_schedule_horizon_epochs)
    from experiments.g4_future_items.configs.control import (
        build_control,
        control_runtime_projection,
    )

    run_name = trial.run_name
    experiment = build_control(
        run_name=run_name,
        seed=42,
        batch_size=batch_size,
        embedding_learning_rate=embedding_learning_rate,
        deep_learning_rate=deep_learning_rate,
        lr_schedule_horizon_epochs=lr_schedule_horizon_epochs,
    )
    seed_configuration = {
        "run_name": run_name,
        "seed": 42,
        **control_runtime_projection(experiment),
    }
    return {
        "version": 2,
        "kind": "g4_selected_control",
        "control_semantics_manifest_sha256": control_semantics_manifest_sha256,
        "ledger_sha256": ledger["sha256"],
        "selection": {
            "row_id": trial.row_id,
            "run_name": run_name,
            "validation_recall_at_100": trial.validation_recall_at_100,
            "validation_loss": trial.validation_loss,
            "best_epoch": best_epoch,
            "epochs_trained": trial.epochs_trained,
            "canonical_parameters": {
                "batch_size": batch_size,
                "embedding_learning_rate": embedding_learning_rate,
                "deep_learning_rate": deep_learning_rate,
                "lr_schedule_horizon_epochs": lr_schedule_horizon_epochs,
            },
        },
        "seed_42_configuration": seed_configuration,
        "seed_42_configuration_sha256": canonical_sha256(seed_configuration),
        "evidence": {
            "ledgers": [_file_identity(path) for path in ledger_paths],
            "runs": [
                {
                    "job_contract": _file_identity(path / "g4_job.json"),
                    "training_metadata": _file_identity(
                        path / "training_metadata.json"
                    ),
                    "sweep_log": _file_identity(path / "sweep.log"),
                }
                for path in run_directories
            ],
        },
    }


def write_frozen_manifest(path: Path, document: dict[str, Any]) -> None:
    content = canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"frozen manifest differs: {path}")
        return
    path.write_bytes(content)
    path.chmod(0o444)


def _resolved_job(
    *,
    run_name: str,
    stage: str,
    seed: int,
    batch_size: int,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    lr_schedule_horizon_epochs: int,
    trial_id: int | None = None,
    objective: dict[str, Any] | None = None,
    loss: dict[str, Any] | None = None,
    boundary_round: int | None = None,
) -> dict[str, Any]:
    protocol: dict[str, Any] = {"stage": stage}
    if trial_id is not None:
        protocol["trial_id"] = trial_id
    if boundary_round is not None:
        protocol["boundary_round"] = boundary_round
    job: dict[str, Any] = {
        "run_name": run_name,
        "protocol": protocol,
        "dataloader": {"batch_size": batch_size},
        "embedding_learning_rate": embedding_learning_rate,
        "deep_learning_rate": deep_learning_rate,
        "lr_schedule_horizon_epochs": lr_schedule_horizon_epochs,
        "seed": seed,
    }
    if objective is not None:
        job["objective"] = objective
    if loss is not None:
        job["loss"] = loss
    return job


def _sample_recommender_trials(
    count: int,
    *,
    seed: int,
    anchor: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=seed)
    )
    if anchor is not None:
        study.enqueue_trial(anchor)
    result: list[dict[str, Any]] = []
    for _ in range(count):
        trial = study.ask()
        parameters: dict[str, Any] = {}
        parameters["embedding_learning_rate"] = trial.suggest_float(
            "embedding_learning_rate", *EMBEDDING_LR_BOUNDS, log=True
        )
        parameters["deep_learning_rate"] = trial.suggest_float(
            "deep_learning_rate", *DEEP_LR_BOUNDS, log=True
        )
        parameters["lr_schedule_horizon_epochs"] = trial.suggest_categorical(
            "lr_schedule_horizon_epochs", list(BASE_HORIZON_VALUES)
        )
        study.tell(trial, 0.0)
        result.append(parameters)
    return result


def _seal_ledger(document: dict[str, Any]) -> dict[str, Any]:
    if "sha256" in document:
        raise ValueError("unsealed ledger must not contain sha256")
    return document | {"sha256": canonical_sha256(document)}


def compile_control_tuning_ledger(
    control_semantics_manifest_sha256: str,
) -> dict[str, Any]:
    anchor = load_control_manifest()["anchor"]
    anchor_parameters = {
        "embedding_learning_rate": anchor["embedding_learning_rate"],
        "deep_learning_rate": anchor["deep_learning_rate"],
        "lr_schedule_horizon_epochs": anchor["lr_schedule_horizon_epochs"],
    }
    trials = _sample_recommender_trials(20, seed=42, anchor=anchor_parameters)
    rows = []
    for trial_id, parameters in enumerate(trials, 1):
        rows.append(
            {
                "id": f"control_tuning:{trial_id:02d}",
                "job": _resolved_job(
                    run_name=f"g4_control_trial_{trial_id:02d}_native50m",
                    stage="control_tuning",
                    trial_id=trial_id,
                    seed=42,
                    batch_size=CONTROL_BATCH_SIZE,
                    **parameters,
                ),
            }
        )
    return _seal_ledger(
        {
            "version": LEDGER_VERSION,
            "stage": "control_tuning",
            "control_semantics_manifest_sha256": (control_semantics_manifest_sha256),
            "rows": rows,
        }
    )


_OBJECTIVE_FIELDS: dict[ObjectiveId, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "rq1_24h": (
        "rq1_tuning",
        {"id": "rq1_24h", "window_seconds": 86400},
        {"valid_positive_mask_mode": "next_24h_unique"},
    ),
    "rq2_next10": (
        "rq2_tuning",
        {"id": "rq2_next10", "event_lookahead": 10},
        {"valid_positive_mask_mode": "next_10_unique"},
    ),
    "rq3_deterministic_hard": (
        "rq3_deterministic_tuning",
        {"id": "rq3_deterministic_hard"},
        {"valid_positive_mask_mode": "selected_period_union_unique"},
    ),
    "rq3_learned_hard": (
        "rq3_learned_hard_tuning",
        {"id": "rq3_learned_hard"},
        {"valid_positive_mask_mode": "selected_period_union_unique"},
    ),
    "rq3_learned_proportional": (
        "rq3_learned_proportional_tuning",
        {"id": "rq3_learned_proportional"},
        {"valid_positive_mask_mode": "all_positive_probability_periods_unique"},
    ),
}


def compile_treatment_tuning_ledger(
    *,
    objective_id: ObjectiveId,
    selected_control_manifest_sha256: str,
    treatment_semantics_manifest_sha256: str,
    batch_size: int,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    lr_schedule_horizon_epochs: int,
    selector_artifact_sha256: str | None = None,
    materialization_cost_evidence: dict[str, Any] | None = None,
    _materialization_cost_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    if objective_id not in _OBJECTIVE_FIELDS:
        raise ValueError(f"objective {objective_id!r} is not approved")
    if batch_size != CONTROL_BATCH_SIZE or isinstance(batch_size, bool):
        raise ValueError("G4 treatment batch must be 512")
    _approved_selected_horizon(lr_schedule_horizon_epochs)
    stage, objective_template, loss = _OBJECTIVE_FIELDS[objective_id]
    if objective_id.startswith("rq3_") and not selector_artifact_sha256:
        raise ValueError("RQ3 objective requires selector artifact SHA-256")
    materialization_evidence_sha256 = None
    if objective_id.startswith("rq3_"):
        if (
            materialization_cost_evidence is None
            and _materialization_cost_evidence_sha256 is None
        ):
            raise ValueError("RQ3 objective requires passing materialization evidence")
        if materialization_cost_evidence is not None:
            _validate_materialization_cost_evidence(materialization_cost_evidence)
            expected_artifact = materialization_cost_evidence[
                (
                    "deterministic_artifact_sha256"
                    if objective_id == "rq3_deterministic_hard"
                    else "learned_artifact_sha256"
                )
            ]
            if selector_artifact_sha256 != expected_artifact:
                raise ValueError("RQ3 selector artifact differs from measured artifact")
            materialization_evidence_sha256 = canonical_sha256(
                materialization_cost_evidence
            )
        else:
            _validate_sha256_text(
                "materialization_cost_evidence_sha256",
                _materialization_cost_evidence_sha256,
            )
            materialization_evidence_sha256 = _materialization_cost_evidence_sha256
    elif materialization_cost_evidence is not None:
        raise ValueError("materialization evidence applies only to RQ3")
    anchor = {
        "embedding_learning_rate": embedding_learning_rate,
        "deep_learning_rate": deep_learning_rate,
        "lr_schedule_horizon_epochs": lr_schedule_horizon_epochs,
    }
    if objective_id.startswith("rq3_"):
        anchor["period_count"] = 1
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42)
    )
    study.enqueue_trial(anchor)
    trials = []
    for _ in range(12):
        trial = study.ask()
        parameters = {
            "embedding_learning_rate": trial.suggest_float(
                "embedding_learning_rate", *EMBEDDING_LR_BOUNDS, log=True
            ),
            "deep_learning_rate": trial.suggest_float(
                "deep_learning_rate", *DEEP_LR_BOUNDS, log=True
            ),
            "lr_schedule_horizon_epochs": trial.suggest_categorical(
                "lr_schedule_horizon_epochs", list(BASE_HORIZON_VALUES)
            ),
        }
        if objective_id.startswith("rq3_"):
            parameters["period_count"] = trial.suggest_categorical(
                "period_count", [1, 2, 4]
            )
        study.tell(trial, 0.0)
        trials.append(parameters)
    rows = []
    for trial_id, parameters in enumerate(trials, 1):
        objective = dict(objective_template)
        if objective_id.startswith("rq3_"):
            objective["selector_artifact_sha256"] = selector_artifact_sha256
            objective["period_count"] = parameters.pop("period_count")
        rows.append(
            {
                "id": f"{stage}:{trial_id:02d}",
                "job": _resolved_job(
                    run_name=f"g4_{objective_id}_trial_{trial_id:02d}_native50m",
                    stage=stage,
                    trial_id=trial_id,
                    seed=42,
                    batch_size=batch_size,
                    objective=objective,
                    loss=loss,
                    **parameters,
                ),
            }
        )
    root = {
        "version": LEDGER_VERSION,
        "stage": stage,
        "selected_control_manifest_sha256": selected_control_manifest_sha256,
        "treatment_semantics_manifest_sha256": (treatment_semantics_manifest_sha256),
        "anchor_parameters": anchor,
        "rows": rows,
    }
    if materialization_evidence_sha256 is not None:
        root["materialization_cost_evidence_sha256"] = materialization_evidence_sha256
    return _seal_ledger(root)


def _boundary_side(
    value: float, bounds: tuple[float, float]
) -> Literal["lower", "upper"] | None:
    lower, upper = bounds
    if not lower <= value <= upper:
        raise ValueError("selected rate is outside its entering search interval")
    position = math.log(value / lower) / math.log(upper / lower)
    if position <= 0.1:
        return "lower"
    if position >= 0.9:
        return "upper"
    return None


def _extended_bounds(
    bounds: tuple[float, float], side: Literal["lower", "upper"] | None
) -> tuple[float, float]:
    if side == "lower":
        return bounds[0] / 4, bounds[1]
    if side == "upper":
        return bounds[0], bounds[1] * 4
    return bounds


def _horizon_boundary_values(
    value: int,
    entering_values: tuple[int, ...],
    boundary_round: int,
) -> tuple[int, ...] | None:
    _positive_horizon(value)
    if value not in entering_values:
        raise ValueError("selected horizon is outside its entering search domain")
    if boundary_round == 1:
        if entering_values != BASE_HORIZON_VALUES:
            raise ValueError("round-one horizon domain differs from the base surface")
        if value == 5:
            return LOWER_ROUND_ONE_HORIZON_VALUES
        if value == 30:
            return UPPER_ROUND_ONE_HORIZON_VALUES
        return None
    if len(entering_values) == 1:
        return None
    if entering_values == LOWER_ROUND_ONE_HORIZON_VALUES:
        return LOWER_ROUND_TWO_HORIZON_VALUES if value in {2, 3, 4} else None
    if entering_values == UPPER_ROUND_ONE_HORIZON_VALUES:
        return UPPER_ROUND_TWO_HORIZON_VALUES if value in {37, 38, 39, 40} else None
    raise ValueError("round-two horizon domain differs from the approved surface")


def _compile_recommender_boundary_ledger(
    *,
    entering_row: dict[str, Any],
    boundary_round: int,
    control_semantics_manifest_sha256: str | None = None,
    selected_control_manifest_sha256: str | None = None,
    treatment_semantics_manifest_sha256: str | None = None,
    materialization_cost_evidence_sha256: str | None = None,
    entering_rate_bounds: dict[str, list[float]] | None = None,
    entering_horizon_values: list[int] | None = None,
) -> dict[str, Any]:
    if boundary_round not in {1, 2}:
        raise ValueError("boundary round must be 1 or 2")
    if not isinstance(entering_row, dict) or set(entering_row) != {"id", "job"}:
        raise ValueError("entering boundary row is invalid")
    source = entering_row["job"]
    if not isinstance(source, dict):
        raise ValueError("entering boundary job is invalid")
    source_stage = source.get("protocol", {}).get("stage")
    base_stage = (
        source_stage.removesuffix("_boundary")
        if isinstance(source_stage, str)
        else source_stage
    )
    allowed = {"control_tuning", *[value[0] for value in _OBJECTIVE_FIELDS.values()]}
    if base_stage not in allowed:
        raise ValueError("entering row is not an approved recommender study")
    prior = entering_rate_bounds or {
        "embedding_learning_rate": list(EMBEDDING_LR_BOUNDS),
        "deep_learning_rate": list(DEEP_LR_BOUNDS),
    }
    if set(prior) != {"embedding_learning_rate", "deep_learning_rate"}:
        raise ValueError("entering rate bounds are incomplete")
    embedding_bounds = tuple(prior["embedding_learning_rate"])
    deep_bounds = tuple(prior["deep_learning_rate"])
    if len(embedding_bounds) != 2 or len(deep_bounds) != 2:
        raise ValueError("entering rate bounds are invalid")
    embedding_side = _boundary_side(
        source["embedding_learning_rate"], cast(tuple[float, float], embedding_bounds)
    )
    deep_side = _boundary_side(
        source["deep_learning_rate"], cast(tuple[float, float], deep_bounds)
    )
    prior_horizons = tuple(entering_horizon_values or BASE_HORIZON_VALUES)
    horizon_values = _horizon_boundary_values(
        source["lr_schedule_horizon_epochs"], prior_horizons, boundary_round
    )
    if embedding_side is None and deep_side is None and horizon_values is None:
        raise ValueError("entering winner does not trigger a recommender boundary")
    expanded_embedding = _extended_bounds(
        cast(tuple[float, float], embedding_bounds), embedding_side
    )
    expanded_deep = _extended_bounds(cast(tuple[float, float], deep_bounds), deep_side)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=42 + boundary_round),
    )
    rows = []
    for trial_id in range(1, 5):
        trial = study.ask()
        embedding_rate = (
            trial.suggest_float(
                "embedding_learning_rate", *expanded_embedding, log=True
            )
            if embedding_side is not None
            else source["embedding_learning_rate"]
        )
        deep_rate = (
            trial.suggest_float("deep_learning_rate", *expanded_deep, log=True)
            if deep_side is not None
            else source["deep_learning_rate"]
        )
        horizon = (
            trial.suggest_categorical(
                "lr_schedule_horizon_epochs", list(horizon_values)
            )
            if horizon_values is not None
            else source["lr_schedule_horizon_epochs"]
        )
        study.tell(trial, 0.0)
        job = json.loads(json.dumps(source, allow_nan=False))
        stage = f"{base_stage}_boundary"
        job["run_name"] = (
            f"g4_{base_stage}_boundary_r{boundary_round}_"
            f"trial_{trial_id:02d}_native50m"
        )
        job["protocol"] = {
            "stage": stage,
            "trial_id": trial_id,
            "boundary_round": boundary_round,
        }
        job["embedding_learning_rate"] = embedding_rate
        job["deep_learning_rate"] = deep_rate
        job["lr_schedule_horizon_epochs"] = horizon
        rows.append(
            {
                "id": f"{stage}:{boundary_round}:{trial_id:02d}",
                "job": job,
            }
        )
    references: dict[str, str] = {}
    if base_stage == "control_tuning":
        if selected_control_manifest_sha256 or treatment_semantics_manifest_sha256:
            raise ValueError(
                "control boundary cannot reference selected treatment state"
            )
        if not control_semantics_manifest_sha256:
            raise ValueError("control boundary requires control semantics")
        references["control_semantics_manifest_sha256"] = (
            control_semantics_manifest_sha256
        )
    else:
        if (
            not selected_control_manifest_sha256
            or not treatment_semantics_manifest_sha256
        ):
            raise ValueError(
                "treatment boundary requires selected and treatment semantics"
            )
        references.update(
            {
                "selected_control_manifest_sha256": (selected_control_manifest_sha256),
                "treatment_semantics_manifest_sha256": (
                    treatment_semantics_manifest_sha256
                ),
            }
        )
        if base_stage.startswith("rq3_"):
            _validate_sha256_text(
                "materialization_cost_evidence_sha256",
                materialization_cost_evidence_sha256,
            )
            references["materialization_cost_evidence_sha256"] = cast(
                str, materialization_cost_evidence_sha256
            )
        elif materialization_cost_evidence_sha256 is not None:
            raise ValueError("materialization evidence applies only to RQ3")
    return _seal_ledger(
        {
            "version": LEDGER_VERSION,
            "stage": f"{base_stage}_boundary",
            **references,
            "entering_row_sha256": canonical_sha256(entering_row),
            "entering_row": entering_row,
            "entering_rate_bounds": {
                name: list(bounds) for name, bounds in prior.items()
            },
            "entering_horizon_values": list(prior_horizons),
            "rate_bounds": {
                "embedding_learning_rate": list(expanded_embedding),
                "deep_learning_rate": list(expanded_deep),
            },
            "horizon_values": list(
                horizon_values
                if horizon_values is not None
                else (source["lr_schedule_horizon_epochs"],)
            ),
            "rows": rows,
        }
    )


def _recommender_predecessor_winner(
    *,
    ledger_paths: list[Path],
    run_directories: list[Path],
    boundary_round: int,
) -> tuple[dict[str, Any], list[Path], list[Path], list[dict[str, Any]]]:
    from experiments.g4_future_items.report.artifacts import read_recommender_trial
    from experiments.g4_future_items.report.selection import select_recommender_trial

    resolved_ledgers = [path.resolve() for path in ledger_paths]
    resolved_runs = [path.resolve() for path in run_directories]
    if (
        boundary_round not in {1, 2}
        or len(resolved_ledgers) != boundary_round
        or len(set(resolved_ledgers)) != len(resolved_ledgers)
        or len(set(resolved_runs)) != len(resolved_runs)
    ):
        raise ValueError("boundary predecessor ledger set is incomplete")
    ledgers = [load_ledger(path) for path in resolved_ledgers]
    base_stage = ledgers[0]["stage"]
    if base_stage.endswith("_boundary") or base_stage not in {
        "control_tuning",
        *[value[0] for value in _OBJECTIVE_FIELDS.values()],
    }:
        raise ValueError("boundary predecessor base stage differs")
    for round_index, ledger in enumerate(ledgers[1:], 1):
        if ledger["stage"] != f"{base_stage}_boundary" or any(
            row["job"]["protocol"]["boundary_round"] != round_index
            for row in ledger["rows"]
        ):
            raise ValueError("boundary predecessor stage sequence differs")

    contracts: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for run_directory in resolved_runs:
        contract = load_strict_json(run_directory / "g4_job.json")
        if set(contract) != {
            "ledger_sha256",
            "row_id",
            "job",
            "ledger_path",
            "ledger_stage",
        }:
            raise ValueError("boundary candidate job contract schema differs")
        key = (contract["ledger_sha256"], contract["row_id"])
        if key in contracts:
            raise ValueError("boundary candidate evidence is duplicated")
        contracts[key] = (run_directory, contract)
    candidates: dict[str, tuple[Any, dict[str, Any]]] = {}
    for ledger_path, ledger in zip(resolved_ledgers, ledgers):
        for row in ledger["rows"]:
            candidate = contracts.pop((ledger["sha256"], row["id"]), None)
            if candidate is None:
                raise ValueError("boundary candidate evidence is incomplete")
            run_directory, contract = candidate
            if (
                Path(contract["ledger_path"]).resolve() != ledger_path
                or contract["ledger_stage"] != ledger["stage"]
                or contract["job"] != row["job"]
            ):
                raise ValueError("boundary candidate differs from its frozen row")
            trial = read_recommender_trial(run_directory)
            if not trial.usable:
                raise ValueError("boundary candidate is not usable")
            candidates[row["id"]] = (trial, row)
    if contracts:
        raise ValueError("boundary candidate evidence contains extra runs")

    base_trials = [candidates[row["id"]][0] for row in ledgers[0]["rows"]]
    winner = select_recommender_trial(base_trials, objective=base_stage)
    cumulative = [winner]
    for ledger in ledgers[1:]:
        winner_row = candidates[winner.row_id][1]
        if ledger["entering_row"] != winner_row:
            raise ValueError("boundary entering row is not the cumulative winner")
        cumulative.extend(candidates[row["id"]][0] for row in ledger["rows"])
        winner = select_recommender_trial(cumulative, objective=base_stage)
    return candidates[winner.row_id][1], resolved_ledgers, resolved_runs, ledgers


def compile_recommender_boundary_ledger(
    *,
    predecessor_ledger_paths: list[Path],
    candidate_run_directories: list[Path],
    boundary_round: int,
    control_semantics_manifest_sha256: str | None = None,
    selected_control_manifest_sha256: str | None = None,
    treatment_semantics_manifest_sha256: str | None = None,
    materialization_cost_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    entering_row, ledger_paths, run_directories, ledgers = (
        _recommender_predecessor_winner(
            ledger_paths=predecessor_ledger_paths,
            run_directories=candidate_run_directories,
            boundary_round=boundary_round,
        )
    )
    base_stage = ledgers[0]["stage"]
    if base_stage == "control_tuning":
        if any(
            ledger["control_semantics_manifest_sha256"]
            != control_semantics_manifest_sha256
            for ledger in ledgers
        ):
            raise ValueError("boundary predecessor control semantics differ")
    else:
        expected_references = {
            "selected_control_manifest_sha256": selected_control_manifest_sha256,
            "treatment_semantics_manifest_sha256": (
                treatment_semantics_manifest_sha256
            ),
        }
        if base_stage.startswith("rq3_"):
            expected_references["materialization_cost_evidence_sha256"] = (
                materialization_cost_evidence_sha256
            )
        if any(
            ledger.get(name) != value
            for ledger in ledgers
            for name, value in expected_references.items()
        ):
            raise ValueError("boundary predecessor treatment semantics differ")
    prior = ledgers[-1] if boundary_round == 2 else None
    document = _compile_recommender_boundary_ledger(
        entering_row=entering_row,
        boundary_round=boundary_round,
        control_semantics_manifest_sha256=control_semantics_manifest_sha256,
        selected_control_manifest_sha256=selected_control_manifest_sha256,
        treatment_semantics_manifest_sha256=treatment_semantics_manifest_sha256,
        materialization_cost_evidence_sha256=(materialization_cost_evidence_sha256),
        entering_rate_bounds=(None if prior is None else prior["rate_bounds"]),
        entering_horizon_values=(None if prior is None else prior["horizon_values"]),
    )
    unsigned = {key: value for key, value in document.items() if key != "sha256"}
    unsigned["predecessor_evidence"] = {
        "ledgers": [_file_identity(path) for path in ledger_paths],
        "runs": [
            {
                "job_contract": _file_identity(path / "g4_job.json"),
                "training_metadata": _file_identity(path / "training_metadata.json"),
                "sweep_log": _file_identity(path / "sweep.log"),
            }
            for path in run_directories
        ],
    }
    return _seal_ledger(unsigned)


_SELECTOR_FIELDS = {
    "stage",
    "trial_id",
    "fold_id",
    "boundary_round",
    "family",
    "period_width_seconds",
    "lookahead_seconds",
    "minimum_liked_events",
    "time_tolerance_seconds",
    "frequency_entity",
    "max_leaf_nodes",
    "learning_rate",
    "l2_regularization",
    "seed",
    "input_artifact_sha256",
    "input_payload_sha256",
    "deterministic_artifact_sha256",
    "deterministic_payload_sha256",
    "learned_artifact_sha256",
    "learned_payload_sha256",
    "output_artifact_sha256",
}


def _selector_output_identity(job: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema": "g4-selector-output-slot-v1",
            **{
                key: value
                for key, value in job.items()
                if key != "output_artifact_sha256"
            },
        }
    )


def compile_selector_search_ledger(
    *,
    trials: list[Any] | None = None,
    treatment_semantics_manifest_sha256: str,
    input_artifact_sha256: str,
) -> dict[str, Any]:
    if trials is None:
        from experiments.g4_future_items.configs.selectors import (
            compile_selector_search,
        )

        trials = list(compile_selector_search(seed=42))
    if len(trials) != 48:
        raise ValueError("selector search requires exactly 48 trial rows")
    rows = []
    for trial in trials:
        raw = trial.to_dict() if hasattr(trial, "to_dict") else dict(trial)
        raw.pop("stage", None)
        raw.pop("run_name", None)
        job = {
            "stage": "selector_search",
            "fold_id": None,
            "input_artifact_sha256": input_artifact_sha256,
            "input_payload_sha256": None,
            "deterministic_artifact_sha256": None,
            "deterministic_payload_sha256": None,
            "learned_artifact_sha256": None,
            "learned_payload_sha256": None,
            "output_artifact_sha256": "",
            **raw,
        }
        job["output_artifact_sha256"] = _selector_output_identity(job)
        family = job.get("family")
        trial_id = job.get("trial_id")
        rows.append(
            {
                "id": f"selector_search:{family}:{trial_id:02d}",
                "job": job,
            }
        )
    structural_keys = (
        "family",
        "period_width_seconds",
        "lookahead_seconds",
        "minimum_liked_events",
        "time_tolerance_seconds",
        "frequency_entity",
        "max_leaf_nodes",
    )
    for family in ("time", "content", "frequency", "learned"):
        family_jobs = [row["job"] for row in rows if row["job"]["family"] == family]
        if (
            len(family_jobs) != 12
            or len({tuple(job[key] for key in structural_keys) for job in family_jobs})
            != 12
        ):
            raise ValueError(
                "selector search requires 12 unique structural trials per family"
            )
    return _seal_ledger(
        {
            "version": LEDGER_VERSION,
            "stage": "selector_search",
            "treatment_semantics_manifest_sha256": (
                treatment_semantics_manifest_sha256
            ),
            "rows": rows,
        }
    )


def _empty_selector_job(stage: str, input_artifact_sha256: str) -> dict[str, Any]:
    job = {
        name: None
        for name in _SELECTOR_FIELDS
        if name
        not in {
            "stage",
            "seed",
            "input_artifact_sha256",
            "output_artifact_sha256",
        }
    }
    job.update(
        {
            "stage": stage,
            "seed": 42,
            "input_artifact_sha256": input_artifact_sha256,
            "output_artifact_sha256": "",
        }
    )
    job["output_artifact_sha256"] = _selector_output_identity(job)
    return job


def compile_selector_gate_ledger(
    *,
    treatment_semantics_manifest_sha256: str,
    deterministic_artifact_sha256: str,
    deterministic_payload_sha256: str,
    learned_artifact_sha256: str,
    learned_payload_sha256: str,
) -> dict[str, Any]:
    job = _empty_selector_job("selector_gate", "")
    job.update(
        {
            "input_artifact_sha256": None,
            "input_payload_sha256": None,
            "deterministic_artifact_sha256": deterministic_artifact_sha256,
            "deterministic_payload_sha256": deterministic_payload_sha256,
            "learned_artifact_sha256": learned_artifact_sha256,
            "learned_payload_sha256": learned_payload_sha256,
            "output_artifact_sha256": "",
        }
    )
    job["output_artifact_sha256"] = _selector_output_identity(job)
    return _seal_ledger(
        {
            "version": LEDGER_VERSION,
            "stage": "selector_gate",
            "treatment_semantics_manifest_sha256": (
                treatment_semantics_manifest_sha256
            ),
            "rows": [{"id": "selector_gate:01", "job": job}],
        }
    )


def compile_selector_materialization_ledger(
    *,
    treatment_semantics_manifest_sha256: str,
    selected_configuration: dict[str, Any],
    selector_gate_artifact_sha256: str,
    selector_gate_payload_sha256: str,
) -> dict[str, Any]:
    expected_selected = _SELECTOR_FIELDS - {
        "stage",
        "fold_id",
        "input_artifact_sha256",
        "input_payload_sha256",
        "deterministic_artifact_sha256",
        "deterministic_payload_sha256",
        "learned_artifact_sha256",
        "learned_payload_sha256",
        "output_artifact_sha256",
    }
    if set(selected_configuration) != expected_selected:
        raise ValueError(
            "selected selector configuration has missing or unknown fields"
        )
    rows = []
    for fold_id in range(5):
        job = {
            "stage": "selector_materialization",
            "fold_id": fold_id,
            "input_artifact_sha256": selector_gate_artifact_sha256,
            "input_payload_sha256": selector_gate_payload_sha256,
            "deterministic_artifact_sha256": None,
            "deterministic_payload_sha256": None,
            "learned_artifact_sha256": None,
            "learned_payload_sha256": None,
            "output_artifact_sha256": "",
            **selected_configuration,
        }
        job["trial_id"] = None
        job["boundary_round"] = None
        job["output_artifact_sha256"] = _selector_output_identity(job)
        rows.append(
            {
                "id": f"selector_materialization:{fold_id}",
                "job": job,
            }
        )
    return _seal_ledger(
        {
            "version": LEDGER_VERSION,
            "stage": "selector_materialization",
            "treatment_semantics_manifest_sha256": (
                treatment_semantics_manifest_sha256
            ),
            "rows": rows,
        }
    )


def _compile_selector_boundary_ledger(
    *,
    entering_row: dict[str, Any],
    treatment_semantics_manifest_sha256: str,
    boundary_round: int,
    entering_rate_bounds: list[float] | None = None,
) -> dict[str, Any]:
    if boundary_round not in {1, 2}:
        raise ValueError("selector boundary round must be 1 or 2")
    if not isinstance(entering_row, dict) or set(entering_row) != {"id", "job"}:
        raise ValueError("entering selector row is invalid")
    source = entering_row["job"]
    if (
        not isinstance(source, dict)
        or source.get("family") != "learned"
        or source.get("stage") not in {"selector_search", "selector_search_boundary"}
    ):
        raise ValueError("selector boundary requires a learned search winner")
    bounds = entering_rate_bounds or [0.01, 0.2]
    if len(bounds) != 2:
        raise ValueError("selector entering rate bounds are invalid")
    typed_bounds = cast(tuple[float, float], tuple(bounds))
    side = _boundary_side(source["learning_rate"], typed_bounds)
    if side is None:
        raise ValueError("learned selector winner does not trigger a boundary")
    expanded = _extended_bounds(typed_bounds, side)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=42 + boundary_round),
    )
    rows = []
    for trial_id in range(1, 5):
        trial = study.ask()
        job = json.loads(json.dumps(source, allow_nan=False))
        job.update(
            {
                "stage": "selector_search_boundary",
                "trial_id": trial_id,
                "boundary_round": boundary_round,
                "learning_rate": trial.suggest_float(
                    "learning_rate", *expanded, log=True
                ),
                "output_artifact_sha256": "",
            }
        )
        study.tell(trial, 0.0)
        job["output_artifact_sha256"] = _selector_output_identity(job)
        rows.append(
            {
                "id": f"selector_search_boundary:{boundary_round}:{trial_id:02d}",
                "job": job,
            }
        )
    return _seal_ledger(
        {
            "version": LEDGER_VERSION,
            "stage": "selector_search_boundary",
            "treatment_semantics_manifest_sha256": (
                treatment_semantics_manifest_sha256
            ),
            "entering_row_sha256": canonical_sha256(entering_row),
            "entering_row": entering_row,
            "entering_rate_bounds": list(bounds),
            "rate_bounds": {"learning_rate": list(expanded)},
            "rows": rows,
        }
    )


def _selector_predecessor_winner(
    *,
    ledger_paths: list[Path],
    search_root: Path,
    boundary_round: int,
) -> tuple[dict[str, Any], list[Path], Path, list[dict[str, Any]]]:
    from experiments.g4_future_items.configs.selectors import (
        select_family_winner,
        selector_trial_from_job,
    )
    from experiments.g4_future_items.launchers.run_selectors import (
        load_search_result,
    )

    resolved_ledgers = [path.resolve() for path in ledger_paths]
    search_root = search_root.resolve()
    if (
        boundary_round not in {1, 2}
        or len(resolved_ledgers) != boundary_round
        or len(set(resolved_ledgers)) != len(resolved_ledgers)
    ):
        raise ValueError("selector boundary predecessor set is incomplete")
    ledgers = [load_ledger(path) for path in resolved_ledgers]
    if ledgers[0]["stage"] != "selector_search":
        raise ValueError("selector boundary requires the base search ledger first")
    for round_index, ledger in enumerate(ledgers[1:], 1):
        if ledger["stage"] != "selector_search_boundary" or any(
            row["job"]["boundary_round"] != round_index for row in ledger["rows"]
        ):
            raise ValueError("selector boundary predecessor sequence differs")
    results: dict[str, tuple[Any, dict[str, Any]]] = {}
    for ledger in ledgers:
        for row in ledger["rows"]:
            job = row["job"]
            result = load_search_result(search_root, job["output_artifact_sha256"])
            if result.trial != selector_trial_from_job(job):
                raise ValueError("selector result differs from its frozen ledger row")
            results[row["id"]] = (result.to_trial_result(), row)
    learned = [
        results[row["id"]][0]
        for row in ledgers[0]["rows"]
        if row["job"]["family"] == "learned"
    ]
    winner = select_family_winner(learned)
    cumulative = [winner]
    for ledger in ledgers[1:]:
        winner_row = results[
            (
                f"selector_search:{winner.trial.family}:{winner.trial.trial_id:02d}"
                if winner.trial.boundary_round is None
                else (
                    f"selector_search_boundary:{winner.trial.boundary_round}:"
                    f"{winner.trial.trial_id:02d}"
                )
            )
        ][1]
        if ledger["entering_row"] != winner_row:
            raise ValueError("selector boundary entry is not the cumulative winner")
        round_results = [results[row["id"]][0] for row in ledger["rows"]]
        cumulative.extend(round_results)
        winner = select_family_winner(cumulative)
    winner_id = (
        f"selector_search:learned:{winner.trial.trial_id:02d}"
        if winner.trial.boundary_round is None
        else (
            f"selector_search_boundary:{winner.trial.boundary_round}:"
            f"{winner.trial.trial_id:02d}"
        )
    )
    return results[winner_id][1], resolved_ledgers, search_root, ledgers


def compile_selector_boundary_ledger(
    *,
    predecessor_ledger_paths: list[Path],
    search_root: Path,
    treatment_semantics_manifest_sha256: str,
    boundary_round: int,
) -> dict[str, Any]:
    entering_row, ledger_paths, search_root, ledgers = _selector_predecessor_winner(
        ledger_paths=predecessor_ledger_paths,
        search_root=search_root,
        boundary_round=boundary_round,
    )
    if any(
        ledger["treatment_semantics_manifest_sha256"]
        != treatment_semantics_manifest_sha256
        for ledger in ledgers
    ):
        raise ValueError("selector predecessor treatment semantics differ")
    prior = ledgers[-1] if boundary_round == 2 else None
    document = _compile_selector_boundary_ledger(
        entering_row=entering_row,
        treatment_semantics_manifest_sha256=treatment_semantics_manifest_sha256,
        boundary_round=boundary_round,
        entering_rate_bounds=(
            None if prior is None else prior["rate_bounds"]["learning_rate"]
        ),
    )
    unsigned = {key: value for key, value in document.items() if key != "sha256"}
    unsigned["predecessor_evidence"] = {
        "ledgers": [_file_identity(path) for path in ledger_paths],
        "results": [
            {
                "row_id": row["id"],
                "artifact_sha256": row["job"]["output_artifact_sha256"],
                "artifact": _file_identity(
                    search_root / row["job"]["output_artifact_sha256"] / "artifact.json"
                ),
                "artifact_sha256_file": _file_identity(
                    search_root
                    / row["job"]["output_artifact_sha256"]
                    / "artifact.sha256"
                ),
                "result": _file_identity(
                    search_root / row["job"]["output_artifact_sha256"] / "result.json"
                ),
            }
            for ledger in ledgers
            for row in ledger["rows"]
        ],
    }
    return _seal_ledger(unsigned)


def _validate_sha256_text(name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_materialization_cost_evidence(document: dict[str, Any]) -> None:
    required = {
        "version",
        "measurement_id",
        "passes",
        "deterministic_artifact_sha256",
        "learned_artifact_sha256",
        "runtime",
        "logical_output_scratch_bytes",
        "timed_load_valid",
        "limits",
    }
    if not isinstance(document, dict) or not required <= set(document):
        raise ValueError("materialization cost evidence is incomplete")
    if document["version"] != "g4-materialization-cost-v1":
        raise ValueError("materialization cost evidence version differs")
    for name in (
        "measurement_id",
        "deterministic_artifact_sha256",
        "learned_artifact_sha256",
    ):
        _validate_sha256_text(name, document[name])
    runtime = document["runtime"]
    limits = document["limits"]
    if not isinstance(runtime, dict) or not isinstance(limits, dict):
        raise ValueError("materialization resource evidence is invalid")
    if (
        any(
            not _same_json_type(limits.get(name), value)
            for name, value in MATERIALIZATION_COST_LIMITS.items()
        )
        or limits != MATERIALIZATION_COST_LIMITS
    ):
        raise ValueError("materialization cost limits differ from the protocol")
    if (
        isinstance(runtime.get("wall_seconds"), bool)
        or not isinstance(runtime.get("wall_seconds"), (int, float))
        or not _same_json_type(runtime.get("peak_aggregate_rss_bytes"), 0)
        or not _same_json_type(document.get("logical_output_scratch_bytes"), 0)
    ):
        raise ValueError("materialization resource evidence is invalid")
    try:
        wall_seconds = float(runtime["wall_seconds"])
        peak_rss_bytes = int(runtime["peak_aggregate_rss_bytes"])
        logical_bytes = int(document["logical_output_scratch_bytes"])
        resources_pass = (
            math.isfinite(wall_seconds)
            and 0 <= wall_seconds <= MATERIALIZATION_COST_LIMITS["wall_seconds"]
            and 0
            <= peak_rss_bytes
            <= MATERIALIZATION_COST_LIMITS["peak_aggregate_rss_bytes"]
            and 0
            <= logical_bytes
            <= MATERIALIZATION_COST_LIMITS["logical_output_scratch_bytes"]
        )
        if "decision" in document or "attempt" in document:
            load_contract = (
                document.get("decision") == "pass"
                and document.get("attempt") in {1, 2}
                and not isinstance(document.get("attempt"), bool)
                and isinstance(document.get("post_launch_contention"), bool)
                and isinstance(document.get("timed_load_valid"), bool)
            )
        else:
            load_contract = document["timed_load_valid"] is True
        recomputed = resources_pass and load_contract
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("materialization resource evidence is invalid") from error
    if document["passes"] is not True or not recomputed:
        raise ValueError("materialization cost gate did not pass")


def _verify_materialization_artifacts(
    evidence: dict[str, Any], artifact_root: Path | None = None
) -> None:
    from experiments.g4_future_items.protocol.materialization import (
        DEFAULT_PERIOD_ARTIFACT_ROOT,
        PeriodArtifact,
    )

    root = artifact_root or DEFAULT_PERIOD_ARTIFACT_ROOT
    for selector_kind, name in (
        ("deterministic", "deterministic_artifact_sha256"),
        ("learned", "learned_artifact_sha256"),
    ):
        artifact = PeriodArtifact.open(root, expected_sha256=evidence[name])
        if artifact.manifest.get("selector_kind") != selector_kind:
            raise ValueError("measured selector artifact kind differs")
        if artifact.manifest.get("cost") != {
            "measurement_id": evidence["measurement_id"]
        }:
            raise ValueError("selector artifact measurement identity differs")


def _validate_ledger(document: dict[str, Any]) -> None:
    stage = document.get("stage")
    common = {"version", "stage", "rows", "sha256"}
    treatment_stages = {value[0] for value in _OBJECTIVE_FIELDS.values()}
    if stage == "control_tuning":
        expected_root = common | {"control_semantics_manifest_sha256"}
    elif stage in treatment_stages:
        expected_root = common | {
            "selected_control_manifest_sha256",
            "treatment_semantics_manifest_sha256",
            "anchor_parameters",
        }
        if stage.startswith("rq3_"):
            expected_root.add("materialization_cost_evidence_sha256")
    elif stage == "control_tuning_boundary":
        expected_root = common | {
            "control_semantics_manifest_sha256",
            "entering_row_sha256",
            "entering_row",
            "entering_rate_bounds",
            "entering_horizon_values",
            "rate_bounds",
            "horizon_values",
            "predecessor_evidence",
        }
    elif (
        isinstance(stage, str)
        and stage.endswith("_boundary")
        and stage.removesuffix("_boundary") in treatment_stages
    ):
        expected_root = common | {
            "selected_control_manifest_sha256",
            "treatment_semantics_manifest_sha256",
            "entering_row_sha256",
            "entering_row",
            "entering_rate_bounds",
            "entering_horizon_values",
            "rate_bounds",
            "horizon_values",
            "predecessor_evidence",
        }
        if stage.removesuffix("_boundary").startswith("rq3_"):
            expected_root.add("materialization_cost_evidence_sha256")
    elif stage in {"selector_search", "selector_gate", "selector_materialization"}:
        expected_root = common | {"treatment_semantics_manifest_sha256"}
    elif stage == "selector_search_boundary":
        expected_root = common | {
            "treatment_semantics_manifest_sha256",
            "entering_row_sha256",
            "entering_row",
            "entering_rate_bounds",
            "rate_bounds",
            "predecessor_evidence",
        }
    else:
        raise ValueError(f"ledger stage {stage!r} is not approved")
    if set(document) != expected_root:
        raise ValueError("ledger has missing or unknown root keys")
    if document["version"] != LEDGER_VERSION:
        raise ValueError("ledger version is not supported")
    digest = document.get("sha256")
    unsigned = {key: value for key, value in document.items() if key != "sha256"}
    if digest != canonical_sha256(unsigned):
        raise ValueError("ledger hash does not match its canonical contents")
    for identity_key in (
        "control_semantics_manifest_sha256",
        "selected_control_manifest_sha256",
        "treatment_semantics_manifest_sha256",
        "materialization_cost_evidence_sha256",
        "entering_row_sha256",
    ):
        if identity_key in document:
            _validate_sha256_text(identity_key, document[identity_key])
    rate_bounds = document.get("rate_bounds")
    if rate_bounds is not None:
        expected_rate_keys = (
            {"learning_rate"}
            if stage == "selector_search_boundary"
            else {"embedding_learning_rate", "deep_learning_rate"}
        )
        if not isinstance(rate_bounds, dict) or set(rate_bounds) != expected_rate_keys:
            raise ValueError("boundary ledger rate bounds are invalid")
        for bounds in rate_bounds.values():
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in bounds
                )
                or bounds[0] >= bounds[1]
            ):
                raise ValueError("boundary ledger rate bounds are invalid")
    horizon_values = document.get("horizon_values")
    if horizon_values is not None:
        if (
            not isinstance(horizon_values, list)
            or not horizon_values
            or horizon_values != sorted(set(horizon_values))
        ):
            raise ValueError("boundary ledger horizon values are invalid")
        for value in horizon_values:
            _positive_horizon(value)
        allowed_domains = {
            LOWER_ROUND_ONE_HORIZON_VALUES,
            UPPER_ROUND_ONE_HORIZON_VALUES,
            LOWER_ROUND_TWO_HORIZON_VALUES,
            UPPER_ROUND_TWO_HORIZON_VALUES,
        }
        if len(horizon_values) == 1:
            _approved_selected_horizon(horizon_values[0])
        elif tuple(horizon_values) not in allowed_domains:
            raise ValueError("boundary ledger horizon domain is not approved")
    rows = document["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("ledger rows must be a nonempty list")
    _validate_ledger_topology(stage, rows)
    if isinstance(stage, str) and stage.endswith("_boundary"):
        boundary_round = rows[0]["job"][
            "boundary_round" if stage.startswith("selector_") else "protocol"
        ]
        if isinstance(boundary_round, dict):
            boundary_round = boundary_round["boundary_round"]
        _validate_boundary_metadata(
            stage=stage,
            boundary_round=boundary_round,
            rate_bounds=rate_bounds,
            horizon_values=horizon_values,
        )
    row_ids: set[str] = set()
    run_names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "job"}:
            raise ValueError("ledger row has missing or unknown keys")
        row_id = row["id"]
        job = row["job"]
        if not isinstance(row_id, str) or not isinstance(job, dict):
            raise ValueError("ledger row identity is invalid")
        if row_id in row_ids:
            raise ValueError("ledger contains duplicate row ids")
        row_ids.add(row_id)
        if stage.startswith("selector_"):
            _validate_selector_job(stage, row_id, job, rate_bounds=rate_bounds)
        else:
            run_name = job.get("run_name")
            if not isinstance(run_name, str) or run_name in run_names:
                raise ValueError("ledger contains an invalid or duplicate run name")
            run_names.add(run_name)
            _validate_job(
                stage,
                row_id,
                job,
                rate_bounds=rate_bounds,
                horizon_values=horizon_values,
            )
    if "entering_row" in document and (
        canonical_sha256(document["entering_row"]) != document["entering_row_sha256"]
    ):
        raise ValueError("boundary entering row hash differs")
    _validate_seeded_ledger(document)


def _predecessor_paths(
    evidence: Any,
) -> tuple[list[Path], list[Path]]:
    if not isinstance(evidence, dict) or set(evidence) != {"ledgers", "runs"}:
        raise ValueError("boundary predecessor evidence schema differs")
    ledger_identities = evidence["ledgers"]
    run_evidence = evidence["runs"]
    if (
        not isinstance(ledger_identities, list)
        or not ledger_identities
        or not isinstance(run_evidence, list)
        or not run_evidence
    ):
        raise ValueError("boundary predecessor evidence is incomplete")
    identities = list(ledger_identities)
    for run in run_evidence:
        if not isinstance(run, dict) or set(run) != {
            "job_contract",
            "training_metadata",
            "sweep_log",
        }:
            raise ValueError("boundary predecessor run evidence differs")
        identities.extend(run.values())
    for identity in identities:
        if not isinstance(identity, dict) or set(identity) != _FILE_IDENTITY_KEYS:
            raise ValueError("boundary predecessor file identity differs")
        if _file_identity(Path(identity.get("path", ""))) != identity:
            raise ValueError("boundary predecessor file evidence differs")
    return (
        [Path(identity["path"]) for identity in ledger_identities],
        [Path(run["job_contract"]["path"]).parent for run in run_evidence],
    )


def _selector_predecessor_paths(evidence: Any) -> tuple[list[Path], Path]:
    if not isinstance(evidence, dict) or set(evidence) != {"ledgers", "results"}:
        raise ValueError("selector predecessor evidence schema differs")
    ledgers = evidence["ledgers"]
    results = evidence["results"]
    if (
        not isinstance(ledgers, list)
        or not ledgers
        or not isinstance(results, list)
        or not results
    ):
        raise ValueError("selector predecessor evidence is incomplete")
    identities = list(ledgers)
    roots: set[Path] = set()
    for result in results:
        if not isinstance(result, dict) or set(result) != {
            "row_id",
            "artifact_sha256",
            "artifact",
            "artifact_sha256_file",
            "result",
        }:
            raise ValueError("selector predecessor result evidence differs")
        if not isinstance(result["row_id"], str):
            raise ValueError("selector predecessor row identity differs")
        _validate_sha256_text("artifact_sha256", result["artifact_sha256"])
        identities.extend(
            result[name] for name in ("artifact", "artifact_sha256_file", "result")
        )
        artifact_path = Path(result["artifact"].get("path", ""))
        if (
            artifact_path.name != "artifact.json"
            or artifact_path.parent.name != result["artifact_sha256"]
        ):
            raise ValueError("selector predecessor artifact path differs")
        roots.add(artifact_path.parent.parent.resolve())
    for identity in identities:
        if not isinstance(identity, dict) or set(identity) != _FILE_IDENTITY_KEYS:
            raise ValueError("selector predecessor file identity differs")
        if _file_identity(Path(identity.get("path", ""))) != identity:
            raise ValueError("selector predecessor file evidence differs")
    if len(roots) != 1:
        raise ValueError("selector predecessor search root differs")
    return [Path(identity["path"]) for identity in ledgers], roots.pop()


def _validate_seeded_ledger(document: dict[str, Any]) -> None:
    stage = document["stage"]
    expected: dict[str, Any] | None = None
    if stage == "control_tuning":
        expected = compile_control_tuning_ledger(
            document["control_semantics_manifest_sha256"]
        )
    elif stage in {value[0] for value in _OBJECTIVE_FIELDS.values()}:
        first = document["rows"][0]["job"]
        objective_id = first["objective"]["id"]
        anchor = document["anchor_parameters"]
        expected = compile_treatment_tuning_ledger(
            objective_id=objective_id,
            selected_control_manifest_sha256=document[
                "selected_control_manifest_sha256"
            ],
            treatment_semantics_manifest_sha256=document[
                "treatment_semantics_manifest_sha256"
            ],
            batch_size=first["dataloader"]["batch_size"],
            embedding_learning_rate=anchor["embedding_learning_rate"],
            deep_learning_rate=anchor["deep_learning_rate"],
            lr_schedule_horizon_epochs=anchor["lr_schedule_horizon_epochs"],
            selector_artifact_sha256=first["objective"].get("selector_artifact_sha256"),
            _materialization_cost_evidence_sha256=document.get(
                "materialization_cost_evidence_sha256"
            ),
        )
    elif stage == "selector_search":
        expected = compile_selector_search_ledger(
            treatment_semantics_manifest_sha256=document[
                "treatment_semantics_manifest_sha256"
            ],
            input_artifact_sha256=document["rows"][0]["job"]["input_artifact_sha256"],
        )
    elif stage.endswith("_boundary") and stage != "selector_search_boundary":
        base_stage = stage.removesuffix("_boundary")
        references: dict[str, Any]
        if base_stage == "control_tuning":
            references = {
                "control_semantics_manifest_sha256": document[
                    "control_semantics_manifest_sha256"
                ]
            }
        else:
            references = {
                "selected_control_manifest_sha256": document[
                    "selected_control_manifest_sha256"
                ],
                "treatment_semantics_manifest_sha256": document[
                    "treatment_semantics_manifest_sha256"
                ],
                "materialization_cost_evidence_sha256": document.get(
                    "materialization_cost_evidence_sha256"
                ),
            }
        ledger_paths, run_directories = _predecessor_paths(
            document["predecessor_evidence"]
        )
        expected = compile_recommender_boundary_ledger(
            predecessor_ledger_paths=ledger_paths,
            candidate_run_directories=run_directories,
            boundary_round=document["rows"][0]["job"]["protocol"]["boundary_round"],
            **references,
        )
    elif stage == "selector_search_boundary":
        ledger_paths, search_root = _selector_predecessor_paths(
            document["predecessor_evidence"]
        )
        expected = compile_selector_boundary_ledger(
            predecessor_ledger_paths=ledger_paths,
            search_root=search_root,
            treatment_semantics_manifest_sha256=document[
                "treatment_semantics_manifest_sha256"
            ],
            boundary_round=document["rows"][0]["job"]["boundary_round"],
        )
    if expected is not None and expected != document:
        raise ValueError("ledger differs from its exact seeded compilation")


def _validate_ledger_topology(stage: Any, rows: list[Any]) -> None:
    if not isinstance(stage, str) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("ledger row topology is invalid")
    if stage == "selector_search":
        expected = {
            f"selector_search:{family}:{trial_id:02d}"
            for family in ("time", "content", "frequency", "learned")
            for trial_id in range(1, 13)
        }
    elif stage == "selector_gate":
        expected = {"selector_gate:01"}
    elif stage == "selector_materialization":
        expected = {f"selector_materialization:{fold_id}" for fold_id in range(5)}
    elif stage.endswith("_boundary"):
        try:
            first_job = rows[0]["job"]
            boundary_round = (
                first_job["boundary_round"]
                if stage.startswith("selector_")
                else first_job["protocol"]["boundary_round"]
            )
        except (KeyError, TypeError) as error:
            raise ValueError("ledger row topology is invalid") from error
        expected = {
            f"{stage}:{boundary_round}:{trial_id:02d}" for trial_id in range(1, 5)
        }
    else:
        count = 20 if stage == "control_tuning" else 12
        expected = {f"{stage}:{trial_id:02d}" for trial_id in range(1, count + 1)}
    actual = {row.get("id") for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("ledger row topology differs from the approved stage")
    for row in rows:
        job = row.get("job")
        if not isinstance(job, dict):
            raise ValueError("ledger row topology is invalid")
        if stage == "selector_materialization":
            expected_identity = int(row["id"].rsplit(":", 1)[1])
            actual_identity = job.get("fold_id")
        elif stage == "selector_gate":
            continue
        else:
            expected_identity = int(row["id"].rsplit(":", 1)[1])
            protocol = job if stage.startswith("selector_") else job.get("protocol")
            actual_identity = (
                protocol.get("trial_id") if isinstance(protocol, dict) else None
            )
        if actual_identity != expected_identity:
            raise ValueError("ledger row topology differs from its job identity")


def _rate_transition_values(
    base: tuple[float, float], boundary_round: int
) -> set[tuple[float, float]]:
    values = {base}
    for _ in range(boundary_round):
        values |= {
            _extended_bounds(bounds, side)
            for bounds in tuple(values)
            for side in ("lower", "upper")
        }
    return values


def _validate_boundary_metadata(
    *,
    stage: str,
    boundary_round: Any,
    rate_bounds: Any,
    horizon_values: Any,
) -> None:
    if boundary_round not in {1, 2}:
        raise ValueError("boundary metadata has no approved transition")
    rate_specs = (
        {"learning_rate": (0.01, 0.2)}
        if stage == "selector_search_boundary"
        else {
            "embedding_learning_rate": EMBEDDING_LR_BOUNDS,
            "deep_learning_rate": DEEP_LR_BOUNDS,
        }
    )
    if any(
        tuple(rate_bounds[name]) not in _rate_transition_values(bounds, boundary_round)
        for name, bounds in rate_specs.items()
    ):
        raise ValueError("boundary rate bounds are not an approved transition")
    if stage == "selector_search_boundary":
        return
    values = tuple(horizon_values)
    if boundary_round == 1:
        allowed_singletons = set(BASE_HORIZON_VALUES)
        allowed_domains = {
            LOWER_ROUND_ONE_HORIZON_VALUES,
            UPPER_ROUND_ONE_HORIZON_VALUES,
        }
    else:
        allowed_singletons = set(
            BASE_HORIZON_VALUES
            + LOWER_ROUND_ONE_HORIZON_VALUES
            + UPPER_ROUND_ONE_HORIZON_VALUES
        )
        allowed_domains = {
            LOWER_ROUND_TWO_HORIZON_VALUES,
            UPPER_ROUND_TWO_HORIZON_VALUES,
        }
    if not (
        (len(values) == 1 and values[0] in allowed_singletons)
        or values in allowed_domains
    ):
        raise ValueError("boundary horizon domain is not an approved transition")


def _finite_rate(name: str, value: Any, bounds: tuple[float, float]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} is outside the approved interval")


def _positive_horizon(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("lr_schedule_horizon_epochs must be a positive integer")


def _approved_selected_horizon(value: Any) -> None:
    _positive_horizon(value)
    if value > 50:
        raise ValueError("selected schedule horizon exceeds the approved surface")


def _validate_job(
    stage: Any,
    row_id: str,
    job: dict[str, Any],
    *,
    rate_bounds: dict[str, list[float]] | None,
    horizon_values: list[int] | None,
) -> None:
    base_keys = {
        "run_name",
        "protocol",
        "dataloader",
        "embedding_learning_rate",
        "deep_learning_rate",
        "lr_schedule_horizon_epochs",
        "seed",
    }
    boundary = isinstance(stage, str) and stage.endswith("_boundary")
    base_stage = stage.removesuffix("_boundary") if boundary else stage
    treatment = base_stage in {value[0] for value in _OBJECTIVE_FIELDS.values()}
    expected_keys = base_keys | ({"objective", "loss"} if treatment else set())
    if set(job) != expected_keys:
        raise ValueError(f"{row_id} job has missing or unknown keys")
    protocol = job["protocol"]
    if not isinstance(protocol, dict) or protocol.get("stage") != stage:
        raise ValueError(f"{row_id} protocol stage changed")
    if boundary:
        expected_protocol = {"stage", "trial_id", "boundary_round"}
        expected_seed = 42
        if protocol.get("boundary_round") not in {1, 2}:
            raise ValueError(f"{row_id} boundary round changed")
    else:
        expected_protocol = {"stage", "trial_id"}
        expected_seed = 42
    if set(protocol) != expected_protocol:
        raise ValueError(f"{row_id} protocol fields changed")
    if job["seed"] != expected_seed:
        raise ValueError(f"{row_id} seed changed")
    dataloader = job["dataloader"]
    if not isinstance(dataloader, dict) or set(dataloader) != {"batch_size"}:
        raise ValueError(f"{row_id} dataloader fields changed")
    if dataloader["batch_size"] != CONTROL_BATCH_SIZE:
        raise ValueError(f"{row_id} batch size must be 512")
    treatment_anchor = treatment and not boundary and row_id.endswith(":01")
    approved_embedding_bounds = (
        tuple(rate_bounds["embedding_learning_rate"])
        if rate_bounds is not None
        else (
            EMBEDDING_LR_SELECTION_BOUNDS if treatment_anchor else EMBEDDING_LR_BOUNDS
        )
    )
    approved_deep_bounds = (
        tuple(rate_bounds["deep_learning_rate"])
        if rate_bounds is not None
        else DEEP_LR_SELECTION_BOUNDS if treatment_anchor else DEEP_LR_BOUNDS
    )
    _finite_rate(
        "embedding_learning_rate",
        job["embedding_learning_rate"],
        cast(tuple[float, float], approved_embedding_bounds),
    )
    _finite_rate(
        "deep_learning_rate",
        job["deep_learning_rate"],
        cast(tuple[float, float], approved_deep_bounds),
    )
    approved_horizons = horizon_values or list(BASE_HORIZON_VALUES)
    if treatment_anchor:
        _approved_selected_horizon(job["lr_schedule_horizon_epochs"])
    elif job["lr_schedule_horizon_epochs"] not in approved_horizons:
        raise ValueError(f"{row_id} schedule horizon is not approved")
    if treatment:
        objective = job["objective"]
        objective_id = objective.get("id") if isinstance(objective, dict) else None
        if objective_id not in _OBJECTIVE_FIELDS:
            raise ValueError(f"{row_id} objective changed")
        expected_stage, objective_base, loss = _OBJECTIVE_FIELDS[objective_id]
        if expected_stage != base_stage or job["loss"] != loss:
            raise ValueError(f"{row_id} objective or mask changed")
        expected_objective_keys = set(objective_base)
        if objective_id.startswith("rq3_"):
            expected_objective_keys |= {"selector_artifact_sha256", "period_count"}
            _validate_sha256_text(
                "selector_artifact_sha256", objective.get("selector_artifact_sha256")
            )
            if objective.get("period_count") not in {1, 2, 4}:
                raise ValueError(f"{row_id} period count is not approved")
        if set(objective) != expected_objective_keys:
            raise ValueError(f"{row_id} objective fields changed")
        for name, value in objective_base.items():
            if objective.get(name) != value:
                raise ValueError(f"{row_id} objective changed")


def _validate_selector_job(
    stage: str,
    row_id: str,
    job: dict[str, Any],
    *,
    rate_bounds: dict[str, list[float]] | None,
) -> None:
    if set(job) != _SELECTOR_FIELDS or job.get("stage") != stage:
        raise ValueError(f"{row_id} selector job has missing or unknown fields")
    if job.get("seed") != 42:
        raise ValueError(f"{row_id} selector seed changed")
    _validate_sha256_text("output_artifact_sha256", job.get("output_artifact_sha256"))
    if job["output_artifact_sha256"] != _selector_output_identity(job):
        raise ValueError(f"{row_id} selector output identity changed")
    configurable = {
        "family",
        "period_width_seconds",
        "lookahead_seconds",
        "minimum_liked_events",
        "time_tolerance_seconds",
        "frequency_entity",
        "max_leaf_nodes",
        "learning_rate",
        "l2_regularization",
    }
    if stage == "selector_gate":
        if (
            job["input_artifact_sha256"] is not None
            or job["input_payload_sha256"] is not None
        ):
            raise ValueError("selector gate has a composite input hash")
        for name in (
            "deterministic_artifact_sha256",
            "deterministic_payload_sha256",
            "learned_artifact_sha256",
            "learned_payload_sha256",
        ):
            _validate_sha256_text(name, job[name])
        if any(
            job[name] is not None
            for name in configurable | {"trial_id", "fold_id", "boundary_round"}
        ):
            raise ValueError("selector gate contains tunable fields")
        return
    _validate_sha256_text("input_artifact_sha256", job["input_artifact_sha256"])
    if (
        job["deterministic_artifact_sha256"] is not None
        or job["deterministic_payload_sha256"] is not None
        or job["learned_artifact_sha256"] is not None
        or job["learned_payload_sha256"] is not None
    ):
        raise ValueError("selector search/materialization has gate-only input hashes")
    if stage == "selector_materialization":
        _validate_sha256_text("input_payload_sha256", job["input_payload_sha256"])
        if job["trial_id"] is not None or job["boundary_round"] is not None:
            raise ValueError("selector materialization contains search identity")
        if job["fold_id"] not in range(5):
            raise ValueError("selector materialization fold changed")
    elif stage == "selector_search":
        if job["input_payload_sha256"] is not None:
            raise ValueError("selector search has an unexpected input payload hash")
        if job["fold_id"] is not None or job["boundary_round"] is not None:
            raise ValueError("selector search contains fold or boundary identity")
        if job["trial_id"] not in range(1, 13):
            raise ValueError("selector search trial changed")
    elif stage == "selector_search_boundary":
        if job["input_payload_sha256"] is not None:
            raise ValueError("selector search has an unexpected input payload hash")
        if job["fold_id"] is not None or job["boundary_round"] not in {1, 2}:
            raise ValueError("selector boundary identity changed")
        if job["trial_id"] not in range(1, 5):
            raise ValueError("selector boundary trial changed")
        if job["family"] != "learned" or rate_bounds is None:
            raise ValueError("selector boundary must freeze learned structure")
    else:
        raise ValueError(f"selector stage {stage!r} is unsupported")
    family = job["family"]
    if family not in {"time", "content", "frequency", "learned"}:
        raise ValueError("selector family changed")
    width = job["period_width_seconds"]
    lookahead = job["lookahead_seconds"]
    if width not in {3600, 21600, 86400}:
        raise ValueError("selector period width changed")
    allowed_lookahead = (
        {604800}
        if family == "time" and width in {3600, 21600}
        else ({259200, 604800} if width in {3600, 21600} else {1209600, 2419200})
    )
    if lookahead not in allowed_lookahead or job["minimum_liked_events"] not in {
        1,
        2,
        4,
    }:
        raise ValueError("selector structural fields changed")
    if (job["time_tolerance_seconds"] in {0, 3600, 7200}) != (family == "time"):
        raise ValueError("selector time fields changed")
    if (job["frequency_entity"] in {"item", "artist", "album"}) != (
        family == "frequency"
    ):
        raise ValueError("selector frequency fields changed")
    learned = family == "learned"
    if (job["max_leaf_nodes"] in {7, 15, 31}) != learned:
        raise ValueError("selector learned capacity changed")
    for name, bounds in (
        ("learning_rate", (0.01, 0.2)),
        ("l2_regularization", (0.00001, 1.0)),
    ):
        value = job[name]
        if learned:
            approved_bounds = (
                tuple(rate_bounds["learning_rate"])
                if name == "learning_rate" and rate_bounds is not None
                else bounds
            )
            _finite_rate(name, value, cast(tuple[float, float], approved_bounds))
        elif value is not None:
            raise ValueError(f"selector {name} applies only to learned family")


def load_ledger(path: Path) -> dict[str, Any]:
    document = load_strict_json(path)
    _validate_ledger(document)
    return document


def _validate_control_semantics_document(document: dict[str, Any]) -> None:
    expected = {
        "version",
        "kind",
        "control_manifest_sha256",
        "source_paths",
        "sources",
        "data_identity",
        "training_semantics_revisions",
        "resolved_anchor_configuration",
    }
    if set(document) != expected or document.get("version") != 1:
        raise ValueError("control semantics manifest schema differs")
    if document.get("control_manifest_sha256") != APPROVED_CONTROL_MANIFEST_SHA256:
        raise ValueError("control semantics references another control manifest")
    paths = document["source_paths"]
    sources = document["sources"]
    if (
        not isinstance(paths, list)
        or not all(isinstance(path, str) for path in paths)
        or paths != expected_control_source_paths()
        or not isinstance(sources, dict)
        or list(sources) != paths
    ):
        raise ValueError("control semantics source closure differs")
    for relative, digest in sources.items():
        _validate_sha256_text(f"control source hash for {relative}", digest)
    _validate_data_identity(document["data_identity"])
    revisions = document["training_semantics_revisions"]
    if (
        not isinstance(revisions, dict)
        or not revisions
        or list(revisions) != sorted(revisions)
        or any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            for name, value in revisions.items()
        )
    ):
        raise ValueError("control training semantic revisions differ")
    control = load_control_manifest()
    anchor = control["anchor"]
    expected_configuration = {
        "fixed": control["fixed"],
        "selected": {
            "batch_size": anchor["batch_size"],
            "embedding_learning_rate": anchor["embedding_learning_rate"],
            "deep_learning_rate": anchor["deep_learning_rate"],
            "lr_schedule_horizon_epochs": anchor["lr_schedule_horizon_epochs"],
        },
    }
    if document["resolved_anchor_configuration"] != expected_configuration:
        raise ValueError("control resolved anchor configuration differs")


def _validate_selected_control_document(document: dict[str, Any]) -> None:
    expected = {
        "version",
        "kind",
        "control_semantics_manifest_sha256",
        "ledger_sha256",
        "selection",
        "seed_42_configuration",
        "seed_42_configuration_sha256",
        "evidence",
    }
    if set(document) != expected or document.get("version") != 2:
        raise ValueError("selected-control manifest schema differs")
    for name in (
        "control_semantics_manifest_sha256",
        "ledger_sha256",
        "seed_42_configuration_sha256",
    ):
        _validate_sha256_text(name, document[name])
    if (
        canonical_sha256(document["seed_42_configuration"])
        != document["seed_42_configuration_sha256"]
    ):
        raise ValueError("selected-control seed configuration hash differs")
    selection = document["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "row_id",
        "run_name",
        "validation_recall_at_100",
        "validation_loss",
        "best_epoch",
        "epochs_trained",
        "canonical_parameters",
    }:
        raise ValueError("selected-control selection schema differs")
    if not all(
        isinstance(selection[name], str) and selection[name]
        for name in ("row_id", "run_name")
    ):
        raise ValueError("selected-control selection identity differs")
    for name in ("validation_recall_at_100", "validation_loss"):
        value = selection[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"selected-control {name} differs")
    parameters = selection["canonical_parameters"]
    if not isinstance(parameters, dict) or set(parameters) != {
        "batch_size",
        "embedding_learning_rate",
        "deep_learning_rate",
        "lr_schedule_horizon_epochs",
    }:
        raise ValueError("selected-control parameters schema differs")
    if parameters["batch_size"] != CONTROL_BATCH_SIZE:
        raise ValueError("selected-control batch differs")
    _finite_rate(
        "embedding_learning_rate",
        parameters["embedding_learning_rate"],
        EMBEDDING_LR_SELECTION_BOUNDS,
    )
    _finite_rate(
        "deep_learning_rate",
        parameters["deep_learning_rate"],
        DEEP_LR_SELECTION_BOUNDS,
    )
    _approved_selected_horizon(parameters["lr_schedule_horizon_epochs"])
    if (
        isinstance(selection["best_epoch"], bool)
        or not isinstance(selection["best_epoch"], int)
        or not 1 <= selection["best_epoch"] <= parameters["lr_schedule_horizon_epochs"]
        or selection["epochs_trained"] != parameters["lr_schedule_horizon_epochs"]
        or isinstance(selection["epochs_trained"], bool)
    ):
        raise ValueError("selected-control horizon evidence differs")
    configuration = document["seed_42_configuration"]
    if (
        not isinstance(configuration, dict)
        or set(configuration) != {"run_name", "seed", "fixed", "selected"}
        or configuration["run_name"] != selection["run_name"]
        or configuration["seed"] != 42
        or configuration["fixed"] != load_control_manifest()["fixed"]
        or configuration["selected"] != parameters
    ):
        raise ValueError("selected-control seed configuration differs")
    evidence = document["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"ledgers", "runs"}:
        raise ValueError("selected-control evidence schema differs")
    ledgers = evidence["ledgers"]
    runs = evidence["runs"]
    if (
        not isinstance(ledgers, list)
        or not ledgers
        or not isinstance(runs, list)
        or not runs
    ):
        raise ValueError("selected-control evidence set differs")
    identities = list(ledgers)
    for run in runs:
        if not isinstance(run, dict) or set(run) != {
            "job_contract",
            "training_metadata",
            "sweep_log",
        }:
            raise ValueError("selected-control run evidence differs")
        identities.extend(run.values())
    for identity in identities:
        if not isinstance(identity, dict) or set(identity) != _FILE_IDENTITY_KEYS:
            raise ValueError("selected-control file identity differs")
        if _file_identity(Path(identity.get("path", ""))) != identity:
            raise ValueError("selected-control file evidence differs")
    reconstructed = build_selected_control_manifest(
        control_semantics_manifest_sha256=document["control_semantics_manifest_sha256"],
        ledger_paths=[Path(identity["path"]) for identity in ledgers],
        run_directories=[Path(run["job_contract"]["path"]).parent for run in runs],
    )
    if reconstructed != document:
        raise ValueError("selected-control evidence does not reproduce the manifest")


def _validate_treatment_semantics_v1_document(document: dict[str, Any]) -> None:
    expected = {
        "version",
        "kind",
        "selected_control_manifest_sha256",
        "preimplementation_source_manifest_sha256",
        "source_paths",
        "sources",
        "entrypoint_source_paths",
        "changed_paths",
        "schema_revisions",
        "fixtures",
    }
    if set(document) != expected or document.get("version") != 1:
        raise ValueError("treatment semantics manifest schema differs")
    for name in (
        "selected_control_manifest_sha256",
        "preimplementation_source_manifest_sha256",
    ):
        _validate_sha256_text(name, document[name])
    paths = document["source_paths"]
    sources = document["sources"]
    entrypoints = document["entrypoint_source_paths"]
    if (
        not isinstance(paths, list)
        or not all(isinstance(path, str) for path in paths)
        or paths != sorted(set(paths))
        or not isinstance(sources, dict)
        or list(sources) != paths
        or not isinstance(entrypoints, dict)
        or set(entrypoints) != _G4_ENTRYPOINTS
    ):
        raise ValueError("treatment source closure schema differs")
    for entrypoint, closure in entrypoints.items():
        if (
            not isinstance(closure, list)
            or not all(isinstance(path, str) for path in closure)
            or closure != sorted(set(closure))
            or entrypoint not in closure
        ):
            raise ValueError(f"treatment closure for {entrypoint} differs")
    if sorted({path for closure in entrypoints.values() for path in closure}) != paths:
        raise ValueError("treatment entrypoint closures do not cover source paths")
    for relative, digest in sources.items():
        _validate_sha256_text(f"treatment source hash for {relative}", digest)
    for entrypoint, closure in entrypoints.items():
        if (
            not isinstance(closure, list)
            or closure != sorted(set(closure))
            or entrypoint not in closure
        ):
            raise ValueError("treatment compatibility entrypoint closure differs")
    before = load_preimplementation_source_manifest()
    preimplementation = load_strict_json(PREIMPLEMENTATION_SOURCE_MANIFEST_PATH)
    if (
        canonical_sha256(preimplementation)
        != document["preimplementation_source_manifest_sha256"]
    ):
        raise ValueError("preimplementation source manifest hash differs")
    expected_changes = [
        {
            "path": relative,
            "before_sha256": before[relative],
            "after_sha256": sources[relative],
        }
        for relative in sorted(set(before) & set(paths))
        if before[relative] != sources[relative]
    ]
    if document["changed_paths"] != expected_changes:
        raise ValueError("treatment changed-path evidence differs")
    revisions = document["schema_revisions"]
    if (
        not isinstance(revisions, dict)
        or not revisions
        or list(revisions) != sorted(revisions)
        or any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (str, int))
            for name, value in revisions.items()
        )
    ):
        raise ValueError("treatment schema revisions differ")
    fixtures = document["fixtures"]
    if (
        not isinstance(fixtures, dict)
        or not fixtures
        or list(fixtures) != sorted(fixtures)
    ):
        raise ValueError("treatment fixture identities differ")
    for name, identity in fixtures.items():
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
            raise ValueError(f"treatment fixture {name!r} schema differs")
        if not isinstance(identity["path"], str) or not identity["path"]:
            raise ValueError(f"treatment fixture {name!r} path differs")
        _validate_sha256_text(f"treatment fixture {name!r} hash", identity["sha256"])


def _compatibility_runtime_projection(experiment: Any) -> dict[str, Any]:
    from experiments.g4_future_items.configs.control import (
        control_runtime_projection,
    )

    objective = None
    if hasattr(experiment, "objective_id"):
        objective = {
            "objective_id": experiment.objective_id,
            "objective_window_seconds": experiment.objective_window_seconds,
            "objective_event_lookahead": experiment.objective_event_lookahead,
            "selector_artifact_sha256": experiment.selector_artifact_sha256,
            "objective_period_count": experiment.objective_period_count,
            "valid_positive_mask_mode": experiment.valid_positive_mask_mode,
        }
    return {
        "experiment_class": (
            f"{type(experiment).__module__}.{type(experiment).__qualname__}"
        ),
        "run_name": experiment.run_name,
        "seed": experiment.seed,
        "control": control_runtime_projection(experiment),
        "objective": objective,
    }


def _selector_optimization_regression(project_root: Path) -> dict[str, Any]:
    import numpy as np

    from experiments.g4_future_items.selectors import (
        DAY_SECONDS,
        ChronologicalBounds,
        LikeEvent,
        ListenEvent,
        SelectorConfiguration,
        build_selector_examples,
    )

    day = DAY_SECONDS
    likes = (
        LikeEvent(7, 2 * day + 10, 1, (11,), (21,), np.array([1.0, 0.0, 0.0])),
        LikeEvent(7, 3 * day + 20, 2, (12,), (22,), np.array([0.0, 1.0, 0.0])),
        LikeEvent(7, 4 * day + 30, 3, (11,), (23,), np.array([0.0, 0.0, 1.0])),
        LikeEvent(7, 4 * day + 30, 4, (13,), (24,), np.array([1.0, 1.0, 0.0])),
        LikeEvent(7, 4 * day + 30, 5, (13,), (24,), np.array([1.0, 0.0, 1.0])),
        LikeEvent(7, 4 * day + 30, 6, (14,), (25,), np.array([0.0, 1.0, 1.0])),
        LikeEvent(7, 5 * day + 40, 7, (11,), (21,), np.array([-1.0, 0.0, 0.0])),
    )
    listens = (
        ListenEvent(7, 2 * day + 30, (11,)),
        ListenEvent(7, 3 * day + 30, (12,)),
        ListenEvent(7, 4 * day + 40, (13,)),
        ListenEvent(7, 5 * day + 30, (11,)),
    )
    examples = build_selector_examples(
        likes,
        listens,
        ChronologicalBounds.from_interval(0, 60 * day),
        SelectorConfiguration("content", day, 28 * day, 1),
    )
    digest = canonical_sha256([asdict(example) for example in examples])
    expected = "d6cbb93493af12909a9ea8399288fcf30a9d9c059b95cdb7c5461c67e67dcfcc"
    if len(examples) != 15 or digest != expected:
        raise ValueError("selector optimization changed duplicate-period output")
    representative_likes = []
    for index in range(30):
        embedding = np.zeros(32, dtype=np.float64)
        embedding[index % 32] = 1.0
        representative_likes.append(
            LikeEvent(
                9,
                (2 + index) * day + index,
                index + 1,
                (index % 11 + 1,),
                (index % 7 + 1,),
                embedding,
            )
        )
    burst_timestamp = 33 * day + 123
    for index in range(130):
        embedding = np.zeros(32, dtype=np.float64)
        embedding[index % 32] = 1.0
        representative_likes.append(
            LikeEvent(
                9,
                burst_timestamp,
                1000 + index,
                (index % 11 + 1,),
                (index % 7 + 1,),
                embedding,
            )
        )
    representative = build_selector_examples(
        tuple(representative_likes),
        tuple(
            ListenEvent(9, (2 + index) * day + 50, (index % 11 + 1,))
            for index in range(32)
        ),
        ChronologicalBounds.from_interval(0, 60 * day),
        SelectorConfiguration("content", day, 28 * day, 1),
    )
    representative_digest = canonical_sha256(
        [asdict(example) for example in representative]
    )
    representative_expected = (
        "f31aad7809d99d40a10d2cf696047473186cd101064ff5b171ab146723aa7e0a"
    )
    if len(representative) != 3_812 or representative_digest != representative_expected:
        raise ValueError("selector optimization changed representative output")
    test_path = (
        "dcn/tests/experiments/g4_future_items/"
        "test_selector_preparation_optimization.py"
    )
    return {
        "fixtures": [
            {
                "name": "duplicate-timestamp-period-cache-v1",
                "row_count": len(examples),
                "prechange_and_current_output_sha256": digest,
            },
            {
                "name": "representative-duplicate-burst-v1",
                "row_count": len(representative),
                "prechange_and_current_output_sha256": representative_digest,
            },
        ],
        "test": {
            "path": test_path,
            "sha256": _file_sha256(project_root / test_path),
            "nodes": [
                (
                    "test_selector_preparation_optimization.py::"
                    "test_duplicate_timestamp_examples_match_uncached_reference"
                ),
                (
                    "test_selector_preparation_optimization.py::"
                    "test_representative_duplicate_burst_matches_uncached_reference"
                ),
            ],
        },
    }


def build_runtime_compatibility_evidence(
    *,
    predecessor_treatment_sha256: str,
    source_changes: list[dict[str, str]],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    from experiments.g4_future_items.configs.treatments import build_treatment
    from experiments.g4_future_items.launchers.compiled import (
        build_training_experiment,
    )

    change_paths = {change["path"] for change in source_changes}
    approved_paths = (
        _G4_COMPATIBILITY_CLOSURE_ONLY_PATHS
        | _G4_COMPATIBILITY_OPTIMIZATION_PATHS
        | _G4_COMPATIBILITY_PROTOCOL_PATHS
    )
    if not change_paths <= approved_paths:
        raise ValueError("compatibility evidence contains an unapproved source path")
    source_by_path = {change["path"]: change for change in source_changes}
    if any(
        source_by_path[path]["after_sha256"]
        != _G4_COMPATIBILITY_CLOSURE_ONLY_AFTER_SHA256[path]
        for path in change_paths & _G4_COMPATIBILITY_CLOSURE_ONLY_PATHS
    ):
        raise ValueError("closure-only compatibility hash is not reviewed")
    if any(
        source_by_path[path]["after_sha256"]
        != _G4_COMPATIBILITY_OPTIMIZATION_AFTER_SHA256[path]
        for path in change_paths & _G4_COMPATIBILITY_OPTIMIZATION_PATHS
    ):
        raise ValueError("optimization compatibility hash is not reviewed")
    ledgers = []
    rows = []
    for relative in _G4_COMPATIBILITY_LEDGER_PATHS:
        path = project_root / relative
        ledger = load_ledger(path)
        identity = {
            "stage": ledger["stage"],
            "row_count": len(ledger["rows"]),
            "ledger_sha256": ledger["sha256"],
            "canonical_sha256": canonical_sha256(ledger),
        }
        if identity != _G4_COMPATIBILITY_LEDGER_IDENTITIES[relative]:
            raise ValueError("frozen compatibility ledger identity differs")
        if ledger["stage"].startswith("control_"):
            if ledger["control_semantics_manifest_sha256"] != (
                "acc6a6c7a614f0f84f4f9f69bc8c110625a22965f46b38635b5d6be4037447e8"
            ):
                raise ValueError("frozen control ledger semantics differ")
        elif (
            ledger["selected_control_manifest_sha256"]
            != "c30fb4eafcea2cefa1099631a40ca1531245e412c1cedcdbd02d9f7fea7aafd6"
            or ledger["treatment_semantics_manifest_sha256"]
            != predecessor_treatment_sha256
        ):
            raise ValueError("frozen treatment ledger semantics differ")
        ledgers.append({"path": relative, **identity})
        for row in ledger["rows"]:
            compiled = CompiledJob.from_row(ledger_sha256=ledger["sha256"], row=row)
            projection = _compatibility_runtime_projection(
                build_training_experiment(compiled)
            )
            rows.append(
                {
                    "ledger_sha256": ledger["sha256"],
                    "row_id": row["id"],
                    "job_sha256": canonical_sha256(row["job"]),
                    "runtime_projection": projection,
                    "runtime_projection_sha256": canonical_sha256(projection),
                }
            )
    if len(rows) != 48:
        raise ValueError("compatibility evidence requires 48 frozen rows")
    selected_path = (
        project_root
        / "experiments/g4_future_items/protocol/selected_control_manifest.json"
    )
    selected = load_strict_json(selected_path)
    parameters = selected["selection"]["canonical_parameters"]
    rq3 = []
    for objective_id, mask in (
        ("rq3_deterministic_hard", "selected_period_union_unique"),
        ("rq3_learned_hard", "selected_period_union_unique"),
        (
            "rq3_learned_proportional",
            "all_positive_probability_periods_unique",
        ),
    ):
        projection = _compatibility_runtime_projection(
            build_treatment(
                objective={
                    "id": objective_id,
                    "selector_artifact_sha256": "f" * 64,
                    "period_count": 1,
                },
                valid_positive_mask_mode=mask,
                run_name=f"g4_compatibility_{objective_id}_native50m",
                batch_size=parameters["batch_size"],
                embedding_learning_rate=parameters["embedding_learning_rate"],
                deep_learning_rate=parameters["deep_learning_rate"],
                lr_schedule_horizon_epochs=parameters["lr_schedule_horizon_epochs"],
                seed=42,
            )
        )
        rq3.append(
            {
                "objective_id": objective_id,
                "runtime_projection": projection,
                "runtime_projection_sha256": canonical_sha256(projection),
            }
        )
    generator_path = "experiments/g4_future_items/protocol/manifest.py"
    document = {
        "version": _G4_COMPATIBILITY_EVIDENCE_VERSION,
        "scope": _G4_COMPATIBILITY_SCOPE,
        "predecessor_treatment_semantics_manifest_sha256": (
            predecessor_treatment_sha256
        ),
        "source_changes": source_changes,
        "source_change_classification": {
            "closure_only": sorted(change_paths & _G4_COMPATIBILITY_CLOSURE_ONLY_PATHS),
            "behavior_preserving_optimization": sorted(
                change_paths & _G4_COMPATIBILITY_OPTIMIZATION_PATHS
            ),
            "protocol": sorted(change_paths & _G4_COMPATIBILITY_PROTOCOL_PATHS),
        },
        "generator": {
            "path": generator_path,
            "sha256": source_by_path[generator_path]["after_sha256"],
        },
        "inputs": {
            "ledgers": ledgers,
            "selected_control": {
                "path": selected_path.relative_to(project_root).as_posix(),
                "sha256": canonical_sha256(selected),
            },
        },
        "recommender_rows": rows,
        "planned_rq3": rq3,
    }
    if change_paths & _G4_COMPATIBILITY_OPTIMIZATION_PATHS:
        document["optimization_regression"] = _selector_optimization_regression(
            project_root
        )
    return document


def _validate_compatibility_evidence(
    document: dict[str, Any],
    *,
    predecessor_treatment_sha256: str,
    source_changes: list[dict[str, str]],
    project_root: Path = PROJECT_ROOT,
) -> None:
    expected = build_runtime_compatibility_evidence(
        predecessor_treatment_sha256=predecessor_treatment_sha256,
        source_changes=source_changes,
        project_root=project_root,
    )
    if document != expected:
        raise ValueError("compatibility evidence does not reproduce")


def _validate_treatment_semantics_v2_document(document: dict[str, Any]) -> None:
    expected = {
        "version",
        "kind",
        "selected_control_manifest_sha256",
        "historical_lineage",
        "source_paths",
        "sources",
        "entrypoint_source_paths",
        "source_changes",
        "source_additions",
        "schema_revisions",
        "fixtures",
        "compatibility",
    }
    if set(document) != expected or document.get("version") != 2:
        raise ValueError("treatment compatibility manifest schema differs")
    if document.get("kind") != "g4_treatment_semantics":
        raise ValueError("treatment compatibility manifest kind differs")
    _validate_sha256_text(
        "selected_control_manifest_sha256",
        document["selected_control_manifest_sha256"],
    )
    paths = document["source_paths"]
    sources = document["sources"]
    entrypoints = document["entrypoint_source_paths"]
    if (
        not isinstance(paths, list)
        or paths != sorted(set(paths))
        or not isinstance(sources, dict)
        or list(sources) != paths
        or not isinstance(entrypoints, dict)
        or set(entrypoints) != _G4_ENTRYPOINTS
    ):
        raise ValueError("treatment compatibility source closure differs")
    for relative, digest in sources.items():
        if not isinstance(relative, str):
            raise ValueError("treatment compatibility source path differs")
        _validate_sha256_text(f"treatment source hash for {relative}", digest)
    if sorted({path for closure in entrypoints.values() for path in closure}) != paths:
        raise ValueError("treatment compatibility entrypoint closures differ")
    lineage = document["historical_lineage"]
    if not isinstance(lineage, dict) or set(lineage) != {
        "control_semantics",
        "selected_control",
        "treatment_semantics",
    }:
        raise ValueError("historical semantics lineage schema differs")
    for name, identity in lineage.items():
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
            raise ValueError(f"historical {name} identity differs")
        if not isinstance(identity["path"], str) or not identity["path"]:
            raise ValueError(f"historical {name} path differs")
        _validate_sha256_text(f"historical {name} hash", identity["sha256"])
    if (
        lineage["selected_control"]["sha256"]
        != document["selected_control_manifest_sha256"]
    ):
        raise ValueError("historical selected-control hash differs")
    changes = document["source_changes"]
    if not isinstance(changes, list):
        raise ValueError("compatibility source changes differ")
    change_paths = []
    for change in changes:
        if not isinstance(change, dict) or set(change) != {
            "path",
            "before_sha256",
            "after_sha256",
        }:
            raise ValueError("compatibility source change schema differs")
        change_paths.append(change["path"])
        _validate_sha256_text("compatibility before hash", change["before_sha256"])
        _validate_sha256_text("compatibility after hash", change["after_sha256"])
        if sources.get(change["path"]) != change["after_sha256"]:
            raise ValueError("compatibility current source hash differs")
    if change_paths != sorted(set(change_paths)):
        raise ValueError("compatibility source change paths are not canonical")
    additions = document["source_additions"]
    if not isinstance(additions, list):
        raise ValueError("compatibility source additions differ")
    addition_paths = []
    for addition in additions:
        if not isinstance(addition, dict) or set(addition) != {
            "path",
            "after_sha256",
        }:
            raise ValueError("compatibility source addition schema differs")
        addition_paths.append(addition["path"])
        _validate_sha256_text(
            "compatibility added-source hash", addition["after_sha256"]
        )
        if sources.get(addition["path"]) != addition["after_sha256"]:
            raise ValueError("compatibility added-source hash differs")
    if addition_paths != sorted(set(addition_paths)):
        raise ValueError("compatibility source addition paths are not canonical")
    _validate_compatibility_source_additions(additions)
    compatibility = document["compatibility"]
    if not isinstance(compatibility, dict) or set(compatibility) != {
        "scope",
        "approved_source_changes",
        "evidence",
    }:
        raise ValueError("compatibility declaration differs")
    if (
        compatibility["scope"] != _G4_COMPATIBILITY_SCOPE
        or compatibility["approved_source_changes"] != change_paths
    ):
        raise ValueError("compatibility scope or approved paths differ")
    evidence = compatibility["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
        raise ValueError("compatibility evidence identity differs")
    if not isinstance(evidence["path"], str) or not evidence["path"]:
        raise ValueError("compatibility evidence path differs")
    _validate_sha256_text("compatibility evidence hash", evidence["sha256"])
    revisions = document["schema_revisions"]
    if not isinstance(revisions, dict) or not revisions:
        raise ValueError("treatment compatibility schema revisions differ")
    fixtures = document["fixtures"]
    if not isinstance(fixtures, dict) or not fixtures:
        raise ValueError("treatment compatibility fixtures differ")


def _validate_treatment_semantics_document(document: dict[str, Any]) -> None:
    if document.get("version") == 1:
        _validate_treatment_semantics_v1_document(document)
        return
    if document.get("version") == 2:
        _validate_treatment_semantics_v2_document(document)
        return
    raise ValueError("treatment semantics manifest version is not supported")


def _verify_current_semantics(document: dict[str, Any]) -> None:
    kind = document.get("kind")
    if kind == "g4_control_semantics":
        _validate_control_semantics_document(document)
        if (
            source_manifest(PROJECT_ROOT, document["source_paths"])
            != document["sources"]
        ):
            raise ValueError("current control source hashes differ")
        if (
            _current_control_data_identity(document["data_identity"])
            != document["data_identity"]
        ):
            raise ValueError("current control data identity differs")
        return
    if kind == "g4_selected_control":
        _validate_selected_control_document(document)
        return
    if kind == "g4_treatment_semantics":
        _validate_treatment_semantics_document(document)
        if document["version"] == 2:
            lineage_documents = {}
            for name, identity in document["historical_lineage"].items():
                path = (PROJECT_ROOT / identity["path"]).resolve()
                if (
                    not path.is_relative_to(PROJECT_ROOT.resolve())
                    or not path.is_file()
                ):
                    raise ValueError(f"historical {name} manifest is missing")
                historical = load_strict_json(path)
                if canonical_sha256(historical) != identity["sha256"]:
                    raise ValueError(f"historical {name} manifest hash differs")
                lineage_documents[name] = historical
            control = lineage_documents["control_semantics"]
            selected = lineage_documents["selected_control"]
            predecessor = lineage_documents["treatment_semantics"]
            _validate_control_semantics_document(control)
            if (
                _current_control_data_identity(control["data_identity"])
                != control["data_identity"]
            ):
                raise ValueError("current control data identity differs")
            _validate_selected_control_document(selected)
            _validate_treatment_semantics_v1_document(predecessor)
            if (
                selected["control_semantics_manifest_sha256"]
                != document["historical_lineage"]["control_semantics"]["sha256"]
                or predecessor["selected_control_manifest_sha256"]
                != document["historical_lineage"]["selected_control"]["sha256"]
            ):
                raise ValueError("historical semantics lineage differs")
            if (
                predecessor["schema_revisions"] != document["schema_revisions"]
                or predecessor["fixtures"] != document["fixtures"]
            ):
                raise ValueError("compatibility predecessor semantics differ")
            current_entrypoints = derive_current_entrypoint_source_paths(
                predecessor["entrypoint_source_paths"], project_root=PROJECT_ROOT
            )
            current_paths = sorted(
                {path for paths in current_entrypoints.values() for path in paths}
            )
            if (
                document["entrypoint_source_paths"] != current_entrypoints
                or document["source_paths"] != current_paths
            ):
                raise ValueError("current treatment source closure differs")
            expected_changes = [
                {
                    "path": relative,
                    "before_sha256": predecessor["sources"][relative],
                    "after_sha256": document["sources"][relative],
                }
                for relative in predecessor["source_paths"]
                if predecessor["sources"][relative] != document["sources"][relative]
            ]
            if document["source_changes"] != expected_changes:
                raise ValueError("compatibility source changes differ")
            expected_additions = [
                {
                    "path": relative,
                    "after_sha256": document["sources"][relative],
                }
                for relative in sorted(
                    set(document["source_paths"]) - set(predecessor["source_paths"])
                )
            ]
            if document["source_additions"] != expected_additions:
                raise ValueError("compatibility source additions differ")
            _validate_compatibility_source_additions(expected_additions)
            evidence_identity = document["compatibility"]["evidence"]
            evidence_path = (PROJECT_ROOT / evidence_identity["path"]).resolve()
            if (
                not evidence_path.is_relative_to(PROJECT_ROOT.resolve())
                or not evidence_path.is_file()
            ):
                raise ValueError("compatibility evidence is missing")
            evidence = load_strict_json(evidence_path)
            if canonical_sha256(evidence) != evidence_identity["sha256"]:
                raise ValueError("compatibility evidence hash differs")
            _validate_compatibility_evidence(
                evidence,
                predecessor_treatment_sha256=document["historical_lineage"][
                    "treatment_semantics"
                ]["sha256"],
                source_changes=document["source_changes"],
                project_root=PROJECT_ROOT,
            )
        if (
            source_manifest(PROJECT_ROOT, document["source_paths"])
            != document["sources"]
        ):
            raise ValueError("current treatment source hashes differ")
        for name, identity in document["fixtures"].items():
            path = (PROJECT_ROOT / identity["path"]).resolve()
            if (
                not path.is_relative_to(PROJECT_ROOT.resolve())
                or not path.is_file()
                or _file_sha256(path) != identity["sha256"]
            ):
                raise ValueError(f"current treatment fixture differs: {name}")
        return
    raise ValueError(f"semantics manifest kind {kind!r} is not approved")


def _verify_historical_semantics(document: dict[str, Any]) -> None:
    kind = document.get("kind")
    if kind == "g4_control_semantics":
        _validate_control_semantics_document(document)
        if (
            _current_control_data_identity(document["data_identity"])
            != document["data_identity"]
        ):
            raise ValueError("current control data identity differs")
        return
    if kind == "g4_selected_control":
        _validate_selected_control_document(document)
        return
    if kind == "g4_treatment_semantics" and document.get("version") == 1:
        _validate_treatment_semantics_v1_document(document)
        return
    raise ValueError("historical semantics document is not approved")


def verify_ledger_semantics(
    ledger: dict[str, Any],
    reference_paths: dict[str, Path],
    *,
    compatibility_path: Path | None = None,
) -> None:
    _validate_ledger(ledger)
    evidence_key = "materialization_cost_evidence_sha256"
    visited: set[str] = set()
    if evidence_key in ledger:
        if evidence_key not in reference_paths:
            raise ValueError("materialization cost evidence path is missing")
        evidence = load_strict_json(reference_paths[evidence_key])
        if canonical_sha256(evidence) != ledger[evidence_key]:
            raise ValueError("materialization cost evidence hash differs")
        _validate_materialization_cost_evidence(evidence)
        _verify_materialization_artifacts(evidence)
        expected_artifact = (
            evidence["deterministic_artifact_sha256"]
            if ledger["stage"].removesuffix("_boundary") == "rq3_deterministic_tuning"
            else evidence["learned_artifact_sha256"]
        )
        if any(
            row["job"]["objective"]["selector_artifact_sha256"] != expected_artifact
            for row in ledger["rows"]
        ):
            raise ValueError("RQ3 ledger does not consume the measured artifact")
        visited.add(evidence_key)
    pending = {
        key
        for key in ledger
        if key.endswith("_semantics_manifest_sha256")
        or key == "selected_control_manifest_sha256"
    }
    expected_kinds = {
        "control_semantics_manifest_sha256": "g4_control_semantics",
        "selected_control_manifest_sha256": "g4_selected_control",
        "treatment_semantics_manifest_sha256": "g4_treatment_semantics",
    }
    expected_hashes = {key: cast(str, ledger[key]) for key in pending}
    documents: dict[str, dict[str, Any]] = {}
    historical_hashes: set[str] = set()
    if compatibility_path is not None:
        compatibility = load_strict_json(compatibility_path)
        _verify_current_semantics(compatibility)
        if compatibility.get("version") != 2 or compatibility.get("kind") != (
            "g4_treatment_semantics"
        ):
            raise ValueError("historical compatibility manifest differs")
        historical_hashes.update(
            identity["sha256"]
            for identity in compatibility["historical_lineage"].values()
        )
    while pending:
        key = min(
            pending,
            key=lambda candidate: (
                candidate != "treatment_semantics_manifest_sha256",
                candidate,
            ),
        )
        pending.remove(key)
        if key in visited:
            continue
        if key not in reference_paths:
            raise ValueError(f"semantics manifest path is missing for {key}")
        path = reference_paths[key]
        try:
            document = load_strict_json(path)
        except ValueError as error:
            raise ValueError(
                f"semantics manifest hash cannot be verified: {path}"
            ) from error
        actual = canonical_sha256(document)
        if actual != expected_hashes[key]:
            raise ValueError(
                f"semantics manifest hash differs for {key}: "
                f"expected {expected_hashes[key]}, got {actual}"
            )
        if document.get("kind") != expected_kinds[key]:
            raise ValueError(f"semantics manifest kind differs for {key}")
        if actual in historical_hashes:
            _verify_historical_semantics(document)
        else:
            _verify_current_semantics(document)
        if document.get("version") == 2 and document.get("kind") == (
            "g4_treatment_semantics"
        ):
            historical_hashes.update(
                identity["sha256"]
                for identity in document["historical_lineage"].values()
            )
        documents[key] = document
        visited.add(key)
        linked = {
            "g4_selected_control": ("control_semantics_manifest_sha256",),
            "g4_treatment_semantics": ("selected_control_manifest_sha256",),
        }.get(document["kind"], ())
        if document.get("version") == 2 and document.get("kind") == (
            "g4_treatment_semantics"
        ):
            linked = ()
        for linked_key in linked:
            linked_hash = document[linked_key]
            previous = expected_hashes.get(linked_key)
            if previous is not None and previous != linked_hash:
                raise ValueError(f"conflicting semantics hash for {linked_key}")
            expected_hashes[linked_key] = linked_hash
            pending.add(linked_key)
    stage = ledger["stage"]
    selected = documents.get("selected_control_manifest_sha256")
    if selected is not None:
        parameters = selected["selection"]["canonical_parameters"]
        if stage.startswith(("rq1_", "rq2_", "rq3_")):
            if any(
                row["job"]["dataloader"]["batch_size"] != parameters["batch_size"]
                for row in ledger["rows"]
            ):
                raise ValueError("treatment batch differs from selected control")
            if not stage.endswith("_boundary"):
                anchor = ledger["rows"][0]["job"]
                if (
                    anchor["embedding_learning_rate"]
                    != parameters["embedding_learning_rate"]
                    or anchor["deep_learning_rate"] != parameters["deep_learning_rate"]
                    or anchor["lr_schedule_horizon_epochs"]
                    != parameters["lr_schedule_horizon_epochs"]
                ):
                    raise ValueError("treatment anchor differs from selected control")
    if stage == "control_tuning":
        anchor = load_control_manifest()["anchor"]
        first = ledger["rows"][0]["job"]
        if (
            first["dataloader"]["batch_size"] != anchor["batch_size"]
            or first["embedding_learning_rate"] != anchor["embedding_learning_rate"]
            or first["deep_learning_rate"] != anchor["deep_learning_rate"]
            or first["lr_schedule_horizon_epochs"]
            != anchor["lr_schedule_horizon_epochs"]
        ):
            raise ValueError("control tuning anchor differs from approved manifest")
    if set(reference_paths) != visited:
        raise ValueError("semantics manifest paths contain unknown references")


def write_frozen_ledger(path: Path, document: dict[str, Any]) -> None:
    _validate_ledger(document)
    content = canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"frozen ledger differs: {path}")
        return
    path.write_bytes(content)
    path.chmod(0o444)


def resolve_ledger_row(path: Path, row_id: str) -> dict[str, Any]:
    ledger = load_ledger(path)
    matches = [row for row in ledger["rows"] if row["id"] == row_id]
    if len(matches) != 1:
        raise ValueError(f"ledger row {row_id!r} is not approved")
    return cast(dict[str, Any], matches[0])


@dataclass(frozen=True)
class CompiledJob:
    ledger_sha256: str
    row_id: str
    job: dict[str, Any]

    @classmethod
    def from_row(cls, *, ledger_sha256: str, row: dict[str, Any]) -> "CompiledJob":
        return cls(
            ledger_sha256=ledger_sha256,
            row_id=cast(str, row["id"]),
            job=cast(dict[str, Any], row["job"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_sha256": self.ledger_sha256,
            "row_id": self.row_id,
            "job": self.job,
        }


def verify_compiled_job(compiled: CompiledJob, ledger_path: Path) -> None:
    ledger = load_ledger(ledger_path)
    if compiled.ledger_sha256 != ledger["sha256"]:
        raise ValueError("compiled job references a different ledger hash")
    row = resolve_ledger_row(ledger_path, compiled.row_id)
    if compiled.job != row["job"]:
        raise ValueError("compiled job differs from its frozen ledger row")
