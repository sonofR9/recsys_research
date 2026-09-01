from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from dcn.training_metadata import (
    GENERATION_TRAINING_SEMANTICS_REVISION,
    NEGATIVE_SAMPLING_SEMANTICS_REVISION,
    TIMESTAMP_BIN_SEMANTICS_REVISION,
)
from experiments.g4_future_items.configs.control import (
    build_anchor_control,
    control_runtime_projection,
)
from experiments.g4_future_items.protocol.manifest import (
    PROJECT_ROOT,
    build_control_semantics_manifest,
    canonical_bytes,
    canonical_sha256,
    compile_control_tuning_ledger,
    expected_control_source_paths,
    resolve_control_data_identity,
    source_manifest,
    write_frozen_ledger,
    write_frozen_manifest,
)
from utils.global_config import config as global_config


PROTOCOL_ROOT = Path(__file__).resolve().parents[1] / "protocol"
CONTROL_SEMANTICS_PATH = PROTOCOL_ROOT / "control_semantics_manifest.json"
CONTROL_TUNING_LEDGER_PATH = PROTOCOL_ROOT / "ledgers" / "control_tuning.json"


@dataclass(frozen=True)
class ControlFreeze:
    semantics: dict[str, Any]
    ledger: dict[str, Any]


def _training_semantics_revisions(experiment: Any) -> dict[str, int]:
    projection = control_runtime_projection(experiment)
    fixed = projection["fixed"]
    if (
        fixed["training"]["training_semantics_revision"]
        != GENERATION_TRAINING_SEMANTICS_REVISION
    ):
        raise RuntimeError("control training semantics revision is stale")
    if fixed["model"]["timestamp_delta"] != "bins":
        raise RuntimeError("control timestamp semantics are not revisioned")
    return {
        "generation": GENERATION_TRAINING_SEMANTICS_REVISION,
        "negative_sampling": NEGATIVE_SAMPLING_SEMANTICS_REVISION,
        "timestamp_bins": TIMESTAMP_BIN_SEMANTICS_REVISION,
    }


def compile_control_freeze(experiment: Any | None = None) -> ControlFreeze:
    anchor = build_anchor_control()
    global_config.initialize(Path(anchor.base_path))
    source_paths = expected_control_source_paths()
    semantics = build_control_semantics_manifest(
        source_paths=source_paths,
        sources=source_manifest(PROJECT_ROOT, source_paths),
        data_identity=resolve_control_data_identity(
            anchor if experiment is None else experiment
        ),
        training_semantics_revisions=_training_semantics_revisions(anchor),
    )
    ledger = compile_control_tuning_ledger(canonical_sha256(semantics))
    return ControlFreeze(semantics=semantics, ledger=ledger)


def _destination_status(path: Path, document: dict[str, Any], kind: str) -> str:
    if not path.exists():
        return "absent"
    if path.read_bytes() != canonical_bytes(document):
        raise RuntimeError(f"frozen {kind} differs: {path}")
    return "matching"


def freeze_control(
    *,
    semantics_path: Path = CONTROL_SEMANTICS_PATH,
    ledger_path: Path = CONTROL_TUNING_LEDGER_PATH,
    experiment: Any | None = None,
    write: bool = False,
) -> dict[str, Any]:
    compiled = compile_control_freeze(experiment)
    destinations = {
        "control_semantics_manifest": _destination_status(
            semantics_path, compiled.semantics, "manifest"
        ),
        "control_tuning_ledger": _destination_status(
            ledger_path, compiled.ledger, "ledger"
        ),
    }
    if write:
        write_frozen_manifest(semantics_path, compiled.semantics)
        write_frozen_ledger(ledger_path, compiled.ledger)
    return {
        "write": write,
        "control_semantics_manifest_sha256": canonical_sha256(compiled.semantics),
        "control_tuning_ledger_sha256": compiled.ledger["sha256"],
        "paths": {
            "control_semantics_manifest": str(semantics_path.resolve()),
            "control_tuning_ledger": str(ledger_path.resolve()),
        },
        "destinations": destinations,
        "control_tuning_rows": len(compiled.ledger["rows"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="freeze the verified documents; omit for a read-only preview",
    )
    arguments = parser.parse_args(argv)
    print(json.dumps(freeze_control(write=arguments.write), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
