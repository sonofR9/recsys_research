from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.g1_sasrec_item_ids_likes.analysis import select_native_500m


EXPERIMENT = Path(__file__).resolve().parent.parent
MANIFEST = EXPERIMENT / "evidence/native_500m_provenance.json"
EVIDENCE_FILES = ("training_metadata.json", "final_metrics.json")
EXPECTED_RATES = ("0.001", "0.002")
EXPECTED_WINNER = (
    "g1_transfer_batchscale_b1280_e0p001_d0p002_cap60_ts2_r2_50m"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selector_record(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_digest": selection["source_digest"],
        "source_id": selection["source_id"],
        "source_artifacts": selection["source_artifacts"],
        "winner_run": selection["winner_run"],
        "embedding_lr": selection["embedding_lr"],
        "deep_lr": selection["deep_lr"],
    }


def build_manifest(directory: Path, selection: dict[str, Any]) -> dict[str, Any]:
    selector = _selector_record(selection)
    source_digest = selector["source_digest"]
    if (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
    ):
        raise ValueError("selector source digest must be lowercase SHA-256")
    if selector["source_id"] != source_digest[:12]:
        raise ValueError("selector source id is not its digest prefix")
    if selector["source_artifacts"] != 42:
        raise ValueError("native selector must bind all 42 approved artifacts")
    if (selector["embedding_lr"], selector["deep_lr"]) != EXPECTED_RATES:
        raise ValueError("native selector rates do not match the accepted winner")
    if selector["winner_run"] != EXPECTED_WINNER:
        raise ValueError("native selector winner run is not the accepted source")
    expected_identity = (
        f"selected_native50_{selector['source_id']}_e0p001_d0p002_"
    )
    if not directory.name.startswith(f"g1_transfer_{expected_identity}"):
        raise ValueError("target run does not encode the selected source and rates")
    return {
        "schema_version": 1,
        "selector": selector,
        "target": {
            "run": directory.name,
            "files": {name: _sha256(directory / name) for name in EVIDENCE_FILES},
        },
    }


def validate(
    directory: Path,
    *,
    selection: dict[str, Any] | None = None,
    manifest_path: Path = MANIFEST,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if set(manifest) != {"schema_version", "selector", "target"}:
        raise ValueError("native provenance manifest has unexpected fields")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported native provenance manifest schema")
    selector = manifest["selector"]
    if set(selector) != {
        "source_digest",
        "source_id",
        "source_artifacts",
        "winner_run",
        "embedding_lr",
        "deep_lr",
    }:
        raise ValueError("native provenance selector has unexpected fields")
    if selection is not None and selector != _selector_record(selection):
        raise ValueError("native provenance does not match the current selector")
    expected = build_manifest(directory, selector)
    if manifest != expected:
        raise ValueError("native provenance target or evidence hashes do not match")
    return manifest


def write_manifest(
    directory: Path,
    selection: dict[str, Any],
    *,
    manifest_path: Path = MANIFEST,
) -> dict[str, Any]:
    manifest = build_manifest(directory, selection)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--winner-run", required=True)
    parser.add_argument("--embedding-lr", required=True)
    parser.add_argument("--deep-lr", required=True)
    parser.add_argument("--source-artifacts", type=int, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    selection = {
        "source_digest": args.source_digest,
        "source_id": args.source_digest[:12],
        "source_artifacts": args.source_artifacts,
        "winner_run": args.winner_run,
        "embedding_lr": args.embedding_lr,
        "deep_lr": args.deep_lr,
    }
    if args.write:
        write_manifest(args.target, selection)
    validate(args.target, selection=selection)


if __name__ == "__main__":
    main()
