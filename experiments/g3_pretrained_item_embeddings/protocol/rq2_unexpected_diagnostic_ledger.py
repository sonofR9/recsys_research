from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.g3_pretrained_item_embeddings.analysis.rq1_results import (
    APPROVED_RQ1_EVIDENCE_SHA256,
    RQ1_EVIDENCE_PATH,
    load_rq1_evidence,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq2_content_deep_lr_boundary_results import (
    RQ2_CONTENT_DEEP_LR_BOUNDARY_EVIDENCE_PATH,
)

from .constants import APPROVED_PROTOCOL, APPROVED_PROTOCOL_SHA256


RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq2_unexpected_result_diagnostic.json"
)
APPROVED_RQ2_BOUNDARY_EVIDENCE_SHA256 = (
    "6fe504dc0e2207ebc0c2389b472a52063bbdff500892d2aaac4b4deac8f608ea"
)
RQ2_BOUNDARY_EVIDENCE_FILE_SHA256 = (
    "23d34eed361781fa121a969ee25bf662e7a13ec65d13ca97c4e8acc1aacc5175"
)
RQ1_EVIDENCE_FILE_SHA256 = (
    "632b521be737badf996f3dadf1a38ba6218fee7402d9eb3419fe02669845c90d"
)
APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256 = (
    "6214d7d00ccc27e6457c479f1a83597667784245bf6e362f87e51ef278de4edf"
)


@dataclass(frozen=True)
class EvidenceReference:
    path: str
    size_bytes: int
    sha256: str
    logical_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "logical_sha256": self.logical_sha256,
        }


@dataclass(frozen=True)
class Rq2UnexpectedDiagnosticJob:
    id: str
    run_name: str
    role: str
    representation_id: str
    history: str
    history_hidden_dim: int
    embedding_learning_rate: float
    deep_learning_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.representation_id,
            "phase": "unexpected_result_diagnostic",
            "run_name": self.run_name,
            "stage": "rq2_unexpected_result_diagnostic",
            "role": self.role,
            "reused_from": None,
            "representation": {
                "id": self.representation_id,
                "history": self.history,
                "catalog": "learned_item_id",
                "history_hidden_dim": self.history_hidden_dim,
                "separate_history_catalog_tables": True,
                "content_trainable": False,
                "content_width": 128,
            },
            "dataset": {
                "size": APPROVED_PROTOCOL.main_dataset_size,
                "source": "likes",
                "event_limit": 50_000_000,
                "sampling": "none",
                "minimum_user_interactions": 5,
                "validation_interval_seconds": 604800,
                "candidate_catalog": "full",
                "exclude_seen": False,
            },
            "training": {
                "batch_size": APPROVED_PROTOCOL.batch_size,
                "seed": APPROVED_PROTOCOL.seed,
                "embedding_learning_rate": self.embedding_learning_rate,
                "deep_learning_rate": self.deep_learning_rate,
                "horizon_epochs": 40,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
        }


@dataclass(frozen=True)
class Rq2UnexpectedDiagnosticLedger:
    schema_version: int
    kind: str
    protocol_sha256: str
    maximum_jobs: int
    boundary_evidence: EvidenceReference
    rq1_evidence: EvidenceReference
    source_assertions: dict[str, object]
    isolation_contract: dict[str, object]
    rows: tuple[Rq2UnexpectedDiagnosticJob, ...]

    @property
    def inputs(self) -> dict[str, EvidenceReference]:
        return {
            "boundary_evidence": self.boundary_evidence,
            "rq1_evidence": self.rq1_evidence,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._payload()).encode()).hexdigest()

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "maximum_jobs": self.maximum_jobs,
            "inputs": {name: value.to_dict() for name, value in self.inputs.items()},
            "source_assertions": self.source_assertions,
            "isolation_contract": self.isolation_contract,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq2_unexpected_diagnostic_ledger(
    root: Path,
) -> Rq2UnexpectedDiagnosticLedger:
    root = root.resolve(strict=True)
    boundary_reference = _evidence_reference(
        root,
        path=RQ2_CONTENT_DEEP_LR_BOUNDARY_EVIDENCE_PATH,
        size_bytes=135_971,
        file_sha256=RQ2_BOUNDARY_EVIDENCE_FILE_SHA256,
        logical_sha256=APPROVED_RQ2_BOUNDARY_EVIDENCE_SHA256,
    )
    rq1_reference = _evidence_reference(
        root,
        path=RQ1_EVIDENCE_PATH,
        size_bytes=57_213,
        file_sha256=RQ1_EVIDENCE_FILE_SHA256,
        logical_sha256=APPROVED_RQ1_EVIDENCE_SHA256,
    )
    boundary = _load_json(root / boundary_reference.path)
    boundary_payload = {
        name: value for name, value in boundary.items() if name != "sha256"
    }
    if (
        boundary.get("sha256")
        != hashlib.sha256(_canonical_json(boundary_payload).encode()).hexdigest()
    ):
        raise ValueError("RQ2 diagnostic boundary evidence logical hash changed")
    rq1 = load_rq1_evidence(root / rq1_reference.path)
    _validate_sources(boundary=boundary, rq1=rq1)
    rq1_embedding_lr = 0.2183583071089141
    rq1_deep_lr = 0.021004505318001004
    selected_embedding_lr = 0.3041556165944196
    selected_deep_lr = 0.014506684820055783
    rows = (
        Rq2UnexpectedDiagnosticJob(
            id="rq2_unexpected_diagnostic:01",
            run_name="g3_rq2_diag_concat_width_32_rq1_lrs_native50m",
            role="crossed_learning_rates",
            representation_id="rq2_content_concat",
            history="learned_item_id_plus_frozen_content",
            history_hidden_dim=32,
            embedding_learning_rate=rq1_embedding_lr,
            deep_learning_rate=rq1_deep_lr,
        ),
        Rq2UnexpectedDiagnosticJob(
            id="rq2_unexpected_diagnostic:02",
            run_name="g3_rq2_diag_zero_id_width_32_rq1_lrs_native50m",
            role="learned_id_branch_ablation",
            representation_id="rq2_content_zero_id",
            history="zero_frozen_item_id_plus_frozen_content",
            history_hidden_dim=32,
            embedding_learning_rate=rq1_embedding_lr,
            deep_learning_rate=rq1_deep_lr,
        ),
        Rq2UnexpectedDiagnosticJob(
            id="rq2_unexpected_diagnostic:03",
            run_name="g3_rq2_diag_concat_width_128_selected_lrs_native50m",
            role="bottleneck_capacity_check",
            representation_id="rq2_content_concat",
            history="learned_item_id_plus_frozen_content",
            history_hidden_dim=128,
            embedding_learning_rate=selected_embedding_lr,
            deep_learning_rate=selected_deep_lr,
        ),
    )
    ledger = Rq2UnexpectedDiagnosticLedger(
        schema_version=1,
        kind="g3_rq2_unexpected_result_diagnostic",
        protocol_sha256=APPROVED_PROTOCOL_SHA256,
        maximum_jobs=3,
        boundary_evidence=boundary_reference,
        rq1_evidence=rq1_reference,
        source_assertions={
            "unexpected_result": {
                "selected_concat_row_id": "rq2_content_concat:12",
                "selected_concat_recall_at_100": 0.08893693160875873,
                "selected_id_only_row_id": "rq2_id_only_densenet:12",
                "selected_id_only_recall_at_100": 0.09074562121371973,
                "concat_beats_id_only": False,
            },
            "rq1_learning_rates": {
                "source_row_id": "rq1_content_input:02",
                "embedding_learning_rate": rq1_embedding_lr,
                "deep_learning_rate": rq1_deep_lr,
                "horizon_epochs": 40,
            },
            "selected_concat_learning_rates": {
                "source_row_id": "rq2_content_concat:12",
                "embedding_learning_rate": selected_embedding_lr,
                "deep_learning_rate": selected_deep_lr,
                "horizon_epochs": 40,
            },
        },
        isolation_contract={
            "rows": [
                "rq2_unexpected_diagnostic:01",
                "rq2_unexpected_diagnostic:02",
            ],
            "common_initialization": "bit_identical_common_parameters",
            "isolated_factor": "learned_history_item_id_branch",
            "ablated_table": "instantiated_then_zeroed_after_global_initialization",
            "ablated_table_trainable": False,
            "ablated_table_in_optimizer": False,
            "id_only_padding_idx": 0,
        },
        rows=rows,
    )
    if (
        APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256 != "0" * 64
        and ledger.sha256 != APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256
    ):
        raise ValueError("approved RQ2 unexpected diagnostic ledger drifted")
    return ledger


def validate_rq2_unexpected_diagnostic_ledger_document(
    document: object, *, root: Path
) -> Rq2UnexpectedDiagnosticLedger:
    if not isinstance(document, dict):
        raise ValueError("RQ2 unexpected diagnostic ledger must be an object")
    expected = compile_rq2_unexpected_diagnostic_ledger(root)
    expected_document = expected.to_dict()
    _validate_exact_json_types(document, expected_document, path="ledger")
    if document != expected_document:
        raise ValueError("RQ2 unexpected diagnostic ledger differs from approval")
    return expected


def load_rq2_unexpected_diagnostic_ledger(
    path: Path, *, root: Path
) -> Rq2UnexpectedDiagnosticLedger:
    ledger = validate_rq2_unexpected_diagnostic_ledger_document(
        _load_json(path), root=root
    )
    if ledger.sha256 != APPROVED_RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_SHA256:
        raise ValueError("RQ2 unexpected diagnostic ledger is not approved")
    return ledger


def persist_rq2_unexpected_diagnostic_ledger(
    path: Path,
    ledger: Rq2UnexpectedDiagnosticLedger,
    *,
    root: Path,
) -> Path:
    validate_rq2_unexpected_diagnostic_ledger_document(ledger.to_dict(), root=root)
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(
                f"immutable RQ2 unexpected diagnostic ledger differs: {path}"
            )
    return path


def _validate_sources(
    *, boundary: Mapping[str, object], rq1: Mapping[str, object]
) -> None:
    selection = boundary.get("final_content_selection")
    comparison = boundary.get("final_rq2_comparison")
    rq1_selected = rq1.get("selected_treatment")
    if not all(
        isinstance(value, dict) for value in (selection, comparison, rq1_selected)
    ):
        raise ValueError("RQ2 diagnostic source evidence lacks selections")
    concat = selection.get("selected")
    id_only = comparison.get("id_only_densenet")
    if not isinstance(concat, dict) or not isinstance(id_only, dict):
        raise ValueError("RQ2 diagnostic boundary evidence lacks selected rows")
    if (
        boundary.get("sha256") != APPROVED_RQ2_BOUNDARY_EVIDENCE_SHA256
        or selection.get("status") != "resolved"
        or selection.get("boundary_decision", {}).get("required_actions") != []
        or concat.get("row_id") != "rq2_content_concat:12"
        or concat.get("capacity") != 32
        or concat.get("horizon_epochs") != 40
        or concat.get("embedding_learning_rate") != 0.3041556165944196
        or concat.get("deep_learning_rate") != 0.014506684820055783
        or concat.get("metrics", {}).get("recall@100") != 0.08893693160875873
        or id_only.get("row_id") != "rq2_id_only_densenet:12"
        or id_only.get("metrics", {}).get("recall@100") != 0.09074562121371973
        or comparison.get("content_beats_id_only") is not False
        or rq1.get("sha256") != APPROVED_RQ1_EVIDENCE_SHA256
        or rq1_selected.get("row_id") != "rq1_content_input:02"
        or rq1_selected.get("embedding_learning_rate") != 0.2183583071089141
        or rq1_selected.get("deep_learning_rate") != 0.021004505318001004
        or rq1_selected.get("horizon_epochs") != 40
    ):
        raise ValueError("RQ2 unexpected diagnostic source selection changed")


def _evidence_reference(
    root: Path,
    *,
    path: str,
    size_bytes: int,
    file_sha256: str,
    logical_sha256: str,
) -> EvidenceReference:
    target = _resolve_reference(root, path)
    if target.stat().st_size != size_bytes or _file_sha256(target) != file_sha256:
        raise ValueError(f"RQ2 diagnostic evidence file changed: {path}")
    return EvidenceReference(path, size_bytes, file_sha256, logical_sha256)


def _resolve_reference(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RQ2 diagnostic reference must be project-relative")
    path = root / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError(f"RQ2 diagnostic reference is not a project file: {value}")
    return path


def _validate_exact_json_types(actual: object, expected: object, *, path: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"RQ2 unexpected diagnostic {path} has an invalid JSON type")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"RQ2 unexpected diagnostic {path} has invalid keys")
        for name, expected_value in expected.items():
            _validate_exact_json_types(
                actual[name], expected_value, path=f"{path}.{name}"
            )
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"RQ2 unexpected diagnostic {path} has invalid length")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _validate_exact_json_types(
                actual_value, expected_value, path=f"{path}[{index}]"
            )


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot load RQ2 unexpected diagnostic input {path}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"RQ2 unexpected diagnostic input is not an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key {name!r}")
        result[name] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    ledger = compile_rq2_unexpected_diagnostic_ledger(root)
    path = root / RQ2_UNEXPECTED_DIAGNOSTIC_LEDGER_PATH
    if arguments.write:
        persist_rq2_unexpected_diagnostic_ledger(path, ledger, root=root)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": ledger.sha256,
                "jobs": len(ledger.rows),
                "status": "materialized" if arguments.write else "preview",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
