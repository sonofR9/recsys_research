from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from experiments.g3_pretrained_item_embeddings.analysis.rq5_collection import (
    RQ5_OUTCOME_EVIDENCE_PATH,
)

from .constants import APPROVED_PROTOCOL_SHA256
from .rq3_post_boundary import Rq3ArtifactContract
from .rq5_initial import (
    RQ5_ARTIFACT_CONTRACTS,
    RQ5_INITIAL_LEDGER_LOGICAL_SHA256,
    RQ5_INITIAL_LEDGER_PATH,
    Rq5FileReference,
    load_rq5_initial_ledger,
    verify_rq5_initial_input_files,
)


RQ5_MECHANISM_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq5_selected_gate_mechanism_reproductions.json"
)
RQ5_OUTCOME_EVIDENCE_LOGICAL_SHA256 = (
    "a42314f93ffad3c51d6ea43a7e7c07d87486e6cadb7bb03f60c75be98ca4d442"
)
RQ5_MECHANISM_ARTIFACT_CONTRACTS = (
    *(
        contract
        if contract.name != "job_contract"
        else Rq3ArtifactContract(
            "job_contract",
            "g3_rq5_mechanism_job.json",
            contract.required_keys,
            contract.schema_versions,
        )
        for contract in RQ5_ARTIFACT_CONTRACTS
    ),
    Rq3ArtifactContract(
        "gate_diagnostics",
        "g3_gate_diagnostics.json",
        (
            "schema_version",
            "frequency_terciles",
            "training_count_reference",
            "slice_membership_reference",
            "frequency_input_parity",
            "epochs",
        ),
        (1,),
    ),
)


@dataclass(frozen=True)
class Rq5MechanismJob:
    id: str
    source_row_id: str
    family_id: str
    run_name: str
    content_gate: str
    gate_hidden_dim: int | None
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    history_hidden_dim: int
    source_best_epoch: int
    source_metrics: tuple[tuple[str, float], ...]
    source_slice_recall_at_100: tuple[tuple[str, float], ...]

    @property
    def batch_size(self) -> int:
        return 512

    @property
    def seed(self) -> int:
        return 42

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "rq5_content_gate",
            "stage": "rq5_selected_gate_mechanism_reproductions",
            "role": "evidence_only_exact_selected_config_reproduction",
            "run_name": self.run_name,
            "source_row_id": self.source_row_id,
            "selection_eligible": False,
            "representation": {
                "history_representation": "id_content",
                "history_hidden_dim": self.history_hidden_dim,
                "catalog_representation": "learned_id",
                "content_gate": self.content_gate,
                "gate_hidden_dim": self.gate_hidden_dim,
                "gate_input": (
                    "standardized_log1p_training_count"
                    if self.content_gate == "frequency"
                    else None
                ),
                "gate_activation": "sigmoid",
                "content_attachment": "before_id_content_densenet",
            },
            "dataset": {
                "size": "native-50m",
                "source": "likes",
                "event_limit": 50_000_000,
                "sampling": "none",
                "batch_size": 512,
                "seed": 42,
            },
            "training": {
                "batch_size": 512,
                "seed": 42,
                "embedding_learning_rate": self.embedding_learning_rate,
                "deep_learning_rate": self.deep_learning_rate,
                "horizon_epochs": self.horizon_epochs,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
            "parity_target": {
                "best_epoch": self.source_best_epoch,
                "metrics": dict(self.source_metrics),
                "slice_recall_at_100": dict(self.source_slice_recall_at_100),
            },
        }


@dataclass(frozen=True)
class Rq5MechanismLedger:
    initial_ledger: Rq5FileReference
    outcome_evidence: Rq5FileReference
    rows: tuple[Rq5MechanismJob, ...]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    @property
    def physical_rows(self) -> tuple[Rq5MechanismJob, ...]:
        return self.rows

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "g3_rq5_selected_gate_mechanism_reproductions",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": {
                "initial_ledger": self.initial_ledger.to_dict(),
                "premechanism_outcome_evidence": self.outcome_evidence.to_dict(),
            },
            "interpretation_contract": {
                "selection_eligible": False,
                "require_metric_parity_before_mechanism_interpretation": True,
                "purpose": "explain_predeclared_tail_acceptance_failure",
            },
            "opportunity_accounting": {
                "logical": 2,
                "physical": 2,
            },
            "artifact_contracts": [
                contract.to_dict() for contract in RQ5_MECHANISM_ARTIFACT_CONTRACTS
            ],
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq5_mechanism_ledger(*, root: Path) -> Rq5MechanismLedger:
    root = root.resolve(strict=True)
    initial_path = root / RQ5_INITIAL_LEDGER_PATH
    initial = load_rq5_initial_ledger(initial_path)
    if initial.sha256 != RQ5_INITIAL_LEDGER_LOGICAL_SHA256:
        raise ValueError("RQ5 mechanism initial ledger changed")
    outcome_path = root / RQ5_OUTCOME_EVIDENCE_PATH
    outcome = _load_logical_document(
        root,
        outcome_path,
        expected_sha256=RQ5_OUTCOME_EVIDENCE_LOGICAL_SHA256,
    )
    if (
        outcome.get("kind") != "g3_rq5_outcome_premechanism_native50m"
        or outcome.get("selection_resolved") is not True
        or outcome.get("mechanism_evidence")
        != {
            "status": "pending_targeted_selected_config_reproductions",
            "final_closure_allowed": False,
        }
    ):
        raise ValueError("RQ5 premechanism outcome is not the approved pending state")
    global_selected = outcome["global_selection"]["selected"]
    frequency_selected = outcome["frequency_selection"]["selected"]
    rows = (
        _job_from_selected(
            selected=global_selected,
            id="rq5_gate_mechanism:01",
            source_row_id="rq5_global_gate:10",
            run_name="g3_rq5_mechanism_global_selected_native50m",
            family_id="rq5_global_gate",
            content_gate="global",
            gate_hidden_dim=None,
        ),
        _job_from_selected(
            selected=frequency_selected,
            id="rq5_gate_mechanism:02",
            source_row_id="rq5_frequency_gate:04",
            run_name="g3_rq5_mechanism_frequency_width8_selected_native50m",
            family_id="rq5_frequency_gate",
            content_gate="frequency",
            gate_hidden_dim=8,
        ),
    )
    ledger = Rq5MechanismLedger(
        initial_ledger=_reference(root, initial_path, initial.sha256),
        outcome_evidence=_reference(
            root, outcome_path, RQ5_OUTCOME_EVIDENCE_LOGICAL_SHA256
        ),
        rows=rows,
    )
    _validate_program(ledger)
    return ledger


def load_rq5_mechanism_ledger(
    path: Path, *, root: Path, expected_ledger_sha256: str | None = None
) -> Rq5MechanismLedger:
    document = _load_json(path)
    payload = {name: value for name, value in document.items() if name != "sha256"}
    logical_sha256 = document.get("sha256")
    if (
        not isinstance(logical_sha256, str)
        or _canonical_sha256(payload) != logical_sha256
        or (
            expected_ledger_sha256 is not None
            and logical_sha256 != expected_ledger_sha256
        )
    ):
        raise ValueError("RQ5 mechanism ledger logical SHA changed")
    rebuilt = compile_rq5_mechanism_ledger(root=root)
    if rebuilt.sha256 != logical_sha256 or not _exact_json_equal(
        document, rebuilt.to_dict()
    ):
        raise ValueError("RQ5 mechanism ledger differs from frozen inputs")
    return rebuilt


def persist_rq5_mechanism_ledger(
    path: Path, ledger: Rq5MechanismLedger, *, root: Path
) -> Path:
    root = root.resolve(strict=True)
    destination = (root / RQ5_MECHANISM_LEDGER_PATH).resolve()
    if path.resolve() != destination or destination.is_symlink():
        raise ValueError("RQ5 mechanism ledger destination is not canonical")
    rebuilt = compile_rq5_mechanism_ledger(root=root)
    if rebuilt != ledger:
        raise ValueError("RQ5 mechanism ledger differs from authenticated inputs")
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 mechanism ledger differs: {destination}")
    return destination


def verify_rq5_mechanism_inputs(
    root: Path, ledger: Rq5MechanismLedger
) -> Path:
    root = root.resolve(strict=True)
    for reference in (ledger.initial_ledger, ledger.outcome_evidence):
        if _reference(root, root / reference.path, reference.logical_sha256) != reference:
            raise ValueError(f"RQ5 mechanism input changed: {reference.path}")
    initial = load_rq5_initial_ledger(root / ledger.initial_ledger.path)
    return verify_rq5_initial_input_files(root, initial)


def _job_from_selected(
    *,
    selected: object,
    id: str,
    source_row_id: str,
    run_name: str,
    family_id: str,
    content_gate: str,
    gate_hidden_dim: int | None,
) -> Rq5MechanismJob:
    if not isinstance(selected, dict) or selected.get("row_id") != source_row_id:
        raise ValueError(f"RQ5 selected source changed for {source_row_id}")
    metrics = selected.get("metrics")
    slices = selected.get("slices")
    if not isinstance(metrics, dict) or not isinstance(slices, dict):
        raise ValueError(f"RQ5 selected metrics or slices absent for {source_row_id}")
    return Rq5MechanismJob(
        id=id,
        source_row_id=source_row_id,
        family_id=family_id,
        run_name=run_name,
        content_gate=content_gate,
        gate_hidden_dim=gate_hidden_dim,
        embedding_learning_rate=float(selected["embedding_learning_rate"]),
        deep_learning_rate=float(selected["deep_learning_rate"]),
        horizon_epochs=int(selected["horizon_epochs"]),
        history_hidden_dim=128,
        source_best_epoch=int(selected["best_epoch"]),
        source_metrics=tuple(
            (name, float(metrics[name]))
            for name in ("recall@100", "ndcg@100")
        ),
        source_slice_recall_at_100=tuple(
            (name, float(slices[name]["recall@100"]))
            for name in ("tail", "mid", "head")
        ),
    )


def _validate_program(ledger: Rq5MechanismLedger) -> None:
    expected = (
        (
            "rq5_gate_mechanism:01",
            "rq5_global_gate:10",
            "global",
            None,
            0.12305770976863895,
            0.011338899623382975,
            40,
            22,
            0.09343741789063942,
            0.034549496580629245,
            (0.013108593728748766, 0.03502721620566738, 0.1310283868970138),
        ),
        (
            "rq5_gate_mechanism:02",
            "rq5_frequency_gate:04",
            "frequency",
            8,
            0.11386115952375567,
            0.021533016497665633,
            25,
            22,
            0.09460930081879586,
            0.03281573437633283,
            (0.011429165208234974, 0.0360954826443042, 0.1293747193354421),
        ),
    )
    actual = tuple(
        (
            row.id,
            row.source_row_id,
            row.content_gate,
            row.gate_hidden_dim,
            row.embedding_learning_rate,
            row.deep_learning_rate,
            row.horizon_epochs,
            row.source_best_epoch,
            dict(row.source_metrics)["recall@100"],
            dict(row.source_metrics)["ndcg@100"],
            tuple(dict(row.source_slice_recall_at_100).values()),
        )
        for row in ledger.rows
    )
    if actual != expected or any(
        row.batch_size != 512
        or row.seed != 42
        or row.history_hidden_dim != 128
        for row in ledger.rows
    ):
        raise ValueError("RQ5 mechanism lost its exact two-run approved design")


def _reference(root: Path, path: Path, logical_sha256: str) -> Rq5FileReference:
    if path.is_symlink():
        raise ValueError("RQ5 mechanism input must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("RQ5 mechanism input escapes project root")
    return Rq5FileReference(
        path=str(resolved.relative_to(root)),
        size_bytes=resolved.stat().st_size,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        logical_sha256=logical_sha256,
    )


def _load_logical_document(
    root: Path, path: Path, *, expected_sha256: str
) -> dict[str, object]:
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("RQ5 mechanism input path is invalid")
    document = _load_json(path)
    payload = {name: value for name, value in document.items() if name != "sha256"}
    if document.get("sha256") != expected_sha256 or _canonical_sha256(
        payload
    ) != expected_sha256:
        raise ValueError("RQ5 mechanism logical input changed")
    return document


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load RQ5 mechanism JSON {path}") from error
    if not isinstance(value, dict):
        raise ValueError("RQ5 mechanism JSON must be an object")
    return value


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")
