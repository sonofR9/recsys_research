from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from experiments.g3_pretrained_item_embeddings.analysis.rq5_collection import (
    RQ5_OUTCOME_EVIDENCE_PATH,
)
from experiments.g3_pretrained_item_embeddings.launchers import rq5_mechanism

from .constants import APPROVED_PROTOCOL_SHA256
from .rq3_post_boundary import Rq3ArtifactContract
from .rq5_initial import (
    RQ5_ARTIFACT_CONTRACTS,
    RQ5_INITIAL_LEDGER_LOGICAL_SHA256,
    RQ5_INITIAL_LEDGER_PATH,
    Rq5FileReference,
    Rq5PhysicalFileReference,
    load_rq5_initial_ledger,
    verify_rq5_initial_input_files,
)
from .rq5_mechanism_ledger import (
    RQ5_MECHANISM_LEDGER_PATH,
    load_rq5_mechanism_ledger,
)


RQ5_FREQUENCY_V2_LEDGER_PATH = (
    "experiments/g3_pretrained_item_embeddings/protocol/ledgers/"
    "rq5_frequency_gate_fp32_p09_v2_initial.json"
)
RQ5_MECHANISM_BATCH_ID = "70f1b7d1589e4ac7a6208844d8ef2b28"
RQ5_PRECISION_PROBE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq5_frequency_gate_bfloat16_precision_probe_a100.json"
)
RQ5_PRECISION_PROBE_LOGICAL_SHA256 = (
    "aa3e2e82155e1031a43c93f9be1523cda47bbe972a6757a07848aaf258d3a582"
)
RQ5_FREQUENCY_V2_ARTIFACT_CONTRACTS = (
    *(
        contract
        if contract.name != "job_contract"
        else Rq3ArtifactContract(
            "job_contract",
            "g3_rq5_frequency_v2_job.json",
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
class Rq5FrequencyV2Job:
    id: str
    run_name: str
    gate_hidden_dim: int
    embedding_learning_rate: float
    deep_learning_rate: float

    @property
    def family_id(self) -> str:
        return "rq5_frequency_gate_v2"

    @property
    def content_gate(self) -> str:
        return "frequency"

    @property
    def history_hidden_dim(self) -> int:
        return 128

    @property
    def batch_size(self) -> int:
        return 512

    @property
    def seed(self) -> int:
        return 42

    @property
    def horizon_epochs(self) -> int:
        return 25

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family_id": self.family_id,
            "phase": "rq5_content_gate",
            "stage": "rq5_frequency_gate_fp32_p09_v2_initial",
            "role": "corrected_frequency_gate_search",
            "run_name": self.run_name,
            "replaces_invalid_family": "rq5_frequency_gate",
            "representation": {
                "history_representation": "id_content",
                "history_hidden_dim": self.history_hidden_dim,
                "catalog_representation": "learned_id",
                "content_gate": "frequency",
                "gate_hidden_dim": self.gate_hidden_dim,
                "gate_input": "standardized_log1p_training_count",
                "gate_activation": "sigmoid",
                "content_attachment": "before_id_content_densenet",
                "frequency_gate_semantics": "fp32_p09_v2",
                "initial_probability": 0.9,
                "math_dtype": "float32",
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
                "horizon_epochs": 25,
                "validate_every_epoch": True,
                "restore_best_validation_epoch": True,
            },
        }


@dataclass(frozen=True)
class Rq5FrequencyV2Ledger:
    initial_ledger: Rq5FileReference
    premechanism_outcome: Rq5FileReference
    mechanism_ledger: Rq5FileReference
    precision_probe: Rq5FileReference
    mechanism_batch: Rq5PhysicalFileReference
    defect_artifacts: tuple[tuple[str, Rq5PhysicalFileReference], ...]
    rows: tuple[Rq5FrequencyV2Job, ...]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self._payload())

    @property
    def physical_rows(self) -> tuple[Rq5FrequencyV2Job, ...]:
        return self.rows

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "g3_rq5_frequency_gate_fp32_p09_v2_initial",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": {
                "initial_ledger": self.initial_ledger.to_dict(),
                "premechanism_outcome": self.premechanism_outcome.to_dict(),
                "mechanism_ledger": self.mechanism_ledger.to_dict(),
                "precision_probe": self.precision_probe.to_dict(),
                "mechanism_batch": self.mechanism_batch.to_dict()
                | {"batch_id": RQ5_MECHANISM_BATCH_ID},
                "defect_artifacts": {
                    name: reference.to_dict()
                    for name, reference in self.defect_artifacts
                },
            },
            "defect_resolution": {
                "invalid_family": "rq5_frequency_gate",
                "reader_and_tuning_eligible": False,
                "metric_parity": "exact_bytes_for_metrics_rankings_metadata_and_training_diagnostics",
                "observed_failure": "zero_gate_gradients_all_epochs_and_constant_p09999_output",
                "root_cause": "bfloat16_sigmoid_saturation",
                "corrected_semantics": "fp32_p09_v2",
                "valid_fixed_and_global_evidence_frozen": True,
            },
            "opportunity_accounting": {
                "widths": [4, 8, 16],
                "learning_rate_probes_per_width": 3,
                "logical": 9,
                "physical": 9,
                "deferred_selected_width_horizons": [15, 25, 40],
            },
            "artifact_contracts": [
                contract.to_dict() for contract in RQ5_FREQUENCY_V2_ARTIFACT_CONTRACTS
            ],
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload() | {"sha256": self.sha256}


def compile_rq5_frequency_v2_ledger(*, root: Path) -> Rq5FrequencyV2Ledger:
    root = root.resolve(strict=True)
    initial_path = root / RQ5_INITIAL_LEDGER_PATH
    initial = load_rq5_initial_ledger(initial_path)
    if initial.sha256 != RQ5_INITIAL_LEDGER_LOGICAL_SHA256:
        raise ValueError("RQ5 frequency v2 initial ledger changed")
    mechanism_path = root / RQ5_MECHANISM_LEDGER_PATH
    mechanism = load_rq5_mechanism_ledger(
        mechanism_path,
        root=root,
        expected_ledger_sha256=rq5_mechanism.RQ5_MECHANISM_LEDGER_LOGICAL_SHA256,
    )
    outcome_path = root / RQ5_OUTCOME_EVIDENCE_PATH
    outcome = _load_logical(root, outcome_path)
    probe_path = root / RQ5_PRECISION_PROBE_PATH
    probe = _load_logical(root, probe_path)
    _validate_probe(root, probe)
    batch_path = (
        root
        / "generated/training-queue-service/batches"
        / f"{RQ5_MECHANISM_BATCH_ID}.json"
    )
    batch = _load_json(batch_path)
    if batch.get("sealed") is not True or len(batch.get("jobs", [])) != 2:
        raise ValueError("RQ5 mechanism batch is not the exact sealed pair")
    stable = (
        "final_metrics.json",
        "ranking_evidence.pt",
        "top_item_rankings.json",
        "training_metadata.json",
        "g3_training_diagnostics.json",
    )
    selected = {
        "global": outcome["global_selection"]["selected"],
        "frequency": outcome["frequency_selection"]["selected"],
    }
    artifacts = []
    for row, job_id in zip(mechanism.rows, batch["jobs"], strict=True):
        queue_path = root / "generated/training-queue-service/completed" / f"{job_id}.json"
        queue = _load_json(queue_path)
        if (
            queue.get("exit_code") != 0
            or queue.get("batch_id") != RQ5_MECHANISM_BATCH_ID
            or queue.get("run") != row.run_name
        ):
            raise ValueError("RQ5 mechanism queue identity changed")
        key = "frequency" if row.content_gate == "frequency" else "global"
        source_artifacts = selected[key]["artifacts"]
        run_dir = root / "generated/logs" / row.run_name
        for filename in stable:
            source_name = {
                "final_metrics.json": "final_metrics",
                "ranking_evidence.pt": "ranking_evidence",
                "top_item_rankings.json": "top_item_rankings",
                "training_metadata.json": "training_metadata",
                "g3_training_diagnostics.json": "training_diagnostics",
            }[filename]
            path = run_dir / filename
            if _physical(root, path).sha256 != source_artifacts[source_name]["sha256"]:
                raise ValueError("RQ5 mechanism reproduction lost exact source parity")
            artifacts.append((f"{key}_{source_name}", _physical(root, path)))
        gate_path = run_dir / "g3_gate_diagnostics.json"
        _validate_gate_diagnostics(_load_json(gate_path), kind=key, horizon=row.horizon_epochs)
        artifacts.extend(
            (
                (f"{key}_gate_diagnostics", _physical(root, gate_path)),
                (f"{key}_queue_job", _physical(root, queue_path)),
            )
        )
    legacy_rows = [
        row for row in initial.rows if row.family_id == "rq5_frequency_gate"
    ]
    if len(legacy_rows) != 9:
        raise ValueError("RQ5 legacy frequency coordinate surface changed")
    rows = tuple(
        Rq5FrequencyV2Job(
            id=f"rq5_frequency_gate_v2:{index:02d}",
            run_name=(
                f"g3_rq5_frequency_gate_v2_width_{row.gate_hidden_dim}_"
                f"trial_{index:02d}_native50m"
            ),
            gate_hidden_dim=int(row.gate_hidden_dim),
            embedding_learning_rate=row.embedding_learning_rate,
            deep_learning_rate=row.deep_learning_rate,
        )
        for index, row in enumerate(legacy_rows, start=1)
    )
    ledger = Rq5FrequencyV2Ledger(
        initial_ledger=_reference(root, initial_path, initial.sha256),
        premechanism_outcome=_reference(
            root, outcome_path, str(outcome["sha256"])
        ),
        mechanism_ledger=_reference(root, mechanism_path, mechanism.sha256),
        precision_probe=_reference(root, probe_path, str(probe["sha256"])),
        mechanism_batch=_physical(root, batch_path),
        defect_artifacts=tuple(artifacts),
        rows=rows,
    )
    _validate_program(ledger)
    return ledger


def load_rq5_frequency_v2_ledger(
    path: Path, *, root: Path, expected_ledger_sha256: str | None = None
) -> Rq5FrequencyV2Ledger:
    document = _load_json(path)
    payload = {key: value for key, value in document.items() if key != "sha256"}
    logical = document.get("sha256")
    if (
        not isinstance(logical, str)
        or _canonical_sha256(payload) != logical
        or (expected_ledger_sha256 is not None and logical != expected_ledger_sha256)
    ):
        raise ValueError("RQ5 frequency v2 ledger logical SHA changed")
    rebuilt = compile_rq5_frequency_v2_ledger(root=root)
    if rebuilt.sha256 != logical or not _exact_json_equal(
        document, rebuilt.to_dict()
    ):
        raise ValueError("RQ5 frequency v2 ledger differs from frozen inputs")
    return rebuilt


def persist_rq5_frequency_v2_ledger(
    path: Path, ledger: Rq5FrequencyV2Ledger, *, root: Path
) -> Path:
    root = root.resolve(strict=True)
    destination = (root / RQ5_FREQUENCY_V2_LEDGER_PATH).resolve()
    if path.resolve() != destination or destination.is_symlink():
        raise ValueError("RQ5 frequency v2 ledger destination is not canonical")
    if compile_rq5_frequency_v2_ledger(root=root) != ledger:
        raise ValueError("RQ5 frequency v2 ledger differs from authenticated inputs")
    content = (_canonical_json(ledger.to_dict()) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 frequency v2 ledger differs: {destination}")
    return destination


def verify_rq5_frequency_v2_inputs(
    root: Path, ledger: Rq5FrequencyV2Ledger
) -> Path:
    rebuilt = compile_rq5_frequency_v2_ledger(root=root)
    if rebuilt != ledger:
        raise ValueError("RQ5 frequency v2 bound inputs changed")
    initial = load_rq5_initial_ledger(root / ledger.initial_ledger.path)
    return verify_rq5_initial_input_files(root, initial)


def _validate_gate_diagnostics(
    document: dict[str, object], *, kind: str, horizon: int
) -> None:
    epochs = document.get("epochs")
    if (
        document.get("schema_version") != 1
        or not isinstance(epochs, list)
        or len(epochs) != horizon
        or [entry.get("epoch") for entry in epochs] != list(range(horizon))
    ):
        raise ValueError("RQ5 gate diagnostic horizon changed")
    gradients = [entry["gate_parameter_gradient_norm"] for entry in epochs]
    if kind == "frequency":
        if document.get("frequency_input_parity") is not True or any(
            value["mean"] != 0 or value["nonfinite_count"] != 0
            for value in gradients
        ):
            raise ValueError("RQ5 frequency defect evidence changed")
    elif document.get("frequency_input_parity") is not None or not any(
        value["mean"] > 0 for value in gradients
    ):
        raise ValueError("RQ5 global mechanism control evidence changed")


def _validate_probe(root: Path, probe: dict[str, object]) -> None:
    rows = probe.get("rows")
    source = probe.get("source")
    if (
        probe.get("sha256") != RQ5_PRECISION_PROBE_LOGICAL_SHA256
        or probe.get("autocast_dtype") != "torch.bfloat16"
        or probe.get("device", {}).get("capability") != [8, 0]
        or not isinstance(rows, list)
        or len(rows) != 2
        or rows[0].get("all_output_exact_one") is not True
        or any(value != 0 for value in rows[0]["gradient_norms"].values())
        or rows[1].get("all_output_exact_one") is not False
        or not any(value > 0 for value in rows[1]["gradient_norms"].values())
        or not isinstance(source, dict)
        or _physical(root, root / str(source["path"])).sha256 != source.get("sha256")
    ):
        raise ValueError("RQ5 A100 frequency gate precision proof changed")


def _validate_program(ledger: Rq5FrequencyV2Ledger) -> None:
    expected_widths = (4, 4, 4, 8, 8, 8, 16, 16, 16)
    if (
        len(ledger.rows) != 9
        or tuple(row.gate_hidden_dim for row in ledger.rows) != expected_widths
        or any(row.batch_size != 512 or row.seed != 42 or row.horizon_epochs != 25 for row in ledger.rows)
    ):
        raise ValueError("RQ5 frequency v2 lost its exact nine-cell surface")


def _reference(root: Path, path: Path, logical: str) -> Rq5FileReference:
    physical = _physical(root, path)
    return Rq5FileReference(
        path=physical.path,
        size_bytes=physical.size_bytes,
        sha256=physical.sha256,
        logical_sha256=logical,
    )


def _physical(root: Path, path: Path) -> Rq5PhysicalFileReference:
    if path.is_symlink():
        raise ValueError("RQ5 frequency v2 input must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("RQ5 frequency v2 input escapes project root")
    return Rq5PhysicalFileReference(
        path=str(resolved.relative_to(root)),
        size_bytes=resolved.stat().st_size,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
    )


def _load_logical(root: Path, path: Path) -> dict[str, object]:
    document = _load_json(path)
    payload = {key: value for key, value in document.items() if key != "sha256"}
    logical = document.get("sha256")
    if not isinstance(logical, str) or _canonical_sha256(payload) != logical:
        raise ValueError("RQ5 frequency v2 logical input changed")
    return document


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("RQ5 frequency v2 JSON must be an object")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
