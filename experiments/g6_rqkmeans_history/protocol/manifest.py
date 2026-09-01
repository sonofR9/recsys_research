from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, get_args


Representation = Literal[
    "learned_sid_event",
    "item_frozen_sid_event",
    "item_learned_frozen_sid_event",
    "learned_sid_tokens",
    "learned_frozen_sid_tokens",
    "frozen_sid_tokens",
    "interleaved_item_sid_tokens",
]
Backbone = Literal["original_g1", "best_g1"]
Stage = Literal[
    "primary_control_tuning",
    "original_control_tuning",
    "primary_control_repeats",
    "treatment_tuning",
    "bridge_tuning",
    "lr_boundary",
]

REPRESENTATIONS: tuple[Representation, ...] = get_args(Representation)
CONTROL_BATCHES = (128, 256, 512, 1024, 1280)
NUM_LEVELS = (2, 3, 4)
NUM_CODES = (32, 64, 128, 256, 512)
REPRESENTATION_WIDTHS = (32, 64, 128)
RANKING_EVIDENCE_GROUP = "g6-rq0-native50m"
EMBEDDING_LR_BOUNDS = (1e-4, 0.256)
DEEP_LR_BOUNDS = (1e-4, 0.128)
INITIAL_RUNS = 165
MAX_RUNS = 245
MANIFEST_VERSION = 1
_BOUNDARY_FACTORS = (2**0.5, 2.0, 2**1.5, 4.0)

EXPECTED_STAGE_COUNTS = {
    "primary_control_tuning": 20,
    "original_control_tuning": 12,
    "primary_control_repeats": 9,
    "treatment_tuning": 112,
    "bridge_tuning": 12,
    "lr_boundary": 80,
}


@dataclass(frozen=True)
class ApprovedJob:
    id: str
    run_name: str
    stage: Stage
    method: str
    seed: int = 42
    trial: int | None = None
    conditional: bool = False
    forced_parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_name": self.run_name,
            "stage": self.stage,
            "method": self.method,
            "seed": self.seed,
            "trial": self.trial,
            "conditional": self.conditional,
            "forced_parameters": self.forced_parameters,
        }


@dataclass(frozen=True)
class ApprovedManifest:
    jobs: tuple[ApprovedJob, ...]

    @property
    def stage_counts(self) -> dict[str, int]:
        counts = Counter(job.stage for job in self.jobs)
        return {stage: counts[stage] for stage in EXPECTED_STAGE_COUNTS}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    def jobs_for_stage(self, stage: Stage) -> list[ApprovedJob]:
        return [job for job in self.jobs if job.stage == stage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "dataset_size": "native-50m",
            "initial_runs": INITIAL_RUNS,
            "maximum_runs": MAX_RUNS,
            "jobs": [job.to_dict() for job in self.jobs],
            "search_spaces": {
                "batch_size": list(CONTROL_BATCHES),
                "validation_batch_size": "validated preflight evidence",
                "embedding_learning_rate": [*EMBEDDING_LR_BOUNDS, "log_uniform"],
                "deep_learning_rate": [*DEEP_LR_BOUNDS, "log_uniform"],
                "num_levels": list(NUM_LEVELS),
                "shared_num_codes": list(NUM_CODES),
                "representation_width": list(REPRESENTATION_WIDTHS),
                "lr_boundary": {
                    "trigger_outer_log_fraction": 0.1,
                    "points_per_learning_rate": 4,
                    "outward_factors": [2**0.5, 2.0, 2**1.5, 4.0],
                    "maximum_extensions_per_learning_rate": 1,
                },
            },
        }


@dataclass(frozen=True)
class CompiledJob:
    approved: ApprovedJob
    parameters: dict[str, Any]
    attempt: int = 0
    cap_epochs: int | None = None

    @property
    def run_name(self) -> str:
        if self.attempt == 0:
            return self.approved.run_name
        return f"{self.approved.run_name}_cap{self.cap_epochs}_a{self.attempt}"

    @property
    def identity(self) -> tuple[str, int]:
        return self.approved.id, self.attempt

    def to_contract(self, manifest: ApprovedManifest) -> dict[str, Any]:
        contract = {
            "manifest_sha256": manifest.sha256,
            "job": self.approved.to_dict(),
            "parameters": self.parameters,
        }
        if self.attempt:
            contract["continuation"] = {
                "attempt": self.attempt,
                "cap_epochs": self.cap_epochs,
                "source_run_name": self.approved.run_name,
            }
        return contract


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def boundary_side(
    value: float, bounds: tuple[float, float]
) -> Literal["lower", "upper"] | None:
    lower, upper = bounds
    if not lower <= value <= upper:
        raise ValueError("selected learning rate is outside its approved search")
    position = math.log(value / lower) / math.log(upper / lower)
    if position <= 0.1:
        return "lower"
    if position >= 0.9:
        return "upper"
    return None


def outside_boundary_rates(
    bounds: tuple[float, float], side: Literal["lower", "upper"]
) -> tuple[float, ...]:
    if side == "lower":
        return tuple(bounds[0] / factor for factor in _BOUNDARY_FACTORS)
    return tuple(bounds[1] * factor for factor in _BOUNDARY_FACTORS)


def _job(
    stage: Stage,
    suffix: str,
    method: str,
    *,
    seed: int = 42,
    trial: int | None = None,
    conditional: bool = False,
    forced_parameters: dict[str, Any] | None = None,
) -> ApprovedJob:
    return ApprovedJob(
        id=f"{stage}:{suffix}",
        run_name=f"g6_rq0_{suffix}_native50m",
        stage=stage,
        method=method,
        seed=seed,
        trial=trial,
        conditional=conditional,
        forced_parameters=forced_parameters or {},
    )


def _build_jobs() -> tuple[ApprovedJob, ...]:
    jobs: list[ApprovedJob] = []
    jobs.extend(
        _job(
            "primary_control_tuning",
            f"best_g1_control_trial_{trial:02d}",
            "best_g1_item_ids",
            trial=trial,
        )
        for trial in range(20)
    )
    jobs.extend(
        _job(
            "original_control_tuning",
            f"original_g1_control_trial_{trial:02d}",
            "original_g1_item_ids",
            trial=trial,
        )
        for trial in range(12)
    )
    jobs.extend(
        _job(
            "primary_control_repeats",
            f"best_g1_control_seed_{seed}",
            "best_g1_item_ids",
            seed=seed,
        )
        for seed in range(43, 52)
    )
    anchors = (
        {"num_levels": 2, "num_codes": 256},
        {"num_levels": 3, "num_codes": 64},
        {"num_levels": 3, "num_codes": 128},
        {"num_levels": 4, "num_codes": 32},
    )
    for representation in REPRESENTATIONS:
        jobs.extend(
            _job(
                "treatment_tuning",
                f"{representation}_trial_{trial:02d}",
                representation,
                trial=trial,
                forced_parameters=anchors[trial] if trial < len(anchors) else {},
            )
            for trial in range(16)
        )
    jobs.extend(
        _job(
            "bridge_tuning",
            f"selected_semantic_original_g1_trial_{trial:02d}",
            "selected_semantic_bridge",
            trial=trial,
        )
        for trial in range(12)
    )
    surfaces = (
        "primary_control",
        "original_control",
        *REPRESENTATIONS,
        "bridge",
    )
    for surface in surfaces:
        for learning_rate in (
            "embedding_learning_rate",
            "deep_learning_rate",
        ):
            jobs.extend(
                _job(
                    "lr_boundary",
                    f"boundary_{surface}_{learning_rate}_{slot}",
                    surface,
                    trial=slot,
                    conditional=True,
                    forced_parameters={
                        "learning_rate": learning_rate,
                        "boundary_slot": slot,
                    },
                )
                for slot in range(4)
            )
    return tuple(jobs)


_APPROVED_MANIFEST = ApprovedManifest(_build_jobs())
APPROVED_MANIFEST_PATH = Path(__file__).with_name("approved_manifest.json")


def approved_manifest() -> ApprovedManifest:
    validate_approved_manifest(_APPROVED_MANIFEST.to_dict())
    return _APPROVED_MANIFEST


def validate_approved_manifest(document: dict[str, Any]) -> None:
    if _canonical_json(document) != _canonical_json(_APPROVED_MANIFEST.to_dict()):
        raise ValueError("document does not match the approved manifest")
    jobs = document["jobs"]
    if len(jobs) != MAX_RUNS:
        raise ValueError("approved manifest does not account for 245 runs")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("approved manifest contains duplicate job IDs")
    if len({job["run_name"] for job in jobs}) != len(jobs):
        raise ValueError("approved manifest contains duplicate run names")
    if _APPROVED_MANIFEST.stage_counts != EXPECTED_STAGE_COUNTS:
        raise ValueError("approved manifest stage counts changed")
    initial = sum(not job["conditional"] for job in jobs)
    if initial != INITIAL_RUNS:
        raise ValueError("approved manifest does not account for 165 initial runs")


def write_approved_manifest(path: Path = APPROVED_MANIFEST_PATH) -> None:
    manifest = approved_manifest()
    document = manifest.to_dict() | {"sha256": manifest.sha256}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


_BASE_PARAMETERS = {
    "batch_size",
    "validation_batch_size",
    "embedding_learning_rate",
    "deep_learning_rate",
}
_SEMANTIC_PARAMETERS = {
    "representation",
    "num_levels",
    "num_codes",
    "representation_width",
}


def _require_exact_parameters(
    job: ApprovedJob,
    parameters: dict[str, Any],
    required: set[str],
) -> None:
    missing = required - parameters.keys()
    unexpected = parameters.keys() - required
    if missing:
        raise ValueError(f"{job.id} missing parameters: {', '.join(sorted(missing))}")
    if unexpected:
        raise ValueError(
            f"{job.id} has unexpected parameters: {', '.join(sorted(unexpected))}"
        )


def _job_by_id(job_id: Any) -> ApprovedJob:
    matches = [job for job in _APPROVED_MANIFEST.jobs if job.id == job_id]
    if len(matches) != 1:
        raise ValueError(f"prerequisite job {job_id!r} is not approved")
    return matches[0]


def _is_boundary(job: ApprovedJob, surface: str) -> bool:
    return job.stage == "lr_boundary" and job.method == surface


def _require_primary(job_id: Any) -> ApprovedJob:
    job = _job_by_id(job_id)
    if job.stage != "primary_control_tuning" and not _is_boundary(
        job, "primary_control"
    ):
        raise ValueError("global batch must come from the selected primary control")
    return job


def _require_original(job_id: Any) -> ApprovedJob:
    job = _job_by_id(job_id)
    if job.stage != "original_control_tuning" and not _is_boundary(
        job, "original_control"
    ):
        raise ValueError("original baseline must come from its approved study")
    return job


def _require_treatment(job_id: Any, representation: Any) -> ApprovedJob:
    job = _job_by_id(job_id)
    if representation not in REPRESENTATIONS:
        raise ValueError("representation is not approved")
    if job.stage == "treatment_tuning" and job.method == representation:
        return job
    if _is_boundary(job, str(representation)):
        return job
    raise ValueError("semantic representation must come from its approved study")


def _positive_finite(parameters: dict[str, Any], name: str) -> float:
    value = parameters.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be resolved")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")
    return value


def _validate_domains(job: ApprovedJob, parameters: dict[str, Any]) -> None:
    if parameters["batch_size"] not in CONTROL_BATCHES:
        raise ValueError("batch_size is outside the approved domain")
    validation_batch_size = parameters["validation_batch_size"]
    if (
        not isinstance(validation_batch_size, int)
        or isinstance(validation_batch_size, bool)
        or validation_batch_size < 1
    ):
        raise ValueError("validation_batch_size must be a positive integer")
    embedding_rate = _positive_finite(parameters, "embedding_learning_rate")
    deep_rate = _positive_finite(parameters, "deep_learning_rate")
    if job.stage != "lr_boundary" and job.stage != "primary_control_repeats":
        if not EMBEDDING_LR_BOUNDS[0] <= embedding_rate <= EMBEDDING_LR_BOUNDS[1]:
            raise ValueError("embedding learning rate is outside the approved search")
        if not DEEP_LR_BOUNDS[0] <= deep_rate <= DEEP_LR_BOUNDS[1]:
            raise ValueError("deep learning rate is outside the approved search")
    if "representation" not in parameters:
        return
    if parameters["representation"] not in REPRESENTATIONS:
        raise ValueError("representation is outside the approved domain")
    if parameters["num_levels"] not in NUM_LEVELS:
        raise ValueError("num_levels is outside the approved domain")
    if parameters["num_codes"] not in NUM_CODES:
        raise ValueError("num_codes is outside the approved domain")
    if parameters["representation_width"] not in REPRESENTATION_WIDTHS:
        raise ValueError("representation_width is outside the approved domain")


def _boundary_required(parameters: dict[str, Any]) -> set[str]:
    source = {"source_parameters"}
    builder = parameters.get("builder")
    if builder == "primary_control":
        return (
            _BASE_PARAMETERS
            | source
            | {
                "builder",
                "source_job_id",
                "boundary_side",
            }
        )
    if builder == "original_control":
        return (
            _BASE_PARAMETERS
            | source
            | {
                "builder",
                "source_job_id",
                "selected_primary_control_job_id",
                "boundary_side",
            }
        )
    if builder == "treatment":
        return (
            _BASE_PARAMETERS
            | _SEMANTIC_PARAMETERS
            | source
            | {
                "builder",
                "source_job_id",
                "selected_primary_control_job_id",
                "boundary_side",
            }
        )
    if builder == "bridge":
        return (
            _BASE_PARAMETERS
            | _SEMANTIC_PARAMETERS
            | source
            | {
                "builder",
                "source_job_id",
                "selected_primary_control_job_id",
                "selected_original_control_job_id",
                "selected_treatment_job_id",
                "boundary_side",
            }
        )
    raise ValueError(f"unknown boundary builder {builder!r}")


def _validate_parameters(job: ApprovedJob, parameters: dict[str, Any]) -> None:
    for name, expected in job.forced_parameters.items():
        if name in {"learning_rate", "boundary_slot"}:
            continue
        if parameters.get(name) != expected:
            raise ValueError(f"{job.id} must preserve forced {name}={expected!r}")
    if job.stage == "primary_control_tuning":
        required = set(_BASE_PARAMETERS)
    elif job.stage == "original_control_tuning":
        required = _BASE_PARAMETERS | {"selected_primary_control_job_id"}
        _require_primary(parameters.get("selected_primary_control_job_id"))
    elif job.stage == "primary_control_repeats":
        required = _BASE_PARAMETERS | {"selected_primary_control_job_id"}
        _require_primary(parameters.get("selected_primary_control_job_id"))
    elif job.stage == "treatment_tuning":
        required = (
            _BASE_PARAMETERS
            | _SEMANTIC_PARAMETERS
            | {"selected_primary_control_job_id"}
        )
        _require_primary(parameters.get("selected_primary_control_job_id"))
        if parameters.get("representation") != job.method:
            raise ValueError(f"{job.id} representation changed")
    elif job.stage == "bridge_tuning":
        required = (
            _BASE_PARAMETERS
            | _SEMANTIC_PARAMETERS
            | {
                "selected_primary_control_job_id",
                "selected_original_control_job_id",
                "selected_treatment_job_id",
            }
        )
        _require_primary(parameters.get("selected_primary_control_job_id"))
        _require_original(parameters.get("selected_original_control_job_id"))
        _require_treatment(
            parameters.get("selected_treatment_job_id"),
            parameters.get("representation"),
        )
    elif job.stage == "lr_boundary":
        required = _boundary_required(parameters)
        source = _job_by_id(parameters.get("source_job_id"))
        builder = parameters["builder"]
        expected = {
            "primary_control": ("primary_control_tuning", "best_g1_item_ids"),
            "original_control": ("original_control_tuning", "original_g1_item_ids"),
            "treatment": ("treatment_tuning", parameters.get("representation")),
            "bridge": ("bridge_tuning", "selected_semantic_bridge"),
        }[builder]
        if (source.stage, source.method) != expected:
            raise ValueError(f"{job.id} boundary source changed")
        expected_surface = (
            str(parameters["representation"]) if builder == "treatment" else builder
        )
        if job.method != expected_surface:
            raise ValueError(f"{job.id} boundary surface changed")
        if parameters.get("boundary_side") not in {"lower", "upper"}:
            raise ValueError("boundary_side must be lower or upper")
        source_parameters = parameters.get("source_parameters")
        if not isinstance(source_parameters, dict):
            raise ValueError("boundary source parameters are absent")
        validate_compiled_job(CompiledJob(source, source_parameters))
        learning_rate = str(job.forced_parameters["learning_rate"])
        bounds = (
            EMBEDDING_LR_BOUNDS
            if learning_rate == "embedding_learning_rate"
            else DEEP_LR_BOUNDS
        )
        expected_side = boundary_side(float(source_parameters[learning_rate]), bounds)
        if parameters["boundary_side"] != expected_side:
            raise ValueError("boundary side changed from the selected source")
        slot = job.forced_parameters["boundary_slot"]
        expected_rate = outside_boundary_rates(bounds, parameters["boundary_side"])[
            slot
        ]
        if parameters[learning_rate] != expected_rate:
            raise ValueError("boundary learning rate changed from its approved slot")
        boundary_metadata = {
            "builder",
            "source_job_id",
            "source_parameters",
            "boundary_side",
        }
        training_parameters = {
            name: value
            for name, value in parameters.items()
            if name not in boundary_metadata
        }
        expected_parameters = dict(source_parameters)
        expected_parameters[learning_rate] = expected_rate
        if training_parameters != expected_parameters:
            raise ValueError("boundary source configuration changed")
        if builder != "primary_control":
            _require_primary(parameters.get("selected_primary_control_job_id"))
        if builder == "bridge":
            _require_original(parameters.get("selected_original_control_job_id"))
            _require_treatment(
                parameters.get("selected_treatment_job_id"),
                parameters.get("representation"),
            )
    else:
        raise ValueError(f"unknown stage {job.stage!r}")
    _require_exact_parameters(job, parameters, required)
    _validate_domains(job, parameters)


def validate_compiled_job(compiled: CompiledJob) -> None:
    matches = [job for job in approved_manifest().jobs if job == compiled.approved]
    if len(matches) != 1:
        raise ValueError("compiled job identity is not approved")
    _validate_parameters(compiled.approved, compiled.parameters)
    _validate_continuation(compiled)


def validate_boundary_source(
    compiled: CompiledJob,
    compiled_ledger: list[CompiledJob] | tuple[CompiledJob, ...],
) -> None:
    validate_compiled_job(compiled)
    if compiled.approved.stage != "lr_boundary":
        return
    source_id = compiled.parameters["source_job_id"]
    sources = [
        candidate for candidate in compiled_ledger if candidate.approved.id == source_id
    ]
    if not sources:
        raise ValueError(f"{compiled.approved.id}: compiled boundary source is absent")
    source_parameters = compiled.parameters["source_parameters"]
    if any(source.parameters != source_parameters for source in sources):
        raise ValueError(
            f"{compiled.approved.id}: compiled boundary source contract changed"
        )


def _uses_original_backbone(compiled: CompiledJob) -> bool:
    stage = compiled.approved.stage
    if stage in {"original_control_tuning", "bridge_tuning"}:
        return True
    return stage == "lr_boundary" and compiled.parameters.get("builder") in {
        "original_control",
        "bridge",
    }


def _cap_for_attempt(attempt: int) -> int:
    cap = 40
    for _ in range(attempt):
        cap = math.ceil(1.5 * cap)
    return cap


def _validate_continuation(compiled: CompiledJob) -> None:
    if (
        not isinstance(compiled.attempt, int)
        or isinstance(compiled.attempt, bool)
        or compiled.attempt < 0
    ):
        raise ValueError("continuation attempt must be a non-negative integer")
    if compiled.attempt == 0:
        if compiled.cap_epochs is not None:
            raise ValueError("initial run cannot override the epoch cap")
        return
    if not _uses_original_backbone(compiled):
        raise ValueError("cap continuations require the original G1 backbone")
    expected_cap = _cap_for_attempt(compiled.attempt)
    if compiled.cap_epochs != expected_cap:
        raise ValueError(
            f"continuation attempt {compiled.attempt} requires cap {expected_cap}"
        )


def compile_cap_continuation(compiled: CompiledJob) -> CompiledJob:
    validate_compiled_job(compiled)
    continuation = CompiledJob(
        approved=compiled.approved,
        parameters=dict(compiled.parameters),
        attempt=compiled.attempt + 1,
        cap_epochs=_cap_for_attempt(compiled.attempt + 1),
    )
    validate_compiled_job(continuation)
    return continuation


def load_compiled_jobs(path: Path) -> list[CompiledJob]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read compiled manifest {path}") from error
    manifest = approved_manifest()
    if document.get("manifest_sha256") != manifest.sha256:
        raise ValueError("compiled manifest references a different approved manifest")
    rows = document.get("jobs")
    if not isinstance(rows, list):
        raise ValueError("compiled manifest jobs must be a list")
    approved_by_id = {job.id: job for job in manifest.jobs}
    seen: set[tuple[str, int]] = set()
    compiled_jobs: list[CompiledJob] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("compiled job must be an object")
        job_id = row.get("id")
        if job_id not in approved_by_id:
            raise ValueError(f"unknown compiled job {job_id!r}")
        approved = approved_by_id[str(job_id)]
        attempt = row.get("attempt", 0)
        cap_epochs = row.get("cap_epochs")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            raise ValueError("compiled continuation attempt is invalid")
        identity = (str(job_id), attempt)
        if identity in seen:
            raise ValueError(f"duplicate compiled job attempt {identity!r}")
        seen.add(identity)
        parameters = row.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"parameters are unresolved for {job_id}")
        compiled = CompiledJob(approved, parameters, attempt, cap_epochs)
        validate_compiled_job(compiled)
        if row.get("run_name") != compiled.run_name:
            raise ValueError(f"run name changed for {job_id}")
        compiled_jobs.append(compiled)
    return compiled_jobs
