from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping

from .design import (
    BATCH_SIZE,
    DATASET_SIZE,
    FIXED_HORIZON,
    KMEANS_MAX_ITERATIONS,
    KMEANS_TOLERANCE,
    REPRESENTATION_WIDTH,
    REPRESENTATIONS,
    SHARED_CODEBOOK_SIZES,
    TOKENIZER_LEVELS,
    EXPECTED_RUN_TOTALS,
    MAXIMUM_RUNS,
)


PLAN_SHA256 = "3561064c58087cff75b0029c62eb477104b8bce51ff77b4868f3733d0b218910"
APPROVAL_SHA256 = "427d86df9523cee795945e1d83d01681672da6120a1667d0af2a2ab94b3c2382"
JOB_SCHEMA = "g6-native500m-job/v2"
LEGACY_JOB_SCHEMA = "g6-native500m-job/v1"
MANIFEST_SCHEMA = "g6-native500m-stage-manifest/v1"
APPROVAL_SCHEMA = "g6-native500m-approval/v1"
RUNNER_PATH = "experiments/g6_rqkmeans_history/native500m/launchers/run_native500m.py"
APPROVED_STAGES = (
    "controls",
    "controls_boundary",
    "rq0_surface",
    "rq0_boundary",
    "rq0_bridge",
    "rq0_bridge_boundary",
    "rq1_surface",
    "rq1_boundary",
    "rq1_confirmation",
    "rq2_rq3_surface",
    "rq2_rq3_refinement",
    "rq2_rq3_boundary",
    "rq2_rq3_confirmation",
    "terminal_bridge",
    "terminal_bridge_boundary",
    "aggregate",
)
APPROVED_PREDECESSORS = {
    "controls": frozenset(),
    "controls_boundary": frozenset({"controls"}),
    "rq0_surface": frozenset({"controls", "controls_boundary", "rq0_surface"}),
    "rq0_boundary": frozenset({"rq0_surface"}),
    "rq0_bridge": frozenset({"rq0_surface", "rq0_boundary"}),
    "rq0_bridge_boundary": frozenset({"rq0_bridge"}),
    "rq1_surface": frozenset({"rq0_surface", "rq0_boundary"}),
    "rq1_boundary": frozenset({"rq1_surface"}),
    "rq1_confirmation": frozenset({"rq1_surface", "rq1_boundary"}),
    "rq2_rq3_surface": frozenset({"rq1_confirmation"}),
    "rq2_rq3_refinement": frozenset({"rq2_rq3_surface"}),
    "rq2_rq3_boundary": frozenset({"rq2_rq3_refinement"}),
    "rq2_rq3_confirmation": frozenset({"rq2_rq3_refinement", "rq2_rq3_boundary"}),
    "terminal_bridge": frozenset({"rq2_rq3_confirmation"}),
    "terminal_bridge_boundary": frozenset({"terminal_bridge"}),
    "aggregate": frozenset(
        {
            "rq2_rq3_confirmation",
            "terminal_bridge",
            "terminal_bridge_boundary",
        }
    ),
}
_COMMON_PARAMETERS = {
    "builder",
    "runner",
    "run_name",
    "config_logical_sha256",
    "data_group",
    "environment",
    "backbone",
    "embedding_learning_rate",
    "deep_learning_rate",
    "seed",
}
_SEMANTIC_PARAMETERS = {
    "representation",
    "levels",
    "shared_codes",
    "representation_width",
    "collision_policy",
    "sid_initialization",
}


def job_id_has_coordinate(job_id: object, family: str, index: int) -> bool:
    if not isinstance(job_id, str):
        return False
    fields = job_id.split(":")
    return len(fields) >= 3 and fields[1] == family and fields[2] == f"{index:02d}"


_CONTROL_STAGES = {"controls", "controls_boundary"}
_CONFIRMATION_STAGES = {
    stage for stage in APPROVED_STAGES if stage.endswith("confirmation")
}
_ORIGINAL_BACKBONE_STAGES = {
    "rq0_bridge",
    "rq0_bridge_boundary",
    "terminal_bridge",
    "terminal_bridge_boundary",
}
_RQ0_STAGES = {
    "rq0_surface",
    "rq0_boundary",
    "rq0_bridge",
    "rq0_bridge_boundary",
}
_RQ1_STAGES = {"rq1_surface", "rq1_boundary", "rq1_confirmation"}
_TRAINABLE_SID_REPRESENTATIONS = {
    "learned_sid_event",
    "item_learned_frozen_sid_event",
    "learned_sid_tokens",
    "learned_frozen_sid_tokens",
    "interleaved_item_sid_tokens",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_NAME = re.compile(r"[A-Za-z0-9_.-]+")
_DATA_GROUP = re.compile(r"[A-Za-z0-9_.-]+")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOKENIZER_ENVIRONMENT_FIELDS = {
    "G6_NATIVE500M_TOKENIZER_BINDING_REVISION",
    "G6_NATIVE500M_TOKENIZER_REGISTRY_SHA256",
    "G6_NATIVE500M_TOKENIZER_BASE_CACHE_KEY",
    "G6_NATIVE500M_TOKENIZER_FIT_SHA256",
    "G6_NATIVE500M_TOKENIZER_CODES_SHA256",
    "G6_NATIVE500M_TOKENIZER_MATERIALIZATION_SHA256",
}
_G6_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_PATH = _G6_ROOT / "protocol/native500m_rerun_plan.md"
DEFAULT_APPROVAL_PATH = _G6_ROOT / "protocol/native500m_approval.json"

Schedule = Literal["annealed", "constant"]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class DocumentIdentity:
    logical_sha256: str
    physical_sha256: str


def document_identity(content: bytes) -> DocumentIdentity:
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("document is not valid JSON") from error
    return DocumentIdentity(
        logical_sha256=canonical_sha256(document),
        physical_sha256=hashlib.sha256(content).hexdigest(),
    )


@dataclass(frozen=True)
class ApprovalBinding:
    plan_sha256: str
    approval_sha256: str
    expected_run_totals: tuple[int, ...]
    maximum_runs: int


def load_approval_binding(
    *,
    plan_path: Path = DEFAULT_PLAN_PATH,
    approval_path: Path = DEFAULT_APPROVAL_PATH,
) -> ApprovalBinding:
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if plan_sha256 != PLAN_SHA256:
        raise ValueError("approved plan SHA-256 changed")
    approval_content = approval_path.read_bytes()
    approval_sha256 = hashlib.sha256(approval_content).hexdigest()
    if approval_sha256 != APPROVAL_SHA256:
        raise ValueError("approval artifact SHA-256 changed")
    try:
        approval = json.loads(approval_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("approval artifact is not valid JSON") from error
    expected = {
        "schema": APPROVAL_SCHEMA,
        "plan_sha256": PLAN_SHA256,
        "dataset_size": DATASET_SIZE,
        "expected_run_totals": list(EXPECTED_RUN_TOTALS),
        "maximum_runs": MAXIMUM_RUNS,
        "fixed_training_horizon": FIXED_HORIZON,
        "representation_width": REPRESENTATION_WIDTH,
        "tokenizer_levels": list(TOKENIZER_LEVELS),
        "shared_codebook_sizes": list(SHARED_CODEBOOK_SIZES),
        "fixed_kmeans_max_iterations": KMEANS_MAX_ITERATIONS,
        "fixed_kmeans_tolerance": KMEANS_TOLERANCE,
        "approved_at": "2026-08-31",
    }
    if approval != expected:
        raise ValueError("approval artifact does not match the approved protocol")
    return ApprovalBinding(
        plan_sha256=plan_sha256,
        approval_sha256=approval_sha256,
        expected_run_totals=EXPECTED_RUN_TOTALS,
        maximum_runs=MAXIMUM_RUNS,
    )


@dataclass(frozen=True)
class SelectionBinding:
    stage: str
    selection_sha256: str
    resolved: bool

    def __post_init__(self) -> None:
        if self.stage not in APPROVED_STAGES:
            raise ValueError("selection has an unknown stage")
        _require_sha256(self.selection_sha256, "selection")
        if not isinstance(self.resolved, bool):
            raise ValueError("selection resolved flag must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "selection_sha256": self.selection_sha256,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SelectionBinding:
        _require_keys(value, {"stage", "selection_sha256", "resolved"}, "selection")
        if not isinstance(value["resolved"], bool):
            raise ValueError("selection resolved flag must be boolean")
        return cls(
            _string(value["stage"], "selection stage"),
            _string(value["selection_sha256"], "selection SHA-256"),
            value["resolved"],
        )


@dataclass(frozen=True)
class ExactReuse:
    source_job_id: str
    source_contract_sha256: str
    fields: tuple[str, ...]
    source_selection_stage: str | None = None
    source_selection_sha256: str | None = None
    source_selection_physical_sha256: str | None = None
    source_selection_path: str | None = None

    def __post_init__(self) -> None:
        if not self.source_job_id:
            raise ValueError("exact reuse source job must be nonempty")
        _require_sha256(self.source_contract_sha256, "source contract")
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("exact reuse fields must be nonempty and unique")
        bindings = (
            self.source_selection_stage,
            self.source_selection_sha256,
            self.source_selection_physical_sha256,
            self.source_selection_path,
        )
        if any(value is not None for value in bindings):
            if not all(isinstance(value, str) and value for value in bindings):
                raise ValueError("exact reuse selection binding is incomplete")
            _require_sha256(self.source_selection_sha256, "reuse selection")
            _require_sha256(
                self.source_selection_physical_sha256, "reuse selection physical"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_job_id": self.source_job_id,
            "source_contract_sha256": self.source_contract_sha256,
            "fields": list(self.fields),
            "source_selection_stage": self.source_selection_stage,
            "source_selection_sha256": self.source_selection_sha256,
            "source_selection_physical_sha256": self.source_selection_physical_sha256,
            "source_selection_path": self.source_selection_path,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExactReuse:
        _require_keys(
            value,
            {
                "source_job_id",
                "source_contract_sha256",
                "fields",
                "source_selection_stage",
                "source_selection_sha256",
                "source_selection_physical_sha256",
                "source_selection_path",
            },
            "exact reuse",
        )
        fields = value["fields"]
        if not isinstance(fields, list) or not all(
            isinstance(field, str) for field in fields
        ):
            raise ValueError("exact reuse fields must be strings")
        return cls(
            str(value["source_job_id"]),
            str(value["source_contract_sha256"]),
            tuple(fields),
            value["source_selection_stage"],
            value["source_selection_sha256"],
            value["source_selection_physical_sha256"],
            value["source_selection_path"],
        )


@dataclass(frozen=True)
class JobContract:
    job_id: str
    stage: str
    parameters_json: str
    source_selection: SelectionBinding | None
    exact_reuse: tuple[ExactReuse, ...]
    schedule: Schedule
    dataset_size: str = DATASET_SIZE
    batch_size: int = BATCH_SIZE
    training_horizon: int = FIXED_HORIZON
    plan_sha256: str = PLAN_SHA256
    approval_sha256: str = APPROVAL_SHA256
    schema: str = JOB_SCHEMA

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        stage: str,
        parameters: Mapping[str, object],
        source_selection: SelectionBinding | None = None,
        exact_reuse: tuple[ExactReuse, ...] = (),
        schedule: Schedule = "annealed",
        dataset_size: str = DATASET_SIZE,
        batch_size: int = BATCH_SIZE,
        training_horizon: int = FIXED_HORIZON,
        plan_sha256: str = PLAN_SHA256,
        approval_sha256: str = APPROVAL_SHA256,
    ) -> JobContract:
        contract = cls(
            job_id=job_id,
            stage=stage,
            parameters_json=canonical_json(dict(parameters)),
            source_selection=source_selection,
            exact_reuse=exact_reuse,
            schedule=schedule,
            dataset_size=dataset_size,
            batch_size=batch_size,
            training_horizon=training_horizon,
            plan_sha256=plan_sha256,
            approval_sha256=approval_sha256,
        )
        contract.validate()
        return contract

    @property
    def parameters(self) -> dict[str, object]:
        value = json.loads(self.parameters_json)
        if not isinstance(value, dict):
            raise ValueError("job parameters must be a JSON object")
        return value

    @property
    def logical_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def run_name(self) -> str:
        return self._string_parameter("run_name")

    @property
    def backbone(self) -> str:
        return self._string_parameter("backbone")

    @property
    def representation(self) -> str | None:
        return self._optional_string_parameter("representation")

    @property
    def embedding_learning_rate(self) -> float:
        return self._positive_float_parameter("embedding_learning_rate")

    @property
    def deep_learning_rate(self) -> float:
        return self._positive_float_parameter("deep_learning_rate")

    @property
    def levels(self) -> int | None:
        return self._optional_int_parameter("levels")

    @property
    def shared_codes(self) -> int | None:
        return self._optional_int_parameter("shared_codes")

    @property
    def collision_policy(self) -> str | None:
        return self._optional_string_parameter("collision_policy")

    @property
    def sid_initialization(self) -> str | None:
        return self._optional_string_parameter("sid_initialization")

    @property
    def seed(self) -> int:
        value = self.parameters.get("seed")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("job seed is not an integer")
        return value

    def validate(self) -> None:
        if (
            not isinstance(self.job_id, str)
            or not self.job_id
            or self.stage not in APPROVED_STAGES
        ):
            raise ValueError("job ID or stage is invalid")
        if self.stage == "aggregate":
            raise ValueError("aggregate is not a runnable training stage")
        if self.dataset_size != DATASET_SIZE:
            raise ValueError("job dataset is not native-500m")
        if self.batch_size != BATCH_SIZE:
            raise ValueError("job batch size is not the approved 512")
        if self.training_horizon != FIXED_HORIZON:
            raise ValueError("job training horizon is not the approved 26")
        if self.schedule not in ("annealed", "constant"):
            raise ValueError("job schedule is not approved")
        if self.plan_sha256 != PLAN_SHA256 or self.approval_sha256 != APPROVAL_SHA256:
            raise ValueError("job approval binding changed")
        if self.schema not in {JOB_SCHEMA, LEGACY_JOB_SCHEMA}:
            raise ValueError("job schema changed")
        if canonical_json(self.parameters) != self.parameters_json:
            raise ValueError("job parameters are not canonical")
        parameters = self.parameters
        self._validate_parameter_schema(parameters)
        backbone = parameters["backbone"]
        if backbone == "original_g1" and self.schedule != "constant":
            raise ValueError("original-G1 jobs must preserve the constant-LR schedule")
        if (
            backbone is not None
            and backbone != "original_g1"
            and self.schedule != "annealed"
        ):
            raise ValueError("best-G1 and SID jobs must use the annealed schedule")
        if len({reuse.source_job_id for reuse in self.exact_reuse}) != len(
            self.exact_reuse
        ):
            raise ValueError("job has duplicate exact-reuse sources")
        allowed_predecessors = APPROVED_PREDECESSORS[self.stage]
        if not allowed_predecessors and self.source_selection is not None:
            raise ValueError("root job cannot bind a predecessor")
        if allowed_predecessors and self.source_selection is None:
            raise ValueError("dependent job lacks a source selection")
        if (
            self.source_selection is not None
            and self.source_selection.stage not in allowed_predecessors
        ):
            raise ValueError("job source has an invalid predecessor stage")
        if self.source_selection is not None and not self.source_selection.resolved:
            raise ValueError("job source selection is unresolved")

    def validate_reuse(self, sources: Mapping[str, JobContract]) -> None:
        target_parameters = self.parameters
        for declaration in self.exact_reuse:
            source = sources.get(declaration.source_job_id)
            if source is None:
                raise ValueError("exact-reuse source job is absent")
            if source.logical_sha256 != declaration.source_contract_sha256:
                raise ValueError("exact-reuse source contract changed")
            source_parameters = source.parameters
            for field in declaration.fields:
                if field not in source_parameters or field not in target_parameters:
                    raise ValueError(f"exact-reuse field {field!r} is absent")
                if source_parameters[field] != target_parameters[field]:
                    raise ValueError(f"exact-reuse field {field!r} changed")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "job_id": self.job_id,
            "stage": self.stage,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "training_horizon": self.training_horizon,
            "schedule": self.schedule,
            "plan_sha256": self.plan_sha256,
            "approval_sha256": self.approval_sha256,
            "parameters": self.parameters,
            "source_selection": (
                None
                if self.source_selection is None
                else self.source_selection.to_dict()
            ),
            "exact_reuse": [declaration.to_dict() for declaration in self.exact_reuse],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> JobContract:
        _require_keys(
            value,
            {
                "schema",
                "job_id",
                "stage",
                "dataset_size",
                "batch_size",
                "training_horizon",
                "schedule",
                "plan_sha256",
                "approval_sha256",
                "parameters",
                "source_selection",
                "exact_reuse",
            },
            "job contract",
        )
        parameters = value["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("job parameters must be an object")
        source_value = value["source_selection"]
        if source_value is not None and not isinstance(source_value, dict):
            raise ValueError("job source selection must be an object or null")
        reuse_value = value["exact_reuse"]
        if not isinstance(reuse_value, list) or not all(
            isinstance(row, dict) for row in reuse_value
        ):
            raise ValueError("job exact reuse must be a list of objects")
        contract = cls(
            job_id=str(value["job_id"]),
            stage=str(value["stage"]),
            parameters_json=canonical_json(parameters),
            source_selection=(
                None
                if source_value is None
                else SelectionBinding.from_dict(source_value)
            ),
            exact_reuse=tuple(ExactReuse.from_dict(row) for row in reuse_value),
            schedule=str(value["schedule"]),  # type: ignore[arg-type]
            dataset_size=str(value["dataset_size"]),
            batch_size=_integer(value["batch_size"], "batch size"),
            training_horizon=_integer(value["training_horizon"], "training horizon"),
            plan_sha256=str(value["plan_sha256"]),
            approval_sha256=str(value["approval_sha256"]),
            schema=str(value["schema"]),
        )
        contract.validate()
        return contract

    def _string_parameter(self, name: str) -> str:
        value = self.parameters.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"job {name} is not a nonempty string")
        return value

    def _optional_string_parameter(self, name: str) -> str | None:
        value = self.parameters.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(f"job {name} is not a nonempty string or null")
        return value

    def _positive_float_parameter(self, name: str) -> float:
        value = self.parameters.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"job {name} is not numeric")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"job {name} is not positive finite")
        return result

    def _optional_int_parameter(self, name: str) -> int | None:
        value = self.parameters.get(name)
        if value is None:
            return None
        return _integer(value, name)

    def _validate_parameter_schema(self, parameters: Mapping[str, object]) -> None:
        expected = set(_COMMON_PARAMETERS)
        if self.stage not in _CONTROL_STAGES:
            expected |= _SEMANTIC_PARAMETERS
        if set(parameters) != expected:
            raise ValueError(f"{self.stage} parameter fields changed")

        expected_builder = "control" if self.stage in _CONTROL_STAGES else "semantic"
        if parameters["builder"] != expected_builder:
            raise ValueError(f"{self.stage} builder is invalid")
        if _safe_path(parameters["runner"], "runner") != RUNNER_PATH:
            raise ValueError("runner is not the approved native-500M entry point")
        _matching_string(parameters["run_name"], _RUN_NAME, "run name")
        _require_sha256(
            _string(parameters["config_logical_sha256"], "config SHA-256"),
            "config",
        )
        _matching_string(parameters["data_group"], _DATA_GROUP, "data group")
        _environment(parameters["environment"])
        environment = parameters["environment"]
        if not isinstance(environment, dict):
            raise ValueError("job environment is not an object")
        tokenizer_fields = _TOKENIZER_ENVIRONMENT_FIELDS & environment.keys()
        if tokenizer_fields and tokenizer_fields != _TOKENIZER_ENVIRONMENT_FIELDS:
            raise ValueError("tokenizer artifact binding is incomplete")
        if (
            self.stage not in _CONTROL_STAGES
            and self.schema == JOB_SCHEMA
            and tokenizer_fields != _TOKENIZER_ENVIRONMENT_FIELDS
        ):
            raise ValueError("semantic job lacks its tokenizer artifact binding")
        if tokenizer_fields:
            if (
                environment["G6_NATIVE500M_TOKENIZER_BINDING_REVISION"]
                != "shared-base-v2"
                or not environment["G6_NATIVE500M_TOKENIZER_BASE_CACHE_KEY"]
            ):
                raise ValueError("tokenizer artifact binding identity differs")
            for name in (
                "G6_NATIVE500M_TOKENIZER_REGISTRY_SHA256",
                "G6_NATIVE500M_TOKENIZER_FIT_SHA256",
                "G6_NATIVE500M_TOKENIZER_CODES_SHA256",
                "G6_NATIVE500M_TOKENIZER_MATERIALIZATION_SHA256",
            ):
                _require_sha256(environment[name], name)
        if parameters["backbone"] not in {"original_g1", "best_g1"}:
            raise ValueError("job backbone is invalid")
        expected_backbone = (
            "original_g1" if self.stage in _ORIGINAL_BACKBONE_STAGES else "best_g1"
        )
        if (
            self.stage not in _CONTROL_STAGES
            and parameters["backbone"] != expected_backbone
        ):
            raise ValueError(f"{self.stage} backbone changed")
        self._positive_float_parameter("embedding_learning_rate")
        self._positive_float_parameter("deep_learning_rate")
        seed = _integer(parameters["seed"], "seed")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        if (
            self.stage == "rq1_confirmation"
            and seed not in {43, 44, 45}
            and not (seed == 42 and self.exact_reuse)
        ):
            raise ValueError("RQ1 confirmation seed is outside 43-45")
        if (
            self.stage == "rq2_rq3_confirmation"
            and seed not in {43, 44}
            and not (seed == 42 and self.exact_reuse)
        ):
            raise ValueError("RQ2/RQ3 confirmation seed is outside 43-44")
        if self.stage not in _CONFIRMATION_STAGES and seed != 42:
            raise ValueError("non-confirmation jobs must use seed 42")
        if self.stage in _CONTROL_STAGES:
            return
        if parameters["representation"] not in REPRESENTATIONS:
            raise ValueError("semantic representation is invalid")
        if parameters["levels"] not in TOKENIZER_LEVELS:
            raise ValueError("semantic levels are outside the approved domain")
        if parameters["shared_codes"] not in SHARED_CODEBOOK_SIZES:
            raise ValueError("shared codes are outside the approved domain")
        if parameters["representation_width"] != REPRESENTATION_WIDTH:
            raise ValueError("representation width changed")
        if parameters["collision_policy"] not in {"suffix", "none"}:
            raise ValueError("collision policy is invalid")
        if parameters["sid_initialization"] not in {"random", "content_pca"}:
            raise ValueError("SID initialization is invalid")
        if self.stage in _RQ0_STAGES and (
            parameters["collision_policy"] != "suffix"
            or parameters["sid_initialization"] != "random"
        ):
            raise ValueError(
                "RQ0 jobs must use suffix policy and random SID initialization"
            )
        if self.stage in _RQ1_STAGES and parameters["collision_policy"] != "suffix":
            raise ValueError("RQ1 jobs must preserve the suffix policy")
        if self.stage in _RQ1_STAGES and parameters["representation"] not in (
            _TRAINABLE_SID_REPRESENTATIONS
        ):
            raise ValueError("RQ1 requires a trainable-SID representation")
        if (
            parameters["sid_initialization"] == "content_pca"
            and parameters["representation"] not in _TRAINABLE_SID_REPRESENTATIONS
        ):
            raise ValueError("content PCA requires a trainable-SID representation")


@dataclass(frozen=True)
class StageManifest:
    stage: str
    jobs: tuple[JobContract, ...]
    predecessor: SelectionBinding | None
    dataset_size: str = DATASET_SIZE
    batch_size: int = BATCH_SIZE
    training_horizon: int = FIXED_HORIZON
    plan_sha256: str = PLAN_SHA256
    approval_sha256: str = APPROVAL_SHA256
    schema: str = MANIFEST_SCHEMA

    @classmethod
    def create(
        cls,
        *,
        stage: str,
        jobs: tuple[JobContract, ...],
        predecessor: SelectionBinding | None = None,
        requires_predecessor: bool | None = None,
    ) -> StageManifest:
        manifest = cls(stage=stage, jobs=jobs, predecessor=predecessor)
        manifest.validate(requires_predecessor=requires_predecessor)
        return manifest

    @property
    def logical_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def validate(self, *, requires_predecessor: bool | None = None) -> None:
        if self.stage not in APPROVED_STAGES:
            raise ValueError("stage manifest has an unknown stage")
        if self.stage != "aggregate" and not self.jobs:
            raise ValueError("stage manifest needs a stage and at least one job")
        if self.stage == "aggregate" and self.jobs:
            raise ValueError("aggregate manifest cannot contain runnable jobs")
        if (
            self.dataset_size != DATASET_SIZE
            or self.batch_size != BATCH_SIZE
            or self.training_horizon != FIXED_HORIZON
        ):
            raise ValueError("stage manifest fixed training contract changed")
        if self.plan_sha256 != PLAN_SHA256 or self.approval_sha256 != APPROVAL_SHA256:
            raise ValueError("stage manifest approval binding changed")
        if self.schema != MANIFEST_SCHEMA:
            raise ValueError("stage manifest schema changed")
        if len({job.job_id for job in self.jobs}) != len(self.jobs):
            raise ValueError("stage manifest has duplicate job IDs")
        if self.predecessor is not None and not self.predecessor.resolved:
            raise ValueError("stage predecessor is not selection-resolved")
        allowed_predecessors = APPROVED_PREDECESSORS[self.stage]
        if self.predecessor is None and allowed_predecessors:
            raise ValueError("stage requires a source selection predecessor")
        if (
            self.predecessor is not None
            and self.predecessor.stage not in allowed_predecessors
        ):
            raise ValueError("stage has an invalid predecessor stage")
        for job in self.jobs:
            job.validate()
            if job.stage != self.stage:
                raise ValueError("job belongs to a different stage")
            if job.source_selection != self.predecessor:
                raise ValueError("job source selection differs from stage predecessor")
        predecessor_required = bool(allowed_predecessors)
        if requires_predecessor is False and predecessor_required:
            raise ValueError("only controls may be predecessor-free")
        if (
            requires_predecessor is True or predecessor_required
        ) and self.predecessor is None:
            raise ValueError("stage requires a source selection predecessor")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "stage": self.stage,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "training_horizon": self.training_horizon,
            "plan_sha256": self.plan_sha256,
            "approval_sha256": self.approval_sha256,
            "predecessor": (
                None if self.predecessor is None else self.predecessor.to_dict()
            ),
            "jobs": [job.to_dict() for job in self.jobs],
        }

    def to_document(self) -> dict[str, object]:
        return self.to_dict() | {"sha256": self.logical_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> StageManifest:
        _require_keys(
            value,
            {
                "schema",
                "stage",
                "dataset_size",
                "batch_size",
                "training_horizon",
                "plan_sha256",
                "approval_sha256",
                "predecessor",
                "jobs",
            },
            "stage manifest",
        )
        jobs_value = value["jobs"]
        if not isinstance(jobs_value, list) or not all(
            isinstance(job, dict) for job in jobs_value
        ):
            raise ValueError("stage manifest jobs must be a list of objects")
        predecessor_value = value["predecessor"]
        if predecessor_value is not None and not isinstance(predecessor_value, dict):
            raise ValueError("stage predecessor must be an object or null")
        manifest = cls(
            stage=str(value["stage"]),
            jobs=tuple(JobContract.from_dict(job) for job in jobs_value),
            predecessor=(
                None
                if predecessor_value is None
                else SelectionBinding.from_dict(predecessor_value)
            ),
            dataset_size=str(value["dataset_size"]),
            batch_size=_integer(value["batch_size"], "batch size"),
            training_horizon=_integer(value["training_horizon"], "training horizon"),
            plan_sha256=str(value["plan_sha256"]),
            approval_sha256=str(value["approval_sha256"]),
            schema=str(value["schema"]),
        )
        manifest.validate()
        return manifest


def load_stage_manifest(
    path: Path, *, expected_stage: str | None = None
) -> StageManifest:
    try:
        document = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("stage manifest is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("stage manifest must be a JSON object")
    sha256 = document.pop("sha256", None)
    manifest = StageManifest.from_dict(document)
    if sha256 != manifest.logical_sha256:
        raise ValueError("stage manifest logical SHA-256 changed")
    if expected_stage is not None and manifest.stage != expected_stage:
        raise ValueError("stage manifest is for a different stage")
    return manifest


@dataclass(frozen=True)
class EnvironmentContract:
    manifest_sha256: str
    stage: str
    dataset_size: str = DATASET_SIZE
    batch_size: int = BATCH_SIZE
    training_horizon: int = FIXED_HORIZON
    plan_sha256: str = PLAN_SHA256
    approval_sha256: str = APPROVAL_SHA256

    @classmethod
    def from_manifest(cls, manifest: StageManifest) -> EnvironmentContract:
        manifest.validate()
        return cls(manifest.logical_sha256, manifest.stage)

    def to_environ(self) -> dict[str, str]:
        return {
            "G6_MANIFEST_SHA256": self.manifest_sha256,
            "G6_STAGE": self.stage,
            "G6_DATASET_SIZE": self.dataset_size,
            "G6_BATCH_SIZE": str(self.batch_size),
            "G6_TRAINING_HORIZON": str(self.training_horizon),
            "G6_PLAN_SHA256": self.plan_sha256,
            "G6_APPROVAL_SHA256": self.approval_sha256,
        }

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str],
        *,
        manifest: StageManifest | None = None,
    ) -> EnvironmentContract:
        required = {
            "G6_MANIFEST_SHA256",
            "G6_STAGE",
            "G6_DATASET_SIZE",
            "G6_BATCH_SIZE",
            "G6_TRAINING_HORIZON",
            "G6_PLAN_SHA256",
            "G6_APPROVAL_SHA256",
        }
        missing = required - environ.keys()
        if missing:
            raise ValueError(f"environment contract is missing {sorted(missing)}")
        contract = cls(
            manifest_sha256=environ["G6_MANIFEST_SHA256"],
            stage=environ["G6_STAGE"],
            dataset_size=environ["G6_DATASET_SIZE"],
            batch_size=int(environ["G6_BATCH_SIZE"]),
            training_horizon=int(environ["G6_TRAINING_HORIZON"]),
            plan_sha256=environ["G6_PLAN_SHA256"],
            approval_sha256=environ["G6_APPROVAL_SHA256"],
        )
        _require_sha256(contract.manifest_sha256, "manifest")
        if contract.stage not in APPROVED_STAGES:
            raise ValueError("environment has an unknown stage")
        if contract.dataset_size != DATASET_SIZE:
            raise ValueError("environment dataset changed")
        if (
            contract.batch_size != BATCH_SIZE
            or contract.training_horizon != FIXED_HORIZON
        ):
            raise ValueError("environment training contract changed")
        if (
            contract.plan_sha256 != PLAN_SHA256
            or contract.approval_sha256 != APPROVAL_SHA256
        ):
            raise ValueError("environment approval binding changed")
        if manifest is not None:
            manifest.validate()
            if contract != cls.from_manifest(manifest):
                raise ValueError(
                    "environment contract differs from manifest stage semantics"
                )
        return contract


def _require_sha256(value: str, name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} SHA-256 is invalid")


def _require_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields changed")


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _matching_string(value: object, pattern: re.Pattern[str], name: str) -> str:
    text = _string(value, name)
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{name} is invalid")
    return text


def _safe_path(value: object, name: str) -> str:
    text = _string(value, name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise ValueError(f"{name} path is invalid")
    return text


def _environment(value: object) -> None:
    if not isinstance(value, dict) or any(
        not isinstance(name, str)
        or _ENVIRONMENT_NAME.fullmatch(name) is None
        or not isinstance(setting, str)
        for name, setting in value.items()
    ):
        raise ValueError("job environment is invalid")
