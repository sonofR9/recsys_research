from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from experiments.g2_esasrec.configs.local import COMPONENT_METHODS, LIGR_WIDTHS

MAX_RUNS = 135
MANIFEST_VERSION = 1
Stage = Literal[
    "control_tuning",
    "control_repeats",
    "component_tuning",
    "mixed_tuning",
    "official",
    "lr_boundary",
    "reversal_confirmation",
]

EXPECTED_STAGE_COUNTS = {
    "control_tuning": 20,
    "control_repeats": 10,
    "component_tuning": 72,
    "mixed_tuning": 12,
    "official": 3,
    "lr_boundary": 14,
    "reversal_confirmation": 4,
}


@dataclass(frozen=True)
class ApprovedJob:
    id: str
    run_name: str
    stage: Stage
    method: str
    seed: int
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
            "maximum_runs": MAX_RUNS,
            "jobs": [job.to_dict() for job in self.jobs],
            "search_spaces": _search_spaces(),
        }


@dataclass(frozen=True)
class CompiledJob:
    approved: ApprovedJob
    parameters: dict[str, Any]

    def to_contract(self, manifest: ApprovedManifest) -> dict[str, Any]:
        return {
            "manifest_sha256": manifest.sha256,
            "job": self.approved.to_dict(),
            "parameters": self.parameters,
        }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
        run_name=f"g2_{suffix}_native50m",
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
        _job("control_tuning", f"control_trial_{trial:02d}", "control", trial=trial)
        for trial in range(20)
    )
    jobs.extend(
        _job(
            "control_repeats",
            f"control_seed_{seed}",
            "control",
            seed=seed,
            conditional=seed == 42,
            forced_parameters={"selected_control": True},
        )
        for seed in range(42, 52)
    )
    for method in COMPONENT_METHODS:
        for trial in range(12):
            forced: dict[str, Any] = {}
            if trial == 0:
                forced = {
                    "embedding_learning_rate": 0.001,
                    "deep_learning_rate": 0.001,
                }
                if method == "ligr_sampled_softmax":
                    forced["ligr_multiplier"] = 4
                if method.endswith("_gbce"):
                    forced["gbce_t"] = 0.75
            jobs.append(
                _job(
                    "component_tuning",
                    f"{method}_trial_{trial:02d}",
                    method,
                    trial=trial,
                    forced_parameters=forced,
                )
            )
    mixed_anchors = (
        {"uniform_fraction": 0.6, "logq_correction": "none"},
        {"uniform_fraction": 0.6, "logq_correction": "yi2019"},
    )
    jobs.extend(
        _job(
            "mixed_tuning",
            f"mixed_trial_{trial:02d}",
            "mixed_sampler",
            trial=trial,
            forced_parameters=mixed_anchors[trial] if trial < 2 else {},
        )
        for trial in range(12)
    )
    jobs.extend(
        _job(
            "official",
            f"official_esasrec_s{seed}",
            "official_rectools",
            seed=seed,
            forced_parameters={"rectools_version": "0.19.0"},
        )
        for seed in (42, 43, 44)
    )
    for method in ("control", *COMPONENT_METHODS):
        jobs.extend(
            _job(
                "lr_boundary",
                f"boundary_{method}_{index}",
                method,
                trial=index,
                conditional=True,
                forced_parameters={"boundary_slot": index},
            )
            for index in range(2)
        )
    for configuration in range(2):
        jobs.extend(
            _job(
                "reversal_confirmation",
                f"confirmation_{configuration}_s{seed}",
                "implicated_configuration",
                seed=seed,
                trial=configuration,
                conditional=True,
                forced_parameters={"configuration_slot": configuration},
            )
            for seed in (43, 44)
        )
    return tuple(jobs)


def _search_spaces() -> dict[str, Any]:
    return {
        "control": {
            "batch_size": [128, 256, 512, 1024, 1280],
            "embedding_learning_rate": [0.0001, 0.256, "log_uniform"],
            "deep_learning_rate": [0.0001, 0.128, "log_uniform"],
        },
        "component": {
            "embedding_learning_rate": [0.0001, 0.256, "log_uniform"],
            "deep_learning_rate": [0.0001, 0.128, "log_uniform"],
            "ligr_multiplier": sorted(LIGR_WIDTHS),
            "gbce_t": [0.25, 1.0, "continuous"],
        },
        "mixed": {
            "uniform_fraction": [0.2, 0.8, "continuous"],
            "logq_correction": ["none", "yi2019"],
        },
        "lr_boundary": {
            "trigger_distance_fraction": 0.05,
            "outside_multiplier": 3.0,
            "slots_per_method": 2,
        },
        "reversal_confirmation": {"configuration_slots": 2, "seeds": [43, 44]},
    }


_APPROVED_MANIFEST = ApprovedManifest(_build_jobs())


def approved_manifest() -> ApprovedManifest:
    validate_approved_manifest(_APPROVED_MANIFEST.to_dict())
    return _APPROVED_MANIFEST


def validate_approved_manifest(document: dict[str, Any]) -> None:
    if _canonical_json(document) != _canonical_json(_APPROVED_MANIFEST.to_dict()):
        raise ValueError("document does not match the approved manifest")
    jobs = document["jobs"]
    if len(jobs) != MAX_RUNS:
        raise ValueError("approved manifest does not account for 135 runs")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("approved manifest contains duplicate job IDs")
    if len({job["run_name"] for job in jobs}) != len(jobs):
        raise ValueError("approved manifest contains duplicate run names")
    if _APPROVED_MANIFEST.stage_counts != EXPECTED_STAGE_COUNTS:
        raise ValueError("approved manifest stage counts changed")


def write_approved_manifest(path: Path) -> None:
    manifest = approved_manifest()
    document = manifest.to_dict() | {"sha256": manifest.sha256}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _finite_rate(parameters: dict[str, Any], name: str) -> float:
    value = parameters.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be resolved")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")
    return value


_BASE_PARAMETERS = {
    "batch_size",
    "embedding_learning_rate",
    "deep_learning_rate",
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


def _inherits_ligr_capacity(method: str) -> bool:
    return method.startswith("matched_standard_") or method == "ligr_gbce"


def _component_parameters(method: str) -> set[str]:
    required = _BASE_PARAMETERS | {"selected_control_job_id"}
    if method.startswith(("ligr_", "matched_standard_")):
        required |= {"ligr_multiplier"}
    if method.endswith("_gbce"):
        required |= {"gbce_t"}
    if _inherits_ligr_capacity(method):
        required |= {"source_job_id"}
    return required


def _builder_parameters(builder: str, method: Any) -> set[str]:
    if builder == "control":
        return _BASE_PARAMETERS | {"builder"}
    if builder == "component":
        if method not in COMPONENT_METHODS:
            raise ValueError("conditional component has an unknown method")
        return _component_parameters(str(method)) | {"builder", "method"}
    if builder == "mixed_sampler":
        return _BASE_PARAMETERS | {
            "builder",
            "selected_control_job_id",
            "ligr_multiplier",
            "uniform_fraction",
            "logq_correction",
        }
    raise ValueError(f"unknown conditional builder {builder!r}")


def _job_by_id(job_id: Any) -> ApprovedJob:
    if not isinstance(job_id, str):
        raise ValueError("prerequisite job ID is missing")
    matches = [job for job in _APPROVED_MANIFEST.jobs if job.id == job_id]
    if len(matches) != 1:
        raise ValueError(f"prerequisite job {job_id!r} is not approved")
    return matches[0]


def _validate_selected_control(parameters: dict[str, Any]) -> None:
    selected = _job_by_id(parameters.get("selected_control_job_id"))
    if selected.stage == "control_tuning":
        return
    if selected.stage == "lr_boundary" and selected.method == "control":
        return
    raise ValueError("selected batch does not come from an approved control selection")


def _validate_component_domains(job: ApprovedJob, parameters: dict[str, Any]) -> None:
    if job.method.startswith(("ligr_", "matched_standard_")):
        if parameters.get("ligr_multiplier") not in LIGR_WIDTHS:
            raise ValueError("ligr_multiplier is outside the approved domain")
    if job.method.endswith("_gbce"):
        value = parameters.get("gbce_t")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.25 <= float(value) <= 1.0
        ):
            raise ValueError("gbce_t is outside the approved domain")


def _validate_ligr_capacity_source(parameters: dict[str, Any]) -> None:
    source = _job_by_id(parameters.get("source_job_id"))
    if not (
        source.method == "ligr_sampled_softmax"
        and source.stage in {"component_tuning", "lr_boundary"}
    ):
        raise ValueError("LiGR capacity must come from the selected LiGR source")


def _validate_mixed_domains(parameters: dict[str, Any]) -> None:
    if parameters.get("ligr_multiplier") not in LIGR_WIDTHS:
        raise ValueError("ligr_multiplier is outside the approved domain")
    fraction = parameters.get("uniform_fraction")
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not math.isfinite(float(fraction))
        or not 0.2 <= float(fraction) <= 0.8
    ):
        raise ValueError("uniform_fraction is outside the approved domain")
    if parameters.get("logq_correction") not in {"none", "yi2019"}:
        raise ValueError("logq_correction is outside the approved domain")


def _validate_conditional_source(job: ApprovedJob, parameters: dict[str, Any]) -> None:
    source = _job_by_id(parameters.get("source_job_id"))
    builder = parameters.get("builder")
    if job.stage == "lr_boundary":
        expected_stage = (
            "control_tuning" if job.method == "control" else "component_tuning"
        )
        if source.stage != expected_stage or source.method != job.method:
            raise ValueError(f"{job.id} conditional family changed")
        expected_builder = "control" if job.method == "control" else "component"
        if builder != expected_builder or (
            builder == "component" and parameters.get("method") != job.method
        ):
            raise ValueError(f"{job.id} conditional family changed")
        return
    allowed = {
        "reversal_confirmation": {
            "control_tuning",
            "component_tuning",
            "mixed_tuning",
            "lr_boundary",
        },
    }[job.stage]
    if source.stage not in allowed:
        raise ValueError(f"{job.id} prerequisite stage changed")
    if source.stage == "control_tuning" and builder != "control":
        raise ValueError(f"{job.id} conditional family changed")
    if source.stage == "component_tuning" and (
        builder != "component" or parameters.get("method") != source.method
    ):
        raise ValueError(f"{job.id} conditional family changed")
    if source.stage == "mixed_tuning" and builder != "mixed_sampler":
        raise ValueError(f"{job.id} conditional family changed")
    if (
        source.stage == "lr_boundary"
        and source.method != "control"
        and (builder != "component" or parameters.get("method") != source.method)
    ):
        raise ValueError(f"{job.id} conditional family changed")
    if (
        source.stage == "lr_boundary"
        and source.method == "control"
        and builder != "control"
    ):
        raise ValueError(f"{job.id} conditional family changed")


def _validate_parameters(job: ApprovedJob, parameters: dict[str, Any]) -> None:
    for name, expected in job.forced_parameters.items():
        if name in {
            "selected_control",
            "rectools_version",
            "boundary_slot",
            "configuration_slot",
        }:
            continue
        if parameters.get(name) != expected:
            raise ValueError(f"{job.id} must preserve forced {name}={expected!r}")
    if job.stage == "official":
        _require_exact_parameters(job, parameters, {"rectools_version"})
        if parameters["rectools_version"] != "0.19.0":
            raise ValueError("official jobs require RecTools 0.19.0")
        return
    if job.stage == "control_tuning":
        required = set(_BASE_PARAMETERS)
    elif job.stage == "control_repeats":
        required = _BASE_PARAMETERS | {"selected_control", "selected_control_job_id"}
        if parameters.get("selected_control") is not True:
            raise ValueError("control repeat must preserve the selected control")
        _validate_selected_control(parameters)
    elif job.stage == "component_tuning":
        required = _component_parameters(job.method)
        _validate_selected_control(parameters)
        if _inherits_ligr_capacity(job.method):
            _validate_ligr_capacity_source(parameters)
    elif job.stage == "mixed_tuning":
        required = _BASE_PARAMETERS | {
            "selected_control_job_id",
            "source_job_id",
            "ligr_multiplier",
            "uniform_fraction",
            "logq_correction",
        }
        _validate_selected_control(parameters)
        source = _job_by_id(parameters.get("source_job_id"))
        if not (
            source.method == "ligr_sampled_softmax"
            and source.stage in {"component_tuning", "lr_boundary"}
        ):
            raise ValueError("mixed tuning requires the selected LiGR family")
    elif job.stage in {"lr_boundary", "reversal_confirmation"}:
        required = _builder_parameters(
            str(parameters.get("builder")), parameters.get("method")
        ) | {"source_job_id"}
        if parameters.get("builder") != "control":
            _validate_selected_control(parameters)
        _validate_conditional_source(job, parameters)
    else:
        raise ValueError(f"unknown stage {job.stage!r}")
    _require_exact_parameters(job, parameters, required)
    batch_size = parameters.get("batch_size")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size not in {128, 256, 512, 1024, 1280}
    ):
        raise ValueError("batch_size is outside the approved domain")
    embedding_rate = _finite_rate(parameters, "embedding_learning_rate")
    deep_rate = _finite_rate(parameters, "deep_learning_rate")
    if job.stage in {"control_tuning", "component_tuning"} and not (
        0.0001 <= embedding_rate <= 0.256 and 0.0001 <= deep_rate <= 0.128
    ):
        raise ValueError("learning rate is outside the approved search range")
    if job.stage == "component_tuning":
        _validate_component_domains(job, parameters)
    if parameters.get("builder") == "component":
        proxy = ApprovedJob(
            "resolved", "resolved", "component_tuning", str(parameters["method"]), 42
        )
        _validate_component_domains(proxy, parameters)
    if job.stage == "mixed_tuning" or parameters.get("builder") == "mixed_sampler":
        _validate_mixed_domains(parameters)


def validate_compiled_job(compiled: CompiledJob) -> None:
    manifest = approved_manifest()
    matches = [job for job in manifest.jobs if job == compiled.approved]
    if len(matches) != 1:
        raise ValueError("compiled job identity is not approved")
    _validate_parameters(compiled.approved, compiled.parameters)


def load_compiled_jobs(path: Path) -> list[CompiledJob]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read compiled manifest {path}") from error
    manifest = approved_manifest()
    if document.get("manifest_sha256") != manifest.sha256:
        raise ValueError("compiled manifest does not reference the approved manifest")
    rows = document.get("jobs")
    if not isinstance(rows, list):
        raise ValueError("compiled manifest jobs must be a list")
    by_id = {job.id: job for job in manifest.jobs}
    seen: set[str] = set()
    compiled: list[CompiledJob] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("compiled job must be an object")
        job_id = row.get("id")
        if job_id in seen:
            raise ValueError(f"duplicate compiled job {job_id!r}")
        seen.add(job_id)
        if job_id not in by_id:
            raise ValueError(f"unknown compiled job {job_id!r}")
        approved = by_id[job_id]
        if row.get("run_name") != approved.run_name:
            raise ValueError(f"run name changed for {job_id}")
        parameters = row.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"parameters are unresolved for {job_id}")
        _validate_parameters(approved, parameters)
        job = CompiledJob(approved, parameters)
        validate_compiled_job(job)
        compiled.append(job)
    return compiled
