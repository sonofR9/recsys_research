from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import optuna
from optuna.trial import TrialState

from experiments.g6_rqkmeans_history.protocol.manifest import (
    DEEP_LR_BOUNDS,
    EMBEDDING_LR_BOUNDS,
    ApprovedJob,
    CompiledJob,
    boundary_side,
    outside_boundary_rates,
)


REPRESENTATION = "item_frozen_sid_learned_residual_event"
REPRESENTATION_WIDTHS = (32, 64, 128)
FROZEN_EVENT_WIDTH = 128
CONTROL_JOB_ID = "lr_boundary:boundary_item_frozen_sid_event_embedding_learning_rate_0"
CONTROL_RUN_NAME = (
    "g6_rq0_boundary_item_frozen_sid_event_embedding_learning_rate_0_native50m"
)
ORIGINAL_CONTROL_JOB_ID = "original_control_tuning:original_g1_control_trial_01"
ORIGINAL_CONTROL_RUN_NAME = "g6_rq0_original_g1_control_trial_01_native50m"
RECALL_BAND = 0.002
NDCG_BAND = 0.002
CARRYOVER_MANIFEST_SHA256 = (
    "8eff0bbc85d16d86da8fbedf410569181efe46cda7c691e9e0ba703ec2b896c0"
)
CONTROL_ANCHOR_DEEP_LR = 0.03463626154088337
SAMPLER_SEED = 42
MAXIMUM_PHYSICAL_RUNS = 44
_STAGES = (
    "remediation_tuning",
    "remediation_lr_boundary",
    "remediation_bridge_tuning",
    "remediation_bridge_lr_boundary",
)
_FIXED = {
    "batch_size": 256,
    "validation_batch_size": 8192,
    "representation": REPRESENTATION,
    "num_levels": 3,
    "num_codes": 512,
    "frozen_event_width": FROZEN_EVENT_WIDTH,
    "source_control_job_id": CONTROL_JOB_ID,
    "source_control_run_name": CONTROL_RUN_NAME,
}
_CARRYOVER_PARAMETERS = (
    _FIXED
    | {
        "representation_width": 32,
        "embedding_learning_rate": 0.0018902646022772652,
        "deep_learning_rate": 0.08996404438599374,
    },
    _FIXED
    | {
        "representation_width": 64,
        "embedding_learning_rate": 0.0018902646022772652,
        "deep_learning_rate": 0.08996404438599374,
    },
)
_ANCHOR_PARAMETERS = tuple(
    {
        "representation_width": width,
        "embedding_learning_rate": EMBEDDING_LR_BOUNDS[1],
        "deep_learning_rate": CONTROL_ANCHOR_DEEP_LR,
    }
    for width in REPRESENTATION_WIDTHS
)


@dataclass(frozen=True)
class RemediationManifest:
    jobs: tuple[ApprovedJob, ...]

    @property
    def stage_counts(self) -> dict[str, int]:
        counts = Counter(job.stage for job in self.jobs)
        return {stage: counts[stage] for stage in _STAGES}

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def jobs_for_stage(self, stage: str) -> list[ApprovedJob]:
        return [job for job in self.jobs if job.stage == stage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "dataset_size": "native-50m",
            "initial_runs": 16,
            "maximum_runs": 44,
            "jobs": [job.to_dict() for job in self.jobs],
            "carryovers": [
                {
                    "job_id": job.approved.id,
                    "parameters": job.parameters,
                    "source_manifest_sha256": CARRYOVER_MANIFEST_SHA256,
                }
                for job in carryover_compiled_jobs()
            ],
            "anchors": [
                {"trial": trial, "parameters": parameters}
                for trial, parameters in zip(
                    range(2, 5), _ANCHOR_PARAMETERS, strict=True
                )
            ],
            "sampler": {
                "name": "TPESampler",
                "seed": SAMPLER_SEED,
                "restart_stable_trial_seed": {
                    "hash": "sha256",
                    "material": "{seed}:{stage}:{trial_number}",
                    "integer": "first_4_bytes_big_endian",
                },
            },
            "search_spaces": {
                "fixed_batch_size": 256,
                "fixed_validation_batch_size": 8192,
                "fixed_num_levels": 3,
                "fixed_shared_num_codes": 512,
                "fixed_frozen_event_width": FROZEN_EVENT_WIDTH,
                "learned_sid_width": list(REPRESENTATION_WIDTHS),
                "embedding_learning_rate": [*EMBEDDING_LR_BOUNDS, "log_uniform"],
                "deep_learning_rate": [*DEEP_LR_BOUNDS, "log_uniform"],
                "promotion_bands": {
                    "recall@100": RECALL_BAND,
                    "ndcg@100": NDCG_BAND,
                },
            },
        }


def _job(
    stage: str,
    suffix: str,
    method: str,
    *,
    trial: int,
    conditional: bool,
    learning_rate: str | None = None,
    version: int = 3,
    forced_parameters: dict[str, Any] | None = None,
) -> ApprovedJob:
    if learning_rate is not None and forced_parameters is not None:
        raise ValueError("boundary and fixed parameters cannot both be forced")
    forced = dict(forced_parameters or {})
    if learning_rate is not None:
        forced = {
            "learning_rate": learning_rate,
            "boundary_slot": trial,
        }
    return ApprovedJob(
        id=f"{stage}:{suffix}",
        run_name=f"g6_rq0_remediation_v{version}_{suffix}_native50m",
        stage=stage,
        method=method,
        seed=42,
        trial=trial,
        conditional=conditional,
        forced_parameters=forced,
    )


def _build_jobs() -> tuple[ApprovedJob, ...]:
    jobs = [
        _job(
            "remediation_tuning",
            f"learned_sid_residual_trial_{trial:02d}",
            REPRESENTATION,
            trial=trial,
            conditional=False,
            version=2,
        )
        for trial in range(2)
    ]
    jobs.extend(
        _job(
            "remediation_tuning",
            f"learned_sid_residual_trial_{trial:02d}",
            REPRESENTATION,
            trial=trial,
            conditional=False,
            forced_parameters=(_ANCHOR_PARAMETERS[trial - 2] if trial < 5 else None),
        )
        for trial in range(2, 16)
    )
    jobs.extend(
        _job(
            "remediation_lr_boundary",
            f"boundary_learned_sid_residual_{learning_rate}_{slot}",
            REPRESENTATION,
            trial=slot,
            conditional=True,
            learning_rate=learning_rate,
        )
        for learning_rate in ("embedding_learning_rate", "deep_learning_rate")
        for slot in range(4)
    )
    jobs.extend(
        _job(
            "remediation_bridge_tuning",
            f"original_g1_bridge_trial_{trial:02d}",
            "original_g1_bridge",
            trial=trial,
            conditional=True,
        )
        for trial in range(12)
    )
    jobs.extend(
        _job(
            "remediation_bridge_lr_boundary",
            f"boundary_original_g1_bridge_{learning_rate}_{slot}",
            "original_g1_bridge",
            trial=slot,
            conditional=True,
            learning_rate=learning_rate,
        )
        for learning_rate in ("embedding_learning_rate", "deep_learning_rate")
        for slot in range(4)
    )
    return tuple(jobs)


def carryover_compiled_jobs() -> tuple[CompiledJob, ...]:
    jobs = _MANIFEST.jobs_for_stage("remediation_tuning")[:2]
    return tuple(
        CompiledJob(job, dict(parameters))
        for job, parameters in zip(jobs, _CARRYOVER_PARAMETERS, strict=True)
    )


_MANIFEST = RemediationManifest(_build_jobs())


class RunBudgetApprovalRequired(RuntimeError):
    pass


def remediation_manifest() -> RemediationManifest:
    counts = _MANIFEST.stage_counts
    if counts != dict(zip(_STAGES, (16, 8, 12, 8), strict=True)):
        raise RuntimeError("approved remediation stage counts changed")
    if len(_MANIFEST.jobs) != 44:
        raise RuntimeError("approved remediation budget changed")
    if len({job.id for job in _MANIFEST.jobs}) != 44:
        raise RuntimeError("approved remediation job IDs are not unique")
    if len({job.run_name for job in _MANIFEST.jobs}) != 44:
        raise RuntimeError("approved remediation run names are not unique")
    return _MANIFEST


def _positive_rate(parameters: dict[str, Any], name: str) -> float:
    value = parameters.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")
    return value


def _source_job(job_id: Any) -> ApprovedJob:
    matches = [job for job in remediation_manifest().jobs if job.id == job_id]
    if len(matches) != 1:
        raise ValueError("remediation boundary source is not approved")
    return matches[0]


def _validate_base(parameters: dict[str, Any], *, boundary: bool) -> None:
    for name, expected in _FIXED.items():
        if parameters.get(name) != expected:
            raise ValueError(f"remediation fixed parameter {name} changed")
    if parameters.get("representation_width") not in REPRESENTATION_WIDTHS:
        raise ValueError("learned SID width is outside the approved domain")
    embedding_rate = _positive_rate(parameters, "embedding_learning_rate")
    deep_rate = _positive_rate(parameters, "deep_learning_rate")
    if not boundary:
        if not EMBEDDING_LR_BOUNDS[0] <= embedding_rate <= EMBEDDING_LR_BOUNDS[1]:
            raise ValueError("embedding learning rate is outside the approved search")
        if not DEEP_LR_BOUNDS[0] <= deep_rate <= DEEP_LR_BOUNDS[1]:
            raise ValueError("deep learning rate is outside the approved search")


def validate_remediation_job(compiled: CompiledJob) -> None:
    if compiled.approved not in remediation_manifest().jobs:
        raise ValueError("compiled job identity is not approved for remediation")
    job = compiled.approved
    parameters = compiled.parameters
    bridge = job.stage in {
        "remediation_bridge_tuning",
        "remediation_bridge_lr_boundary",
    }
    boundary = job.stage in {
        "remediation_lr_boundary",
        "remediation_bridge_lr_boundary",
    }
    required = set(_FIXED) | {
        "representation_width",
        "embedding_learning_rate",
        "deep_learning_rate",
    }
    if bridge:
        required |= {
            "selected_treatment_job_id",
            "selected_treatment_run_name",
            "selected_treatment_attempt",
            "selected_treatment_cap_epochs",
            "selected_treatment_parameters",
            "selected_original_control_job_id",
            "selected_original_control_run_name",
        }
        if (
            parameters.get("selected_original_control_job_id")
            != ORIGINAL_CONTROL_JOB_ID
        ):
            raise ValueError("original control identity changed")
        if (
            parameters.get("selected_original_control_run_name")
            != ORIGINAL_CONTROL_RUN_NAME
        ):
            raise ValueError("original control run changed")
        try:
            treatment = _source_job(parameters.get("selected_treatment_job_id"))
        except ValueError as error:
            raise ValueError("selected treatment is not approved") from error
        if treatment.stage not in {"remediation_tuning", "remediation_lr_boundary"}:
            raise ValueError("selected treatment stage changed")
        treatment_parameters = parameters.get("selected_treatment_parameters")
        if not isinstance(treatment_parameters, dict):
            raise ValueError("selected treatment parameters are absent")
        selected_treatment = CompiledJob(
            treatment,
            treatment_parameters,
            parameters.get("selected_treatment_attempt"),
            parameters.get("selected_treatment_cap_epochs"),
        )
        validate_remediation_job(selected_treatment)
        if parameters.get("selected_treatment_run_name") != selected_treatment.run_name:
            raise ValueError("selected treatment run changed")
    if boundary:
        required |= {
            "builder",
            "source_job_id",
            "source_run_name",
            "source_attempt",
            "source_cap_epochs",
            "source_parameters",
            "boundary_side",
        }
    missing = required - parameters.keys()
    unexpected = parameters.keys() - required
    if missing or unexpected:
        raise ValueError(
            f"{job.id} parameter contract differs: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    _validate_base(parameters, boundary=boundary)
    _validate_continuation(compiled, bridge=bridge, boundary=boundary)
    if job.stage == "remediation_tuning" and int(job.trial) < 2:
        expected = _CARRYOVER_PARAMETERS[int(job.trial)]
        if parameters != expected:
            raise ValueError("remediation carryover parameters changed")
    if job.stage == "remediation_tuning" and 2 <= int(job.trial) < 5:
        expected = _FIXED | _ANCHOR_PARAMETERS[int(job.trial) - 2]
        if parameters != expected:
            raise ValueError("remediation anchor parameters changed")
    if not boundary:
        return
    source = _source_job(parameters["source_job_id"])
    expected_stage = "remediation_bridge_tuning" if bridge else "remediation_tuning"
    if source.stage != expected_stage:
        raise ValueError("remediation boundary source stage changed")
    source_parameters = parameters["source_parameters"]
    if not isinstance(source_parameters, dict):
        raise ValueError("remediation boundary source parameters are absent")
    source_compiled = CompiledJob(
        source,
        source_parameters,
        parameters["source_attempt"],
        parameters["source_cap_epochs"],
    )
    validate_remediation_job(source_compiled)
    if parameters["source_run_name"] != source_compiled.run_name:
        raise ValueError("remediation boundary source run changed")
    learning_rate = str(job.forced_parameters["learning_rate"])
    bounds = (
        EMBEDDING_LR_BOUNDS
        if learning_rate == "embedding_learning_rate"
        else DEEP_LR_BOUNDS
    )
    side = boundary_side(float(source_parameters[learning_rate]), bounds)
    if side is None or parameters["boundary_side"] != side:
        raise ValueError("remediation boundary side changed")
    slot = int(job.forced_parameters["boundary_slot"])
    rate = outside_boundary_rates(bounds, side)[slot]
    if parameters[learning_rate] != rate:
        raise ValueError("remediation boundary rate changed")
    metadata = {
        "builder",
        "source_job_id",
        "source_run_name",
        "source_attempt",
        "source_cap_epochs",
        "source_parameters",
        "boundary_side",
    }
    training = {
        name: value for name, value in parameters.items() if name not in metadata
    }
    expected = dict(source_parameters)
    expected[learning_rate] = rate
    if training != expected:
        raise ValueError("remediation boundary changed a non-LR parameter")


class RemediationDriver:
    def __init__(self, storage_path: Path, *, seed: int = SAMPLER_SEED) -> None:
        if seed != SAMPLER_SEED:
            raise ValueError("remediation sampler seed changed")
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage = f"sqlite:///{storage_path.resolve()}"
        self._seed = seed

    def register_carryovers(
        self,
        observations: list[tuple[CompiledJob, float, Path]],
    ) -> None:
        expected = carryover_compiled_jobs()
        if tuple(compiled for compiled, _, _ in observations) != expected:
            raise ValueError("remediation carryover identities changed")
        study = self._study("remediation_tuning", {}, tune_width=True)
        trials = study.trials
        if not trials:
            distributions = {
                "embedding_learning_rate": optuna.distributions.FloatDistribution(
                    *EMBEDDING_LR_BOUNDS, log=True
                ),
                "deep_learning_rate": optuna.distributions.FloatDistribution(
                    *DEEP_LR_BOUNDS, log=True
                ),
                "representation_width": optuna.distributions.CategoricalDistribution(
                    REPRESENTATION_WIDTHS
                ),
            }
            for compiled, recall, artifact in observations:
                study.add_trial(
                    optuna.trial.create_trial(
                        params={
                            name: compiled.parameters[name] for name in distributions
                        },
                        distributions=distributions,
                        value=float(recall),
                        state=TrialState.COMPLETE,
                        user_attrs={"artifact": str(artifact), "carryover": True},
                    )
                )
            trials = study.trials
        self._validate_registered_carryovers(trials, observations)
        anchor_spec = self._anchor_identity()
        existing = study.user_attrs.get("control_anchors")
        if existing is None:
            if len(trials) != len(expected):
                raise ValueError("remediation anchors are absent from a started study")
            for parameters in _ANCHOR_PARAMETERS:
                study.enqueue_trial(parameters)
            study.set_user_attr("control_anchors", anchor_spec)
        elif existing != anchor_spec:
            raise ValueError("remediation anchor identity changed")

    def next_treatment(self) -> CompiledJob | None:
        return self._next("remediation_tuning", {}, tune_width=True)

    def next_bridge(self, treatment: CompiledJob) -> CompiledJob | None:
        validate_remediation_job(treatment)
        if treatment.approved.stage not in {
            "remediation_tuning",
            "remediation_lr_boundary",
        }:
            raise ValueError("bridge source is not an approved treatment")
        fixed = {
            "representation_width": treatment.parameters["representation_width"],
            "selected_treatment_job_id": treatment.approved.id,
            "selected_treatment_run_name": treatment.run_name,
            "selected_treatment_attempt": treatment.attempt,
            "selected_treatment_cap_epochs": treatment.cap_epochs,
            "selected_treatment_parameters": dict(treatment.parameters),
            "selected_original_control_job_id": ORIGINAL_CONTROL_JOB_ID,
            "selected_original_control_run_name": ORIGINAL_CONTROL_RUN_NAME,
        }
        return self._next("remediation_bridge_tuning", fixed, tune_width=False)

    def _next(
        self, stage: str, fixed: dict[str, Any], *, tune_width: bool
    ) -> CompiledJob | None:
        jobs = remediation_manifest().jobs_for_stage(stage)
        study = self._study(stage, fixed, tune_width=tune_width)
        if (
            tune_width
            and study.user_attrs.get("control_anchors") != self._anchor_identity()
        ):
            raise RuntimeError("remediation carryovers must be registered first")
        running = [trial for trial in study.trials if trial.state == TrialState.RUNNING]
        if running:
            trial = running[0]
            return self._compiled(jobs[trial.number], fixed | trial.params)
        completed = [
            trial for trial in study.trials if trial.state == TrialState.COMPLETE
        ]
        if len(completed) == len(jobs):
            return None
        waiting = [trial for trial in study.trials if trial.state == TrialState.WAITING]
        if not waiting:
            study = self._study(
                stage,
                fixed,
                tune_width=tune_width,
                sampler_seed=self._trial_seed(stage, len(study.trials)),
            )
        trial = study.ask()
        parameters = {
            "embedding_learning_rate": trial.suggest_float(
                "embedding_learning_rate", *EMBEDDING_LR_BOUNDS, log=True
            ),
            "deep_learning_rate": trial.suggest_float(
                "deep_learning_rate", *DEEP_LR_BOUNDS, log=True
            ),
        }
        if tune_width:
            parameters["representation_width"] = trial.suggest_categorical(
                "representation_width", list(REPRESENTATION_WIDTHS)
            )
        return self._compiled(jobs[trial.number], fixed | parameters)

    def tell(self, compiled: CompiledJob, recall_at_100: float, artifact: Path) -> None:
        validate_remediation_job(compiled)
        stage = compiled.approved.stage
        if stage not in {"remediation_tuning", "remediation_bridge_tuning"}:
            raise ValueError("only remediation tuning jobs belong to Optuna")
        fixed = self._study_fixed(compiled)
        study = self._study(stage, fixed, tune_width=stage == "remediation_tuning")
        trial = study.trials[int(compiled.approved.trial)]
        if trial.state == TrialState.COMPLETE:
            if trial.value != float(recall_at_100):
                raise ValueError("completed remediation observation changed")
            return
        live = optuna.trial.Trial(study, trial._trial_id)
        live.set_user_attr("artifact", str(artifact))
        study.tell(live, float(recall_at_100))

    def compile_lr_boundaries(self, source: CompiledJob) -> tuple[CompiledJob, ...]:
        validate_remediation_job(source)
        bridge = source.approved.stage == "remediation_bridge_tuning"
        if source.approved.stage not in {
            "remediation_tuning",
            "remediation_bridge_tuning",
        }:
            raise ValueError("boundary source must be an initial remediation trial")
        stage = (
            "remediation_bridge_lr_boundary" if bridge else "remediation_lr_boundary"
        )
        jobs = remediation_manifest().jobs_for_stage(stage)
        compiled: list[CompiledJob] = []
        for learning_rate, bounds in (
            ("embedding_learning_rate", EMBEDDING_LR_BOUNDS),
            ("deep_learning_rate", DEEP_LR_BOUNDS),
        ):
            side = boundary_side(float(source.parameters[learning_rate]), bounds)
            if side is None:
                continue
            slots = [
                job
                for job in jobs
                if job.forced_parameters["learning_rate"] == learning_rate
            ]
            for job, rate in zip(
                slots, outside_boundary_rates(bounds, side), strict=True
            ):
                parameters = dict(source.parameters)
                parameters[learning_rate] = rate
                parameters |= {
                    "builder": "bridge" if bridge else "treatment",
                    "source_job_id": source.approved.id,
                    "source_run_name": source.run_name,
                    "source_attempt": source.attempt,
                    "source_cap_epochs": source.cap_epochs,
                    "source_parameters": dict(source.parameters),
                    "boundary_side": side,
                }
                candidate = CompiledJob(job, parameters)
                validate_remediation_job(candidate)
                compiled.append(candidate)
        return tuple(compiled)

    @staticmethod
    def promotion_eligible(
        *,
        control_recall: float,
        control_ndcg: float,
        treatment_recall: float,
        treatment_ndcg: float,
    ) -> bool:
        return (
            treatment_recall > control_recall + RECALL_BAND
            and treatment_ndcg >= control_ndcg - NDCG_BAND
        )

    def _study(
        self,
        stage: str,
        fixed: dict[str, Any],
        *,
        tune_width: bool,
        sampler_seed: int | None = None,
    ) -> optuna.Study:
        study = optuna.create_study(
            study_name=f"g6-rq0-remediation-v3-{stage}",
            storage=self._storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=self._seed if sampler_seed is None else sampler_seed
            ),
            load_if_exists=True,
        )
        identity = json.dumps(fixed, sort_keys=True)
        existing = study.user_attrs.get("fixed_parameters")
        if existing is None:
            study.set_user_attr("fixed_parameters", identity)
        elif existing != identity:
            raise ValueError("remediation study fixed parameters changed")
        protocol_identity = {
            "manifest_sha256": remediation_manifest().sha256,
            "sampler": remediation_manifest().to_dict()["sampler"],
        }
        existing_protocol = study.user_attrs.get("protocol_identity")
        if existing_protocol is None:
            if study.trials:
                raise ValueError("remediation study protocol identity is absent")
            study.set_user_attr("protocol_identity", protocol_identity)
        elif existing_protocol != protocol_identity:
            raise ValueError("remediation study protocol identity changed")
        return study

    @staticmethod
    def _anchor_identity() -> str:
        return json.dumps(
            [
                {"trial": trial, "parameters": parameters}
                for trial, parameters in zip(
                    range(2, 5), _ANCHOR_PARAMETERS, strict=True
                )
            ],
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _validate_registered_carryovers(
        trials: list[optuna.trial.FrozenTrial],
        observations: list[tuple[CompiledJob, float, Path]],
    ) -> None:
        if len(trials) < len(observations):
            raise ValueError("remediation carryover study is incomplete")
        for trial, (compiled, recall, artifact) in zip(
            trials, observations, strict=False
        ):
            expected = {
                "embedding_learning_rate": compiled.parameters[
                    "embedding_learning_rate"
                ],
                "deep_learning_rate": compiled.parameters["deep_learning_rate"],
                "representation_width": compiled.parameters["representation_width"],
            }
            if (
                trial.state != TrialState.COMPLETE
                or trial.params != expected
                or trial.value != float(recall)
                or trial.user_attrs.get("artifact") != str(artifact)
                or trial.user_attrs.get("carryover") is not True
            ):
                raise ValueError("remediation carryover observation changed")

    def _trial_seed(self, stage: str, trial_number: int) -> int:
        material = f"{self._seed}:{stage}:{trial_number}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")

    @staticmethod
    def _study_fixed(compiled: CompiledJob) -> dict[str, Any]:
        if compiled.approved.stage == "remediation_tuning":
            return {}
        return {
            name: compiled.parameters[name]
            for name in (
                "representation_width",
                "selected_treatment_job_id",
                "selected_treatment_run_name",
                "selected_treatment_attempt",
                "selected_treatment_cap_epochs",
                "selected_treatment_parameters",
                "selected_original_control_job_id",
                "selected_original_control_run_name",
            )
        }

    @staticmethod
    def _compiled(job: ApprovedJob, parameters: dict[str, Any]) -> CompiledJob:
        compiled = CompiledJob(job, _FIXED | parameters)
        validate_remediation_job(compiled)
        return compiled


def _cap_for_attempt(attempt: int) -> int:
    cap = 40
    for _ in range(attempt):
        cap = math.ceil(1.5 * cap)
    return cap


def _validate_continuation(
    compiled: CompiledJob, *, bridge: bool, boundary: bool
) -> None:
    attempt = compiled.attempt
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        raise ValueError("remediation continuation attempt is invalid")
    if attempt == 0:
        if compiled.cap_epochs is not None:
            raise ValueError("initial remediation run cannot override the cap")
        return
    if not bridge:
        raise ValueError("only remediation bridge runs can extend their cap")
    expected = _cap_for_attempt(attempt)
    if compiled.cap_epochs != expected:
        raise ValueError(
            f"remediation continuation attempt {attempt} requires cap {expected}"
        )


def compile_remediation_cap_continuation(compiled: CompiledJob) -> CompiledJob:
    validate_remediation_job(compiled)
    continuation = CompiledJob(
        compiled.approved,
        dict(compiled.parameters),
        attempt=compiled.attempt + 1,
        cap_epochs=_cap_for_attempt(compiled.attempt + 1),
    )
    validate_remediation_job(continuation)
    return continuation
