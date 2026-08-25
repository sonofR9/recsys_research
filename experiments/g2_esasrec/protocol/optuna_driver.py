from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

import optuna
from optuna.trial import TrialState

from experiments.g2_esasrec.analysis.evidence import VerifiedArtifact
from experiments.g2_esasrec.configs.local import (
    COMPONENT_METHODS,
    CONTROL_BATCHES,
    DEEP_LR_BOUNDS,
    EMBEDDING_LR_BOUNDS,
    GBCE_T_BOUNDS,
    LIGR_WIDTHS,
    MIXED_UNIFORM_FRACTION_BOUNDS,
    ComponentMethod,
)
from experiments.g2_esasrec.protocol.manifest import (
    ApprovedJob,
    CompiledJob,
    approved_manifest,
    validate_compiled_job,
)

TuningStage = Literal["control_tuning", "component_tuning", "mixed_tuning"]


@dataclass(frozen=True)
class TrialRecord:
    study_name: str
    slot_id: str
    trial_number: int
    parameters: dict[str, Any]
    objective: float
    artifact: Path


def compile_lr_boundary_jobs(
    winner: VerifiedArtifact,
) -> tuple[CompiledJob, ...]:
    if not isinstance(winner, VerifiedArtifact):
        raise TypeError("prerequisite must be a VerifiedArtifact")
    if winner.job.stage not in {"control_tuning", "component_tuning"}:
        raise ValueError(
            f"verified prerequisite stage {winner.job.stage!r} is not allowed"
        )
    validate_compiled_job(CompiledJob(winner.job, winner.parameters))
    if winner.metadata.get("selection_resolved") is not True:
        raise ValueError("verified prerequisite is not selection-resolved")
    bounds = {
        "embedding_learning_rate": EMBEDDING_LR_BOUNDS,
        "deep_learning_rate": DEEP_LR_BOUNDS,
    }
    edges: list[tuple[str, str]] = []
    for name, (lower, upper) in bounds.items():
        value = float(winner.parameters[name])
        if not lower <= value <= upper:
            raise ValueError(f"selected {name} is outside its approved search")
        if value <= lower * 1.05:
            edges.append((name, "lower"))
        if value >= upper * 0.95:
            edges.append((name, "upper"))
    if len(edges) > 1:
        raise ValueError(
            "both learning rates reached a boundary; two slots cannot resolve both"
        )
    if not edges:
        return ()
    name, edge = edges[0]
    lower, upper = bounds[name]
    outside = lower / 3 if edge == "lower" else upper * 3
    midpoint = lower / math.sqrt(3) if edge == "lower" else upper * math.sqrt(3)
    method = "control" if winner.job.stage == "control_tuning" else winner.job.method
    slots = [
        job
        for job in approved_manifest().jobs_for_stage("lr_boundary")
        if job.method == method
    ]
    compiled = []
    for job, rate in zip(slots, (outside, midpoint), strict=True):
        parameters = dict(winner.parameters)
        parameters["builder"] = (
            "control" if winner.job.stage == "control_tuning" else "component"
        )
        if winner.job.stage == "component_tuning":
            parameters["method"] = winner.job.method
        parameters["source_job_id"] = winner.job.id
        parameters[name] = rate
        candidate = CompiledJob(job, parameters)
        validate_compiled_job(candidate)
        compiled.append(candidate)
    return tuple(compiled)


def require_triggered_lr_boundaries(
    initial_winners: Sequence[VerifiedArtifact],
    compiled_jobs: Sequence[CompiledJob],
) -> tuple[CompiledJob, ...]:
    required = tuple(
        candidate
        for winner in initial_winners
        for candidate in compile_lr_boundary_jobs(winner)
    )
    observed = {compiled.approved.id: compiled for compiled in compiled_jobs}
    missing = [
        candidate.approved.id
        for candidate in required
        if candidate.approved.id not in observed
    ]
    if missing:
        raise ValueError(
            "compiled manifest omits triggered LR boundaries: "
            + ", ".join(sorted(missing))
        )
    for candidate in required:
        if observed[candidate.approved.id] != candidate:
            raise ValueError(f"triggered LR boundary changed: {candidate.approved.id}")
    return required


class G2OptunaDriver:
    def __init__(self, storage_path: Path, *, seed: int = 42) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage = f"sqlite:///{storage_path.resolve()}"
        self._seed = seed

    def next_control(self) -> CompiledJob | None:
        jobs = approved_manifest().jobs_for_stage("control_tuning")

        def suggest(trial: optuna.Trial) -> dict[str, Any]:
            return {
                "batch_size": trial.suggest_categorical(
                    "batch_size", list(CONTROL_BATCHES)
                ),
                "embedding_learning_rate": trial.suggest_float(
                    "embedding_learning_rate", *EMBEDDING_LR_BOUNDS, log=True
                ),
                "deep_learning_rate": trial.suggest_float(
                    "deep_learning_rate", *DEEP_LR_BOUNDS, log=True
                ),
            }

        return self._next("control_tuning", "control", jobs, {}, suggest)

    def next_component(
        self,
        method: ComponentMethod,
        selected_control: VerifiedArtifact,
        *,
        ligr_selection: VerifiedArtifact | None = None,
    ) -> CompiledJob | None:
        if method not in COMPONENT_METHODS:
            raise ValueError(f"{method!r} is not an approved component method")
        self._require_source(selected_control, {"control_tuning", "lr_boundary"})
        self._require_resolved_family(selected_control, "control")
        fixed: dict[str, Any] = {
            "batch_size": selected_control.parameters["batch_size"],
            "selected_control_job_id": selected_control.job.id,
        }
        capacity_is_fixed = (
            method.startswith("matched_standard_") or method == "ligr_gbce"
        )
        if capacity_is_fixed:
            if ligr_selection is None:
                raise ValueError(f"{method} requires the verified LiGR selection")
            self._require_source(ligr_selection, {"component_tuning", "lr_boundary"})
            self._require_resolved_family(ligr_selection, "ligr_sampled_softmax")
            if self._resolved_method(ligr_selection) != "ligr_sampled_softmax":
                raise ValueError(
                    "capacity must come from the LiGR sampled-softmax study"
                )
            fixed |= {
                "source_job_id": ligr_selection.job.id,
                "ligr_multiplier": ligr_selection.parameters["ligr_multiplier"],
            }
        elif ligr_selection is not None:
            raise ValueError(f"{method} does not accept a LiGR capacity prerequisite")
        jobs = [
            job
            for job in approved_manifest().jobs_for_stage("component_tuning")
            if job.method == method
        ]

        def suggest(trial: optuna.Trial) -> dict[str, Any]:
            parameters = dict(fixed)
            parameters.update(
                embedding_learning_rate=trial.suggest_float(
                    "embedding_learning_rate", *EMBEDDING_LR_BOUNDS, log=True
                ),
                deep_learning_rate=trial.suggest_float(
                    "deep_learning_rate", *DEEP_LR_BOUNDS, log=True
                ),
            )
            if method == "ligr_sampled_softmax":
                parameters["ligr_multiplier"] = trial.suggest_categorical(
                    "ligr_multiplier", sorted(LIGR_WIDTHS)
                )
            if method.endswith("_gbce"):
                parameters["gbce_t"] = trial.suggest_float("gbce_t", *GBCE_T_BOUNDS)
            return parameters

        return self._next("component_tuning", method, jobs, fixed, suggest)

    def next_mixed(self, ligr_selection: VerifiedArtifact) -> CompiledJob | None:
        self._require_source(ligr_selection, {"component_tuning", "lr_boundary"})
        self._require_resolved_family(ligr_selection, "ligr_sampled_softmax")
        if self._resolved_method(ligr_selection) != "ligr_sampled_softmax":
            raise ValueError("mixed tuning requires the LiGR sampled-softmax selection")
        fixed = {
            name: ligr_selection.parameters[name]
            for name in (
                "batch_size",
                "embedding_learning_rate",
                "deep_learning_rate",
                "ligr_multiplier",
            )
        }
        selected_control_job_id = ligr_selection.parameters.get(
            "selected_control_job_id"
        )
        if not isinstance(selected_control_job_id, str):
            raise ValueError("LiGR selection has no verified selected-control lineage")
        fixed |= {
            "selected_control_job_id": selected_control_job_id,
            "source_job_id": ligr_selection.job.id,
        }
        jobs = approved_manifest().jobs_for_stage("mixed_tuning")

        def suggest(trial: optuna.Trial) -> dict[str, Any]:
            return dict(fixed) | {
                "uniform_fraction": trial.suggest_float(
                    "uniform_fraction", *MIXED_UNIFORM_FRACTION_BOUNDS
                ),
                "logq_correction": trial.suggest_categorical(
                    "logq_correction", ["none", "yi2019"]
                ),
            }

        return self._next("mixed_tuning", "mixed_sampler", jobs, fixed, suggest)

    def record_observation(
        self, compiled: CompiledJob, artifact: VerifiedArtifact
    ) -> TrialRecord:
        validate_compiled_job(compiled)
        if compiled.approved.stage not in {
            "control_tuning",
            "component_tuning",
            "mixed_tuning",
        }:
            raise ValueError("only tuning jobs belong to Optuna studies")
        self._require_source(artifact, {compiled.approved.stage})
        if (
            artifact.job != compiled.approved
            or artifact.parameters != compiled.parameters
        ):
            raise ValueError("verified artifact does not match the emitted trial")
        objective = artifact.metrics.get("recall@100")
        if (
            not isinstance(objective, (int, float))
            or isinstance(objective, bool)
            or not math.isfinite(float(objective))
            or float(objective) < 0
        ):
            raise ValueError("verified artifact has no finite recall@100 objective")
        objective = float(objective)
        study = self._load_study(
            cast(TuningStage, compiled.approved.stage), compiled.approved.method
        )
        trial_number = compiled.approved.trial
        if trial_number is None:
            raise ValueError("tuning job has no trial number")
        frozen = self._trial(study, trial_number)
        self._require_trial_identity(frozen, compiled)
        artifact_value = str(artifact.path)
        if frozen.state == TrialState.COMPLETE:
            record = self._record(study.study_name, frozen)
            if record.objective != objective or record.artifact != artifact.path:
                raise ValueError("completed trial observation changed")
            return record
        if frozen.state != TrialState.RUNNING:
            raise ValueError("trial is not awaiting an observation")
        live_trial = optuna.trial.Trial(study, frozen._trial_id)
        live_trial.set_user_attr("artifact", artifact_value)
        study.tell(live_trial, objective)
        return self._record(study.study_name, self._trial(study, trial_number))

    def observations(self, stage: TuningStage, method: str) -> tuple[TrialRecord, ...]:
        study = self._load_study(stage, method)
        return tuple(
            self._record(study.study_name, trial)
            for trial in study.get_trials(deepcopy=False)
            if trial.state == TrialState.COMPLETE
        )

    def compile_lr_boundary(self, winner: VerifiedArtifact) -> tuple[CompiledJob, ...]:
        return compile_lr_boundary_jobs(winner)

    def compile_control_repeats(
        self, selected_control: VerifiedArtifact
    ) -> tuple[CompiledJob, ...]:
        self._require_source(selected_control, {"control_tuning", "lr_boundary"})
        self._require_resolved_family(selected_control, "control")
        parameters = dict(selected_control.parameters)
        parameters.pop("builder", None)
        parameters.pop("source_job_id", None)
        parameters |= {
            "selected_control": True,
            "selected_control_job_id": selected_control.job.id,
        }
        compiled = []
        for job in approved_manifest().jobs_for_stage("control_repeats"):
            if job.seed == selected_control.job.seed == 42:
                continue
            candidate = CompiledJob(job, dict(parameters))
            validate_compiled_job(candidate)
            compiled.append(candidate)
        return tuple(compiled)

    def compile_reversal_confirmation(
        self, implicated: Sequence[VerifiedArtifact]
    ) -> tuple[CompiledJob, ...]:
        if len(implicated) != 2:
            raise ValueError(
                "reversal confirmation requires exactly two configurations"
            )
        for artifact in implicated:
            self._require_source(
                artifact,
                {
                    "control_tuning",
                    "component_tuning",
                    "mixed_tuning",
                    "lr_boundary",
                },
            )
        if implicated[0].parameters == implicated[1].parameters:
            raise ValueError("reversal confirmation configurations must differ")
        jobs = approved_manifest().jobs_for_stage("reversal_confirmation")
        compiled = []
        for job in jobs:
            slot = int(job.forced_parameters["configuration_slot"])
            source = implicated[slot]
            parameters = self._resolved_parameters(source)
            parameters["source_job_id"] = source.job.id
            candidate = CompiledJob(job, parameters)
            validate_compiled_job(candidate)
            compiled.append(candidate)
        return tuple(compiled)

    def _next(
        self,
        stage: TuningStage,
        method: str,
        jobs: list[ApprovedJob],
        fixed_parameters: dict[str, Any],
        suggest: Callable[[optuna.Trial], dict[str, Any]],
    ) -> CompiledJob | None:
        study = self._create_study(stage, method, fixed_parameters)
        trials = study.get_trials(deepcopy=False)
        self._validate_trial_sequence(trials, jobs)
        active = [
            trial
            for trial in trials
            if trial.state in {TrialState.RUNNING, TrialState.WAITING}
        ]
        if active:
            frozen = active[0]
            if frozen.state == TrialState.WAITING:
                study = self._study_with_trial_seed(stage, method, frozen.number)
                trial = study.ask()
                parameters = suggest(trial)
                parameters.update(jobs[trial.number].forced_parameters)
                return self._store_compiled(trial, jobs[trial.number], parameters)
            return self._compiled_from_trial(frozen, jobs[frozen.number])
        completed = [trial for trial in trials if trial.state == TrialState.COMPLETE]
        if len(completed) == len(jobs):
            return None
        number = len(completed)
        job = jobs[number]
        study = self._study_with_trial_seed(stage, method, number)
        if job.forced_parameters:
            study.enqueue_trial(job.forced_parameters, skip_if_exists=True)
        trial = study.ask()
        if trial.number != number:
            raise RuntimeError("Optuna trial number diverged from the approved slot")
        parameters = suggest(trial)
        parameters.update(job.forced_parameters)
        return self._store_compiled(trial, job, parameters)

    def _create_study(
        self,
        stage: TuningStage,
        method: str,
        fixed_parameters: dict[str, Any],
    ) -> optuna.Study:
        study_name = self._study_name(stage, method)
        study = optuna.create_study(
            storage=self._storage,
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self._trial_seed(study_name, 0)),
            load_if_exists=True,
        )
        expected = {
            "manifest_sha256": approved_manifest().sha256,
            "driver_seed": self._seed,
            "fixed_parameters": fixed_parameters,
        }
        for name, value in expected.items():
            if name in study.user_attrs and study.user_attrs[name] != value:
                message = (
                    "fixed parameters changed"
                    if name == "fixed_parameters"
                    else f"study {name} changed"
                )
                raise ValueError(message)
            if name not in study.user_attrs:
                study.set_user_attr(name, value)
        return study

    def _load_study(self, stage: TuningStage, method: str) -> optuna.Study:
        study_name = self._study_name(stage, method)
        try:
            study = optuna.load_study(study_name=study_name, storage=self._storage)
        except KeyError as error:
            raise ValueError(f"study {study_name!r} does not exist") from error
        if study.user_attrs.get("manifest_sha256") != approved_manifest().sha256:
            raise ValueError("study manifest_sha256 changed")
        if study.user_attrs.get("driver_seed") != self._seed:
            raise ValueError("study driver_seed changed")
        return study

    def _study_with_trial_seed(
        self, stage: TuningStage, method: str, trial_number: int
    ) -> optuna.Study:
        study_name = self._study_name(stage, method)
        return optuna.load_study(
            study_name=study_name,
            storage=self._storage,
            sampler=optuna.samplers.TPESampler(
                seed=self._trial_seed(study_name, trial_number)
            ),
        )

    def _study_name(self, stage: TuningStage, method: str) -> str:
        if stage == "control_tuning" and method != "control":
            raise ValueError("control study method changed")
        if stage == "component_tuning" and method not in COMPONENT_METHODS:
            raise ValueError(f"{method!r} is not an approved component method")
        if stage == "mixed_tuning" and method != "mixed_sampler":
            raise ValueError("mixed study method changed")
        return f"g2-v1:{stage}:{method}"

    def _trial_seed(self, study_name: str, trial_number: int) -> int:
        payload = f"{self._seed}:{study_name}:{trial_number}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

    def _store_compiled(
        self, trial: optuna.Trial, job: ApprovedJob, parameters: dict[str, Any]
    ) -> CompiledJob:
        compiled = CompiledJob(job, parameters)
        validate_compiled_job(compiled)
        trial.set_user_attr("job_id", job.id)
        trial.set_user_attr("compiled_parameters", parameters)
        return compiled

    def _compiled_from_trial(
        self, trial: optuna.trial.FrozenTrial, job: ApprovedJob
    ) -> CompiledJob:
        parameters = trial.user_attrs.get("compiled_parameters")
        if trial.user_attrs.get("job_id") != job.id or not isinstance(parameters, dict):
            raise ValueError("pending trial does not match its approved slot")
        compiled = CompiledJob(job, parameters)
        validate_compiled_job(compiled)
        return compiled

    def _validate_trial_sequence(
        self,
        trials: Sequence[optuna.trial.FrozenTrial],
        jobs: Sequence[ApprovedJob],
    ) -> None:
        if len(trials) > len(jobs):
            raise ValueError("study contains an unapproved trial slot")
        if [trial.number for trial in trials] != list(range(len(trials))):
            raise ValueError("study trial numbering is not contiguous")
        active_seen = False
        for trial in trials:
            if trial.state == TrialState.COMPLETE:
                if active_seen:
                    raise ValueError("study observed a slot before its predecessor")
            elif trial.state in {TrialState.RUNNING, TrialState.WAITING}:
                if active_seen:
                    raise ValueError("study has more than one pending slot")
                active_seen = True
            else:
                raise ValueError("study contains a non-observed terminal trial")

    def _trial(
        self, study: optuna.Study, trial_number: int
    ) -> optuna.trial.FrozenTrial:
        matches = [
            trial
            for trial in study.get_trials(deepcopy=False)
            if trial.number == trial_number
        ]
        if len(matches) != 1:
            raise ValueError(f"study has no unique trial {trial_number}")
        return matches[0]

    def _require_trial_identity(
        self, trial: optuna.trial.FrozenTrial, compiled: CompiledJob
    ) -> None:
        if trial.user_attrs.get("job_id") != compiled.approved.id:
            raise ValueError("Optuna trial job identity changed")
        if trial.user_attrs.get("compiled_parameters") != compiled.parameters:
            raise ValueError("Optuna trial parameters changed")

    def _record(self, study_name: str, trial: optuna.trial.FrozenTrial) -> TrialRecord:
        parameters = trial.user_attrs.get("compiled_parameters")
        artifact = trial.user_attrs.get("artifact")
        if (
            trial.state != TrialState.COMPLETE
            or trial.value is None
            or not isinstance(parameters, dict)
            or not isinstance(artifact, str)
        ):
            raise ValueError("completed trial record is incomplete")
        return TrialRecord(
            study_name=study_name,
            slot_id=str(trial.user_attrs["job_id"]),
            trial_number=trial.number,
            parameters=parameters,
            objective=float(trial.value),
            artifact=Path(artifact),
        )

    def _require_source(self, artifact: VerifiedArtifact, stages: set[str]) -> None:
        if not isinstance(artifact, VerifiedArtifact):
            raise TypeError("prerequisite must be a VerifiedArtifact")
        if artifact.job.stage not in stages:
            raise ValueError(
                f"verified prerequisite stage {artifact.job.stage!r} is not allowed"
            )
        validate_compiled_job(CompiledJob(artifact.job, artifact.parameters))
        if artifact.metadata.get("selection_resolved") is not True:
            raise ValueError("verified prerequisite is not selection-resolved")

    def _resolved_parameters(self, artifact: VerifiedArtifact) -> dict[str, Any]:
        parameters = dict(artifact.parameters)
        if artifact.job.stage == "control_tuning":
            parameters["builder"] = "control"
        elif artifact.job.stage == "component_tuning":
            parameters["builder"] = "component"
            parameters["method"] = artifact.job.method
        elif artifact.job.stage == "mixed_tuning":
            parameters["builder"] = "mixed_sampler"
        elif "builder" not in parameters:
            raise ValueError("conditional prerequisite has no resolved builder")
        return parameters

    @staticmethod
    def _resolved_method(artifact: VerifiedArtifact) -> str:
        if artifact.job.stage == "lr_boundary":
            return str(artifact.parameters.get("method", artifact.job.method))
        return artifact.job.method

    def _require_resolved_family(
        self, artifact: VerifiedArtifact, expected: str
    ) -> None:
        if artifact.job.stage != "lr_boundary":
            return
        builder = artifact.parameters.get("builder")
        if expected == "control":
            matches = builder == "control"
        else:
            matches = (
                builder == "component" and self._resolved_method(artifact) == expected
            )
        if not matches:
            raise ValueError(f"boundary prerequisite is not resolved {expected}")


OptunaDriver = G2OptunaDriver
