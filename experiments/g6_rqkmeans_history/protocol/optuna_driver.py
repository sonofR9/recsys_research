from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Literal, cast

import optuna
from optuna.trial import TrialState

from experiments.g6_rqkmeans_history.protocol.manifest import (
    CONTROL_BATCHES,
    DEEP_LR_BOUNDS,
    EMBEDDING_LR_BOUNDS,
    NUM_CODES,
    NUM_LEVELS,
    REPRESENTATIONS,
    REPRESENTATION_WIDTHS,
    ApprovedJob,
    CompiledJob,
    Representation,
    approved_manifest,
    boundary_side,
    outside_boundary_rates,
    validate_compiled_job,
)


TuningStage = Literal[
    "primary_control_tuning",
    "original_control_tuning",
    "treatment_tuning",
    "bridge_tuning",
]


@dataclass(frozen=True)
class Selection:
    compiled: CompiledJob
    objective: float
    selection_resolved: bool


@dataclass(frozen=True)
class TrialRecord:
    study_name: str
    job_id: str
    trial_number: int
    parameters: dict[str, Any]
    objective: float
    artifact: Path


class G6Rq0OptunaDriver:
    def __init__(
        self,
        storage_path: Path,
        *,
        feasible_training_batches: tuple[int, ...],
        validation_batch_size: int,
        seed: int = 42,
    ) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if (
            not feasible_training_batches
            or len(set(feasible_training_batches)) != len(feasible_training_batches)
            or any(batch not in CONTROL_BATCHES for batch in feasible_training_batches)
        ):
            raise ValueError(
                "feasible training batches must be a unique approved subset"
            )
        if (
            not isinstance(validation_batch_size, int)
            or isinstance(validation_batch_size, bool)
            or validation_batch_size < 1
        ):
            raise ValueError("validation_batch_size must be a positive integer")
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage = f"sqlite:///{storage_path.resolve()}"
        self._seed = seed
        self._feasible_training_batches = tuple(sorted(feasible_training_batches))
        self._validation_batch_size = validation_batch_size

    def next_primary_control(self) -> CompiledJob | None:
        jobs = approved_manifest().jobs_for_stage("primary_control_tuning")

        def suggest(trial: optuna.Trial) -> dict[str, Any]:
            return {
                "batch_size": trial.suggest_categorical(
                    "batch_size", list(self._feasible_training_batches)
                ),
                "validation_batch_size": self._validation_batch_size,
                **self._suggest_rates(trial),
            }

        return self._next(
            "primary_control_tuning",
            "best_g1_item_ids",
            jobs,
            {},
            suggest,
        )

    def next_original_control(self, primary: Selection) -> CompiledJob | None:
        self._require_primary(primary)
        fixed = {
            "batch_size": primary.compiled.parameters["batch_size"],
            "validation_batch_size": primary.compiled.parameters[
                "validation_batch_size"
            ],
            "selected_primary_control_job_id": primary.compiled.approved.id,
        }
        jobs = approved_manifest().jobs_for_stage("original_control_tuning")

        def suggest(trial: optuna.Trial) -> dict[str, Any]:
            return fixed | self._suggest_rates(trial)

        return self._next(
            "original_control_tuning",
            "original_g1_item_ids",
            jobs,
            fixed,
            suggest,
        )

    def next_treatment(
        self,
        representation: Representation,
        primary: Selection,
    ) -> CompiledJob | None:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"{representation!r} is not an approved representation")
        self._require_primary(primary)
        fixed = {
            "batch_size": primary.compiled.parameters["batch_size"],
            "validation_batch_size": primary.compiled.parameters[
                "validation_batch_size"
            ],
            "selected_primary_control_job_id": primary.compiled.approved.id,
            "representation": representation,
        }
        jobs = [
            job
            for job in approved_manifest().jobs_for_stage("treatment_tuning")
            if job.method == representation
        ]

        def suggest(trial: optuna.Trial) -> dict[str, Any]:
            return (
                fixed
                | self._suggest_rates(trial)
                | {
                    "num_levels": trial.suggest_categorical(
                        "num_levels", list(NUM_LEVELS)
                    ),
                    "num_codes": trial.suggest_categorical(
                        "num_codes", list(NUM_CODES)
                    ),
                    "representation_width": trial.suggest_categorical(
                        "representation_width", list(REPRESENTATION_WIDTHS)
                    ),
                }
            )

        return self._next(
            "treatment_tuning",
            representation,
            jobs,
            fixed,
            suggest,
        )

    def next_bridge(
        self,
        primary: Selection,
        original: Selection,
        treatment: Selection,
    ) -> CompiledJob | None:
        self._require_primary(primary)
        self._require_treatment(treatment)
        self._require_original(original)
        semantic = {
            name: treatment.compiled.parameters[name]
            for name in (
                "representation",
                "num_levels",
                "num_codes",
                "representation_width",
            )
        }
        fixed = semantic | {
            "batch_size": primary.compiled.parameters["batch_size"],
            "validation_batch_size": primary.compiled.parameters[
                "validation_batch_size"
            ],
            "selected_primary_control_job_id": primary.compiled.approved.id,
            "selected_original_control_job_id": original.compiled.approved.id,
            "selected_treatment_job_id": treatment.compiled.approved.id,
        }
        jobs = approved_manifest().jobs_for_stage("bridge_tuning")

        def suggest(trial: optuna.Trial) -> dict[str, Any]:
            return fixed | self._suggest_rates(trial)

        return self._next(
            "bridge_tuning",
            "selected_semantic_bridge",
            jobs,
            fixed,
            suggest,
        )

    def tell(
        self,
        compiled: CompiledJob,
        recall_at_100: float,
        artifact: Path,
    ) -> TrialRecord:
        validate_compiled_job(compiled)
        if compiled.approved.stage not in {
            "primary_control_tuning",
            "original_control_tuning",
            "treatment_tuning",
            "bridge_tuning",
        }:
            raise ValueError("only initial tuning jobs belong to Optuna studies")
        if (
            not isinstance(recall_at_100, (int, float))
            or isinstance(recall_at_100, bool)
            or not math.isfinite(float(recall_at_100))
            or float(recall_at_100) < 0
        ):
            raise ValueError("recall@100 must be finite and non-negative")
        stage = cast(TuningStage, compiled.approved.stage)
        study = self._load_study(stage, compiled.approved.method)
        trial_number = compiled.approved.trial
        if trial_number is None:
            raise ValueError("tuning job has no trial number")
        frozen = self._trial(study, trial_number)
        self._require_trial_identity(frozen, compiled)
        if frozen.state == TrialState.COMPLETE:
            existing = self._record(study.study_name, frozen)
            if (
                existing.objective != float(recall_at_100)
                or existing.artifact != artifact
            ):
                raise ValueError("completed trial observation changed")
            return existing
        if frozen.state != TrialState.RUNNING:
            raise ValueError("trial is not awaiting an observation")
        live = optuna.trial.Trial(study, frozen._trial_id)
        live.set_user_attr("artifact", str(artifact))
        study.tell(live, float(recall_at_100))
        return self._record(study.study_name, self._trial(study, trial_number))

    def compile_primary_repeats(self, primary: Selection) -> tuple[CompiledJob, ...]:
        self._require_primary(primary)
        parameters = self._base_parameters(primary.compiled) | {
            "selected_primary_control_job_id": primary.compiled.approved.id
        }
        compiled = tuple(
            CompiledJob(job, dict(parameters))
            for job in approved_manifest().jobs_for_stage("primary_control_repeats")
        )
        for candidate in compiled:
            validate_compiled_job(candidate)
        return compiled

    def compile_lr_boundaries(
        self, initial_winner: Selection
    ) -> tuple[CompiledJob, ...]:
        self._require_resolved(initial_winner)
        source = initial_winner.compiled
        if source.approved.stage == "lr_boundary":
            raise ValueError("another LR boundary win requires new approval")
        builder, surface = self._boundary_family(source)
        parameters = dict(source.parameters)
        boundaries: list[CompiledJob] = []
        for learning_rate, bounds in (
            ("embedding_learning_rate", EMBEDDING_LR_BOUNDS),
            ("deep_learning_rate", DEEP_LR_BOUNDS),
        ):
            side = boundary_side(float(parameters[learning_rate]), bounds)
            if side is None:
                continue
            slots = [
                job
                for job in approved_manifest().jobs_for_stage("lr_boundary")
                if job.method == surface
                and job.forced_parameters["learning_rate"] == learning_rate
            ]
            for job, rate in zip(
                slots,
                outside_boundary_rates(bounds, side),
                strict=True,
            ):
                resolved = dict(parameters)
                resolved |= {
                    "builder": builder,
                    "source_job_id": source.approved.id,
                    "source_parameters": dict(parameters),
                    "boundary_side": side,
                    learning_rate: rate,
                }
                candidate = CompiledJob(job, resolved)
                validate_compiled_job(candidate)
                boundaries.append(candidate)
        return tuple(boundaries)

    @staticmethod
    def _suggest_rates(trial: optuna.Trial) -> dict[str, float]:
        return {
            "embedding_learning_rate": trial.suggest_float(
                "embedding_learning_rate", *EMBEDDING_LR_BOUNDS, log=True
            ),
            "deep_learning_rate": trial.suggest_float(
                "deep_learning_rate", *DEEP_LR_BOUNDS, log=True
            ),
        }

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
        active = [trial for trial in trials if trial.state == TrialState.RUNNING]
        if active:
            return self._compiled_from_trial(active[0], jobs[active[0].number])
        completed = [trial for trial in trials if trial.state == TrialState.COMPLETE]
        if len(completed) == len(jobs):
            return None
        job = jobs[len(completed)]
        if job.forced_parameters:
            study.enqueue_trial(job.forced_parameters, skip_if_exists=True)
        study = optuna.load_study(
            storage=self._storage,
            study_name=study.study_name,
            sampler=optuna.samplers.TPESampler(
                seed=self._study_seed(f"{study.study_name}:{len(completed)}")
            ),
        )
        trial = study.ask()
        if trial.number != len(completed):
            raise RuntimeError("Optuna trial number diverged from the approved slot")
        parameters = suggest(trial)
        parameters.update(job.forced_parameters)
        compiled = CompiledJob(job, parameters)
        validate_compiled_job(compiled)
        trial.set_user_attr("job_id", job.id)
        trial.set_user_attr("compiled_parameters", parameters)
        return compiled

    def _create_study(
        self,
        stage: TuningStage,
        method: str,
        fixed_parameters: dict[str, Any],
    ) -> optuna.Study:
        name = f"g6-rq0-{stage}-{method}"
        study = optuna.create_study(
            storage=self._storage,
            study_name=name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self._study_seed(name)),
            load_if_exists=True,
        )
        expected = {
            "manifest_sha256": approved_manifest().sha256,
            "driver_seed": self._seed,
            "feasible_training_batches": list(self._feasible_training_batches),
            "validation_batch_size": self._validation_batch_size,
            "fixed_parameters": fixed_parameters,
        }
        for key, value in expected.items():
            prior = study.user_attrs.get(key)
            if prior is not None and prior != value:
                label = "fixed parameters" if key == "fixed_parameters" else key
                raise ValueError(f"{label} changed for {name}")
            study.set_user_attr(key, value)
        return study

    def _load_study(self, stage: TuningStage, method: str) -> optuna.Study:
        name = f"g6-rq0-{stage}-{method}"
        return optuna.load_study(storage=self._storage, study_name=name)

    def _study_seed(self, name: str) -> int:
        digest = hashlib.sha256(f"{self._seed}:{name}".encode()).digest()
        return int.from_bytes(digest[:4], "big")

    @staticmethod
    def _validate_trial_sequence(
        trials: list[optuna.trial.FrozenTrial], jobs: list[ApprovedJob]
    ) -> None:
        if len(trials) > len(jobs):
            raise ValueError("study exceeds the approved job slots")
        for trial in trials:
            if trial.number >= len(jobs):
                raise ValueError("study trial has no approved job slot")
            if trial.state not in {TrialState.RUNNING, TrialState.COMPLETE}:
                raise ValueError("study contains a non-observed terminal trial")
            job_id = trial.user_attrs.get("job_id")
            if job_id is not None and job_id != jobs[trial.number].id:
                raise ValueError("Optuna trial job identity changed")

    @staticmethod
    def _compiled_from_trial(
        trial: optuna.trial.FrozenTrial, job: ApprovedJob
    ) -> CompiledJob:
        parameters = trial.user_attrs.get("compiled_parameters")
        if trial.user_attrs.get("job_id") != job.id or not isinstance(parameters, dict):
            raise ValueError("running Optuna trial is missing its compiled contract")
        compiled = CompiledJob(job, parameters)
        validate_compiled_job(compiled)
        return compiled

    @staticmethod
    def _trial(study: optuna.Study, trial_number: int) -> optuna.trial.FrozenTrial:
        matches = [
            trial
            for trial in study.get_trials(deepcopy=False)
            if trial.number == trial_number
        ]
        if len(matches) != 1:
            raise ValueError(f"study has no unique trial {trial_number}")
        return matches[0]

    @staticmethod
    def _require_trial_identity(
        trial: optuna.trial.FrozenTrial, compiled: CompiledJob
    ) -> None:
        if trial.user_attrs.get("job_id") != compiled.approved.id:
            raise ValueError("Optuna trial job identity changed")
        if trial.user_attrs.get("compiled_parameters") != compiled.parameters:
            raise ValueError("Optuna trial parameters changed")

    @staticmethod
    def _record(study_name: str, trial: optuna.trial.FrozenTrial) -> TrialRecord:
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
            job_id=str(trial.user_attrs["job_id"]),
            trial_number=trial.number,
            parameters=parameters,
            objective=float(trial.value),
            artifact=Path(artifact),
        )

    @staticmethod
    def _require_resolved(selection: Selection) -> None:
        if not isinstance(selection, Selection):
            raise TypeError("prerequisite must be a Selection")
        validate_compiled_job(selection.compiled)
        if selection.selection_resolved is not True:
            raise ValueError("prerequisite is not selection-resolved")
        if not math.isfinite(selection.objective) or selection.objective < 0:
            raise ValueError("selection objective must be finite and non-negative")

    def _require_primary(self, selection: Selection) -> None:
        self._require_resolved(selection)
        job = selection.compiled.approved
        if job.stage == "primary_control_tuning":
            return
        if job.stage == "lr_boundary" and job.method == "primary_control":
            return
        raise ValueError("prerequisite is not a primary control selection")

    def _require_original(self, selection: Selection) -> None:
        self._require_resolved(selection)
        job = selection.compiled.approved
        if job.stage == "original_control_tuning":
            return
        if job.stage == "lr_boundary" and job.method == "original_control":
            return
        raise ValueError("prerequisite is not an original control selection")

    def _require_treatment(self, selection: Selection) -> None:
        self._require_resolved(selection)
        job = selection.compiled.approved
        representation = selection.compiled.parameters.get("representation")
        if job.stage == "treatment_tuning" and job.method == representation:
            return
        if job.stage == "lr_boundary" and job.method == representation:
            return
        raise ValueError("prerequisite is not a treatment selection")

    @staticmethod
    def _base_parameters(compiled: CompiledJob) -> dict[str, Any]:
        return {
            name: compiled.parameters[name]
            for name in (
                "batch_size",
                "validation_batch_size",
                "embedding_learning_rate",
                "deep_learning_rate",
            )
        }

    @staticmethod
    def _boundary_family(compiled: CompiledJob) -> tuple[str, str]:
        job = compiled.approved
        if job.stage == "primary_control_tuning":
            return "primary_control", "primary_control"
        if job.stage == "original_control_tuning":
            return "original_control", "original_control"
        if job.stage == "treatment_tuning":
            return "treatment", job.method
        if job.stage == "bridge_tuning":
            return "bridge", "bridge"
        raise ValueError(f"{job.stage!r} is not an initial tuning surface")


OptunaDriver = G6Rq0OptunaDriver
