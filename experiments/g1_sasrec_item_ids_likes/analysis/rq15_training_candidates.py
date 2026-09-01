from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
import json
import math
from pathlib import Path
import re
from typing import Literal


Rq15TrainingMethod = Literal[
    "scratch_candidate_only", "pretrained_finetune", "auxiliary_ntp"
]
Rq15Stage = Literal["initial", "lr_boundary", "auxiliary_weight"]
Rq15BoundaryAxis = Literal["embedding", "deep"]

RQ15_SOURCE_RECIPE_RUN = (
    "g1_rq8_query_standard_s128_e0p064_d0p048_b1280_seed42_cap20_"
    "ts2_boundaryhigh1_r1_500m"
)
RQ15_SOURCE_EXPORT_RUN = (
    "g1_rq15_rq8_standard_ntp_checkpoint_s128_e0p064_d0p048_"
    "b1280_seed42_h20_r1_500m"
)
RQ15_SOURCE_CHECKPOINT_NAME = "rq15_first_stage_checkpoint.pt"

EMBEDDING_LRS = (0.032, 0.064, 0.128)
CANDIDATE_ONLY_DEEP_LRS = (0.00075, 0.0015, 0.003)
AUXILIARY_DEEP_LRS = (0.003, 0.012, 0.048)
SOURCE_DEEP_LRS = (0.024, 0.048, 0.096)
SCRATCH_REUSED_CENTER = (0.064, 0.0015)
DEEP_LR_BOUNDARY_RATIOS: dict[Rq15TrainingMethod, int] = {
    "scratch_candidate_only": 2,
    "pretrained_finetune": 2,
    "auxiliary_ntp": 4,
}
PRETRAINED_FROZEN_EMBEDDING_STEP = 8


@dataclass(frozen=True)
class Rq15SourceCandidate:
    run_name: str
    source_recipe_run_name: str
    embedding_lr: float
    deep_lr: float
    checkpoint_name: str = RQ15_SOURCE_CHECKPOINT_NAME

    def checkpoint_path(self, logs: Path) -> Path:
        return logs / self.run_name / self.checkpoint_name


@dataclass(frozen=True)
class Rq15Candidate:
    training_method: Rq15TrainingMethod
    embedding_lr: float
    deep_lr: float
    dataset_size: str = "500m"
    batch_size: int = 1280
    seed: int = 42
    horizon_epochs: int = 20
    stage: Rq15Stage = "initial"
    boundary_axis: Rq15BoundaryAxis | None = None
    boundary_direction: Literal["low", "high"] | None = None
    boundary_step: int | None = None
    auxiliary_ntp_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.training_method not in {
            "scratch_candidate_only",
            "pretrained_finetune",
            "auxiliary_ntp",
        }:
            raise ValueError("unknown RQ15 training method")
        for name, value in (
            ("embedding_lr", self.embedding_lr),
            ("deep_lr", self.deep_lr),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.deep_lr == 0:
            raise ValueError("deep_lr must be positive")
        if self.dataset_size != "500m" or self.batch_size != 1280:
            raise ValueError("RQ15 uses native Yambda-500M and batch 1280")
        if self.seed != 42 or self.horizon_epochs != 20:
            raise ValueError("RQ15 uses seed 42 and a 20-epoch horizon")
        if self.stage not in {"initial", "lr_boundary", "auxiliary_weight"}:
            raise ValueError("unknown RQ15 candidate stage")
        if self.stage == "initial":
            if any(
                value is not None
                for value in (
                    self.boundary_axis,
                    self.boundary_direction,
                    self.boundary_step,
                )
            ):
                raise ValueError("initial RQ15 candidate cannot carry a boundary")
            if self.auxiliary_ntp_weight != 1.0:
                raise ValueError("initial RQ15 candidate uses auxiliary weight 1.0")
        elif self.stage == "lr_boundary":
            if self.boundary_axis not in {"embedding", "deep"}:
                raise ValueError("RQ15 boundary axis is required")
            if self.boundary_direction not in {"low", "high"}:
                raise ValueError("RQ15 boundary direction is required")
            if (
                not isinstance(self.boundary_step, int)
                or isinstance(self.boundary_step, bool)
                or self.boundary_step < 1
            ):
                raise ValueError("RQ15 boundary step must be positive")
            if self.auxiliary_ntp_weight != 1.0:
                raise ValueError("RQ15 LR boundary keeps auxiliary weight 1.0")
        else:
            if self.training_method != "auxiliary_ntp":
                raise ValueError("auxiliary-weight follow-up requires auxiliary NTP")
            if any(
                value is not None
                for value in (
                    self.boundary_axis,
                    self.boundary_direction,
                    self.boundary_step,
                )
            ):
                raise ValueError("auxiliary-weight follow-up is not an LR boundary")
            if self.auxiliary_ntp_weight not in {0.1, 0.3}:
                raise ValueError("RQ15 auxiliary-weight follow-up must use 0.1 or 0.3")
        if self.embedding_lr == 0 and (
            math.copysign(1.0, float(self.embedding_lr)) < 0
            or not self._is_frozen_embedding_candidate()
        ):
            raise ValueError(
                "zero embedding LR is only valid at the pretrained terminal boundary"
            )

    def _is_frozen_embedding_candidate(self) -> bool:
        if self.training_method != "pretrained_finetune" or self.stage != "lr_boundary":
            return False
        if self.boundary_axis == "embedding":
            return (
                self.boundary_direction == "low"
                and self.boundary_step == PRETRAINED_FROZEN_EMBEDDING_STEP
                and self.deep_lr in CANDIDATE_ONLY_DEEP_LRS
            )
        if self.boundary_axis != "deep" or self.boundary_direction not in {
            "low",
            "high",
        }:
            return False
        assert self.boundary_step is not None
        initial = (
            CANDIDATE_ONLY_DEEP_LRS[0]
            if self.boundary_direction == "low"
            else CANDIDATE_ONLY_DEEP_LRS[-1]
        )
        factor = 2 ** (
            -self.boundary_step
            if self.boundary_direction == "low"
            else self.boundary_step
        )
        return self.deep_lr == initial * factor

    @property
    def run_name(self) -> str:
        prefix = (
            f"g1_rq15_{self.training_method}_e{_slug(self.embedding_lr)}_"
            f"d{_slug(self.deep_lr)}_b{self.batch_size}_seed{self.seed}_"
            f"h{self.horizon_epochs}"
        )
        if self.stage == "initial":
            suffix = ""
        elif self.stage == "lr_boundary":
            suffix = (
                f"_lr{self.boundary_axis}{self.boundary_direction}"
                f"{self.boundary_step}"
            )
        else:
            suffix = f"_auxw{_slug(self.auxiliary_ntp_weight)}"
        return f"{prefix}{suffix}_r1_{self.dataset_size}"

    def environment(self) -> dict[str, str]:
        return {"G1_RQ15_RUN": self.run_name}


@cache
def initial_candidates() -> tuple[Rq15Candidate, ...]:
    candidates = tuple(
        Rq15Candidate(method, embedding_lr=embedding_lr, deep_lr=deep_lr)
        for method, deep_lrs in (
            ("scratch_candidate_only", CANDIDATE_ONLY_DEEP_LRS),
            ("pretrained_finetune", CANDIDATE_ONLY_DEEP_LRS),
            ("auxiliary_ntp", AUXILIARY_DEEP_LRS),
        )
        for embedding_lr in EMBEDDING_LRS
        for deep_lr in deep_lrs
    )
    if len(candidates) != 27 or len({item.run_name for item in candidates}) != 27:
        raise RuntimeError("RQ15 initial manifest must contain 27 unique runs")
    return candidates


@cache
def launch_initial_candidates() -> tuple[Rq15Candidate, ...]:
    candidates = tuple(
        candidate
        for candidate in initial_candidates()
        if not (
            candidate.training_method == "scratch_candidate_only"
            and (candidate.embedding_lr, candidate.deep_lr) == SCRATCH_REUSED_CENTER
        )
    )
    if len(candidates) != 26:
        raise RuntimeError("RQ15 launch manifest must reuse exactly one scratch cell")
    return candidates


@cache
def source_candidates() -> tuple[Rq15SourceCandidate, ...]:
    recipe_runs = {
        0.024: (
            "g1_rq8_query_standard_s128_e0p064_d0p024_b1280_seed42_cap20_"
            "ts2_r1_500m"
        ),
        0.048: RQ15_SOURCE_RECIPE_RUN,
        0.096: (
            "g1_rq8_query_standard_s128_e0p064_d0p096_b1280_seed42_cap20_"
            "ts2_boundaryhigh2_r1_500m"
        ),
    }
    candidates = tuple(
        Rq15SourceCandidate(
            run_name=(
                "g1_rq15_rq8_standard_ntp_checkpoint_s128_e0p064_"
                f"d{_slug(deep_lr)}_b1280_seed42_h20_r1_500m"
            ),
            source_recipe_run_name=recipe_runs[deep_lr],
            embedding_lr=0.064,
            deep_lr=deep_lr,
        )
        for deep_lr in SOURCE_DEEP_LRS
    )
    if len({candidate.run_name for candidate in candidates}) != 3:
        raise RuntimeError("RQ15 source manifest must contain three unique runs")
    return candidates


def source_candidate_by_run(run_name: str) -> Rq15SourceCandidate:
    matches = [
        candidate for candidate in source_candidates() if candidate.run_name == run_name
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown RQ15 source run {run_name!r}")
    return matches[0]


def source_checkpoint_metadata(
    candidate: Rq15SourceCandidate | None = None,
) -> dict[str, object]:
    if candidate is None:
        candidate = source_candidate_by_run(RQ15_SOURCE_EXPORT_RUN)
    return {
        "dataset_size": "500m",
        "source_recipe_run_name": candidate.source_recipe_run_name,
        "training_objective": "standard_next_item_prediction",
        "max_seq_len": 128,
        "embedding_learning_rate": candidate.embedding_lr,
        "deep_learning_rate": candidate.deep_lr,
        "batch_size": 1280,
        "seed": 42,
        "horizon_epochs": 20,
    }


def selected_source_candidate(logs: Path) -> Rq15SourceCandidate:
    ranked: list[tuple[tuple[float, float, float], Rq15SourceCandidate]] = []
    for candidate in source_candidates():
        directory = logs / candidate.run_name
        checkpoint = candidate.checkpoint_path(logs)
        required = (directory / "training_metadata.json", directory / "sweep.log", checkpoint)
        required = (*required, directory / "final_metrics.json")
        if not all(path.is_file() for path in required):
            raise ValueError(f"missing complete RQ15 source artifact {candidate.run_name}")
        metadata = json.loads((directory / "training_metadata.json").read_text())
        final_metrics = json.loads((directory / "final_metrics.json").read_text())
        if (
            metadata.get("embedding_learning_rate") != candidate.embedding_lr
            or metadata.get("deep_learning_rate") != candidate.deep_lr
            or metadata.get("best_epoch") not in range(1, 21)
            or metadata.get("stopped_epoch") != 20
            or metadata.get("lr_horizon_complete") is not True
            or metadata.get("selection_resolved") is not True
            or not all(
                isinstance(final_metrics.get(metric), (int, float))
                and math.isfinite(final_metrics[metric])
                for metric in ("recall@100", "ndcg@100")
            )
        ):
            raise ValueError(f"incompatible RQ15 source artifact {candidate.run_name}")
        curve = _source_validation_curve(directory / "sweep.log")
        best_epoch = int(metadata["best_epoch"])
        epoch, recall, ndcg, _ = curve[best_epoch - 1]
        if epoch != best_epoch or best_epoch != _best_source_epoch(curve):
            raise ValueError(f"invalid RQ15 source selection {candidate.run_name}")
        ranked.append(((recall, ndcg, -curve[-1][3]), candidate))
    return max(ranked, key=lambda item: item[0])[1]


def selected_source_checkpoint(logs: Path) -> Path:
    return selected_source_candidate(logs).checkpoint_path(logs)


def candidate_by_run(run_name: str) -> Rq15Candidate:
    matches = [item for item in initial_candidates() if item.run_name == run_name]
    if len(matches) == 1:
        return matches[0]
    match = _RUN_PATTERN.fullmatch(run_name)
    if match is None:
        raise ValueError(f"unknown RQ15 candidate run {run_name!r}")
    stage_token = match.group("stage")
    values: dict[str, object] = {
        "training_method": match.group("method"),
        "embedding_lr": _unslug(match.group("embedding")),
        "deep_lr": _unslug(match.group("deep")),
    }
    if stage_token.startswith("lr"):
        direction_step = re.fullmatch(
            r"lr(embedding|deep)(low|high)(\d+)", stage_token
        )
        assert direction_step is not None
        values.update(
            stage="lr_boundary",
            boundary_axis=direction_step.group(1),
            boundary_direction=direction_step.group(2),
            boundary_step=int(direction_step.group(3)),
        )
    else:
        values.update(
            stage="auxiliary_weight",
            auxiliary_ntp_weight=_unslug(stage_token.removeprefix("auxw")),
        )
    candidate = Rq15Candidate(**values)
    if candidate.run_name != run_name:
        raise ValueError(f"noncanonical RQ15 candidate run {run_name!r}")
    return candidate


def make_boundary_candidate(
    anchor: Rq15Candidate,
    axis: Rq15BoundaryAxis,
    direction: Literal["low", "high"],
    step: int,
) -> Rq15Candidate:
    if (
        anchor.stage != "initial"
        or axis not in {"embedding", "deep"}
        or direction not in {"low", "high"}
        or not isinstance(step, int)
        or isinstance(step, bool)
        or step < 1
    ):
        raise ValueError("RQ15 boundary requires an initial anchor and positive step")
    ratio = (
        DEEP_LR_BOUNDARY_RATIOS[anchor.training_method]
        if axis == "deep"
        else 2
    )
    if (
        axis == "embedding"
        and direction == "low"
        and anchor.training_method == "pretrained_finetune"
    ):
        if step > PRETRAINED_FROZEN_EMBEDDING_STEP:
            raise ValueError("pretrained embedding boundary is terminal at zero")
        if step == PRETRAINED_FROZEN_EMBEDDING_STEP:
            if anchor.embedding_lr != EMBEDDING_LRS[0]:
                raise ValueError(
                    "frozen embedding boundary requires the pretrained low-edge anchor"
                )
            return replace(
                anchor,
                embedding_lr=0.0,
                stage="lr_boundary",
                boundary_axis=axis,
                boundary_direction=direction,
                boundary_step=step,
            )
    factor = ratio**step if direction == "high" else ratio ** (-step)
    rate_update = (
        {"embedding_lr": anchor.embedding_lr * factor}
        if axis == "embedding"
        else {"deep_lr": anchor.deep_lr * factor}
    )
    return replace(
        anchor,
        **rate_update,
        stage="lr_boundary",
        boundary_axis=axis,
        boundary_direction=direction,
        boundary_step=step,
    )


def make_auxiliary_weight_candidate(
    anchor: Rq15Candidate, weight: Literal[0.1, 0.3]
) -> Rq15Candidate:
    if anchor.training_method != "auxiliary_ntp" or anchor.auxiliary_ntp_weight != 1.0:
        raise ValueError("RQ15 auxiliary weight must follow the weight-1 surface")
    return replace(
        anchor,
        stage="auxiliary_weight",
        boundary_axis=None,
        boundary_direction=None,
        boundary_step=None,
        auxiliary_ntp_weight=weight,
    )


def candidate_followup_record(candidate: Rq15Candidate) -> dict[str, object]:
    if candidate.stage == "initial":
        raise ValueError("initial RQ15 candidate is not a follow-up")
    return {
        "run_name": candidate.run_name,
        "training_method": candidate.training_method,
        "stage": candidate.stage,
        "embedding_lr": candidate.embedding_lr,
        "deep_lr": candidate.deep_lr,
        "boundary_axis": candidate.boundary_axis,
        "boundary_direction": candidate.boundary_direction,
        "boundary_step": candidate.boundary_step,
        "auxiliary_ntp_weight": candidate.auxiliary_ntp_weight,
    }


def validated_required_followup_candidates(
    evidence: object,
    *,
    authoritative_evidence: object,
) -> tuple[Rq15Candidate, ...]:
    if evidence != authoritative_evidence:
        raise ValueError(
            "RQ15 evidence is stale or differs from the artifact-derived report"
        )
    if not isinstance(evidence, dict):
        raise ValueError("RQ15 result evidence must be an object")
    expected_stage = {
        "pending_boundary": "lr_boundary",
        "pending_auxiliary_weights": "auxiliary_weight",
    }.get(evidence.get("claims_status"))
    if (
        evidence.get("schema_version") != 1
        or evidence.get("research_question")
        != "RQ15 decoder-decoder training method"
        or evidence.get("dataset_size") != "500m"
        or evidence.get("result_claims_user_validated") is not False
        or evidence.get("missing_artifacts") != []
        or not isinstance(evidence.get("scratch_control"), dict)
        or not isinstance(evidence.get("checkpoint_pretraining"), dict)
        or not isinstance(evidence.get("treatments"), dict)
        or not isinstance(evidence.get("artifact_audit"), dict)
        or evidence["artifact_audit"].get("status") != "passed"
        or expected_stage is None
    ):
        raise ValueError("RQ15 evidence is foreign, stale, or not a follow-up stage")
    required = evidence.get("required_followups")
    if not isinstance(required, list) or not required:
        raise ValueError("RQ15 follow-up evidence is empty")
    candidates = []
    for record in required:
        if not isinstance(record, dict) or not isinstance(record.get("run_name"), str):
            raise ValueError("RQ15 follow-up record is malformed")
        candidate = candidate_by_run(record["run_name"])
        if candidate.stage != expected_stage:
            raise ValueError("RQ15 follow-up stage contradicts claims status")
        if record != candidate_followup_record(candidate):
            raise ValueError("RQ15 follow-up record is stale or manually altered")
        candidates.append(candidate)
    if len({candidate.run_name for candidate in candidates}) != len(candidates):
        raise ValueError("RQ15 follow-up evidence repeats a candidate")
    if expected_stage == "auxiliary_weight" and {
        candidate.training_method for candidate in candidates
    } != {"auxiliary_ntp"}:
        raise ValueError("RQ15 auxiliary follow-up contains a foreign method")
    return tuple(candidates)


def _source_validation_curve(path: Path) -> tuple[tuple[int, float, float, float], ...]:
    values: dict[int, tuple[float, float, float]] = {}
    elapsed = 0.0
    for line in path.read_text().splitlines():
        timing = re.search(
            r"\bepoch (\d+) finished\b.*?"
            r"timing\.train_epoch_time=([0-9.eE+-]+).*?"
            r"timing\.val_inference_time=([0-9.eE+-]+).*?"
            r"timing\.val_save_time=([0-9.eE+-]+)",
            line,
        )
        if timing is not None:
            elapsed += sum(float(timing.group(index)) for index in range(2, 5))
        epoch = re.search(r"\bepoch (\d+) finished\b", line)
        recall = re.search(r"\bepoch/val_true\.recall@100=([0-9.eE+-]+)\b", line)
        ndcg = re.search(r"\bepoch/val_true\.ndcg@100=([0-9.eE+-]+)\b", line)
        if epoch is not None and recall is not None and ndcg is not None:
            number = int(epoch.group(1)) + 1
            values[number] = (float(recall.group(1)), float(ndcg.group(1)), elapsed)
    if sorted(values) != list(range(1, 21)):
        raise ValueError(f"incomplete RQ15 source curve {path.parent.name}")
    return tuple((epoch, *values[epoch]) for epoch in range(1, 21))


def _best_source_epoch(
    curve: tuple[tuple[int, float, float, float], ...],
) -> int:
    return max(curve, key=lambda point: (point[1], -point[0]))[0]


def _slug(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _unslug(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


_RUN_PATTERN = re.compile(
    r"g1_rq15_(?P<method>scratch_candidate_only|pretrained_finetune|auxiliary_ntp)_"
    r"e(?P<embedding>[0-9mp]+)_d(?P<deep>[0-9mp]+)_b1280_seed42_h20_"
    r"(?P<stage>lr(?:embedding|deep)(?:low|high)\d+|auxw[0-9mp]+)_r1_500m"
)
