from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_candidates import (
    EMBEDDING_LR,
    TREATMENTS,
    Treatment,
)


@dataclass(frozen=True)
class Rq14LesionDiagnosticCandidate:
    treatment: Treatment
    dataset_size: Literal["500m"] = "500m"
    embedding_lr: float = EMBEDDING_LR
    deep_lr: float = 0.00075
    batch_size: int = 1280
    seed: int = 42
    horizon_epochs: int = 20

    def __post_init__(self) -> None:
        if self.treatment not in TREATMENTS:
            raise ValueError("unknown RQ14 lesion treatment")
        if (
            self.dataset_size != "500m"
            or self.embedding_lr != 0.00025
            or self.deep_lr != 0.00075
            or self.batch_size != 1280
            or self.seed != 42
            or self.horizon_epochs != 20
        ):
            raise ValueError("RQ14 lesions use the exact selected-cell protocol")

    @property
    def query_slots_shared(self) -> bool:
        return self.treatment.startswith("shared_")

    @property
    def include_history_memory(self) -> bool:
        return self.treatment.endswith("_history")

    @property
    def lesions(self) -> tuple[str, ...]:
        return tuple(["remove_history"] if self.include_history_memory else []) + tuple(
            f"drop_cls_{slot}" for slot in range(4)
        )

    @property
    def run_name(self) -> str:
        return (
            f"g1_rq14_pretrained_{self.treatment}_e0p00025_d0p00075_"
            "b1280_seed42_h20_lesions_r1_500m"
        )


@cache
def diagnostic_candidates() -> tuple[Rq14LesionDiagnosticCandidate, ...]:
    candidates = tuple(
        Rq14LesionDiagnosticCandidate(treatment) for treatment in TREATMENTS
    )
    if len(candidates) != 4 or len({item.run_name for item in candidates}) != 4:
        raise RuntimeError(
            "RQ14 lesion manifest must contain four unique selected cells"
        )
    return candidates


def diagnostic_candidate_by_run(run_name: str) -> Rq14LesionDiagnosticCandidate:
    matches = [
        candidate
        for candidate in diagnostic_candidates()
        if candidate.run_name == run_name
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown RQ14 lesion run {run_name!r}")
    return matches[0]
