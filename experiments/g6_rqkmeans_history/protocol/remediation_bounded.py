from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.g6_rqkmeans_history.protocol.manifest import ApprovedJob, CompiledJob


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SELECTION_PATH = (
    PROJECT_ROOT / "experiments/g6_rqkmeans_history/evidence/"
    "rq0_remediation_v3_selection_native50m.json"
)
SOURCE_SELECTION_SHA256 = (
    "09b4ece7d5535f0ffc45a0b72e2d3b9993e5c44660139e0c3da1df516cd4fce4"
)
SOURCE_MANIFEST_SHA256 = (
    "0cbda03a524c55bd2630f68467a55bbb4c5424faa5ef50862cd64c34f3a3f493"
)
SOURCE_TREATMENT_JOB_ID = "remediation_tuning:learned_sid_residual_trial_02"
SOURCE_TREATMENT_RUN_NAME = (
    "g6_rq0_remediation_v3_learned_sid_residual_trial_02_native50m"
)
BOUNDED_GATE_SCALES = (0.0, 0.01, 0.025, 0.05, 0.1)
_FIXED_PARAMETERS = {
    "batch_size": 256,
    "validation_batch_size": 8192,
    "representation": "item_frozen_sid_learned_residual_event",
    "representation_width": 32,
    "num_levels": 3,
    "num_codes": 512,
    "frozen_event_width": 128,
    "embedding_learning_rate": 0.256,
    "deep_learning_rate": 0.03463626154088337,
    "source_selection_sha256": SOURCE_SELECTION_SHA256,
    "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
    "source_treatment_job_id": SOURCE_TREATMENT_JOB_ID,
    "source_treatment_run_name": SOURCE_TREATMENT_RUN_NAME,
}


@dataclass(frozen=True)
class BoundedGateManifest:
    jobs: tuple[ApprovedJob, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dataset_size": "native-50m",
            "prior_physical_runs": 20,
            "new_physical_runs": 5,
            "maximum_total_physical_runs": 44,
            "source_selection_sha256": SOURCE_SELECTION_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "grid": {"learned_residual_max_scale": list(BOUNDED_GATE_SCALES)},
            "jobs": [job.to_dict() for job in self.jobs],
        }

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def _scale_slug(scale: float) -> str:
    return f"{round(scale * 1000):03d}"


_MANIFEST = BoundedGateManifest(
    tuple(
        ApprovedJob(
            id=f"bounded_gate_tuning:max_scale_{_scale_slug(scale)}",
            run_name=(
                "g6_rq0_remediation_v4_bounded_gate_" f"{_scale_slug(scale)}_native50m"
            ),
            stage="bounded_gate_tuning",
            method="item_frozen_sid_learned_residual_event",
            seed=42,
            trial=trial,
            conditional=False,
            forced_parameters={"learned_residual_max_scale": scale},
        )
        for trial, scale in enumerate(BOUNDED_GATE_SCALES)
    )
)


def bounded_gate_manifest() -> BoundedGateManifest:
    if len(_MANIFEST.jobs) != 5:
        raise RuntimeError("bounded-gate run budget changed")
    if len({job.id for job in _MANIFEST.jobs}) != 5:
        raise RuntimeError("bounded-gate job IDs are not unique")
    if len({job.run_name for job in _MANIFEST.jobs}) != 5:
        raise RuntimeError("bounded-gate run names are not unique")
    return _MANIFEST


def bounded_gate_jobs() -> tuple[CompiledJob, ...]:
    return tuple(
        CompiledJob(
            job,
            _FIXED_PARAMETERS
            | {
                "learned_residual_max_scale": job.forced_parameters[
                    "learned_residual_max_scale"
                ]
            },
        )
        for job in bounded_gate_manifest().jobs
    )


def validate_bounded_gate_job(compiled: CompiledJob) -> None:
    if compiled not in bounded_gate_jobs():
        raise ValueError("bounded-gate parameters changed")
    if compiled.attempt != 0 or compiled.cap_epochs is not None:
        raise ValueError("bounded-gate grid does not permit continuations")
    scale = compiled.parameters["learned_residual_max_scale"]
    if (
        not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not math.isfinite(scale)
        or scale not in BOUNDED_GATE_SCALES
    ):
        raise ValueError("bounded-gate scale is invalid")


def load_bounded_gate_source(path: Path = SOURCE_SELECTION_PATH) -> dict[str, Any]:
    try:
        content = path.read_bytes()
        document = json.loads(content)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read bounded-gate source selection {path}") from error
    if hashlib.sha256(content).hexdigest() != SOURCE_SELECTION_SHA256:
        raise ValueError("bounded-gate source selection changed")
    if document.get("manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise ValueError("bounded-gate source manifest changed")
    if document.get("run_counts", {}).get("total_including_carryovers") != 20:
        raise ValueError("bounded-gate source run count changed")
    winner = document.get("treatment_winner")
    if not isinstance(winner, dict):
        raise ValueError("bounded-gate source winner is absent")
    if winner.get("job_id") != SOURCE_TREATMENT_JOB_ID:
        raise ValueError("bounded-gate source winner changed")
    if winner.get("run_name") != SOURCE_TREATMENT_RUN_NAME:
        raise ValueError("bounded-gate source winner run changed")
    return document
