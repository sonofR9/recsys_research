from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "native50m_fixed26_calibration_manifest.json"
)
CALIBRATION_METRICS = (
    "recall@10",
    "recall@50",
    "recall@100",
    "ndcg@10",
    "ndcg@50",
    "ndcg@100",
    "mrr@10",
    "mrr@50",
    "mrr@100",
    "capped_recall@10",
    "capped_recall@50",
    "capped_recall@100",
    "coverage@10",
    "coverage@50",
    "coverage@100",
)
_SOURCE_RUN = (
    "g1_aggregate_dataset_size_baseline_none_l2_b512_s42_"
    "e0p003261002414691765_d0p025343654763668278_hnone_"
    "batch_lr_calibration_ts2_r1_50m"
)
_SOURCE_HASHES = {
    "training_metadata.json": (
        "fe330a0e00141787de45faa60b9bfa8afb14fcf796279ba8283634d5ba9b39fd"
    ),
    "final_metrics.json": (
        "6fbafdb72a3d2e143be5de69ec90f983e78b7253dc26a3bf7c1881846d1e1406"
    ),
    "sweep.log": (
        "10ea037da81b9463bcc2618ed6b7d81009a8c3f49bf0f741a5cde0f6b1b8fdd2"
    ),
}


@dataclass(frozen=True)
class Fixed26CalibrationJob:
    id: str
    run_name: str
    seed: int

    def __post_init__(self) -> None:
        expected_id = f"native50m_shared_fixed26_calibration:seed_{self.seed}"
        expected_run = (
            f"g1_shared_calibration_fixed26_mup_control_seed_{self.seed}_native50m"
        )
        if self.seed not in range(42, 52):
            raise ValueError("fixed-26 calibration seed must be 42 through 51")
        if self.id != expected_id or self.run_name != expected_run:
            raise ValueError("fixed-26 calibration job identity changed")

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "run_name": self.run_name, "seed": self.seed}


@dataclass(frozen=True)
class Fixed26CalibrationManifest:
    path: Path
    sha256: str
    version: int
    stage: str
    dataset_size: str
    batch_size: int
    embedding_learning_rate: float
    deep_learning_rate: float
    num_epochs: int
    early_stopping: bool
    restore_best_weights: bool
    recipe_sha256: str
    experiment_fingerprint_sha256: str
    source_run_name: str
    source_artifact_sha256: Mapping[str, str]
    jobs: tuple[Fixed26CalibrationJob, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_artifact_sha256",
            MappingProxyType(dict(self.source_artifact_sha256)),
        )
        expected = (
            self.version == 1
            and self.stage == "native50m_shared_fixed26_calibration"
            and self.dataset_size == "50m"
            and self.batch_size == 512
            and self.embedding_learning_rate == 0.003261002414691765
            and self.deep_learning_rate == 0.025343654763668278
            and self.num_epochs == 26
            and self.early_stopping is False
            and self.restore_best_weights is True
            and len(self.recipe_sha256) == 64
            and set(self.recipe_sha256) <= set("0123456789abcdef")
            and len(self.experiment_fingerprint_sha256) == 64
            and set(self.experiment_fingerprint_sha256)
            <= set("0123456789abcdef")
            and self.source_run_name == _SOURCE_RUN
            and dict(self.source_artifact_sha256) == _SOURCE_HASHES
            and [job.seed for job in self.jobs] == list(range(42, 52))
        )
        if not expected:
            raise ValueError("fixed-26 calibration manifest contract changed")
        if len(self.sha256) != 64 or not set(self.sha256) <= set(
            "0123456789abcdef"
        ):
            raise ValueError("fixed-26 calibration manifest hash is invalid")

    def job_by_run(self, run_name: str) -> Fixed26CalibrationJob:
        matches = [job for job in self.jobs if job.run_name == run_name]
        if len(matches) != 1:
            raise ValueError(f"unknown fixed-26 calibration run {run_name!r}")
        return matches[0]


def load_fixed26_manifest(
    path: Path = DEFAULT_MANIFEST_PATH,
) -> Fixed26CalibrationManifest:
    raw = path.read_bytes()
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("fixed-26 calibration manifest is invalid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("fixed-26 calibration manifest must be an object")
    _validate_document(document)
    source = document["source_control"]
    jobs = tuple(Fixed26CalibrationJob(**row) for row in document["jobs"])
    return Fixed26CalibrationManifest(
        path=path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        version=document["version"],
        stage=document["stage"],
        dataset_size=document["dataset_size"],
        batch_size=document["batch_size"],
        embedding_learning_rate=document["embedding_learning_rate"],
        deep_learning_rate=document["deep_learning_rate"],
        num_epochs=document["num_epochs"],
        early_stopping=document["early_stopping"],
        restore_best_weights=document["restore_best_weights"],
        recipe_sha256=document["recipe_sha256"],
        experiment_fingerprint_sha256=document[
            "experiment_fingerprint_sha256"
        ],
        source_run_name=source["run_name"],
        source_artifact_sha256=source["artifact_sha256"],
        jobs=jobs,
    )


def _validate_document(document: dict[str, Any]) -> None:
    expected_keys = {
        "version",
        "stage",
        "dataset_size",
        "batch_size",
        "embedding_learning_rate",
        "deep_learning_rate",
        "num_epochs",
        "early_stopping",
        "restore_best_weights",
        "recipe_sha256",
        "experiment_fingerprint_sha256",
        "source_control",
        "jobs",
    }
    if set(document) != expected_keys:
        raise ValueError("fixed-26 calibration manifest fields changed")
    expected = {
        "version": 1,
        "stage": "native50m_shared_fixed26_calibration",
        "dataset_size": "50m",
        "batch_size": 512,
        "embedding_learning_rate": 0.003261002414691765,
        "deep_learning_rate": 0.025343654763668278,
        "num_epochs": 26,
        "early_stopping": False,
        "restore_best_weights": True,
    }
    if any(
        type(document[key]) is not type(value) or document[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("fixed-26 calibration protocol changed")
    source = document["source_control"]
    if (
        not isinstance(source, dict)
        or set(source) != {"run_name", "artifact_sha256"}
        or source["run_name"] != _SOURCE_RUN
        or source["artifact_sha256"] != _SOURCE_HASHES
    ):
        raise ValueError("fixed-26 source-control binding changed")
    rows = document["jobs"]
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("fixed-26 calibration requires ten jobs")
    for seed, row in zip(range(42, 52), rows, strict=True):
        expected_row = {
            "id": f"native50m_shared_fixed26_calibration:seed_{seed}",
            "run_name": (
                f"g1_shared_calibration_fixed26_mup_control_seed_{seed}_native50m"
            ),
            "seed": seed,
        }
        if not isinstance(row, dict) or row != expected_row:
            raise ValueError("fixed-26 calibration job identities changed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")
