from __future__ import annotations

from dataclasses import asdict
import base64
import hashlib
import json
import os
from pathlib import Path
from types import MethodType
from typing import Any, Mapping

from experiments.g6_rqkmeans_history.native500m.configs.runtime import (
    build_control,
    build_semantic_treatment,
)
from experiments.g6_rqkmeans_history.native500m.launchers.queue import (
    CONFIG_LOGICAL_SHA256_ENVIRONMENT,
    JOB_ENVIRONMENT,
    JOB_LOGICAL_SHA256_ENVIRONMENT,
    MANIFEST_LOGICAL_SHA256_ENVIRONMENT,
    MANIFEST_PATH_ENVIRONMENT,
    MANIFEST_PHYSICAL_SHA256_ENVIRONMENT,
    QueueJob,
    QueueManifest,
    canonical_bytes,
    load_queue_manifest,
    persist_immutable_bytes,
    source_identity_sha256,
)
from experiments.g6_rqkmeans_history.native500m.protocol.tokenizer_registry import (
    DEFAULT_REGISTRY_PATH,
    ENVIRONMENT_FIELDS as TOKENIZER_ENVIRONMENT_FIELDS,
    load_registry as load_tokenizer_registry,
    verify_binding as verify_tokenizer_binding,
)
from neuralrec.run.callbacks import Callback
from neuralrec.utils import EXTRA_METRICS, to_float
from experiments.g6_rqkmeans_history.native500m.protocol.contracts import JOB_SCHEMA


class ValidationHistoryCallback(Callback):
    def __init__(self, *, experiment: object, job: QueueJob) -> None:
        self.experiment = experiment
        self.job = job
        self.rows: list[dict[str, float | int]] = []

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        metrics = state.get(EXTRA_METRICS, {}).get("epoch/val_true", {})
        recall = to_float(metrics.get("recall@100"))
        ndcg = to_float(metrics.get("ndcg@100"))
        if recall is None or ndcg is None:
            raise RuntimeError("native-500M validation Recall/NDCG is missing")
        epoch = state["train_runner"].current_epoch + 1
        self.rows.append({"epoch": epoch, "recall@100": recall, "ndcg@100": ndcg})

    def on_train_end(self, state: dict[str, Any]) -> None:
        if [row["epoch"] for row in self.rows] != list(range(1, 27)):
            raise RuntimeError("native-500M validation history must cover epochs 1-26")
        best_epoch_index = self.experiment.callbacks.best_weights.best_epoch
        if best_epoch_index is None:
            raise RuntimeError("native-500M best validation epoch is unresolved")
        document = {
            "schema": "g6-native500m-validation-history/v1",
            "job_id": self.job.job_id,
            "job_logical_sha256": self.job.logical_sha256,
            "config_logical_sha256": self.job.config_logical_sha256,
            "selection_metric": "recall@100",
            "best_epoch": best_epoch_index + 1,
            "epochs": self.rows,
        }
        path = (
            Path(self.experiment.base_path)
            / "logs"
            / self.job.run_name
            / "validation_history.json"
        )
        _write_immutable_json(path, document)


def load_runtime_job(
    environment: Mapping[str, str] = os.environ,
) -> tuple[QueueManifest, QueueJob]:
    required = {
        JOB_ENVIRONMENT,
        MANIFEST_PATH_ENVIRONMENT,
        MANIFEST_LOGICAL_SHA256_ENVIRONMENT,
        MANIFEST_PHYSICAL_SHA256_ENVIRONMENT,
        JOB_LOGICAL_SHA256_ENVIRONMENT,
        CONFIG_LOGICAL_SHA256_ENVIRONMENT,
    }
    missing = required - environment.keys()
    if missing:
        raise RuntimeError(
            f"native-500M runtime environment is missing {sorted(missing)}"
        )
    manifest = load_queue_manifest(
        Path(environment[MANIFEST_PATH_ENVIRONMENT]), verify_current_source=True
    )
    if (
        environment[MANIFEST_LOGICAL_SHA256_ENVIRONMENT] != manifest.logical_sha256
        or environment[MANIFEST_PHYSICAL_SHA256_ENVIRONMENT] != manifest.physical_sha256
    ):
        raise RuntimeError("native-500M runtime manifest identity differs")
    try:
        payload = json.loads(
            base64.urlsafe_b64decode(environment[JOB_ENVIRONMENT]).decode()
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("native-500M runtime job payload is invalid") from error
    matches = [job for job in manifest.jobs if job.payload == payload]
    if len(matches) != 1:
        raise RuntimeError("native-500M runtime job is absent from its manifest")
    job = matches[0]
    if (
        environment[JOB_LOGICAL_SHA256_ENVIRONMENT] != job.logical_sha256
        or environment[CONFIG_LOGICAL_SHA256_ENVIRONMENT] != job.config_logical_sha256
    ):
        raise RuntimeError("native-500M runtime job identity differs")
    return manifest, job


def build_experiment(job: QueueJob) -> object:
    if job.payload.get("schema") != JOB_SCHEMA:
        raise RuntimeError("native-500M runtime requires a current job contract")
    expected_source = source_identity_sha256()
    supplied_source = job.environment.get("G6_NATIVE500M_SOURCE_SHA256")
    if supplied_source is not None and supplied_source != expected_source:
        raise RuntimeError("native-500M source identity differs")
    parameters = dict(job.payload["parameters"])
    for name in (
        "runner",
        "config_logical_sha256",
        "data_group",
        "environment",
    ):
        parameters.pop(name, None)
    parameters.pop("builder", None)
    representation = parameters.pop("representation", None)
    if representation is None:
        experiment = build_control(**parameters)
    else:
        levels = parameters.pop("levels")
        shared_codes = parameters.pop("shared_codes")
        initialization = parameters.pop("sid_initialization", "random")
        experiment = build_semantic_treatment(
            **parameters,
            representation=representation,
            num_levels=levels,
            num_codes=shared_codes,
            sid_lookup_initialization=initialization,
        )
    if experiment_logical_sha256(experiment) != job.config_logical_sha256:
        raise RuntimeError("native-500M runtime configuration identity differs")
    tokenizer_registry_sha256: str | None = None
    if representation is not None:
        if TOKENIZER_ENVIRONMENT_FIELDS & job.environment.keys() != (
            TOKENIZER_ENVIRONMENT_FIELDS
        ):
            raise RuntimeError("native-500M semantic job lacks tokenizer binding")
        registry = load_tokenizer_registry(
            DEFAULT_REGISTRY_PATH, source_sha256=expected_source
        )
        tokenizer_registry_sha256 = str(registry["sha256"])
    expected_schedule = job.payload["schedule"]
    actual_schedule = (
        "constant" if experiment.lr_schedule.shape == "constant" else "annealed"
    )
    if actual_schedule != expected_schedule:
        raise RuntimeError("native-500M runtime schedule differs")
    original_setup = experiment.setup

    def setup(self: object) -> None:
        original_setup()
        if tokenizer_registry_sha256 is not None:
            verify_tokenizer_binding(
                experiment=self,
                environment=job.environment,
                registry_sha256=tokenizer_registry_sha256,
            )

    experiment.setup = MethodType(setup, experiment)
    original_create_callbacks = experiment.create_callbacks

    def create_callbacks(self: object) -> object:
        callbacks = original_create_callbacks()
        callbacks.all.insert(
            callbacks.all.index(callbacks.best_weights) + 1,
            ValidationHistoryCallback(experiment=self, job=job),
        )
        return callbacks

    experiment.create_callbacks = MethodType(create_callbacks, experiment)
    return experiment


def experiment_logical_sha256(
    experiment: object, *, source_sha256: str | None = None
) -> str:
    fields = asdict(experiment)
    source_sha256 = source_sha256 or source_identity_sha256()
    return hashlib.sha256(
        json.dumps(
            {
                "experiment_class": (
                    f"{type(experiment).__module__}.{type(experiment).__qualname__}"
                ),
                "fields": fields,
                "source_identity_sha256": source_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_canonical_value,
        ).encode()
    ).hexdigest()


def write_run_contract(logs_root: Path, manifest: QueueManifest, job: QueueJob) -> Path:
    path = logs_root / job.run_name / "g6_native500m_job.json"
    document = {
        "schema": "g6-native500m-run-contract/v1",
        "manifest_logical_sha256": manifest.logical_sha256,
        "manifest_physical_sha256": manifest.physical_sha256,
        "job_logical_sha256": job.logical_sha256,
        "config_logical_sha256": job.config_logical_sha256,
        "job": job.payload,
    }
    content = canonical_bytes(document)
    return persist_immutable_bytes(path, content, label="native-500M run contract")


def _canonical_value(value: object) -> object:
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    if value_type == "torch.dtype":
        return {"type": value_type, "value": str(value)}
    raise TypeError(f"unsupported experiment value: {value_type}")


def _write_immutable_json(path: Path, document: dict[str, Any]) -> None:
    persist_immutable_bytes(
        path, canonical_bytes(document), label="native-500M runtime artifact"
    )
