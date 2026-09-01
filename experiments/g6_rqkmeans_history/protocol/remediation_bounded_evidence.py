from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

from dcn.eval.ranking_evidence import load_ranking_evidence
from experiments.g6_rqkmeans_history.launchers.remediation_bounded import (
    CONTRACT_FILENAME,
    build_bounded_gate_experiment,
)
from experiments.g6_rqkmeans_history.protocol.evidence import (
    REQUIRED_METRICS,
    _numeric_metrics,
    _read_json,
    _validate_metadata,
    _validate_semantic_diagnostics,
    _validate_sid_metrics,
)
from experiments.g6_rqkmeans_history.protocol.manifest import (
    CompiledJob,
    RANKING_EVIDENCE_GROUP,
)
from experiments.g6_rqkmeans_history.protocol.remediation_bounded import (
    bounded_gate_manifest,
    validate_bounded_gate_job,
)


@dataclass(frozen=True)
class BoundedGateArtifact:
    compiled: CompiledJob
    path: Path
    metrics: dict[str, float]
    metadata: dict[str, Any]
    semantic_diagnostics: dict[str, Any]


def load_bounded_gate_artifact(
    compiled: CompiledJob, logs_root: Path
) -> BoundedGateArtifact:
    validate_bounded_gate_job(compiled)
    directory = logs_root / compiled.run_name
    contract = _read_json(directory / CONTRACT_FILENAME)
    if contract != compiled.to_contract(bounded_gate_manifest()):
        raise ValueError(f"{compiled.run_name}: bounded-gate contract changed")
    metrics = _numeric_metrics(_read_json(directory / "final_metrics.json"))
    for name in REQUIRED_METRICS:
        if name not in metrics:
            raise ValueError(f"{compiled.run_name}: missing {name}")
    metadata = _read_json(directory / "training_metadata.json")
    _validate_metadata(
        compiled,
        metadata,
        experiment_builder=build_bounded_gate_experiment,
        builder_resolver=lambda _: "treatment",
    )
    expected_scale = compiled.parameters["learned_residual_max_scale"]
    expected_architecture = {
        "history_representation": compiled.parameters["representation"],
        "representation_width": compiled.parameters["representation_width"],
        "frozen_event_width": compiled.parameters["frozen_event_width"],
        "learned_residual_max_scale": expected_scale,
    }
    if any(
        metadata.get(name) != value for name, value in expected_architecture.items()
    ):
        raise ValueError(f"{compiled.run_name}: bounded-gate architecture changed")
    load_ranking_evidence(
        logs_root / ".ranking-evidence" / RANKING_EVIDENCE_GROUP / "context.pt",
        directory / "ranking_evidence.pt",
    )
    _validate_sid_metrics(compiled, metrics)
    diagnostics = _read_json(directory / "semantic_id_diagnostics.json")
    _validate_semantic_diagnostics(compiled, diagnostics)
    raw_scale = diagnostics.get("learned_residual_raw_scale")
    effective_scale = diagnostics.get("learned_residual_effective_scale")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in (raw_scale, effective_scale)
    ):
        raise ValueError(f"{compiled.run_name}: bounded-gate scale is absent")
    if abs(effective_scale) > expected_scale + 1e-7:
        raise ValueError(f"{compiled.run_name}: effective gate exceeds its bound")
    return BoundedGateArtifact(compiled, directory, metrics, metadata, diagnostics)


def bounded_gate_artifact_state(compiled: CompiledJob, logs_root: Path) -> str:
    directory = logs_root / compiled.run_name
    terminal = (
        CONTRACT_FILENAME,
        "final_metrics.json",
        "training_metadata.json",
        "ranking_evidence.pt",
        "semantic_id_diagnostics.json",
    )
    if all((directory / name).is_file() for name in terminal):
        load_bounded_gate_artifact(compiled, logs_root)
        return "complete"
    if directory.exists() and any(directory.iterdir()):
        return "partial"
    return "missing"


def select_positive_bounded_gate(
    artifacts: Sequence[BoundedGateArtifact],
) -> BoundedGateArtifact:
    positive = [
        artifact
        for artifact in artifacts
        if artifact.compiled.parameters["learned_residual_max_scale"] > 0
    ]
    if not positive:
        raise ValueError("bounded-gate selection requires positive bounds")
    return max(
        positive,
        key=lambda artifact: (
            artifact.metrics["recall@100"],
            artifact.metrics["ndcg@100"],
            -artifact.compiled.approved.trial,
        ),
    )


def bounded_gate_row(artifact: BoundedGateArtifact) -> dict[str, Any]:
    return {
        "job_id": artifact.compiled.approved.id,
        "run_name": artifact.compiled.run_name,
        "parameters": artifact.compiled.parameters,
        "metrics": artifact.metrics,
        "best_epoch": artifact.metadata["best_epoch"],
        "epochs_trained": artifact.metadata["epochs_trained"],
        "learned_residual_raw_scale": artifact.semantic_diagnostics[
            "learned_residual_raw_scale"
        ],
        "learned_residual_effective_scale": artifact.semantic_diagnostics[
            "learned_residual_effective_scale"
        ],
    }


def write_bounded_gate_selection(path: Path, document: dict[str, Any]) -> None:
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"existing bounded-gate selection differs: {path}")
    path.write_text(content)
