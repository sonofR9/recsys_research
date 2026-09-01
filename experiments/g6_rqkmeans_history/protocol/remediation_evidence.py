from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from dcn.eval.ranking_evidence import load_ranking_evidence
from experiments.g6_rqkmeans_history.launchers.remediation_compiled import (
    CONTRACT_FILENAME,
    build_remediation_experiment,
)
from experiments.g6_rqkmeans_history.protocol.evidence import (
    BoundaryApprovalRequired,
    CapExtensionRequired,
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
from experiments.g6_rqkmeans_history.protocol.remediation import (
    CARRYOVER_MANIFEST_SHA256,
    carryover_compiled_jobs,
    remediation_manifest,
    validate_remediation_job,
)


@dataclass(frozen=True)
class RemediationArtifact:
    compiled: CompiledJob
    path: Path
    metrics: dict[str, float]
    metadata: dict[str, Any]
    semantic_diagnostics: dict[str, Any]


def _builder(compiled: CompiledJob) -> str:
    if compiled.approved.stage in {
        "remediation_bridge_tuning",
        "remediation_bridge_lr_boundary",
    }:
        return "bridge"
    return "treatment"


def _expected_contract(compiled: CompiledJob) -> dict[str, Any]:
    if compiled in carryover_compiled_jobs():
        return {
            "manifest_sha256": CARRYOVER_MANIFEST_SHA256,
            "job": compiled.approved.to_dict(),
            "parameters": compiled.parameters,
        }
    return compiled.to_contract(remediation_manifest())


def load_remediation_artifact(
    compiled: CompiledJob, logs_root: Path
) -> RemediationArtifact:
    validate_remediation_job(compiled)
    directory = logs_root / compiled.run_name
    contract = _read_json(directory / CONTRACT_FILENAME)
    if contract != _expected_contract(compiled):
        raise ValueError(f"{compiled.run_name}: remediation job contract changed")
    metrics = _numeric_metrics(_read_json(directory / "final_metrics.json"))
    for name in REQUIRED_METRICS:
        if name not in metrics:
            raise ValueError(f"{compiled.run_name}: missing {name}")
    metadata = _read_json(directory / "training_metadata.json")
    _validate_metadata(
        compiled,
        metadata,
        experiment_builder=build_remediation_experiment,
        builder_resolver=_builder,
    )
    load_ranking_evidence(
        logs_root / ".ranking-evidence" / RANKING_EVIDENCE_GROUP / "context.pt",
        directory / "ranking_evidence.pt",
    )
    _validate_sid_metrics(compiled, metrics)
    diagnostics = _read_json(directory / "semantic_id_diagnostics.json")
    _validate_semantic_diagnostics(compiled, diagnostics)
    return RemediationArtifact(compiled, directory, metrics, metadata, diagnostics)


def remediation_artifact_state(compiled: CompiledJob, logs_root: Path) -> str:
    directory = logs_root / compiled.run_name
    terminal = (
        CONTRACT_FILENAME,
        "final_metrics.json",
        "training_metadata.json",
        "ranking_evidence.pt",
        "semantic_id_diagnostics.json",
    )
    if all((directory / name).is_file() for name in terminal):
        try:
            load_remediation_artifact(compiled, logs_root)
        except CapExtensionRequired:
            return "extend_cap"
        return "complete"
    if directory.exists() and any(directory.iterdir()):
        return "partial"
    return "missing"


def select_remediation_best(
    artifacts: Sequence[RemediationArtifact],
) -> RemediationArtifact:
    if not artifacts:
        raise ValueError("remediation selection requires artifacts")
    return max(
        artifacts,
        key=lambda artifact: (
            artifact.metrics["recall@100"],
            artifact.metrics["ndcg@100"],
            -int(artifact.compiled.approved.trial or 0),
        ),
    )


def require_remediation_boundary_resolved(artifact: RemediationArtifact) -> None:
    if artifact.compiled.approved.stage not in {
        "remediation_lr_boundary",
        "remediation_bridge_lr_boundary",
    }:
        return
    if artifact.compiled.approved.forced_parameters.get("boundary_slot") == 3:
        raise BoundaryApprovalRequired(
            f"{artifact.compiled.run_name}: outermost LR extension won"
        )


def selection_row(artifact: RemediationArtifact) -> dict[str, Any]:
    return {
        "job_id": artifact.compiled.approved.id,
        "run_name": artifact.compiled.run_name,
        "parameters": artifact.compiled.parameters,
        "metrics": artifact.metrics,
        "best_epoch": artifact.metadata.get("best_epoch"),
        "epochs_trained": artifact.metadata.get("epochs_trained"),
    }


def write_remediation_selection(path: Path, document: dict[str, Any]) -> None:
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"existing remediation selection differs: {path}")
    path.write_text(content)
