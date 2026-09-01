from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.g6_rqkmeans_history.launchers.remediation_manifest import (
    load_remediation_jobs,
    write_remediation_jobs,
)
from experiments.g6_rqkmeans_history.protocol.manifest import (
    CompiledJob,
    approved_manifest,
)
from experiments.g6_rqkmeans_history.protocol.remediation import (
    CONTROL_JOB_ID,
    CONTROL_RUN_NAME,
    NDCG_BAND,
    ORIGINAL_CONTROL_JOB_ID,
    ORIGINAL_CONTROL_RUN_NAME,
    RECALL_BAND,
    MAXIMUM_PHYSICAL_RUNS,
    RemediationDriver,
    RunBudgetApprovalRequired,
    carryover_compiled_jobs,
    compile_remediation_cap_continuation,
    remediation_manifest,
)
from experiments.g6_rqkmeans_history.protocol.remediation_evidence import (
    RemediationArtifact,
    load_remediation_artifact,
    remediation_artifact_state,
    require_remediation_boundary_resolved,
    select_remediation_best,
    selection_row,
    write_remediation_selection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUEUE_LAUNCHER = Path(__file__).with_name("queue_remediation.sh")
CONTROL_SELECTION_PATH = (
    PROJECT_ROOT
    / "experiments/g6_rqkmeans_history/evidence/rq0_selection_native50m.json"
)
CONTROL_RECALL = 0.13018225318522603
CONTROL_NDCG = 0.05168322558159157
CONTROL_ROW_SHA256 = "f61f41ba4b21f784bbd4cc678f97308b9c3a438acb578a8151200a0150bf05c8"
ORIGINAL_CONTROL_ROW_SHA256 = (
    "9e14f35ace73502fe3809f810cbe56dc2c99229149fddbbcc79c24f094a9974d"
)


def _row_sha256(row: dict) -> str:
    content = json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(content).hexdigest()


def load_control_reference(path: Path = CONTROL_SELECTION_PATH) -> dict:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read remediation control selection {path}") from error
    if document.get("manifest_sha256") != approved_manifest().sha256:
        raise ValueError("remediation control manifest changed")
    control = document.get("semantic_winner")
    original = document.get("original_control")
    if not isinstance(control, dict) or not isinstance(original, dict):
        raise ValueError("remediation controls are absent")
    if _row_sha256(control) != CONTROL_ROW_SHA256:
        raise ValueError("remediation control row changed")
    if _row_sha256(original) != ORIGINAL_CONTROL_ROW_SHA256:
        raise ValueError("remediation original control row changed")
    if control.get("job_id") != CONTROL_JOB_ID:
        raise ValueError("remediation control job changed")
    if control.get("run_name") != CONTROL_RUN_NAME:
        raise ValueError("remediation control run changed")
    if original.get("job_id") != ORIGINAL_CONTROL_JOB_ID:
        raise ValueError("remediation original control job changed")
    if original.get("run_name") != ORIGINAL_CONTROL_RUN_NAME:
        raise ValueError("remediation original control run changed")
    parameters = control.get("parameters")
    metrics = control.get("metrics")
    expected_parameters = {
        "batch_size": 256,
        "validation_batch_size": 8192,
        "num_levels": 3,
        "num_codes": 512,
        "representation": "item_frozen_sid_event",
        "representation_width": 128,
    }
    if not isinstance(parameters, dict) or any(
        parameters.get(name) != value for name, value in expected_parameters.items()
    ):
        raise ValueError("remediation control configuration changed")
    if not isinstance(metrics, dict) or metrics.get("recall@100") != CONTROL_RECALL:
        raise ValueError("remediation control Recall changed")
    if metrics.get("ndcg@100") != CONTROL_NDCG:
        raise ValueError("remediation control NDCG changed")
    return {"control": control, "original_control": original}


class RemediationLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, compiled: CompiledJob) -> None:
        existing = [] if not self.path.exists() else load_remediation_jobs(self.path)
        by_identity = {job.identity: job for job in existing}
        previous = by_identity.get(compiled.identity)
        if previous is not None and previous != compiled:
            raise ValueError(f"remediation job changed: {compiled.identity}")
        if any(
            job.approved.id == compiled.approved.id
            and job.parameters != compiled.parameters
            for job in existing
        ):
            raise ValueError(f"remediation parameters changed: {compiled.approved.id}")
        if previous is None and len(existing) >= MAXIMUM_PHYSICAL_RUNS:
            raise RunBudgetApprovalRequired(
                f"remediation is limited to {MAXIMUM_PHYSICAL_RUNS} physical runs"
            )
        by_identity[compiled.identity] = compiled
        order = {job.id: index for index, job in enumerate(remediation_manifest().jobs)}
        write_remediation_jobs(
            self.path,
            sorted(
                by_identity.values(),
                key=lambda job: (order[job.approved.id], job.attempt),
            ),
        )

    def require_capacity(self, jobs: tuple[CompiledJob, ...]) -> None:
        existing = [] if not self.path.exists() else load_remediation_jobs(self.path)
        identities = {compiled.identity for compiled in existing}
        additions = {
            compiled.identity
            for compiled in jobs
            if compiled.identity not in identities
        }
        if len(existing) + len(additions) > MAXIMUM_PHYSICAL_RUNS:
            raise RunBudgetApprovalRequired(
                f"remediation is limited to {MAXIMUM_PHYSICAL_RUNS} physical runs"
            )


class QueueSubmitter:
    def __init__(self, batches_root: Path) -> None:
        self.batches_root = batches_root

    def __call__(self, jobs: tuple[CompiledJob, ...]) -> None:
        digest = hashlib.sha256(
            json.dumps(
                [job.to_contract(remediation_manifest()) for job in jobs],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:16]
        batch = self.batches_root / f"batch_{digest}.json"
        write_remediation_jobs(batch, list(jobs))
        subprocess.run(
            ["bash", str(QUEUE_LAUNCHER), str(batch)],
            cwd=PROJECT_ROOT,
            check=True,
        )


class RemediationWorkflow:
    def __init__(
        self,
        driver: RemediationDriver,
        *,
        logs_root: Path,
        ledger_path: Path,
        submit: QueueSubmitter,
    ) -> None:
        self.driver = driver
        self.logs_root = logs_root
        self.ledger = RemediationLedger(ledger_path)
        self.ledger_path = ledger_path
        self.submit = submit

    def advance(self, ask) -> list[RemediationArtifact]:
        artifacts: list[RemediationArtifact] = []
        while (compiled := ask()) is not None:
            artifact = self.run((compiled,))[0]
            self.driver.tell(compiled, artifact.metrics["recall@100"], artifact.path)
            artifacts.append(artifact)
        return artifacts

    def run(self, jobs: tuple[CompiledJob, ...]) -> list[RemediationArtifact]:
        unresolved = list(jobs)
        resolved: dict[str, CompiledJob] = {}
        while unresolved:
            self.ledger.require_capacity(tuple(unresolved))
            pending: list[CompiledJob] = []
            for compiled in unresolved:
                self.ledger.append(compiled)
                state = remediation_artifact_state(compiled, self.logs_root)
                if state == "partial":
                    raise RuntimeError(
                        f"{compiled.run_name}: partial artifact requires audit"
                    )
                if state == "extend_cap":
                    pending.append(compile_remediation_cap_continuation(compiled))
                elif state == "missing":
                    if compiled in carryover_compiled_jobs():
                        raise RuntimeError(
                            f"{compiled.run_name}: bound carryover artifact is missing"
                        )
                    pending.append(compiled)
                else:
                    resolved[compiled.approved.id] = compiled
            if pending:
                self.ledger.require_capacity(tuple(pending))
                self.submit(tuple(pending))
            unresolved = pending
        return [
            load_remediation_artifact(resolved[job.approved.id], self.logs_root)
            for job in jobs
        ]

    def artifacts_for_stage(self, stage: str) -> list[RemediationArtifact]:
        compiled: dict[str, CompiledJob] = {}
        for candidate in load_remediation_jobs(self.ledger_path):
            previous = compiled.get(candidate.approved.id)
            if previous is None or candidate.attempt > previous.attempt:
                compiled[candidate.approved.id] = candidate
        return [
            load_remediation_artifact(compiled[job.id], self.logs_root)
            for job in remediation_manifest().jobs_for_stage(stage)
            if job.id in compiled
        ]

    def total_compiled_runs(self) -> int:
        return len(load_remediation_jobs(self.ledger_path))

    def total_new_runs(self) -> int:
        carryovers = carryover_compiled_jobs()
        return sum(
            compiled not in carryovers
            for compiled in load_remediation_jobs(self.ledger_path)
        )


def run_program(
    workflow: RemediationWorkflow,
    driver: RemediationDriver,
    *,
    control_reference: dict,
) -> dict:
    carryover_artifacts = workflow.run(carryover_compiled_jobs())
    driver.register_carryovers(
        [
            (artifact.compiled, artifact.metrics["recall@100"], artifact.path)
            for artifact in carryover_artifacts
        ]
    )
    workflow.advance(driver.next_treatment)
    initial = workflow.artifacts_for_stage("remediation_tuning")
    initial_winner = select_remediation_best(initial)
    boundaries = workflow.run(driver.compile_lr_boundaries(initial_winner.compiled))
    winner = select_remediation_best([*initial, *boundaries])
    require_remediation_boundary_resolved(winner)
    promoted = driver.promotion_eligible(
        control_recall=CONTROL_RECALL,
        control_ndcg=CONTROL_NDCG,
        treatment_recall=winner.metrics["recall@100"],
        treatment_ndcg=winner.metrics["ndcg@100"],
    )
    bridge_winner = None
    bridge_runs: list[RemediationArtifact] = []
    if promoted:
        workflow.advance(lambda: driver.next_bridge(winner.compiled))
        bridge_initial = workflow.artifacts_for_stage("remediation_bridge_tuning")
        first_bridge = select_remediation_best(bridge_initial)
        bridge_boundaries = workflow.run(
            driver.compile_lr_boundaries(first_bridge.compiled)
        )
        bridge_runs = [*bridge_initial, *bridge_boundaries]
        bridge_winner = select_remediation_best(bridge_runs)
        require_remediation_boundary_resolved(bridge_winner)
    return {
        "manifest_sha256": remediation_manifest().sha256,
        "dataset_size": "native-50m",
        "control": control_reference["control"],
        "original_control": control_reference["original_control"],
        "metric_bands": {"recall@100": RECALL_BAND, "ndcg@100": NDCG_BAND},
        "treatment_winner": selection_row(winner),
        "promoted": promoted,
        "bridge_winner": (
            None if bridge_winner is None else selection_row(bridge_winner)
        ),
        "run_counts": {
            "treatment_initial": len(initial),
            "treatment_boundary": len(boundaries),
            "bridge": len(bridge_runs),
            "total_new": workflow.total_new_runs(),
            "total_including_carryovers": workflow.total_compiled_runs(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, default=Path("generated/logs"))
    parser.add_argument("--batches-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--control-selection", type=Path, default=CONTROL_SELECTION_PATH
    )
    arguments = parser.parse_args()
    driver = RemediationDriver(arguments.database)
    workflow = RemediationWorkflow(
        driver,
        logs_root=arguments.logs_root,
        ledger_path=arguments.ledger,
        submit=QueueSubmitter(arguments.batches_root),
    )
    write_remediation_selection(
        arguments.selection,
        run_program(
            workflow,
            driver,
            control_reference=load_control_reference(arguments.control_selection),
        ),
    )


if __name__ == "__main__":
    main()
