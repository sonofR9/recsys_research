from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from experiments.g6_rqkmeans_history.launchers.bounded_gate_manifest import (
    write_bounded_gate_jobs,
)
from experiments.g6_rqkmeans_history.protocol.remediation_bounded import (
    SOURCE_SELECTION_PATH,
    bounded_gate_jobs,
    bounded_gate_manifest,
    load_bounded_gate_source,
)
from experiments.g6_rqkmeans_history.protocol.remediation_bounded_evidence import (
    bounded_gate_artifact_state,
    bounded_gate_row,
    load_bounded_gate_artifact,
    select_positive_bounded_gate,
    write_bounded_gate_selection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUEUE_LAUNCHER = Path(__file__).with_name("queue_remediation_bounded.sh")
RECALL_BAND = 0.002
NDCG_BAND = 0.002


def _record_launch_or_reuse_complete(*, ledger_path: Path, logs_root: Path) -> bool:
    jobs = bounded_gate_jobs()
    recorded = ledger_path.exists()
    if recorded:
        write_bounded_gate_jobs(ledger_path, jobs)
    states = [bounded_gate_artifact_state(job, logs_root) for job in jobs]
    if "partial" in states:
        raise RuntimeError("bounded-gate grid has a partial artifact requiring audit")
    if len(set(states)) != 1:
        raise RuntimeError("bounded-gate grid has mixed artifact state requiring audit")
    if states[0] == "complete":
        if not recorded:
            write_bounded_gate_jobs(ledger_path, jobs)
        return False
    if recorded:
        raise RuntimeError(
            "bounded-gate launch is recorded but artifacts are missing; audit required"
        )
    write_bounded_gate_jobs(ledger_path, jobs)
    return True


def run_bounded_gate_grid(
    *,
    ledger_path: Path,
    logs_root: Path,
    selection_path: Path,
    source_path: Path = SOURCE_SELECTION_PATH,
) -> dict:
    source = load_bounded_gate_source(source_path)
    jobs = bounded_gate_jobs()
    manifest = bounded_gate_manifest()
    if (
        manifest.to_dict()["prior_physical_runs"]
        + manifest.to_dict()["new_physical_runs"]
        > manifest.to_dict()["maximum_total_physical_runs"]
    ):
        raise RuntimeError("bounded-gate grid exceeds the approved physical budget")
    if _record_launch_or_reuse_complete(ledger_path=ledger_path, logs_root=logs_root):
        subprocess.run(
            ["bash", str(QUEUE_LAUNCHER), str(ledger_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    artifacts = [load_bounded_gate_artifact(job, logs_root) for job in jobs]
    zero = artifacts[0]
    positive = select_positive_bounded_gate(artifacts)
    control = source["control"]
    control_recall = control["metrics"]["recall@100"]
    control_ndcg = control["metrics"]["ndcg@100"]
    document = {
        "manifest_sha256": manifest.sha256,
        "dataset_size": "native-50m",
        "source_selection_sha256": manifest.to_dict()["source_selection_sha256"],
        "control": control,
        "zero_bound_diagnostic": bounded_gate_row(zero),
        "positive_winner": bounded_gate_row(positive),
        "positive_noninferior": (
            positive.metrics["recall@100"] >= control_recall - RECALL_BAND
            and positive.metrics["ndcg@100"] >= control_ndcg - NDCG_BAND
        ),
        "positive_promoted": (
            positive.metrics["recall@100"] > control_recall + RECALL_BAND
            and positive.metrics["ndcg@100"] >= control_ndcg - NDCG_BAND
        ),
        "run_count": len(artifacts),
        "rows": [bounded_gate_row(artifact) for artifact in artifacts],
    }
    write_bounded_gate_selection(selection_path, document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, default=Path("generated/logs"))
    parser.add_argument("--source", type=Path, default=SOURCE_SELECTION_PATH)
    arguments = parser.parse_args()
    run_bounded_gate_grid(
        ledger_path=arguments.ledger,
        logs_root=arguments.logs_root,
        selection_path=arguments.selection,
        source_path=arguments.source,
    )


if __name__ == "__main__":
    main()
