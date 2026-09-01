from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dcn.eval.ranking_evidence import load_ranking_evidence
from experiments.g6_rqkmeans_history.analysis.rq0_slices import _metrics
from experiments.g6_rqkmeans_history.configs.collision_policy import (
    build_collision_policy_experiment,
)
from experiments.g6_rqkmeans_history.launchers.collision_runtime import (
    CONTRACT_FILENAME,
)
from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    collision_search_manifest,
)
from experiments.g6_rqkmeans_history.protocol.manifest import RANKING_EVIDENCE_GROUP


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_ROOT = PROJECT_ROOT / "generated/logs"


def repair(job_id: str, logs_root: Path = LOGS_ROOT) -> Path:
    manifest = collision_search_manifest()
    matches = [job for job in manifest.new_physical_jobs if job.id == job_id]
    if len(matches) != 1:
        raise ValueError(f"collision job {job_id!r} is not approved")
    job = matches[0]
    directory = logs_root / job.run_name
    contract = json.loads((directory / CONTRACT_FILENAME).read_text())
    if contract != {"manifest_sha256": manifest.sha256, "job": job.to_dict()}:
        raise ValueError(f"{job.run_name}: job contract changed")
    final_metrics = json.loads((directory / "final_metrics.json").read_text())
    evidence = load_ranking_evidence(
        logs_root
        / ".ranking-evidence"
        / RANKING_EVIDENCE_GROUP
        / "context.pt",
        directory / "ranking_evidence.pt",
    )
    recomputed = _metrics(
        evidence,
        torch.ones_like(evidence.relevant_item_ids, dtype=torch.bool),
    )
    for name, value in recomputed.items():
        if abs(final_metrics[name] - value) > 1e-12:
            raise ValueError(f"{job.run_name}: {name} disagrees with ranking evidence")
    metadata = json.loads((directory / "training_metadata.json").read_text())
    expected_metadata = {
        "dataset_size": "50m",
        "batch_size": 256,
        "seed": 42,
        "embedding_learning_rate": job.coordinate.embedding_learning_rate,
        "deep_learning_rate": job.coordinate.deep_learning_rate,
        "num_epochs": 15,
        "lr_schedule_horizon_epochs": 15,
        "lr_horizon_complete": True,
    }
    if any(metadata.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError(f"{job.run_name}: training metadata changed")
    experiment = build_collision_policy_experiment(job)
    experiment.setup()
    document = experiment.semantic_diagnostics_document()
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    destination = directory / "semantic_id_diagnostics.json"
    if destination.exists():
        if destination.read_text() != content:
            raise ValueError(f"{job.run_name}: diagnostics already differ")
        return destination
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(content)
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_ids", nargs="+")
    arguments = parser.parse_args()
    for job_id in arguments.job_ids:
        print(repair(job_id))


if __name__ == "__main__":
    main()
