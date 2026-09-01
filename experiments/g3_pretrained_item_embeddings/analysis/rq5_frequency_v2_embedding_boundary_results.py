from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from experiments.g3_pretrained_item_embeddings.analysis.control_calibration import (
    _file_fact,
    _load_json,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq1_results import (
    load_training_item_counts,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq5_collection import (
    _ARTIFACT_FILENAMES,
    _collect_run,
    _document,
    _exact_json_equal,
    _feature_identity,
    _require_completed_batch,
    _rq5_acceptance,
    _select,
    _validate_batch,
)
from experiments.g3_pretrained_item_embeddings.analysis.rq5_frequency_v2_results import (
    _validate_gate_mechanism,
)
from experiments.g3_pretrained_item_embeddings.launchers import (
    rq5_frequency_v2_embedding_boundary,
)
from experiments.g3_pretrained_item_embeddings.launchers.control import PROJECT_ROOT
from experiments.g3_pretrained_item_embeddings.protocol.constants import (
    APPROVED_PROTOCOL_SHA256,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5_frequency_v2_embedding_boundary_ledger import (
    RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_LEDGER_PATH,
    Rq5FrequencyV2EmbeddingBoundaryLedger,
    load_rq5_frequency_v2_embedding_boundary_ledger,
    verify_rq5_frequency_v2_embedding_boundary_inputs,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5_frequency_v2_horizon_ledger import (
    load_rq5_frequency_v2_horizon_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5_frequency_v2_ledger import (
    load_rq5_frequency_v2_ledger,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5_initial import (
    load_rq5_initial_ledger,
)


RQ5_FREQUENCY_V2_FINAL_EVIDENCE_PATH = (
    "experiments/g3_pretrained_item_embeddings/evidence/"
    "rq5_frequency_gate_fp32_p09_v2_final_native50m.json"
)


def select_corrected_frequency_boundary_outcome(
    *,
    runs: Sequence[Mapping[str, object]],
    fixed: Mapping[str, object],
    global_gate: Mapping[str, object],
) -> dict[str, object]:
    order = {
        f"rq5_frequency_gate_v2:{index:02d}": index for index in range(1, 16)
    }
    winner = _select(runs, order=order)
    values = sorted({float(run["embedding_learning_rate"]) for run in runs})
    selected = float(winner["embedding_learning_rate"])
    span = values[-1] - values[0]
    lower = values[0] + 0.1 * span
    upper = values[-1] - 0.1 * span
    direction = "lower" if selected <= lower else "upper" if selected >= upper else None
    acceptance = _rq5_acceptance(
        fixed=fixed,
        global_gate=global_gate,
        frequency_gate=winner,
    )
    return {
        "selected": dict(winner),
        "selected_row_id": winner["row_id"],
        "embedding_learning_rate_boundary": {
            "selected": selected,
            "tested_min": values[0],
            "tested_max": values[-1],
            "outer_fraction": 0.1,
            "direction": direction,
        },
        "selection_resolved": direction is None,
        "next_action": "none" if direction is None else "renewed_approval",
        "acceptance_analysis": acceptance,
    }


def build_rq5_frequency_v2_final_collection(
    *,
    root: Path,
    ledger_path: Path,
    expected_ledger_sha256: str,
    batch_id: str,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    ledger = load_rq5_frequency_v2_embedding_boundary_ledger(
        ledger_path,
        root=root,
        expected_ledger_sha256=expected_ledger_sha256,
    )
    feature_path = verify_rq5_frequency_v2_embedding_boundary_inputs(root, ledger)
    horizon_ledger = load_rq5_frequency_v2_horizon_ledger(
        root / ledger.horizon_ledger.path,
        root=root,
        expected_ledger_sha256=ledger.horizon_ledger.logical_sha256,
    )
    initial_v2 = load_rq5_frequency_v2_ledger(
        root / horizon_ledger.initial_ledger.path,
        root=root,
        expected_ledger_sha256=horizon_ledger.initial_ledger.logical_sha256,
    )
    original_initial = load_rq5_initial_ledger(root / initial_v2.initial_ledger.path)
    initial_evidence = _load_json(root / horizon_ledger.initial_evidence.path)
    horizon_evidence = _load_json(root / ledger.horizon_evidence.path)
    outcome = _load_json(root / ledger.premechanism_outcome.path)
    _verify_logical_reference(initial_evidence, horizon_ledger.initial_evidence)
    _verify_logical_reference(horizon_evidence, ledger.horizon_evidence)
    _verify_logical_reference(outcome, ledger.premechanism_outcome)

    batch_path = root / "generated/training-queue-service/batches" / f"{batch_id}.json"
    job_ids = _validate_batch(_load_json(batch_path), batch_id, 3)
    _require_completed_batch(root, job_ids)
    context_path = root / "generated/logs/.ranking-evidence/g3-native50m/context.pt"
    item_counts = load_training_item_counts(feature_path)
    artifact_filenames = dict(_ARTIFACT_FILENAMES) | {
        "job_contract": "g3_rq5_frequency_v2_embedding_boundary_job.json",
        "gate_diagnostics": "g3_gate_diagnostics.json",
    }
    new_runs = [
        _collect_run(
            root=root,
            ledger=ledger,
            ledger_path=ledger_path,
            row=row,
            batch_id=batch_id,
            job_id=job_id,
            context_path=context_path,
            item_counts=item_counts,
            feature_identity=_feature_identity(original_initial),
            runner_filename="run_rq5_frequency_v2_embedding_boundary.py",
            job_environment=rq5_frequency_v2_embedding_boundary.JOB_ENVIRONMENT,
            ledger_environment=rq5_frequency_v2_embedding_boundary.LEDGER_ENVIRONMENT,
            artifact_filenames=artifact_filenames,
        )
        for row, job_id in zip(ledger.rows, job_ids, strict=True)
    ]
    _validate_gate_mechanism(root, new_runs, ledger.rows)
    prior = [*initial_evidence["runs"], *horizon_evidence["runs"]]
    valid_corrected_runs = [*prior, *new_runs]
    _validate_valid_corrected_surface(valid_corrected_runs)

    global_selection = outcome.get("global_selection")
    fixed = outcome.get("fixed_comparator")
    global_gate = (
        global_selection.get("selected") if isinstance(global_selection, dict) else None
    )
    if (
        not isinstance(fixed, dict)
        or not isinstance(global_gate, dict)
        or global_selection.get("selection_resolved") is not True
    ):
        raise ValueError("RQ5 fixed/global comparators are unresolved")
    selection = select_corrected_frequency_boundary_outcome(
        runs=valid_corrected_runs,
        fixed=fixed,
        global_gate=global_gate,
    )
    return _document(
        {
            "schema_version": 1,
            "kind": "g3_rq5_frequency_gate_fp32_p09_v2_final_native50m",
            "protocol_sha256": APPROVED_PROTOCOL_SHA256,
            "inputs": {
                "ledger": _file_fact(root, ledger_path)
                | {"logical_sha256": ledger.sha256},
                "corrected_initial_evidence": horizon_ledger.initial_evidence.to_dict(),
                "corrected_horizon_evidence": ledger.horizon_evidence.to_dict(),
                "fixed_global_outcome": ledger.premechanism_outcome.to_dict(),
            },
            "queue_batch": _file_fact(root, batch_path) | {"batch_id": batch_id},
            "ranking_context": _file_fact(root, context_path),
            "new_runs": new_runs,
            "valid_corrected_frequency_tuning_rows": valid_corrected_runs,
            "legacy_frequency_artifact_policy": {
                "semantics": "bfloat16_p09999_saturated_zero_gradient",
                "reader_and_tuning_eligible": False,
                "raw_artifacts_preserved_by_bound_audit_inputs": True,
            },
            "fixed_comparator": fixed,
            "global_comparator": global_gate,
            "final_selection": selection,
        }
    )


def persist_rq5_frequency_v2_final_collection(
    path: Path,
    document: dict[str, object],
    *,
    root: Path,
    ledger_path: Path,
    expected_ledger_sha256: str,
    batch_id: str,
) -> Path:
    root = root.resolve(strict=True)
    destination = (root / RQ5_FREQUENCY_V2_FINAL_EVIDENCE_PATH).resolve()
    if path.resolve() != destination or destination.is_symlink():
        raise ValueError("RQ5 corrected-frequency final evidence destination is not canonical")
    rebuilt = build_rq5_frequency_v2_final_collection(
        root=root,
        ledger_path=ledger_path,
        expected_ledger_sha256=expected_ledger_sha256,
        batch_id=batch_id,
    )
    if not _exact_json_equal(document, rebuilt):
        raise ValueError("RQ5 corrected-frequency final evidence differs from rebuilt evidence")
    content = (json.dumps(rebuilt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise RuntimeError(f"immutable RQ5 corrected-frequency evidence differs: {destination}")
    return destination


def _verify_logical_reference(
    document: Mapping[str, object], reference: object
) -> None:
    logical = getattr(reference, "logical_sha256")
    if document.get("sha256") != logical:
        raise ValueError("RQ5 referenced logical evidence changed")


def _validate_valid_corrected_surface(runs: Sequence[Mapping[str, object]]) -> None:
    expected_ids = {
        f"rq5_frequency_gate_v2:{index:02d}" for index in range(1, 16)
    }
    ids = {run.get("row_id") for run in runs}
    if len(runs) != 15 or ids != expected_ids:
        raise ValueError("RQ5 corrected-frequency opportunity accounting changed")
    if any(
        run.get("family_id") != "rq5_frequency_gate_v2"
        or run.get("content_gate") != "frequency"
        for run in runs
    ):
        raise ValueError("RQ5 legacy frequency row entered the valid surface")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=PROJECT_ROOT / RQ5_FREQUENCY_V2_EMBEDDING_BOUNDARY_LEDGER_PATH,
    )
    parser.add_argument("--expected-ledger-sha256", required=True)
    arguments = parser.parse_args()
    document = build_rq5_frequency_v2_final_collection(
        root=PROJECT_ROOT,
        ledger_path=arguments.ledger,
        expected_ledger_sha256=arguments.expected_ledger_sha256,
        batch_id=arguments.batch_id,
    )
    path = persist_rq5_frequency_v2_final_collection(
        PROJECT_ROOT / RQ5_FREQUENCY_V2_FINAL_EVIDENCE_PATH,
        document,
        root=PROJECT_ROOT,
        ledger_path=arguments.ledger,
        expected_ledger_sha256=arguments.expected_ledger_sha256,
        batch_id=arguments.batch_id,
    )
    print(path)
    print(document["sha256"])


if __name__ == "__main__":
    main()
