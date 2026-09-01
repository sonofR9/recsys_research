from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

import polars as pl
import torch

from dcn.eval.ranking_evidence import load_ranking_evidence
from experiments.g3_pretrained_item_embeddings.analysis.queue_attribution import (
    verify_artifacts_in_job_window,
    verify_unique_completed_run,
)
from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
    DATA_GROUP,
    JOB_ENVIRONMENT,
    MANIFEST_ENVIRONMENT,
    MANIFEST_LOGICAL_SHA256_ENVIRONMENT,
    MANIFEST_PHYSICAL_SHA256_ENVIRONMENT,
    PROJECT_ROOT,
    build_batch_specification,
    load_execution_manifest,
    load_queue_submission_binding,
    resolve_input_manifest_path,
)
from experiments.g3_pretrained_item_embeddings.analysis.slices import (
    compute_ranking_slices,
)
from experiments.g3_pretrained_item_embeddings.protocol.native500m.constants import (
    PROTOCOL,
)


_METRIC_NAMES = tuple(
    f"{name}@{cutoff}"
    for name in ("recall", "ndcg", "mrr", "capped_recall", "coverage")
    for cutoff in (10, 50, 100)
)
_METRIC_ABSOLUTE_TOLERANCE = 32 * math.ulp(1.0)
_ARTIFACT_FILENAMES = {
    "job_contract": "g3_native500m_job.json",
    "training_metadata": "training_metadata.json",
    "final_metrics": "final_metrics.json",
    "ranking_evidence": "ranking_evidence.pt",
    "top_item_rankings": "top_item_rankings.json",
    "training_diagnostics": "g3_training_diagnostics.json",
    "restored_best_checkpoint": "restored_best_checkpoint.pt",
    "final_evaluation_proof": "final_evaluation_proof.json",
    "sweep_log": "sweep.log",
}
_METRIC_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_AUTHENTICATION_CONTEXT: ContextVar[dict[str, object] | None] = ContextVar(
    "g3_native500m_authentication_context", default=None
)


@contextmanager
def _authentication_scope():
    context = _AUTHENTICATION_CONTEXT.get()
    owner = context is None
    token = None
    if owner:
        context = {"batch_documents": {}, "source_facts": {}}
        token = _AUTHENTICATION_CONTEXT.set(context)
    assert context is not None
    try:
        yield context
        if owner:
            source_facts = context["source_facts"]
            assert isinstance(source_facts, dict)
            for root, path, fact, label in source_facts.values():
                _require_unchanged_fact(root, path, fact, label)
    finally:
        if owner:
            assert token is not None
            _AUTHENTICATION_CONTEXT.reset(token)


def _authenticated_operation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _authentication_scope():
            return function(*args, **kwargs)

    return wrapped


def _catalog_sha256(num_items: int) -> str:
    return hashlib.sha256(
        json.dumps(list(range(1, num_items + 1)), separators=(",", ":")).encode()
    ).hexdigest()


def collect_batch_evidence(
    *,
    root: Path,
    manifest_path: Path,
    batch_specification_path: Path,
    batch_id: str,
    expected_protocol_sha256: str | None = None,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    manifest = load_execution_manifest(
        manifest_path,
        expected_protocol_sha256=expected_protocol_sha256,
        validate_inputs=True,
    )
    specification_path = batch_specification_path.resolve(strict=True)
    if specification_path.is_symlink() or not specification_path.is_file():
        raise ValueError("batch specification is not a regular file")
    manifest_fact = _file_fact(root, manifest.path)
    specification_document, specification_fact = _load_json_with_fact(
        root, specification_path
    )
    specification = build_batch_specification(
        manifest_path,
        expected_protocol_sha256=manifest.protocol_sha256,
    )
    if (
        specification_document != specification.document
        or _canonical_sha256(specification_document) != specification.sha256
        or _file_sha256(specification_path) != specification.sha256
    ):
        raise ValueError("batch specification differs from the execution manifest")
    queue_root = root / "generated/training-queue-service"
    batch_path = queue_root / "batches" / f"{batch_id}.json"
    batch, batch_fact = _load_json_with_fact(root, batch_path)
    job_ids = batch.get("jobs")
    if (
        batch.get("id") != batch_id
        or batch.get("sealed") is not True
        or batch.get("atomic_submission") is not True
        or batch.get("expected_job_count") != len(manifest.rows)
        or batch.get("specification_sha256") != specification.sha256
        or not isinstance(job_ids, list)
        or len(job_ids) != len(manifest.rows)
        or len(set(job_ids)) != len(job_ids)
    ):
        raise ValueError("queue batch is not the exact sealed atomic submission")
    submission_binding = load_queue_submission_binding(
        specification_path=specification_path,
        manifest=manifest,
        batch_id=batch_id,
    )
    submission_binding_path = specification_path.with_name(
        f"{specification.sha256}-q"
        f"{str(submission_binding['queue_service_identity']['sha256'])[:16]}-"
        f"{batch_id}.submission.json"
    )
    submission_binding_fact = _file_fact(root, submission_binding_path) | {
        "logical_sha256": submission_binding["sha256"],
        "queue_service_identity_sha256": submission_binding["queue_service_identity"][
            "sha256"
        ],
    }
    context_path = (
        root / "generated/logs/.ranking-evidence/g3-native500m-likes/context.pt"
    )
    context_fact = _file_fact(root, context_path)
    expected_population = manifest.evaluation_population
    training_item_counts, training_history_lengths = _load_slice_inputs(root, manifest)
    runs = [
        _collect_run(
            root=root,
            queue_root=queue_root,
            context_path=context_path,
            manifest=manifest,
            specification_row=specification.document["jobs"][index],
            batch_id=batch_id,
            job_id=str(job_ids[index]),
            row=manifest.rows[index],
            expected_population=expected_population,
            training_item_counts=training_item_counts,
            training_history_lengths=training_history_lengths,
        )
        for index in range(len(manifest.rows))
    ]
    body = {
        "schema_version": 1,
        "kind": "g3_native500m_batch_evidence",
        "protocol_sha256": manifest.protocol_sha256,
        "data_group": DATA_GROUP,
        "execution_manifest": manifest_fact
        | {
            "logical_sha256": manifest.logical_sha256,
            "physical_sha256": manifest.physical_sha256,
            "stage": manifest.stage,
        },
        "input_manifests": [
            reference.to_dict() for reference in manifest.input_manifests
        ],
        "batch_specification": specification_fact | {"sha256": specification.sha256},
        "queue_batch": batch_fact | {"batch_id": batch_id},
        "queue_submission_binding": submission_binding_fact,
        "ranking_context": context_fact,
        "runs": runs,
    }
    for label, path, fact in (
        ("execution manifest", manifest.path, manifest_fact),
        ("batch specification", specification_path, specification_fact),
        ("queue batch", batch_path, batch_fact),
        ("ranking context", context_path, context_fact),
    ):
        _require_unchanged_fact(root, path, fact, label)
    _require_unchanged_fact(
        root,
        submission_binding_path,
        submission_binding_fact,
        "queue submission binding",
    )
    return {**body, "sha256": _canonical_sha256(body)}


def persist_batch_evidence(path: Path, document: Mapping[str, object]) -> Path:
    value = dict(document)
    supplied = value.pop("sha256", None)
    if supplied != _canonical_sha256(value):
        raise ValueError("batch evidence logical SHA-256 differs")
    content = _canonical_bytes(document) + b"\n"
    _write_immutable(path, content)
    return path


@_authenticated_operation
def load_batch_evidence(
    path: Path,
    *,
    expected_protocol_sha256: str | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    path = path.resolve(strict=True)
    root = _evidence_root(path) if root is None else root.resolve(strict=True)
    document, evidence_fact = _load_json_with_fact(root, path)
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "data_group",
        "execution_manifest",
        "input_manifests",
        "batch_specification",
        "queue_batch",
        "queue_submission_binding",
        "ranking_context",
        "runs",
        "sha256",
    }
    supplied = document.get("sha256")
    body = {key: value for key, value in document.items() if key != "sha256"}
    if set(document) != expected_keys or supplied != _canonical_sha256(body):
        raise ValueError("batch evidence logical SHA-256 differs")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "g3_native500m_batch_evidence"
        or (
            expected_protocol_sha256 is not None
            and document.get("protocol_sha256") != expected_protocol_sha256
        )
    ):
        raise ValueError("batch evidence identity differs")
    context = _AUTHENTICATION_CONTEXT.get()
    assert context is not None
    batch_documents = context["batch_documents"]
    assert isinstance(batch_documents, dict)
    cache_key = (
        root.as_posix(),
        path.as_posix(),
        evidence_fact["size_bytes"],
        evidence_fact["sha256"],
        supplied,
    )
    cached = batch_documents.get(cache_key)
    if cached is not None:
        if cached != document:
            raise ValueError("memoized batch evidence snapshot differs")
        _register_batch_source_facts(root, path, evidence_fact, document)
        return document
    manifest_reference = document["execution_manifest"]
    specification_reference = document["batch_specification"]
    queue_reference = document["queue_batch"]
    for label, reference in (
        ("execution manifest", manifest_reference),
        ("batch specification", specification_reference),
        ("queue batch", queue_reference),
        ("ranking context", document["ranking_context"]),
    ):
        _authenticate_file_fact(root, reference, label)
    submission_reference = document["queue_submission_binding"]
    _authenticate_file_fact(root, submission_reference, "queue submission binding")
    if not isinstance(queue_reference, dict) or not isinstance(
        queue_reference.get("batch_id"), str
    ):
        raise ValueError("batch evidence queue identity differs")
    if not isinstance(manifest_reference, dict) or not isinstance(
        specification_reference, dict
    ):
        raise ValueError("batch evidence source identity differs")
    derived = collect_batch_evidence(
        root=root,
        manifest_path=root / str(manifest_reference["path"]),
        batch_specification_path=root / str(specification_reference["path"]),
        batch_id=str(queue_reference["batch_id"]),
        expected_protocol_sha256=str(document["protocol_sha256"]),
    )
    if derived != document:
        raise ValueError("batch evidence differs from authenticated queue artifacts")
    batch_documents[cache_key] = document
    _register_batch_source_facts(root, path, evidence_fact, document)
    _require_unchanged_fact(root, path, evidence_fact, "batch evidence")
    return document


@_authenticated_operation
def derive_family_selection(
    *,
    family_id: str,
    evidence_paths: tuple[Path, ...],
    predecessor_selection_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.configs.model import (
        G3Representation,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        compile_baseline_rows,
        compile_boundary_rows,
        compile_capacity_first_stage,
        compile_capacity_followup,
        compile_nine_cell_family,
        compile_rq5_frequency_first_stage,
        compile_rq5_frequency_followup,
        compile_rq5_global_rows,
        family_spec,
        required_boundary_extensions,
        select_preliminary_winner,
        select_winner,
    )

    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    if not evidence_paths:
        raise ValueError("family selection requires batch evidence")
    spec = family_spec(family_id)
    if spec.conditional:
        raise ValueError("conditional selection requires a resolved aggregate manifest")
    predecessor_document = None
    authenticated_predecessor = None
    predecessor_representation = None
    predecessor = None
    if family_id == "baseline":
        if predecessor_selection_path is not None:
            raise ValueError("baseline does not accept predecessor selection")
        initial_rows = compile_baseline_rows()
        predecessor_reference = None
    else:
        if predecessor_selection_path is None:
            raise ValueError("family selection requires its predecessor selection")
        predecessor_document, authenticated_predecessor = authenticate_family_selection(
            predecessor_selection_path, root=root
        )
        winner_job = predecessor_document["winner"]["job"]
        if winner_job.get("family_id") != spec.search_predecessor_id:
            raise ValueError("family selection has the wrong predecessor family")
        predecessor = authenticated_predecessor.coordinate
        predecessor_representation = G3Representation.from_dict(
            winner_job["resolved_representation"]
        )
        if spec.design == "nine_cell":
            initial_rows = compile_nine_cell_family(spec, predecessor)
        elif spec.design == "capacity":
            initial_rows = compile_capacity_first_stage(spec, predecessor)
        elif spec.design == "rq5_global":
            initial_rows = compile_rq5_global_rows(predecessor)
        elif spec.design == "rq5_frequency":
            initial_rows = compile_rq5_frequency_first_stage(predecessor)
        else:
            raise ValueError("family selection design is unsupported")
        predecessor_reference = _selection_file_reference(
            predecessor_selection_path,
            predecessor_document,
            "search_predecessor",
            root=root,
        )
    evidence_documents = [
        load_batch_evidence(path, root=root) for path in evidence_paths
    ]
    if len({document["protocol_sha256"] for document in evidence_documents}) != 1:
        raise ValueError("family evidence mixes protocol identities")
    runs = [run for document in evidence_documents for run in document["runs"]]
    if any(not isinstance(run, dict) for run in runs):
        raise ValueError("family evidence contains an invalid run")
    jobs = [run.get("job") for run in runs]
    if any(
        not isinstance(job, dict) or job.get("family_id") != family_id for job in jobs
    ):
        raise ValueError("family evidence contains the wrong role")
    if len({str(job["id"]) for job in jobs}) != len(jobs):
        raise ValueError("family evidence contains duplicate rows")
    for job in jobs:
        _validate_evidence_representation(job, predecessor_representation)
    results_by_id = {
        str(job["id"]): _candidate_result(run, _search_row(job))
        for run, job in zip(runs, jobs, strict=True)
    }
    _require_rows(initial_rows, results_by_id, predecessor_representation)
    expected_rows = list(initial_rows)
    if spec.design == "capacity":
        assert predecessor is not None
        interim = select_preliminary_winner(
            [results_by_id[row.id] for row in initial_rows],
            expected_rows=initial_rows,
            predecessor=authenticated_predecessor,
        )
        followup = compile_capacity_followup(spec, predecessor, interim.row)
        _require_rows(followup, results_by_id, predecessor_representation)
        expected_rows.extend(followup)
    elif spec.design == "rq5_frequency":
        assert predecessor is not None
        interim = select_preliminary_winner(
            [results_by_id[row.id] for row in initial_rows],
            expected_rows=initial_rows,
            predecessor=authenticated_predecessor,
        )
        followup = compile_rq5_frequency_followup(predecessor, interim.row)
        _require_rows(followup, results_by_id, predecessor_representation)
        expected_rows.extend(followup)
    provisional_results = [results_by_id[row.id] for row in expected_rows]
    provisional = select_preliminary_winner(
        provisional_results,
        expected_rows=expected_rows,
        predecessor=authenticated_predecessor,
    )
    requests = required_boundary_extensions(provisional, expected_rows)
    if requests:
        boundary = compile_boundary_rows(
            provisional,
            expected_rows,
            existing_results=provisional_results,
            predecessor=authenticated_predecessor,
            requests=requests,
            round_number=1,
        )
        _require_rows(boundary, results_by_id, predecessor_representation)
        expected_rows.extend(boundary)
    if set(results_by_id) != {row.id for row in expected_rows}:
        raise ValueError("family evidence is incomplete or contains unapproved rows")
    winner = select_winner(
        [results_by_id[row.id] for row in expected_rows],
        expected_rows=expected_rows,
        predecessor=authenticated_predecessor,
    )
    runs_by_id = {str(run["row_id"]): run for run in runs}
    winner_run = runs_by_id[winner.row.id]
    body = {
        "schema_version": 1,
        "kind": "g3_native500m_family_selection",
        "protocol_sha256": evidence_documents[0]["protocol_sha256"],
        "family_id": family_id,
        "predecessor": predecessor_reference,
        "evidence": [
            _selection_file_reference(path, document, "family_evidence", root=root)
            for path, document in zip(evidence_paths, evidence_documents, strict=True)
        ],
        "expected_row_ids": [row.id for row in expected_rows],
        "boundary_requests": [request.__dict__ for request in requests],
        "candidates": [
            {
                "row_id": row.id,
                "selection_metrics": runs_by_id[row.id]["selection_metrics"],
                "metrics": runs_by_id[row.id]["metrics"],
            }
            for row in expected_rows
        ],
        "winner": {
            "row_id": winner.row.id,
            "job": winner_run["job"],
            "selection_metrics": winner_run["selection_metrics"],
            "metrics": winner_run["metrics"],
            "slices": winner_run["slices"],
        },
    }
    return {**body, "sha256": _canonical_sha256(body)}


@_authenticated_operation
def derive_conditional_family_selection(
    *,
    family_id: str,
    evidence_paths: tuple[Path, ...],
    compatibility_state_path: Path,
    root: Path | None = None,
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.configs.model import G3Representation
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        authenticate_resolved_conditional_predecessor,
        compile_boundary_rows,
        compile_nine_cell_family,
        family_spec,
        required_boundary_extensions,
        select_preliminary_winner,
        select_winner,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    if not evidence_paths:
        raise ValueError("conditional family selection requires batch evidence")
    spec = family_spec(family_id)
    if not spec.conditional or spec.design != "nine_cell":
        raise ValueError("conditional family design is unsupported")
    state, authenticated_state = authenticate_compatibility_resolution(
        compatibility_state_path, root=root
    )
    resolved = authenticate_resolved_conditional_predecessor(
        target_family_id=family_id, compatibility_state=authenticated_state
    )
    expected_rows = list(compile_nine_cell_family(spec, resolved.coordinate))
    predecessor = _winner_from_reference(state["most_specific_selection"], root=root)
    predecessor_representation = G3Representation.from_dict(
        predecessor["job"]["resolved_representation"]
    )
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        _conditional_representation,
    )

    expected_representation = _conditional_representation(
        family_id, state=state, root=root
    )
    evidence_documents = [
        load_batch_evidence(path, root=root) for path in evidence_paths
    ]
    if len({document["protocol_sha256"] for document in evidence_documents}) != 1:
        raise ValueError("conditional evidence mixes protocol identities")
    runs = [run for document in evidence_documents for run in document["runs"]]
    jobs = [run.get("job") for run in runs]
    if any(
        not isinstance(job, dict) or job.get("family_id") != family_id for job in jobs
    ):
        raise ValueError("conditional evidence contains the wrong family")
    if len({str(job["id"]) for job in jobs}) != len(jobs):
        raise ValueError("conditional evidence contains duplicate rows")
    for job in jobs:
        _validate_evidence_representation(job, predecessor_representation)
        if job["resolved_representation"] != expected_representation:
            raise ValueError(
                "conditional evidence representation differs from compatibility state"
            )
    results = {
        str(job["id"]): _candidate_result(run, _search_row(job))
        for run, job in zip(runs, jobs, strict=True)
    }
    _require_rows(expected_rows, results, predecessor_representation)
    provisional_results = [results[row.id] for row in expected_rows]
    provisional = select_preliminary_winner(
        provisional_results, expected_rows=expected_rows, predecessor=resolved
    )
    requests = required_boundary_extensions(provisional, expected_rows)
    if requests:
        boundary = compile_boundary_rows(
            provisional,
            expected_rows,
            existing_results=provisional_results,
            predecessor=resolved,
            requests=requests,
            round_number=1,
        )
        _require_rows(boundary, results, predecessor_representation)
        expected_rows.extend(boundary)
    if set(results) != {row.id for row in expected_rows}:
        raise ValueError(
            "conditional evidence is incomplete or contains unapproved rows"
        )
    winner = select_winner(
        [results[row.id] for row in expected_rows],
        expected_rows=expected_rows,
        predecessor=resolved,
    )
    runs_by_id = {str(run["row_id"]): run for run in runs}
    winner_run = runs_by_id[winner.row.id]
    body = {
        "schema_version": 1,
        "kind": "g3_native500m_family_selection",
        "protocol_sha256": evidence_documents[0]["protocol_sha256"],
        "family_id": family_id,
        "predecessor": state["most_specific_selection"],
        "compatibility_state": _authenticated_compatibility_reference(
            compatibility_state_path,
            state,
            authenticated_state,
            "compatibility_state",
            root=root,
        ),
        "evidence": [
            _selection_file_reference(path, document, "family_evidence", root=root)
            for path, document in zip(evidence_paths, evidence_documents, strict=True)
        ],
        "expected_row_ids": [row.id for row in expected_rows],
        "boundary_requests": [request.__dict__ for request in requests],
        "candidates": [
            {
                "row_id": row.id,
                "selection_metrics": runs_by_id[row.id]["selection_metrics"],
                "metrics": runs_by_id[row.id]["metrics"],
            }
            for row in expected_rows
        ],
        "winner": {
            "row_id": winner.row.id,
            "job": winner_run["job"],
            "selection_metrics": winner_run["selection_metrics"],
            "metrics": winner_run["metrics"],
            "slices": winner_run["slices"],
        },
    }
    return {**body, "sha256": _canonical_sha256(body)}


def persist_family_selection(path: Path, document: Mapping[str, object]) -> Path:
    body = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _canonical_sha256(body):
        raise ValueError("family selection logical SHA-256 differs")
    _write_immutable(path, _canonical_bytes(document) + b"\n")
    return path


@_authenticated_operation
def load_family_selection(path: Path, *, root: Path | None = None) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        PROTOCOL_SHA256,
        family_spec,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    document = _load_json(path)
    body = {key: value for key, value in document.items() if key != "sha256"}
    if (
        document.get("kind") != "g3_native500m_family_selection"
        or document.get("schema_version") != 1
        or document.get("protocol_sha256") != PROTOCOL_SHA256
        or document.get("sha256") != _canonical_sha256(body)
        or not isinstance(document.get("evidence"), list)
    ):
        raise ValueError("family selection identity differs")
    for reference in document["evidence"]:
        _authenticate_selection_reference(path, reference, root=root)
    predecessor = document.get("predecessor")
    spec = family_spec(str(document.get("family_id")))
    compatibility_state = document.get("compatibility_state")
    if spec.conditional:
        _authenticate_row_selection_reference(predecessor, root=root)
        if not isinstance(compatibility_state, dict):
            raise ValueError("conditional selection has no compatibility state")
        _authenticate_selection_reference(path, compatibility_state, root=root)
    elif compatibility_state is not None:
        raise ValueError("standalone selection has a compatibility state")
    elif predecessor is not None:
        _authenticate_selection_reference(path, predecessor, root=root)
    evidence_paths = tuple(
        root / str(reference["path"])
        for reference in document["evidence"]
        if isinstance(reference, dict) and reference.get("role") == "family_evidence"
    )
    predecessor_path = None if predecessor is None else root / str(predecessor["path"])
    if spec.conditional:
        derived = derive_conditional_family_selection(
            family_id=str(document.get("family_id")),
            evidence_paths=evidence_paths,
            compatibility_state_path=root / str(compatibility_state["path"]),
            root=root,
        )
    else:
        derived = derive_family_selection(
            family_id=str(document.get("family_id")),
            evidence_paths=evidence_paths,
            predecessor_selection_path=predecessor_path,
            root=root,
        )
    if derived != document:
        raise ValueError("family selection differs from complete protocol evidence")
    return document


@_authenticated_operation
def authenticate_family_selection(
    path: Path, *, root: Path | None = None
) -> tuple[dict[str, object], object]:
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        authenticate_selected_coordinate,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    document = load_family_selection(path, root=root)
    predecessor_reference = document.get("predecessor")
    compatibility_state = document.get("compatibility_state")
    authenticated_predecessor = None
    if compatibility_state is not None:
        from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
            authenticate_resolved_conditional_predecessor,
        )

        state = authenticate_compatibility_resolution(
            root / str(compatibility_state["path"]), root=root
        )[1]
        authenticated_predecessor = authenticate_resolved_conditional_predecessor(
            target_family_id=str(document["family_id"]),
            compatibility_state=state,
        )
    elif predecessor_reference is not None:
        authenticated_predecessor = authenticate_family_selection(
            root / str(predecessor_reference["path"]), root=root
        )[1]
    runs = [
        run
        for reference in document["evidence"]
        for run in load_batch_evidence(root / str(reference["path"]), root=root)["runs"]
    ]
    runs_by_id = {str(run["row_id"]): run for run in runs}
    row_ids = document["expected_row_ids"]
    if (
        not isinstance(row_ids, list)
        or set(runs_by_id) != set(row_ids)
        or document["winner"]["row_id"] not in runs_by_id
    ):
        raise ValueError("family selection cannot be authenticated")
    rows = tuple(_search_row(runs_by_id[str(row_id)]["job"]) for row_id in row_ids)
    results = tuple(_candidate_result(runs_by_id[row.id], row) for row in rows)
    authenticated = authenticate_selected_coordinate(
        results,
        expected_rows=rows,
        predecessor=authenticated_predecessor,
    )
    if authenticated.selected_result.row.id != document["winner"]["row_id"]:
        raise ValueError("authenticated family winner differs")
    return document, authenticated


@_authenticated_operation
def derive_continuation_authorization(
    *,
    family_id: str,
    continuation: str,
    evidence_paths: tuple[Path, ...],
    predecessor_selection_path: Path | None,
    root: Path | None = None,
) -> tuple[dict[str, object], tuple[object, ...], dict[str, object]]:
    from experiments.g3_pretrained_item_embeddings.configs.model import G3Representation
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        compile_boundary_rows,
        compile_capacity_first_stage,
        compile_capacity_followup,
        compile_nine_cell_family,
        compile_rq5_frequency_first_stage,
        compile_rq5_frequency_followup,
        compile_rq5_global_rows,
        family_spec,
        required_boundary_extensions,
        select_preliminary_winner,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    if continuation not in {"followup", "boundary"} or not evidence_paths:
        raise ValueError("continuation authorization request is invalid")
    spec = family_spec(family_id)
    if spec.conditional:
        raise ValueError("conditional families use compatibility resolution")
    if family_id == "baseline":
        if predecessor_selection_path is not None:
            raise ValueError("baseline continuation has no predecessor selection")
        from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
            compile_baseline_rows,
        )

        predecessor_document = None
        authenticated_predecessor = None
        predecessor = None
        predecessor_representation = G3Representation(item_id_tying="tied")
        initial = compile_baseline_rows()
    else:
        if predecessor_selection_path is None:
            raise ValueError("non-baseline continuation requires a predecessor")
        predecessor_document, authenticated_predecessor = authenticate_family_selection(
            predecessor_selection_path, root=root
        )
        if predecessor_document["family_id"] != spec.search_predecessor_id:
            raise ValueError("continuation has the wrong authenticated predecessor")
        predecessor = authenticated_predecessor.coordinate
        predecessor_representation = G3Representation.from_dict(
            predecessor_document["winner"]["job"]["resolved_representation"]
        )
    if family_id == "baseline":
        pass
    elif spec.design == "nine_cell":
        initial = compile_nine_cell_family(spec, predecessor)
    elif spec.design == "capacity":
        initial = compile_capacity_first_stage(spec, predecessor)
    elif spec.design == "rq5_global":
        initial = compile_rq5_global_rows(predecessor)
    elif spec.design == "rq5_frequency":
        initial = compile_rq5_frequency_first_stage(predecessor)
    else:
        raise ValueError("continuation family design is unsupported")
    documents = [load_batch_evidence(path, root=root) for path in evidence_paths]
    runs = [run for document in documents for run in document["runs"]]
    if any(run.get("family_id") != family_id for run in runs):
        raise ValueError("continuation evidence contains another family")
    jobs = [run["job"] for run in runs]
    for job in jobs:
        _validate_evidence_representation(job, predecessor_representation)
    results = {
        str(run["row_id"]): _candidate_result(run, _search_row(run["job"]))
        for run in runs
    }
    if len(results) != len(runs):
        raise ValueError("continuation evidence contains duplicate rows")
    _require_rows(initial, results, predecessor_representation)
    initial_winner = select_preliminary_winner(
        [results[row.id] for row in initial],
        expected_rows=initial,
        predecessor=authenticated_predecessor,
    )
    family_rows = list(initial)
    if spec.design == "capacity":
        family_rows.extend(
            compile_capacity_followup(spec, predecessor, initial_winner.row)
        )
    elif spec.design == "rq5_frequency":
        family_rows.extend(
            compile_rq5_frequency_followup(predecessor, initial_winner.row)
        )
    if continuation == "followup":
        if len(family_rows) == len(initial):
            raise ValueError("family has no compiler-required followup")
        if set(results) != {row.id for row in initial}:
            raise ValueError("followup requires exactly the complete initial evidence")
        continuation_rows = tuple(family_rows[len(initial) :])
        selected = initial_winner
        requests = ()
        source_rows = tuple(initial)
    else:
        _require_rows(family_rows, results, predecessor_representation)
        if set(results) != {row.id for row in family_rows}:
            raise ValueError("boundary requires exactly the complete family evidence")
        source_rows = tuple(family_rows)
        selected = select_preliminary_winner(
            [results[row.id] for row in source_rows],
            expected_rows=source_rows,
            predecessor=authenticated_predecessor,
        )
        requests = required_boundary_extensions(selected, source_rows)
        if not requests:
            raise ValueError("family is already resolved and needs no boundary")
        continuation_rows = compile_boundary_rows(
            selected,
            source_rows,
            existing_results=[results[row.id] for row in source_rows],
            predecessor=authenticated_predecessor,
            requests=requests,
        )
    body = {
        "schema_version": 1,
        "kind": "g3_native500m_continuation_authorization",
        "protocol_sha256": documents[0]["protocol_sha256"],
        "family_id": family_id,
        "continuation": continuation,
        "predecessor": (
            None
            if predecessor_document is None
            else _selection_file_reference(
                predecessor_selection_path,
                predecessor_document,
                "search_predecessor",
                root=root,
            )
        ),
        "evidence": [
            _selection_file_reference(path, document, "family_evidence", root=root)
            for path, document in zip(evidence_paths, documents, strict=True)
        ],
        "source_row_ids": [row.id for row in source_rows],
        "selected_row_id": selected.row.id,
        "boundary_requests": [request.__dict__ for request in requests],
        "continuation_rows": [row.to_dict() for row in continuation_rows],
    }
    document = {**body, "sha256": _canonical_sha256(body)}
    return document, continuation_rows, predecessor_representation.to_dict()


def persist_continuation_authorization(
    path: Path, document: Mapping[str, object]
) -> Path:
    body = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _canonical_sha256(body):
        raise ValueError("continuation authorization logical SHA-256 differs")
    _write_immutable(path, _canonical_bytes(document) + b"\n")
    return path


@_authenticated_operation
def load_continuation_authorization(
    path: Path, *, root: Path | None = None
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    document = _load_json(path)
    body = {key: value for key, value in document.items() if key != "sha256"}
    if (
        document.get("kind") != "g3_native500m_continuation_authorization"
        or document.get("schema_version") != 1
        or document.get("sha256") != _canonical_sha256(body)
    ):
        raise ValueError("continuation authorization identity differs")
    predecessor = document.get("predecessor")
    evidence = document.get("evidence")
    if predecessor is not None and not isinstance(predecessor, dict):
        raise ValueError("continuation authorization predecessor differs")
    if not isinstance(evidence, list):
        raise ValueError("continuation authorization references differ")
    if predecessor is not None:
        _authenticate_selection_reference(path, predecessor, root=root)
    for reference in evidence:
        _authenticate_selection_reference(path, reference, root=root)
    derived, _, _ = derive_continuation_authorization(
        family_id=str(document.get("family_id")),
        continuation=str(document.get("continuation")),
        evidence_paths=tuple(root / str(reference["path"]) for reference in evidence),
        predecessor_selection_path=(
            None if predecessor is None else root / str(predecessor["path"])
        ),
        root=root,
    )
    if derived != document:
        raise ValueError("continuation authorization differs from protocol evidence")
    return document


@_authenticated_operation
def derive_conditional_boundary_authorization(
    *,
    family_id: str,
    evidence_paths: tuple[Path, ...],
    compatibility_state_path: Path,
    root: Path | None = None,
) -> tuple[dict[str, object], tuple[object, ...], dict[str, object]]:
    from experiments.g3_pretrained_item_embeddings.configs.model import G3Representation
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        authenticate_resolved_conditional_predecessor,
        compile_boundary_rows,
        compile_nine_cell_family,
        family_spec,
        required_boundary_extensions,
        select_preliminary_winner,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    spec = family_spec(family_id)
    if not spec.conditional or spec.design != "nine_cell" or not evidence_paths:
        raise ValueError("conditional boundary request is invalid")
    state, authenticated_state = authenticate_compatibility_resolution(
        compatibility_state_path, root=root
    )
    resolved = authenticate_resolved_conditional_predecessor(
        target_family_id=family_id, compatibility_state=authenticated_state
    )
    initial = tuple(compile_nine_cell_family(spec, resolved.coordinate))
    documents = [load_batch_evidence(path, root=root) for path in evidence_paths]
    runs = [run for document in documents for run in document["runs"]]
    jobs = [run.get("job") for run in runs]
    if len({document["protocol_sha256"] for document in documents}) != 1 or any(
        not isinstance(job, dict) or job.get("family_id") != family_id for job in jobs
    ):
        raise ValueError("conditional boundary evidence identity differs")
    if {str(job["id"]) for job in jobs} != {row.id for row in initial} or len(
        jobs
    ) != len(initial):
        raise ValueError(
            "conditional boundary requires exactly complete initial evidence"
        )
    predecessor = _winner_from_reference(state["most_specific_selection"], root=root)
    predecessor_representation = G3Representation.from_dict(
        predecessor["job"]["resolved_representation"]
    )
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        _conditional_representation,
    )

    expected_representation = _conditional_representation(
        family_id, state=state, root=root
    )
    for job in jobs:
        _validate_evidence_representation(job, predecessor_representation)
        if job["resolved_representation"] != expected_representation:
            raise ValueError(
                "conditional evidence representation differs from compatibility state"
            )
    results = {
        str(job["id"]): _candidate_result(run, _search_row(job))
        for run, job in zip(runs, jobs, strict=True)
    }
    _require_rows(initial, results, predecessor_representation)
    provisional = select_preliminary_winner(
        [results[row.id] for row in initial],
        expected_rows=initial,
        predecessor=resolved,
    )
    requests = required_boundary_extensions(provisional, initial)
    if not requests:
        raise ValueError("conditional winner is interior and requires no continuation")
    boundary = tuple(
        compile_boundary_rows(
            provisional,
            initial,
            existing_results=[results[row.id] for row in initial],
            predecessor=resolved,
            requests=requests,
            round_number=1,
        )
    )
    state_reference = _authenticated_compatibility_reference(
        compatibility_state_path,
        state,
        authenticated_state,
        "compatibility_state",
        root=root,
    )
    body = {
        "schema_version": 1,
        "kind": "g3_native500m_conditional_boundary_authorization",
        "protocol_sha256": documents[0]["protocol_sha256"],
        "family_id": family_id,
        "compatibility_state": state_reference,
        "evidence": [
            _selection_file_reference(path, document, "family_evidence", root=root)
            for path, document in zip(evidence_paths, documents, strict=True)
        ],
        "selected_row_id": provisional.row.id,
        "boundary_requests": [request.__dict__ for request in requests],
        "boundary_rows": [row.to_dict() for row in boundary],
    }
    return (
        {**body, "sha256": _canonical_sha256(body)},
        boundary,
        predecessor_representation.to_dict(),
    )


def persist_conditional_boundary_authorization(
    path: Path, document: Mapping[str, object]
) -> Path:
    body = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _canonical_sha256(body):
        raise ValueError("conditional boundary authorization logical SHA-256 differs")
    _write_immutable(path, _canonical_bytes(document) + b"\n")
    return path


@_authenticated_operation
def load_conditional_boundary_authorization(
    path: Path, *, root: Path | None = None
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    document = _load_json(path)
    body = {key: value for key, value in document.items() if key != "sha256"}
    if (
        document.get("kind") != "g3_native500m_conditional_boundary_authorization"
        or document.get("schema_version") != 1
        or document.get("sha256") != _canonical_sha256(body)
        or not isinstance(document.get("evidence"), list)
        or not isinstance(document.get("compatibility_state"), dict)
    ):
        raise ValueError("conditional boundary authorization identity differs")
    for reference in document["evidence"]:
        _authenticate_selection_reference(path, reference, root=root)
    _authenticate_selection_reference(path, document["compatibility_state"], root=root)
    derived, _, _ = derive_conditional_boundary_authorization(
        family_id=str(document.get("family_id")),
        evidence_paths=tuple(
            root / str(reference["path"]) for reference in document["evidence"]
        ),
        compatibility_state_path=(root / str(document["compatibility_state"]["path"])),
        root=root,
    )
    if derived != document:
        raise ValueError("conditional boundary authorization differs from evidence")
    return document


@_authenticated_operation
def derive_compatibility_resolution(
    *, selection_paths: Mapping[str, Path], root: Path | None = None
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.analysis import native500m_report
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    expected = {
        "baseline",
        "untied_control",
        "rq1_content_input",
        "rq2_content_concat",
        "rq3_output_learned",
        "rq3_output_frozen_content",
        "rq3_output_trainable_content",
        "rq3_output_learned_frozen_content",
        "rq3_output_learned_trainable_content",
        "rq4_artist",
        "rq4_album",
        "rq4_artist_album",
        "rq5_global_gate",
        "rq5_frequency_gate",
    }
    if set(selection_paths) != expected:
        raise ValueError("compatibility resolution requires every standalone selection")
    selections = {
        family: authenticate_family_selection(path, root=root)[0]
        for family, path in selection_paths.items()
    }
    baseline = selections["baseline"]["winner"]
    relative = native500m_report._relative_noise_bands()
    thresholds = {
        metric: float(baseline["metrics"][metric]) * relative[metric]
        for metric in ("recall@100", "ndcg@100")
    }
    baseline_tail = native500m_report._tail_recall(baseline)
    if baseline_tail is None:
        raise ValueError("compatibility baseline lacks tail evidence")
    tail_band = baseline_tail * relative["recall@100"]
    rq1_families = ("baseline", "untied_control", "rq1_content_input")
    rq1_rows = [baseline, *(selections[name]["winner"] for name in rq1_families[1:])]
    rq1 = native500m_report._approved_winner_index(
        1, rq1_rows, rq1_families, thresholds, tail_band
    )
    rq2_rows = [baseline, selections["rq2_content_concat"]["winner"]]
    rq2 = native500m_report._approved_winner_index(
        2, rq2_rows, ("baseline", "rq2_content_concat"), thresholds, tail_band
    )
    rq5_families = (
        "baseline",
        "rq2_content_concat",
        "rq5_global_gate",
        "rq5_frequency_gate",
    )
    rq5_rows = [baseline, *(selections[name]["winner"] for name in rq5_families[1:])]
    rq5 = native500m_report._approved_winner_index(
        5, rq5_rows, rq5_families, thresholds, tail_band
    )
    input_candidates = []
    if rq1:
        input_candidates.append(rq1_rows[rq1])
    if rq2:
        input_candidates.append(rq2_rows[rq2])
    if rq5:
        input_candidates.append(rq5_rows[rq5])
    if input_candidates:
        selected_input = input_candidates[
            native500m_report._band_aware_choice(
                input_candidates,
                tuple(range(len(input_candidates))),
                thresholds["recall@100"],
            )
        ]
    else:
        selected_input = baseline
    rq3_names = native500m_report._RQ_FAMILIES[3]
    rq3_rows = [baseline, *(selections[name]["winner"] for name in rq3_names)]
    rq3 = native500m_report._approved_winner_index(
        3, rq3_rows, ("baseline", *rq3_names), thresholds, tail_band
    )
    learned_index = ("baseline", *rq3_names).index("rq3_output_learned")
    selected_output = None if rq3 == learned_index else rq3_rows[rq3]
    rq4_names = native500m_report._RQ_FAMILIES[4]
    rq4_rows = [baseline, *(selections[name]["winner"] for name in rq4_names)]
    rq4 = native500m_report._approved_winner_index(
        4, rq4_rows, ("baseline", *rq4_names), thresholds, tail_band
    )
    selected_metadata = None if rq4 == 0 else rq4_rows[rq4]
    selected_by_row = {
        selection["winner"]["row_id"]: (family, selection)
        for family, selection in selections.items()
    }

    def reference(row: Mapping[str, object], role: str) -> dict[str, object]:
        family, selection = selected_by_row[str(row["row_id"])]
        return _selection_file_reference(
            selection_paths[family], selection, role, root=root
        ) | {"row_id": row["row_id"]}

    targets = {
        "input": reference(selected_input, "component_target_input"),
        "output": (
            None
            if selected_output is None
            else reference(selected_output, "component_target_output")
        ),
        "metadata": (
            None
            if selected_metadata is None
            else reference(selected_metadata, "component_target_metadata")
        ),
    }
    included = {"input": targets["input"], "output": None, "metadata": None}
    most_specific = targets["input"]
    if (
        targets["output"] is not None
        and selected_input["job"]["family_id"] == "rq2_content_concat"
    ):
        included["output"] = targets["output"]
        most_specific = targets["output"]
    if (
        targets["metadata"] is not None
        and selected_input["job"]["family_id"] == "baseline"
        and targets["output"] is None
    ):
        included["metadata"] = targets["metadata"]
        most_specific = targets["metadata"]
    omissions = [
        {"component": component, "reason": "standalone_not_qualified"}
        for component in ("output", "metadata")
        if targets[component] is None
    ]
    selected_component_rows = {
        str(value["row_id"]): component
        for component, value in targets.items()
        if isinstance(value, dict)
    }
    comparator_families = {
        "untied_control": ("baseline",),
        "rq1_content_input": ("baseline",),
        "rq2_content_concat": ("baseline",),
        "rq3_output_learned": ("rq2_content_concat",),
        "rq3_output_frozen_content": ("rq3_output_learned",),
        "rq3_output_trainable_content": ("rq3_output_learned",),
        "rq3_output_learned_frozen_content": ("rq3_output_learned",),
        "rq3_output_learned_trainable_content": ("rq3_output_learned",),
        "rq4_artist": ("baseline",),
        "rq4_album": ("baseline",),
        "rq4_artist_album": ("baseline",),
        "rq5_global_gate": ("baseline", "rq2_content_concat"),
        "rq5_frequency_gate": ("baseline", "rq2_content_concat", "rq5_global_gate"),
    }
    eligible_families = set()
    if rq1:
        eligible_families.add("rq1_content_input")
    if rq2:
        eligible_families.add("rq2_content_concat")
    learned_output_recall = float(
        selections["rq3_output_learned"]["winner"]["metrics"]["recall@100"]
    )
    for family in rq3_names:
        if (
            family != "rq3_output_learned"
            and float(selections[family]["winner"]["metrics"]["recall@100"])
            > learned_output_recall + thresholds["recall@100"]
        ):
            eligible_families.add(family)
    baseline_recall = float(baseline["metrics"]["recall@100"])
    for family in rq4_names:
        row = selections[family]["winner"]
        recall = float(row["metrics"]["recall@100"])
        tail = native500m_report._tail_recall(row)
        if recall > baseline_recall + thresholds["recall@100"] or (
            recall >= baseline_recall - thresholds["recall@100"]
            and tail is not None
            and tail > baseline_tail + tail_band
        ):
            eligible_families.add(family)
    rq5_recall = {
        family: float(selections[family]["winner"]["metrics"]["recall@100"])
        for family in rq5_families
    }
    if (
        rq5_recall["rq5_global_gate"]
        > max(rq5_recall["baseline"], rq5_recall["rq2_content_concat"])
        + thresholds["recall@100"]
    ):
        eligible_families.add("rq5_global_gate")
    frequency_row = selections["rq5_frequency_gate"]["winner"]
    frequency_tail = native500m_report._tail_recall(frequency_row)
    frequency_comparator_tails = [
        native500m_report._tail_recall(selections[family]["winner"])
        for family in rq5_families[:-1]
    ]
    if (
        all(
            rq5_recall["rq5_frequency_gate"]
            >= rq5_recall[family] - thresholds["recall@100"]
            for family in rq5_families[:-1]
        )
        and frequency_tail is not None
        and all(
            value is not None and frequency_tail > value + tail_band
            for value in frequency_comparator_tails
        )
    ):
        eligible_families.add("rq5_frequency_gate")
    standalone_decisions = []
    for family in sorted(expected):
        winner = selections[family]["winner"]
        comparators = comparator_families.get(family, ())
        component = selected_component_rows.get(str(winner["row_id"]))
        if family == "baseline":
            status, reason = "included", "reader_baseline"
        elif component is not None:
            status, reason = "included", f"selected_{component}_component"
        elif family == "untied_control":
            status, reason = "omitted", "secondary_mechanism_control"
        elif family == "rq3_output_learned":
            status, reason = "omitted", "scientific_chain_control"
        elif family in eligible_families:
            status, reason = "omitted", "eligible_but_superseded"
        else:
            status, reason = "omitted", "approved_gate_failed"
        winner_recall = float(winner["metrics"]["recall@100"])
        winner_tail = native500m_report._tail_recall(winner)
        comparisons = []
        for comparator in comparators:
            comparator_row = selections[comparator]["winner"]
            comparator_recall = float(comparator_row["metrics"]["recall@100"])
            comparator_tail = native500m_report._tail_recall(comparator_row)
            comparisons.append(
                {
                    "family_id": comparator,
                    "recall@100": comparator_recall,
                    "recall_delta": winner_recall - comparator_recall,
                    "tail_recall@100": comparator_tail,
                    "tail_recall_delta": (
                        None
                        if winner_tail is None or comparator_tail is None
                        else winner_tail - comparator_tail
                    ),
                }
            )
        standalone_decisions.append(
            {
                "family_id": family,
                "selection": reference(winner, "standalone_decision"),
                "status": status,
                "reason": reason,
                "component": component,
                "predecessor": selections[family].get("predecessor"),
                "eligible": family in eligible_families,
                "recall@100": winner_recall,
                "tail_recall@100": winner_tail,
                "comparisons": comparisons,
                "applicable_bands": {
                    "recall@100": thresholds["recall@100"],
                    "tail_recall@100": (
                        tail_band
                        if family
                        in {
                            "rq1_content_input",
                            "rq4_artist",
                            "rq4_album",
                            "rq4_artist_album",
                            "rq5_frequency_gate",
                        }
                        else None
                    ),
                },
            }
        )
    body = {
        "schema_version": 1,
        "kind": "g3_native500m_compatibility_state",
        "protocol_sha256": baseline["job"].get(
            "protocol_sha256", selections["baseline"]["protocol_sha256"]
        ),
        "generation": 0,
        "standalone_selections": {
            family: _selection_file_reference(
                path, selections[family], "family_evidence", root=root
            )
            | {"row_id": selections[family]["winner"]["row_id"]}
            for family, path in selection_paths.items()
        },
        "standalone_decisions": standalone_decisions,
        "component_targets": targets,
        "prior_state": None,
        "completed_transition": None,
        "gate_thresholds": {
            "recall@100": thresholds["recall@100"],
            "tail_recall@100": tail_band,
        },
        "included": included,
        "omissions": omissions,
        "most_specific_selection": most_specific,
        "next_conditional_family": _next_conditional_family(
            targets, included, omissions
        ),
    }
    return {**body, "sha256": _canonical_sha256(body)}


@_authenticated_operation
def derive_compatibility_transition(
    *,
    prior_resolution_path: Path,
    completed_selection_path: Path,
    root: Path | None = None,
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.analysis import native500m_report
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    prior, prior_authenticated = authenticate_compatibility_resolution(
        prior_resolution_path, root=root
    )
    family_id = prior.get("next_conditional_family")
    if family_id not in {"bridge_rq3_output", "bridge_rq4_metadata", "aggregate"}:
        raise ValueError("compatibility state authorizes no conditional family")
    completed, _ = authenticate_family_selection(completed_selection_path, root=root)
    if completed.get("family_id") != family_id:
        raise ValueError("completed conditional family differs from authorization")
    completed_state = completed.get("compatibility_state")
    expected_prior = _authenticated_compatibility_reference(
        prior_resolution_path,
        prior,
        prior_authenticated,
        "compatibility_state",
        root=root,
    )
    if completed_state != expected_prior:
        raise ValueError(
            "completed conditional selection used another compatibility state"
        )
    winner = completed["winner"]
    predecessor_reference = prior["most_specific_selection"]
    predecessor = _winner_from_reference(predecessor_reference, root=root)
    recall_band = float(prior["gate_thresholds"]["recall@100"])
    tail_band = float(prior["gate_thresholds"]["tail_recall@100"])
    marginal_recall = float(winner["metrics"]["recall@100"]) - float(
        predecessor["metrics"]["recall@100"]
    )
    predecessor_tail = native500m_report._tail_recall(predecessor)
    winner_tail = native500m_report._tail_recall(winner)
    tail_eligible = (
        family_id == "bridge_rq4_metadata"
        and marginal_recall >= -recall_band
        and predecessor_tail is not None
        and winner_tail is not None
        and winner_tail > predecessor_tail + tail_band
    )
    decision = "accept" if marginal_recall > recall_band or tail_eligible else "omit"
    component = {
        "bridge_rq3_output": "output",
        "bridge_rq4_metadata": "metadata",
        "aggregate": "aggregate",
    }[family_id]
    completed_reference = _selection_file_reference(
        completed_selection_path, completed, "conditional_result", root=root
    ) | {"row_id": winner["row_id"]}
    included = dict(prior["included"])
    omissions = list(prior["omissions"])
    most_specific = predecessor_reference
    if decision == "accept":
        if component == "aggregate":
            included = dict(prior["component_targets"])
        else:
            included[component] = prior["component_targets"][component]
        most_specific = completed_reference
    else:
        omissions.append(
            {
                "component": component,
                "reason": "conditional_marginal_gate_failed",
                "target": (
                    None
                    if component == "aggregate"
                    else prior["component_targets"][component]
                ),
                "conditional_selection": completed_reference,
            }
        )
    transition = {
        "family_id": family_id,
        "selected_family_id": winner["job"]["family_id"],
        "selected_selection": completed_reference,
        "selected_row_id": winner["row_id"],
        "selected_job": winner["job"],
        "selection_metrics": winner["selection_metrics"],
        "metrics": winner["metrics"],
        "predecessor_reference": predecessor_reference,
        "marginal_recall@100": marginal_recall,
        "gate_thresholds": prior["gate_thresholds"],
        "decision": decision,
    }
    body = {
        "schema_version": 1,
        "kind": "g3_native500m_compatibility_state",
        "protocol_sha256": prior["protocol_sha256"],
        "generation": int(prior["generation"]) + 1,
        "standalone_selections": prior["standalone_selections"],
        "standalone_decisions": prior["standalone_decisions"],
        "component_targets": prior["component_targets"],
        "prior_state": expected_prior,
        "completed_transition": transition,
        "gate_thresholds": prior["gate_thresholds"],
        "included": included,
        "omissions": omissions,
        "most_specific_selection": most_specific,
        "next_conditional_family": _next_conditional_family(
            prior["component_targets"], included, omissions
        ),
    }
    return {**body, "sha256": _canonical_sha256(body)}


def persist_compatibility_resolution(
    path: Path, document: Mapping[str, object]
) -> Path:
    body = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _canonical_sha256(body):
        raise ValueError("compatibility resolution logical SHA-256 differs")
    _write_immutable(path, _canonical_bytes(document) + b"\n")
    return path


@_authenticated_operation
def load_compatibility_resolution(
    path: Path, *, root: Path | None = None
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    document = _load_json(path)
    if document.get("kind") != "g3_native500m_compatibility_state":
        raise ValueError("compatibility resolution identity differs")
    generation = document.get("generation")
    if type(generation) is not int or generation < 0:
        raise ValueError("compatibility resolution generation differs")
    selections = document.get("standalone_selections")
    if not isinstance(selections, dict):
        raise ValueError("compatibility resolution selections differ")
    for reference in selections.values():
        _authenticate_row_selection_reference(reference, root=root)
    if generation == 0:
        derived = derive_compatibility_resolution(
            selection_paths={
                family: root / str(reference["path"])
                for family, reference in selections.items()
            },
            root=root,
        )
    else:
        prior = document.get("prior_state")
        transition = document.get("completed_transition")
        if not isinstance(prior, dict) or not isinstance(transition, dict):
            raise ValueError("compatibility transition references differ")
        _authenticate_selection_reference(path, prior, root=root)
        selection = transition.get("selected_selection")
        if not isinstance(selection, dict):
            raise ValueError("compatibility transition selection differs")
        _authenticate_row_selection_reference(selection, root=root)
        derived = derive_compatibility_transition(
            prior_resolution_path=root / str(prior["path"]),
            completed_selection_path=root / str(selection["path"]),
            root=root,
        )
    if derived != document:
        raise ValueError("compatibility resolution differs from selected evidence")
    return document


@_authenticated_operation
def authenticate_compatibility_resolution(
    path: Path, *, root: Path | None = None
) -> tuple[dict[str, object], object]:
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        authenticate_compatibility_state,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    document = load_compatibility_resolution(path, root=root)
    return document, authenticate_compatibility_state(path, root=root)


def _next_conditional_family(
    targets: Mapping[str, object],
    included: Mapping[str, object],
    omissions: object,
) -> str | None:
    if not isinstance(omissions, list):
        raise ValueError("compatibility omissions differ")
    terminal = {
        str(value.get("component"))
        for value in omissions
        if isinstance(value, dict)
        and value.get("reason") == "conditional_marginal_gate_failed"
    }
    if (
        targets.get("output") is not None
        and included.get("output") is None
        and "output" not in terminal
    ):
        return "bridge_rq3_output"
    if (
        targets.get("metadata") is not None
        and included.get("metadata") is None
        and "metadata" not in terminal
    ):
        return "bridge_rq4_metadata"
    return None


def _winner_from_reference(reference: object, *, root: Path) -> dict[str, object]:
    _authenticate_row_selection_reference(reference, root=root)
    assert isinstance(reference, dict)
    document = authenticate_family_selection(root / str(reference["path"]), root=root)[
        0
    ]
    winner = document["winner"]
    if winner["row_id"] != reference["row_id"]:
        raise ValueError("compatibility row reference differs from selected winner")
    return winner


def _authenticate_row_selection_reference(reference: object, *, root: Path) -> None:
    if not isinstance(reference, dict) or set(reference) != {
        "role",
        "path",
        "size_bytes",
        "sha256",
        "logical_sha256",
        "row_id",
    }:
        raise ValueError("compatibility row reference schema differs")
    generic = {key: value for key, value in reference.items() if key != "row_id"}
    _authenticate_selection_reference(root / str(reference["path"]), generic, root=root)


def _search_row(job: Mapping[str, object]):
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import SearchRow

    return SearchRow(
        id=str(job["id"]),
        family_id=str(job["family_id"]),
        family_code=int(job["family_code"]),
        research_question=str(job["research_question"]),
        predecessor_id=str(job["predecessor_id"]),
        promotion_predecessor_id=str(job["promotion_predecessor_id"]),
        manifest_order=int(job["manifest_order"]),
        stage=job["stage"],
        batch_size=int(job["batch_size"]),
        seed=int(job["seed"]),
        horizon_epochs=int(job["horizon_epochs"]),
        embedding_learning_rate_text=str(job["embedding_learning_rate"]),
        deep_learning_rate_text=str(job["deep_learning_rate"]),
        anchor_embedding_learning_rate_text=str(job["anchor_embedding_learning_rate"]),
        anchor_deep_learning_rate_text=str(job["anchor_deep_learning_rate"]),
        capacity=job["capacity"],
    )


def _candidate_result(run: Mapping[str, object], row: Any):
    from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
        CandidateResult,
    )

    selection = run["selection_metrics"]
    restored = run["restored_checkpoint"]
    return CandidateResult(
        row=row,
        recall_at_100=float(selection["recall@100"]),
        ndcg_at_100=float(selection["ndcg@100"]),
        best_epoch=int(restored["best_epoch"]),
        epochs_trained=row.horizon_epochs,
    )


def _require_rows(
    rows: Any,
    results_by_id: Mapping[str, object],
    predecessor_representation: object,
) -> None:
    del predecessor_representation
    for row in rows:
        result = results_by_id.get(row.id)
        if result is None or result.row != row:
            raise ValueError("family evidence is missing a compiler-derived row")


def _validate_evidence_representation(
    job: Mapping[str, object], predecessor_representation: object
) -> None:
    from experiments.g3_pretrained_item_embeddings.configs.model import (
        G3Representation,
        _validate_native500m_family,
    )

    representation = G3Representation.from_dict(job["resolved_representation"])
    _validate_native500m_family(str(job["family_id"]), job["capacity"], representation)
    if str(job["family_id"]).startswith("rq3_") or str(job["family_id"]).startswith(
        "rq5_"
    ):
        if not isinstance(predecessor_representation, G3Representation) or (
            representation.history_hidden_dim
            != predecessor_representation.history_hidden_dim
        ):
            raise ValueError("family representation differs from selected predecessor")


def _selection_file_reference(
    path: Path,
    document: Mapping[str, object],
    role: str,
    *,
    root: Path | None = None,
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )

    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if (
        resolved.is_symlink()
        or not resolved.is_file()
        or not resolved.is_relative_to(root)
    ):
        raise ValueError("selection evidence is not a project artifact")
    return {
        "role": role,
        "path": resolved.relative_to(root).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
        "logical_sha256": document["sha256"],
    }


def _authenticated_compatibility_reference(
    path: Path,
    document: Mapping[str, object],
    authenticated: object,
    role: str,
    *,
    root: Path,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if (
        resolved.is_symlink()
        or not resolved.is_file()
        or not resolved.is_relative_to(root)
        or resolved.relative_to(root).as_posix() != authenticated.relative_path
        or resolved.stat().st_size != authenticated.size_bytes
        or _file_sha256(resolved) != authenticated.physical_sha256
        or document.get("sha256") != authenticated.logical_sha256
    ):
        raise ValueError("authenticated compatibility state changed before binding")
    return {
        "role": role,
        "path": authenticated.relative_path,
        "size_bytes": authenticated.size_bytes,
        "sha256": authenticated.physical_sha256,
        "logical_sha256": authenticated.logical_sha256,
    }


def _authenticate_selection_reference(
    selection_path: Path, reference: object, *, root: Path | None = None
) -> None:
    from experiments.g3_pretrained_item_embeddings.launchers.native500m import (
        PROJECT_ROOT,
    )

    del selection_path
    root = PROJECT_ROOT if root is None else root.resolve(strict=True)
    if not isinstance(reference, dict) or set(reference) != {
        "role",
        "path",
        "size_bytes",
        "sha256",
        "logical_sha256",
    }:
        raise ValueError("selection evidence reference schema differs")
    path = root / str(reference["path"])
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != reference["size_bytes"]
        or _file_sha256(path) != reference["sha256"]
    ):
        raise ValueError("selection evidence reference drifted")
    document = _load_json(path)
    body = {key: value for key, value in document.items() if key != "sha256"}
    if (
        document.get("sha256") != _canonical_sha256(body)
        or document.get("sha256") != reference["logical_sha256"]
    ):
        raise ValueError("selection evidence logical identity drifted")


def _collect_run(
    *,
    root: Path,
    queue_root: Path,
    context_path: Path,
    manifest: Any,
    specification_row: object,
    batch_id: str,
    job_id: str,
    row: dict[str, object],
    expected_population: Mapping[str, object],
    training_item_counts: Mapping[int, int],
    training_history_lengths: Mapping[int, int],
) -> dict[str, object]:
    from experiments.g3_pretrained_item_embeddings.launchers.run_native500m import (
        decode_compiled_job,
    )

    if not isinstance(specification_row, dict):
        raise ValueError("batch specification row is invalid")
    completed_path = queue_root / "completed" / f"{job_id}.json"
    completed, completed_fact = _load_json_with_fact(root, completed_path)
    expected_queue_fields = {
        "script": specification_row["script"],
        "run": specification_row["run"],
        "data_group": DATA_GROUP,
        "environment": specification_row["environment"],
    }
    if (
        completed.get("id") != job_id
        or completed.get("batch_id") != batch_id
        or completed.get("exit_code") != 0
        or any(
            completed.get(key) != value for key, value in expected_queue_fields.items()
        )
    ):
        raise ValueError(f"queue completion differs for {row['id']}")
    environment = completed.get("environment")
    if not isinstance(environment, list):
        raise ValueError(f"queue environment is absent for {row['id']}")
    assignments = _environment_assignments(environment)
    if (
        assignments.get("WANDB_MODE") != "offline"
        or assignments.get(MANIFEST_ENVIRONMENT) != str(manifest.path)
        or assignments.get(MANIFEST_LOGICAL_SHA256_ENVIRONMENT)
        != manifest.logical_sha256
        or assignments.get(MANIFEST_PHYSICAL_SHA256_ENVIRONMENT)
        != manifest.physical_sha256
    ):
        raise ValueError(f"queue environment differs for {row['id']}")
    compiled = decode_compiled_job(assignments.get(JOB_ENVIRONMENT, ""), manifest)
    if compiled.row_id != row["id"] or compiled.job != row["job"]:
        raise ValueError(f"compiled job differs for {row['id']}")
    run_directory = root / "generated/logs" / str(compiled.job["run_name"])
    paths = {
        name: run_directory / filename for name, filename in _ARTIFACT_FILENAMES.items()
    }
    artifacts = {name: _file_fact(root, path) for name, path in paths.items()}
    contract = _load_json(paths["job_contract"])
    if contract != compiled.to_dict():
        raise ValueError(f"job contract differs for {row['id']}")
    metadata = _load_json(paths["training_metadata"])
    _validate_training_metadata(
        metadata,
        compiled.job,
        str(row["id"]),
        implementation_identity=manifest.implementation_identity,
        evaluation_population=expected_population,
    )
    selection_metrics = _best_epoch_selection_metrics(
        paths["sweep_log"], int(metadata["best_epoch"])
    )
    metrics = _load_json(paths["final_metrics"])
    _validate_metrics(metrics, str(row["id"]))
    full_user_ranking, recomputed, slices = _verify_full_user_ranking(
        context_path=context_path,
        ranking_path=paths["ranking_evidence"],
        top_rankings_path=paths["top_item_rankings"],
        metadata=metadata,
        expected_population=expected_population,
        training_item_counts=training_item_counts,
        training_history_lengths=training_history_lengths,
    )
    checkpoint = _verify_final_evaluation_proof(
        paths=paths,
        metadata=metadata,
        implementation_identity=manifest.implementation_identity,
        evaluation_population=expected_population,
    )
    if any(
        not _same_metric(float(metrics[name]), float(recomputed[name]))
        for name in (*_METRIC_NAMES, "num_users")
    ):
        raise ValueError(f"final metrics differ from ranking evidence for {row['id']}")
    verify_unique_completed_run(
        queue_root,
        run_name=str(compiled.job["run_name"]),
        expected_job_id=job_id,
    )
    verify_artifacts_in_job_window(
        tuple(paths.values()),
        dispatched_at=float(completed["dispatched_at"]),
        finished_at=float(completed["finished_at"]),
        run_label=str(row["id"]),
    )
    for name, artifact_path in paths.items():
        _require_unchanged_fact(
            root, artifact_path, artifacts[name], f"{row['id']} {name}"
        )
    _require_unchanged_fact(
        root, completed_path, completed_fact, f"{row['id']} queue completion"
    )
    return {
        "row_id": row["id"],
        "family_id": compiled.job["family_id"],
        "manifest_order": compiled.job["manifest_order"],
        "run_name": compiled.job["run_name"],
        "job": compiled.job,
        "coordinate": {
            key: compiled.job[key]
            for key in (
                "embedding_learning_rate",
                "deep_learning_rate",
                "horizon_epochs",
                "capacity",
            )
        },
        "selection_metrics": selection_metrics,
        "metrics": metrics,
        "slices": slices,
        "restored_checkpoint": {
            "best_epoch": metadata["best_epoch"],
            **checkpoint,
        },
        "full_user_ranking": full_user_ranking,
        "queue_completion": completed_fact | {"job_id": job_id},
        "artifacts": artifacts,
    }


def _validate_training_metadata(
    metadata: dict[str, Any],
    job: dict[str, object],
    row_id: str,
    *,
    implementation_identity: Mapping[str, object],
    evaluation_population: Mapping[str, object],
) -> None:
    horizon = int(job["horizon_epochs"])
    embedding_lr = float(str(job["embedding_learning_rate"]))
    deep_lr = float(str(job["deep_learning_rate"]))
    required = {
        "dataset_size": "500m",
        "g3_dataset_size": "native-500m",
        "seed": job["seed"],
        "batch_size": 512,
        "physical_batch_size": 512,
        "effective_batch_size": 512,
        "num_epochs": horizon,
        "max_epochs": horizon,
        "epochs_trained": horizon,
        "stopped_epoch": horizon,
        "early_stopped": False,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "embedding_learning_rate": embedding_lr,
        "deep_learning_rate": deep_lr,
        "lr_schedule_horizon_epochs": horizon,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError(f"runtime metadata differs from contract for {row_id}")
    if (
        metadata.get("g3_execution_identity") != implementation_identity
        or metadata.get("g3_evaluation_population") != evaluation_population
    ):
        raise ValueError(f"runtime implementation or population differs for {row_id}")
    best_epoch = metadata.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= horizon:
        raise ValueError(f"restored best epoch is invalid for {row_id}")
    invariants = metadata.get("transfer_invariants")
    if not isinstance(invariants, dict):
        raise ValueError(f"training invariants are absent for {row_id}")
    expected_invariants = {
        "batch_size": 512,
        "dataset_size": "500m",
        "event_type_filter": "like",
        "eval_max_users": 20_000,
        "evaluation_catalog": "all",
        "exclude_seen_from_evaluation": False,
        "restore_best_weights": True,
        "user_sample": None,
    }
    if any(invariants.get(key) != value for key, value in expected_invariants.items()):
        raise ValueError(f"full-user or restore invariants differ for {row_id}")
    schedule = invariants.get("lr_schedule")
    expected_schedule = {
        "shape": "cosine",
        "warmup_fraction": 0.05,
        "cycles": 1,
        "min_lr_fraction": 0.0,
        "optimizer_group_scope": "deep_only",
    }
    if not isinstance(schedule, dict) or any(
        schedule.get(key) != value for key, value in expected_schedule.items()
    ):
        raise ValueError(f"deep-only one-cycle schedule differs for {row_id}")
    traces = metadata.get("lr_group_traces")
    if not isinstance(traces, dict) or set(traces) != {"embedding", "deep"}:
        raise ValueError(f"optimizer-group traces differ for {row_id}")
    embedding_trace = traces["embedding"]
    deep_trace = traces["deep"]
    if (
        not isinstance(embedding_trace, list)
        or len(embedding_trace) != horizon
        or any(value != embedding_lr for value in embedding_trace)
    ):
        raise ValueError(f"constant embedding learning rate differs for {row_id}")
    expected_deep = _expected_cosine_trace(metadata, deep_lr)
    if (
        not isinstance(deep_trace, list)
        or len(deep_trace) != horizon
        or any(
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12)
            for actual, expected in zip(deep_trace, expected_deep, strict=True)
        )
    ):
        raise ValueError(f"deep one-cycle learning-rate trace differs for {row_id}")


def _expected_cosine_trace(
    metadata: Mapping[str, object], deep_lr: float
) -> list[float]:
    epochs = int(metadata["epochs_trained"])
    steps_per_epoch = metadata.get("optimizer_steps_per_epoch")
    total = metadata.get("lr_schedule_horizon_steps")
    if type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError("schedule has invalid optimizer steps per epoch")
    if type(total) is not int or total != epochs * steps_per_epoch:
        raise ValueError("schedule horizon steps differ from completed horizon")
    warmup = int(total * 0.05)
    if warmup < 1:
        raise ValueError("schedule warmup has no optimizer step")
    decay = max(1, total - warmup - 1)
    result = []
    for epoch in range(1, epochs + 1):
        step = epoch * steps_per_epoch - 1
        if step < warmup:
            factor = (step + 1) / warmup
        else:
            progress = min(1.0, (step - warmup) / decay)
            factor = (
                0.0 if progress == 1.0 else 0.5 * (1 + math.cos(math.pi * progress))
            )
        result.append(deep_lr * factor)
    return result


def _best_epoch_selection_metrics(path: Path, best_epoch: int) -> dict[str, float]:
    epoch_index = best_epoch - 1
    values = []
    for line in path.read_text().splitlines():
        if re.search(rf"\bepoch {epoch_index} finished\b", line) is None:
            continue
        recall = re.search(rf"\bepoch/val_true\.recall@100=({_METRIC_NUMBER})\b", line)
        ndcg = re.search(rf"\bepoch/val_true\.ndcg@100=({_METRIC_NUMBER})\b", line)
        if recall is not None and ndcg is not None:
            values.append((float(recall.group(1)), float(ndcg.group(1))))
    if len(set(values)) != 1:
        raise ValueError("best epoch has missing or conflicting validation metrics")
    recall, ndcg = values[0]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (recall, ndcg)):
        raise ValueError("best-epoch validation metrics are invalid")
    return {"recall@100": recall, "ndcg@100": ndcg}


def _validate_metrics(metrics: dict[str, Any], row_id: str) -> None:
    if set(metrics) != {*_METRIC_NAMES, "num_users"} or any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ValueError(f"final metric schema differs for {row_id}")


def _same_metric(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=_METRIC_ABSOLUTE_TOLERANCE,
    )


def _verify_full_user_ranking(
    *,
    context_path: Path,
    ranking_path: Path,
    top_rankings_path: Path,
    metadata: Mapping[str, object],
    expected_population: Mapping[str, object],
    training_item_counts: Mapping[int, int],
    training_history_lengths: Mapping[int, int],
) -> tuple[dict[str, object], dict[str, float], dict[str, object]]:
    evidence = load_ranking_evidence(context_path, ranking_path)
    users = [int(value) for value in evidence.user_ids.tolist()]
    if (
        len(users) != expected_population.get("num_users")
        or len(users) != len(set(users))
        or evidence.max_k < 100
        or _user_ids_sha256(users) != expected_population.get("user_ids_sha256")
    ):
        raise ValueError("full-user ranking evidence is incomplete")
    snapshot = _load_json(top_rankings_path)
    catalog_size = snapshot.get("catalog_size")
    rows = snapshot.get("rankings")
    expected_snapshot_keys = {
        "schema_version",
        "catalog_sha256",
        "catalog_size",
        "exclude_seen",
        "max_k",
        "rankings",
    }
    if (
        set(snapshot) != expected_snapshot_keys
        or snapshot.get("schema_version") != 1
        or catalog_size != PROTOCOL.num_items
        or snapshot.get("catalog_sha256") != _catalog_sha256(PROTOCOL.num_items)
        or snapshot.get("exclude_seen") is not False
        or snapshot.get("max_k") != 100
        or evidence.max_k != 100
        or not isinstance(rows, list)
    ):
        raise ValueError("top-item ranking snapshot is invalid")
    rankings: dict[int, list[int]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"user_id", "item_ids"}:
            raise ValueError("top-item ranking row schema differs")
        user_id = row["user_id"]
        item_ids = row["item_ids"]
        if (
            type(user_id) is not int
            or not isinstance(item_ids, list)
            or len(item_ids) != 100
            or len(item_ids) != len(set(item_ids))
            or any(
                type(item_id) is not int or not 1 <= item_id <= PROTOCOL.num_items
                for item_id in item_ids
            )
        ):
            raise ValueError("top-item ranking row is invalid")
        rankings[user_id] = item_ids
    if set(rankings) != set(users) or len(rankings) != len(users):
        raise ValueError("top-item rankings do not cover every evaluation user")
    offsets = [int(value) for value in evidence.relevance_offsets.tolist()]
    relevant_ids = [int(value) for value in evidence.relevant_item_ids.tolist()]
    ranks = [int(value) for value in evidence.relevant_ranks.tolist()]
    if any(not 1 <= item_id <= PROTOCOL.num_items for item_id in relevant_ids):
        raise ValueError("full-user ranking contains an out-of-catalog relevant item")
    if any(offsets[index] == offsets[index + 1] for index in range(len(users))):
        raise ValueError("full-user ranking contains a user without relevance")
    for index, user_id in enumerate(users):
        positions = {
            item_id: rank for rank, item_id in enumerate(rankings[user_id], start=1)
        }
        expected = [
            positions.get(item_id, 0)
            for item_id in relevant_ids[offsets[index] : offsets[index + 1]]
        ]
        if expected != ranks[offsets[index] : offsets[index + 1]]:
            raise ValueError("ranking evidence differs from the top-item snapshot")
    result: dict[str, float] = {}
    for cutoff in (10, 50, 100):
        totals = {name: 0.0 for name in ("recall", "capped_recall", "ndcg", "mrr")}
        covered: set[int] = set()
        for index, user_id in enumerate(users):
            relevant = ranks[offsets[index] : offsets[index + 1]]
            hits = [rank for rank in relevant if 0 < rank <= cutoff]
            ideal_length = min(cutoff, len(relevant))
            totals["recall"] += len(hits) / len(relevant)
            totals["capped_recall"] += len(hits) / ideal_length
            totals["ndcg"] += sum(1 / math.log2(rank + 1) for rank in hits) / sum(
                1 / math.log2(rank + 1) for rank in range(1, ideal_length + 1)
            )
            totals["mrr"] += 1 / min(hits) if hits else 0.0
            covered.update(rankings[user_id][:cutoff])
        for name, total in totals.items():
            result[f"{name}@{cutoff}"] = total / len(users)
        result[f"coverage@{cutoff}"] = len(covered) / catalog_size
    result["num_users"] = float(len(users))
    relevant = {
        user_id: relevant_ids[offsets[index] : offsets[index + 1]]
        for index, user_id in enumerate(users)
    }
    slice_report = compute_ranking_slices(
        rankings=rankings,
        relevant_items=relevant,
        training_item_counts=training_item_counts,
        training_history_lengths={
            user_id: training_history_lengths[user_id] for user_id in users
        },
    )
    slices: dict[str, object] = {"item_frequency": {}, "user_history": {}}
    for item in slice_report.slices:
        membership = item.item_ids if item.item_ids else item.user_ids
        slices[item.axis][item.name] = {
            "num_users": item.num_users,
            "num_targets": item.num_targets,
            "metrics": dict(item.metrics),
            "membership_sha256": _canonical_sha256(list(membership)),
        }
    return (
        {
            "group": DATA_GROUP,
            "num_users": len(users),
            "max_k": evidence.max_k,
            "catalog_size": catalog_size,
            "user_sampling": metadata["transfer_invariants"].get("user_sample"),
        },
        result,
        slices,
    )


def _load_slice_inputs(
    root: Path, manifest: Any
) -> tuple[dict[int, int], dict[int, int]]:
    references = {reference.role: reference for reference in manifest.input_manifests}
    if "features" not in references:
        raise ValueError("native-500M execution has no feature manifest")
    feature_manifest = _load_json(
        resolve_input_manifest_path(manifest.path, references["features"])
    )
    artifacts = feature_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("native-500M feature manifest has no artifacts")
    paths = {
        artifact.get("role"): root / str(artifact.get("path"))
        for artifact in artifacts
        if isinstance(artifact, dict)
    }
    if set(paths) < {"item_features", "training_user_histories"}:
        raise ValueError("native-500M feature manifest lacks slice inputs")
    items = pl.read_parquet(
        paths["item_features"], columns=("compact_item_id", "training_count")
    ).sort("compact_item_id")
    expected_ids = list(range(1, PROTOCOL.num_items + 1))
    item_ids = [int(value) for value in items["compact_item_id"].to_list()]
    counts = [int(value) for value in items["training_count"].to_list()]
    if item_ids != expected_ids or any(value < 0 for value in counts):
        raise ValueError("native-500M item-frequency slice source differs")
    histories = pl.read_parquet(
        paths["training_user_histories"],
        columns=("uid", "training_history_length"),
    )
    user_ids = [int(value) for value in histories["uid"].to_list()]
    lengths = [int(value) for value in histories["training_history_length"].to_list()]
    if len(user_ids) != len(set(user_ids)) or any(value < 0 for value in lengths):
        raise ValueError("native-500M history slice source differs")
    return dict(zip(item_ids, counts, strict=True)), dict(
        zip(user_ids, lengths, strict=True)
    )


def _verify_final_evaluation_proof(
    *,
    paths: Mapping[str, Path],
    metadata: Mapping[str, object],
    implementation_identity: Mapping[str, object],
    evaluation_population: Mapping[str, object],
) -> dict[str, object]:
    proof_path = paths["final_evaluation_proof"]
    proof = _load_json(proof_path)
    supplied = proof.get("sha256")
    body = {key: value for key, value in proof.items() if key != "sha256"}
    expected_keys = {
        "schema_version",
        "best_epoch",
        "checkpoint",
        "checkpoint_state_sha256",
        "execution_identity_sha256",
        "evaluation_population",
        "final_metrics",
        "ranking_evidence",
        "top_item_rankings",
        "sha256",
    }
    if (
        set(proof) != expected_keys
        or supplied != _canonical_sha256(body)
        or proof.get("schema_version") != 1
        or proof.get("best_epoch") != metadata.get("best_epoch")
        or proof.get("execution_identity_sha256")
        != implementation_identity.get("sha256")
        or proof.get("evaluation_population") != evaluation_population
    ):
        raise ValueError("final evaluation proof identity differs")
    artifact_names = {
        "checkpoint": "restored_best_checkpoint",
        "final_metrics": "final_metrics",
        "ranking_evidence": "ranking_evidence",
        "top_item_rankings": "top_item_rankings",
    }
    for proof_name, path_name in artifact_names.items():
        path = paths[path_name]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"final evaluation artifact is absent: {proof_name}")
        expected = {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        if proof.get(proof_name) != expected:
            raise ValueError(f"final evaluation proof differs for {proof_name}")
        if proof_path.stat().st_mtime_ns < path.stat().st_mtime_ns:
            raise ValueError("final evaluation proof predates a bound artifact")
    try:
        checkpoint = torch.load(
            paths["restored_best_checkpoint"],
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("restored-best checkpoint is unreadable") from error
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "schema_version",
        "best_epoch",
        "state_sha256",
        "state_dict",
    }:
        raise ValueError("restored-best checkpoint schema differs")
    state = checkpoint["state_dict"]
    if (
        checkpoint["schema_version"] != 1
        or checkpoint["best_epoch"] != metadata.get("best_epoch")
        or not isinstance(state, dict)
        or checkpoint["state_sha256"] != _state_dict_sha256(state)
        or checkpoint["state_sha256"] != proof.get("checkpoint_state_sha256")
    ):
        raise ValueError("restored-best checkpoint identity differs")
    return {
        "artifact_sha256": proof["checkpoint"]["sha256"],
        "state_sha256": checkpoint["state_sha256"],
        "final_evaluation_after_restore": True,
    }


def _state_dict_sha256(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("restored-best checkpoint state differs")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _user_ids_sha256(user_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(user_ids), separators=(",", ":")).encode()
    ).hexdigest()


def _environment_assignments(values: list[object]) -> dict[str, str]:
    if not all(isinstance(value, str) and "=" in value for value in values):
        raise ValueError("queue environment assignment is invalid")
    pairs = [str(value).split("=", 1) for value in values]
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError("queue environment has duplicate assignments")
    return dict(pairs)


def _file_fact(root: Path, path: Path) -> dict[str, object]:
    try:
        invalid = path.is_symlink()
        resolved = path.resolve(strict=True)
        invalid = (
            invalid
            or resolved.is_symlink()
            or not resolved.is_file()
            or not resolved.is_relative_to(root)
        )
        size_bytes = resolved.stat().st_size
        sha256 = _file_sha256(resolved)
    except OSError as error:
        raise ValueError(
            f"evidence artifact is not a regular project file: {path}"
        ) from error
    if invalid:
        raise ValueError(f"evidence artifact is not a regular project file: {path}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _load_json_with_fact(
    root: Path, path: Path
) -> tuple[dict[str, Any], dict[str, object]]:
    fact_before = _file_fact(root, path)
    document = _load_json(path)
    _require_unchanged_fact(root, path, fact_before, "JSON artifact")
    return document, fact_before


def _require_unchanged_fact(
    root: Path,
    path: Path,
    expected: Mapping[str, object],
    label: str,
) -> None:
    if _file_fact(root, path) != {
        key: expected[key] for key in ("path", "size_bytes", "sha256")
    }:
        raise ValueError(f"{label} changed while evidence was verified")


def _authenticate_file_fact(root: Path, reference: object, label: str) -> None:
    if not isinstance(reference, Mapping) or not {
        "path",
        "size_bytes",
        "sha256",
    }.issubset(reference):
        raise ValueError(f"batch evidence {label} reference differs")
    relative = Path(str(reference["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"batch evidence {label} path differs")
    expected = {key: reference[key] for key in ("path", "size_bytes", "sha256")}
    _require_unchanged_fact(root, root / relative, expected, label)


def _register_batch_source_facts(
    root: Path,
    evidence_path: Path,
    evidence_fact: Mapping[str, object],
    document: Mapping[str, object],
) -> None:
    _register_source_fact(root, evidence_path, evidence_fact, "batch evidence")
    references: list[tuple[str, object]] = [
        ("execution manifest", document.get("execution_manifest")),
        ("batch specification", document.get("batch_specification")),
        ("queue batch", document.get("queue_batch")),
        ("ranking context", document.get("ranking_context")),
    ]
    if isinstance(document.get("queue_submission_binding"), Mapping):
        references.append(
            ("queue submission binding", document["queue_submission_binding"])
        )
    inputs = document.get("input_manifests")
    runs = document.get("runs")
    if not isinstance(inputs, list) or not isinstance(runs, list):
        raise ValueError("batch evidence source bindings differ")
    references.extend(("input manifest", value) for value in inputs)
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("batch evidence run binding differs")
        references.append(("queue completion", run.get("queue_completion")))
        artifacts = run.get("artifacts")
        job = run.get("job")
        if not isinstance(artifacts, Mapping) or not isinstance(job, Mapping):
            raise ValueError("batch evidence run source bindings differ")
        references.extend(
            (f"run artifact {name}", value) for name, value in artifacts.items()
        )
        predecessors = job.get("predecessor_artifacts")
        if not isinstance(predecessors, list):
            raise ValueError("batch evidence predecessor bindings differ")
        references.extend(("execution predecessor", value) for value in predecessors)
    for label, reference in references:
        if not isinstance(reference, Mapping) or not {
            "path",
            "size_bytes",
            "sha256",
        }.issubset(reference):
            raise ValueError(f"batch evidence {label} reference differs")
        relative = Path(str(reference["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"batch evidence {label} path differs")
        fact = {key: reference[key] for key in ("path", "size_bytes", "sha256")}
        _register_source_fact(root, root / relative, fact, label)


def _register_source_fact(
    root: Path,
    path: Path,
    fact: Mapping[str, object],
    label: str,
) -> None:
    _require_unchanged_fact(root, path, fact, label)
    context = _AUTHENTICATION_CONTEXT.get()
    assert context is not None
    source_facts = context["source_facts"]
    assert isinstance(source_facts, dict)
    key = (root.as_posix(), path.resolve(strict=True).as_posix())
    previous = source_facts.get(key)
    value = (root, path, dict(fact), label)
    if previous is not None and previous[2] != value[2]:
        raise ValueError(f"{label} has conflicting authenticated identities")
    source_facts[key] = value


def _evidence_root(path: Path) -> Path:
    project = PROJECT_ROOT.resolve(strict=True)
    return project if path.is_relative_to(project) else path.parent


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable batch evidence differs: {path}")
        return
    with NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != content:
        raise RuntimeError(f"immutable batch evidence differs: {path}")
