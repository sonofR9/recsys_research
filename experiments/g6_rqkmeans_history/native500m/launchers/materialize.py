from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from experiments.g6_rqkmeans_history.native500m.configs.runtime import (
    build_control,
    build_semantic_treatment,
)
from experiments.g6_rqkmeans_history.native500m.analysis.collect import (
    NATIVE500M_RELATIVE_DISPERSIONS,
    RECALL_RELATIVE_DISPERSION,
    candidate_selection_group,
    collect_stage_candidates,
)
from experiments.g6_rqkmeans_history.native500m.launchers.queue import (
    PROJECT_ROOT,
    canonical_bytes,
    load_queue_manifest,
    persist_immutable_bytes,
)
from experiments.g6_rqkmeans_history.native500m.launchers.runtime import (
    experiment_logical_sha256,
    source_identity_sha256,
)
from experiments.g6_rqkmeans_history.native500m.protocol.contracts import (
    ExactReuse,
    JobContract,
    SelectionBinding,
    StageManifest,
    job_id_has_coordinate,
    load_approval_binding,
    load_stage_manifest,
)
from experiments.g6_rqkmeans_history.native500m.protocol.design import (
    BEST_G1_ANCHOR,
    ORIGINAL_G1_ANCHOR,
    FIRST_RQ0_REPRESENTATION,
    REPRESENTATIONS,
    LearningRateCoordinate,
    SurfaceCoordinate,
    TokenizerCoordinate,
    boundary_coordinates,
    bridge_surface,
    control_surface,
    inherited_rq0_surface,
    rq1_paired_surface,
    rq23_paired_surface,
    rq0_first_surface,
    tokenizer_coordinates,
)
from experiments.g6_rqkmeans_history.native500m.protocol.selection import (
    Candidate,
    MetricValues,
    SeedEvidence,
    decide_rq1_initialization,
    decide_rq23,
    promote_against_two_baselines,
    select_by_quality,
)
from experiments.g6_rqkmeans_history.native500m.protocol.tokenizer_registry import (
    DEFAULT_REGISTRY_PATH,
    binding_environment,
    load_registry,
)


RUNNER = "experiments/g6_rqkmeans_history/native500m/launchers/run_native500m.py"
COMPILER_RECIPE_SCHEMA = "g6-native500m-compiler-recipe/v1"
CACHEFIX_REVISION = "cachefix01"


def build_controls_manifest(retry_revision: int = 0) -> StageManifest:
    load_approval_binding()
    if (
        not isinstance(retry_revision, int)
        or isinstance(retry_revision, bool)
        or retry_revision < 0
    ):
        raise ValueError("control retry revision must be a non-negative integer")
    source_sha256 = source_identity_sha256()
    jobs = tuple(
        _control_job(
            backbone,
            index,
            rates.embedding,
            rates.deep,
            source_sha256=source_sha256,
            run_revision=retry_revision,
        )
        for backbone, anchor in (
            ("original_g1", ORIGINAL_G1_ANCHOR),
            ("best_g1", BEST_G1_ANCHOR),
        )
        for index, rates in enumerate(control_surface(anchor))
    )
    return StageManifest.create(stage="controls", jobs=jobs)


def load_selection_binding(
    path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> tuple[SelectionBinding, dict[str, object]]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("stage selection must be a regular file")
    try:
        document = json.loads(resolved.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("stage selection is not valid JSON") from error
    expected = {
        "schema",
        "stage",
        "manifest_logical_sha256",
        "manifest_physical_sha256",
        "batch_id",
        "recall_relative_dispersion",
        "selection_group_field",
        "selected_job_ids",
        "candidates",
        "selection_sha256",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError("stage selection schema differs")
    body = {key: value for key, value in document.items() if key != "selection_sha256"}
    sha256 = hashlib.sha256(canonical_bytes(body)).hexdigest()
    if (
        document["schema"] != "g6-native500m-stage-selection/v1"
        or document["selection_sha256"] != sha256
        or document["recall_relative_dispersion"] != RECALL_RELATIVE_DISPERSION
    ):
        raise ValueError("stage selection identity or Recall dispersion differs")
    source = load_queue_manifest(source_manifest_path)
    if (
        document["stage"] != source.stage
        or document["manifest_logical_sha256"] != source.logical_sha256
        or document["manifest_physical_sha256"] != source.physical_sha256
        or document["selection_group_field"] != _selection_group_field(source.stage)
    ):
        raise ValueError("stage selection source manifest differs")
    candidates = document["candidates"]
    selected = document["selected_job_ids"]
    candidate_ids = (
        [row.get("job_id") for row in candidates]
        if isinstance(candidates, list)
        else []
    )
    if (
        not isinstance(selected, dict)
        or not selected
        or any(
            not isinstance(group, str) or not isinstance(job_id, str)
            for group, job_id in selected.items()
        )
        or any(candidate_ids.count(job_id) != 1 for job_id in selected.values())
    ):
        raise ValueError("stage selection winners are absent or ambiguous")
    if candidate_ids != [job.job_id for job in source.jobs]:
        raise ValueError("stage selection candidates differ from the source manifest")
    grouped: dict[str, list[Candidate]] = {}
    group_field = document["selection_group_field"]
    for order, (row, job) in enumerate(zip(candidates, source.jobs, strict=True)):
        metrics = row.get("validation_metrics")
        if (
            row.get("job_logical_sha256") != job.logical_sha256
            or row.get("parameters") != job.payload["parameters"]
            or not isinstance(metrics, dict)
        ):
            raise ValueError("stage selection candidate identity differs")
        candidate = Candidate(
            job.job_id,
            MetricValues(metrics.get("recall@100"), metrics.get("ndcg@100")),
            order,
        )
        group = candidate_selection_group(source.stage, job, group_field)
        if group is None:
            continue
        grouped.setdefault(group, []).append(candidate)
    if source.stage.endswith("confirmation"):
        recomputed = {
            group: next(
                job.job_id
                for job in source.jobs
                if job.seed == 42
                and candidate_selection_group(source.stage, job, group_field) == group
            )
            for group in grouped
        }
    else:
        recomputed = {
            group: select_by_quality(
                values,
                recall_relative_dispersion=RECALL_RELATIVE_DISPERSION,
            ).identifier
            for group, values in grouped.items()
        }
    if selected != recomputed:
        raise ValueError("stage selection winners differ from grouped recomputation")
    verified = collect_stage_candidates(
        manifest=source,
        logs_root=logs_root,
        queue_state_directory=queue_state_directory,
        batch_id=document["batch_id"],
        output_path=resolved,
    )
    if verified != document:
        raise ValueError("stage selection differs from authenticated run evidence")
    return SelectionBinding(str(document["stage"]), sha256, True), document


def build_rq0_first_surface_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    predecessor, selection = load_selection_binding(
        selection_path,
        source_manifest_path=source_manifest_path,
        logs_root=logs_root,
        queue_state_directory=queue_state_directory,
    )
    if (
        selection["stage"] not in {"controls", "controls_boundary"}
        or selection["selection_group_field"] != "backbone"
        or set(selection["selected_job_ids"]) != {"original_g1", "best_g1"}
    ):
        raise ValueError("RQ0 requires the grouped controls selection")
    selected_id = selection["selected_job_ids"]["best_g1"]
    selected = [row for row in selection["candidates"] if row["job_id"] == selected_id][
        0
    ]
    parameters = selected.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("backbone") != "best_g1":
        raise ValueError("RQ0 source is not the selected best-G1 control")
    return _semantic_manifest(
        "rq0_surface",
        predecessor,
        tuple(
            _semantic_row(
                stage="rq0_surface",
                family=FIRST_RQ0_REPRESENTATION,
                index=index,
                backbone="best_g1",
                representation=FIRST_RQ0_REPRESENTATION,
                coordinate=coordinate,
            )
            for index, coordinate in enumerate(rq0_first_surface())
        ),
    )


def build_controls_boundary_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    _require_selection(selection, "controls", {"original_g1", "best_g1"})
    source_sha256 = source_identity_sha256()
    source_manifest = load_queue_manifest(source_manifest_path)
    jobs: list[JobContract] = []
    for backbone in ("original_g1", "best_g1"):
        parameters = _selected_parameters(selection, backbone)
        source = _queue_job_by_id(
            source_manifest, str(selection["selected_job_ids"][backbone])
        )
        carried = _control_job(
            backbone,
            0,
            float(parameters["embedding_learning_rate"]),
            float(parameters["deep_learning_rate"]),
            source_sha256=source_sha256,
            stage="controls_boundary",
            predecessor=predecessor,
            exact_reuse=(_exact_reuse(source, selection_path),),
            reuse_selection_path=selection_path,
        )
        carried.validate_reuse({source.job_id: JobContract.from_dict(source.payload)})
        jobs.append(carried)
        design = boundary_coordinates(_rates(parameters), round_number=0)
        for index, coordinate in enumerate(design.coordinates, 1):
            jobs.append(
                _control_job(
                    backbone,
                    index,
                    coordinate.embedding,
                    coordinate.deep,
                    source_sha256=source_sha256,
                    stage="controls_boundary",
                    predecessor=predecessor,
                )
            )
    return StageManifest.create(
        stage="controls_boundary", jobs=tuple(jobs), predecessor=predecessor
    )


def build_rq0_inherited_surfaces_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    _require_selection(selection, "rq0_surface", {FIRST_RQ0_REPRESENTATION})
    anchor = _selected_parameters(selection, FIRST_RQ0_REPRESENTATION)
    tokenizer = _tokenizer(anchor)
    rates = _rates(anchor)
    rows = tuple(
        _semantic_row(
            stage="rq0_surface",
            family=representation,
            index=index,
            backbone="best_g1",
            representation=representation,
            coordinate=coordinate,
        )
        for representation in REPRESENTATIONS
        if representation != FIRST_RQ0_REPRESENTATION
        for index, coordinate in enumerate(inherited_rq0_surface(tokenizer, rates))
    )
    carried = _carried_semantic_job(
        stage="rq0_surface",
        predecessor=predecessor,
        selection=selection,
        selection_path=selection_path,
        source_manifest_path=source_manifest_path,
        group=FIRST_RQ0_REPRESENTATION,
        index=0,
    )
    manifest = _semantic_manifest("rq0_surface", predecessor, rows)
    return StageManifest.create(
        stage="rq0_surface",
        jobs=(carried, *manifest.jobs),
        predecessor=predecessor,
    )


def build_rq0_boundary_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    return _semantic_boundary_manifest(
        "rq0_boundary",
        selection_path,
        source_manifest_path,
        logs_root,
        queue_state_directory,
        paired_field=None,
    )


def build_rq0_bridge_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    if selection["stage"] not in {"rq0_surface", "rq0_boundary"}:
        raise ValueError("RQ0 bridge requires a resolved RQ0 selection")
    parameters = _best_selected_parameters(selection)
    rows = tuple(
        _semantic_row(
            stage="rq0_bridge",
            family=str(parameters["representation"]),
            index=index,
            backbone="original_g1",
            representation=str(parameters["representation"]),
            coordinate=SurfaceCoordinate(_tokenizer(parameters), rates),
            collision_policy=str(parameters["collision_policy"]),
            sid_initialization=str(parameters["sid_initialization"]),
        )
        for index, rates in enumerate(bridge_surface(_rates(parameters)))
    )
    return _semantic_manifest("rq0_bridge", predecessor, rows)


def build_rq0_bridge_boundary_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    return _semantic_boundary_manifest(
        "rq0_bridge_boundary",
        selection_path,
        source_manifest_path,
        logs_root,
        queue_state_directory,
        paired_field=None,
    )


def build_rq1_surface_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    if selection["stage"] not in {"rq0_surface", "rq0_boundary"}:
        raise ValueError("RQ1 requires a resolved RQ0 selection")
    parameters = _best_trainable_selected_parameters(selection)
    rows: list[tuple[dict[str, object], tuple[ExactReuse, ...]]] = []
    carried: JobContract | None = None
    for index, candidate in enumerate(
        rq1_paired_surface(_tokenizer(parameters), _rates(parameters))
    ):
        if index == 0 and candidate.initialization == "random":
            group = next(
                group
                for group, job_id in selection["selected_job_ids"].items()
                if job_id == _selected_id_for_parameters(selection, parameters)
            )
            carried = _carried_semantic_job(
                stage="rq1_surface",
                predecessor=predecessor,
                selection=selection,
                selection_path=selection_path,
                source_manifest_path=source_manifest_path,
                group=group,
                index=0,
            )
            continue
        row = _semantic_row(
            stage="rq1_surface",
            family=candidate.initialization,
            index=index,
            backbone="best_g1",
            representation=str(parameters["representation"]),
            coordinate=candidate.coordinate,
            sid_initialization=candidate.initialization,
        )
        rows.append((row, ()))
    if carried is None:
        raise ValueError("RQ1 exact random anchor is absent")
    manifest = _semantic_manifest_with_reuse("rq1_surface", predecessor, tuple(rows))
    return StageManifest.create(
        stage="rq1_surface", jobs=(carried, *manifest.jobs), predecessor=predecessor
    )


def build_rq1_boundary_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    return _semantic_boundary_manifest(
        "rq1_boundary",
        selection_path,
        source_manifest_path,
        logs_root,
        queue_state_directory,
        paired_field="sid_initialization",
    )


def build_rq1_confirmation_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    if selection["stage"] not in {"rq1_surface", "rq1_boundary"}:
        raise ValueError("RQ1 confirmation requires a resolved RQ1 selection")
    rows = tuple(
        _semantic_row_from_parameters(
            stage="rq1_confirmation",
            family=initialization,
            index=seed,
            parameters=_selected_parameters(selection, initialization),
            seed=seed,
        )
        for initialization in ("random", "content_pca")
        for seed in (43, 44, 45)
    )
    carried = tuple(
        _carried_semantic_job(
            stage="rq1_confirmation",
            predecessor=predecessor,
            selection=selection,
            selection_path=selection_path,
            source_manifest_path=source_manifest_path,
            group=initialization,
            index=42,
        )
        for initialization in ("random", "content_pca")
    )
    manifest = _semantic_manifest("rq1_confirmation", predecessor, rows)
    return StageManifest.create(
        stage="rq1_confirmation",
        jobs=(*carried, *manifest.jobs),
        predecessor=predecessor,
    )


def build_cachefix_rq0_anchor_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    _require_selection(
        selection,
        "rq0_surface",
        {
            "frozen_sid_tokens",
            "interleaved_item_sid_tokens",
            "item_frozen_sid_event",
            "item_learned_frozen_sid_event",
            "learned_frozen_sid_tokens",
            "learned_sid_event",
            "learned_sid_tokens",
        },
    )
    parameters = _selected_parameters(selection, "learned_sid_tokens")
    row = _semantic_row_from_parameters(
        stage="rq0_surface",
        family="learned_sid_tokens",
        index=0,
        parameters=parameters,
        seed=42,
    )
    manifest = _semantic_manifest("rq0_surface", predecessor, (row,))
    return _revised_manifest(manifest, CACHEFIX_REVISION, rerun_reuse=True)


def build_cachefix_rq1_surface_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    historical_selection_path: Path,
    historical_source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    _require_selection(selection, "rq0_surface", {"learned_sid_tokens"})
    _, historical = _authenticated_selection(
        historical_selection_path,
        historical_source_manifest_path,
        logs_root,
        queue_state_directory,
    )
    _require_selection(historical, "rq1_surface", {"random", "content_pca"})
    anchor_id = str(selection["selected_job_ids"]["learned_sid_tokens"])
    random = _carried_candidate_job(
        stage="rq1_surface",
        predecessor=predecessor,
        selection=selection,
        selection_path=selection_path,
        source_manifest_path=source_manifest_path,
        source_id=anchor_id,
        family="random",
        index=0,
    )
    content_parameters = _selected_parameters(historical, "content_pca")
    content_row = _semantic_row_from_parameters(
        stage="rq1_surface",
        family="content_pca",
        index=1,
        parameters=content_parameters,
        seed=42,
    )
    content = _semantic_manifest("rq1_surface", predecessor, (content_row,)).jobs[0]
    manifest = StageManifest.create(
        stage="rq1_surface",
        jobs=(random, content),
        predecessor=predecessor,
    )
    return _revised_manifest(manifest, CACHEFIX_REVISION)


def build_cachefix_rq1_confirmation_manifest(
    *, retry_revision: int = 0, **arguments: object
) -> StageManifest:
    if (
        not isinstance(retry_revision, int)
        or isinstance(retry_revision, bool)
        or retry_revision < 0
    ):
        raise ValueError("cachefix confirmation retry revision is invalid")
    revision = (
        CACHEFIX_REVISION
        if retry_revision == 0
        else f"{CACHEFIX_REVISION}_retry{retry_revision:02d}"
    )
    return _revised_manifest(build_rq1_confirmation_manifest(**arguments), revision)


def build_cachefix_rq23_initial_surface_manifest(**arguments: object) -> StageManifest:
    return _revised_manifest(
        build_rq23_initial_surface_manifest(**arguments), CACHEFIX_REVISION
    )


def build_cachefix_rq23_refinement_surface_manifest(
    **arguments: object,
) -> StageManifest:
    return _revised_manifest(
        build_rq23_refinement_surface_manifest(**arguments), CACHEFIX_REVISION
    )


def build_cachefix_rq23_boundary_manifest(**arguments: object) -> StageManifest:
    return _revised_manifest(
        build_rq23_boundary_manifest(**arguments), CACHEFIX_REVISION
    )


def build_cachefix_rq23_confirmation_manifest(**arguments: object) -> StageManifest:
    return _revised_manifest(
        build_rq23_confirmation_manifest(**arguments), CACHEFIX_REVISION
    )


def build_rq23_initial_surface_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    if selection["stage"] != "rq1_confirmation":
        raise ValueError("RQ2/RQ3 requires the resolved RQ1 confirmation")
    parameters = _rq1_decision_parameters(selection)
    rows: list[dict[str, object]] = []
    carried: JobContract | None = None
    selected_group = next(
        group
        for group, job_id in selection["selected_job_ids"].items()
        if job_id == _selected_id_for_parameters(selection, parameters)
    )
    for index, tokenizer in enumerate(tokenizer_coordinates()):
        for policy in ("suffix", "none"):
            if policy == "suffix" and tokenizer == _tokenizer(parameters):
                carried = _carried_semantic_job(
                    stage="rq2_rq3_surface",
                    predecessor=predecessor,
                    selection=selection,
                    selection_path=selection_path,
                    source_manifest_path=source_manifest_path,
                    group=selected_group,
                    index=index * 2,
                )
                continue
            rows.append(
                _semantic_row(
                    stage="rq2_rq3_surface",
                    family=policy,
                    index=index * 2 + int(policy == "none"),
                    backbone="best_g1",
                    representation=str(parameters["representation"]),
                    coordinate=SurfaceCoordinate(tokenizer, _rates(parameters)),
                    collision_policy=policy,
                    sid_initialization=str(parameters["sid_initialization"]),
                )
            )
    if carried is None:
        raise ValueError("RQ2/RQ3 exact suffix anchor is absent")
    rq0_anchor = _carried_semantic_job(
        stage="rq2_rq3_surface",
        predecessor=predecessor,
        selection=selection,
        selection_path=selection_path,
        source_manifest_path=source_manifest_path,
        group="random",
        index=99,
    )
    manifest = _semantic_manifest("rq2_rq3_surface", predecessor, tuple(rows))
    return StageManifest.create(
        stage="rq2_rq3_surface",
        jobs=(carried, rq0_anchor, *manifest.jobs),
        predecessor=predecessor,
    )


def build_rq23_refinement_surface_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    _require_selection(selection, "rq2_rq3_surface", {"suffix", "none"})
    suffix = _selected_parameters(selection, "suffix")
    none = _selected_parameters(selection, "none")
    inherited = _rates(suffix)
    coordinates = rq23_paired_surface(
        _tokenizer(suffix),
        inherited,
        suffix_winner=_tokenizer(suffix),
        no_suffix_winner=_tokenizer(none),
    )[12:]
    rows = tuple(
        _semantic_row(
            stage="rq2_rq3_refinement",
            family=candidate.policy,
            index=index // 2 + 1,
            backbone="best_g1",
            representation=str(suffix["representation"]),
            coordinate=candidate.coordinate,
            collision_policy=candidate.policy,
            sid_initialization=str(suffix["sid_initialization"]),
        )
        for index, candidate in enumerate(coordinates)
    )
    carried = tuple(
        _carried_semantic_job(
            stage="rq2_rq3_refinement",
            predecessor=predecessor,
            selection=selection,
            selection_path=selection_path,
            source_manifest_path=source_manifest_path,
            group=policy,
            index=0,
        )
        for policy in ("suffix", "none")
    )
    rq0_sources = [
        row["job_id"]
        for row in selection["candidates"]
        if job_id_has_coordinate(row.get("job_id"), "random", 99)
    ]
    rq0_anchor = (
        _carried_candidate_job(
            stage="rq2_rq3_refinement",
            predecessor=predecessor,
            selection=selection,
            selection_path=selection_path,
            source_manifest_path=source_manifest_path,
            source_id=rq0_sources[0],
            family="rq0_anchor",
            index=99,
        )
        if len(rq0_sources) == 1
        else None
    )
    manifest = _semantic_manifest("rq2_rq3_refinement", predecessor, rows)
    return StageManifest.create(
        stage="rq2_rq3_refinement",
        jobs=(
            *carried,
            *((rq0_anchor,) if rq0_anchor is not None else ()),
            *manifest.jobs,
        ),
        predecessor=predecessor,
    )


def build_rq23_boundary_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    return _semantic_boundary_manifest(
        "rq2_rq3_boundary",
        selection_path,
        source_manifest_path,
        logs_root,
        queue_state_directory,
        paired_field="collision_policy",
    )


def build_rq23_confirmation_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
    rq1_selection_path: Path | None = None,
    rq1_source_manifest_path: Path | None = None,
    rq0_selection_path: Path | None = None,
    rq0_source_manifest_path: Path | None = None,
    rq0_lineage_selection_path: Path | None = None,
    rq0_lineage_source_manifest_path: Path | None = None,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    if selection["stage"] not in {"rq2_rq3_refinement", "rq2_rq3_boundary"}:
        raise ValueError("RQ2/RQ3 confirmation requires a resolved surface")
    if (rq1_selection_path is None) != (rq1_source_manifest_path is None):
        raise ValueError("RQ1 overlap reuse inputs must be supplied together")
    if (rq0_selection_path is None) != (rq0_source_manifest_path is None):
        raise ValueError("RQ0 anchor reuse inputs must be supplied together")
    if (rq0_lineage_selection_path is None) != (
        rq0_lineage_source_manifest_path is None
    ):
        raise ValueError("RQ0 lineage inputs must be supplied together")
    rq1_selection: dict[str, object] | None = None
    if rq1_selection_path is not None and rq1_source_manifest_path is not None:
        _, rq1_selection = _authenticated_selection(
            rq1_selection_path,
            rq1_source_manifest_path,
            logs_root,
            queue_state_directory,
        )
        if rq1_selection["stage"] != "rq1_confirmation":
            raise ValueError("RQ23 overlap source is not RQ1 confirmation")
    carried = [
        _carried_semantic_job(
            stage="rq2_rq3_confirmation",
            predecessor=predecessor,
            selection=selection,
            selection_path=selection_path,
            source_manifest_path=source_manifest_path,
            group=policy,
            index=42,
        )
        for policy in ("suffix", "none")
    ]
    rq0_selection = selection
    rq0_selection_source_path = selection_path
    rq0_manifest_source_path = source_manifest_path
    rq0_sources = [
        row["job_id"]
        for row in rq0_selection["candidates"]
        if job_id_has_coordinate(row.get("job_id"), "rq0_anchor", 99)
        or job_id_has_coordinate(row.get("job_id"), "random", 99)
    ]
    if not rq0_sources and rq0_selection_path is not None:
        _, rq0_selection = _authenticated_selection(
            rq0_selection_path,
            rq0_source_manifest_path,
            logs_root,
            queue_state_directory,
        )
        if rq0_selection["stage"] != "rq2_rq3_surface":
            raise ValueError("RQ0 anchor fallback is not an RQ2/RQ3 surface")
        lineage = load_stage_manifest(source_manifest_path).predecessor
        if selection["stage"] == "rq2_rq3_boundary":
            if (
                rq0_lineage_selection_path is None
                or rq0_lineage_source_manifest_path is None
            ):
                raise ValueError("RQ0 boundary fallback lacks refinement lineage")
            _, lineage_selection = _authenticated_selection(
                rq0_lineage_selection_path,
                rq0_lineage_source_manifest_path,
                logs_root,
                queue_state_directory,
            )
            if (
                lineage is None
                or lineage_selection["stage"] != "rq2_rq3_refinement"
                or lineage.stage != lineage_selection["stage"]
                or lineage.selection_sha256 != lineage_selection.get("selection_sha256")
            ):
                raise ValueError(
                    "RQ0 boundary fallback differs from refinement lineage"
                )
            lineage = load_stage_manifest(rq0_lineage_source_manifest_path).predecessor
        elif rq0_lineage_selection_path is not None:
            raise ValueError("RQ0 refinement fallback has extraneous lineage inputs")
        if (
            lineage is None
            or lineage.stage != rq0_selection["stage"]
            or lineage.selection_sha256 != rq0_selection.get("selection_sha256")
        ):
            raise ValueError("RQ0 anchor fallback differs from refinement lineage")
        rq0_selection_source_path = rq0_selection_path
        rq0_manifest_source_path = rq0_source_manifest_path
        rq0_sources = [
            row["job_id"]
            for row in rq0_selection["candidates"]
            if job_id_has_coordinate(row.get("job_id"), "rq0_anchor", 99)
            or job_id_has_coordinate(row.get("job_id"), "random", 99)
        ]
    if len(rq0_sources) != 1:
        raise ValueError("RQ2/RQ3 confirmation lacks the frozen RQ0 anchor")
    rq0_candidate = next(
        row for row in rq0_selection["candidates"] if row["job_id"] == rq0_sources[0]
    )
    carried.append(
        _carried_candidate_job(
            stage="rq2_rq3_confirmation",
            predecessor=predecessor,
            selection=rq0_selection,
            selection_path=rq0_selection_source_path,
            source_manifest_path=rq0_manifest_source_path,
            source_id=rq0_sources[0],
            family="rq0_anchor",
            index=42,
        )
    )
    new_rows: list[dict[str, object]] = []
    overlap_jobs: list[JobContract] = []
    arms = {
        "rq0_anchor": dict(rq0_candidate["parameters"]),
        "suffix": _selected_parameters(selection, "suffix"),
        "none": _selected_parameters(selection, "none"),
    }
    rq1_candidates = [] if rq1_selection is None else list(rq1_selection["candidates"])
    for family, seed, target_parameters, overlap in resolve_rq23_confirmation_reuse(
        arms, tuple(rq1_candidates)
    ):
        if overlap is not None:
            overlap_jobs.append(
                _carried_candidate_job(
                    stage="rq2_rq3_confirmation",
                    predecessor=predecessor,
                    selection=rq1_selection,
                    selection_path=rq1_selection_path,
                    source_manifest_path=rq1_source_manifest_path,
                    source_id=str(overlap["job_id"]),
                    family=family,
                    index=seed,
                )
            )
        else:
            new_rows.append(
                _semantic_row_from_parameters(
                    stage="rq2_rq3_confirmation",
                    family=family,
                    index=seed,
                    parameters=target_parameters,
                    seed=seed,
                )
            )
    if len(new_rows) not in {2, 4, 6}:
        raise ValueError("RQ23 overlap reuse produced an unapproved new-run count")
    manifest = _semantic_manifest("rq2_rq3_confirmation", predecessor, tuple(new_rows))
    return StageManifest.create(
        stage="rq2_rq3_confirmation",
        jobs=(*carried, *overlap_jobs, *manifest.jobs),
        predecessor=predecessor,
    )


def resolve_rq23_confirmation_reuse(
    arms: dict[str, dict[str, object]],
    rq1_candidates: tuple[dict[str, object], ...],
) -> tuple[tuple[str, int, dict[str, object], dict[str, object] | None], ...]:
    resolved = []
    for family, parameters in arms.items():
        for seed in (43, 44):
            target = parameters | {"seed": seed}
            matches = [
                row
                for row in rq1_candidates
                if _same_scientific_parameters(row.get("parameters"), target)
            ]
            if len(matches) > 1:
                raise ValueError("RQ1 overlap reuse source is ambiguous")
            resolved.append((family, seed, target, matches[0] if matches else None))
    new_count = sum(overlap is None for _, _, _, overlap in resolved)
    if new_count not in {2, 4, 6}:
        raise ValueError("RQ23 overlap reuse produced an unapproved new-run count")
    return tuple(resolved)


def build_terminal_bridge_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    _require_selection(
        selection, "rq2_rq3_confirmation", {"rq0_anchor", "suffix", "none"}
    )
    parameters = _rq23_terminal_parameters(selection)
    rows = tuple(
        _semantic_row(
            stage="terminal_bridge",
            family=str(parameters["collision_policy"]),
            index=index,
            backbone="original_g1",
            representation=str(parameters["representation"]),
            coordinate=SurfaceCoordinate(_tokenizer(parameters), rates),
            collision_policy=str(parameters["collision_policy"]),
            sid_initialization=str(parameters["sid_initialization"]),
        )
        for index, rates in enumerate(bridge_surface(_rates(parameters)))
    )
    return _semantic_manifest("terminal_bridge", predecessor, rows)


def build_terminal_bridge_boundary_manifest(
    selection_path: Path,
    *,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> StageManifest:
    return _semantic_boundary_manifest(
        "terminal_bridge_boundary",
        selection_path,
        source_manifest_path,
        logs_root,
        queue_state_directory,
        paired_field=None,
    )


def rederive_for_queue_admission(
    persisted: StageManifest, rederived: StageManifest
) -> StageManifest:
    if persisted.to_document() != rederived.to_document():
        raise ValueError(
            "persisted stage manifest differs from deterministic rederivation"
        )
    current_source = source_identity_sha256()
    if any(
        job.parameters["environment"].get("G6_NATIVE500M_SOURCE_SHA256")
        != current_source
        for job in rederived.jobs
    ):
        raise ValueError("rederived stage manifest does not bind current source")
    return rederived


def _authenticated_selection(
    selection_path: Path,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
) -> tuple[SelectionBinding, dict[str, object]]:
    load_approval_binding()
    return load_selection_binding(
        selection_path,
        source_manifest_path=source_manifest_path,
        logs_root=logs_root,
        queue_state_directory=queue_state_directory,
    )


def _require_selection(
    selection: dict[str, object], stage: str, groups: set[str]
) -> None:
    selected = selection.get("selected_job_ids")
    if (
        selection.get("stage") != stage
        or not isinstance(selected, dict)
        or set(selected) != groups
    ):
        raise ValueError(f"{stage} selection groups differ from the approved design")


def _selected_parameters(selection: dict[str, object], group: str) -> dict[str, object]:
    selected = selection["selected_job_ids"]
    candidates = selection["candidates"]
    if not isinstance(selected, dict) or not isinstance(candidates, list):
        raise ValueError("selection rows are invalid")
    job_id = selected.get(group)
    matches = [row for row in candidates if row.get("job_id") == job_id]
    if len(matches) != 1 or not isinstance(matches[0].get("parameters"), dict):
        raise ValueError(f"selected group {group!r} is absent")
    return dict(matches[0]["parameters"])


def _selected_id_for_parameters(
    selection: dict[str, object], parameters: dict[str, object]
) -> str:
    candidates = selection["candidates"]
    matches = [
        row.get("job_id")
        for row in candidates
        if row.get("parameters") == parameters
        and row.get("job_id") in selection["selected_job_ids"].values()
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError("selected source job is ambiguous")
    return matches[0]


def _best_selected_parameters(selection: dict[str, object]) -> dict[str, object]:
    selected = selection.get("selected_job_ids")
    candidates = selection.get("candidates")
    if not isinstance(selected, dict) or not isinstance(candidates, list):
        raise ValueError("selection rows are invalid")
    rows = [
        row
        for order, row in enumerate(candidates)
        if row.get("job_id") in selected.values()
        and isinstance(row.get("validation_metrics"), dict)
    ]
    selectable = [
        Candidate(
            str(row["job_id"]),
            MetricValues(
                row["validation_metrics"]["recall@100"],
                row["validation_metrics"]["ndcg@100"],
            ),
            candidates.index(row),
        )
        for row in rows
    ]
    winner = select_by_quality(
        selectable, recall_relative_dispersion=RECALL_RELATIVE_DISPERSION
    ).identifier
    matches = [row for row in rows if row["job_id"] == winner]
    return dict(matches[0]["parameters"])


def _best_trainable_selected_parameters(
    selection: dict[str, object],
) -> dict[str, object]:
    trainable = {
        "learned_sid_event",
        "item_learned_frozen_sid_event",
        "learned_sid_tokens",
        "learned_frozen_sid_tokens",
        "interleaved_item_sid_tokens",
    }
    selected = selection.get("selected_job_ids")
    candidates = selection.get("candidates")
    if not isinstance(selected, dict) or not isinstance(candidates, list):
        raise ValueError("selection rows are invalid")
    eligible = {
        group: job_id for group, job_id in selected.items() if group in trainable
    }
    if not eligible:
        raise ValueError("RQ1 has no selected trainable-SID representation")
    narrowed = dict(selection) | {"selected_job_ids": eligible}
    return _best_selected_parameters(narrowed)


def _rq1_decision_parameters(selection: dict[str, object]) -> dict[str, object]:
    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("RQ1 confirmation candidates are invalid")
    grouped = {
        initialization: sorted(
            [
                row
                for row in candidates
                if row.get("parameters", {}).get("sid_initialization") == initialization
            ],
            key=lambda row: row["parameters"]["seed"],
        )
        for initialization in ("random", "content_pca")
    }
    decision = decide_rq1_initialization(
        [_seed_evidence(row) for row in grouped["random"]],
        [_seed_evidence(row) for row in grouped["content_pca"]],
        recall_relative_dispersion=RECALL_RELATIVE_DISPERSION,
        ndcg_relative_dispersion=NATIVE500M_RELATIVE_DISPERSIONS["ndcg@100"],
    )
    rows = grouped[decision.selected]
    seed42 = [row for row in rows if row["parameters"]["seed"] == 42]
    if len(seed42) != 1:
        raise ValueError("RQ1 decision lacks its seed-42 anchor")
    return dict(seed42[0]["parameters"])


def _seed_evidence(row: dict[str, object]) -> SeedEvidence:
    metrics = row.get("validation_metrics")
    convergence = row.get("convergence")
    parameters = row.get("parameters")
    if not isinstance(metrics, dict) or not isinstance(parameters, dict):
        raise ValueError("confirmation candidate evidence is incomplete")
    return SeedEvidence(
        seed=int(parameters["seed"]),
        metrics=MetricValues(metrics["recall@100"], metrics["ndcg@100"]),
        first_epoch_at_95_percent=(
            None
            if not isinstance(convergence, dict)
            else convergence["first_epoch_at_95_percent"]
        ),
        normalized_recall_auc=(
            None
            if not isinstance(convergence, dict)
            else convergence["normalized_recall_auc"]
        ),
    )


def _same_scientific_parameters(source: object, target: dict[str, object]) -> bool:
    if not isinstance(source, dict):
        return False
    fields = (
        "backbone",
        "embedding_learning_rate",
        "deep_learning_rate",
        "seed",
        "representation",
        "levels",
        "shared_codes",
        "representation_width",
        "collision_policy",
        "sid_initialization",
    )
    return all(source.get(field) == target.get(field) for field in fields)


def _rq23_terminal_parameters(selection: dict[str, object]) -> dict[str, object]:
    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("RQ2/RQ3 confirmation candidates are invalid")
    groups = {
        "rq0": [row for row in candidates if "rq0_anchor" in str(row["job_id"])],
        "suffix": [
            row
            for row in candidates
            if "rq0_anchor" not in str(row["job_id"])
            and row["parameters"]["collision_policy"] == "suffix"
        ],
        "none": [
            row for row in candidates if row["parameters"]["collision_policy"] == "none"
        ],
    }
    for rows in groups.values():
        rows.sort(key=lambda row: row["parameters"]["seed"])
    decision = decide_rq23(
        [_seed_evidence(row) for row in groups["rq0"]],
        [_seed_evidence(row) for row in groups["suffix"]],
        [_seed_evidence(row) for row in groups["none"]],
        recall_relative_dispersion=RECALL_RELATIVE_DISPERSION,
        ndcg_relative_dispersion=NATIVE500M_RELATIVE_DISPERSIONS["ndcg@100"],
    )
    selected = (
        "rq0" if decision.terminal_selected == "rq0" else decision.terminal_selected
    )
    seed42 = [row for row in groups[selected] if row["parameters"]["seed"] == 42]
    if len(seed42) != 1:
        raise ValueError("RQ2/RQ3 decision lacks its seed-42 anchor")
    return dict(seed42[0]["parameters"])


def _rates(parameters: dict[str, object]) -> LearningRateCoordinate:
    return LearningRateCoordinate(
        float(parameters["embedding_learning_rate"]),
        float(parameters["deep_learning_rate"]),
    )


def _tokenizer(parameters: dict[str, object]) -> TokenizerCoordinate:
    return TokenizerCoordinate(
        int(parameters["levels"]), int(parameters["shared_codes"])
    )


def _semantic_row(
    *,
    stage: str,
    family: str,
    index: int,
    backbone: str,
    representation: str,
    coordinate: SurfaceCoordinate,
    collision_policy: str = "suffix",
    sid_initialization: str = "random",
    seed: int = 42,
) -> dict[str, object]:
    slug = family.replace("_", "-")
    registry = load_registry(
        DEFAULT_REGISTRY_PATH, source_sha256=source_identity_sha256()
    )
    return {
        "job_id": f"{stage}:{family}:{index:02d}",
        "run_name": f"g6_native500m_{stage}_{slug}_{index:02d}",
        "data_group": (
            f"g6-native500m-{stage}-{slug}-l{coordinate.tokenizer.levels}-"
            f"c{coordinate.tokenizer.shared_codes}-v1"
        ),
        "environment": binding_environment(
            registry,
            levels=coordinate.tokenizer.levels,
            shared_codes=coordinate.tokenizer.shared_codes,
            collision_policy=collision_policy,
        ),
        "backbone": backbone,
        "embedding_learning_rate": coordinate.learning_rates.embedding,
        "deep_learning_rate": coordinate.learning_rates.deep,
        "seed": seed,
        "representation": representation,
        "levels": coordinate.tokenizer.levels,
        "shared_codes": coordinate.tokenizer.shared_codes,
        "representation_width": 128,
        "collision_policy": collision_policy,
        "sid_initialization": sid_initialization,
    }


def _semantic_row_from_parameters(
    *,
    stage: str,
    family: str,
    index: int,
    parameters: dict[str, object],
    seed: int,
) -> dict[str, object]:
    return _semantic_row(
        stage=stage,
        family=family,
        index=index,
        backbone=str(parameters["backbone"]),
        representation=str(parameters["representation"]),
        coordinate=SurfaceCoordinate(_tokenizer(parameters), _rates(parameters)),
        collision_policy=str(parameters["collision_policy"]),
        sid_initialization=str(parameters["sid_initialization"]),
        seed=seed,
    )


def _semantic_manifest(
    stage: str,
    predecessor: SelectionBinding,
    rows: tuple[dict[str, object], ...],
) -> StageManifest:
    return _semantic_manifest_with_reuse(
        stage, predecessor, tuple((row, ()) for row in rows)
    )


def _semantic_manifest_with_reuse(
    stage: str,
    predecessor: SelectionBinding,
    rows: tuple[tuple[dict[str, object], tuple[ExactReuse, ...]], ...],
) -> StageManifest:
    source_sha256 = source_identity_sha256()
    jobs = tuple(
        _dependent_job(
            stage,
            predecessor,
            row,
            source_sha256=source_sha256,
            exact_reuse=exact_reuse,
        )
        for row, exact_reuse in rows
    )
    return StageManifest.create(stage=stage, jobs=jobs, predecessor=predecessor)


def _revised_manifest(
    manifest: StageManifest,
    revision: str,
    *,
    rerun_reuse: bool = False,
) -> StageManifest:
    if not revision or not revision.replace("_", "").isalnum():
        raise ValueError("run revision is invalid")
    if manifest.predecessor is None:
        raise ValueError("revised semantic manifest lacks its predecessor")
    source_sha256 = source_identity_sha256()
    jobs = []
    for job in manifest.jobs:
        row = dict(job.parameters)
        row["job_id"] = f"{job.job_id}:{revision}"
        row["run_name"] = f"{job.run_name}_{revision}"
        row["data_group"] = f"{row['data_group']}-{revision}"
        jobs.append(
            _dependent_job(
                manifest.stage,
                manifest.predecessor,
                row,
                source_sha256=source_sha256,
                exact_reuse=() if rerun_reuse else job.exact_reuse,
            )
        )
    return StageManifest.create(
        stage=manifest.stage,
        jobs=tuple(jobs),
        predecessor=manifest.predecessor,
    )


def _semantic_boundary_manifest(
    stage: str,
    selection_path: Path,
    source_manifest_path: Path,
    logs_root: Path,
    queue_state_directory: Path,
    *,
    paired_field: str | None,
) -> StageManifest:
    predecessor, selection = _authenticated_selection(
        selection_path, source_manifest_path, logs_root, queue_state_directory
    )
    selected = selection.get("selected_job_ids")
    if not isinstance(selected, dict):
        raise ValueError("boundary source selection is invalid")
    parameters_by_group = {
        group: _selected_parameters(selection, group) for group in selected
    }
    carried = tuple(
        _carried_semantic_job(
            stage=stage,
            predecessor=predecessor,
            selection=selection,
            selection_path=selection_path,
            source_manifest_path=source_manifest_path,
            group=group,
            index=0,
        )
        for group in selected
    )
    rows: list[dict[str, object]] = []
    if paired_field is None:
        for group, parameters in parameters_by_group.items():
            for index, rates in enumerate(
                boundary_coordinates(_rates(parameters), round_number=0).coordinates,
                1,
            ):
                rows.append(
                    _semantic_row_from_parameters(
                        stage=stage,
                        family=group,
                        index=index,
                        parameters=parameters
                        | {
                            "embedding_learning_rate": rates.embedding,
                            "deep_learning_rate": rates.deep,
                        },
                        seed=42,
                    )
                )
    else:
        boundary_rates = []
        for parameters in parameters_by_group.values():
            boundary_rates.extend(
                boundary_coordinates(_rates(parameters), round_number=0).coordinates
            )
        unique_rates = tuple(dict.fromkeys(boundary_rates))
        if len(unique_rates) > 8:
            raise ValueError(
                "paired boundary requires more than eight coordinates; approval required"
            )
        for index, rates in enumerate(unique_rates, 1):
            for group, parameters in parameters_by_group.items():
                rows.append(
                    _semantic_row_from_parameters(
                        stage=stage,
                        family=group,
                        index=index,
                        parameters=parameters
                        | {
                            "embedding_learning_rate": rates.embedding,
                            "deep_learning_rate": rates.deep,
                        },
                        seed=42,
                    )
                )
    if not rows:
        raise ValueError(
            "selected learning rates do not require an approved boundary stage"
        )
    manifest = _semantic_manifest(stage, predecessor, tuple(rows))
    return StageManifest.create(
        stage=stage, jobs=(*carried, *manifest.jobs), predecessor=predecessor
    )


def _queue_job_by_id(manifest: object, job_id: str) -> object:
    jobs = getattr(manifest, "jobs", ())
    matches = [job for job in jobs if job.job_id == job_id]
    if len(matches) != 1:
        raise ValueError("exact-reuse source job is absent")
    return matches[0]


def _exact_reuse(source: object, selection_path: Path | None = None) -> ExactReuse:
    parameters = source.payload["parameters"]
    fields = [
        "backbone",
        "embedding_learning_rate",
        "deep_learning_rate",
        "seed",
    ]
    if "representation" in parameters:
        fields.extend(
            (
                "representation",
                "levels",
                "shared_codes",
                "representation_width",
                "collision_policy",
                "sid_initialization",
            )
        )
    binding: dict[str, object] = {}
    if selection_path is not None:
        resolved = selection_path.resolve(strict=True)
        selection = json.loads(resolved.read_text())
        binding = {
            "source_selection_stage": selection["stage"],
            "source_selection_sha256": selection["selection_sha256"],
            "source_selection_physical_sha256": hashlib.sha256(
                resolved.read_bytes()
            ).hexdigest(),
            "source_selection_path": str(resolved),
        }
    return ExactReuse(
        source_job_id=source.job_id,
        source_contract_sha256=source.logical_sha256,
        fields=tuple(fields),
        **binding,
    )


def _reuse_selection_environment(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {
        "G6_NATIVE500M_REUSE_SELECTION_PATH": str(resolved),
        "G6_NATIVE500M_REUSE_SELECTION_PHYSICAL_SHA256": hashlib.sha256(
            resolved.read_bytes()
        ).hexdigest(),
    }


def _carried_semantic_job(
    *,
    stage: str,
    predecessor: SelectionBinding,
    selection: dict[str, object],
    selection_path: Path,
    source_manifest_path: Path,
    group: str,
    index: int,
) -> JobContract:
    source_id = selection["selected_job_ids"][group]
    return _carried_candidate_job(
        stage=stage,
        predecessor=predecessor,
        selection=selection,
        selection_path=selection_path,
        source_manifest_path=source_manifest_path,
        source_id=str(source_id),
        family=group,
        index=index,
    )


def _carried_candidate_job(
    *,
    stage: str,
    predecessor: SelectionBinding,
    selection: dict[str, object],
    selection_path: Path,
    source_manifest_path: Path,
    source_id: str,
    family: str,
    index: int,
) -> JobContract:
    candidates = selection["candidates"]
    matches = [row for row in candidates if row.get("job_id") == source_id]
    if len(matches) != 1 or not isinstance(matches[0].get("parameters"), dict):
        raise ValueError("carried candidate is absent from its selection")
    parameters = dict(matches[0]["parameters"])
    source_manifest = load_queue_manifest(source_manifest_path)
    source = _queue_job_by_id(source_manifest, str(source_id))
    row = _semantic_row_from_parameters(
        stage=stage,
        family=family,
        index=index,
        parameters=parameters,
        seed=int(parameters["seed"]),
    )
    row_environment = row["environment"]
    if not isinstance(row_environment, dict):
        raise ValueError("carried candidate environment is invalid")
    row_environment.update(_reuse_selection_environment(selection_path))
    job = _dependent_job(
        stage,
        predecessor,
        row,
        source_sha256=source_identity_sha256(),
        exact_reuse=(_exact_reuse(source, selection_path),),
    )
    job.validate_reuse({source.job_id: JobContract.from_dict(source.payload)})
    return job


def _selection_group_field(stage: str) -> str:
    if stage.startswith("controls"):
        return "backbone"
    if stage.startswith("rq0"):
        return "representation"
    if stage.startswith("rq1"):
        return "sid_initialization"
    if stage.startswith("rq2_rq3"):
        return "collision_policy"
    return "backbone"


def persist_stage_manifest(path: Path, manifest: StageManifest) -> Path:
    manifest.validate()
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_bytes(manifest.to_document())
    return persist_immutable_bytes(destination, content, label="stage manifest")


def compile_and_persist_stage(
    path: Path,
    *,
    compiler_name: str,
    arguments: dict[str, object],
) -> tuple[Path, Path]:
    compiler = _stage_compilers().get(compiler_name)
    if compiler is None:
        raise ValueError("unknown native-500M stage compiler")
    normalized, call_arguments = _bind_compiler_arguments(arguments)
    manifest = compiler(**call_arguments)
    manifest_path = persist_stage_manifest(path, manifest)
    manifest_content = manifest_path.read_bytes()
    body = {
        "schema": COMPILER_RECIPE_SCHEMA,
        "compiler": compiler_name,
        "arguments": normalized,
        "manifest_logical_sha256": manifest.logical_sha256,
        "manifest_physical_sha256": hashlib.sha256(manifest_content).hexdigest(),
    }
    recipe = {
        **body,
        "recipe_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest(),
    }
    recipe_path = _recipe_path(manifest_path)
    persist_immutable_bytes(
        recipe_path, canonical_bytes(recipe), label="stage compiler recipe"
    )
    return manifest_path, recipe_path


def rederive_manifest_for_admission(
    manifest_path: Path, *, recipe_path: Path | None = None
) -> str:
    resolved_manifest = manifest_path.resolve(strict=True)
    resolved_recipe = (
        _recipe_path(resolved_manifest)
        if recipe_path is None
        else recipe_path.resolve(strict=True)
    )
    try:
        recipe = json.loads(resolved_recipe.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("stage compiler recipe is invalid JSON") from error
    expected_keys = {
        "schema",
        "compiler",
        "arguments",
        "manifest_logical_sha256",
        "manifest_physical_sha256",
        "recipe_sha256",
    }
    if not isinstance(recipe, dict) or set(recipe) != expected_keys:
        raise ValueError("stage compiler recipe schema differs")
    body = {key: value for key, value in recipe.items() if key != "recipe_sha256"}
    if (
        recipe["schema"] != COMPILER_RECIPE_SCHEMA
        or recipe["recipe_sha256"] != hashlib.sha256(canonical_bytes(body)).hexdigest()
    ):
        raise ValueError("stage compiler recipe identity differs")
    compiler = _stage_compilers().get(recipe["compiler"])
    if compiler is None:
        raise ValueError("stage compiler recipe names an unknown compiler")
    arguments = _replay_compiler_arguments(recipe["arguments"])
    rederived = compiler(**arguments)
    content = canonical_bytes(rederived.to_document())
    actual = resolved_manifest.read_bytes()
    if (
        content != actual
        or rederived.logical_sha256 != recipe["manifest_logical_sha256"]
        or hashlib.sha256(actual).hexdigest() != recipe["manifest_physical_sha256"]
    ):
        raise ValueError("stage manifest differs from compiler recipe replay")
    return str(recipe["recipe_sha256"])


def _stage_compilers() -> dict[str, Callable[..., StageManifest]]:
    return {
        "controls": build_controls_manifest,
        "controls_boundary": build_controls_boundary_manifest,
        "rq0_first_surface": build_rq0_first_surface_manifest,
        "rq0_inherited_surfaces": build_rq0_inherited_surfaces_manifest,
        "rq0_boundary": build_rq0_boundary_manifest,
        "rq0_bridge": build_rq0_bridge_manifest,
        "rq0_bridge_boundary": build_rq0_bridge_boundary_manifest,
        "rq1_surface": build_rq1_surface_manifest,
        "rq1_boundary": build_rq1_boundary_manifest,
        "rq1_confirmation": build_rq1_confirmation_manifest,
        "rq23_initial_surface": build_rq23_initial_surface_manifest,
        "rq23_refinement_surface": build_rq23_refinement_surface_manifest,
        "rq23_boundary": build_rq23_boundary_manifest,
        "rq23_confirmation": build_rq23_confirmation_manifest,
        "terminal_bridge": build_terminal_bridge_manifest,
        "terminal_bridge_boundary": build_terminal_bridge_boundary_manifest,
        "cachefix_rq0_anchor": build_cachefix_rq0_anchor_manifest,
        "cachefix_rq1_surface": build_cachefix_rq1_surface_manifest,
        "cachefix_rq1_confirmation": build_cachefix_rq1_confirmation_manifest,
        "cachefix_rq23_initial_surface": build_cachefix_rq23_initial_surface_manifest,
        "cachefix_rq23_refinement_surface": (
            build_cachefix_rq23_refinement_surface_manifest
        ),
        "cachefix_rq23_boundary": build_cachefix_rq23_boundary_manifest,
        "cachefix_rq23_confirmation": build_cachefix_rq23_confirmation_manifest,
    }


def _bind_compiler_arguments(
    arguments: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    normalized: dict[str, object] = {}
    call_arguments: dict[str, object] = {}
    for name, value in arguments.items():
        if isinstance(value, Path):
            resolved = value.resolve(strict=True)
            call_arguments[name] = resolved
            if resolved.is_file():
                content = resolved.read_bytes()
                normalized[name] = {
                    "kind": "file",
                    "path": str(resolved),
                    "physical_sha256": hashlib.sha256(content).hexdigest(),
                }
            elif resolved.is_dir():
                normalized[name] = {"kind": "directory", "path": str(resolved)}
            else:
                raise ValueError("compiler path input is neither file nor directory")
        elif isinstance(value, (str, int, float, bool)) or value is None:
            normalized[name] = {"kind": "value", "value": value}
            call_arguments[name] = value
        else:
            raise ValueError(f"unsupported compiler argument {name!r}")
    return normalized, call_arguments


def _replay_compiler_arguments(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("compiler recipe arguments are invalid")
    arguments: dict[str, object] = {}
    for name, binding in value.items():
        if not isinstance(name, str) or not isinstance(binding, dict):
            raise ValueError("compiler recipe argument binding is invalid")
        if binding.get("kind") == "value" and set(binding) == {"kind", "value"}:
            arguments[name] = binding["value"]
            continue
        if binding.get("kind") == "directory" and set(binding) == {"kind", "path"}:
            path = Path(binding["path"]).resolve(strict=True)
            if not path.is_dir():
                raise ValueError("compiler recipe directory input differs")
            arguments[name] = path
            continue
        if binding.get("kind") == "file" and set(binding) == {
            "kind",
            "path",
            "physical_sha256",
        }:
            path = Path(binding["path"]).resolve(strict=True)
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != binding["physical_sha256"]
            ):
                raise ValueError("compiler recipe file input differs")
            arguments[name] = path
            continue
        raise ValueError("compiler recipe argument schema differs")
    return arguments


def _recipe_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.name}.recipe.json")


def _control_job(
    backbone: str,
    index: int,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    *,
    source_sha256: str,
    stage: str = "controls",
    predecessor: SelectionBinding | None = None,
    exact_reuse: tuple[ExactReuse, ...] = (),
    reuse_selection_path: Path | None = None,
    run_revision: int = 0,
) -> JobContract:
    revision = "" if run_revision == 0 else f"_retry{run_revision:02d}"
    run_name = f"g6_native500m_{stage}_{backbone}_{index:02d}{revision}"
    experiment = build_control(
        backbone=backbone,
        embedding_learning_rate=embedding_learning_rate,
        deep_learning_rate=deep_learning_rate,
        run_name=run_name,
        seed=42,
    )
    environment = {"G6_NATIVE500M_SOURCE_SHA256": source_sha256}
    if reuse_selection_path is not None:
        environment.update(_reuse_selection_environment(reuse_selection_path))
    job = JobContract.create(
        job_id=f"{stage}:{backbone}:{index:02d}{revision}",
        stage=stage,
        schedule="constant" if backbone == "original_g1" else "annealed",
        parameters={
            "builder": "control",
            "runner": RUNNER,
            "run_name": run_name,
            "config_logical_sha256": experiment_logical_sha256(
                experiment, source_sha256=source_sha256
            ),
            "data_group": "g6-native500m-controls-v1",
            "environment": environment,
            "backbone": backbone,
            "embedding_learning_rate": embedding_learning_rate,
            "deep_learning_rate": deep_learning_rate,
            "seed": 42,
        },
        source_selection=predecessor,
        exact_reuse=exact_reuse,
    )
    return job


def _dependent_job(
    stage: str,
    predecessor: SelectionBinding,
    row: dict[str, object],
    *,
    source_sha256: str,
    exact_reuse: tuple[ExactReuse, ...] = (),
) -> JobContract:
    parameters = dict(row)
    job_id = parameters.pop("job_id", None)
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("dependent manifest row has no job ID")
    parameters.setdefault("builder", "semantic")
    parameters.setdefault("runner", RUNNER)
    environment = parameters.setdefault("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("dependent environment is invalid")
    environment.setdefault("G6_NATIVE500M_SOURCE_SHA256", source_sha256)
    experiment = build_semantic_treatment(
        backbone=parameters["backbone"],
        representation=parameters["representation"],
        embedding_learning_rate=parameters["embedding_learning_rate"],
        deep_learning_rate=parameters["deep_learning_rate"],
        num_levels=parameters["levels"],
        num_codes=parameters["shared_codes"],
        run_name=parameters["run_name"],
        seed=parameters["seed"],
        representation_width=parameters["representation_width"],
        collision_policy=parameters["collision_policy"],
        sid_lookup_initialization=parameters["sid_initialization"],
    )
    parameters["config_logical_sha256"] = experiment_logical_sha256(
        experiment, source_sha256=source_sha256
    )
    return JobContract.create(
        job_id=job_id,
        stage=stage,
        schedule="constant" if parameters["backbone"] == "original_g1" else "annealed",
        parameters=parameters,
        source_selection=predecessor,
        exact_reuse=exact_reuse,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "generated/g6-native500m/stage-manifests/controls.json"
        ),
    )
    parser.add_argument("--retry-revision", type=int, default=0)
    arguments = parser.parse_args()
    path, recipe_path = compile_and_persist_stage(
        arguments.output,
        compiler_name="controls",
        arguments={"retry_revision": arguments.retry_revision},
    )
    manifest = load_queue_manifest(path)
    print(f"{path}\n{recipe_path}\n{manifest.logical_sha256}")


if __name__ == "__main__":
    main()
