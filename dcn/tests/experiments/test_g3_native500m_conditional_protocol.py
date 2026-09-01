from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.g3_pretrained_item_embeddings.analysis import (
    native500m_evidence as evidence_module,
)
from experiments.g3_pretrained_item_embeddings.protocol.native500m import (
    AuthenticatedCompatibilityState,
    AuthenticatedResolvedConditionalPredecessor,
    CandidateResult,
    PROTOCOL_SHA256,
    authenticate_compatibility_state,
    authenticate_resolved_conditional_predecessor,
    authenticate_selected_coordinate,
    compile_baseline_rows,
    compile_boundary_rows,
    compile_nine_cell_family,
    family_spec,
    required_boundary_extensions,
    select_preliminary_winner,
    select_winner,
)

CONDITIONAL_FAMILIES = (
    "bridge_rq3_output",
    "bridge_rq4_metadata",
    "aggregate",
)


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _results(
    rows,
    *,
    winner_index: int,
    winning_recall: float = 0.13,
) -> tuple[CandidateResult, ...]:
    return tuple(
        CandidateResult(
            row=row,
            recall_at_100=winning_recall if index == winner_index else 0.12,
            ndcg_at_100=0.05 if index == winner_index else 0.04,
            best_epoch=min(8, row.horizon_epochs),
            epochs_trained=row.horizon_epochs,
        )
        for index, row in enumerate(rows)
    )


def _baseline_selected():
    rows = compile_baseline_rows()
    return authenticate_selected_coordinate(
        _results(rows, winner_index=1),
        expected_rows=rows,
    )


class _StateFactory:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.documents = {}
        self.selections = {}

    def _selected_reference(self, selected, *, role: str) -> tuple[dict, dict]:
        coordinate = selected.coordinate
        job = {
            "id": coordinate.source_id,
            "family_id": coordinate.family_id,
            "embedding_learning_rate": coordinate.embedding_learning_rate_text,
            "deep_learning_rate": coordinate.deep_learning_rate_text,
            "horizon_epochs": coordinate.horizon_epochs,
            "capacity": coordinate.capacity,
        }
        body = {
            "winner": {
                "row_id": coordinate.source_id,
                "job": job,
                "selection_metrics": {"recall@100": 0.13, "ndcg@100": 0.05},
                "metrics": {"recall@100": 0.13, "ndcg@100": 0.05},
            }
        }
        document = {
            **body,
            "sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
        }
        path = self.root / f"selection-{len(self.selections)}.json"
        path.write_bytes(_canonical_bytes(document))
        self.documents[path] = document
        self.selections[path] = selected
        reference = {
            "role": role,
            "path": path.relative_to(self.root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "logical_sha256": document["sha256"],
            "row_id": coordinate.source_id,
        }
        return reference, document

    def __call__(
        self,
        next_family: str | None,
        *,
        selected=None,
        generation: int = 0,
        prior_state=None,
        completed_winner=None,
        decision=None,
        suffix: str = "0",
    ):
        selected = _baseline_selected() if selected is None else selected
        prior_reference = None
        transition = None
        if generation == 0:
            most_specific, _ = self._selected_reference(
                selected, role="most_specific_selection"
            )
        else:
            completed_reference, completed_document = self._selected_reference(
                completed_winner, role="conditional_result"
            )
            winner = completed_document["winner"]
            prior_reference = {
                "role": "compatibility_state",
                "path": prior_state.relative_path,
                "size_bytes": prior_state.size_bytes,
                "sha256": prior_state.physical_sha256,
                "logical_sha256": prior_state.logical_sha256,
            }
            transition = {
                "family_id": prior_state.next_conditional_family,
                "selected_family_id": winner["job"]["family_id"],
                "selected_selection": completed_reference,
                "selected_row_id": winner["row_id"],
                "selected_job": winner["job"],
                "selection_metrics": winner["selection_metrics"],
                "metrics": winner["metrics"],
                "decision": decision,
            }
            if decision == "accept":
                most_specific = completed_reference
            else:
                most_specific = {
                    "role": "most_specific_selection",
                    "path": prior_state.most_specific_selection_path,
                    "size_bytes": prior_state.most_specific_selection_size_bytes,
                    "sha256": prior_state.most_specific_selection_physical_sha256,
                    "logical_sha256": prior_state.most_specific_selection_logical_sha256,
                    "row_id": selected.coordinate.source_id,
                }
        body = {
            "schema_version": 1,
            "kind": "g3_native500m_compatibility_state",
            "protocol_sha256": PROTOCOL_SHA256,
            "generation": generation,
            "prior_state": prior_reference,
            "completed_transition": transition,
            "most_specific_selection": most_specific,
            "next_conditional_family": next_family,
        }
        document = {
            **body,
            "sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
        }
        path = self.root / f"compatibility-{suffix}.json"
        path.write_bytes(_canonical_bytes(document))
        self.documents[path] = document

        def load(candidate, *, root):
            return self.documents[Path(candidate).resolve()]

        def authenticate(candidate, *, root):
            resolved = Path(candidate).resolve()
            return self.documents[resolved], self.selections[resolved]

        with patch.object(
            evidence_module, "load_compatibility_resolution", load
        ), patch.object(evidence_module, "authenticate_family_selection", authenticate):
            return authenticate_compatibility_state(path, root=self.root)


@pytest.fixture
def state_factory(tmp_path: Path) -> _StateFactory:
    return _StateFactory(tmp_path)


def _interior_index(rows) -> int:
    for index, row in enumerate(rows):
        result = CandidateResult(
            row, 0.13, 0.05, min(8, row.horizon_epochs), row.horizon_epochs
        )
        if not required_boundary_extensions(result, rows):
            return index
    raise AssertionError("conditional family has no interior candidate")


@pytest.mark.parametrize("family_id", CONDITIONAL_FAMILIES)
def test_conditional_family_initial_final_and_boundary_flows(
    family_id: str, state_factory: _StateFactory
) -> None:
    state = state_factory(family_id, suffix=family_id)
    predecessor = authenticate_resolved_conditional_predecessor(
        target_family_id=family_id,
        compatibility_state=state,
    )
    spec = family_spec(family_id)
    rows = compile_nine_cell_family(spec, predecessor.coordinate)
    interior_index = _interior_index(rows)
    interior_results = _results(rows, winner_index=interior_index)

    assert (
        select_preliminary_winner(
            interior_results,
            expected_rows=rows,
            predecessor=predecessor,
        ).row
        == rows[interior_index]
    )
    assert (
        select_winner(
            interior_results,
            expected_rows=rows,
            predecessor=predecessor,
        ).row
        == rows[interior_index]
    )
    authenticated = authenticate_selected_coordinate(
        interior_results,
        expected_rows=rows,
        predecessor=predecessor,
    )
    assert authenticated.conditional_state_identity == state.identity_sha256
    assert authenticated.conditional_generation == state.generation
    assert authenticated.conditional_family_id == family_id

    boundary_index = max(
        range(len(rows)),
        key=lambda index: rows[index].embedding_learning_rate,
    )
    boundary_results = _results(rows, winner_index=boundary_index)
    boundary_winner = boundary_results[boundary_index]
    requests = required_boundary_extensions(boundary_winner, rows)
    extension = compile_boundary_rows(
        boundary_winner,
        rows,
        existing_results=boundary_results,
        predecessor=predecessor,
        requests=requests,
    )
    extension_results = _results(extension, winner_index=0, winning_recall=0.12)
    assert (
        select_winner(
            (*boundary_results, *extension_results),
            expected_rows=(*rows, *extension),
            predecessor=predecessor,
        )
        == boundary_winner
    )


def test_conditional_state_authorizes_only_its_exact_next_role(
    state_factory: _StateFactory,
) -> None:
    state = state_factory("bridge_rq3_output")
    predecessor = authenticate_resolved_conditional_predecessor(
        target_family_id="bridge_rq3_output",
        compatibility_state=state,
    )

    assert predecessor.synthetic_role == "aggregate_selected_input"
    assert predecessor.actual_selected.coordinate.family_id == "baseline"
    assert (
        predecessor.coordinate.source_id
        == predecessor.actual_selected.coordinate.source_id
    )
    assert predecessor.coordinate.learning_rate_pair == (
        predecessor.actual_selected.coordinate.learning_rate_pair
    )
    assert predecessor.coordinate.horizon_epochs == (
        predecessor.actual_selected.coordinate.horizon_epochs
    )
    assert (
        predecessor.coordinate.capacity
        == predecessor.actual_selected.coordinate.capacity
    )
    with pytest.raises(ValueError, match="does not authorize"):
        authenticate_resolved_conditional_predecessor(
            target_family_id="bridge_rq4_metadata",
            compatibility_state=state,
        )
    with pytest.raises(ValueError, match="only valid for conditional"):
        authenticate_resolved_conditional_predecessor(
            target_family_id="rq1_content_input",
            compatibility_state=state,
        )

    concrete_rows = compile_nine_cell_family(
        family_spec("rq1_content_input"),
        state.most_specific_selected.coordinate,
    )
    with pytest.raises(ValueError, match="authenticated predecessor"):
        select_winner(
            _results(concrete_rows, winner_index=_interior_index(concrete_rows)),
            expected_rows=concrete_rows,
            predecessor=predecessor,
        )

    conditional_rows = compile_nine_cell_family(
        family_spec("bridge_rq3_output"),
        predecessor.coordinate,
    )
    with pytest.raises(ValueError, match="resolved predecessor"):
        select_winner(
            _results(
                conditional_rows,
                winner_index=_interior_index(conditional_rows),
            ),
            expected_rows=conditional_rows,
            predecessor=state.most_specific_selected,
        )


def test_transition_requires_the_exact_prior_authorized_winner(
    state_factory: _StateFactory,
) -> None:
    initial = state_factory("bridge_rq3_output")
    predecessor = authenticate_resolved_conditional_predecessor(
        target_family_id="bridge_rq3_output",
        compatibility_state=initial,
    )
    rows = compile_nine_cell_family(
        family_spec("bridge_rq3_output"), predecessor.coordinate
    )
    results = _results(rows, winner_index=_interior_index(rows))
    completed = authenticate_selected_coordinate(
        results,
        expected_rows=rows,
        predecessor=predecessor,
    )
    transitioned = state_factory(
        "bridge_rq4_metadata",
        selected=completed,
        generation=1,
        prior_state=initial,
        completed_winner=completed,
        decision="accept",
        suffix="1",
    )
    next_predecessor = authenticate_resolved_conditional_predecessor(
        target_family_id="bridge_rq4_metadata",
        compatibility_state=transitioned,
    )

    assert next_predecessor.actual_selected is completed
    assert (
        next_predecessor.compatibility_state.identity_sha256 != initial.identity_sha256
    )
    omitted = state_factory(
        "bridge_rq4_metadata",
        selected=initial.most_specific_selected,
        generation=1,
        prior_state=initial,
        completed_winner=completed,
        decision="omit",
        suffix="omitted",
    )
    omitted_predecessor = authenticate_resolved_conditional_predecessor(
        target_family_id="bridge_rq4_metadata",
        compatibility_state=omitted,
    )
    assert omitted_predecessor.actual_selected is initial.most_specific_selected
    with pytest.raises(ValueError, match="winner identity|prior state"):
        state_factory(
            "aggregate",
            selected=_baseline_selected(),
            generation=1,
            prior_state=initial,
            completed_winner=_baseline_selected(),
            decision="accept",
            suffix="forged",
        )


def test_unverified_forged_and_stale_states_cannot_mint_capabilities(
    state_factory: _StateFactory,
) -> None:
    selected = _baseline_selected()
    with pytest.raises(ValueError, match="not produced by authentication"):
        AuthenticatedCompatibilityState(
            relative_path="compatibility.json",
            size_bytes=1,
            physical_sha256="a" * 64,
            logical_sha256="b" * 64,
            generation=0,
            most_specific_selected=selected,
            most_specific_job_sha256="c" * 64,
            most_specific_selection_path="selected.json",
            most_specific_selection_size_bytes=1,
            most_specific_selection_physical_sha256="e" * 64,
            most_specific_selection_logical_sha256="f" * 64,
            next_conditional_family="aggregate",
            decision=None,
            identity_sha256="d" * 64,
            protocol_sha256=PROTOCOL_SHA256,
            _seal=object(),
        )
    with pytest.raises(ValueError, match="not produced by authentication"):
        AuthenticatedResolvedConditionalPredecessor(
            target_family_id="aggregate",
            synthetic_role="forged-role",
            actual_selected=selected,
            compatibility_state=state_factory("aggregate"),
            protocol_sha256=PROTOCOL_SHA256,
            _seal=object(),
        )
    stale = state_factory("aggregate", suffix="stale")
    object.__setattr__(stale, "_protocol_sha256", "0" * 64)
    with pytest.raises(ValueError, match="current authenticated state"):
        authenticate_resolved_conditional_predecessor(
            target_family_id="aggregate",
            compatibility_state=stale,
        )


def test_fixed_loader_factory_binds_logical_and_physical_state_identity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _baseline_selected()
    selection_path = tmp_path / "baseline-selection.json"
    selection_document = {
        "sha256": "2" * 64,
        "winner": {
            "row_id": selected.coordinate.source_id,
            "job": {"family_id": selected.coordinate.family_id},
        },
    }
    selection_path.write_bytes(_canonical_bytes(selection_document))
    reference = {
        "role": "aggregate_input",
        "path": selection_path.relative_to(tmp_path).as_posix(),
        "size_bytes": selection_path.stat().st_size,
        "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "logical_sha256": selection_document["sha256"],
        "row_id": selected.coordinate.source_id,
    }
    body = {
        "schema_version": 1,
        "kind": "g3_native500m_compatibility_state",
        "protocol_sha256": PROTOCOL_SHA256,
        "generation": 0,
        "prior_state": None,
        "completed_transition": None,
        "most_specific_selection": reference,
        "next_conditional_family": "bridge_rq3_output",
    }
    document = {
        **body,
        "sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }
    state_path = tmp_path / "compatibility.json"
    state_path.write_bytes(_canonical_bytes(document))
    monkeypatch.setattr(
        evidence_module,
        "load_compatibility_resolution",
        lambda path, root: document,
    )
    monkeypatch.setattr(
        evidence_module,
        "authenticate_family_selection",
        lambda path, root: (selection_document, selected),
    )

    state = authenticate_compatibility_state(state_path, root=tmp_path)

    assert state.relative_path == "compatibility.json"
    assert state.logical_sha256 == document["sha256"]
    assert state.physical_sha256 == hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert (
        state.most_specific_job_sha256
        == hashlib.sha256(
            _canonical_bytes(selection_document["winner"]["job"])
        ).hexdigest()
    )

    forged = document | {"sha256": "0" * 64}
    state_path.write_bytes(_canonical_bytes(forged))
    monkeypatch.setattr(
        evidence_module,
        "load_compatibility_resolution",
        lambda path, root: forged,
    )
    with pytest.raises(ValueError, match="logical identity"):
        authenticate_compatibility_state(state_path, root=tmp_path)
    invalid_body = body | {"generation": False}
    invalid = {
        **invalid_body,
        "sha256": hashlib.sha256(_canonical_bytes(invalid_body)).hexdigest(),
    }
    state_path.write_bytes(_canonical_bytes(invalid))
    monkeypatch.setattr(
        evidence_module,
        "load_compatibility_resolution",
        lambda path, root: invalid,
    )
    with pytest.raises(ValueError, match="generation is invalid"):
        authenticate_compatibility_state(state_path, root=tmp_path)
