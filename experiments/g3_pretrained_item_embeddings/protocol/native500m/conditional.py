from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Literal, Mapping

from .compiler import SelectedCoordinate
from .constants import PROTOCOL_SHA256, family_spec
from .selection import AuthenticatedSelectedCoordinate


ConditionalDecision = Literal["accept", "omit"]
_SHA256 = re.compile(r"[0-9a-f]{64}")


class AuthenticatedCompatibilityState:
    __slots__ = (
        "_decision",
        "_generation",
        "_identity_sha256",
        "_logical_sha256",
        "_most_specific_job_sha256",
        "_most_specific_selection_logical_sha256",
        "_most_specific_selection_path",
        "_most_specific_selection_physical_sha256",
        "_most_specific_selection_size_bytes",
        "_most_specific_selected",
        "_next_conditional_family",
        "_physical_sha256",
        "_protocol_sha256",
        "_relative_path",
        "_size_bytes",
    )

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> AuthenticatedCompatibilityState:
        raise ValueError("compatibility state was not produced by authentication")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("authenticated compatibility states are immutable")

    @property
    def relative_path(self) -> str:
        return self._relative_path

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def physical_sha256(self) -> str:
        return self._physical_sha256

    @property
    def logical_sha256(self) -> str:
        return self._logical_sha256

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def most_specific_selected(self) -> AuthenticatedSelectedCoordinate:
        return self._most_specific_selected

    @property
    def most_specific_job_sha256(self) -> str:
        return self._most_specific_job_sha256

    @property
    def most_specific_selection_path(self) -> str:
        return self._most_specific_selection_path

    @property
    def most_specific_selection_size_bytes(self) -> int:
        return self._most_specific_selection_size_bytes

    @property
    def most_specific_selection_physical_sha256(self) -> str:
        return self._most_specific_selection_physical_sha256

    @property
    def most_specific_selection_logical_sha256(self) -> str:
        return self._most_specific_selection_logical_sha256

    @property
    def next_conditional_family(self) -> str | None:
        return self._next_conditional_family

    @property
    def decision(self) -> ConditionalDecision | None:
        return self._decision

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    @property
    def protocol_sha256(self) -> str:
        return self._protocol_sha256


class AuthenticatedResolvedConditionalPredecessor:
    __slots__ = (
        "_actual_selected",
        "_compatibility_state",
        "_protocol_sha256",
        "_synthetic_role",
        "_target_family_id",
    )

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> AuthenticatedResolvedConditionalPredecessor:
        raise ValueError("resolved predecessor was not produced by authentication")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("authenticated resolved predecessors are immutable")

    @property
    def target_family_id(self) -> str:
        return self._target_family_id

    @property
    def synthetic_role(self) -> str:
        return self._synthetic_role

    @property
    def actual_selected(self) -> AuthenticatedSelectedCoordinate:
        return self._actual_selected

    @property
    def compatibility_state(self) -> AuthenticatedCompatibilityState:
        return self._compatibility_state

    @property
    def protocol_sha256(self) -> str:
        return self._protocol_sha256

    @property
    def coordinate(self) -> SelectedCoordinate:
        actual = self.actual_selected.coordinate
        return SelectedCoordinate.create(
            source_id=actual.source_id,
            family_id=self.synthetic_role,
            embedding_learning_rate=actual.embedding_learning_rate_text,
            deep_learning_rate=actual.deep_learning_rate_text,
            horizon_epochs=actual.horizon_epochs,
            capacity=actual.capacity,
        )


def authenticate_resolved_conditional_predecessor(
    *,
    target_family_id: str,
    compatibility_state: AuthenticatedCompatibilityState,
) -> AuthenticatedResolvedConditionalPredecessor:
    spec = family_spec(target_family_id)
    if not spec.conditional:
        raise ValueError(
            "resolved predecessors are only valid for conditional families"
        )
    if (
        not isinstance(compatibility_state, AuthenticatedCompatibilityState)
        or compatibility_state.protocol_sha256 != PROTOCOL_SHA256
    ):
        raise ValueError(
            "conditional predecessor requires a current authenticated state"
        )
    if compatibility_state.next_conditional_family != target_family_id:
        raise ValueError("compatibility state does not authorize this next family")
    actual = compatibility_state.most_specific_selected
    if actual.protocol_sha256 != PROTOCOL_SHA256:
        raise ValueError("compatibility state contains a stale selected coordinate")
    authenticated = object.__new__(AuthenticatedResolvedConditionalPredecessor)
    values = {
        "_target_family_id": target_family_id,
        "_synthetic_role": spec.search_predecessor_id,
        "_actual_selected": actual,
        "_compatibility_state": compatibility_state,
        "_protocol_sha256": PROTOCOL_SHA256,
    }
    for name, value in values.items():
        object.__setattr__(authenticated, name, value)
    return authenticated


def authenticate_compatibility_state(
    path: Path,
    *,
    root: Path,
) -> AuthenticatedCompatibilityState:
    from experiments.g3_pretrained_item_embeddings.analysis.native500m_evidence import (
        authenticate_family_selection,
        load_compatibility_resolution,
    )

    root = root.resolve(strict=True)
    if path.is_symlink():
        raise ValueError("compatibility state is not a regular project artifact")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError("compatibility state is not a regular project artifact")
    state_bytes = resolved.read_bytes()
    snapshot = _load_json_bytes(state_bytes)
    document = load_compatibility_resolution(resolved, root=root)
    if snapshot != document:
        raise ValueError("compatibility state bytes changed during authentication")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "g3_native500m_compatibility_state"
        or document.get("protocol_sha256") != PROTOCOL_SHA256
    ):
        raise ValueError("compatibility state identity differs")
    body = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _canonical_sha256(body):
        raise ValueError("compatibility state logical identity differs")
    generation = document.get("generation")
    if type(generation) is not int or generation < 0:
        raise ValueError("compatibility state generation is invalid")
    relative_path = resolved.relative_to(root).as_posix()
    state_size = len(state_bytes)
    state_physical_sha256 = hashlib.sha256(state_bytes).hexdigest()
    state_logical_sha256 = str(document.get("sha256"))
    next_family = document.get("next_conditional_family")
    if next_family is not None and not isinstance(next_family, str):
        raise ValueError("compatibility state next family differs")

    if generation == 0:
        if (
            document.get("prior_state") is not None
            or document.get("completed_transition") is not None
        ):
            raise ValueError("initial compatibility state contains a transition")
        selection_document, selected, selection_identity = _selected_reference(
            document.get("most_specific_selection"),
            root=root,
            authenticate=authenticate_family_selection,
        )
        decision = None
        prior_state = None
        completed_winner = None
    else:
        prior_reference = document.get("prior_state")
        if not isinstance(prior_reference, dict):
            raise ValueError("compatibility transition prior state differs")
        prior_path = root / str(prior_reference.get("path"))
        prior_state = authenticate_compatibility_state(prior_path, root=root)
        _validate_state_reference(prior_reference, prior_state)
        transition = document.get("completed_transition")
        if not isinstance(transition, dict):
            raise ValueError("compatibility transition evidence differs")
        completed_document, completed_winner, completed_identity = _selected_reference(
            transition.get("selected_selection"),
            root=root,
            authenticate=authenticate_family_selection,
        )
        winner = completed_document["winner"]
        if (
            transition.get("family_id") != prior_state.next_conditional_family
            or transition.get("selected_family_id") != winner["job"]["family_id"]
            or transition.get("selected_family_id")
            != prior_state.next_conditional_family
            or transition.get("selected_row_id") != winner["row_id"]
            or transition.get("selected_job") != winner["job"]
            or transition.get("selection_metrics") != winner["selection_metrics"]
            or transition.get("metrics") != winner["metrics"]
        ):
            raise ValueError("compatibility transition winner identity differs")
        decision = transition.get("decision")
        if decision == "accept":
            selected = completed_winner
            selection_document = completed_document
            selection_identity = completed_identity
        elif decision == "omit":
            selected = prior_state.most_specific_selected
            selection_document = {"winner": {"job": None}}
            selection_identity = (
                prior_state.most_specific_selection_path,
                prior_state.most_specific_selection_size_bytes,
                prior_state.most_specific_selection_physical_sha256,
                prior_state.most_specific_selection_logical_sha256,
            )
        else:
            raise ValueError("compatibility transition decision differs")
        _validate_selected_reference_identity(
            document.get("most_specific_selection"),
            selection_identity,
        )

    job_sha256 = (
        prior_state.most_specific_job_sha256
        if generation > 0 and decision == "omit"
        else _canonical_sha256(selection_document["winner"]["job"])
    )
    state_inputs = {
        "relative_path": relative_path,
        "size_bytes": state_size,
        "physical_sha256": state_physical_sha256,
        "logical_sha256": state_logical_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "generation": generation,
        "most_specific_selected": selected,
        "most_specific_job_sha256": job_sha256,
        "most_specific_selection_path": selection_identity[0],
        "most_specific_selection_size_bytes": selection_identity[1],
        "most_specific_selection_physical_sha256": selection_identity[2],
        "most_specific_selection_logical_sha256": selection_identity[3],
        "next_conditional_family": next_family,
        "prior_state": prior_state,
        "completed_winner": completed_winner,
        "decision": decision,
    }
    _validate_state_inputs(**state_inputs)
    identity = _compatibility_state_identity(**state_inputs)
    authenticated = object.__new__(AuthenticatedCompatibilityState)
    values = {
        "_relative_path": relative_path,
        "_size_bytes": state_size,
        "_physical_sha256": state_physical_sha256,
        "_logical_sha256": state_logical_sha256,
        "_generation": generation,
        "_most_specific_selected": selected,
        "_most_specific_job_sha256": job_sha256,
        "_most_specific_selection_path": selection_identity[0],
        "_most_specific_selection_size_bytes": selection_identity[1],
        "_most_specific_selection_physical_sha256": selection_identity[2],
        "_most_specific_selection_logical_sha256": selection_identity[3],
        "_next_conditional_family": next_family,
        "_decision": decision,
        "_identity_sha256": identity,
        "_protocol_sha256": PROTOCOL_SHA256,
    }
    for name, value in values.items():
        object.__setattr__(authenticated, name, value)
    return authenticated


def _compatibility_state_identity(
    *,
    relative_path: str,
    size_bytes: int,
    physical_sha256: str,
    logical_sha256: str,
    protocol_sha256: str,
    generation: int,
    most_specific_selected: AuthenticatedSelectedCoordinate,
    most_specific_job_sha256: str,
    most_specific_selection_path: str,
    most_specific_selection_size_bytes: int,
    most_specific_selection_physical_sha256: str,
    most_specific_selection_logical_sha256: str,
    next_conditional_family: str | None,
    prior_state: AuthenticatedCompatibilityState | None,
    completed_winner: AuthenticatedSelectedCoordinate | None,
    decision: ConditionalDecision | None,
) -> str:
    return _canonical_sha256(
        {
            "relative_path": relative_path,
            "size_bytes": size_bytes,
            "physical_sha256": physical_sha256,
            "logical_sha256": logical_sha256,
            "protocol_sha256": protocol_sha256,
            "generation": generation,
            "most_specific_family_id": most_specific_selected.coordinate.family_id,
            "most_specific_row_id": most_specific_selected.coordinate.source_id,
            "most_specific_learning_rates": most_specific_selected.coordinate.learning_rate_pair,
            "most_specific_horizon": most_specific_selected.coordinate.horizon_epochs,
            "most_specific_capacity": most_specific_selected.coordinate.capacity,
            "most_specific_job_sha256": most_specific_job_sha256,
            "most_specific_selection_path": most_specific_selection_path,
            "most_specific_selection_size_bytes": most_specific_selection_size_bytes,
            "most_specific_selection_physical_sha256": (
                most_specific_selection_physical_sha256
            ),
            "most_specific_selection_logical_sha256": (
                most_specific_selection_logical_sha256
            ),
            "next_conditional_family": next_conditional_family,
            "prior_state_identity": (
                None if prior_state is None else prior_state.identity_sha256
            ),
            "completed_family_id": (
                None
                if completed_winner is None
                else completed_winner.coordinate.family_id
            ),
            "completed_row_id": (
                None
                if completed_winner is None
                else completed_winner.coordinate.source_id
            ),
            "decision": decision,
        }
    )


def _validate_state_inputs(
    *,
    relative_path: str,
    size_bytes: int,
    physical_sha256: str,
    logical_sha256: str,
    protocol_sha256: str,
    generation: int,
    most_specific_selected: AuthenticatedSelectedCoordinate,
    most_specific_job_sha256: str,
    most_specific_selection_path: str,
    most_specific_selection_size_bytes: int,
    most_specific_selection_physical_sha256: str,
    most_specific_selection_logical_sha256: str,
    next_conditional_family: str | None,
    prior_state: AuthenticatedCompatibilityState | None,
    completed_winner: AuthenticatedSelectedCoordinate | None,
    decision: ConditionalDecision | None,
) -> None:
    path = PurePosixPath(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts:
        raise ValueError("compatibility state path is not project-relative")
    if type(size_bytes) is not int or size_bytes < 1:
        raise ValueError("compatibility state size is invalid")
    if any(
        _SHA256.fullmatch(value) is None
        for value in (
            physical_sha256,
            logical_sha256,
            most_specific_job_sha256,
            most_specific_selection_physical_sha256,
            most_specific_selection_logical_sha256,
        )
    ):
        raise ValueError("compatibility state SHA-256 identity is invalid")
    if protocol_sha256 != PROTOCOL_SHA256:
        raise ValueError("compatibility state protocol identity is stale")
    if (
        not isinstance(most_specific_selected, AuthenticatedSelectedCoordinate)
        or most_specific_selected.protocol_sha256 != PROTOCOL_SHA256
    ):
        raise ValueError("compatibility state selected coordinate is not authenticated")
    selection_path = PurePosixPath(most_specific_selection_path)
    if (
        not most_specific_selection_path
        or selection_path.is_absolute()
        or ".." in selection_path.parts
        or type(most_specific_selection_size_bytes) is not int
        or most_specific_selection_size_bytes < 1
    ):
        raise ValueError("compatibility selected artifact identity is invalid")
    if next_conditional_family is not None:
        if not family_spec(next_conditional_family).conditional:
            raise ValueError("compatibility state next family is not conditional")
    if type(generation) is not int or generation < 0:
        raise ValueError("compatibility state generation is invalid")
    if generation == 0:
        if (
            prior_state is not None
            or completed_winner is not None
            or decision is not None
        ):
            raise ValueError("initial compatibility state cannot contain a transition")
        return
    if (
        not isinstance(prior_state, AuthenticatedCompatibilityState)
        or not isinstance(completed_winner, AuthenticatedSelectedCoordinate)
        or decision not in {"accept", "omit"}
    ):
        raise ValueError("compatibility transition is not authenticated")
    if generation != prior_state.generation + 1:
        raise ValueError("compatibility transition generation is not consecutive")
    if completed_winner.protocol_sha256 != PROTOCOL_SHA256:
        raise ValueError("compatibility transition winner is stale")
    if completed_winner.conditional_state_identity != prior_state.identity_sha256:
        raise ValueError("conditional winner was not selected from the prior state")
    if completed_winner.coordinate.family_id != prior_state.next_conditional_family:
        raise ValueError("conditional winner does not complete the authorized family")
    if decision == "accept" and most_specific_selected is not completed_winner:
        raise ValueError("accepted transition must advance to its conditional winner")
    if (
        decision == "omit"
        and most_specific_selected is not prior_state.most_specific_selected
    ):
        raise ValueError("omitted transition must retain the prior predecessor")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _selected_reference(reference: object, *, root: Path, authenticate):
    if not isinstance(reference, dict) or set(reference) != {
        "role",
        "path",
        "size_bytes",
        "sha256",
        "logical_sha256",
        "row_id",
    }:
        raise ValueError("compatibility selected reference schema differs")
    path = root / str(reference["path"])
    if path.is_symlink():
        raise ValueError("compatibility selected artifact identity differs")
    resolved = path.resolve(strict=True)
    selected_bytes = resolved.read_bytes()
    selected_physical_sha256 = hashlib.sha256(selected_bytes).hexdigest()
    if (
        not resolved.is_file()
        or not resolved.is_relative_to(root)
        or type(reference["size_bytes"]) is not int
        or len(selected_bytes) != reference["size_bytes"]
        or selected_physical_sha256 != reference["sha256"]
    ):
        raise ValueError("compatibility selected artifact identity differs")
    snapshot = _load_json_bytes(selected_bytes)
    document, selected = authenticate(resolved, root=root)
    if (
        document != snapshot
        or document.get("sha256") != reference["logical_sha256"]
        or selected.selected_result.row.id != reference["row_id"]
    ):
        raise ValueError("compatibility selected logical identity differs")
    identity = (
        resolved.relative_to(root).as_posix(),
        len(selected_bytes),
        selected_physical_sha256,
        str(document["sha256"]),
    )
    return document, selected, identity


def _validate_state_reference(
    reference: Mapping[str, object],
    state: AuthenticatedCompatibilityState,
) -> None:
    expected = {
        "role": "compatibility_state",
        "path": state.relative_path,
        "size_bytes": state.size_bytes,
        "sha256": state.physical_sha256,
        "logical_sha256": state.logical_sha256,
    }
    if dict(reference) != expected:
        raise ValueError("compatibility transition prior-state identity differs")


def _validate_selected_reference_identity(
    reference: object,
    identity: tuple[str, int, str, str],
) -> None:
    if not isinstance(reference, dict):
        raise ValueError("compatibility most-specific selection differs")
    actual = (
        str(reference.get("path")),
        reference.get("size_bytes"),
        str(reference.get("sha256")),
        str(reference.get("logical_sha256")),
    )
    if actual != identity:
        raise ValueError("compatibility most-specific selection differs")


def _load_json_bytes(value: bytes) -> object:
    try:
        return json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("compatibility state bytes are invalid") from error
