from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.g3_pretrained_item_embeddings.analysis.native500m_report as report_module


_FAMILIES = {
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(family: str, order: int) -> dict[str, object]:
    recall = 0.1 if family == "baseline" else 0.099
    return {
        "row_id": f"{family}:winner",
        "job": {"family_id": family, "manifest_order": order},
        "selection_metrics": {"recall@100": recall, "ndcg@100": 0.05},
        "metrics": {"recall@100": recall, "ndcg@100": 0.05},
        "slices": {
            "item_frequency": {
                name: {"metrics": {"recall@100": value}}
                for name, value in (("head", 0.2), ("mid", 0.1), ("tail", 0.02))
            },
            "user_history": {
                name: {"metrics": {"recall@100": value}}
                for name, value in (("low", 0.08), ("mid", 0.1), ("high", 0.12))
            },
        },
    }


def _reference(
    root: Path,
    path: Path,
    logical_sha256: str,
    *,
    row_id: str | None = None,
) -> dict[str, object]:
    reference = {
        "role": "family_evidence",
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "logical_sha256": logical_sha256,
    }
    if row_id is not None:
        reference["row_id"] = row_id
    return reference


def _closure_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    selections = {}
    selection_paths = {}
    for order, family in enumerate(sorted(_FAMILIES)):
        path = tmp_path / f"{family}.json"
        path.write_text(json.dumps({"family": family}))
        document = {
            "family_id": family,
            "sha256": hashlib.sha256(f"logical:{family}".encode()).hexdigest(),
            "winner": _row(family, order),
        }
        selections[family] = document
        selection_paths[family] = path

    standalone = {
        family: _reference(
            tmp_path,
            selection_paths[family],
            selections[family]["sha256"],
            row_id=selections[family]["winner"]["row_id"],
        )
        for family in sorted(_FAMILIES)
    }
    baseline_reference = dict(standalone["baseline"])
    baseline_reference["role"] = "component_target_input"
    thresholds = {"recall@100": 0.1 * 0.1, "tail_recall@100": 0.02 * 0.1}
    state0_path = tmp_path / "state0.json"
    state1_path = tmp_path / "state1.json"
    state0_path.write_text("state zero")
    state1_path.write_text("state one")
    state0 = {
        "generation": 0,
        "standalone_selections": standalone,
        "component_targets": {
            "input": baseline_reference,
            "output": None,
            "metadata": None,
        },
        "included": {
            "input": baseline_reference,
            "output": None,
            "metadata": None,
        },
        "prior_state": None,
        "completed_transition": None,
        "gate_thresholds": thresholds,
        "most_specific_selection": baseline_reference,
        "next_conditional_family": "bridge_rq3_output",
    }
    state0_logical = "0" * 64
    prior_reference = _reference(tmp_path, state0_path, state0_logical)
    prior_reference["role"] = "compatibility_state"
    state1 = {
        **state0,
        "generation": 1,
        "prior_state": prior_reference,
        "completed_transition": {
            "decision": "omit",
            "selected_selection": baseline_reference,
            "predecessor_reference": baseline_reference,
        },
        "next_conditional_family": None,
    }

    def authenticated(path: Path, *, root: Path):
        assert root == tmp_path
        document = state0 if Path(path).name == state0_path.name else state1
        logical = state0_logical if document is state0 else "1" * 64
        state_path = state0_path if document is state0 else state1_path
        identity = SimpleNamespace(
            relative_path=state_path.relative_to(tmp_path).as_posix(),
            size_bytes=state_path.stat().st_size,
            physical_sha256=_sha256(state_path),
            logical_sha256=logical,
            generation=document["generation"],
            most_specific_selection_path=baseline_reference["path"],
            most_specific_selection_size_bytes=baseline_reference["size_bytes"],
            most_specific_selection_physical_sha256=baseline_reference["sha256"],
            most_specific_selection_logical_sha256=baseline_reference["logical_sha256"],
            most_specific_selected=SimpleNamespace(
                coordinate=SimpleNamespace(source_id=baseline_reference["row_id"])
            ),
        )
        return document, identity

    monkeypatch.setattr(report_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        report_module,
        "load_family_selection",
        lambda path: selections[Path(path).stem],
    )
    monkeypatch.setattr(
        report_module, "authenticate_compatibility_resolution", authenticated
    )
    monkeypatch.setattr(
        report_module,
        "_relative_noise_bands",
        lambda: {"recall@100": 0.1, "ndcg@100": 0.1},
    )
    return {
        "selection_paths": selection_paths,
        "conclusions": {
            number: (
                "Selection: control.",
                "Observed result: control.",
                "Conclusion: control.",
            )
            for number in range(1, 6)
        },
        "compatibility_resolution_path": state1_path,
        "selections": selections,
        "state0": state0,
        "state1": state1,
    }


def test_report_accepts_one_authenticated_closure_for_reuse_arithmetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _closure_fixture(tmp_path, monkeypatch)

    rendered = report_module.render_native500m_reports(
        selection_paths=arguments["selection_paths"],
        conclusions=arguments["conclusions"],
        compatibility_resolution_path=arguments["compatibility_resolution_path"],
    )

    assert "No compatible treatment qualified" in rendered.reader
    assert "| recall@100 | 0.100000 | 0.100000 | +0.000000 |" in rendered.reader
    assert "runtime" not in rendered.reader


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("path", "rq1_content_input.json"),
        ("size_bytes", 0),
        ("sha256", "f" * 64),
        ("logical_sha256", "e" * 64),
        ("row_id", "baseline:substituted"),
    ),
)
def test_report_rejects_substituted_standalone_selection_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    arguments = _closure_fixture(tmp_path, monkeypatch)
    arguments["state0"]["standalone_selections"]["baseline"][field] = replacement

    with pytest.raises(ValueError, match="standalone selection identity"):
        report_module.render_native500m_reports(
            selection_paths=arguments["selection_paths"],
            conclusions=arguments["conclusions"],
            compatibility_resolution_path=arguments["compatibility_resolution_path"],
        )


@pytest.mark.parametrize("change", ("missing", "extra"))
def test_report_rejects_missing_or_extra_closure_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    arguments = _closure_fixture(tmp_path, monkeypatch)
    standalone = arguments["state0"]["standalone_selections"]
    if change == "missing":
        standalone.pop("baseline")
    else:
        standalone["extra"] = dict(standalone["baseline"])

    with pytest.raises(ValueError, match="standalone selections"):
        report_module.render_native500m_reports(
            selection_paths=arguments["selection_paths"],
            conclusions=arguments["conclusions"],
            compatibility_resolution_path=arguments["compatibility_resolution_path"],
        )


@pytest.mark.parametrize("change", ("missing", "extra"))
def test_report_rejects_missing_or_extra_caller_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    arguments = _closure_fixture(tmp_path, monkeypatch)
    selection_paths = dict(arguments["selection_paths"])
    if change == "missing":
        selection_paths.pop("baseline")
    else:
        selection_paths["aggregate"] = tmp_path / "aggregate.json"

    with pytest.raises(ValueError, match="every approved family selection"):
        report_module.render_native500m_reports(
            selection_paths=selection_paths,
            conclusions=arguments["conclusions"],
            compatibility_resolution_path=arguments["compatibility_resolution_path"],
        )


def test_report_rejects_mixed_closure_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _closure_fixture(tmp_path, monkeypatch)
    arguments["state1"]["generation"] = 2

    with pytest.raises(ValueError, match="generation chain"):
        report_module.render_native500m_reports(
            selection_paths=arguments["selection_paths"],
            conclusions=arguments["conclusions"],
            compatibility_resolution_path=arguments["compatibility_resolution_path"],
        )


def test_report_rejects_stale_baseline_threshold_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _closure_fixture(tmp_path, monkeypatch)
    arguments["state0"]["gate_thresholds"] = {
        "recall@100": 0.02,
        "tail_recall@100": 0.002,
    }

    with pytest.raises(ValueError, match="threshold source"):
        report_module.render_native500m_reports(
            selection_paths=arguments["selection_paths"],
            conclusions=arguments["conclusions"],
            compatibility_resolution_path=arguments["compatibility_resolution_path"],
        )


def test_report_rejects_final_selection_outside_authenticated_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _closure_fixture(tmp_path, monkeypatch)
    arguments["state1"]["most_specific_selection"] = dict(
        arguments["state0"]["standalone_selections"]["rq1_content_input"]
    )

    with pytest.raises(ValueError, match="final selection identity"):
        report_module.render_native500m_reports(
            selection_paths=arguments["selection_paths"],
            conclusions=arguments["conclusions"],
            compatibility_resolution_path=arguments["compatibility_resolution_path"],
        )
