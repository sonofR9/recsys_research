import copy

import pytest

from experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_cap_fit import (
    Rq13CapFitError,
    build_cap_fit,
    practical_cap_ceiling,
)


def _rq12() -> dict:
    artifacts = [
        {"validation_metrics": {"recall@100": value}}
        for value in (0.1367, 0.1343, 0.1363)
    ]
    return {
        "research_question": "RQ12 decoder-only query layout",
        "dataset_size": "500m",
        "methods": [
            {
                "method": "standard",
                "artifacts": artifacts,
                "mean_full_user_metrics": {
                    "recall@100": 0.13468336146286186,
                },
            }
        ],
    }


def _rq13() -> dict:
    values = {
        "one_example": (0.0766, 0.0774),
        "truncated_4": (0.1070, 0.1060),
        "truncated_8": (0.1164, 0.1157),
        "truncated_16": (0.1233, 0.1240),
    }
    winners = {
        treatment: {
            "deep_learning_rate": 0.012,
            "validation": {"recall@100": validation},
            "full_user_metrics": {"recall@100": full},
            "artifact_sha256": {
                "training_metadata.json": f"{treatment}-metadata",
                "final_metrics.json": f"{treatment}-metrics",
                "sweep.log": f"{treatment}-log",
            },
            "source_manifest_sha256": "source",
        }
        for treatment, (validation, full) in values.items()
    }
    return {
        "research_question": "RQ13 encoder-decoder prefix expansion",
        "dataset_size": "500m",
        "missing_initial_artifacts": [],
        "required_followups": [],
        "required_boundary_followups": [],
        "surface_winners": winners,
        "treatments": {
            treatment: {
                "artifacts": [
                    {"deep_learning_rate": rate} for rate in (0.006, 0.012, 0.024)
                ]
            }
            for treatment in values
        },
    }


def test_practical_cap_ceiling_combines_support_compute_and_extrapolation() -> None:
    counts = [100] * 5 + [45] * 5

    ceiling = practical_cap_ceiling(counts)

    assert ceiling["support"] == 100
    assert ceiling["compute"] >= 32
    assert ceiling["extrapolation"] == 32
    assert ceiling["selected"] == 32


def test_cap_fit_selects_only_from_validation_and_keeps_reader_target_distinct() -> (
    None
):
    evidence = build_cap_fit(_rq13(), _rq12(), [100] * 10, 14_960)

    assert (
        evidence["metric"]
        == "validation Recall@100 from validation-selected checkpoints"
    )
    assert evidence["selection_target"] == {
        "metric": "mean validation Recall@100",
        "control_values": [0.1367, 0.1343, 0.1363],
        "control_mean": pytest.approx(0.13576666666666667),
        "multiplier": 1.10,
        "value": pytest.approx(0.14934333333333335),
    }
    assert evidence["reader_success_target"] == {
        "metric": "mean full-user Recall@100",
        "control_mean": pytest.approx(0.13468336146286186),
        "multiplier": 1.10,
        "value": pytest.approx(0.14815169760914806),
    }
    assert set(evidence["fit_points"]) == {"1", "4", "8", "16"}
    assert evidence["selected_cap"] == 32
    assert 17 <= evidence["selected_cap"] <= evidence["practical_ceiling"]["selected"]

    changed_final = _rq13()
    for row in changed_final["surface_winners"].values():
        row["full_user_metrics"]["recall@100"] = 0.99
    changed = build_cap_fit(changed_final, _rq12(), [100] * 10, 14_960)
    assert changed["fit"] == evidence["fit"]
    assert changed["selected_cap"] == evidence["selected_cap"]
    assert len(evidence["sensitivity"]["fits"]) == 16
    assert all(
        set(fit["perturbations"]) == {"1", "4", "8", "16"}
        for fit in evidence["sensitivity"]["fits"]
    )
    assert len(evidence["leave_one_out"]["fits"]) == 4


@pytest.mark.parametrize(
    "failure", ["missing_cap", "wrong_control", "nan", "bad_history"]
)
def test_cap_fit_fails_closed_on_invalid_inputs(failure: str) -> None:
    rq13 = _rq13()
    rq12 = _rq12()
    history = [100] * 10
    if failure == "missing_cap":
        rq13["surface_winners"].pop("truncated_4")
    elif failure == "wrong_control":
        rq12["methods"][0]["artifacts"][0]["validation_metrics"]["recall@100"] = 0.2
    elif failure == "nan":
        rq13["surface_winners"]["truncated_4"]["validation"]["recall@100"] = float(
            "nan"
        )
    else:
        history = [0, 100]

    with pytest.raises(Rq13CapFitError):
        build_cap_fit(rq13, rq12, history, 14_960)


def test_full_user_metrics_cannot_replace_validation_fit_points() -> None:
    rq13 = _rq13()
    invalid = copy.deepcopy(rq13)
    for row in invalid["surface_winners"].values():
        row.pop("validation")

    with pytest.raises(Rq13CapFitError, match="validation"):
        build_cap_fit(invalid, _rq12(), [100] * 10, 14_960)


def test_cap_fit_rejects_unresolved_lr_surface_and_missing_hash_binding() -> None:
    unresolved = _rq13()
    unresolved["required_boundary_followups"] = ["cap4-high"]
    unresolved["required_followups"] = ["cap4-high"]
    with pytest.raises(Rq13CapFitError, match="unresolved"):
        build_cap_fit(unresolved, _rq12(), [100] * 10, 14_960)

    unbound = _rq13()
    unbound["surface_winners"]["truncated_4"].pop("artifact_sha256")
    with pytest.raises(Rq13CapFitError, match="binding"):
        build_cap_fit(unbound, _rq12(), [100] * 10, 14_960)

    with pytest.raises(Rq13CapFitError, match="reproduce logged"):
        build_cap_fit(_rq13(), _rq12(), [100] * 10, 14_959)
