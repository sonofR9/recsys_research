import os
import runpy
import subprocess
from pathlib import Path

import pytest

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION


EXPERIMENT = (
    Path(__file__).resolve().parents[3] / "experiments/g1_sasrec_item_ids_likes"
)
MANIFEST = EXPERIMENT / "launchers/architecture/manifest.sh"
CONFIG = EXPERIMENT / "configs/rq_tuning_variant.py"
TUNE_RUN_SUFFIX = f"_ts{GENERATION_TRAINING_SEMANTICS_REVISION}_r2"

RQ7_TREATMENTS = {
    "none": (None, None, False),
    "learned_forward": ("forward", None, False),
    "learned_reverse": ("reverse", None, False),
    "learned_forward_reverse": (("forward", "reverse"), None, False),
    "rope_forward": (None, "forward", False),
    "rope_reverse": (None, "reverse", False),
    "alibi": (None, None, True),
    "rope_forward_alibi": (None, "forward", True),
    "rope_reverse_alibi": (None, "reverse", True),
    "learned_forward_alibi": ("forward", None, True),
    "learned_reverse_alibi": ("reverse", None, True),
    "rope_forward_learned_forward": ("forward", "forward", False),
    "rope_forward_learned_reverse": ("reverse", "forward", False),
    "rope_reverse_learned_forward": ("forward", "reverse", False),
    "rope_reverse_learned_reverse": ("reverse", "reverse", False),
    "rope_forward_learned_forward_alibi": ("forward", "forward", True),
    "rope_forward_learned_reverse_alibi": ("reverse", "forward", True),
    "rope_reverse_learned_forward_alibi": ("forward", "reverse", True),
    "rope_reverse_learned_reverse_alibi": ("reverse", "reverse", True),
}


def _position_rows() -> list[list[str]]:
    result = subprocess.run(
        ["bash", "-c", f'source "{MANIFEST}"; g1_manifest_rows'],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.split("|")
        for line in result.stdout.splitlines()
        if line.startswith("position|")
    ]


def test_rq7_manifest_has_every_exact_position_treatment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _position_rows()

    assert len(rows) == len(RQ7_TREATMENTS) == 19
    assert {row[1] for row in rows} == set(RQ7_TREATMENTS)
    assert {row[1]: row[6] for row in rows if row[6]} == {
        "learned_forward": "control/control"
    }

    for _, treatment, source, transformer_fields, experiment_fields, extras, alias in rows:
        assert experiment_fields == extras == ""
        if alias:
            assert source == "selected_quality_b1280"
            assert transformer_fields == ""
        else:
            assert transformer_fields == "alibi,rope,learned_positions"
        for name in tuple(os.environ):
            if name.startswith("G1_TUNE_"):
                monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("G1_DATASET_SIZE", "50m")
        monkeypatch.setenv("G1_TUNE_RUN", f"rq7_test_{treatment}{TUNE_RUN_SUFFIX}")
        monkeypatch.setenv("G1_TUNE_SOURCE_VARIANT", source)
        monkeypatch.setenv("G1_TUNE_TRANSFORMER_FIELDS", transformer_fields)
        monkeypatch.setenv("G1_TUNE_EXPERIMENT_FIELDS", experiment_fields)
        monkeypatch.setenv("G1_TUNE_EMBEDDING_LR", "0.016")
        monkeypatch.setenv("G1_TUNE_DEEP_LR", "0.006")
        for assignment in extras.split():
            name, value = assignment.split("=", 1)
            monkeypatch.setenv(name, value)

        experiment = runpy.run_path(str(CONFIG))["experiment"]
        transformer = experiment.transformer

        assert (
            transformer.learned_positions,
            transformer.rope,
            transformer.alibi,
        ) == RQ7_TREATMENTS[treatment]
        assert type(experiment).__name__ == "MuTransferGenerationExperiment"
        assert (experiment.mup_base_dim, experiment.mup_delta_dim) == (16, 32)
        assert experiment.item_embedding_dim == 64
        assert experiment.dataloader.batch_size == 1280
