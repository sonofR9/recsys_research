import json
import os
import shutil
import subprocess
from pathlib import Path

from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/g1_sasrec_item_ids_likes"


def _queue_stub(tmp_path: Path) -> Path:
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\" >&2; return 0; }\n"
        "drain() { return 0; }\n"
    )
    return queue


def _run(tmp_path: Path, **environment: str) -> subprocess.CompletedProcess:
    launchers = tmp_path / "experiments/g1_sasrec_item_ids_likes/launchers"
    shutil.copytree(EXPERIMENT / "launchers", launchers)
    generated_logs = tmp_path / "generated/logs"
    generated_logs.mkdir(parents=True)
    for rate in ("0p006", "0p012", "0p024"):
        name = (
            f"g1_rqtune_rqfinal_ffn_gelu171_e0p064_d{rate}" "_b1280_cap40_ts2_r3_500m"
        )
        (generated_logs / name).mkdir()
    (launchers / "verify_artifact.py").write_text(
        "import sys\n" "for line in sys.stdin:\n" "    print(0, flush=True)\n"
    )
    return subprocess.run(
        ["bash", str(launchers / "ffn/activation_depth_500m.sh")],
        capture_output=True,
        text=True,
        env=os.environ
        | {"G1_TRAINING_QUEUE_LIBRARY": str(_queue_stub(tmp_path))}
        | environment,
    )


def test_launcher_submits_the_approved_activation_depth_matrix(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result.returncode == 0
    assert result.stderr.count("ENQUEUE ") == 27
    lines = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    assert {
        (
            line.split("G1_TUNE_SOURCE_VARIANT=")[1].split()[0],
            line.split("G1_TUNE_NUM_LAYERS=")[1].split()[0],
            line.split("G1_TUNE_FFN_DIM=")[1].split()[0],
        )
        for line in lines
    } == {
        ("ffn_relu", "2", "171"),
        ("ffn_silu", "2", "171"),
        ("ffn_reglu", "2", "114"),
        ("ffn_geglu", "2", "114"),
        ("ffn_swiglu_dropout", "2", "114"),
        ("ffn_gelu", "4", "171"),
        ("ffn_swiglu_dropout", "4", "114"),
        ("ffn_gelu", "8", "171"),
        ("ffn_swiglu_dropout", "8", "114"),
    }
    assert {line.split("G1_TUNE_DEEP_LR=")[1].split()[0] for line in lines} == {
        "0.006",
        "0.012",
        "0.024",
    }
    assert all("G1_TUNE_EMBEDDING_LR=0.064" in line for line in lines)
    assert all("G1_TUNE_BATCH_SIZE=1280" in line for line in lines)
    assert all("G1_TUNE_EPOCHS=40" in line for line in lines)
    assert result.stdout.count("=== reused compatible generated/logs/") == 3


def test_launcher_arm_selection_is_configurable(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        G1_FFN_STUDY_ARMS="silu:2 geglu:8",
        G1_FFN_STUDY_DEEP_LRS="0.003 0.006 0.012",
    )

    assert result.returncode == 0
    assert result.stderr.count("ENQUEUE ") == 6
    assert "G1_TUNE_SOURCE_VARIANT=ffn_silu" in result.stderr
    assert "G1_TUNE_SOURCE_VARIANT=ffn_geglu" in result.stderr
    assert "G1_TUNE_TRANSFORMER_FIELDS=ffn,gated_ffn_dropout" in result.stderr
    assert "G1_TUNE_NUM_LAYERS=8" in result.stderr
    assert "G1_TUNE_DEEP_LR=0.003" in result.stderr


def test_launcher_targets_the_gelu_depth_two_upper_boundary(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        G1_FFN_STUDY_ARMS="gelu:2",
        G1_FFN_STUDY_DEEP_LRS="0.048 0.096",
    )

    assert result.returncode == 0
    lines = [line for line in result.stderr.splitlines() if line.startswith("ENQUEUE ")]
    assert [line.split()[1] for line in lines] == [
        "g1_rqtune_rqffnact_gelu_l2_w171_e0p064_d0p048_b1280_cap40_ts2_r4_500m",
        "g1_rqtune_rqffnact_gelu_l2_w171_e0p064_d0p096_b1280_cap40_ts2_r4_500m",
    ]
    for line, deep_lr in zip(lines, (0.048, 0.096), strict=True):
        assignments = line.split()[2:]
        parsed = verify_artifact._tuning_assignments(assignments)
        experiment = verify_artifact._tuning_experiment("500m", parsed)

        assert parsed["G1_TUNE_SOURCE_VARIANT"] == "ffn_gelu"
        assert parsed["G1_TUNE_TRANSFORMER_FIELDS"] == "ffn"
        assert experiment.transformer.ffn == "gelu"
        assert experiment.transformer.ffn_intermediate_dim == 171
        assert experiment.transformer.ffn_dropout == 0.1
        assert experiment.transformer.num_layers == 2
        assert experiment.embedding_learning_rate == 0.064
        assert experiment.deep_learning_rate == deep_lr
        assert experiment.dataloader.batch_size == 1280
        assert experiment.num_epochs == 40
        assert experiment.lr_schedule.shape == "linear"
        assert experiment.lr_schedule_horizon_epochs == 20
        assert experiment.mup_base_dim == 16
        assert experiment.mup_delta_dim == 32


def test_launcher_rejects_invalid_or_duplicate_arms_before_submission(
    tmp_path: Path,
) -> None:
    invalid = _run(tmp_path / "invalid", G1_FFN_STUDY_ARMS="relu:3")
    duplicate = _run(tmp_path / "duplicate", G1_FFN_STUDY_ARMS="relu:2 relu:2")

    assert invalid.returncode == 2
    assert "Invalid G1_FFN_STUDY_ARMS value: relu:3" in invalid.stderr
    assert "ENQUEUE " not in invalid.stderr
    assert duplicate.returncode == 2
    assert "Duplicate G1_FFN_STUDY_ARMS value: relu:2" in duplicate.stderr
    assert "ENQUEUE " not in duplicate.stderr


def test_launcher_rejects_zero_deep_rate_before_submission(tmp_path: Path) -> None:
    result = _run(tmp_path, G1_FFN_STUDY_DEEP_LRS="0.000")

    assert result.returncode == 2
    assert "must be positive canonical decimals: 0.000" in result.stderr
    assert "ENQUEUE " not in result.stderr


def test_launcher_rejects_embedding_rate_override_before_submission(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, G1_FFN_STUDY_EMBEDDING_LR="0.032")

    assert result.returncode == 2
    assert "must remain fixed at 0.064" in result.stderr
    assert "ENQUEUE " not in result.stderr


def test_legacy_artifact_normalization_preserves_gated_dropout_semantics() -> None:
    missing = {"transfer_invariants": {"transformer": {"ffn": "swiglu"}}}
    explicit = {
        "transfer_invariants": {
            "transformer": {"ffn": "swiglu", "gated_ffn_dropout": True}
        }
    }

    normalized_missing = verify_artifact._with_legacy_accumulation_defaults(missing)
    normalized_explicit = verify_artifact._with_legacy_accumulation_defaults(explicit)

    assert not normalized_missing["transfer_invariants"]["transformer"][
        "gated_ffn_dropout"
    ]
    assert normalized_explicit["transfer_invariants"]["transformer"][
        "gated_ffn_dropout"
    ]


def test_tuning_verifier_matches_mup_ffn_shape_invariants(tmp_path: Path) -> None:
    assignments = [
        "G1_TUNE_RUN=artifact_mup_cap40_ts2_r3",
        "G1_TUNE_RUN_REVISION=3",
        "G1_TUNE_EPOCHS=40",
        "G1_TUNE_SOURCE_VARIANT=baseline",
        "G1_TUNE_TRANSFORMER_FIELDS=ffn",
        "G1_TUNE_EXPERIMENT_FIELDS=",
        "G1_TUNE_EMBEDDING_LR=0.064",
        "G1_TUNE_DEEP_LR=0.012",
        "G1_TUNE_BATCH_SIZE=1280",
        "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1",
        "G1_TUNE_FFN_DIM=171",
    ]
    parsed = verify_artifact._tuning_assignments(assignments)
    experiment = verify_artifact._tuning_experiment("500m", parsed)
    top_level, invariants = verify_artifact._expected_metadata(experiment)
    metadata = top_level | {
        "max_epochs": 40,
        "epochs_trained": 20,
        "stopped_epoch": 20,
        "best_epoch": 14,
        "early_stopped": False,
        "best_epoch_at_cap": False,
        "selection_resolved": True,
        "targets_per_epoch": 11,
        "tokens_per_epoch": 13,
        "training_horizon": 220,
        "token_horizon": 260,
        "tokens_seen": 260,
        "optimizer_steps": 5,
        "validation_loss": 0.5,
        "training_semantics_revision": 2,
        "transfer_invariants": invariants,
    }
    directory = tmp_path / experiment.run_name
    directory.mkdir()
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text(json.dumps({"recall@100": 0.13}))

    assert verify_artifact.verify(directory, "500m", assignments)

    metadata["transfer_invariants"].pop("mup_base_ffn_dim")
    metadata["transfer_invariants"].pop("mup_delta_ffn_dim")
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text(json.dumps({"recall@100": 0.13}))
    assert verify_artifact.verify(directory, "500m", assignments)

    metadata["transfer_invariants"]["mup_base_ffn_dim"] = 99
    (directory / "training_metadata.json").write_text(json.dumps(metadata))
    (directory / "final_metrics.json").write_text(json.dumps({"recall@100": 0.13}))
    assert not verify_artifact.verify(directory, "500m", assignments)
