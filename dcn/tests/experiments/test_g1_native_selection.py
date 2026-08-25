from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest


EXPERIMENT = (
    Path(__file__).resolve().parents[3]
    / "experiments/g1_sasrec_item_ids_likes"
)
SELECTOR = EXPERIMENT / "analysis/select_native_500m.py"
LAUNCHER = EXPERIMENT / "launchers/transfer/selected_native_500m.sh"


def _synthetic_grid(tmp_path: Path):
    namespace = runpy.run_path(str(SELECTOR))
    for epochs, points in namespace["APPROVED_STAGES"].items():
        for embedding_lr, deep_lr in points:
            run = namespace["_run"](epochs, embedding_lr, deep_lr)
            directory = tmp_path / "logs" / f"g1_transfer_{run}_50m"
            directory.mkdir(parents=True)
            winner = (
                epochs == 60
                and (embedding_lr, deep_lr) == namespace["EXPECTED_WINNER"]
            )
            (directory / "training_metadata.json").write_text(
                json.dumps(
                    {
                        "selection_resolved": True,
                        "max_epochs": epochs,
                        "embedding_learning_rate": float(embedding_lr),
                        "deep_learning_rate": float(deep_lr),
                    }
                )
            )
            (directory / "final_metrics.json").write_text(
                json.dumps({"recall@100": 1.0 if winner else 0.5})
            )
    return namespace


def test_selector_uses_only_exact_approved_grid_and_ignores_incidental_run(
    tmp_path: Path,
) -> None:
    namespace = _synthetic_grid(tmp_path)
    verified = []

    def verify(directory, _config, assignments):
        verified.append((directory.name, tuple(assignments)))
        return directory.is_dir()

    selected = namespace["select_native_500m"](tmp_path, verify)
    incidental = (
        tmp_path
        / "logs/g1_transfer_batchscale_b1280_e0p007_d0p001_cap40_ts2_r2_50m"
    )
    incidental.mkdir()
    (incidental / "training_metadata.json").write_text(
        json.dumps({"selection_resolved": True})
    )
    (incidental / "final_metrics.json").write_text(
        json.dumps({"recall@100": 2.0})
    )
    selected_with_incidental = namespace["select_native_500m"](tmp_path, verify)

    assert selected["embedding_lr"] == "0.001"
    assert selected["deep_lr"] == "0.002"
    assert selected["source_artifacts"] == 42
    assert selected["source_digest"] == selected_with_incidental["source_digest"]
    assert all("e0p007_d0p001" not in name for name, _ in verified)


def test_selector_rejects_missing_or_unresolved_source(tmp_path: Path) -> None:
    namespace = _synthetic_grid(tmp_path)
    missing = namespace["_run"](20, "0.008", "0.002")
    missing_directory = tmp_path / "logs" / f"g1_transfer_{missing}_50m"
    missing_directory.rename(missing_directory.with_name("missing"))

    with pytest.raises(ValueError, match="missing or incompatible approved artifact"):
        namespace["select_native_500m"](
            tmp_path, lambda directory, _config, _assignments: directory.is_dir()
        )

    namespace = _synthetic_grid(tmp_path / "unresolved")
    winner = namespace["_run"](60, "0.001", "0.002")
    metadata_path = (
        tmp_path
        / "unresolved/logs"
        / f"g1_transfer_{winner}_50m"
        / "training_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    metadata["selection_resolved"] = False
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="winner is not selection_resolved"):
        namespace["select_native_500m"](
            tmp_path / "unresolved", lambda *_: True
        )


def test_selector_retains_cap_limited_sources_with_resolved_continuations(
    tmp_path: Path,
) -> None:
    namespace = _synthetic_grid(tmp_path)
    unresolved = {
        (20, "0.008", "0.002"),
        (20, "0.008", "0.004"),
        (20, "0.016", "0.032"),
        (40, "0.002", "0.002"),
        (40, "0.004", "0.001"),
        (60, "0.0005", "0.002"),
    }
    for epochs, embedding_lr, deep_lr in unresolved:
        run = namespace["_run"](epochs, embedding_lr, deep_lr)
        metadata_path = (
            tmp_path
            / "logs"
            / f"g1_transfer_{run}_50m"
            / "training_metadata.json"
        )
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            selection_resolved=False,
            num_epochs=epochs,
            max_epochs=epochs,
            epochs_trained=epochs,
            stopped_epoch=epochs,
        )
        metadata_path.write_text(json.dumps(metadata))

    def selectable(directory, _config, _assignments):
        metadata = json.loads((directory / "training_metadata.json").read_text())
        return metadata["selection_resolved"]

    selected = namespace["select_native_500m"](
        tmp_path, selectable, lambda directory, *_: directory.is_dir()
    )
    assert selected["source_artifacts"] == 42

    retained = namespace["_run"](20, "0.016", "0.032")
    retained_metrics = (
        tmp_path
        / "logs"
        / f"g1_transfer_{retained}_50m"
        / "final_metrics.json"
    )
    retained_metrics.write_text(json.dumps({"recall@100": 0.99}))
    changed = namespace["select_native_500m"](
        tmp_path, selectable, lambda directory, *_: directory.is_dir()
    )
    assert changed["source_digest"] != selected["source_digest"]
    assert changed["winner_run"] == selected["winner_run"]


def test_selector_rejects_unresolved_terminal_continuation(tmp_path: Path) -> None:
    namespace = _synthetic_grid(tmp_path)
    terminal = namespace["_run"](40, "0.016", "0.032")
    metadata_path = (
        tmp_path
        / "logs"
        / f"g1_transfer_{terminal}_50m"
        / "training_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        selection_resolved=False,
        num_epochs=40,
        max_epochs=40,
        epochs_trained=40,
        stopped_epoch=40,
    )
    metadata_path.write_text(json.dumps(metadata))

    def selectable(directory, _config, _assignments):
        artifact_metadata = json.loads(
            (directory / "training_metadata.json").read_text()
        )
        return artifact_metadata["selection_resolved"]

    with pytest.raises(ValueError, match="terminal artifact is not selection_resolved"):
        namespace["select_native_500m"](
            tmp_path, selectable, lambda directory, *_: directory.is_dir()
        )


def test_selector_rejects_boundary_winner(tmp_path: Path) -> None:
    namespace = _synthetic_grid(tmp_path)
    namespace["select_native_500m"].__globals__["EXPECTED_WINNER"] = (
        "0.0005",
        "0.002",
    )
    old_winner = namespace["_run"](60, "0.001", "0.002")
    old_metrics = tmp_path / "logs" / f"g1_transfer_{old_winner}_50m/final_metrics.json"
    old_metrics.write_text(json.dumps({"recall@100": 0.5}))
    boundary = namespace["_run"](60, "0.0005", "0.002")
    boundary_metrics = (
        tmp_path / "logs" / f"g1_transfer_{boundary}_50m/final_metrics.json"
    )
    boundary_metrics.write_text(json.dumps({"recall@100": 1.0}))

    with pytest.raises(ValueError, match="not interior"):
        namespace["select_native_500m"](tmp_path, lambda *_: True)


def test_selected_launcher_enqueues_one_digest_bound_confirmation(
    tmp_path: Path,
) -> None:
    digest = "abcdef012345" + "6" * 52
    selector = tmp_path / "selector.py"
    selector.write_text(
        f"print('{digest}\\tabcdef012345\\t0.001\\t0.002\\t42\\t"
        "g1_transfer_batchscale_b1280_e0p001_d0p002_cap60_ts2_r2_50m')\n"
    )
    provenance = tmp_path / "provenance.py"
    provenance.write_text("print('PROVENANCE')\n")
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\"; }\n"
        "drain() { echo DRAIN; return 0; }\n"
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_NATIVE_SELECTOR": str(selector),
            "G1_NATIVE_PROVENANCE_TOOL": str(provenance),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
        },
    )

    assert result.returncode == 0, result.stderr
    enqueues = [line for line in result.stdout.splitlines() if line.startswith("ENQUEUE")]
    assert len(enqueues) == 1
    assert "selected_native50_abcdef012345_e0p001_d0p002_cap40_ts2_r2_500m" in enqueues[0]
    assert "G1_TRANSFER_EPOCHS=40" in enqueues[0]
    assert "G1_TRANSFER_EMBEDDING_LR=0.001" in enqueues[0]
    assert "G1_TRANSFER_DEEP_LR=0.002" in enqueues[0]
    assert (
        result.stdout.index("ENQUEUE")
        < result.stdout.index("DRAIN")
        < result.stdout.index("PROVENANCE")
    )


def test_selected_launcher_extended_cap_has_collision_safe_provenance(
    tmp_path: Path,
) -> None:
    digest = "abcdef012345" + "6" * 52
    selector = tmp_path / "selector.py"
    selector.write_text(
        f"print('{digest}\\tabcdef012345\\t0.001\\t0.002\\t42\\t"
        "g1_transfer_batchscale_b1280_e0p001_d0p002_cap60_ts2_r2_50m')\n"
    )
    provenance = tmp_path / "provenance.py"
    provenance.write_text("raise SystemExit(0)\n")
    queue = tmp_path / "queue.sh"
    queue.write_text(
        "enqueue() { printf 'ENQUEUE %s\\n' \"$*\"; }\n"
        "drain() { return 0; }\n"
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "G1_NATIVE_SELECTOR": str(selector),
            "G1_NATIVE_PROVENANCE_TOOL": str(provenance),
            "G1_TRAINING_QUEUE_LIBRARY": str(queue),
            "G1_FINAL_EPOCHS": "40",
            "G1_FINAL_RUN_REVISION": "3",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "_cap40_ts2_r3_500m" in result.stdout
    assert "G1_TRANSFER_EPOCHS=40" in result.stdout
    assert "G1_TRANSFER_RUN_REVISION=3" in result.stdout


def test_selected_launcher_requires_cap_and_revision_together() -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        text=True,
        env=os.environ | {"G1_FINAL_EPOCHS": "40"},
    )

    assert result.returncode == 2
    assert "must be set together" in result.stderr
