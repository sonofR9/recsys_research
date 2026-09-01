from pathlib import Path
import json
import subprocess
import sys

import pytest
import experiments.g4_future_items.launchers.run_selectors as selector_launcher

from experiments.g4_future_items.launchers.run_selectors import (
    _hold_native_materialization_lock,
    _is_native_materialization_command,
    PreparedSelectorData,
)
from experiments.g4_future_items.protocol.manifest import canonical_sha256


def test_native_materialization_command_matches_module_and_script_forms() -> None:
    assert _is_native_materialization_command(
        [
            "python",
            "-m",
            "experiments.g4_future_items.launchers.run_selectors",
            "native-materialization",
        ]
    )


def test_module_cli_materialization_child_is_spawn_serializable() -> None:
    probe = """
import multiprocessing.reduction
import runpy
import sys

sys.argv = [
    "run_selectors.py",
    "compile",
]
module = runpy.run_module(
    "experiments.g4_future_items.launchers.run_selectors",
    run_name="__main__",
    alter_sys=True,
)
multiprocessing.reduction.ForkingPickler.dumps(
    module["_spawn_callable"](module["_native_materialization_child"])
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert _is_native_materialization_command(
        ["python", "/repo/run_selectors.py", "native-materialization"]
    )
    assert not _is_native_materialization_command(
        [
            "python",
            "-m",
            "experiments.g4_future_items.launchers.run_selectors",
            "prepare",
        ]
    )


def test_native_materialization_lock_fails_closed(tmp_path: Path) -> None:
    lock_path = tmp_path / "materialization.lock"

    with _hold_native_materialization_lock(lock_path):
        with pytest.raises(RuntimeError, match="holds the lock"):
            with _hold_native_materialization_lock(lock_path):
                pass


def test_public_native_materialization_path_cannot_bypass_lock(
    tmp_path: Path, monkeypatch
) -> None:
    lock_path = tmp_path / "materialization.lock"
    monkeypatch.setattr(
        selector_launcher, "NATIVE_MATERIALIZATION_LOCK_PATH", lock_path
    )

    with _hold_native_materialization_lock(lock_path):
        with pytest.raises(RuntimeError, match="holds the lock"):
            selector_launcher.run_native_materialization(
                None,
                None,
                {},
                (),
                (),
                measurement_directory=tmp_path / "measurement",
                enforce_reference_fixture=False,
            )


def test_prepared_input_paths_resolve_from_frozen_control(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "generated/yambda_data/flat/50m"
    cache = tmp_path / "cache"
    dataset.mkdir(parents=True)
    cache.mkdir()
    paths = {
        "control_likes": cache / "events_remapped.parquet",
        "raw_events": dataset / "multi_event.parquet",
        "item_id_remap": cache / "item_id_remap.parquet",
        "compact_embeddings": cache / "embeddings_compact.parquet",
    }
    for index, path in enumerate(paths.values(), 1):
        path.write_bytes(bytes([index]) * 17)
    control = {
        "data_identity": {
            "main": {"path": str(paths["control_likes"])},
            "remap": {"path": str(paths["item_id_remap"])},
            "content_embeddings": {"path": str(paths["compact_embeddings"])},
        },
        "resolved_anchor_configuration": {"fixed": {"data": {"size": "50m"}}},
    }
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps(control))
    identities = {
        name: selector_launcher._file_identity(path) for name, path in paths.items()
    }
    prepared = PreparedSelectorData(
        tmp_path,
        "a" * 64,
        {
            "inputs": identities,
            "provenance": {
                "control_semantics_manifest_sha256": canonical_sha256(control)
            },
        },
    )
    monkeypatch.setattr(selector_launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        selector_launcher, "CONTROL_SEMANTICS_MANIFEST_PATH", control_path
    )

    resolved = selector_launcher._selector_input_paths(prepared)

    assert resolved.paths == tuple(paths.values())
