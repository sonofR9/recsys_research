from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment


SCRIPT = (
    Path(__file__).parents[3]
    / "experiments"
    / "g1_sasrec_item_ids_likes"
    / "configs/transfer_variant.py"
)
VERIFY_ARTIFACT = SCRIPT.parents[1] / "launchers" / "verify_artifact.py"
BATCH_SCALING = SCRIPT.parents[1] / "launchers" / "transfer" / "batch_scaling_50m.sh"


def _load(monkeypatch: pytest.MonkeyPatch, **environment: str):
    monkeypatch.setenv("G1_VARIANT", "heads_8")
    monkeypatch.setenv("G1_DATASET_SIZE", "50m")
    monkeypatch.setenv("G1_TRANSFER_RUN", "control_h20_e4e3_d12e3_ts2_r2")
    monkeypatch.setenv("G1_TRANSFER_EPOCHS", "20")
    monkeypatch.setenv("G1_TRANSFER_EMBEDDING_LR", "0.004")
    monkeypatch.setenv("G1_TRANSFER_DEEP_LR", "0.012")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return runpy.run_path(str(SCRIPT))["experiment"]


def test_standard_transfer_run_uses_selected_control_and_early_stopping_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(monkeypatch)

    assert type(experiment) is GenerationExperiment
    assert experiment.run_name == "g1_transfer_control_h20_e4e3_d12e3_ts2_r2_50m"
    assert experiment.num_epochs == 20
    assert experiment.eval_every_n_epochs == 1
    assert experiment.restore_best_weights
    assert experiment.early_stopping_patience == 3
    assert experiment.early_stopping_min_delta == 0.0
    assert experiment.dataloader.batch_size == 1280
    assert experiment.embedding_learning_rate == 0.004
    assert experiment.deep_learning_rate == 0.012
    assert experiment.transformer.nhead == 2
    assert experiment.transformer.num_kv_heads == 1


def test_extended_transfer_cap_is_encoded_in_run_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TRANSFER_EPOCHS="40",
        G1_TRANSFER_RUN="control_h20_e4e3_d12e3_cap40_ts2_r2",
    )

    assert experiment.num_epochs == 40
    assert experiment.run_name.endswith("_cap40_ts2_r2_50m")
    assert experiment.lr_schedule_horizon_epochs == 20

    with pytest.raises(ValueError, match="cap40_ts2_r2"):
        _load(
            monkeypatch,
            G1_TRANSFER_EPOCHS="40",
            G1_TRANSFER_RUN="control_h20_e4e3_d12e3_ts2_r2",
        )


def test_transfer_run_revision_is_encoded_in_run_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TRANSFER_RUN_REVISION="3",
        G1_TRANSFER_RUN="control_h20_e4e3_d12e3_ts2_r3",
    )

    assert experiment.run_name.endswith("_ts2_r3_50m")


def test_selected_confirmation_can_encode_initial_cap_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TRANSFER_RUN=(
            "selected_native50_abcdef012345_e0p001_d0p002_cap20_ts2_r2"
        ),
    )

    assert experiment.num_epochs == 20
    assert experiment.run_name.endswith("_cap20_ts2_r2_50m")


def test_batch_scaling_launcher_encodes_extended_cap() -> None:
    source = BATCH_SCALING.read_text()

    assert "transfer_epochs=${G1_TRANSFER_EPOCHS:-20}" in source
    assert 'cap="_cap${transfer_epochs}"' in source
    assert '${cap}_ts2_r2' in source
    assert '"$transfer_epochs"' in source


def test_mup_transfer_keeps_item_table_fixed_while_changing_model_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TRANSFER_PARAMETERIZATION="mup",
        G1_TRANSFER_DIM="32",
        G1_TRANSFER_RUN="mup_dim32_ts2_r2",
    )

    assert type(experiment) is MuTransferGenerationExperiment
    assert experiment.item_embedding_dim == 64
    assert experiment.transformer.dim == 32
    assert experiment.transformer.ffn_intermediate_dim == 86
    assert experiment.mup_base_dim == 16
    assert experiment.mup_delta_dim == 32


def test_mup_transfer_takes_an_explicit_ffn_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TRANSFER_PARAMETERIZATION="mup",
        G1_TRANSFER_FFN_DIM="128",
        G1_TRANSFER_RUN="mup_ffn128_ts2_r2",
    )

    assert experiment.transformer.dim == 64
    assert experiment.transformer.ffn_intermediate_dim == 128
    assert experiment.mup_base_ffn_dim is None


def test_mup_transfer_gives_the_ffn_width_a_base_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TRANSFER_PARAMETERIZATION="mup",
        G1_TRANSFER_FFN_DIM="128",
        G1_TRANSFER_MUP_FFN_BASE="32",
        G1_TRANSFER_RUN="mupffn_ffn128_ts2_r2",
    )

    assert experiment.mup_base_ffn_dim == 32
    assert experiment.mup_delta_ffn_dim == 64


def test_an_ffn_base_needs_the_mup_parameterization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="G1_TRANSFER_PARAMETERIZATION"):
        _load(
            monkeypatch,
            G1_TRANSFER_FFN_DIM="128",
            G1_TRANSFER_MUP_FFN_BASE="32",
            G1_TRANSFER_RUN="mupffn_ffn128_ts2_r2",
        )


def test_transfer_can_use_homework_control_for_batch_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TRANSFER_SOURCE_VARIANT="homework_fixed_leave_one_out",
        G1_TRANSFER_BATCH_SIZE="128",
        G1_TRANSFER_RUN="batchscale_b128_e0p001_d0p001_ts2_r2",
        G1_TRANSFER_EPOCHS="20",
        G1_TRANSFER_EMBEDDING_LR="0.001",
        G1_TRANSFER_DEEP_LR="0.001",
    )

    assert type(experiment) is GenerationExperiment
    assert experiment.dataloader.batch_size == 128
    assert experiment.negative_sampling == "offline_logq"
    assert experiment.logq_correction == "baseline"
    assert experiment.lr_schedule.shape == "constant"


def test_power_schedule_uses_explicit_token_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _load(
        monkeypatch,
        G1_TRANSFER_POWER_TOKENS="1500000",
        G1_TRANSFER_RUN="power_t1500k_ts2_r2",
    )

    assert experiment.lr_schedule.shape == "power"
    assert experiment.lr_schedule.power_exponent == -0.51
    assert experiment.lr_schedule.power_transition_tokens == 1_500_000


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("G1_TRANSFER_EPOCHS", "0", "positive integer"),
        ("G1_TRANSFER_EPOCHS", "19", "20-epoch safety cap"),
        ("G1_TRANSFER_EMBEDDING_LR", "nan", "positive finite"),
        ("G1_TRANSFER_PARAMETERIZATION", "standard", "conventional or mup"),
        ("G1_TRANSFER_SOURCE_VARIANT", "baseline", "source variant"),
        ("G1_TRANSFER_POWER_TOKENS", "0", "positive integer"),
    ],
)
def test_transfer_environment_is_validated(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _load(monkeypatch, **{name: value})


def test_artifact_horizon_uses_actual_early_stopped_epochs() -> None:
    valid = runpy.run_path(str(VERIFY_ARTIFACT))["_valid_dynamic_metadata"]
    metadata = {
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 7,
        "stopped_epoch": 7,
        "best_epoch": 4,
        "early_stopped": True,
        "best_epoch_at_cap": False,
        "selection_resolved": True,
        "targets_per_epoch": 11,
        "tokens_per_epoch": 13,
        "training_horizon": 77,
        "token_horizon": 91,
        "tokens_seen": 91,
        "optimizer_steps": 5,
        "validation_loss": 0.5,
        "transfer_invariants": {"lr_schedule": {"shape": "linear"}},
    }

    assert valid(metadata)

    metadata["training_horizon"] = 11 * 20
    metadata["token_horizon"] = 13 * 20
    metadata["tokens_seen"] = 13 * 20
    assert not valid(metadata)


def test_cap_limited_artifact_is_recipe_valid_but_not_selectable() -> None:
    namespace = runpy.run_path(str(VERIFY_ARTIFACT))
    metadata = {
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 20,
        "stopped_epoch": 20,
        "best_epoch": 18,
        "early_stopped": False,
        "best_epoch_at_cap": False,
        "selection_resolved": False,
        "targets_per_epoch": 11,
        "tokens_per_epoch": 13,
        "training_horizon": 220,
        "token_horizon": 260,
        "tokens_seen": 260,
        "optimizer_steps": 5,
        "validation_loss": 0.5,
        "transfer_invariants": {"lr_schedule": {"shape": "constant"}},
    }

    assert namespace["_valid_recipe_dynamic_metadata"](metadata)
    assert not namespace["_valid_dynamic_metadata"](metadata)


def test_a_spent_annealing_horizon_needs_no_early_stop() -> None:
    valid = runpy.run_path(str(VERIFY_ARTIFACT))["_valid_dynamic_metadata"]
    metadata = {
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 20,
        "stopped_epoch": 20,
        "best_epoch": 18,
        "early_stopped": False,
        "best_epoch_at_cap": False,
        "selection_resolved": True,
        "lr_schedule_horizon_epochs": 20,
        "targets_per_epoch": 11,
        "tokens_per_epoch": 13,
        "training_horizon": 220,
        "token_horizon": 260,
        "tokens_seen": 260,
        "optimizer_steps": 5,
        "validation_loss": 0.5,
        "transfer_invariants": {"lr_schedule": {"shape": "linear"}},
    }

    assert valid(metadata)


def test_a_best_epoch_past_the_annealing_horizon_is_rejected() -> None:
    valid = runpy.run_path(str(VERIFY_ARTIFACT))["_valid_dynamic_metadata"]
    metadata = {
        "num_epochs": 40,
        "max_epochs": 40,
        "epochs_trained": 26,
        "stopped_epoch": 26,
        "best_epoch": 23,
        "early_stopped": True,
        "best_epoch_at_cap": False,
        "selection_resolved": True,
        "lr_schedule_horizon_epochs": 20,
        "targets_per_epoch": 11,
        "tokens_per_epoch": 13,
        "training_horizon": 286,
        "token_horizon": 338,
        "tokens_seen": 338,
        "optimizer_steps": 5,
        "validation_loss": 0.5,
        "transfer_invariants": {"lr_schedule": {"shape": "exponential"}},
    }

    assert not valid(metadata)

    metadata["best_epoch"] = 19
    assert valid(metadata)


@pytest.mark.parametrize(
    "field,value",
    [
        ("early_stopped", False),
        ("selection_resolved", False),
        ("best_epoch", None),
        ("best_epoch", 0),
        ("best_epoch", 8),
        ("best_epoch_at_cap", True),
        ("stopped_epoch", 20),
    ],
)
def test_artifact_horizon_rejects_unresolved_selection(
    field: str, value: object
) -> None:
    valid = runpy.run_path(str(VERIFY_ARTIFACT))["_valid_dynamic_metadata"]
    metadata = {
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 7,
        "stopped_epoch": 7,
        "best_epoch": 4,
        "early_stopped": True,
        "best_epoch_at_cap": False,
        "selection_resolved": True,
        "targets_per_epoch": 11,
        "tokens_per_epoch": 13,
        "training_horizon": 77,
        "token_horizon": 91,
        "tokens_seen": 91,
        "optimizer_steps": 5,
        "validation_loss": 0.5,
        "transfer_invariants": {"lr_schedule": {"shape": "constant"}},
    }
    metadata[field] = value
    if field == "stopped_epoch":
        metadata["epochs_trained"] = value
        metadata["training_horizon"] = 11 * int(value)
        metadata["token_horizon"] = 13 * int(value)
        metadata["tokens_seen"] = 13 * int(value)

    assert not valid(metadata)


def test_artifact_recipe_includes_early_stopping_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(VERIFY_ARTIFACT))
    _, invariants = namespace["_expected_metadata"](_load(monkeypatch))

    assert invariants["early_stopping_patience"] == 3
    assert invariants["early_stopping_min_delta"] == 0.0
    assert invariants["early_stopping_metric"] == "recall@100"
    assert invariants["early_stopping_metric_prefix"] == "epoch/val_true"


def test_in_process_verification_is_order_independent_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(VERIFY_ARTIFACT))
    monkeypatch.setenv("G1_AMBIENT_SENTINEL", "keep")
    monkeypatch.setenv("G1_TRANSFER_RUN", "ambient")
    ambient = {
        name: value for name, value in os.environ.items() if name.startswith("G1_")
    }

    def assignments(run: str, embedding_lr: str) -> dict[str, str]:
        return {
            "G1_DATASET_SIZE": "50m",
            "G1_TRANSFER_RUN": run,
            "G1_TRANSFER_RUN_REVISION": "2",
            "G1_TRANSFER_EPOCHS": "20",
            "G1_TRANSFER_EMBEDDING_LR": embedding_lr,
            "G1_TRANSFER_DEEP_LR": "0.002",
            "G1_TRANSFER_PARAMETERIZATION": "conventional",
            "G1_TRANSFER_BATCH_SIZE": "1280",
            "G1_TRANSFER_DIM": "64",
            "G1_TRANSFER_SOURCE_VARIANT": "homework_fixed_leave_one_out",
        }

    first = namespace["_config_experiment"](
        SCRIPT,
        assignments("batchscale_b1280_e0p001_d0p002_ts2_r2", "0.001"),
    )
    second = namespace["_config_experiment"](
        SCRIPT,
        assignments("batchscale_b1280_e0p004_d0p002_ts2_r2", "0.004"),
    )

    assert first.embedding_learning_rate == 0.001
    assert second.embedding_learning_rate == 0.004
    assert {
        name: value for name, value in os.environ.items() if name.startswith("G1_")
    } == ambient

    with pytest.raises(ValueError, match="must end"):
        namespace["_config_experiment"](
            SCRIPT,
            assignments("batchscale_b1280_e0p008_d0p002_ts2_r3", "0.008"),
        )
    assert {
        name: value for name, value in os.environ.items() if name.startswith("G1_")
    } == ambient


def test_in_process_verification_reloads_config_modules_between_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(VERIFY_ARTIFACT))
    homework = SCRIPT.with_name("homework_500m.py")
    monkeypatch.setenv("G1_AMBIENT_SENTINEL", "keep")

    cap20 = namespace["_config_experiment"](
        homework, {"G1_MAX_EPOCHS": "20"}
    )
    cap30 = namespace["_config_experiment"](
        homework, {"G1_MAX_EPOCHS": "30"}
    )

    assert cap20.num_epochs == 20
    assert cap30.num_epochs == 30
    assert cap20.run_name != cap30.run_name
    assert os.environ["G1_AMBIENT_SENTINEL"] == "keep"


def test_config_module_discovery_does_not_resolve_every_loaded_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(VERIFY_ARTIFACT))
    module = ModuleType("g1_config_discovery_probe")
    module.__file__ = str(tmp_path / "configs/probe.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace["_modules_from"](tmp_path / "configs")
    calls = 0
    original = Path.resolve

    def counted_resolve(path: Path, *args, **kwargs) -> Path:
        nonlocal calls
        calls += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted_resolve)

    modules = namespace["_modules_from"](tmp_path / "configs")

    assert modules[module.__name__] is module
    assert calls == 0


def test_in_process_verification_canonicalizes_equivalent_config_paths() -> None:
    namespace = runpy.run_path(str(VERIFY_ARTIFACT))
    homework = SCRIPT.with_name("homework_500m.py")
    equivalent = homework.parent / ".." / homework.parent.name / homework.name

    cap20 = namespace["_config_experiment"](
        equivalent, {"G1_MAX_EPOCHS": "20"}
    )
    cap30 = namespace["_config_experiment"](
        equivalent, {"G1_MAX_EPOCHS": "30"}
    )

    assert cap20.num_epochs == 20
    assert cap30.num_epochs == 30
    assert cap20.run_name != cap30.run_name


def test_unaccumulated_batch_contract_rejects_other_batch_contracts() -> None:
    contract = runpy.run_path(str(VERIFY_ARTIFACT))[
        "has_unaccumulated_batch_contract"
    ]
    valid = {
        "batch_size": 1280,
        "physical_batch_size": 1280,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 1280,
        "transfer_invariants": {
            "batch_size": 1280,
            "physical_batch_size": 1280,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 1280,
        },
    }
    accumulated = {
        **valid,
        "batch_size": 640,
        "physical_batch_size": 640,
        "gradient_accumulation_steps": 2,
        "transfer_invariants": {
            **valid["transfer_invariants"],
            "batch_size": 640,
            "physical_batch_size": 640,
            "gradient_accumulation_steps": 2,
        },
    }
    different_effective = {
        **valid,
        "effective_batch_size": 640,
        "transfer_invariants": {
            **valid["transfer_invariants"],
            "effective_batch_size": 640,
        },
    }

    assert contract(valid, 1280)
    assert not contract(accumulated, 1280)
    assert not contract(different_effective, 1280)
