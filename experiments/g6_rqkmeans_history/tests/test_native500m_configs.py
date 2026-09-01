from dataclasses import fields, replace
import hashlib
import json
import stat
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dcn.config import (
    GenerationExperiment,
    MuTransferGenerationExperiment,
    SemanticHistoryExperiment,
)
from dcn.semantic import ResidualCodebooks, SemanticCodes
from experiments.g6_rqkmeans_history.native500m.configs.runtime import (
    APPLICABLE_INITIALIZATION_REPRESENTATIONS,
    REPRESENTATIONS,
    ConventionalSemanticHistoryExperiment,
    Native500MSemanticHistoryExperiment,
    build_collision_pair,
    build_control,
    build_semantic_treatment,
    learned_sid_lookup,
)
from utils.global_config import config as global_config


TOKENIZER_LEVELS = (3, 4)
SHARED_CODEBOOK_SIZES = (512, 2048, 8192)


def _semantic_artifacts(
    *, num_levels: int = 3, num_codes: int = 512
) -> tuple[SemanticCodes, ResidualCodebooks]:
    item_ids = torch.tensor([1, 2, 3, 4])
    raw_codes = torch.tensor(
        [[0] * num_levels, [0] * num_levels, [1] * num_levels, [2] * num_levels]
    )
    generator = torch.Generator().manual_seed(11)
    centroids = torch.randn(
        num_levels, num_codes, 128, generator=generator, dtype=torch.float32
    )
    return (
        SemanticCodes.with_collision_suffix(item_ids, raw_codes, num_codes),
        ResidualCodebooks(centroids),
    )


@pytest.mark.parametrize("backbone", ["original_g1", "best_g1"])
def test_controls_use_the_native_500m_protocol_and_fixed_horizon(backbone: str) -> None:
    experiment = build_control(
        backbone=backbone,
        embedding_learning_rate=0.03,
        deep_learning_rate=0.01,
        run_name=f"g6_{backbone}_control_native500m",
    )

    assert experiment.size == "500m"
    assert experiment.user_sample is None
    assert experiment.event_type_filter == "like"
    assert experiment.validation_interval_seconds == 7 * 24 * 60 * 60
    assert experiment.drop_unmapped_items is True
    assert experiment.evaluation_catalog == "all"
    assert experiment.exclude_seen_from_evaluation is False
    assert experiment.dataloader.batch_size == 512
    assert experiment.dataloader.gradient_accumulation_steps == 1
    assert experiment.dataloader.num_workers == 4
    assert experiment.dataloader.prefetch_factor == 4
    assert experiment.num_epochs == 26
    assert experiment.eval_max_users is None
    assert experiment.lr_schedule_horizon_epochs == 26
    assert experiment.early_stopping_patience is None
    assert experiment.restore_best_weights is True


def test_original_control_is_the_exact_conventional_g1_recipe() -> None:
    experiment = build_control(
        backbone="original_g1",
        embedding_learning_rate=0.001,
        deep_learning_rate=0.002,
        run_name="g6_original_control_native500m",
    )

    assert isinstance(experiment, GenerationExperiment)
    assert not isinstance(experiment, MuTransferGenerationExperiment)
    assert experiment.transformer == replace(
        GenerationExperiment.transformer,
        dim=64,
        num_layers=2,
        nhead=2,
        num_kv_heads=2,
        ffn_intermediate_dim=256,
        ffn="gelu",
        norm="layer",
        norm_place="pre",
        input_norm=None,
        final_norm="layer",
        alibi=False,
        rope=None,
        learned_positions="forward",
        learned_position_fusion="add",
        learned_position_fusion_normalization=None,
        learned_position_fusion_residual=None,
        learned_position_initialization="default",
        learned_position_reverse_correction=None,
        learned_position_reverse_max_scale=0.1,
        learned_position_reverse_initializer_rng_nonadvancing=False,
        attention_window=None,
        input_dropout=0.1,
        ffn_dropout=0.1,
        gated_ffn_dropout=False,
    )
    assert experiment.max_seq_len == 100
    assert experiment.bos is False
    assert experiment.cls_token_mode == "none"
    assert experiment.negative_sampling == "offline_logq"
    assert experiment.logq_correction == "baseline"
    assert experiment.correct_positive_logq is False
    assert experiment.num_in_batch_negatives == 512
    assert experiment.dense_random_negative_scores is False
    assert experiment.lr_schedule.shape == "constant"


def test_best_control_is_the_exact_four_layer_g1_aggregate_recipe() -> None:
    experiment = build_control(
        backbone="best_g1",
        embedding_learning_rate=0.0468526465053628,
        deep_learning_rate=0.032703745675187676,
        run_name="g6_best_control_native500m",
    )

    assert isinstance(experiment, MuTransferGenerationExperiment)
    assert experiment.transformer.num_layers == 4
    assert experiment.transformer.dim == 64
    assert experiment.transformer.num_kv_heads == 1
    assert experiment.transformer.ffn == "swiglu"
    assert experiment.transformer.ffn_intermediate_dim == 192
    assert experiment.transformer.norm_place == "post"
    assert experiment.transformer.input_norm == "rms"
    assert experiment.transformer.final_norm == "rms"
    assert experiment.transformer.alibi is True
    assert experiment.transformer.rope == "timestamp_reverse"
    assert experiment.transformer.learned_positions == ("forward", "reverse")
    assert experiment.transformer.learned_position_fusion == "concat"
    assert experiment.transformer.learned_position_fusion_residual == "rezero"
    assert experiment.bos is True
    assert experiment.cls_token_mode == "end_only"
    assert experiment.timestamp_delta == "bins"
    assert experiment.negative_sampling == "random_offline_logq"
    assert experiment.logq_correction == "yi2019"
    assert experiment.correct_positive_logq is True
    assert experiment.num_in_batch_negatives == 2048
    assert experiment.dense_random_negative_scores is True
    assert experiment.lr_schedule.shape == "cosine"
    assert experiment.lr_schedule.warmup_fraction == 0.05
    assert experiment.lr_schedule.optimizer_group_scope == "deep_only"
    assert experiment.mup_base_dim == 16
    assert experiment.mup_delta_dim == 32
    assert experiment.item_embedding_dim == 64


@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("num_levels", TOKENIZER_LEVELS)
@pytest.mark.parametrize("num_codes", SHARED_CODEBOOK_SIZES)
def test_semantic_treatments_expose_only_the_approved_tokenizer_domain(
    representation: str, num_levels: int, num_codes: int
) -> None:
    experiment = build_semantic_treatment(
        backbone="best_g1",
        representation=representation,
        embedding_learning_rate=0.03,
        deep_learning_rate=0.01,
        num_levels=num_levels,
        num_codes=num_codes,
        run_name="g6_semantic_native500m",
    )

    assert isinstance(experiment, Native500MSemanticHistoryExperiment)
    assert experiment.history_representation == representation
    assert experiment.representation_width == 128
    assert experiment.semantic.num_levels == num_levels
    assert experiment.semantic.num_codes == num_codes
    assert experiment.semantic.kmeans_iterations == 300
    assert experiment.semantic.kmeans_relative_inertia_tolerance == 1e-4
    assert experiment.semantic.kmeans_assignment_early_stopping is True
    assert experiment.semantic.collision_policy == "suffix"


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_levels", 2), ("num_codes", 1024), ("representation_width", 64)],
)
def test_semantic_builder_rejects_values_outside_the_approved_domain(
    field: str, value: int
) -> None:
    arguments = {
        "backbone": "best_g1",
        "representation": "learned_sid_event",
        "embedding_learning_rate": 0.03,
        "deep_learning_rate": 0.01,
        "num_levels": 3,
        "num_codes": 512,
        "representation_width": 128,
        "run_name": "g6_invalid_native500m",
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        build_semantic_treatment(**arguments)


@pytest.mark.parametrize(
    "representation", APPLICABLE_INITIALIZATION_REPRESENTATIONS
)
@pytest.mark.parametrize("num_codes", SHARED_CODEBOOK_SIZES)
def test_content_initialization_is_available_for_every_trainable_family_and_code_count(
    representation: str, num_codes: int
) -> None:
    experiment = build_semantic_treatment(
        backbone="best_g1",
        representation=representation,
        embedding_learning_rate=0.03,
        deep_learning_rate=0.01,
        num_levels=3,
        num_codes=num_codes,
        sid_lookup_initialization="content_pca",
        run_name="g6_content_native500m",
    )

    assert experiment.sid_lookup_initialization == "content_pca"


def test_original_bridge_changes_only_history_representation() -> None:
    control = build_control(
        backbone="original_g1",
        embedding_learning_rate=0.04,
        deep_learning_rate=0.02,
        run_name="g6_original_control_native500m",
    )
    treatment = build_semantic_treatment(
        backbone="original_g1",
        representation="item_frozen_sid_event",
        embedding_learning_rate=0.04,
        deep_learning_rate=0.02,
        num_levels=3,
        num_codes=512,
        run_name="g6_original_bridge_native500m",
    )

    assert isinstance(treatment, ConventionalSemanticHistoryExperiment)
    assert not isinstance(treatment, MuTransferGenerationExperiment)
    assert not hasattr(treatment, "mup_base_dim")
    assert treatment.transformer == control.transformer
    shared = {
        field.name
        for field in fields(GenerationExperiment)
        if field.init
        and field.name
        not in {
            "run_name",
            "final_ranking_evidence_group",
        }
    }
    assert {
        name: getattr(treatment, name) for name in shared
    } == {name: getattr(control, name) for name in shared}


def test_original_bridge_builds_a_standard_model_and_adam_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "0")
    experiment = build_semantic_treatment(
        backbone="original_g1",
        representation="item_frozen_sid_event",
        embedding_learning_rate=0.04,
        deep_learning_rate=0.02,
        num_levels=3,
        num_codes=512,
        run_name="g6_original_bridge_native500m",
    )
    codes, codebooks = _semantic_artifacts()
    experiment.__dict__["item_embeddings"] = SimpleNamespace(num_known_ids=4)
    experiment.__dict__["artifacts"] = SimpleNamespace(
        item_id_column="compact_item_id"
    )
    experiment.__dict__["semantic_codes"] = codes
    experiment.__dict__["semantic_codebooks"] = codebooks

    model = experiment.base_model
    optimizer = experiment.create_optimizers()

    assert model.query_projection is None
    assert type(optimizer) is torch.optim.Adam


@pytest.mark.parametrize(
    "representation", APPLICABLE_INITIALIZATION_REPRESENTATIONS
)
def test_content_pca_initialization_is_generic_and_rng_nonadvancing(
    representation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "0")
    codes, codebooks = _semantic_artifacts()
    monkeypatch.setattr(
        Native500MSemanticHistoryExperiment,
        "semantic_codes",
        property(lambda _: codes),
    )
    monkeypatch.setattr(
        Native500MSemanticHistoryExperiment,
        "semantic_codebooks",
        property(lambda _: codebooks),
    )
    experiments = []
    models = []
    rng_states = []
    for initialization in ("random", "content_pca"):
        torch.manual_seed(37)
        experiment = build_semantic_treatment(
            backbone="best_g1",
            representation=representation,
            embedding_learning_rate=0.03,
            deep_learning_rate=0.01,
            num_levels=3,
            num_codes=512,
            sid_lookup_initialization=initialization,
            run_name=f"g6_{representation}_{initialization}_native500m",
        )
        experiment.__dict__["item_embeddings"] = SimpleNamespace(num_known_ids=4)
        experiment.__dict__["artifacts"] = SimpleNamespace(
            item_id_column="compact_item_id"
        )
        models.append(experiment.base_model)
        experiments.append(experiment)
        rng_states.append(torch.get_rng_state())

    random_lookup = learned_sid_lookup(models[0])
    content_lookup = learned_sid_lookup(models[1])
    random_weight = random_lookup.embedding.weight.detach()
    content_weight = content_lookup.embedding.weight.detach()
    vocabulary = experiments[0].semantic_codes.vocabulary
    base_rows = torch.zeros(len(random_weight), dtype=torch.bool)
    for level in range(3):
        first, last = vocabulary.level_range(level)
        base_rows[first:last] = True
        assert not torch.equal(random_weight[first:last], content_weight[first:last])
        torch.testing.assert_close(
            random_weight[first:last].square().mean().sqrt(),
            content_weight[first:last].square().mean().sqrt(),
            rtol=1e-5,
            atol=1e-8,
        )
    torch.testing.assert_close(random_weight[~base_rows], content_weight[~base_rows])
    assert torch.equal(rng_states[0], rng_states[1])
    assert experiments[1].sid_initialization_diagnostics["rng_nonadvancing"] is True


@pytest.mark.parametrize("representation", ["item_frozen_sid_event", "frozen_sid_tokens"])
def test_content_initialization_rejects_frozen_only_representations(
    representation: str,
) -> None:
    with pytest.raises(ValueError, match="trainable SID lookup"):
        build_semantic_treatment(
            backbone="best_g1",
            representation=representation,
            embedding_learning_rate=0.03,
            deep_learning_rate=0.01,
            num_levels=3,
            num_codes=512,
            sid_lookup_initialization="content_pca",
            run_name="g6_frozen_content_native500m",
        )


def test_collision_pair_changes_only_the_collision_policy() -> None:
    suffix, no_suffix = build_collision_pair(
        backbone="best_g1",
        representation="item_learned_frozen_sid_event",
        embedding_learning_rate=0.03,
        deep_learning_rate=0.01,
        num_levels=4,
        num_codes=2048,
        sid_lookup_initialization="content_pca",
        suffix_run_name="g6_rq2_suffix_native500m",
        no_suffix_run_name="g6_rq3_none_native500m",
    )

    assert suffix.semantic.collision_policy == "suffix"
    assert no_suffix.semantic.collision_policy == "none"
    assert suffix.sid_lookup_initialization == "content_pca"
    assert no_suffix.sid_lookup_initialization == "content_pca"
    suffix_semantic = replace(suffix.semantic, collision_policy="none")
    assert suffix_semantic == no_suffix.semantic
    common = {
        field.name
        for field in fields(Native500MSemanticHistoryExperiment)
        if field.init and field.name not in {"run_name", "semantic"}
    }
    assert {name: getattr(suffix, name) for name in common} == {
        name: getattr(no_suffix, name) for name in common
    }


def test_collision_pair_enforces_the_suffix_symbol_cap_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix, _ = build_collision_pair(
        backbone="best_g1",
        representation="item_frozen_sid_event",
        embedding_learning_rate=0.03,
        deep_learning_rate=0.01,
        num_levels=3,
        num_codes=8192,
        suffix_run_name="g6_rq2_suffix_native500m",
        no_suffix_run_name="g6_rq3_none_native500m",
    )
    suffix.__dict__["semantic_codes"] = SemanticCodes(
        item_ids=torch.tensor([1]),
        codes=torch.tensor([[0, 0, 0, 8192]]),
        codes_per_level=(8192, 8192, 8192, 8193),
    )
    suffix.__dict__["semantic_stage"] = SimpleNamespace(
        convergence_diagnostics=lambda: (
            {
                "levels": [
                    {"level": level, "stop_reason": "relative_inertia"}
                    for level in range(3)
                ]
            },
            "unused",
        )
    )
    monkeypatch.setattr(
        SemanticHistoryExperiment,
        "create_runner",
        lambda self: pytest.fail("training runner was created"),
    )

    with pytest.raises(ValueError, match="suffix level exceeds"):
        suffix.create_runner()


def test_semantic_run_rejects_a_kmeans_iteration_cap_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = build_semantic_treatment(
        backbone="best_g1",
        representation="item_frozen_sid_event",
        embedding_learning_rate=0.03,
        deep_learning_rate=0.01,
        num_levels=3,
        num_codes=512,
        run_name="g6_kmeans_cap_native500m",
    )
    experiment.__dict__["semantic_stage"] = SimpleNamespace(
        convergence_diagnostics=lambda: (
            {
                "levels": [
                    {"level": 0, "stop_reason": "relative_inertia"},
                    {"level": 1, "stop_reason": "max_iterations"},
                    {"level": 2, "stop_reason": "assignments_stable"},
                ]
            },
            "unused",
        )
    )
    monkeypatch.setattr(
        SemanticHistoryExperiment,
        "create_runner",
        lambda self: pytest.fail("training runner was created"),
    )

    with pytest.raises(RuntimeError, match="iteration cap at levels \\[1\\]"):
        experiment.create_runner()


@pytest.mark.parametrize(
    ("levels", "message"),
    [
        ([], "exactly 3 unique expected levels"),
        (
            [
                {"level": 0, "stop_reason": "relative_inertia"},
                {"level": 1, "stop_reason": "assignments_stable"},
            ],
            "exactly 3 unique expected levels",
        ),
        (
            [
                {"level": 0, "stop_reason": "relative_inertia"},
                {"level": 1, "stop_reason": "assignments_stable"},
                {"level": 1, "stop_reason": "relative_inertia"},
            ],
            "exactly 3 unique expected levels",
        ),
        (
            [
                {"level": 0, "stop_reason": "relative_inertia"},
                {"level": 1, "stop_reason": "assignments_stable"},
                {"level": 3, "stop_reason": "relative_inertia"},
            ],
            "exactly 3 unique expected levels",
        ),
        (
            [
                {"level": 0, "stop_reason": "relative_inertia"},
                "invalid",
                {"level": 2, "stop_reason": "assignments_stable"},
            ],
            "well-formed",
        ),
        (
            [
                {"level": 0, "stop_reason": "relative_inertia"},
                {"level": True, "stop_reason": "assignments_stable"},
                {"level": 2, "stop_reason": "relative_inertia"},
            ],
            "well-formed",
        ),
        (
            [
                {"level": 0, "stop_reason": "relative_inertia"},
                {"level": 1},
                {"level": 2, "stop_reason": "assignments_stable"},
            ],
            "well-formed",
        ),
        (
            [
                {"level": 0, "stop_reason": "relative_inertia"},
                {"level": 1, "stop_reason": "converged"},
                {"level": 2, "stop_reason": "assignments_stable"},
            ],
            "unknown stop reason",
        ),
    ],
)
def test_semantic_run_rejects_incomplete_or_malformed_kmeans_convergence(
    levels: list[object], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = build_semantic_treatment(
        backbone="best_g1",
        representation="item_frozen_sid_event",
        embedding_learning_rate=0.03,
        deep_learning_rate=0.01,
        num_levels=3,
        num_codes=512,
        run_name="g6_kmeans_invalid_native500m",
    )
    experiment.__dict__["semantic_stage"] = SimpleNamespace(
        convergence_diagnostics=lambda: ({"levels": levels}, "unused")
    )
    experiment.__dict__["semantic_codes"] = SemanticCodes.with_collision_suffix(
        item_ids=torch.tensor([1, 2, 3]),
        codes=torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        num_codes=512,
    )
    monkeypatch.setattr(
        SemanticHistoryExperiment,
        "create_runner",
        lambda self: pytest.fail("training runner was created"),
    )

    with pytest.raises(RuntimeError, match=message):
        experiment.create_runner()


def test_semantic_run_accepts_exact_successful_kmeans_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = build_semantic_treatment(
        backbone="best_g1",
        representation="item_frozen_sid_event",
        embedding_learning_rate=0.03,
        deep_learning_rate=0.01,
        num_levels=3,
        num_codes=512,
        run_name="g6_kmeans_converged_native500m",
    )
    experiment.__dict__["semantic_stage"] = SimpleNamespace(
        convergence_diagnostics=lambda: (
            {
                "levels": [
                    {"level": 0, "stop_reason": "relative_inertia"},
                    {"level": 1, "stop_reason": "assignments_stable"},
                    {"level": 2, "stop_reason": "relative_inertia"},
                ]
            },
            "unused",
        )
    )
    experiment.__dict__["semantic_codes"] = SemanticCodes.with_collision_suffix(
        item_ids=torch.tensor([1, 2, 3]),
        codes=torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        num_codes=512,
    )
    expected = SimpleNamespace()
    monkeypatch.setattr(
        SemanticHistoryExperiment,
        "create_runner",
        lambda self: expected,
    )

    assert experiment.create_runner() is expected


def test_learned_lookup_resolver_requires_exactly_one_base_lookup() -> None:
    with pytest.raises(RuntimeError, match="exactly one trainable base SID lookup"):
        learned_sid_lookup(nn.Sequential(nn.Linear(2, 2)))


@pytest.mark.parametrize("kind", ["control", "semantic"])
def test_full_catalog_and_sid_diagnostics_are_deferred_to_selected_winners(
    kind: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = (
        build_control(
            backbone="best_g1",
            embedding_learning_rate=0.03,
            deep_learning_rate=0.01,
            run_name="g6_control_native500m",
        )
        if kind == "control"
        else build_semantic_treatment(
            backbone="best_g1",
            representation="item_frozen_sid_event",
            embedding_learning_rate=0.03,
            deep_learning_rate=0.01,
            num_levels=3,
            num_codes=512,
            run_name="g6_semantic_native500m",
        )
    )
    calls: list[str] = []
    monkeypatch.setattr(global_config, "_base_path", tmp_path)
    model = nn.Linear(3, 2)
    selected_state = {
        name: torch.full_like(tensor, 0.25)
        for name, tensor in model.state_dict().items()
    }

    def restore_selected(target: nn.Module) -> bool:
        target.load_state_dict(selected_state)
        return True

    experiment.__dict__["base_model"] = model
    experiment.__dict__["callbacks"] = SimpleNamespace(
        best_weights=SimpleNamespace(
            best_epoch=7,
            restore=restore_selected,
        )
    )
    monkeypatch.setattr(
        "dcn.config.experiment.Experiment.finish",
        lambda self, runner: calls.append("checkpoint"),
    )
    metadata_path = global_config.logs_path / experiment.run_name / "training_metadata.json"

    def write_metadata(runner) -> None:
        calls.append("metadata")
        destination = (
            global_config.logs_path / experiment.run_name / "training_metadata.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        architecture = experiment.generation_architecture_metadata()
        destination.write_text(
            json.dumps(
                {
                    "best_epoch": 8,
                    "best_model_artifact": architecture["best_model_artifact"],
                }
            )
        )

    monkeypatch.setattr(experiment, "_report_training_metadata", write_metadata)
    monkeypatch.setattr(
        experiment,
        "_report_final_metrics",
        lambda runner: pytest.fail("full-catalog scoring ran"),
    )
    if hasattr(experiment, "semantic_diagnostics_document"):
        monkeypatch.setattr(
            experiment,
            "semantic_diagnostics_document",
            lambda: pytest.fail("SID diagnostics ran"),
        )

    assert experiment.true_metric_options() == {}

    experiment.finish(SimpleNamespace(model=model))

    assert calls == ["checkpoint", "metadata"]
    artifact_path = global_config.logs_path / experiment.run_name / "best_model_state.pt"
    payload = artifact_path.read_bytes()
    assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o444
    saved = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert saved.keys() == selected_state.keys()
    for name, tensor in selected_state.items():
        torch.testing.assert_close(saved[name], tensor)
    artifact = json.loads(metadata_path.read_text())["best_model_artifact"]
    assert artifact == {
        "schema": "g6-best-model-state/v1",
        "path": f"logs/{experiment.run_name}/best_model_state.pt",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_deferred_finish_requires_a_validation_selected_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = build_control(
        backbone="best_g1",
        embedding_learning_rate=0.03,
        deep_learning_rate=0.01,
        run_name="g6_no_best_native500m",
    )
    monkeypatch.setattr(global_config, "_base_path", tmp_path)
    model = nn.Linear(2, 2)
    experiment.__dict__["base_model"] = model
    experiment.__dict__["callbacks"] = SimpleNamespace(
        best_weights=SimpleNamespace(best_epoch=None, restore=lambda target: False)
    )
    monkeypatch.setattr("dcn.config.experiment.Experiment.finish", lambda *args: None)

    with pytest.raises(RuntimeError, match="validation-selected best weights"):
        experiment.finish(SimpleNamespace(model=model))

    assert not (global_config.logs_path / experiment.run_name).exists()


def test_best_model_artifact_is_immutable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = build_control(
        backbone="best_g1",
        embedding_learning_rate=0.03,
        deep_learning_rate=0.01,
        run_name="g6_immutable_best_native500m",
    )
    monkeypatch.setattr(global_config, "_base_path", tmp_path)
    model = nn.Linear(2, 2)
    selected = {
        name: torch.zeros_like(tensor) for name, tensor in model.state_dict().items()
    }

    def restore_selected(target: nn.Module) -> bool:
        target.load_state_dict(selected)
        return True

    experiment.__dict__["base_model"] = model
    experiment.__dict__["callbacks"] = SimpleNamespace(
        best_weights=SimpleNamespace(
            best_epoch=1,
            restore=restore_selected,
        )
    )
    monkeypatch.setattr("dcn.config.experiment.Experiment.finish", lambda *args: None)

    def write_metadata(runner) -> None:
        destination = global_config.logs_path / experiment.run_name / "training_metadata.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}")

    monkeypatch.setattr(experiment, "_report_training_metadata", write_metadata)
    runner = SimpleNamespace(model=model)
    experiment.finish(runner)
    for tensor in selected.values():
        tensor.fill_(1)

    with pytest.raises(RuntimeError, match="immutable best-model artifact changed"):
        experiment.finish(runner)
