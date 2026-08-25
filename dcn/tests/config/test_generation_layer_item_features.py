from collections.abc import Iterator
from dataclasses import replace

import pytest
import torch

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment
from dcn.config.settings import TRANSFORMER
from dcn.nn.layer_item_features import (
    ConcatenatedItemFeatureResidual,
    DirectAddItemFeature,
    GemmaItemFeatureResidual,
)
from dcn.tests.helpers import packed_batch
from dcn.tests.miniature_yambda import configure
from utils.global_config import config as global_config

pytestmark = pytest.mark.usefixtures("cpu_attention")


@pytest.fixture(autouse=True)
def _build_models_on_cpu(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    flash = torch.backends.cuda.flash_sdp_enabled()
    memory_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
    math_sdp = torch.backends.cuda.math_sdp_enabled()
    monkeypatch.setenv("DCN_GPU_LOCK_SLOT", "test")
    yield
    torch.backends.cuda.enable_flash_sdp(flash)
    torch.backends.cuda.enable_mem_efficient_sdp(memory_efficient)
    torch.backends.cuda.enable_math_sdp(math_sdp)


def _transformer():
    return replace(
        TRANSFORMER,
        dim=8,
        num_layers=4,
        nhead=2,
        num_kv_heads=1,
        ffn_intermediate_dim=16,
        dropout=0.0,
        ffn_dropout=0.0,
    )


def _experiment(base_path, family: str, feature_dim: int | None = None):
    experiment = configure(
        MuTransferGenerationExperiment,
        base_path,
        transformer=_transformer(),
        mup_base_dim=4,
        mup_delta_dim=8,
        per_layer_item_features=family,
        per_layer_item_feature_dim=feature_dim,
        initializer_std=0.02,
    )
    experiment.setup()
    global_config.set_cpu_attention(True)
    return experiment


@pytest.mark.parametrize(
    "family, feature_dim, fusion_type, expected_width",
    [
        ("direct_add", None, DirectAddItemFeature, 8),
        ("concat_residual", 4, ConcatenatedItemFeatureResidual, 4),
        ("gemma_ple", 4, GemmaItemFeatureResidual, 4),
    ],
)
def test_mu_transfer_builds_four_compatible_layer_features(
    base_path,
    family: str,
    feature_dim: int | None,
    fusion_type: type,
    expected_width: int,
) -> None:
    model = _experiment(base_path, family, feature_dim).base_model

    assert len(model.layer_item_embeddings) == 4
    assert len(model.layer_item_feature_fusions) == 4
    assert all(
        embedding.embedding_dim == expected_width
        for embedding in model.layer_item_embeddings
    )
    assert all(
        isinstance(fusion, fusion_type) for fusion in model.layer_item_feature_fusions
    )
    assert all(hasattr(parameter, "infshape") for parameter in model.parameters())


def test_concat_mu_transfer_keeps_feature_and_model_width_axes_separate(
    base_path,
) -> None:
    model = _experiment(base_path, "concat_residual", 4).base_model

    for fusion in model.layer_item_feature_fusions:
        feature_shape = fusion.feature_projection.weight.infshape
        dense_input_shape = fusion.encoder.input_projection.weight.infshape
        dense_output_shape = fusion.encoder.output_projection.weight.infshape

        assert feature_shape[0].isinf()
        assert not feature_shape[1].isinf()
        assert all(dimension.isinf() for dimension in dense_input_shape)
        assert all(dimension.isinf() for dimension in dense_output_shape)


def test_auxiliary_embeddings_and_fusions_use_their_approved_optimizer_groups(
    base_path,
) -> None:
    experiment = _experiment(base_path, "concat_residual", 4)
    model = experiment.base_model
    optimizer = experiment.create_optimizers()
    schedule_group_by_parameter = {
        id(parameter): group["schedule_group"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert all(
        schedule_group_by_parameter[id(embedding.weight)] == "embedding"
        for embedding in model.layer_item_embeddings
    )
    assert all(
        schedule_group_by_parameter[id(parameter)] == "deep"
        for fusion in model.layer_item_feature_fusions
        for parameter in fusion.parameters()
    )


@pytest.mark.parametrize("family", ["concat_residual", "gemma_ple"])
def test_zero_start_mu_transfer_model_equals_its_seeded_control(
    base_path, family: str
) -> None:
    control = _experiment(base_path, "none").base_model.eval()
    treatment = _experiment(base_path, family, 4).base_model.eval()
    control_parameters = dict(control.named_parameters())
    treatment_parameters = dict(treatment.named_parameters())

    for name in control_parameters.keys() & treatment_parameters.keys():
        assert torch.equal(control_parameters[name], treatment_parameters[name]), name
    assert all(
        fusion.residual_scale.item() == 0
        for fusion in treatment.layer_item_feature_fusions
    )

    projection = torch.arange(64 * 8, dtype=torch.float32).reshape(64, 8) / 100
    control.query_projection.weight.data.copy_(projection)
    treatment.query_projection.weight.data.copy_(projection)
    batch = packed_batch([1, 2, 3, 4, 5], [3, 2])

    assert torch.equal(control.encode_queries(batch), treatment.encode_queries(batch))


def test_legacy_per_layer_embedding_flag_selects_direct_add(base_path) -> None:
    experiment = configure(
        MuTransferGenerationExperiment,
        base_path,
        transformer=_transformer(),
        mup_base_dim=4,
        mup_delta_dim=8,
        per_layer_item_embeddings=True,
    )
    experiment.setup()

    assert experiment.effective_per_layer_item_features == "direct_add"
    assert all(
        isinstance(fusion, DirectAddItemFeature)
        for fusion in experiment.base_model.layer_item_feature_fusions
    )


@pytest.mark.parametrize("family", ["concat_residual", "gemma_ple"])
def test_compact_families_require_a_positive_width(family: str) -> None:
    with pytest.raises(ValueError, match="positive per_layer_item_feature_dim"):
        GenerationExperiment(per_layer_item_features=family)


def test_legacy_direct_add_rejects_another_explicit_family() -> None:
    with pytest.raises(ValueError, match="legacy direct_add"):
        GenerationExperiment(
            per_layer_item_embeddings=True,
            per_layer_item_features="gemma_ple",
            per_layer_item_feature_dim=4,
        )
