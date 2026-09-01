from dataclasses import fields, replace
import os
from pathlib import Path
import runpy

import torch
from torch import nn
import pytest

from dcn.config import GenerationExperiment, MuTransferGenerationExperiment
from dcn.datasets.base import DatasetSourceArtifacts
from dcn.nn import (
    ContentProjection,
    FrequencyContentGate,
    ItemContentCatalogEncoder,
    ItemContentDenseNetEncoder,
    ItemMetadataDenseNetEncoder,
    PrecomputedEmbeddingLookup,
    PretrainedCatalogEncoder,
)
from dcn.tests.helpers import packed_batch
from experiments.g3_pretrained_item_embeddings.configs.model import (
    G3Representation,
    RQ3_CATALOG_REPRESENTATIONS,
    build_native500m_job,
    build_rq3_representation,
    build_g3_experiment,
)
from experiments.g3_pretrained_item_embeddings.data import LoadedFeatureData
from experiments.g3_pretrained_item_embeddings.diagnostics import (
    G3DiagnosticsCallback,
    G3GateDiagnosticsCallback,
)
from experiments.g4_future_items.configs.control import G4GenerationExperiment
from utils.global_config import config as global_config

pytestmark = pytest.mark.usefixtures("cpu_attention")


def _content(num_items: int = 5) -> PrecomputedEmbeddingLookup:
    values = torch.arange(num_items * 128, dtype=torch.float32).reshape(num_items, 128)
    values = torch.nn.functional.normalize(values + 1, dim=-1)
    return PrecomputedEmbeddingLookup(values, learnable_default=False, strict=False)


def _experiment(
    representation: G3Representation,
    *,
    feature_data: LoadedFeatureData | None = None,
    gate_mechanism_diagnostics: bool = False,
):
    feature_data = _feature_data() if feature_data is None else feature_data
    experiment = build_g3_experiment(
        run_name="g3_test",
        dataset_size="native-50m",
        embedding_learning_rate=0.12,
        deep_learning_rate=0.03,
        lr_schedule_horizon_epochs=15,
        representation=representation,
        feature_data_path=Path("unused.parquet"),
        gate_mechanism_diagnostics=gate_mechanism_diagnostics,
    )
    experiment.__dict__["item_embeddings"] = _content()
    experiment.__dict__["device"] = torch.device("cpu")
    experiment.__dict__["artifacts"] = DatasetSourceArtifacts(
        main_parquet=Path("unused-events.parquet"),
        columns=["compact_item_id"],
        precomputed_embeddings={"compact_item_id": Path("unused-content.parquet")},
        timestamp_column="timestamp",
        user_column="uid",
        item_id_column="compact_item_id",
    )
    experiment.__dict__["g3_feature_data"] = feature_data
    return experiment


def _native500m_experiment(
    representation: G3Representation,
    *,
    feature_data: LoadedFeatureData | None = None,
):
    experiment = build_g3_experiment(
        run_name="g3_native500m_test",
        dataset_size="native-500m",
        embedding_learning_rate=0.0468526465053628,
        deep_learning_rate=0.032703745675187676,
        lr_schedule_horizon_epochs=20,
        representation=representation,
        feature_data_path=Path("unused-native500m-features.pt"),
    )
    experiment.__dict__["item_embeddings"] = _content()
    experiment.__dict__["device"] = torch.device("cpu")
    experiment.__dict__["artifacts"] = DatasetSourceArtifacts(
        main_parquet=Path("unused-native500m-events.parquet"),
        columns=["compact_item_id"],
        precomputed_embeddings={
            "compact_item_id": Path("unused-native500m-content.parquet")
        },
        timestamp_column="timestamp",
        user_column="uid",
        item_id_column="compact_item_id",
    )
    experiment.__dict__["g3_feature_data"] = (
        _feature_data() if feature_data is None else feature_data
    )
    return experiment


def _feature_data() -> LoadedFeatureData:
    return LoadedFeatureData(
        training_counts=torch.tensor([0, 10, 8, 6, 4, 2]),
        training_history_lengths={1: 3, 2: 2, 3: 1},
        artist_rows=((), (1, 2), (2,), (), (3,), (1,)),
        album_rows=((), (1,), (), (2,), (2, 3), (3,)),
        artist_vocab_size=3,
        album_vocab_size=3,
    )


@pytest.mark.parametrize(
    ("field", "error"),
    (
        ("history_representation", "history representation"),
        ("catalog_representation", "catalog representation"),
        ("content_gate", "content gate"),
    ),
)
def test_representation_schema_rejects_unknown_values(field: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        G3Representation(**{field: "typo"})


def test_representation_schema_requires_compatible_explicit_item_id_tying() -> None:
    with pytest.raises(ValueError, match="item-ID tying"):
        G3Representation(item_id_tying="typo")
    with pytest.raises(ValueError, match="item-ID tying"):
        G3Representation(
            history_representation="content",
            catalog_representation="learned_id",
            item_id_tying="tied",
        )
    with pytest.raises(ValueError, match="item-ID tying"):
        G3Representation(
            history_representation="id_content",
            history_hidden_dim=64,
            catalog_representation="frozen_content",
            item_id_tying="tied",
        )


def test_representation_payload_round_trips_and_rejects_schema_drift() -> None:
    representation = G3Representation(
        history_representation="id_content",
        history_hidden_dim=128,
        catalog_representation="id_trainable_content",
        item_id_tying="untied",
    )

    assert G3Representation.from_dict(representation.to_dict()) == representation
    with pytest.raises(ValueError, match="unknown representation fields"):
        G3Representation.from_dict({**representation.to_dict(), "typo": 1})
    incomplete = representation.to_dict()
    incomplete.pop("content_gate")
    with pytest.raises(ValueError, match="missing representation fields"):
        G3Representation.from_dict(incomplete)


def test_native500m_job_adapter_builds_only_the_explicit_matching_payload() -> None:
    representation = G3Representation(
        history_representation="id_content",
        history_hidden_dim=128,
        item_id_tying="tied",
    )
    job = {
        "run_name": "g3_native500m_rq2_test",
        "family_id": "rq2_content_concat",
        "batch_size": 512,
        "seed": 42,
        "horizon_epochs": 20,
        "embedding_learning_rate": "0.046852646505362798",
        "deep_learning_rate": "0.032703745675187676",
        "capacity": 128,
        "resolved_representation": representation.to_dict(),
    }

    experiment = build_native500m_job(
        job,
        feature_data_path=Path("unused-native500m-features.pt"),
    )

    assert experiment.representation == representation
    assert experiment.run_name == job["run_name"]
    assert experiment.embedding_learning_rate == float(job["embedding_learning_rate"])
    assert experiment.deep_learning_rate == float(job["deep_learning_rate"])

    mismatched = {**job, "family_id": "baseline"}
    with pytest.raises(ValueError, match="does not match family"):
        build_native500m_job(
            mismatched,
            feature_data_path=Path("unused-native500m-features.pt"),
        )


def test_native500m_job_adapter_fails_closed_without_resolved_representation() -> None:
    with pytest.raises(ValueError, match="resolved_representation"):
        build_native500m_job(
            {
                "run_name": "g3_native500m_missing_representation",
                "family_id": "baseline",
                "batch_size": 512,
                "seed": 42,
                "horizon_epochs": 20,
                "embedding_learning_rate": "0.046852646505362798",
                "deep_learning_rate": "0.032703745675187676",
                "capacity": None,
                "resolved_representation": None,
            },
            feature_data_path=Path("unused-native500m-features.pt"),
        )


def test_native500m_direct_builder_rejects_undeclared_extra_item_id_branch() -> None:
    with pytest.raises(ValueError, match="extra_item_id_dim"):
        build_g3_experiment(
            run_name="g3_native500m_invalid_direct_extra_item_id",
            dataset_size="native-500m",
            embedding_learning_rate=0.0468526465053628,
            deep_learning_rate=0.032703745675187676,
            lr_schedule_horizon_epochs=20,
            representation=G3Representation(
                extra_item_id_dim=32,
                item_id_tying="tied",
            ),
            feature_data_path=Path("unused-native500m-features.pt"),
        )


@pytest.mark.parametrize(
    ("family_id", "capacity", "representation"),
    (
        (
            "rq3_output_frozen_content",
            None,
            G3Representation(
                history_representation="id_content",
                history_hidden_dim=128,
                catalog_representation="frozen_content",
                extra_item_id_dim=32,
                item_id_tying="untied",
            ),
        ),
        (
            "rq5_global_gate",
            None,
            G3Representation(
                history_representation="id_content",
                history_hidden_dim=128,
                content_gate="global",
                extra_item_id_dim=32,
                item_id_tying="tied",
            ),
        ),
        (
            "rq5_frequency_gate",
            16.0,
            G3Representation(
                history_representation="id_content",
                history_hidden_dim=128,
                content_gate="frequency",
                gate_hidden_dim=16,
                frequency_gate_semantics="fp32_p09_v2",
                item_id_tying="tied",
            ),
        ),
    ),
)
def test_native500m_active_families_reject_undeclared_fields_and_float_capacity(
    family_id: str,
    capacity: object,
    representation: G3Representation,
) -> None:
    with pytest.raises(ValueError, match="representation|capacity"):
        build_native500m_job(
            {
                "run_name": f"g3_native500m_invalid_{family_id}",
                "family_id": family_id,
                "batch_size": 512,
                "seed": 42,
                "horizon_epochs": 20,
                "embedding_learning_rate": "0.046852646505362798",
                "deep_learning_rate": "0.032703745675187676",
                "capacity": capacity,
                "resolved_representation": representation.to_dict(),
            },
            feature_data_path=Path("unused-native500m-features.pt"),
        )


@pytest.mark.parametrize(
    ("family_id", "capacity", "representation"),
    (
        (
            "rq2_content_concat",
            32,
            G3Representation(
                history_representation="id_content",
                history_hidden_dim=32,
                item_id_tying="tied",
            ),
        ),
        (
            "rq2_content_concat",
            512,
            G3Representation(
                history_representation="id_content",
                history_hidden_dim=512,
                item_id_tying="tied",
            ),
        ),
        (
            "rq4_artist",
            128,
            G3Representation(
                metadata=("artist",),
                metadata_dim=128,
                item_id_tying="tied",
            ),
        ),
        (
            "rq5_frequency_gate",
            16,
            G3Representation(
                history_representation="id_content",
                history_hidden_dim=512,
                content_gate="frequency",
                gate_hidden_dim=16,
                frequency_gate_semantics="fp32_p09_v2",
                item_id_tying="tied",
            ),
        ),
        (
            "rq5_frequency_gate",
            128,
            G3Representation(
                history_representation="id_content",
                history_hidden_dim=512,
                content_gate="frequency",
                gate_hidden_dim=128,
                frequency_gate_semantics="fp32_p09_v2",
                item_id_tying="tied",
            ),
        ),
        (
            "rq3_output_frozen_content",
            None,
            G3Representation(
                history_representation="id_content",
                history_hidden_dim=512,
                catalog_representation="frozen_content",
                item_id_tying="untied",
            ),
        ),
    ),
)
def test_native500m_job_adapter_accepts_approved_boundary_capacities(
    family_id: str,
    capacity: int | None,
    representation: G3Representation,
) -> None:
    experiment = build_native500m_job(
        {
            "run_name": f"g3_native500m_{family_id}_{capacity}",
            "family_id": family_id,
            "batch_size": 512,
            "seed": 42,
            "horizon_epochs": 20,
            "embedding_learning_rate": "0.046852646505362798",
            "deep_learning_rate": "0.032703745675187676",
            "capacity": capacity,
            "resolved_representation": representation.to_dict(),
        },
        feature_data_path=Path("unused-native500m-features.pt"),
    )

    assert experiment.representation == representation


@pytest.mark.parametrize(
    ("family_id", "representation"),
    (
        (
            "aggregate",
            G3Representation(
                history_representation="id_only_densenet",
                history_hidden_dim=64,
                item_id_tying="untied",
            ),
        ),
        (
            "aggregate",
            G3Representation(extra_item_id_dim=32, item_id_tying="tied"),
        ),
        (
            "aggregate",
            G3Representation(
                history_representation="id_content",
                history_hidden_dim=128,
                content_gate="global",
                item_id_tying="untied",
            ),
        ),
        ("bridge_rq3_output", G3Representation(item_id_tying="tied")),
        ("bridge_rq4_metadata", G3Representation(item_id_tying="tied")),
    ),
)
def test_native500m_conditional_jobs_reject_out_of_envelope_compositions(
    family_id: str,
    representation: G3Representation,
) -> None:
    with pytest.raises(ValueError, match="representation|composition|bridge"):
        build_native500m_job(
            {
                "run_name": f"g3_native500m_invalid_{family_id}",
                "family_id": family_id,
                "batch_size": 512,
                "seed": 42,
                "horizon_epochs": 20,
                "embedding_learning_rate": "0.046852646505362798",
                "deep_learning_rate": "0.032703745675187676",
                "capacity": None,
                "resolved_representation": representation.to_dict(),
            },
            feature_data_path=Path("unused-native500m-features.pt"),
        )


def test_native500m_control_is_exact_two_layer_g1_best_without_mup() -> None:
    experiment = _native500m_experiment(
        G3Representation(item_id_tying="tied")
    )
    transformer = experiment.transformer

    assert isinstance(experiment, GenerationExperiment)
    assert not isinstance(experiment, MuTransferGenerationExperiment)
    assert experiment.size == "500m"
    assert experiment.g3_dataset_size == "native-500m"
    assert experiment.final_ranking_evidence_group == "g3-native500m-likes"
    assert experiment.dataloader.batch_size == 512
    assert experiment.max_seq_len == 100
    assert experiment.bos
    assert experiment.cls_token_mode == "end_only"
    assert experiment.timestamp_delta == "bins"
    assert experiment.timestamp_combination == "add"
    assert experiment.timestamp_num_bins == 32
    assert experiment.negative_sampling == "random_offline_logq"
    assert experiment.logq_correction == "yi2019"
    assert experiment.logq_alpha == 0.01
    assert experiment.random_negative_fraction == 0.5
    assert experiment.correct_positive_logq
    assert not experiment.mask_false_negatives
    assert not experiment.exclude_own_group_negatives
    assert experiment.num_in_batch_negatives == 2048
    assert experiment.dense_random_negative_scores
    assert experiment.lr_schedule.shape == "cosine"
    assert experiment.lr_schedule.warmup_fraction == 0.05
    assert experiment.lr_schedule.cycles == 1
    assert experiment.lr_schedule.optimizer_group_scope == "deep_only"
    assert experiment.num_epochs == 20
    assert experiment.lr_schedule_horizon_epochs == 20
    assert not experiment.adaptive_schedule_early_stopping
    assert transformer.dim == 64
    assert transformer.num_layers == 2
    assert transformer.nhead == 2
    assert transformer.num_kv_heads == 1
    assert transformer.ffn == "swiglu"
    assert transformer.ffn_intermediate_dim == 192
    assert transformer.gated_ffn_dropout
    assert transformer.dropout == 0.1
    assert transformer.input_dropout == 0.1
    assert transformer.ffn_dropout == 0.1
    assert transformer.norm_place == "post"
    assert transformer.input_norm == "rms"
    assert transformer.final_norm == "rms"
    assert transformer.alibi
    assert transformer.rope == "timestamp_reverse"
    assert transformer.learned_positions == ("forward", "reverse")
    assert transformer.learned_position_fusion == "concat"
    assert transformer.learned_position_fusion_residual == "rezero"
    assert transformer.learned_position_reverse_correction == "bounded_tanh"
    assert transformer.learned_position_reverse_max_scale == 0.025
    assert transformer.learned_position_reverse_initializer_rng_nonadvancing
    assert transformer.attention_window is None


def test_native500m_control_matches_authenticated_g1_aggregate_invariants() -> None:
    run_name = (
        "g1_aggregate_aggregate_none_l4_"
        "e0p0468526465053628_d0p032703745675187676_"
        "h15_c0_initial_ts2_r1_500m"
    )
    previous = os.environ.get("G1_AGGREGATE_RUN")
    os.environ["G1_AGGREGATE_RUN"] = run_name
    try:
        g1 = runpy.run_path(
            str(
                Path(__file__).resolve().parents[3]
                / "experiments/g1_sasrec_item_ids_likes/configs/aggregate_variant.py"
            )
        )["experiment"]
    finally:
        if previous is None:
            os.environ.pop("G1_AGGREGATE_RUN", None)
        else:
            os.environ["G1_AGGREGATE_RUN"] = previous
    g3 = _native500m_experiment(G3Representation(item_id_tying="tied"))

    allowed_differences = {
        "run_name",
        "dataloader",
        "num_epochs",
        "lr_schedule_horizon_epochs",
        "embedding_learning_rate",
        "deep_learning_rate",
        "transformer",
        "final_ranking_evidence_group",
    }
    for field in fields(GenerationExperiment):
        if field.name not in allowed_differences:
            assert getattr(g3, field.name) == getattr(g1, field.name), field.name
    assert g3.dataloader == replace(g1.dataloader, batch_size=512)
    assert g3.transformer == replace(g1.transformer, num_layers=2)


def test_native500m_baseline_shares_one_history_and_catalog_item_table() -> None:
    experiment = _native500m_experiment(
        G3Representation(item_id_tying="tied")
    )
    model = experiment.base_model

    assert isinstance(experiment.item_embedding, nn.Embedding)
    assert experiment.item_embedding is experiment.catalog_item_encoder
    assert experiment.item_embedding.padding_idx == 0
    assert torch.count_nonzero(experiment.item_embedding.weight[0]) == 0
    assert model.catalog_item_encoder is None
    experiment.item_embedding(torch.tensor([0, 1])).sum().backward()
    assert experiment.item_embedding.weight.grad is not None
    assert torch.count_nonzero(experiment.item_embedding.weight.grad[0]) == 0
    assert torch.count_nonzero(experiment.item_embedding.weight.grad[1]) > 0


@pytest.mark.parametrize(
    ("representation", "plain_encoder_names"),
    (
        (G3Representation(item_id_tying="tied"), ("item_embedding",)),
        (
            G3Representation(item_id_tying="untied"),
            ("item_embedding", "catalog_item_encoder"),
        ),
        (
            G3Representation(history_representation="content"),
            ("catalog_item_encoder",),
        ),
        (
            build_rq3_representation(
                "rq3_output_learned",
                history_hidden_dim=128,
                item_id_tying="untied",
            ),
            ("catalog_item_encoder",),
        ),
    ),
)
def test_native500m_plain_item_tables_have_safe_id_value_and_gradient_semantics(
    representation: G3Representation,
    plain_encoder_names: tuple[str, ...],
) -> None:
    experiment = _native500m_experiment(representation)
    _ = experiment.base_model

    for name in plain_encoder_names:
        encoder = getattr(experiment, name)
        assert isinstance(encoder, nn.Embedding)
        encoder.zero_grad(set_to_none=True)
        item_ids = torch.tensor(
            [-1, 0, 1, experiment.num_items, experiment.catalog_size]
        )

        output = encoder(item_ids)

        assert torch.equal(output[[0, 1, 4]], torch.zeros(3, experiment.model_dim))
        torch.testing.assert_close(
            output[2:4],
            encoder.weight.detach()[torch.tensor([1, experiment.num_items])],
        )
        assert torch.count_nonzero(encoder.weight[0]) == 0
        output.sum().backward()
        assert encoder.weight.grad is not None
        assert torch.count_nonzero(encoder.weight.grad[0]) == 0
        assert torch.count_nonzero(encoder.weight.grad[1]) > 0
        assert torch.count_nonzero(encoder.weight.grad[experiment.num_items]) > 0


def test_native500m_rq2_injects_the_catalog_item_table_into_history() -> None:
    experiment = _native500m_experiment(
        G3Representation(
            history_representation="id_content",
            history_hidden_dim=128,
            item_id_tying="tied",
        )
    )
    history = experiment.item_embedding
    catalog = experiment.catalog_item_encoder

    assert isinstance(history, ItemContentDenseNetEncoder)
    assert isinstance(catalog, nn.Embedding)
    assert history.item_embedding is catalog
    assert history.normalize_content
    model = experiment.base_model
    assert catalog.padding_idx == 0
    assert torch.count_nonzero(catalog.weight[0]) == 0
    embedding, deep = experiment.split_parameters(model, experiment.embedding_types)
    assert sum(parameter is catalog.weight for parameter in embedding) == 1
    assert all(parameter is not catalog.weight for parameter in deep)


@pytest.mark.parametrize(
    "item_id_tying",
    (
        pytest.param("tied", id="rq2-tied-history"),
        pytest.param("untied", id="rq3-untied-history"),
    ),
)
def test_native500m_composed_history_item_branch_is_safe_after_initialization(
    item_id_tying: str,
) -> None:
    experiment = _native500m_experiment(
        G3Representation(
            history_representation="id_content",
            history_hidden_dim=128,
            item_id_tying=item_id_tying,
        )
    )
    model = experiment.base_model
    history = model.item_embedding
    assert isinstance(history, ItemContentDenseNetEncoder)
    item_ids = torch.tensor([-1, 0, 1, experiment.num_items, experiment.catalog_size])

    learned = history.composed_features(item_ids)[:, : experiment.model_dim]

    assert torch.equal(learned[[0, 1, 4]], torch.zeros(3, experiment.model_dim))
    assert torch.count_nonzero(history.item_embedding.weight[0]) == 0
    learned.sum().backward()
    assert history.item_embedding.weight.grad is not None
    assert torch.count_nonzero(history.item_embedding.weight.grad[0]) == 0
    assert torch.count_nonzero(history.item_embedding.weight.grad[1]) > 0


def test_native500m_combined_catalog_item_branch_is_safe_after_initialization() -> (
    None
):
    experiment = _native500m_experiment(
        build_rq3_representation(
            "rq3_output_learned_frozen_content",
            history_hidden_dim=128,
            item_id_tying="untied",
        )
    )
    model = experiment.base_model
    catalog = model.catalog_item_encoder
    assert isinstance(catalog, ItemContentCatalogEncoder)
    item_ids = torch.tensor([-1, 0, 1, experiment.num_items, experiment.catalog_size])

    output = catalog(item_ids)

    assert torch.equal(output[[0, 1, 4]], torch.zeros(3, experiment.model_dim))
    assert torch.count_nonzero(catalog.item_embedding.weight[0]) == 0
    output.sum().backward()
    assert catalog.item_embedding.weight.grad is not None
    assert torch.count_nonzero(catalog.item_embedding.weight.grad[0]) == 0
    assert torch.count_nonzero(catalog.item_embedding.weight.grad[1]) > 0


@pytest.mark.parametrize(
    "catalog_representation",
    (
        "frozen_content",
        "trainable_content",
        "id_frozen_content",
        "id_trainable_content",
    ),
)
def test_native500m_enables_lookup_normalization_for_every_content_target(
    catalog_representation: str,
) -> None:
    experiment = _native500m_experiment(
        G3Representation(
            history_representation="content",
            catalog_representation=catalog_representation,
        )
    )

    assert isinstance(experiment.item_embedding, ContentProjection)
    assert experiment.item_embedding.normalize_content
    assert isinstance(
        experiment.catalog_item_encoder,
        (PretrainedCatalogEncoder, ItemContentCatalogEncoder),
    )
    assert experiment.catalog_item_encoder.content.normalize_content


def test_native500m_rq4_shares_one_complete_metadata_encoder() -> None:
    experiment = _native500m_experiment(
        G3Representation(
            metadata=("artist", "album"),
            metadata_dim=32,
            item_id_tying="tied",
        )
    )

    assert isinstance(experiment.item_embedding, ItemMetadataDenseNetEncoder)
    assert experiment.item_embedding is experiment.catalog_item_encoder
    model = experiment.base_model
    assert model.catalog_item_encoder is None
    item_ids = torch.tensor([1, 2, 3, 4, 5])
    (
        experiment.item_embedding(item_ids).sum()
        + model.encode_item_ids(item_ids).sum()
    ).backward()
    assert experiment.item_embedding.item_encoder.weight.grad is not None
    assert all(
        branch.embedding.weight.grad is not None
        for branch in experiment.item_embedding.metadata_branches
    )


def test_native500m_rq4_metadata_branches_are_safe_after_initialization() -> None:
    experiment = _native500m_experiment(
        G3Representation(
            metadata=("artist", "album"),
            metadata_dim=32,
            item_id_tying="tied",
        )
    )
    _ = experiment.base_model
    encoder = experiment.item_embedding
    assert isinstance(encoder, ItemMetadataDenseNetEncoder)
    item_ids = torch.tensor([-1, 0, 1, experiment.num_items, experiment.catalog_size])

    for branch in encoder.metadata_branches:
        branch.zero_grad(set_to_none=True)
        output = branch(item_ids)
        assert torch.equal(output[[0, 1, 4]], torch.zeros(3, branch.out_dim))
        assert torch.count_nonzero(branch.embedding.weight[0]) == 0
        output.sum().backward()
        assert branch.embedding.weight.grad is not None
        assert torch.count_nonzero(branch.embedding.weight.grad[0]) == 0
        assert torch.count_nonzero(branch.embedding.weight.grad[1:]) > 0


def test_native500m_id_only_densenet_is_safe_after_initialization() -> None:
    experiment = _native500m_experiment(
        G3Representation(
            history_representation="id_only_densenet",
            history_hidden_dim=64,
            item_id_tying="untied",
        )
    )
    model = experiment.base_model
    encoder = model.item_embedding
    item_ids = torch.tensor([-1, 0, 1, experiment.num_items, experiment.catalog_size])

    output = encoder(item_ids)

    assert torch.equal(output[[0, 1, 4]], torch.zeros(3, experiment.model_dim))
    assert torch.count_nonzero(encoder.item_embedding.weight[0]) == 0
    output.sum().backward()
    assert encoder.item_embedding.weight.grad is not None
    assert torch.count_nonzero(encoder.item_embedding.weight.grad[0]) == 0
    assert torch.count_nonzero(encoder.item_embedding.weight.grad[1]) > 0


def test_native500m_rq3_arms_have_identical_independent_history_inputs() -> None:
    histories = []
    for family in RQ3_CATALOG_REPRESENTATIONS:
        experiment = _native500m_experiment(
            build_rq3_representation(
                family,
                history_hidden_dim=128,
                item_id_tying="untied",
            )
        )
        torch.manual_seed(123)
        model = experiment.base_model
        assert isinstance(model.item_embedding, ItemContentDenseNetEncoder)
        assert model.catalog_item_encoder is not model.item_embedding
        if isinstance(model.catalog_item_encoder, nn.Embedding):
            assert model.item_embedding.item_embedding is not model.catalog_item_encoder
        if isinstance(model.catalog_item_encoder, ItemContentCatalogEncoder):
            assert (
                model.item_embedding.item_embedding
                is not model.catalog_item_encoder.item_embedding
            )
        histories.append(
            {
                name: value.detach().clone()
                for name, value in model.item_embedding.state_dict().items()
            }
        )

    assert all(history.keys() == histories[0].keys() for history in histories)
    assert all(
        torch.equal(history[name], histories[0][name])
        for history in histories[1:]
        for name in history
    )


def test_local_learned_control_is_untied_and_keeps_embedding_optimizer_groups() -> None:
    experiment = _experiment(G3Representation())

    assert isinstance(experiment.item_embedding, nn.Embedding)
    assert isinstance(experiment.catalog_item_encoder, nn.Embedding)
    assert experiment.item_embedding is not experiment.catalog_item_encoder
    model = experiment.base_model
    embedding, deep = experiment.split_parameters(model, experiment.embedding_types)
    embedding_ids = {id(parameter) for parameter in embedding}
    assert id(experiment.item_embedding.weight) in embedding_ids
    assert id(experiment.catalog_item_encoder.weight) in embedding_ids
    assert not ({id(parameter) for parameter in deep} & embedding_ids)


def test_history_content_families_build_the_approved_encoders() -> None:
    content_only = _experiment(G3Representation(history_representation="content"))
    concat = _experiment(
        G3Representation(
            history_representation="id_content",
            history_hidden_dim=128,
            content_gate="frequency",
            gate_hidden_dim=8,
        ),
        feature_data=_feature_data(),
    )

    assert isinstance(content_only.item_embedding, ContentProjection)
    assert isinstance(concat.item_embedding, ItemContentDenseNetEncoder)
    assert isinstance(concat.item_embedding.content_gate, FrequencyContentGate)
    assert concat.item_embedding.out_dim == 64


def test_corrected_frequency_gate_uses_versioned_fp32_p09_semantics() -> None:
    experiment = _experiment(
        G3Representation(
            history_representation="id_content",
            history_hidden_dim=128,
            content_gate="frequency",
            gate_hidden_dim=8,
            frequency_gate_semantics="fp32_p09_v2",
        ),
        feature_data=_feature_data(),
    )

    gate = experiment.item_embedding.content_gate

    assert isinstance(gate, FrequencyContentGate)
    assert gate.fp32_math is True
    assert experiment.representation.to_dict()["frequency_gate_semantics"] == (
        "fp32_p09_v2"
    )
    torch.testing.assert_close(
        gate(torch.tensor([1, 2])),
        torch.full((2, 1), 0.9),
    )


def test_zero_id_diagnostic_preserves_common_initialization_and_removes_id_signal() -> (
    None
):
    concat = _experiment(
        G3Representation(history_representation="id_content", history_hidden_dim=32)
    )
    zero_id = _experiment(
        G3Representation(
            history_representation="id_content_zero_id",
            history_hidden_dim=32,
        )
    )

    torch.manual_seed(123)
    concat_model = concat.base_model
    torch.manual_seed(123)
    zero_id_model = zero_id.base_model

    assert isinstance(concat_model.item_embedding, ItemContentDenseNetEncoder)
    assert isinstance(zero_id_model.item_embedding, ItemContentDenseNetEncoder)
    concat_state = concat_model.state_dict()
    zero_id_state = zero_id_model.state_dict()
    assert concat_state.keys() == zero_id_state.keys()
    isolated_weights = {
        "tokenizer.tokenizer.item_embedding.item_embedding.weight",
        "item_embedding.item_embedding.weight",
    }
    assert isolated_weights <= concat_state.keys()
    assert all(
        torch.equal(concat_value, zero_id_state[name])
        for name, concat_value in concat_state.items()
        if name not in isolated_weights
    )
    assert all(torch.count_nonzero(concat_state[name]) > 0 for name in isolated_weights)
    assert torch.count_nonzero(zero_id_model.item_embedding.item_embedding.weight) == 0
    assert not zero_id_model.item_embedding.item_embedding.weight.requires_grad

    valid_ids = torch.tensor([1, 2, 5])
    features = zero_id_model.item_embedding.composed_features(valid_ids)
    assert torch.count_nonzero(features[:, :64]) == 0
    assert torch.equal(
        features[:, 64:],
        zero_id_model.item_embedding.content.lookup(valid_ids),
    )

    optimizer_parameters = {
        id(parameter)
        for group in zero_id.create_optimizers().param_groups
        for parameter in group["params"]
    }
    assert id(zero_id_model.item_embedding.item_embedding.weight) not in (
        optimizer_parameters
    )


def test_id_only_padding_row_cannot_affect_valid_packed_sequence_loss_or_gradients() -> (
    None
):
    first_experiment = _experiment(
        G3Representation(
            history_representation="id_only_densenet", history_hidden_dim=32
        )
    )
    second_experiment = _experiment(
        G3Representation(
            history_representation="id_only_densenet", history_hidden_dim=32
        )
    )
    torch.manual_seed(321)
    first = first_experiment.base_model
    torch.manual_seed(321)
    second = second_experiment.base_model
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        second.item_embedding.item_embedding.weight[0].fill_(123.0)

    assert (
        first.item_embedding.item_embedding.padding_idx
        == second.item_embedding.item_embedding.padding_idx
        == 0
    )
    batch = packed_batch([1, 2, 3, 2, 4, 5], [3, 3])
    assert batch["cumulative_lens"].tolist() == [0, 3, 6]
    assert bool((batch["int_columns"]["compact_item_id"].dense() > 0).all())
    first_criterion = first_experiment.create_criterion()
    second_criterion = second_experiment.create_criterion()
    torch.manual_seed(999)
    first_output = first_criterion(batch)
    torch.manual_seed(999)
    second_output = second_criterion(batch)
    assert torch.equal(first_output["loss"], second_output["loss"])

    first_output["loss"].backward()
    second_output["loss"].backward()
    for first_parameter, second_parameter in zip(
        first_criterion.parameters(), second_criterion.parameters(), strict=True
    ):
        assert torch.equal(first_parameter.grad, second_parameter.grad)
    assert torch.count_nonzero(first.item_embedding.item_embedding.weight.grad[0]) == 0
    assert torch.count_nonzero(second.item_embedding.item_embedding.weight.grad[0]) == 0


def test_catalog_variants_are_separate_and_preserve_trainability() -> None:
    frozen = _experiment(G3Representation(catalog_representation="frozen_content"))
    trainable = _experiment(
        G3Representation(catalog_representation="trainable_content")
    )
    combined = _experiment(
        G3Representation(catalog_representation="id_trainable_content")
    )

    assert isinstance(frozen.catalog_item_encoder, PretrainedCatalogEncoder)
    assert list(frozen.catalog_item_encoder.content_parameters()) == []
    assert isinstance(trainable.catalog_item_encoder, PretrainedCatalogEncoder)
    assert isinstance(trainable.catalog_item_encoder.content.embedding, nn.Embedding)
    assert isinstance(combined.catalog_item_encoder, ItemContentCatalogEncoder)
    assert isinstance(combined.catalog_item_encoder.content.embedding, nn.Embedding)
    assert all(
        experiment.item_embedding is not experiment.catalog_item_encoder
        for experiment in (frozen, trainable, combined)
    )


def test_rq3_representation_builder_maps_the_exact_five_output_families() -> None:
    assert RQ3_CATALOG_REPRESENTATIONS == {
        "rq3_output_learned": "learned_id",
        "rq3_output_frozen_content": "frozen_content",
        "rq3_output_trainable_content": "trainable_content",
        "rq3_output_learned_frozen_content": "id_frozen_content",
        "rq3_output_learned_trainable_content": "id_trainable_content",
    }

    representations = {
        family: build_rq3_representation(family, history_hidden_dim=64)
        for family in RQ3_CATALOG_REPRESENTATIONS
    }

    assert {
        representation.history_representation
        for representation in representations.values()
    } == {"id_content"}
    assert {
        representation.history_hidden_dim for representation in representations.values()
    } == {64}
    assert {
        representation.catalog_representation
        for representation in representations.values()
    } == set(RQ3_CATALOG_REPRESENTATIONS.values())


def test_rq3_representation_builder_rejects_unapproved_family() -> None:
    with pytest.raises(ValueError, match="RQ3 output family"):
        build_rq3_representation("rq3_output_typo", history_hidden_dim=64)


def test_metadata_wraps_both_towers_with_independent_tables() -> None:
    experiment = _experiment(
        G3Representation(metadata=("artist", "album"), metadata_dim=16),
        feature_data=_feature_data(),
    )

    history = experiment.item_embedding
    catalog = experiment.catalog_item_encoder
    assert isinstance(history, ItemMetadataDenseNetEncoder)
    assert isinstance(catalog, ItemMetadataDenseNetEncoder)
    assert history is not catalog
    assert history.metadata_branches[0].embedding is not (
        catalog.metadata_branches[0].embedding
    )
    ids = torch.arange(1, 6)
    assert history(ids).shape == (5, 64)
    assert catalog(ids).shape == (5, 64)


def test_approved_id_only_densenet_widths_match_active_capacity() -> None:
    for content_width, control_width in ((64, 128), (128, 255), (256, 510)):
        content = _experiment(
            G3Representation(
                history_representation="id_content",
                history_hidden_dim=content_width,
            )
        )
        control = _experiment(
            G3Representation(
                history_representation="id_only_densenet",
                history_hidden_dim=control_width,
            )
        )
        content_deep = sum(
            parameter.numel()
            for parameter in content.item_embedding.encoder.parameters()
        )
        control_deep = sum(
            parameter.numel()
            for parameter in control.item_embedding.encoder.parameters()
        )
        assert abs(content_deep - control_deep) / content_deep < 0.01


def test_representation_modules_do_not_change_common_transformer_initialization() -> (
    None
):
    learned = _experiment(G3Representation())
    content = _experiment(G3Representation(history_representation="content"))

    torch.manual_seed(123)
    learned_state = {
        name: value.detach().clone()
        for name, value in learned.base_model.sequence_model.state_dict().items()
    }
    torch.manual_seed(123)
    content_state = content.base_model.sequence_model.state_dict()

    assert learned_state.keys() == content_state.keys()
    assert all(
        torch.equal(learned_state[name], content_state[name]) for name in learned_state
    )


@pytest.mark.parametrize(
    "representation",
    (
        G3Representation(),
        G3Representation(history_representation="content"),
        G3Representation(history_representation="id_content", history_hidden_dim=64),
        G3Representation(
            history_representation="id_content_zero_id", history_hidden_dim=64
        ),
        G3Representation(
            history_representation="id_only_densenet", history_hidden_dim=128
        ),
        *(
            G3Representation(
                history_representation="id_content",
                history_hidden_dim=64,
                catalog_representation=catalog,
            )
            for catalog in (
                "frozen_content",
                "trainable_content",
                "id_frozen_content",
                "id_trainable_content",
            )
        ),
        G3Representation(
            history_representation="id_content",
            history_hidden_dim=64,
            content_gate="global",
        ),
        G3Representation(
            history_representation="id_content",
            history_hidden_dim=64,
            content_gate="frequency",
            gate_hidden_dim=4,
        ),
        G3Representation(metadata=("artist",), metadata_dim=16),
        G3Representation(metadata=("album",), metadata_dim=16),
        G3Representation(metadata=("artist", "album"), metadata_dim=16),
        G3Representation(extra_item_id_dim=16),
    ),
)
def test_every_g3_representation_has_a_finite_end_to_end_training_step(
    representation: G3Representation,
) -> None:
    experiment = _experiment(
        representation,
        feature_data=_feature_data() if representation.needs_feature_data else None,
    )
    criterion = experiment.create_criterion()

    output = criterion(packed_batch([1, 2, 3, 2, 4, 5], [3, 3]))
    output["loss"].backward()

    assert torch.isfinite(output["loss"])
    assert any(parameter.grad is not None for parameter in criterion.parameters())


def test_runner_and_callback_share_one_diagnostic_criterion(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(global_config, "_base_path", tmp_path)
    experiment = _experiment(G3Representation())
    criterion = experiment.create_criterion()
    monkeypatch.setattr(
        G4GenerationExperiment,
        "extra_callbacks",
        lambda self, train_days, val_days: ["base"],
    )

    callbacks = experiment.extra_callbacks([], [])

    assert callbacks[0] == "base"
    assert isinstance(callbacks[1], G3DiagnosticsCallback)
    assert callbacks[1].criterion is criterion
    assert callbacks[1].catalog_encoder is experiment.catalog_item_encoder


def test_content_gate_mechanism_diagnostics_are_explicit_and_side_effect_only(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(global_config, "_base_path", tmp_path)
    experiment = _experiment(
        G3Representation(
            history_representation="id_content",
            history_hidden_dim=8,
            content_gate="frequency",
            gate_hidden_dim=4,
        ),
        feature_data=_feature_data(),
        gate_mechanism_diagnostics=True,
    )
    monkeypatch.setattr(
        G4GenerationExperiment,
        "extra_callbacks",
        lambda self, train_days, val_days: [],
    )

    callbacks = experiment.extra_callbacks([], [])

    assert len(callbacks) == 2
    assert isinstance(callbacks[0], G3DiagnosticsCallback)
    assert isinstance(callbacks[1], G3GateDiagnosticsCallback)
    assert callbacks[1].gate is experiment.item_embedding.content_gate


def test_content_gate_does_not_add_mechanism_diagnostics_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(global_config, "_base_path", tmp_path)
    experiment = _experiment(
        G3Representation(
            history_representation="id_content",
            history_hidden_dim=8,
            content_gate="global",
        )
    )
    monkeypatch.setattr(
        G4GenerationExperiment,
        "extra_callbacks",
        lambda self, train_days, val_days: [],
    )

    callbacks = experiment.extra_callbacks([], [])

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], G3DiagnosticsCallback)


def test_rq3_combined_catalog_diagnostics_separate_trainable_components(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(global_config, "_base_path", tmp_path)
    experiment = _experiment(
        G3Representation(catalog_representation="id_trainable_content")
    )
    monkeypatch.setattr(
        G4GenerationExperiment,
        "extra_callbacks",
        lambda self, train_days, val_days: [],
    )

    callback = experiment.extra_callbacks([], [])[0]

    assert set(callback.components) >= {
        "catalog_encoder",
        "catalog_item_table",
        "catalog_content_table",
        "catalog_projection",
    }
    assert callback.components["catalog_item_table"] is (
        experiment.catalog_item_encoder.item_embedding
    )
    assert callback.components["catalog_content_table"] is (
        experiment.catalog_item_encoder.content
    )
    assert callback.components["catalog_projection"] is (
        experiment.catalog_item_encoder.projection
    )
