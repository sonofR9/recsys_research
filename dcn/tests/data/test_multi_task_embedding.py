import pytest
import torch

from dcn.data.features import FeatureValues
from dcn.nn.multi_task_embedding import (
    MultiTaskEmbeddingLayer,
    OutputDims,
)
from dcn.tests.helpers import scalar_feature


@pytest.fixture
def feature_configs() -> dict[str, int]:
    return {
        "item_id": 2,
        "user_id": 3,
        "artist_id": 1,
    }


@pytest.fixture
def split_ratios() -> dict[str, float]:
    return {
        "shared": 0.5,
        "like": 0.25,
        "listen": 0.25,
    }


class TestOutputDims:
    def test_getattr_existing_key(self) -> None:
        dims = OutputDims(dims={"shared": 64, "like": 32})
        assert dims.shared == 64
        assert dims.like == 32

    def test_getattr_missing_key_raises(self) -> None:
        dims = OutputDims(dims={"shared": 64})
        with pytest.raises(AttributeError, match="No dimension named 'missing'"):
            _ = dims.missing


class TestMultiTaskEmbeddingLayer:
    def test_output_shape_sum_mode(
        self,
        feature_configs: dict[str, int],
        split_ratios: dict[str, float],
    ) -> None:
        embedding = MultiTaskEmbeddingLayer(
            feature_configs=feature_configs,
            num_embeddings=1024,
            embedding_dim=64,
            split_ratios=split_ratios,
            mode="sum",
        )

        batch_size = 8
        features: dict[str, FeatureValues] = {
            "item_id": scalar_feature(torch.randint(0, 1_000_000, (batch_size,))),
            "user_id": scalar_feature(torch.randint(0, 1_000_000, (batch_size,))),
            "artist_id": scalar_feature(torch.randint(0, 1_000_000, (batch_size,))),
        }

        output = embedding(features)

        assert "shared" in output
        assert "like" in output
        assert "listen" in output

        num_features = len(feature_configs)
        assert output["shared"].shape == (batch_size, 32 * num_features)
        assert output["like"].shape == (batch_size, 16 * num_features)
        assert output["listen"].shape == (batch_size, 16 * num_features)

    def test_output_shape_concat_mode(
        self,
        feature_configs: dict[str, int],
        split_ratios: dict[str, float],
    ) -> None:
        embedding = MultiTaskEmbeddingLayer(
            feature_configs=feature_configs,
            num_embeddings=1024,
            embedding_dim=64,
            split_ratios=split_ratios,
            mode="concat",
        )

        batch_size = 8
        features: dict[str, FeatureValues] = {
            "item_id": scalar_feature(torch.randint(0, 1_000_000, (batch_size,))),
            "user_id": scalar_feature(torch.randint(0, 1_000_000, (batch_size,))),
            "artist_id": scalar_feature(torch.randint(0, 1_000_000, (batch_size,))),
        }

        output = embedding(features)

        total_hashes = sum(feature_configs.values())
        assert output["shared"].shape == (batch_size, 32 * total_hashes)
        assert output["like"].shape == (batch_size, 16 * total_hashes)
        assert output["listen"].shape == (batch_size, 16 * total_hashes)

    def test_output_dims_calculation_sum_mode(
        self,
        feature_configs: dict[str, int],
        split_ratios: dict[str, float],
    ) -> None:
        embedding = MultiTaskEmbeddingLayer(
            feature_configs=feature_configs,
            num_embeddings=1024,
            embedding_dim=64,
            split_ratios=split_ratios,
            mode="sum",
        )

        dims = embedding.out_dim
        num_features = len(feature_configs)

        assert dims.shared == 32 * num_features
        assert dims.like == 16 * num_features
        assert dims.listen == 16 * num_features

    def test_output_dims_calculation_concat_mode(
        self,
        feature_configs: dict[str, int],
        split_ratios: dict[str, float],
    ) -> None:
        embedding = MultiTaskEmbeddingLayer(
            feature_configs=feature_configs,
            num_embeddings=1024,
            embedding_dim=64,
            split_ratios=split_ratios,
            mode="concat",
        )

        dims = embedding.out_dim
        total_hashes = sum(feature_configs.values())

        assert dims.shared == 32 * total_hashes
        assert dims.like == 16 * total_hashes
        assert dims.listen == 16 * total_hashes

    def test_split_ratios_must_sum_to_one(
        self, feature_configs: dict[str, int]
    ) -> None:
        with pytest.raises(AssertionError, match="split_ratios must sum to 1.0"):
            MultiTaskEmbeddingLayer(
                feature_configs=feature_configs,
                num_embeddings=1024,
                embedding_dim=64,
                split_ratios={"shared": 0.5, "like": 0.3},
                mode="sum",
            )

    def test_split_ratios_must_be_non_negative(
        self, feature_configs: dict[str, int]
    ) -> None:
        with pytest.raises(AssertionError, match="split_ratios must be non-negative"):
            MultiTaskEmbeddingLayer(
                feature_configs=feature_configs,
                num_embeddings=1024,
                embedding_dim=64,
                split_ratios={"shared": 1.5, "like": -0.5},
                mode="sum",
            )

    def test_num_embeddings_must_be_power_of_two(
        self,
        feature_configs: dict[str, int],
        split_ratios: dict[str, float],
    ) -> None:
        with pytest.raises(AssertionError, match="num_embeddings must be a power of 2"):
            MultiTaskEmbeddingLayer(
                feature_configs=feature_configs,
                num_embeddings=1000,
                embedding_dim=64,
                split_ratios=split_ratios,
                mode="sum",
            )

    def test_hash_values_in_valid_range(
        self,
        feature_configs: dict[str, int],
        split_ratios: dict[str, float],
    ) -> None:
        num_embeddings = 1024
        embedding = MultiTaskEmbeddingLayer(
            feature_configs=feature_configs,
            num_embeddings=num_embeddings,
            embedding_dim=64,
            split_ratios=split_ratios,
            mode="sum",
        )

        test_values = torch.randint(0, 10**9, (100,))
        hash_a = embedding.hash_a[:1]

        hashed = embedding._multiply_shift_hash(test_values, hash_a)

        assert hashed.min() >= 0
        assert hashed.max() < num_embeddings

    def test_split_dimensions_sum_to_embedding_dim(
        self,
        feature_configs: dict[str, int],
        split_ratios: dict[str, float],
    ) -> None:
        embedding_dim = 64
        embedding = MultiTaskEmbeddingLayer(
            feature_configs=feature_configs,
            num_embeddings=1024,
            embedding_dim=embedding_dim,
            split_ratios=split_ratios,
            mode="sum",
        )

        total_split_dim = sum(embedding.split_dims.values())
        assert total_split_dim == embedding_dim

    def test_deterministic_output_for_same_input(
        self,
        feature_configs: dict[str, int],
        split_ratios: dict[str, float],
    ) -> None:
        embedding = MultiTaskEmbeddingLayer(
            feature_configs=feature_configs,
            num_embeddings=1024,
            embedding_dim=64,
            split_ratios=split_ratios,
            mode="sum",
        )

        features: dict[str, FeatureValues] = {
            "item_id": scalar_feature(torch.tensor([123, 456])),
            "user_id": scalar_feature(torch.tensor([789, 101])),
            "artist_id": scalar_feature(torch.tensor([112, 131])),
        }

        output1 = embedding(features)
        output2 = embedding(features)

        for key in output1:
            assert torch.allclose(output1[key], output2[key])
