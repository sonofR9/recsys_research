from pathlib import Path

import polars as pl
import pytest
import torch

from dcn.data.features import FeatureValues
from dcn.nn.precomputed_embeddings import PrecomputedEmbeddingLookup
from dcn.tests.helpers import scalar_feature


def test_lookup_returns_correct_rows_for_compact_ids() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    lookup = PrecomputedEmbeddingLookup(
        embeddings, learnable_default=False, strict=False
    )

    output = lookup(scalar_feature(torch.tensor([1, 3, 2, 0])))

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(output, expected)
    assert lookup.embedding_dim == 3
    assert lookup.num_known_ids == 3


def test_out_of_range_ids_route_to_default() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    lookup = PrecomputedEmbeddingLookup(
        embeddings, learnable_default=False, strict=False
    )
    output = lookup(scalar_feature(torch.tensor([1, 2, 99, -1, 0])))
    assert torch.equal(output[2], torch.zeros(2))
    assert torch.equal(output[3], torch.zeros(2))
    assert torch.equal(output[4], torch.zeros(2))


def test_strict_mode_raises_on_out_of_range() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    lookup = PrecomputedEmbeddingLookup(
        embeddings, learnable_default=False, strict=True
    )
    with pytest.raises(AssertionError, match="out-of-range"):
        lookup(scalar_feature(torch.tensor([1, 99])))


def test_learnable_default_is_unit_norm_parameter() -> None:
    embeddings = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    lookup = PrecomputedEmbeddingLookup(
        embeddings, learnable_default=True, strict=False
    )
    assert isinstance(lookup.default, torch.nn.Parameter)
    assert torch.allclose(lookup.default.norm(), torch.tensor(1.0), atol=1e-5)
    output = lookup(scalar_feature(torch.tensor([0])))
    assert torch.allclose(output[0], lookup.default)


def test_multivalent_rows_are_sum_pooled() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    lookup = PrecomputedEmbeddingLookup(
        embeddings, learnable_default=False, strict=False
    )
    feature = FeatureValues(
        values=torch.tensor([1, 2, 3]),
        offsets=torch.tensor([0, 2, 3]),
    )
    output = lookup(feature)
    assert torch.equal(output[0], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(output[1], torch.tensor([0.0, 0.0, 1.0]))


def test_from_parquet_reads_compact_id_column(tmp_path: Path) -> None:
    path = tmp_path / "emb.parquet"
    pl.DataFrame(
        {
            "compact_id": [1, 2, 3],
            "normalized_embed": [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
        }
    ).write_parquet(path)

    lookup = PrecomputedEmbeddingLookup.from_parquet(
        path, learnable_default=False, strict=False
    )
    output = lookup(scalar_feature(torch.tensor([2, 1])))
    assert torch.equal(output, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_from_parquet_orders_rows_by_compact_id(tmp_path: Path) -> None:
    path = tmp_path / "shuffled.parquet"
    pl.DataFrame(
        {
            "compact_id": [3, 1, 2],
            "normalized_embed": [
                [3.0, 3.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ],
        }
    ).write_parquet(path)

    lookup = PrecomputedEmbeddingLookup.from_parquet(
        path, learnable_default=False, strict=False
    )

    output = lookup(scalar_feature(torch.tensor([1, 2, 3])))
    expected = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert torch.equal(output, expected)
