import pytest
import torch

from dcn.semantic import ResidualCodebooks, SemanticCodes, semantic_id_diagnostics


def test_semantic_id_diagnostics_measure_collisions_load_and_similarity() -> None:
    codes = SemanticCodes(
        item_ids=torch.tensor([1, 2, 3, 4]),
        codes=torch.tensor([[0, 0, 1], [0, 0, 2], [0, 1, 1], [1, 1, 1]]),
        codes_per_level=(4, 3, 3),
    )
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )

    diagnostics = semantic_id_diagnostics(codes, embeddings, num_base_levels=2)

    assert diagnostics.identifier_collision_rate == pytest.approx(0.25)
    assert diagnostics.collided_item_fraction == pytest.approx(0.5)
    assert diagnostics.unique_base_tuples == 3
    assert diagnostics.collision_bucket_size_p50 == pytest.approx(1.0)
    assert diagnostics.collision_bucket_size_p95 == pytest.approx(2.0)
    assert diagnostics.collision_bucket_size_p99 == pytest.approx(2.0)
    assert diagnostics.collision_bucket_size_max == 2
    assert diagnostics.occupied_codes == (2, 2)
    assert diagnostics.dead_code_fraction == pytest.approx((0.5, 1 / 3))
    assert diagnostics.p95_occupied_load == pytest.approx((3.0, 2.0))
    assert diagnostics.p95_to_mean_occupied_load == pytest.approx((1.5, 1.0))
    assert diagnostics.intra_code_cosine_similarity == pytest.approx((1 / 3, 0.0))


def test_semantic_id_diagnostics_reject_misaligned_embeddings() -> None:
    codes = SemanticCodes(
        item_ids=torch.tensor([1, 2]),
        codes=torch.tensor([[0, 1], [1, 1]]),
        codes_per_level=(2, 2),
    )

    with pytest.raises(ValueError, match="one embedding per semantic id"):
        semantic_id_diagnostics(codes, torch.ones(1, 2), num_base_levels=1)


def test_semantic_id_diagnostics_measure_residual_error_by_depth() -> None:
    codes = SemanticCodes(
        item_ids=torch.tensor([1, 2]),
        codes=torch.tensor([[0, 0], [1, 1]]),
        codes_per_level=(2, 2),
    )
    codebooks = ResidualCodebooks(
        torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
    )

    diagnostics = semantic_id_diagnostics(
        codes,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        num_base_levels=2,
        codebooks=codebooks,
    )

    assert diagnostics.reconstruction_mse_by_depth == pytest.approx((0.0, 0.0))
