import torch
import torch.nn as nn

from dcn.models import CriterionSpec, MultiCriterion, TargetExtractionWrapper
from dcn.tests.helpers import scalar_feature


def _batch(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> dict:
    batch = {
        "predictions": {"y": scalar_feature(pred)},
        "float_columns": {"t": scalar_feature(target)},
        "int_columns": {},
    }
    if mask is not None:
        batch["int_columns"]["m"] = scalar_feature(mask)
    return batch


def test_extraction_wrapper_forwards_squeezed_tensors_to_inner() -> None:
    wrapper = TargetExtractionWrapper(
        nn.MSELoss(), prediction_column="y", target_column="t"
    )

    pred = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([0.0, 0.0])

    out = wrapper(_batch(pred, target))
    assert torch.allclose(out, nn.MSELoss()(pred.squeeze(-1), target))


def test_extraction_wrapper_applies_mask() -> None:
    wrapper = TargetExtractionWrapper(
        nn.MSELoss(), prediction_column="y", target_column="t", mask_column="m"
    )

    pred = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([0.0, 0.0, 0.0])
    mask = torch.tensor([True, False, True])

    out = wrapper(_batch(pred, target, mask))
    expected = nn.MSELoss()(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 0.0]))
    assert torch.allclose(out, expected)


def test_multicriterion_weighted_sum_over_batch() -> None:
    criterion = MultiCriterion(
        [
            CriterionSpec(
                "a",
                TargetExtractionWrapper(
                    nn.MSELoss(), prediction_column="y", target_column="t"
                ),
                0.5,
            ),
            CriterionSpec(
                "b",
                TargetExtractionWrapper(
                    nn.L1Loss(), prediction_column="y", target_column="t"
                ),
                1.5,
            ),
        ]
    )

    pred = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([0.0, 0.0, 0.0])

    out = criterion(_batch(pred, target))

    assert set(out.keys()) == {"loss", "a", "b"}
    mse = nn.MSELoss()(pred.squeeze(-1), target)
    l1 = nn.L1Loss()(pred.squeeze(-1), target)
    assert torch.allclose(out["a"], mse)
    assert torch.allclose(out["b"], l1)
    assert torch.allclose(out["loss"], 0.5 * mse + 1.5 * l1)
