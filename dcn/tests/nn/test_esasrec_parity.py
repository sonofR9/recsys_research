import hashlib
import json
from pathlib import Path

import pytest
import torch

from dcn.nn.esasrec import LiGRBlock, SASRecBlock
from dcn.nn.sampled_softmax import GeneralizedBCELoss, OfflineInBatchSoftmax


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "rectools_019_esasrec.json"
FIXTURE_SHA256 = "a041454f1cd9f473bc4923aa26bd57f1e66dfedee818a40a11ed5a47bc0721ad"
RECTOOLS_SOURCE_SHA256 = {
    "models/nn/transformers/ligr.py": (
        "1970236e381b1361680903e7327cc219a47fec6a1d5a8cb51840bdb5b3fccb60"
    ),
    "models/nn/transformers/sasrec.py": (
        "464d2cb24552eeeb194c76620573e72a48ab90732ae930f10c45ce57dd822c25"
    ),
    "models/nn/transformers/lightning.py": (
        "fa7fa54fd8db2b888e75a32e105c2bb068f8b0046932030129d1bde94d8e1db9"
    ),
}


@pytest.fixture(scope="module")
def reference() -> dict:
    payload = FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FIXTURE_SHA256
    return json.loads(payload)


@pytest.mark.usefixtures("cpu_attention")
@pytest.mark.parametrize(
    ("family", "block_type"), [("sasrec", SASRecBlock), ("ligr", LiGRBlock)]
)
def test_block_matches_rectools_forward_and_gradients(
    reference: dict, family: str, block_type: type[torch.nn.Module]
) -> None:
    fixture = next(item for item in reference["blocks"] if item["family"] == family)
    block = block_type(**fixture["config"]).double().eval()
    block.load_state_dict(
        {
            name: torch.tensor(value, dtype=torch.float64)
            for name, value in fixture["state"].items()
        }
    )
    inputs = torch.tensor(fixture["input"], dtype=torch.float64, requires_grad=True)

    output = block(inputs, torch.tensor(fixture["cumulative_lens"], dtype=torch.int32))
    output.square().mean().backward()

    torch.testing.assert_close(
        output,
        torch.tensor(fixture["output"], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        inputs.grad,
        torch.tensor(fixture["input_gradient"], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    for name, parameter in block.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            torch.tensor(fixture["parameter_gradients"][name], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )


def test_rectools_fixture_has_pinned_provenance(reference: dict) -> None:
    provenance = reference["provenance"]

    assert provenance["rectools_version"] == "0.19.0"
    assert provenance["source_sha256"] == RECTOOLS_SOURCE_SHA256


def test_sampled_softmax_matches_rectools_loss_and_gradient(reference: dict) -> None:
    fixture = reference["losses"]
    logits = torch.tensor(fixture["logits"], dtype=torch.float64, requires_grad=True)
    criterion = OfflineInBatchSoftmax(
        q=torch.full((fixture["catalog_size"],), 1 / fixture["catalog_size"]),
        num_in_batch_negatives=0,
        correction="none",
    )

    loss = criterion.loss_from_logits(logits)
    loss.backward()

    assert loss.item() == pytest.approx(fixture["sampled_softmax"]["loss"], abs=1e-12)
    torch.testing.assert_close(
        logits.grad,
        torch.tensor(fixture["sampled_softmax"]["gradient"], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_gbce_matches_rectools_transform_loss_and_gradient(reference: dict) -> None:
    fixture = reference["losses"]
    logits = torch.tensor(fixture["logits"], dtype=torch.float64, requires_grad=True)
    criterion = GeneralizedBCELoss(
        catalog_size=fixture["catalog_size"],
        t=fixture["t"],
    )

    transformed = criterion.transform_logits(logits)
    loss = criterion.loss_from_logits(logits)
    loss.backward()

    torch.testing.assert_close(
        transformed,
        torch.tensor(fixture["gbce"]["transformed_logits"], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert loss.item() == pytest.approx(fixture["gbce"]["loss"], abs=1e-12)
    torch.testing.assert_close(
        logits.grad,
        torch.tensor(fixture["gbce"]["gradient"], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("loss_kind", ["sampled_softmax", "gbce"])
def test_fixed_negative_forward_matches_rectools_loss_and_gradients(
    reference: dict, loss_kind: str
) -> None:
    fixture = reference["losses"]
    candidates = fixture["fixed_candidates"]
    query = torch.tensor(candidates["query"], dtype=torch.float64, requires_grad=True)
    item_table = torch.tensor(
        candidates["item_table"], dtype=torch.float64, requires_grad=True
    )
    positive_ids = torch.tensor(candidates["positive_ids"])
    negative_ids = torch.tensor(candidates["negative_ids"])
    if loss_kind == "sampled_softmax":
        criterion = OfflineInBatchSoftmax(
            q=torch.full((fixture["catalog_size"],), 1 / fixture["catalog_size"]),
            num_in_batch_negatives=0,
            correction="none",
            mask_false_negatives=False,
        )
    else:
        criterion = GeneralizedBCELoss(
            catalog_size=fixture["catalog_size"],
            t=fixture["t"],
            mask_false_negatives=False,
        )

    loss = criterion(
        query,
        item_table[positive_ids],
        positive_ids,
        torch.tensor([2, 2]),
        negatives=(item_table[negative_ids], negative_ids),
    )
    loss.backward()
    expected = candidates[loss_kind]

    assert loss.item() == pytest.approx(expected["loss"], abs=1e-12)
    torch.testing.assert_close(
        query.grad,
        torch.tensor(expected["query_gradient"], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        item_table.grad,
        torch.tensor(expected["item_gradient"], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("width", [512, 1024, 1536])
def test_approved_ligr_widths_are_32_aligned(width: int) -> None:
    assert (
        LiGRBlock(dim=256, nhead=4, intermediate_dim=width, dropout=0.2).ffn.out_dim
        == 256
    )


@pytest.mark.parametrize("width", [16, 48, 171])
def test_ligr_rejects_widths_not_divisible_by_32(width: int) -> None:
    with pytest.raises(ValueError, match="divisible by 32"):
        LiGRBlock(dim=256, nhead=4, intermediate_dim=width, dropout=0.2)
