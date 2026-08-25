import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import rectools
from rectools.models.nn.transformers.ligr import LiGRLayer
from rectools.models.nn.transformers.sasrec import SASRecTransformerLayer


RECTOOLS_VERSION = "0.19.0"
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


def _values(shape: torch.Size, offset: int) -> torch.Tensor:
    count = shape.numel()
    values = torch.arange(offset, offset + count, dtype=torch.float64)
    return (values.remainder(37) - 18).reshape(shape) / 100


def _initialize(module: torch.nn.Module) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(_values(parameter.shape, offset))
            offset += parameter.numel()


def _mapped_state(module: torch.nn.Module, family: str) -> dict[str, Any]:
    state = module.state_dict()
    q_weight, k_weight, v_weight = state["multi_head_attn.in_proj_weight"].chunk(3)
    q_bias, k_bias, v_bias = state["multi_head_attn.in_proj_bias"].chunk(3)
    mapped = {
        "q_proj.weight": q_weight,
        "q_proj.bias": q_bias,
        "k_proj.weight": k_weight,
        "k_proj.bias": k_bias,
        "v_proj.weight": v_weight,
        "v_proj.bias": v_bias,
        "out_proj.weight": state["multi_head_attn.out_proj.weight"],
        "out_proj.bias": state["multi_head_attn.out_proj.bias"],
    }
    if family == "sasrec":
        mapped.update(
            {
                "attention_norm.weight": state["q_layer_norm.weight"],
                "attention_norm.bias": state["q_layer_norm.bias"],
                "ffn_norm.weight": state["ff_layer_norm.weight"],
                "ffn_norm.bias": state["ff_layer_norm.bias"],
                "ffn.linear1.weight": state["feed_forward.ff_linear_1.weight"],
                "ffn.linear1.bias": state["feed_forward.ff_linear_1.bias"],
                "ffn.linear2.weight": state["feed_forward.ff_linear_2.weight"],
                "ffn.linear2.bias": state["feed_forward.ff_linear_2.bias"],
            }
        )
    else:
        mapped.update(
            {
                "attention_norm.weight": state["layer_norm_1.weight"],
                "attention_norm.bias": state["layer_norm_1.bias"],
                "ffn_norm.weight": state["layer_norm_2.weight"],
                "ffn_norm.bias": state["layer_norm_2.bias"],
                "ffn.w1.weight": state["feed_forward.ff_linear_1.weight"],
                "ffn.w2.weight": state["feed_forward.ff_linear_3.weight"],
                "ffn.w3.weight": state["feed_forward.ff_linear_2.weight"],
                "attention_gate.weight": state["gating_linear_1.weight"],
                "attention_gate.bias": state["gating_linear_1.bias"],
                "ffn_gate.weight": state["gating_linear_2.weight"],
                "ffn_gate.bias": state["gating_linear_2.bias"],
            }
        )
    return {name: value.tolist() for name, value in mapped.items()}


def _mapped_gradients(module: torch.nn.Module, family: str) -> dict[str, Any]:
    parameters = dict(module.named_parameters())
    q_grad, k_grad, v_grad = parameters["multi_head_attn.in_proj_weight"].grad.chunk(3)
    qb_grad, kb_grad, vb_grad = parameters["multi_head_attn.in_proj_bias"].grad.chunk(3)
    mapped = {
        "q_proj.weight": q_grad,
        "q_proj.bias": qb_grad,
        "k_proj.weight": k_grad,
        "k_proj.bias": kb_grad,
        "v_proj.weight": v_grad,
        "v_proj.bias": vb_grad,
        "out_proj.weight": parameters["multi_head_attn.out_proj.weight"].grad,
        "out_proj.bias": parameters["multi_head_attn.out_proj.bias"].grad,
    }
    if family == "sasrec":
        names = {
            "attention_norm.weight": "q_layer_norm.weight",
            "attention_norm.bias": "q_layer_norm.bias",
            "ffn_norm.weight": "ff_layer_norm.weight",
            "ffn_norm.bias": "ff_layer_norm.bias",
            "ffn.linear1.weight": "feed_forward.ff_linear_1.weight",
            "ffn.linear1.bias": "feed_forward.ff_linear_1.bias",
            "ffn.linear2.weight": "feed_forward.ff_linear_2.weight",
            "ffn.linear2.bias": "feed_forward.ff_linear_2.bias",
        }
    else:
        names = {
            "attention_norm.weight": "layer_norm_1.weight",
            "attention_norm.bias": "layer_norm_1.bias",
            "ffn_norm.weight": "layer_norm_2.weight",
            "ffn_norm.bias": "layer_norm_2.bias",
            "ffn.w1.weight": "feed_forward.ff_linear_1.weight",
            "ffn.w2.weight": "feed_forward.ff_linear_3.weight",
            "ffn.w3.weight": "feed_forward.ff_linear_2.weight",
            "attention_gate.weight": "gating_linear_1.weight",
            "attention_gate.bias": "gating_linear_1.bias",
            "ffn_gate.weight": "gating_linear_2.weight",
            "ffn_gate.bias": "gating_linear_2.bias",
        }
    mapped.update(
        {local: parameters[reference].grad for local, reference in names.items()}
    )
    return {name: value.tolist() for name, value in mapped.items()}


def _block_fixture(family: str) -> dict[str, Any]:
    if family == "sasrec":
        module = SASRecTransformerLayer(4, 2, 0.0).double().eval()
        intermediate_dim = 4
    else:
        module = (
            LiGRLayer(
                4,
                2,
                0.0,
                ff_factors_multiplier=8,
                bias_in_ff=False,
                ff_activation="swiglu",
            )
            .double()
            .eval()
        )
        intermediate_dim = 32
    _initialize(module)
    state = _mapped_state(module, family)
    inputs = _values(torch.Size((2, 3, 4)), 11).requires_grad_()
    causal_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    output = module(inputs, causal_mask, None)
    output.square().mean().backward()
    return {
        "family": family,
        "config": {
            "dim": 4,
            "nhead": 2,
            "intermediate_dim": intermediate_dim,
            "dropout": 0.0,
        },
        "cumulative_lens": [0, 3, 6],
        "input": inputs.detach().reshape(-1, 4).tolist(),
        "state": state,
        "output": output.detach().reshape(-1, 4).tolist(),
        "input_gradient": inputs.grad.reshape(-1, 4).tolist(),
        "parameter_gradients": _mapped_gradients(module, family),
    }


def _loss_fixture() -> dict[str, Any]:
    from rectools.models.nn.transformers.lightning import TransformerLightningModuleBase
    from rectools.models.nn.transformers.similarity import DistanceSimilarityModule

    logits = _values(torch.Size((2, 2, 5)), 7)
    targets = torch.ones((2, 2), dtype=torch.long)
    weights = torch.ones((2, 2), dtype=torch.float64)

    class _DataPreparator:
        n_negatives = 4

    class _ItemModel:
        n_items = 18

    class _TorchModel:
        item_model = _ItemModel()

    oracle = object.__new__(TransformerLightningModuleBase)
    oracle.data_preparator = _DataPreparator()
    oracle.torch_model = _TorchModel()
    oracle.item_extra_tokens = ["PAD", "MASK"]
    oracle.gbce_t = 0.75
    sampled_input = logits.clone().requires_grad_()
    sampled_logits = sampled_input * 1
    sampled = oracle._calc_sampled_softmax_loss(sampled_logits, targets, weights)
    sampled.backward()
    gbce_logits = logits.clone().requires_grad_()
    transformed = oracle._get_reduced_overconfidence_logits(gbce_logits, 16)
    gbce = oracle._calc_bce_loss(transformed, targets, weights)
    gbce.backward()

    positive_ids = torch.tensor([1, 4, 7, 10])
    negative_ids = torch.tensor(
        [
            [2, 3, 5, 6],
            [1, 6, 8, 9],
            [2, 4, 11, 12],
            [3, 5, 8, 13],
        ]
    )
    candidate_ids = torch.cat([positive_ids.unsqueeze(1), negative_ids], dim=1)

    def fixed_candidate_loss(kind: str) -> dict[str, Any]:
        query = _values(torch.Size((4, 3)), 19).requires_grad_()
        item_table = _values(torch.Size((16, 3)), 31).requires_grad_()
        candidate_logits = DistanceSimilarityModule()(
            query.reshape(2, 2, 3),
            item_table,
            candidate_ids.reshape(2, 2, 5),
        )
        if kind == "sampled_softmax":
            value = oracle._calc_sampled_softmax_loss(
                candidate_logits * 1, targets, weights
            )
        else:
            calibrated = oracle._get_reduced_overconfidence_logits(candidate_logits, 16)
            value = oracle._calc_bce_loss(calibrated, targets, weights)
        value.backward()
        return {
            "loss": value.item(),
            "query_gradient": query.grad.tolist(),
            "item_gradient": item_table.grad.tolist(),
        }

    return {
        "catalog_size": 16,
        "t": 0.75,
        "logits": logits.reshape(-1, 5).tolist(),
        "sampled_softmax": {
            "loss": sampled.item(),
            "gradient": sampled_input.grad.reshape(-1, 5).tolist(),
        },
        "gbce": {
            "transformed_logits": transformed.detach().reshape(-1, 5).tolist(),
            "loss": gbce.item(),
            "gradient": gbce_logits.grad.reshape(-1, 5).tolist(),
        },
        "fixed_candidates": {
            "query": _values(torch.Size((4, 3)), 19).tolist(),
            "item_table": _values(torch.Size((16, 3)), 31).tolist(),
            "positive_ids": positive_ids.tolist(),
            "negative_ids": negative_ids.tolist(),
            "sampled_softmax": fixed_candidate_loss("sampled_softmax"),
            "gbce": fixed_candidate_loss("gbce"),
        },
    }


def _source_provenance() -> dict[str, str]:
    package_root = Path(rectools.__file__).parent
    return {
        relative: hashlib.sha256((package_root / relative).read_bytes()).hexdigest()
        for relative in RECTOOLS_SOURCE_SHA256
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if rectools.__version__ != RECTOOLS_VERSION:
        raise RuntimeError(
            f"expected RecTools {RECTOOLS_VERSION}, got {rectools.__version__}"
        )
    source_sha256 = _source_provenance()
    if source_sha256 != RECTOOLS_SOURCE_SHA256:
        raise RuntimeError(f"unexpected RecTools source digests: {source_sha256}")
    fixture = {
        "provenance": {
            "rectools_version": RECTOOLS_VERSION,
            "source_sha256": source_sha256,
            "torch_version": torch.__version__,
        },
        "blocks": [_block_fixture("sasrec"), _block_fixture("ligr")],
        "losses": _loss_fixture(),
    }
    payload = (json.dumps(fixture, indent=2) + "\n").encode()
    fixture_sha256 = hashlib.sha256(payload).hexdigest()
    if fixture_sha256 != FIXTURE_SHA256:
        raise RuntimeError(f"unexpected fixture digest: {fixture_sha256}")
    args.output.write_bytes(payload)


if __name__ == "__main__":
    main()
