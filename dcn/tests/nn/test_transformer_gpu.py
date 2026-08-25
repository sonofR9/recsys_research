import os

import pytest
import torch

from dcn.nn.transformer import Rope, TransformerBlock, TransformerEncoder
from dcn.tests.helpers import TinyFFN
from utils.global_config import config


pytestmark = [
    pytest.mark.slow_gpu,
    pytest.mark.skipif(
        os.environ.get("RUN_SLOW_GPU_TESTS") != "1",
        reason="set RUN_SLOW_GPU_TESTS=1 to run real-GPU integration tests",
    ),
]


def test_real_a100_packed_gqa_rope_matches_separate_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if "A100" not in torch.cuda.get_device_name():
        pytest.skip("requires an NVIDIA A100")

    monkeypatch.setattr(config, "_cpu_attention", False)
    torch.manual_seed(31)
    encoder = TransformerEncoder(
        blocks=[
            TransformerBlock(
                dim=64,
                nhead=2,
                num_kv_heads=1,
                ffn_factory=TinyFFN,
                dropout=0.0,
                use_alibi=False,
                rope=Rope(32),
                attention_window=50,
            )
        ],
        input_norm=torch.nn.Identity(),
        max_seqlen=128,
    ).cuda()
    encoder.eval()
    cumulative_lens = torch.tensor([0, 2, 6, 11], device="cuda")
    packed_input = torch.randn(11, 64, device="cuda", requires_grad=True)
    separate_input = packed_input.detach().clone().requires_grad_(True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        packed_output = encoder(packed_input, cumulative_lens)
        separate_output = torch.cat(
            [
                encoder(
                    separate_input[start:end],
                    torch.tensor([0, end - start], device="cuda"),
                )
                for start, end in zip(
                    cumulative_lens[:-1].tolist(), cumulative_lens[1:].tolist()
                )
            ]
        )
        packed_output.square().mean().backward()
        separate_output.square().mean().backward()

    torch.cuda.synchronize()
    assert torch.allclose(packed_output, separate_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        packed_input.grad, separate_input.grad, atol=1e-5, rtol=1e-5
    )
    assert torch.isfinite(packed_input.grad).all()
