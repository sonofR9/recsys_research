import os

import pytest
import torch

from experiments.g2_esasrec.configs.local import build_component
from utils.global_config import config


pytestmark = [
    pytest.mark.slow_gpu,
    pytest.mark.skipif(
        os.environ.get("RUN_SLOW_GPU_TESTS") != "1",
        reason="set RUN_SLOW_GPU_TESTS=1 to run real-GPU integration tests",
    ),
]


@pytest.mark.parametrize("method", ["standard_sampled_softmax", "ligr_sampled_softmax"])
def test_real_a100_full_esasrec_recipe_matches_separate_packing(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if "A100" not in torch.cuda.get_device_name():
        pytest.skip("requires an NVIDIA A100")

    monkeypatch.setattr(config, "_cpu_attention", False)
    torch.manual_seed(73)
    experiment = build_component(method, ligr_multiplier=4)
    encoder = experiment.create_sequence_model(tokens_per_event=1).cuda().eval()
    encoder.init_weights(experiment.initializer_std)
    cumulative_lens = torch.tensor([0, 3, 8, 12], device="cuda")
    packed_input = torch.randn(
        12,
        experiment.model_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
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
        packed_output.float().square().mean().backward()
        separate_output.float().square().mean().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(packed_output, separate_output, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        packed_input.grad,
        separate_input.grad,
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.isfinite(packed_input.grad).all()
