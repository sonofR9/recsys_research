"""Time the FFN projections at aligned and misaligned intermediate widths.

The G1 architecture control derives its SwiGLU intermediate width from the
Llama 2/3 ratio without the round-up-to-a-multiple step that convention also
carries, so the shipped width is 171. This measures what that costs in the
projections alone; `notes/ffn_width_alignment.md` records what it costs
end to end.
"""

from __future__ import annotations

import argparse

import torch

from dcn.nn.ffn import RegularMLP, SwiGLU


WIDTHS = (128, 160, 171, 176, 192, 256, 342, 352, 684, 704)


def _median_milliseconds(
    module: torch.nn.Module, tokens: torch.Tensor, repeats: int
) -> float:
    for _ in range(5):
        module(tokens)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module(tokens)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=128 * 1024)
    parser.add_argument("--repeats", type=int, default=50)
    arguments = parser.parse_args()

    tokens = torch.randn(
        arguments.tokens,
        arguments.dim,
        device=arguments.device,
        dtype=torch.bfloat16,
    )
    print(f"{'width':>6} {'x16':>4} {'x32':>4} {'GELU ms':>9} {'SwiGLU ms':>10}")
    for width in WIDTHS:
        timings = []
        for family in (RegularMLP, SwiGLU):
            module = family(arguments.dim, width).to(
                arguments.device, torch.bfloat16
            )
            with torch.no_grad():
                timings.append(
                    _median_milliseconds(module, tokens, arguments.repeats)
                )
        print(
            f"{width:>6} {'yes' if width % 16 == 0 else 'no':>4} "
            f"{'yes' if width % 32 == 0 else 'no':>4} "
            f"{timings[0]:>9.3f} {timings[1]:>10.3f}"
        )


if __name__ == "__main__":
    main()
