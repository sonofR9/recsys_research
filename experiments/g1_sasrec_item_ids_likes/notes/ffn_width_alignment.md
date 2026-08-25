# FFN intermediate widths are not multiples of 16

Recorded 2026-08-18.

## What is misaligned

The architecture control's SwiGLU intermediate width is 171 —
`round(2/3 × 4 × 64)`, the Llama ratio without the round-up-to-a-multiple step
that the same convention carries. Tensor-core GEMMs want a multiple of 16, and
32 is the safer default.

Widths in active use that are not multiples of 16:

| width | where |
| ---: | --- |
| 171 | the shared FFN control for both RQ4 families, hardcoded in [`configs/variant.py`](../configs/variant.py) and [`manifest.sh`](../launchers/architecture/manifest.sh) |
| 43, 86, 342, 684 | the RQ8 model-dimension axis, ratio-scaled from 171 at dims 16, 32, 128, 256 |

The ratio scaling itself rounds with a bare `round()`, in
[`generation.py`](../../../dcn/config/generation.py) `_model_at_width` and in
[`configs/transfer_variant.py`](../configs/transfer_variant.py), so any width
derived from a non-multiple base inherits the misalignment.

## What it costs

[`checks/ffn_width_alignment.py`](../checks/ffn_width_alignment.py), bf16,
128k tokens, model dim 64, one idle A100:

```
 width  x16  x32   GELU ms  SwiGLU ms
   128  yes  yes     0.191      0.307
   160  yes  yes     0.246      0.413
   171   no   no     0.389      0.635
   176  yes   no     0.259      0.437
   192  yes  yes     0.265      0.447
   256  yes  yes     0.302      0.518
   342   no   no     0.435      0.730
   352  yes  yes     0.380      0.669
   684   no   no     0.782      1.351
   704  yes  yes     0.675      1.213
```

In the projections alone the penalty is real and large: 171 is 50% slower than
the *wider* 176 and 29% slower than 256, and 342 and 684 each lose 12–13% to
their next multiple of 32.

It does not survive to the training step. Median `train_epoch_time` across
every native-500M FFN confirmation:

| width | family | median s/epoch |
| ---: | --- | ---: |
| 32 | SwiGLU | 11.68, 11.73 |
| 96 | SwiGLU | 11.80 |
| 128 | GELU / SwiGLU | 11.73 / 11.84, 11.86 |
| 171 | GELU | 11.74, 11.78, 11.81 |
| 256 | GELU | 11.79 |
| 224 | SwiGLU | 11.87 |
| 384 | GELU | 11.86 |

A 12× width range moves the epoch by under 2%, and misaligned 171 sits between
aligned 128 and 256. At model dim 64 with a 10M-parameter item table and
sampled-softmax scoring, the FFN is not where the step goes.

## Decision

Not fixed. Changing 171 changes the architecture control, and every axis in
RQ5–RQ11 is measured against that control, so realigning it invalidates the
whole surface for a throughput gain that does not exist end to end at this
size. The RQ8 dimension axis has the same problem for the same reason.

Applies going forward: new width work uses multiples of 32. The SwiGLU width
probe already does — 128, not 171 — and it is the best FFN point measured on
500M, so the aligned grid is where the open questions are anyway. If the ratio
rounding is ever changed to round up to a multiple of 32, it has to land
together with a re-baselined control, not on its own.
