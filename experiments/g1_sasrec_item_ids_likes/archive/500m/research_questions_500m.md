# G1 research questions

> Historical snapshot only. No run in this file records the current
> training-semantics revision 2, so none is current evidence or a current
> research conclusion.

Questions from [../../../list.md](../../../list.md). Results live in
[README.md](../../README.md); this file tracks what each one needs and what it found.

Every historical four-run result is four seeds (0–3) of one unchanged
configuration, not four nearby hyperparameter settings. New tuning uses one
run per configuration and reports `runs=1`. The four existing
`selected_quality` repeats define the shared approximate 500M noise band:
recall@100 **0.14389 ±0.00049** (one sample standard deviation). New treatments
run once; this band is a practical common resolution, not a treatment-specific
confidence interval.

The homework-compatible constant-rate baseline is **0.1271 ±0.0018** and its
cosine-warmup control is **0.1246 ±0.0007**. Final comparisons use full
Yambda-500M and the final-seven-day timestamp holdout. Historical tables use
four fixed seeds; the newer batch tuning uses one run per configuration. Older
accepted 128-dimensional screens are labelled separately and are never used as
the reference for corrected 64-dimensional follow-ups.

| rq | question | answer | best variant | recall@100 | vs baseline | runs |
| --- | --- | --- | --- | --- | --- | --- |
| rq1 | Does µTransfer work? | **yes across model width; no tested method reliably transfers LR across Yambda dataset size** | `mup_dim128_lr5e2` | 0.1303 ±0.0006 | +3.7% vs μP width 32 | 4 |
| rq2 | Best transformer combination for metrics | selected combination plus input RMSNorm | `rqfinal_normalization_input_rms` | **0.14589** | +0.8% vs tuned selected control | 1 |
| rq3 | Best metrics/performance balance | batch 512: 13.2s/epoch and 23.9 GB | `rqfinal_normalization_input_rms` | **0.14589** | +1.6% vs prior throughput baseline | 1 |
| rq4 | Does SwiGLU help? | **yes** after equal-budget width/LR tuning; SwiGLU-192 wins | `rqfinal_neg_random` | **0.14480** | +2.3% vs tuned GELU-128 | 1 |
| rq5 | Which lr scheduler works best? | linear has the highest mean; linear, cosine and polynomial are tied within noise | `lr_linear` | 0.1408 ±0.0006 | **+31.3%** vs constant | 4 |
| rq6 | Does lr warmup help? | **no clear benefit** for constant, cosine or inverse sqrt | `lr_cosine_warmup` | 0.1395 ±0.0007 | −0.4% vs cosine without warmup | 4 |
| rq7 | rope / alibi / position embeddings | learned forward wins the tuned 18-treatment comparison; learned reverse loses 9.6% | `rqfinal_neg_random` | **0.14480** | reference | 1 |
| rq8 | Scaling: dim, layers, seq len, heads, GQA, windows, norm, BOS, ffn ratio | use dim 64, depth 2, seq 128, GQA, window 50, pre-norm plus input RMSNorm, dropout 0.1 | `rqfinal_normalization_input_rms` | **0.14589** | +0.8% vs no input norm | 1 |
| rq9 | Timestamp-delta embeddings | 16 additive log-spaced bins work best; reverse time RoPE adds cost | `time_bins_16` | 0.1322 ±0.0012 | **+6.1%** vs corrected cosine reference | 4 |
| rq10 | Per-layer embeddings (Gemma-style) | **no** — quality is flat while embedding parameters triple | `per_layer_embeddings` | 0.1392 ±0.0005 | −0.2% vs shared embedding | 4 |
| rq11 | Negative sampling and logQ | after family-specific tuning, uniform random wins; no-logQ and both mixed methods lose clearly | `rqfinal_neg_random` | 0.14480 | **+2.9%** vs tuned offline logQ | 1 |

Full per-axis metrics and costs are in [results_500m.md](results_500m.md).

## What already exists

`TransformerConfig` now reaches rope, ALiBi, the learned forward and
from-the-end tables (rq7), the ffn kind (rq4) and the normalization kind and
place (rq8). All are screened in [results_500m.md](results_500m.md).

`LrSchedule` scales each parameter group from its own starting rate through
warmup and constant, linear, cosine, inverse-square-root, step, exponential,
polynomial or warmup-stable-decay shapes.

The BOS token is a tokenizer decorator with `is_target=False`, so N items yield
N pairs. It changes recall by +0.5% at a constant rate and −0.4% under cosine,
both inside the four-run resolution; BOS does not help. In this causal
next-item objective, a CLS position has no supervised target and would test a
different question.

## Completed extensions

- rq1 follows the [official Microsoft μP implementation](https://github.com/microsoft/mup):
  base/delta shapes are registered before initialization, matrix weights use
  μP initialization, query/readout weights use zero initialization, the readout
  is `MuReadout`, optimization uses `MuAdam`, and attention uses width-aware
  scaling. The 64-dimensional item table stays fixed while learned input and
  readout projections connect it to transformer widths 32 and 128.
- rq5 covers eight schedule shapes and rq6 isolates warmup within shape.
- rq8 includes local windows, batch/RMS/LayerNorm, residual/input/final norm
  placement, and corrected BOS evaluation.
- rq9 covers plain/log/binned delta addition, dense concatenation, timestamp
  RoPE in both directions, and combinations.
- rq10 injects a separate item table before every transformer layer.
- rq11 compares independently tuned online/offline logQ, uniform catalog
  sampling, mixed negatives, and uncorrected in-batch negatives under one
  unchanged architecture and data protocol.
- Embedding and deep learning rates are swept separately; embedding LR 0.02 is
  crossed with the best compatible timestamp and position arms.

## μP data-scale control

μP transfers the stable LR across transformer width; it does not promise
transfer across dataset size. The post-initialization-fix width-32 runs confirm
that distinction. These are one-run tuning results, not seed repeats.

| dataset | deep LR | recall@100 | ndcg@100 | change vs best on dataset |
| --- | --- | --- | --- | --- |
| Yambda-50M | 0.1 | 0.07542 | 0.02787 | best |
| Yambda-50M | 0.05 | 0.07021 | 0.02641 | −6.9% |
| Yambda-500M | 0.05 | 0.12567 | 0.04807 | best |
| Yambda-500M | 0.1 | 0.11803 | 0.04490 | −6.1% |

Consequently, 50M screens only shortlist LR and batch-size candidates. Final
tuning compares that shortlist on 500M before transferring the selected
hyperparameters across width.

The same rule is visible in batch tuning: batch-512 LR 0.008/0.012 wins the
50M screen at 0.06941 recall, but LR 0.004/0.012 wins the 500M confirmation at
0.14358. The proxy narrows the search; it does not choose the final setting.

## Hyperparameter transfer across model and data size

Two transfer axes were tested: μP for model width, and two recent methods for
the data/token-horizon axis.

- Model width uses [Tensor Programs V / μP](https://arxiv.org/abs/2203.03466).
  It works here: with the fixed 64-dimensional item table and learned
  projections, deep LR 0.05 is stable at widths 32 and 128, and width 128
  improves recall@100 from 0.1256 to 0.1303.
- Token horizon first used the law from [Scaling Optimal Learning Rates Across
  Token Horizons](https://arxiv.org/abs/2409.19913): fit the terminal optimum
  at several supervised-target horizons, then `lr*(D)=A D^-beta`. Joint
  embedding/deep-LR response surfaces were non-quadratic. A reduced fixed-ratio
  diagnostic produced validation-loss response R² values 0.49–0.91 rather than
  the paper's approximately 0.995 gate; its horizon-law R² was 0.64. The
  extrapolated 500M rates (embedding 0.00149, deep 0.00224) are therefore
  rejected and were not spent on a full run.
- The fallback was [Power Scheduler](https://arxiv.org/abs/2408.13359), using
  exact event tokens seen and exponent −0.51. Proxy tuning selected embedding
  LR cap 0.016, deep LR cap 0.006, and a 4M-token transition. It improved the
  best equal-horizon conventional 50M point by 7.6% (0.09771 versus 0.09080),
  but its single held-out 500M run reached only 0.13707 recall@100 and 0.05181
  NDCG@100, **4.54% below** the directly tuned 0.14358 control.

Conclusion: μP is validated for model-width transfer. The terminal-LR law
failed its proxy fit-quality gate, while Power Scheduler failed its held-out
500M gate. Moreover, 50M changes both the training horizon and the user/item
distribution relative to 500M. Subsequent RQ tuning uses 50M to choose a small
shortlist and transfers each selected treatment to one 500M run, with the
shared empirical noise band above. It does not claim that the exact 50M LR
optimum is invariant.

| transfer test | proxy evidence | 500M recall@100 | decision |
| --- | --- | --- | --- |
| μP width 32 → 128, deep LR 0.05 | stable at both widths | 0.1303 ±0.0006 | validated for width |
| terminal-LR horizon law | response R² 0.49–0.91; law R² 0.64 | rejected at proxy gate | reject extrapolation |
| Power Scheduler, cap 0.016/0.006, transition 4M | +7.6% on 50M | 0.13707 | reject for data-size transfer |

## Implementation notes

- Block, input and final normalization are separate knobs and tables.
- Post-norm changes the residual's dtype under autocast, so the `norm_post*`
  cost columns carry an artifact and should not be read as timings.
- `build_transformer_decoder` honours only part of `TransformerConfig`; the
  variants above all go through `build_causal_transformer`.

## rq8 — why scaling mostly saturates

The corrected screen selects width 64 and two layers. Sequence length 128 is
within 0.4% of length 100 while reducing epoch time from 21.5s to 17.9s. GQA
(2 query heads, 1 KV head) is tied on quality and 7% faster. Window 50 gains
1.2% while cutting epoch time about 10%, so all three are in the future
baseline. FlashAttention consumes ragged sequences, so 128's benefit is
measured throughput, not a claim about Tensor-Core alignment.

Dropout 0.0 loses 3.7%, so keep 0.1. The tuned selected-architecture follow-up
confirms post-norm loses 4.7% and selects input RMSNorm at +0.8%, so keep
pre-norm and add the input RMSNorm. The older 128-dimensional reference-family
screens below are historical context; their exact baseline values remain in
their table rows and are not the final selection.

**Depth (reference-family baseline: depth=2)**

| variant | reference configuration | runs | recall@100 | ndcg@100 |
| --- | --- | --- | --- | --- |
| depth 2 | depth=2 | 4 | 0.1395 ±0.0007 | 0.0533 ±0.0001 |
| depth 4 | — | 4 | +0% (0.1398 ±0.0002) | +0% (0.0535 ±0.0001) |
| depth 1 | — | 4 | −1% (0.1375 ±0.0009) | −2% (0.0525 ±0.0003) |

**Number of attention heads (reference-family baseline: heads=4, KV heads=2)**

| variant | reference configuration | runs | recall@100 | ndcg@100 |
| --- | --- | --- | --- | --- |
| heads 4 | heads=4; KV heads=2 | 4 | 0.1395 ±0.0007 | 0.0533 ±0.0001 |
| heads 8 | — | 4 | +0% (0.1399 ±0.0004) | +1% (0.0536 ±0.0002) |
| heads 2 | — | 4 | −0% (0.1390 ±0.0003) | −1% (0.0530 ±0.0002) |

**FFN ratio (reference-family baseline: width=256, ratio=2x)**

| variant | reference configuration | runs | recall@100 | ndcg@100 |
| --- | --- | --- | --- | --- |
| FFN 256 | ffn_dim=256; ratio=2x | 4 | 0.1395 ±0.0007 | 0.0533 ±0.0001 |
| FFN 512 | — | 4 | +0% (0.1399 ±0.0010) | +0% (0.0535 ±0.0004) |
| FFN 128 | — | 4 | +0% (0.1398 ±0.0004) | +0% (0.0534 ±0.0001) |

**Block normalization kind (reference-family baseline: RMSNorm)**

| variant | reference configuration | runs | recall@100 | ndcg@100 |
| --- | --- | --- | --- | --- |
| RMSNorm | block norm=RMSNorm | 4 | 0.1395 ±0.0007 | 0.0533 ±0.0001 |
| BatchNorm | — | 4 | +1% (0.1404 ±0.0004) | +1% (0.0537 ±0.0003) |
| LayerNorm | — | 4 | −0% (0.1391 ±0.0003) | −0% (0.0532 ±0.0002) |

**Input/final normalization (reference-family baseline: both LayerNorm)**

| variant | reference configuration | runs | recall@100 | ndcg@100 |
| --- | --- | --- | --- | --- |
| both | input LayerNorm; final LayerNorm | 4 | 0.1395 ±0.0007 | 0.0533 ±0.0001 |
| no input norm | — | 4 | +2% (0.1418 ±0.0007) | +1% (0.0540 ±0.0003) |
| no final norm | — | 4 | +1% (0.1415 ±0.0010) | +1% (0.0539 ±0.0003) |
| all RMSNorm | — | 4 | +0% (0.1398 ±0.0002) | +0% (0.0534 ±0.0001) |

**BOS token (reference-family baseline: disabled)**

| variant | reference configuration | runs | recall@100 | ndcg@100 |
| --- | --- | --- | --- | --- |
| no BOS | BOS=false | 4 | 0.1395 ±0.0007 | 0.0533 ±0.0001 |
| BOS | — | 4 | −0% (0.1390 ±0.0008) | −1% (0.0530 ±0.0003) |

## rq5, rq6 — what the schedule rows do and do not say

Linear has the highest mean at 0.1408; cosine, polynomial and cosine-warmup are
tied with it inside noise. They improve roughly 30% over constant. WSD, step
and inverse sqrt are lower, but all decay schedules beat constant.

Warmup has no resolved benefit. It changes constant by +0.25%, cosine by
−0.43% and inverse sqrt by −1.34%. The schedule shape matters; the tested 5%
warmup does not.

**Schedule shape (reference-family baseline: constant, warmup=0%, LR=0.001)**

| variant | reference configuration | runs | recall@100 | ndcg@100 |
| --- | --- | --- | --- | --- |
| constant | constant; warmup=0%; LR=0.001 | 4 | 0.1072 ±0.0009 | 0.0405 ±0.0004 |
| linear | — | 4 | +31% (0.1408 ±0.0006) | +33% (0.0538 ±0.0002) |
| cosine | — | 4 | +31% (0.1401 ±0.0006) | +32% (0.0535 ±0.0001) |
| polynomial | — | 4 | +30% (0.1397 ±0.0004) | +32% (0.0534 ±0.0002) |
| exponential | — | 4 | +29% (0.1381 ±0.0004) | +30% (0.0527 ±0.0001) |
| WSD | — | 4 | +27% (0.1362 ±0.0003) | +29% (0.0522 ±0.0002) |
| step | — | 4 | +26% (0.1353 ±0.0005) | +28% (0.0519 ±0.0001) |
| inverse sqrt | — | 4 | +20% (0.1291 ±0.0015) | +21% (0.0489 ±0.0004) |

**Warmup within schedule shape**

| shape | reference configuration | no warmup recall@100 | warmup=5% recall@100 | change |
| --- | --- | --- | --- | --- |
| constant | LR=0.001; warmup=0% | 0.1072 ±0.0009 | 0.1075 ±0.0014 | +0.25% |
| cosine | LR=0.001; warmup=0% | 0.1401 ±0.0006 | 0.1395 ±0.0007 | −0.43% |
| inverse sqrt | LR=0.001; warmup=0%; timescale=5% | 0.1291 ±0.0015 | 0.1273 ±0.0011 | −1.34% |

## rq7 — what the position result means

The corrected targeted reruns favor learned forward positions. Learned reverse
positions lose 8.2%; pure forward and reverse RoPE lose 2.7% and 3.1%. This
contradicts the initial expectation that alignment from the sequence end would
help, but the result is consistent across four full-500M seeds.

## Additional measured analyses

Embedding LR 0.001 wins over the smaller 0.0001, 0.0002, and 0.0005 rates;
deep LR 0.003 wins over 0.001, 0.002, and 0.005. Additive 16-bin time deltas
reach 0.1322; 8, 32, and 64 bins are slightly lower, and adding reverse time
RoPE reaches 0.1311 with extra cost. Per-layer item tables add no quality and
triple embedding parameters.

**Per-layer item embeddings (reference-family baseline: one shared table)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | embedding parameters | epoch time |
| --- | --- | --- | --- | --- | --- | --- |
| shared | one shared item table | 4 | 0.1395 ±0.0007 | 0.0533 ±0.0001 | 20.1M | 31.3s |
| per-layer | — | 4 | −0% (0.1392 ±0.0005) | −1% (0.0530 ±0.0004) | 60.4M | 32.9s |

RQ11's earlier controlled fixed-rate axis gives offline logQ 0.1271, online
logQ 0.1231, uniform random 0.1332, random sampled from the offline proposal
with exact logQ 0.1266, and uncorrected in-batch 0.0643. Random wins; proposal
correction recovers the offline baseline; omitting correction collapses.

## Future baseline

Use `future_baseline` as the current runnable throughput/quality baseline; its
measured run is `rqfinal_normalization_input_rms`. It keeps the selected-quality
architecture, adds the independently validated input RMSNorm, and uses the
tuned dense random-negative objective. Use
`selected_balanced` only when its lower 13.3 GB memory footprint matters more
than throughput and maximum quality.

| parameter | selected value |
| --- | --- |
| data/evaluation | full Yambda-500M likes; core item interactions ≥5; final 7 days held out; full mapped train catalog; seen items retained |
| run policy | one run per configuration; no seed repeats unless explicitly requested |
| model dimension / item-table dimension | 64 / 64 |
| layers / sequence length / attention window | 2 / 128 / 50 |
| query heads / KV heads | 2 / 1 (GQA) |
| FFN | SwiGLU, intermediate width 192 |
| normalization | pre-LayerNorm; input RMSNorm; final LayerNorm |
| positions | learned forward; no RoPE; no ALiBi |
| time feature | 16 log-spaced timestamp-delta bins, added |
| dropout | attention/input/FFN 0.1 |
| BOS / per-layer item tables | disabled / disabled |
| embedding LR / deep LR | 0.032 / 0.012 |
| schedule | linear decay, no warmup, one cycle |
| optimizer | Adam; weight decay 0; no gradient clipping |
| initialization | truncated normal std 0.02 |
| train/eval | 10 epochs; evaluate at epoch 10; batch 512; validation batch 8192 |
| runtime | bf16; compile disabled |
| negatives | 512 uniform random negatives; dense exact catalog scoring during training |

| final variant | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch time | peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `future_baseline` (`rqfinal_normalization_input_rms`) | **0.14589** | **0.05578** | 0.03055 | 0.02402 | 0.65197 | 13.2s steady | 23.9 GB |
| `rqfinal_neg_random` | 0.14480 | 0.05529 | **0.03156** | **0.02430** | 0.63332 | 12.8s steady | 23.9 GB |
| `selected_quality_b512_e4e3_d12e3` | 0.14358 | 0.05484 | 0.02935 | 0.02362 | 0.57309 | 13.0s steady | 23.9 GB |
| `selected_quality` | 0.1439 ±0.0005 | 0.0556 ±0.0004 | 0.0309 ±0.0005 | 0.0249 ±0.0005 | 0.4589 ±0.0046 | 20.8 ±0.1s | 21.5 GB |
| `selected_balanced` | 0.1407 ±0.0012 | 0.0546 ±0.0006 | 0.0306 ±0.0005 | 0.0245 ±0.0004 | 0.4648 ±0.0116 | 18.0 ±0.1s | 13.3 GB |

# Every result, by question

Written by `analysis/collect.py --write`; edits below this line are overwritten.
Historical tables use four full-Yambda-500M repeats. New tuned tables use one
500M run per treatment and the same final-seven-day timestamp holdout.

<!-- QUESTION TABLES -->

<!-- run-prefix: g1_calibrated_ -->

## rq1 — does µTransfer work?

Hypothesis: with a fixed 64-dimensional item table, learned input and μP
readout projections, μP initialization, MuAdam, and width-aware attention
scaling, the deep learning rate selected at width 32 should transfer to width
128. Result: yes at the stable deep LR 0.05. Recall rises from 0.1256 at width
32 to 0.1303 at width 128 with lower four-seed spread, while LR 0.1 crosses the
width-128 stability boundary. The fixed item table and projections are
necessary: width changes only the transformer, as required by μP.

**muTransfer rate transfer across width (reference: standard width=64)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mup_dim128_lr5e2 | — | 4 | +5% (0.1303 ±0.0006) | +6% (0.0502 ±0.0003) | +10% (0.0268 ±0.0001) | +10% (0.0218 ±0.0001) | +43% (0.5284 ±0.0067) |
| cosine_dim_128 | — | 4 | +2% (0.1266 ±0.0012) | +3% (0.0486 ±0.0007) | +5% (0.0255 ±0.0005) | +5% (0.0209 ±0.0007) | +36% (0.5031 ±0.0142) |
| mup_dim32_lr5e2 | — | 4 | +1% (0.1256 ±0.0008) | +2% (0.0482 ±0.0004) | +3% (0.0250 ±0.0007) | +4% (0.0207 ±0.0005) | +10% (0.4061 ±0.0277) |
| lr_cosine_warmup | standard width=64; item embedding dim=64; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |
| mup_dim32_lr1e1 | — | 4 | -3% (0.1206 ±0.0040) | -3% (0.0460 ±0.0018) | -3% (0.0235 ±0.0013) | -3% (0.0194 ±0.0011) | +2% (0.3760 ±0.0260) |
| cosine_dim_32 | — | 4 | -8% (0.1146 ±0.0015) | -9% (0.0432 ±0.0005) | -9% (0.0220 ±0.0003) | -10% (0.0179 ±0.0002) | -31% (0.2536 ±0.0081) |
| mup_dim128_lr1e1 | — | 4 | -39% (0.0763 ±0.0581) | -39% (0.0290 ±0.0224) | -37% (0.0154 ±0.0119) | -37% (0.0125 ±0.0097) | -40% (0.2232 ±0.2565) |

## rq3 — best metrics/performance balance

Hypothesis: moderate model width and separately tuned embedding and deep
learning rates should retain quality while reducing epoch time. Result: width
64 remains the balance point. Batch 512 with LR 0.032/0.012 and input RMSNorm
reaches 0.14589 recall@100 in 13.2 steady seconds/epoch at 23.9 GB. It is both
faster and better than the historical batch-128 maximum. The older in-batch
`selected_balanced` arm remains the lower-memory choice at 13.3 GB.

**Quality per unit of cost**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rqfinal_normalization_input_rms | batch=512; LR=0.032/0.012; input RMSNorm | 1 | **0.14589** | **0.05578** | 0.03055 | 0.02402 | 0.65197 | 13.2 steady | 23.9 | 0.100M | 10.1M | 9 |
| rqfinal_neg_random | batch=512; LR=0.032/0.012; no input norm | 1 | 0.14480 | 0.05529 | **0.03156** | **0.02430** | 0.63332 | **12.8 steady** | 23.9 | 0.100M | 10.1M | 9 |
| selected_quality_b512_e4e3_d12e3 | batch=512; embedding LR=0.004; deep LR=0.012 | 1 | −0.22% (0.14358) | −1.37% (0.05484) | 0.02935 | 0.02362 | 0.57309 | 13.0 steady | 23.9 | 0.100M | 10.1M | 9 |
| selected_quality | — | 4 | +15% (0.1439 ±0.0005) | +18% (0.0556 ±0.0004) | +27% (0.0309 ±0.0005) | +25% (0.0249 ±0.0005) | +24% (0.4589 ±0.0046) | 20.8 ±0.1 | 21.5 ±0.0 | 0.100M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| selected_balanced | — | 4 | +13% (0.1407 ±0.0012) | +15% (0.0546 ±0.0006) | +26% (0.0306 ±0.0005) | +23% (0.0245 ±0.0004) | +26% (0.4648 ±0.0116) | 18.0 ±0.1 | 13.3 ±0.0 | 0.100M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_dim_128 | — | 4 | +2% (0.1266 ±0.0012) | +3% (0.0486 ±0.0007) | +5% (0.0255 ±0.0005) | +5% (0.0209 ±0.0007) | +36% (0.5031 ±0.0142) | 27.1 ±0.8 | 25.9 ±0.0 | 0.410M ±0.000M | 20.1M ±0.0M | 9 ±0 |
| lr_cosine_warmup | width=64; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| cosine_dim_32 | — | 4 | -8% (0.1146 ±0.0015) | -9% (0.0432 ±0.0005) | -9% (0.0220 ±0.0003) | -10% (0.0179 ±0.0002) | -31% (0.2536 ±0.0081) | 21.9 ±1.4 | 6.9 ±0.0 | 0.029M ±0.000M | 5.0M ±0.0M | 9 ±0 |

**Embedding learning rate under cosine warmup (baseline: embedding LR=0.001; deep LR fixed at 0.001)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | embedding LR=0.001; deep LR=0.001 | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| embedding_lr_5e4 | — | 4 | -4% (0.1202 ±0.0013) | -3% (0.0457 ±0.0004) | -3% (0.0235 ±0.0003) | -3% (0.0193 ±0.0002) | -25% (0.2760 ±0.0059) | 19.3 ±0.9 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| embedding_lr_2e4 | — | 4 | -14% (0.1067 ±0.0010) | -15% (0.0403 ±0.0005) | -16% (0.0204 ±0.0004) | -16% (0.0167 ±0.0003) | -54% (0.1714 ±0.0051) | 19.2 ±0.4 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| embedding_lr_1e4 | — | 4 | -31% (0.0865 ±0.0043) | -31% (0.0326 ±0.0017) | -34% (0.0161 ±0.0012) | -32% (0.0136 ±0.0009) | -71% (0.1077 ±0.0096) | 20.5 ±1.8 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

**Deep learning rate under cosine warmup (baseline: deep LR=0.001; embedding LR fixed at 0.001)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep_lr_3e3 | — | 4 | +4% (0.1302 ±0.0008) | +5% (0.0498 ±0.0004) | +6% (0.0258 ±0.0004) | +7% (0.0213 ±0.0003) | +16% (0.4298 ±0.0084) | 19.2 ±0.7 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| deep_lr_5e3 | — | 4 | +4% (0.1299 ±0.0014) | +5% (0.0497 ±0.0005) | +8% (0.0262 ±0.0006) | +8% (0.0214 ±0.0004) | +18% (0.4380 ±0.0043) | 19.6 ±0.5 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| deep_lr_2e3 | — | 4 | +3% (0.1286 ±0.0010) | +4% (0.0491 ±0.0004) | +6% (0.0258 ±0.0005) | +6% (0.0211 ±0.0002) | +14% (0.4230 ±0.0092) | 19.0 ±0.3 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |
| lr_cosine_warmup | deep LR=0.001; embedding LR=0.001 | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 | 21.5 ±0.1 | 13.2 ±0.0 | 0.107M ±0.000M | 10.1M ±0.0M | 9 ±0 |

## rq4 — does SwiGLU help?

Hypothesis: SwiGLU should improve ranking quality over GELU, but the comparison
must tune each family's width and rates independently. The equal proxy grid
tested GELU widths 128/171/256/384 and SwiGLU widths 96/128/171/224 at three LR
pairs. Its best GELU-128 treatment and reused SwiGLU-192 control then ran once
on 500M. Result: SwiGLU wins by 2.3% recall, far outside shared noise.

**Independently tuned FFN families on 500M (baseline: SwiGLU-192)**

| variant | selected configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SwiGLU | width=171; LR=0.032/0.012 | 1 | **0.14480** | **0.05529** | **0.03156** | **0.02430** | 0.63332 |
| GELU | width=128; LR=0.032/0.012 | 1 | −2.3% (0.14149) | −2.8% (0.05375) | −3.2% (0.03055) | −4.7% (0.02316) | +6.4% (0.67389) |

**Historical fixed-recipe check (baseline: GELU, width=256)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cosine_ffn_swiglu_matched | — | 4 | +2% (0.1267 ±0.0027) | +3% (0.0485 ±0.0010) | +4% (0.0253 ±0.0007) | +5% (0.0208 ±0.0006) | +7% (0.3962 ±0.0154) |
| cosine_ffn_swiglu | — | 4 | +1% (0.1261 ±0.0006) | +2% (0.0481 ±0.0003) | +3% (0.0251 ±0.0002) | +3% (0.0205 ±0.0003) | +9% (0.4019 ±0.0181) |
| lr_cosine_warmup | GELU, ffn_dim=256; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |

## rq5 — which lr scheduler works best?

Hypothesis: decaying schedules should outperform a constant learning rate, with
smooth cosine or linear decay likely to be strongest. Two- and four-cycle
cosine restarts test whether repeated exploration helps. Result: linear led the
accepted schedule screen; two and four cosine cycles are tied with one cycle
and add no benefit.

**Cosine restarts (warmup=5%)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | cosine; cycles=1; warmup=5%; LR=0.001 | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |
| lr_cosine_cycles2 | — | 4 | -0% (0.1243 ±0.0009) | -0% (0.0472 ±0.0002) | -0% (0.0242 ±0.0002) | -1% (0.0198 ±0.0002) | -1% (0.3673 ±0.0034) |
| lr_cosine_cycles4 | — | 4 | -0% (0.1242 ±0.0008) | -0% (0.0472 ±0.0003) | -1% (0.0241 ±0.0001) | -1% (0.0198 ±0.0001) | +0% (0.3702 ±0.0028) |

## rq6 — does lr warmup help?

Hypothesis: a 5% warmup should stabilize early optimization and improve
constant, cosine, and inverse-sqrt schedules. Each table changes only warmup
within one schedule shape. Result: all three changes are inside four-seed
noise, so warmup is not independently beneficial.

**Cosine**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | — | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |

## rq7 — rope / alibi / position embeddings, and from-the-end variants

Hypothesis: relative encodings should generalize better than learned absolute
positions, while counting from the sequence end may align variable histories.
All 18 existing treatments received the same three-LR proxy budget, then each
proxy winner ran once on 500M. Learned forward remains best. Learned reverse
alone loses 9.6%; adding ALiBi or RoPE does not rescue it. Pure forward and
reverse RoPE lose 2.1% and 1.6%, respectively.

**Tuned position encodings on 500M (baseline: learned forward)**

| encoding | selected LR | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned forward | 0.032/0.012 | 1 | **0.14480** | **0.05529** | **0.03156** | **0.02430** | 0.63332 |
| learned forward + ALiBI | 0.032/0.012 | 1 | −0.7% (0.14384) | −1.5% (0.05448) | −5.6% (0.02980) | −4.2% (0.02327) | +3.1% (0.65305) |
| RoPE forward + ALiBi | 0.032/0.012 | 1 | −1.0% (0.14339) | −1.4% (0.05450) | −2.7% (0.03070) | −3.2% (0.02351) | +6.3% (0.67303) |
| ALiBi | 0.032/0.012 | 1 | −1.4% (0.14276) | −1.1% (0.05468) | −5.6% (0.02979) | −2.8% (0.02362) | +7.1% (0.67844) |
| RoPE forward + learned forward + ALiBi | 0.032/0.012 | 1 | −1.6% (0.14251) | −0.8% (0.05486) | −3.9% (0.03033) | −1.3% (0.02397) | +1.3% (0.64185) |
| RoPE reverse | 0.032/0.012 | 1 | −1.6% (0.14247) | −1.9% (0.05423) | −6.3% (0.02956) | −4.5% (0.02320) | +4.0% (0.65864) |
| RoPE reverse + ALiBi | 0.032/0.012 | 1 | −1.7% (0.14237) | −1.1% (0.05468) | −1.1% (0.03121) | −1.1% (0.02402) | +5.6% (0.66869) |
| RoPE reverse + learned forward + ALiBi | 0.032/0.012 | 1 | −1.7% (0.14236) | −1.0% (0.05471) | −2.2% (0.03088) | −1.2% (0.02399) | +0.3% (0.63504) |
| RoPE forward | 0.032/0.012 | 1 | −2.1% (0.14178) | −2.4% (0.05397) | −4.4% (0.03017) | −4.5% (0.02320) | +3.7% (0.65663) |
| none | 0.032/0.012 | 1 | −2.1% (0.14171) | −2.6% (0.05387) | −6.1% (0.02962) | −5.4% (0.02298) | +5.7% (0.66933) |
| RoPE reverse + learned reverse | 0.032/0.012 | 1 | −2.3% (0.14143) | −3.6% (0.05333) | −11.7% (0.02787) | −8.3% (0.02228) | −7.7% (0.58461) |
| RoPE reverse + learned forward | 0.032/0.012 | 1 | −2.5% (0.14116) | −3.1% (0.05357) | −7.7% (0.02913) | −6.1% (0.02282) | +0.4% (0.63587) |
| RoPE forward + learned reverse | 0.032/0.012 | 1 | −2.5% (0.14115) | −4.1% (0.05301) | −11.3% (0.02801) | −9.3% (0.02203) | −6.6% (0.59155) |
| learned reverse + ALiBi | 0.032/0.012 | 1 | −2.6% (0.14107) | −3.4% (0.05343) | −8.1% (0.02900) | −7.0% (0.02259) | −1.8% (0.62220) |
| RoPE reverse + learned reverse + ALiBi | 0.032/0.012 | 1 | −2.6% (0.14105) | −2.8% (0.05374) | −9.2% (0.02866) | −5.7% (0.02292) | −3.6% (0.61024) |
| RoPE forward + learned forward | 0.032/0.012 | 1 | −2.9% (0.14063) | −3.6% (0.05332) | −9.0% (0.02873) | −6.9% (0.02261) | +1.0% (0.63936) |
| RoPE forward + learned reverse + ALiBi | 0.016/0.006 | 1 | −7.3% (0.13416) | −8.3% (0.05073) | −14.5% (0.02697) | −12.2% (0.02133) | −13.6% (0.54749) |
| learned reverse | 0.032/0.012 | 1 | −9.6% (0.13088) | −11.2% (0.04912) | −20.2% (0.02518) | −17.6% (0.02003) | −10.9% (0.56425) |

**Historical fixed-recipe subset (baseline: learned forward)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | learned forward positions; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |
| cosine_pos_rope | — | 4 | -3% (0.1213 ±0.0031) | -3% (0.0461 ±0.0011) | -2% (0.0238 ±0.0006) | -2% (0.0195 ±0.0005) | -8% (0.3399 ±0.0111) |
| cosine_pos_rope_reverse | — | 4 | -3% (0.1207 ±0.0016) | -3% (0.0459 ±0.0008) | -2% (0.0238 ±0.0005) | -3% (0.0194 ±0.0004) | -9% (0.3350 ±0.0117) |
| cosine_pos_learned_reverse | — | 4 | -8% (0.1144 ±0.0025) | -10% (0.0428 ±0.0008) | -14% (0.0210 ±0.0007) | -13% (0.0173 ±0.0004) | -19% (0.2989 ±0.0120) |

## rq8 — scaling

Hypothesis: quality should improve with width, depth, sequence length,
attention capacity, and FFN capacity until regularization or compute cost
dominates. The follow-up specifically checks zero dropout, sequence length 128,
GQA throughput, post-norm stability, and a window of 50. Because FlashAttention
consumes ragged sequences, max_seq_len=128 does not itself create a Tensor-Core
alignment benefit. Each dependence is reported in its own table. Result: keep
width 64, two layers, pre-norm, dropout 0.1 and learned forward positions;
adopt sequence length 128, two-query/one-KV-head GQA, attention window 50, and
input RMSNorm. No full-data MHA arm establishes a recall gain, so GQA stays for
speed. Input RMSNorm improves recall by 0.8%; post-norm loses 4.7%.

**Shared attention window under cosine warmup (baseline: full attention)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| window_50 | — | 4 | +1% (0.1260 ±0.0009) | +1% (0.0479 ±0.0005) | +0% (0.0244 ±0.0006) | +2% (0.0202 ±0.0005) | +5% (0.3888 ±0.0057) |
| lr_cosine_warmup | full attention; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |

**Embedding and model dimension under cosine warmup (baseline: dim=64)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cosine_dim_128 | — | 4 | +2% (0.1266 ±0.0012) | +3% (0.0486 ±0.0007) | +5% (0.0255 ±0.0005) | +5% (0.0209 ±0.0007) | +36% (0.5031 ±0.0142) |
| lr_cosine_warmup | dim=64; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |
| cosine_dim_32 | — | 4 | -8% (0.1146 ±0.0015) | -9% (0.0432 ±0.0005) | -9% (0.0220 ±0.0003) | -10% (0.0179 ±0.0002) | -31% (0.2536 ±0.0081) |

**Sequence length under cosine warmup (baseline: max_seq_len=100)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | max_seq_len=100; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |
| cosine_seq_128 | — | 4 | -0% (0.1241 ±0.0008) | -0% (0.0472 ±0.0006) | -1% (0.0241 ±0.0002) | +0% (0.0199 ±0.0004) | -5% (0.3501 ±0.0127) |

**Tuned attention heads on 500M (baseline: GQA, 2 query / 1 KV)**

| attention | selected LR | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GQA, 2 query / 1 KV | 0.032/0.012 | 1 | **0.14480** | 0.05529 | 0.03156 | 0.02430 | 0.63332 |
| MHA, 8 heads | 0.032/0.012 | 1 | −0.2% (0.14457) | +0.5% (0.05556) | −1.4% (0.03111) | −0.2% (0.02426) | +7.2% (0.67899) |
| MHA, 2 heads | 0.032/0.012 | 1 | −0.4% (0.14420) | +1.1% (0.05592) | +3.1% (0.03254) | +3.2% (0.02507) | +5.9% (0.67065) |
| MHA, 4 heads | 0.032/0.012 | 1 | −0.7% (0.14380) | −0.3% (0.05511) | −1.6% (0.03105) | −0.9% (0.02409) | +8.3% (0.68603) |
| MHA, 1 head | 0.032/0.012 | 1 | −0.7% (0.14379) | −0.7% (0.05489) | −4.5% (0.03014) | −2.5% (0.02370) | +8.3% (0.68587) |

The 0.00023–0.00101 recall gaps span roughly 0.5–2.1 shared control standard
deviations; none establishes an MHA recall gain. GQA was 7% faster in the
controlled throughput comparison and remains selected.

**Historical GQA check (baseline: MHA, 2 heads / 2 KV)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | MHA: heads=2, kv_heads=2; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |
| cosine_heads_gqa | — | 4 | -0% (0.1244 ±0.0007) | -0% (0.0471 ±0.0002) | -0% (0.0242 ±0.0005) | -1% (0.0198 ±0.0003) | -2% (0.3616 ±0.0058) |

**Dropout under cosine warmup (baseline: dropout=0.1)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | dropout=0.1; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |
| cosine_dropout_0 | — | 4 | -4% (0.1200 ±0.0007) | -3% (0.0459 ±0.0003) | -2% (0.0237 ±0.0005) | -2% (0.0196 ±0.0004) | +23% (0.4567 ±0.0075) |

**Tuned normalization on 500M (baseline: pre-LayerNorm, no input norm, final LayerNorm)**

| normalization | selected LR | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| input RMSNorm + pre-LayerNorm + final LayerNorm | 0.032/0.012 | 1 | **+0.8% (0.14589)** | **+0.9% (0.05578)** | −3.2% (0.03055) | −1.1% (0.02402) | +2.9% (0.65197) |
| input LayerNorm + pre-LayerNorm + final LayerNorm | 0.032/0.012 | 1 | +0.4% (0.14543) | +0.7% (0.05566) | +0.3% (0.03165) | +0.1% (0.02431) | +4.6% (0.66234) |
| pre-LayerNorm, no input norm, final LayerNorm | 0.032/0.012 | 1 | 0.14480 | 0.05529 | 0.03156 | 0.02430 | 0.63332 |
| input RMSNorm + pre-LayerNorm + final RMSNorm | 0.032/0.012 | 1 | −0.8% (0.14366) | −0.9% (0.05478) | −3.9% (0.03032) | −2.9% (0.02359) | +7.6% (0.68139) |
| pre-RMSNorm, no input norm, final LayerNorm | 0.032/0.012 | 1 | −2.3% (0.14146) | −2.9% (0.05371) | −5.8% (0.02975) | −5.4% (0.02298) | +7.4% (0.68012) |
| pre-BatchNorm, no input norm, final LayerNorm | 0.032/0.012 | 1 | −2.5% (0.14122) | −3.2% (0.05351) | −6.1% (0.02964) | −6.1% (0.02280) | +3.2% (0.65375) |
| post-LayerNorm, no input norm, final LayerNorm | 0.016/0.006 | 1 | −4.7% (0.13806) | −4.4% (0.05285) | −9.4% (0.02860) | −5.5% (0.02296) | −5.3% (0.59973) |
| pre-LayerNorm, no input/final norm | 0.032/0.012 | 1 | −5.5% (0.13690) | −6.8% (0.05153) | −11.0% (0.02810) | −11.5% (0.02151) | +19.6% (0.75740) |

**Historical norm-place check (baseline: pre-norm)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | pre-norm; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |
| cosine_norm_post | — | 4 | -1% (0.1238 ±0.0019) | -1% (0.0467 ±0.0007) | -3% (0.0236 ±0.0006) | -3% (0.0194 ±0.0003) | +12% (0.4149 ±0.0047) |

## rq9 — timestamp-delta embeddings

Hypothesis: learned binned time deltas and timestamp-aware RoPE should help
distinguish short-term intent from older interactions. The follow-up crosses
32-bin addition with reverse timestamp RoPE and tests 8, 16, 32, and 64
logarithmically spaced delta bins. Result: additive 16-bin deltas are best at
0.1322; reverse timestamp RoPE adds cost without quality.

**Timestamp delta and timestamp RoPE under cosine warmup (baseline: no timestamp-delta feature)**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| time_bins_16 | — | 4 | +6% (0.1322 ±0.0012) | +7% (0.0505 ±0.0008) | +12% (0.0271 ±0.0007) | +11% (0.0220 ±0.0008) | -7% (0.3456 ±0.0159) |
| time_bins_64 | — | 4 | +6% (0.1318 ±0.0009) | +7% (0.0505 ±0.0004) | +11% (0.0270 ±0.0002) | +11% (0.0220 ±0.0004) | -7% (0.3435 ±0.0133) |
| time_bins_add | — | 4 | +6% (0.1315 ±0.0007) | +7% (0.0505 ±0.0003) | +13% (0.0275 ±0.0006) | +12% (0.0222 ±0.0004) | -8% (0.3387 ±0.0108) |
| time_bins_reverse_rope | — | 4 | +5% (0.1311 ±0.0013) | +6% (0.0502 ±0.0006) | +12% (0.0272 ±0.0003) | +10% (0.0219 ±0.0003) | -7% (0.3455 ±0.0125) |
| time_bins_8 | — | 4 | +5% (0.1304 ±0.0009) | +5% (0.0499 ±0.0006) | +11% (0.0269 ±0.0007) | +9% (0.0217 ±0.0006) | -7% (0.3425 ±0.0058) |
| lr_cosine_warmup | no time feature; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |

## rq10 — per-layer embeddings (Gemma-style)

Hypothesis: fresh item embeddings at each transformer layer should add capacity
and improve ranking quality, at increased parameter and memory cost. Result:
recall is flat while embedding parameters triple, so do not use per-layer item
tables.

## rq11 — online/offline logQ, random, random+logQ, or uncorrected in-batch?

Each negative family received its own 50M LR/batch screen; mixed families also
tuned negative count and random share. The selected configuration from every
family then ran once on 500M. This reverses neither the earlier random-negative
winner nor the no-logQ failure: uniform random is 2.9% above tuned offline
logQ, well outside the shared ±0.00049 recall noise band. Fixed offline logQ
is the best in-batch method. Online logQ, random with offline correction, and
both random/in-batch mixtures are worse after tuning.

**Tuned negative sampling on 500M (baseline: tuned fixed offline logQ)**

| variant | selected 50M configuration transferred to 500M | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | batch 512; LR 0.032/0.012; 512 random | 1 | **+2.9% (0.14480)** | **+3.4% (0.05529)** | **+5.4% (0.03156)** | **+5.4% (0.02430)** | −1.6% (0.63332) |
| offline logQ | batch 512; LR 0.016/0.006; 512 in-batch; positive correction, false-negative mask, own-sequence exclusion | 1 | 0.14066 | 0.05348 | 0.02996 | 0.02306 | 0.64385 |
| random + offline logQ | batch 512; LR 0.032/0.012; 512 proposal samples | 1 | −3.8% (0.13533) | −3.6% (0.05154) | −4.0% (0.02876) | −4.1% (0.02212) | +21.0% (0.77895) |
| mixed online logQ | batch 512; LR 0.016/0.006; 75% random; 512 total | 1 | −11.1% (0.12499) | −11.0% (0.04758) | −11.1% (0.02664) | −11.1% (0.02050) | +21.1% (0.77985) |
| online logQ | batch 512; LR 0.008/0.006; alpha 0.01; 512 in-batch | 1 | −14.3% (0.12061) | −14.0% (0.04597) | −16.1% (0.02513) | −14.6% (0.01969) | +20.3% (0.77454) |
| mixed offline logQ | batch 512; LR 0.016/0.006; 50% random; 256 total | 1 | −15.1% (0.11939) | −16.1% (0.04489) | −17.1% (0.02482) | −17.5% (0.01903) | +28.4% (0.82648) |
| in-batch without logQ | batch 1024; LR 0.016/0.006; 512 in-batch | 1 | −47.8% (0.07343) | −48.0% (0.02784) | −51.3% (0.01459) | −48.7% (0.01184) | +45.7% (0.93813) |

## rq2 — best combination for metrics

Hypothesis: combining independently selected parameters should beat every
single-axis corrected-baseline variant. The final combined arm reaches
recall@100 **0.14589** and NDCG@100 **0.05578**. It uses dim 64, depth 2,
sequence length 128, 2 query heads/1 KV head, window 50, SwiGLU width 192,
pre-LayerNorm with input RMSNorm and final LayerNorm, learned forward positions,
dropout 0.1, 16 additive time-delta bins, and 512 uniform random negatives.
Training uses LR 0.032/0.012, linear decay without warmup, 10 epochs, batch 512,
and bf16. Weight decay, gradient clipping, BOS, per-layer item tables, RoPE, and
ALiBi are disabled.

**Final metric candidates**

| variant | reference configuration | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rqfinal_normalization_input_rms | tuned selected control + input RMSNorm | 1 | **+17.1% (0.14589)** | **+17.9% (0.05578)** | +25.7% (0.03055) | +20.7% (0.02402) | +76.2% (0.65197) |
| rqfinal_neg_random | tuned selected control; no input norm | 1 | +16.2% (0.14480) | +16.9% (0.05529) | **+29.9% (0.03156)** | **+22.1% (0.02430)** | +71.2% (0.63332) |
| selected_quality | — | 4 | +15% (0.1439 ±0.0005) | +18% (0.0556 ±0.0004) | +27% (0.0309 ±0.0005) | +25% (0.0249 ±0.0005) | +24% (0.4589 ±0.0046) |
| selected_balanced | — | 4 | +13% (0.1407 ±0.0012) | +15% (0.0546 ±0.0006) | +26% (0.0306 ±0.0005) | +23% (0.0245 ±0.0004) | +26% (0.4648 ±0.0116) |
| lr_cosine_warmup | corrected baseline; cosine, warmup=5% | 4 | 0.1246 ±0.0007 | 0.0473 ±0.0002 | 0.0243 ±0.0002 | 0.0199 ±0.0002 | 0.3700 ±0.0020 |
