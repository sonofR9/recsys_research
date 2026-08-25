# G1 research questions — Yambda-50M results

Frozen before the final protocol moved to Yambda-500M; `analysis/collect.py` does not
regenerate this file.

Questions from [../../../list.md](../../../list.md). Results live in
[results.md](results.md); this file tracks what each one needs and what it found.

The corrected four-run baseline is **0.1120 ±0.0018**; scheduled ablations use
the cosine-warmup reference of **0.1116 ±0.0017**. The pooled per-run σ is
0.00228 (2.0% of baseline), so changes below roughly 4.1% do not clear 2σ.
Every conclusion below uses the same full-50M, future-seven-day protocol.

| rq | question | answer | best variant | recall@100 | vs baseline | runs |
| --- | --- | --- | --- | --- | --- | --- |
| rq1 | Does µTransfer work? | **no** — the width-32 optimum does not transfer and every full-width μP arm collapses | `mup_dim128_lr5e4` | 0.0346 ±0.0011 | **−69.0%** vs standard width 128 | 4 |
| rq2 | Best transformer combination for metrics | embedding LR 0.02 plus reverse position priors | `combo_embedding_position` | 0.1206 ±0.0010 | **+8.0%** vs scheduled reference | 4 |
| rq3 | Best metrics/performance balance | **dim 64 + cosine warmup** keeps 99.5% of recall while halving the item table | `cosine_dim_64` | 0.1111 ±0.0016 | −0.5% vs scheduled width 128 | 4 |
| rq4 | Does SwiGLU help? | **no resolved gain**; parameter-matched GELU is slightly higher inside noise | `cosine_ffn_gelu_matched` | 0.1136 ±0.0033 | +1.7%, inside noise | 4 |
| rq5 | Which lr scheduler works best? | **WSD and step tie**; exponential is much too aggressive | `lr_wsd` | 0.1173 ±0.0024 | **+4.7%** vs constant | 4 |
| rq6 | Does lr warmup help? | **schedule-dependent**: inverse sqrt +11.5%; cosine and constant move <1% | `lr_inverse_sqrt_warmup` | 0.1075 ±0.0024 | **+11.5%** vs no warmup | 4 |
| rq7 | rope / alibi / position embeddings | reverse learned positions with ALiBi have the highest mean, inside noise | `cosine_pos_reverse_all` | 0.1148 ±0.0014 | +2.9%, inside noise | 4 |
| rq8 | Scaling: dim, layers, seq len, heads, GQA, windows, norm, BOS, ffn ratio | removing input LayerNorm is the only clear quality gain; dim 64 is the efficiency winner | `cosine_norm_no_input` | 0.1182 ±0.0029 | **+5.9%** | 4 |
| rq9 | Timestamp-delta embeddings | learned binned deltas concatenated through a dense encoder work best | `time_bins_concat` | 0.1177 ±0.0016 | **+5.4%** | 4 |
| rq10 | Per-layer embeddings (Gemma-style) | **no** — +1.2% is inside noise while embedding parameters triple | `per_layer_embeddings` | 0.1130 ±0.0021 | +1.2%, inside noise | 4 |
| rq11 | Negative sampling and logQ | offline logQ is best; omitting logQ is strongly harmful | `neg_offline_logq` | 0.1220 ±0.0020 | **+3.7%** vs online logQ | 4 |

For rq3, width 64 keeps 99.5% of scheduled width-128 recall while halving the
item-table parameters. Full per-axis metrics and costs are in
[results.md](results.md).

## What already exists

`TransformerConfig` now reaches rope, ALiBi, the learned forward and
from-the-end tables (rq7), the ffn kind (rq4) and the normalization kind and
place (rq8). All are screened in [results.md](results.md).

`LrSchedule` scales each parameter group from its own starting rate through
warmup and constant, linear, cosine, inverse-square-root, step, exponential,
polynomial or warmup-stable-decay shapes.

The BOS token is a tokenizer decorator with `is_target=False`, so N items yield
N pairs. It changes recall by −1.7% at a constant rate and +0.7% under cosine,
both inside the four-run resolution; BOS does not help. In this causal
next-item objective, a CLS position has no supervised target and would test a
different question.

## Completed extensions

- rq1 uses the official Microsoft μP base/delta shapes, tied-readout and
  attention scaling, and MuAdam at widths 32 and 128.
- rq5 covers eight schedule shapes and rq6 isolates warmup within shape.
- rq8 includes local windows, batch/RMS/LayerNorm, residual/input/final norm
  placement, and corrected BOS evaluation.
- rq9 covers plain/log/binned delta addition, dense concatenation, timestamp
  RoPE in both directions, and combinations.
- rq10 injects a separate item table before every transformer layer.
- rq11 compares online/offline logQ, uniform catalog sampling, and uncorrected
  in-batch negatives under identical architecture, rates, data, and seeds.
- Embedding and deep learning rates are swept separately; embedding LR 0.02 is
  crossed with the best compatible timestamp and position arms.

## Implementation notes

- Block, input and final normalization are separate knobs and tables.
- Post-norm changes the residual's dtype under autocast, so the `norm_post*`
  cost columns carry an artifact and should not be read as timings.
- `build_transformer_decoder` honours only part of `TransformerConfig`; the
  variants above all go through `build_causal_transformer`.

## rq8 — why scaling looked like it worked

The corrected catalog has 33,149 trainable items, so the width-128 model is no
longer a 108M-parameter mostly-unused table: it has 4.2M embedding and 0.31M
deep parameters. Width 64 retains 99.5% of scheduled recall while halving the
item table and reducing median epoch time from 2.6s to 2.3s; width 32 loses
6.4%. This makes width 64 the clear quality/cost point.

Each requested dependence has its own four-run table. Removing input
LayerNorm is the only scheduled architecture change that clears the 2σ noise
band (+5.9%). Removing final normalization loses 13.3%. Depth 4 (+3.1%),
sequence length 50 (+2.1%), MHA (+1.7%), local windows, pre/post norm, FFN
ratio, and BOS remain unresolved at this experiment's resolution. BatchNorm is
poor because packed tokens mix statistics across users and positions.

The pooled per-run σ is 0.00228, or 2.0% of baseline; a four-run mean change
below roughly 4.1% is treated as unresolved rather than a categorical win.

## rq5, rq6 — what the schedule rows do and do not say

WSD and step are tied at 0.1173 and 0.1171, roughly +4.7% over constant.
Cosine and linear match constant; polynomial is −5.3%, and exponential is
−23.8% because it decays too aggressively.

Warmup has no universal sign. It changes constant and cosine by less than 1%,
but raises inverse sqrt from 0.0964 to 0.1075 (+11.5%). The answer is therefore
schedule-dependent.

## rq7 — what the position result means

Answered under the scheduled baseline with four runs per combination. ALiBi
plus reverse learned positions has the highest mean at +2.9%, inside noise.
Removing all position information costs 5.9%, while learned positions without
ALiBi cost 3.3–4.6%. The robust conclusion is that a distance prior matters;
reversing or adding priors does not produce a resolved gain.

## Additional measured analyses

The embedding-rate sweep peaks at 0.03 (0.1194), with 0.02 and 0.04 in the same
noise band. The best tested transformer cross is embedding LR 0.02 plus the
reverse-position arm at 0.1206. Learned binned timestamp deltas concatenated
through a dense encoder reach 0.1177. Per-layer item tables add no resolved
quality and triple embedding parameters.

RQ11 is a separate controlled axis: offline logQ reaches 0.1220, uniform
random 0.1192, online logQ 0.1177, and uncorrected in-batch 0.1028. Omitting
proposal correction is the clear failure; offline versus online is +3.7% and
just below the report's conservative 2σ threshold.

# Every result, by question

Written by `analysis/collect.py --write`; edits below this line are overwritten. Every
rq1–rq11 conclusion uses four full-Yambda-50M runs with the final seven days
held out by timestamp.

<!-- QUESTION TABLES -->

## rq1 — does µTransfer work?

No. Width 32 selects 2e-3, but every full-width μP rate collapses to about
0.0345 versus 0.1116 for standard parameterization. Transferring the small-
width optimum therefore does not recover target-width quality. The tied item
input/readout makes width scaling change score temperature as well as the
transformer tower.

**muTransfer rate transfer across width (reference: standard width=128)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
| cosine_dim_32 | 4 | -6% (0.1045 ±0.0029) | -8% (0.0388 ±0.0011) | -6% (0.0215 ±0.0010) | -10% (0.0163 ±0.0006) | -53% (0.2094 ±0.0017) |
| mup_dim32_lr2e3 | 4 | -26% (0.0823 ±0.0024) | -25% (0.0313 ±0.0012) | -26% (0.0169 ±0.0010) | -22% (0.0141 ±0.0008) | -76% (0.1073 ±0.0056) |
| mup_dim32_lr1e3 | 4 | -37% (0.0698 ±0.0041) | -38% (0.0262 ±0.0018) | -38% (0.0141 ±0.0006) | -36% (0.0115 ±0.0010) | -82% (0.0802 ±0.0054) |
| mup_dim32_lr5e4 | 4 | -50% (0.0558 ±0.0034) | -51% (0.0207 ±0.0014) | -54% (0.0104 ±0.0007) | -51% (0.0088 ±0.0010) | -84% (0.0695 ±0.0040) |
| mup_dim128_lr5e4 | 4 | -69% (0.0346 ±0.0011) | -70% (0.0128 ±0.0003) | -73% (0.0062 ±0.0004) | -72% (0.0050 ±0.0003) | -91% (0.0413 ±0.0088) |
| mup_dim128_lr1e3 | 4 | -69% (0.0345 ±0.0007) | -69% (0.0129 ±0.0003) | -71% (0.0067 ±0.0005) | -71% (0.0053 ±0.0005) | -85% (0.0649 ±0.0113) |
| mup_dim128_lr2e3 | 4 | -69% (0.0343 ±0.0012) | -70% (0.0124 ±0.0007) | -77% (0.0053 ±0.0010) | -75% (0.0046 ±0.0007) | -72% (0.1240 ±0.0156) |

## rq3 — best metrics/performance balance

Dimension 64 is the best balance: it retains 99.5% of width-128 recall, halves
embedding parameters from 4.2M to 2.1M, and lowers median epoch time from 2.6s
to 2.3s. Width 32 loses 6.4% recall. Parameter columns separate the tower from
the dominant item table.

**Quality per unit of cost**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 | 2.6 ±0.0 | 4.6 ±0.1 | 0.311M ±0.000M | 4.2M ±0.0M | 12 ±1 |
| cosine_dim_64 | 4 | -0% (0.1111 ±0.0016) | -3% (0.0408 ±0.0006) | -5% (0.0217 ±0.0017) | -7% (0.0169 ±0.0010) | -33% (0.2993 ±0.0122) | 2.3 ±0.1 | 2.5 ±0.0 | 0.082M ±0.000M | 2.1M ±0.0M | 11 ±1 |
| cosine_dim_256 | 4 | -2% (0.1093 ±0.0024) | -3% (0.0406 ±0.0007) | -5% (0.0216 ±0.0008) | -6% (0.0170 ±0.0003) | +31% (0.5853 ±0.0067) | 4.1 ±0.0 | 8.9 ±0.1 | 1.212M ±0.000M | 8.5M ±0.0M | 12 ±0 |
| cosine_dim_32 | 4 | -6% (0.1045 ±0.0029) | -8% (0.0388 ±0.0011) | -6% (0.0215 ±0.0010) | -10% (0.0163 ±0.0006) | -53% (0.2094 ±0.0017) | 2.1 ±0.1 | 1.4 ±0.0 | 0.022M ±0.000M | 1.1M ±0.0M | 16 ±3 |
| cosine_dim_16 | 4 | -20% (0.0898 ±0.0020) | -19% (0.0342 ±0.0011) | -17% (0.0189 ±0.0010) | -15% (0.0154 ±0.0009) | -70% (0.1356 ±0.0039) | 2.2 ±0.1 | 0.8 ±0.0 | 0.007M ±0.000M | 0.5M ±0.0M | 17 ±0 |

**Embedding learning rate under cosine warmup (baseline: embedding LR=0.01; deep LR fixed at 0.001)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| embedding_lr_3e2 | 4 | +7% (0.1194 ±0.0010) | +5% (0.0443 ±0.0007) | +8% (0.0247 ±0.0004) | +4% (0.0189 ±0.0009) | -8% (0.4117 ±0.0024) | 2.6 ±0.0 | 4.6 ±0.1 | 0.311M ±0.000M | 4.2M ±0.0M | 14 ±1 |
| embedding_lr_4e2 | 4 | +7% (0.1189 ±0.0019) | +6% (0.0446 ±0.0005) | +11% (0.0254 ±0.0006) | +8% (0.0196 ±0.0003) | -13% (0.3883 ±0.0027) | 2.6 ±0.1 | 4.6 ±0.1 | 0.311M ±0.000M | 4.2M ±0.0M | 16 ±1 |
| embedding_lr_2e2 | 4 | +5% (0.1177 ±0.0026) | +5% (0.0441 ±0.0008) | +7% (0.0245 ±0.0010) | +5% (0.0190 ±0.0004) | -1% (0.4427 ±0.0054) | 2.6 ±0.1 | 4.6 ±0.1 | 0.311M ±0.000M | 4.2M ±0.0M | 14 ±1 |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 | 2.6 ±0.0 | 4.6 ±0.1 | 0.311M ±0.000M | 4.2M ±0.0M | 12 ±1 |
| embedding_lr_5e3 | 4 | -11% (0.0997 ±0.0023) | -12% (0.0371 ±0.0004) | -10% (0.0206 ±0.0006) | -13% (0.0157 ±0.0002) | -1% (0.4386 ±0.0037) | 2.6 ±0.0 | 4.6 ±0.1 | 0.311M ±0.000M | 4.2M ±0.0M | 18 ±1 |

**Deep learning rate under cosine warmup (baseline: deep LR=0.001; embedding LR fixed at 0.01)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | epoch_time | peak_memory_gb | params_deep | params_embedding | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep_lr_2e3 | 4 | +2% (0.1138 ±0.0021) | +1% (0.0426 ±0.0005) | +1% (0.0231 ±0.0010) | +1% (0.0182 ±0.0003) | +11% (0.4943 ±0.0108) | 2.6 ±0.0 | 4.6 ±0.1 | 0.311M ±0.000M | 4.2M ±0.0M | 11 ±1 |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 | 2.6 ±0.0 | 4.6 ±0.1 | 0.311M ±0.000M | 4.2M ±0.0M | 12 ±1 |
| deep_lr_5e4 | 4 | -3% (0.1082 ±0.0020) | -4% (0.0405 ±0.0005) | -5% (0.0217 ±0.0011) | -6% (0.0171 ±0.0006) | -4% (0.4255 ±0.0015) | 2.7 ±0.0 | 4.6 ±0.1 | 0.311M ±0.000M | 4.2M ±0.0M | 13 ±2 |

## rq4 — does SwiGLU help?

`ffn_gelu` swaps SwiGLU for a plain MLP; `ffn_gelu_matched` does the same and
widens it back to SwiGLU's parameter count, which is the only comparison that
isolates the gate. Both were run at the baseline's hyperparameters, which are
SwiGLU's. Under cosine the parameter-matched GELU row is 1.7% higher, inside
four-run noise. There is no evidence that SwiGLU helps in this shared-
hyperparameter test.

**Feedforward kind under cosine warmup (baseline: SwiGLU, ffn_dim=256)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_ffn_gelu_matched | 4 | +2% (0.1136 ±0.0033) | +1% (0.0424 ±0.0010) | +2% (0.0233 ±0.0011) | +0% (0.0181 ±0.0006) | +3% (0.4564 ±0.0086) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
| cosine_ffn_gelu | 4 | -0% (0.1113 ±0.0023) | -1% (0.0415 ±0.0005) | -7% (0.0212 ±0.0006) | -4% (0.0173 ±0.0004) | +3% (0.4564 ±0.0064) |

## rq5 — which lr scheduler works best?

Warmup-stable-decay and step are tied for best at 0.1173 and 0.1171, about 4.7%
above constant. Cosine and linear match constant, polynomial loses 5.3%, and
exponential decays too aggressively and loses 23.8%.

**Schedule shape, no warmup**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| lr_step | 4 | +5% (0.1171 ±0.0011) | +7% (0.0438 ±0.0007) | +4% (0.0236 ±0.0011) | +9% (0.0186 ±0.0007) | -15% (0.4630 ±0.0119) |
| lr_cosine | 4 | +1% (0.1127 ±0.0014) | +2% (0.0421 ±0.0004) | -4% (0.0219 ±0.0016) | +4% (0.0177 ±0.0003) | -13% (0.4728 ±0.0037) |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| lr_linear | 4 | -0% (0.1117 ±0.0023) | +2% (0.0418 ±0.0003) | +0% (0.0228 ±0.0010) | +5% (0.0179 ±0.0003) | -12% (0.4799 ±0.0060) |
| lr_polynomial | 4 | -5% (0.1060 ±0.0015) | -4% (0.0396 ±0.0005) | -7% (0.0211 ±0.0007) | -3% (0.0165 ±0.0007) | -24% (0.4116 ±0.0080) |
| lr_inverse_sqrt | 4 | -14% (0.0964 ±0.0021) | -12% (0.0361 ±0.0005) | -14% (0.0195 ±0.0007) | -12% (0.0150 ±0.0004) | -19% (0.4416 ±0.0044) |
| lr_exponential | 4 | -24% (0.0853 ±0.0020) | -23% (0.0317 ±0.0003) | -21% (0.0180 ±0.0001) | -22% (0.0133 ±0.0002) | -18% (0.4477 ±0.0040) |

**Warmup-stable-decay**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| lr_wsd | 4 | +5% (0.1173 ±0.0024) | +7% (0.0439 ±0.0007) | +5% (0.0240 ±0.0011) | +11% (0.0188 ±0.0007) | -1% (0.5405 ±0.0064) |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |

## rq6 — does lr warmup help?

Each shape with and without a 5% warmup, which is the only comparison that
isolates it; the baseline row is the constant-rate arm without warmup. The
result is schedule-dependent: inverse sqrt gains 11.5%, while cosine and
constant move by less than 1% and remain within noise.

**Constant**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| lr_warmup | 4 | -1% (0.1113 ±0.0023) | +0% (0.0413 ±0.0008) | -3% (0.0221 ±0.0003) | +1% (0.0172 ±0.0007) | -5% (0.5195 ±0.0270) |

**Cosine**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| lr_cosine | 4 | 0.1127 ±0.0014 | 0.0421 ±0.0004 | 0.0219 ±0.0016 | 0.0177 ±0.0003 | 0.4728 ±0.0037 |
| lr_cosine_warmup | 4 | -1% (0.1116 ±0.0017) | -0% (0.0420 ±0.0005) | +4% (0.0228 ±0.0009) | +2% (0.0181 ±0.0006) | -6% (0.4451 ±0.0032) |

**Inverse sqrt**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| lr_inverse_sqrt_warmup | 4 | +12% (0.1075 ±0.0024) | +12% (0.0405 ±0.0009) | +11% (0.0216 ±0.0011) | +15% (0.0173 ±0.0006) | -3% (0.4262 ±0.0097) |
| lr_inverse_sqrt | 4 | 0.0964 ±0.0021 | 0.0361 ±0.0005 | 0.0195 ±0.0007 | 0.0150 ±0.0004 | 0.4416 ±0.0044 |

## rq7 — rope / alibi / position embeddings, and from-the-end variants

ALiBi plus learned reverse positions has the highest mean at +2.9%, inside the
measured noise band. Removing all position information costs 5.9%; learned
positions without ALiBi cost 3.3–4.6%. Every row has four runs, and `*_reverse`
counts from the end of the window.

**Position encoding under cosine warmup (baseline: ALiBi + learned forward positions)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_pos_reverse_all | 4 | +3% (0.1148 ±0.0014) | +3% (0.0431 ±0.0007) | +1% (0.0230 ±0.0008) | +2% (0.0184 ±0.0005) | -1% (0.4424 ±0.0090) |
| cosine_pos_all | 4 | +2% (0.1133 ±0.0012) | +1% (0.0423 ±0.0004) | +0% (0.0229 ±0.0006) | -1% (0.0180 ±0.0005) | +0% (0.4452 ±0.0055) |
| cosine_pos_rope_learned_reverse | 4 | +1% (0.1132 ±0.0029) | +1% (0.0424 ±0.0007) | +6% (0.0241 ±0.0004) | +2% (0.0185 ±0.0002) | -1% (0.4428 ±0.0047) |
| cosine_pos_rope_reverse_learned_reverse | 4 | +0% (0.1120 ±0.0015) | +1% (0.0423 ±0.0003) | +4% (0.0237 ±0.0009) | +3% (0.0186 ±0.0005) | -1% (0.4426 ±0.0052) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
| cosine_pos_rope_learned | 4 | -0% (0.1111 ±0.0005) | -1% (0.0416 ±0.0007) | -4% (0.0220 ±0.0016) | -2% (0.0178 ±0.0007) | +2% (0.4518 ±0.0057) |
| cosine_pos_rope_alibi | 4 | -0% (0.1111 ±0.0011) | +0% (0.0421 ±0.0007) | -1% (0.0225 ±0.0010) | -1% (0.0180 ±0.0006) | +1% (0.4478 ±0.0089) |
| cosine_pos_alibi | 4 | -1% (0.1110 ±0.0015) | -0% (0.0418 ±0.0008) | -0% (0.0227 ±0.0010) | -1% (0.0179 ±0.0007) | +1% (0.4483 ±0.0105) |
| cosine_pos_rope | 4 | -2% (0.1099 ±0.0036) | -1% (0.0416 ±0.0011) | -6% (0.0215 ±0.0013) | -3% (0.0175 ±0.0008) | +1% (0.4492 ±0.0069) |
| cosine_pos_rope_reverse | 4 | -2% (0.1091 ±0.0024) | -2% (0.0412 ±0.0012) | -3% (0.0222 ±0.0016) | -3% (0.0176 ±0.0010) | +2% (0.4553 ±0.0064) |
| cosine_pos_learned | 4 | -3% (0.1080 ±0.0026) | -3% (0.0408 ±0.0003) | -4% (0.0218 ±0.0012) | -3% (0.0175 ±0.0007) | +1% (0.4503 ±0.0063) |
| cosine_pos_learned_reverse | 4 | -5% (0.1065 ±0.0027) | -4% (0.0405 ±0.0005) | +0% (0.0229 ±0.0002) | -2% (0.0177 ±0.0002) | +1% (0.4484 ±0.0033) |
| cosine_pos_none | 4 | -6% (0.1051 ±0.0014) | -6% (0.0395 ±0.0004) | -4% (0.0219 ±0.0011) | -6% (0.0171 ±0.0004) | +3% (0.4577 ±0.0055) |

**Position encoding (baseline: ALiBi + learned forward positions)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| pos_reverse_all | 4 | +3% (0.1154 ±0.0018) | +3% (0.0424 ±0.0007) | +4% (0.0236 ±0.0010) | +4% (0.0177 ±0.0004) | +1% (0.5498 ±0.0223) |
| pos_rope_reverse_learned_reverse | 4 | +1% (0.1127 ±0.0023) | +1% (0.0414 ±0.0008) | +0% (0.0228 ±0.0014) | +2% (0.0173 ±0.0006) | +2% (0.5555 ±0.0245) |
| pos_rope_learned_reverse | 4 | +0% (0.1123 ±0.0026) | +0% (0.0413 ±0.0007) | +1% (0.0230 ±0.0006) | +2% (0.0173 ±0.0003) | -2% (0.5335 ±0.0298) |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| pos_all | 4 | -0% (0.1119 ±0.0020) | -0% (0.0410 ±0.0004) | -3% (0.0222 ±0.0007) | -1% (0.0168 ±0.0004) | -0% (0.5436 ±0.0256) |
| pos_rope_learned | 4 | -0% (0.1118 ±0.0022) | -0% (0.0409 ±0.0007) | -5% (0.0217 ±0.0012) | -2% (0.0167 ±0.0007) | +1% (0.5512 ±0.0278) |
| pos_alibi | 4 | -1% (0.1112 ±0.0010) | -0% (0.0410 ±0.0004) | -5% (0.0217 ±0.0015) | +0% (0.0170 ±0.0005) | +5% (0.5706 ±0.0404) |
| pos_rope | 4 | -1% (0.1109 ±0.0015) | +0% (0.0412 ±0.0011) | +0% (0.0229 ±0.0011) | +4% (0.0176 ±0.0012) | +1% (0.5509 ±0.0261) |
| pos_rope_reverse | 4 | -1% (0.1105 ±0.0019) | -1% (0.0407 ±0.0013) | -1% (0.0226 ±0.0029) | +1% (0.0172 ±0.0018) | -1% (0.5421 ±0.0341) |
| pos_rope_alibi | 4 | -2% (0.1103 ±0.0013) | -0% (0.0410 ±0.0010) | -0% (0.0227 ±0.0006) | +2% (0.0173 ±0.0006) | +3% (0.5618 ±0.0377) |
| pos_learned | 4 | -2% (0.1093 ±0.0027) | -1% (0.0405 ±0.0013) | -2% (0.0223 ±0.0009) | +1% (0.0171 ±0.0010) | +1% (0.5523 ±0.0162) |
| pos_learned_reverse | 4 | -4% (0.1077 ±0.0015) | -4% (0.0396 ±0.0005) | -7% (0.0212 ±0.0004) | -2% (0.0166 ±0.0004) | +4% (0.5672 ±0.0113) |
| pos_none | 4 | -7% (0.1043 ±0.0022) | -5% (0.0391 ±0.0012) | -5% (0.0217 ±0.0020) | -1% (0.0168 ±0.0012) | +1% (0.5521 ±0.0356) |

## rq8 — scaling

One table per dependence, per list.md. Removing input LayerNorm is the only
clear scheduled architecture gain (+5.9%); removing final norm loses 13.3%.
Width 64 is the efficiency winner. Local windows and BOS are neutral; BatchNorm
is poor because packed tokens share statistics across users and time.

**Embedding and model dimension (baseline: dim=128)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| dim_32 | 4 | +1% (0.1130 ±0.0002) | +3% (0.0422 ±0.0005) | -4% (0.0220 ±0.0003) | +5% (0.0178 ±0.0005) | -36% (0.3512 ±0.0133) |
| dim_64 | 4 | +1% (0.1129 ±0.0010) | +2% (0.0418 ±0.0006) | +1% (0.0231 ±0.0014) | +3% (0.0175 ±0.0009) | -14% (0.4665 ±0.0195) |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| dim_256 | 4 | -4% (0.1073 ±0.0011) | -4% (0.0394 ±0.0007) | -5% (0.0216 ±0.0021) | -4% (0.0164 ±0.0012) | +6% (0.5794 ±0.0174) |
| dim_16 | 4 | -6% (0.1050 ±0.0019) | -4% (0.0396 ±0.0007) | -8% (0.0209 ±0.0013) | +1% (0.0172 ±0.0004) | -60% (0.2172 ±0.0104) |

**Depth (baseline: depth=2)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| depth_4 | 4 | -0% (0.1117 ±0.0007) | -0% (0.0410 ±0.0006) | -11% (0.0202 ±0.0014) | -2% (0.0167 ±0.0008) | +2% (0.5545 ±0.0266) |
| depth_1 | 4 | -2% (0.1101 ±0.0024) | -1% (0.0406 ±0.0010) | -1% (0.0226 ±0.0008) | +1% (0.0172 ±0.0006) | -2% (0.5343 ±0.0284) |

**Sequence length (baseline: max_seq_len=100)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| seq_50 | 4 | -0% (0.1119 ±0.0013) | -1% (0.0408 ±0.0007) | -8% (0.0209 ±0.0016) | -4% (0.0163 ±0.0007) | -0% (0.5429 ±0.0118) |
| seq_200 | 4 | -1% (0.1113 ±0.0010) | +0% (0.0413 ±0.0006) | +0% (0.0228 ±0.0017) | +3% (0.0175 ±0.0009) | +1% (0.5495 ±0.0289) |

**Number of attention heads (baseline: heads=4, kv_heads=2)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| heads_2 | 4 | -0% (0.1117 ±0.0030) | +0% (0.0412 ±0.0007) | -4% (0.0219 ±0.0015) | -1% (0.0169 ±0.0007) | -3% (0.5293 ±0.0136) |
| heads_8 | 4 | -1% (0.1107 ±0.0017) | -0% (0.0410 ±0.0004) | -4% (0.0220 ±0.0012) | +0% (0.0170 ±0.0005) | +2% (0.5545 ±0.0340) |

**Grouped-query attention (baseline: heads=4, kv_heads=2)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| heads_mha | 4 | -0% (0.1115 ±0.0018) | +0% (0.0413 ±0.0010) | -4% (0.0220 ±0.0015) | +3% (0.0175 ±0.0009) | +2% (0.5568 ±0.0232) |

**FFN ratio (baseline: ffn_dim=256 (2x model dim))**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| ffn_512 | 4 | -0% (0.1115 ±0.0032) | -1% (0.0408 ±0.0011) | -4% (0.0220 ±0.0015) | +0% (0.0170 ±0.0006) | +2% (0.5574 ±0.0333) |
| ffn_128 | 4 | -1% (0.1104 ±0.0031) | +1% (0.0414 ±0.0011) | +4% (0.0238 ±0.0014) | +6% (0.0181 ±0.0008) | +2% (0.5532 ±0.0251) |

**Dropout (baseline: dropout=0.1)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| dropout_30 | 4 | -1% (0.1114 ±0.0038) | +1% (0.0415 ±0.0014) | +1% (0.0231 ±0.0006) | +5% (0.0179 ±0.0006) | +1% (0.5486 ±0.0064) |
| dropout_50 | 4 | -5% (0.1069 ±0.0013) | -2% (0.0404 ±0.0002) | -4% (0.0218 ±0.0010) | +3% (0.0175 ±0.0004) | -4% (0.5259 ±0.0079) |

**Block normalization kind (baseline: RMSNorm)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| norm_batch | 4 | -1% (0.1112 ±0.0014) | +1% (0.0414 ±0.0005) | +1% (0.0230 ±0.0009) | +3% (0.0175 ±0.0007) | -1% (0.5392 ±0.0285) |
| norm_layer | 4 | -1% (0.1112 ±0.0019) | +0% (0.0411 ±0.0003) | +2% (0.0232 ±0.0006) | +1% (0.0172 ±0.0004) | -2% (0.5318 ±0.0089) |

**Residual normalization place (baseline: pre-norm)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| norm_post | 4 | +0% (0.1123 ±0.0017) | +1% (0.0414 ±0.0005) | -1% (0.0226 ±0.0007) | +1% (0.0172 ±0.0004) | -1% (0.5382 ±0.0092) |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| norm_post_layer | 4 | -0% (0.1115 ±0.0026) | -0% (0.0410 ±0.0008) | -4% (0.0220 ±0.0006) | -1% (0.0168 ±0.0005) | -0% (0.5448 ±0.0194) |

**Input and final normalization (baseline: input LayerNorm + final LayerNorm)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| norm_no_input | 4 | +3% (0.1158 ±0.0014) | +3% (0.0424 ±0.0008) | -2% (0.0223 ±0.0010) | +1% (0.0171 ±0.0010) | -9% (0.4954 ±0.0308) |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| norm_all_rms | 4 | -0% (0.1116 ±0.0025) | -0% (0.0410 ±0.0007) | -4% (0.0219 ±0.0017) | -1% (0.0168 ±0.0008) | -2% (0.5316 ±0.0359) |
| norm_no_final | 4 | -10% (0.1008 ±0.0007) | -9% (0.0376 ±0.0004) | -13% (0.0199 ±0.0009) | -8% (0.0157 ±0.0002) | +9% (0.5951 ±0.0233) |

**BOS token (baseline: no BOS token)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 4 | 0.1120 ±0.0018 | 0.0411 ±0.0004 | 0.0228 ±0.0011 | 0.0170 ±0.0006 | 0.5450 ±0.0266 |
| bos | 4 | -2% (0.1101 ±0.0015) | +1% (0.0415 ±0.0008) | +0% (0.0228 ±0.0008) | +5% (0.0179 ±0.0009) | +1% (0.5530 ±0.0269) |

**Shared attention window under cosine warmup (baseline: full attention)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| window_50 | 4 | +0% (0.1117 ±0.0008) | +0% (0.0421 ±0.0004) | -1% (0.0225 ±0.0007) | -1% (0.0179 ±0.0007) | +0% (0.4459 ±0.0040) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
| window_25 | 4 | +0% (0.1116 ±0.0007) | +0% (0.0420 ±0.0003) | +3% (0.0234 ±0.0008) | +1% (0.0182 ±0.0002) | +0% (0.4459 ±0.0040) |

**Embedding and model dimension under cosine warmup (baseline: dim=128)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
| cosine_dim_64 | 4 | -0% (0.1111 ±0.0016) | -3% (0.0408 ±0.0006) | -5% (0.0217 ±0.0017) | -7% (0.0169 ±0.0010) | -33% (0.2993 ±0.0122) |
| cosine_dim_256 | 4 | -2% (0.1093 ±0.0024) | -3% (0.0406 ±0.0007) | -5% (0.0216 ±0.0008) | -6% (0.0170 ±0.0003) | +31% (0.5853 ±0.0067) |
| cosine_dim_32 | 4 | -6% (0.1045 ±0.0029) | -8% (0.0388 ±0.0011) | -6% (0.0215 ±0.0010) | -10% (0.0163 ±0.0006) | -53% (0.2094 ±0.0017) |
| cosine_dim_16 | 4 | -20% (0.0898 ±0.0020) | -19% (0.0342 ±0.0011) | -17% (0.0189 ±0.0010) | -15% (0.0154 ±0.0009) | -70% (0.1356 ±0.0039) |

**Depth under cosine warmup (baseline: depth=2)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_depth_4 | 4 | +3% (0.1151 ±0.0015) | +2% (0.0427 ±0.0005) | +1% (0.0231 ±0.0005) | +0% (0.0181 ±0.0004) | +4% (0.4646 ±0.0076) |
| cosine_depth_1 | 4 | +0% (0.1119 ±0.0032) | -0% (0.0419 ±0.0008) | +0% (0.0229 ±0.0002) | -2% (0.0178 ±0.0002) | -5% (0.4233 ±0.0065) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |

**Sequence length under cosine warmup (baseline: max_seq_len=100)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_seq_50 | 4 | +2% (0.1140 ±0.0019) | +1% (0.0424 ±0.0008) | +0% (0.0228 ±0.0010) | -2% (0.0177 ±0.0008) | +6% (0.4711 ±0.0089) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
| cosine_seq_200 | 4 | -3% (0.1082 ±0.0021) | -3% (0.0407 ±0.0008) | -0% (0.0227 ±0.0015) | -3% (0.0176 ±0.0007) | -5% (0.4249 ±0.0068) |

**Number of attention heads under cosine warmup (baseline: heads=4, kv_heads=2)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_heads_8 | 4 | +1% (0.1123 ±0.0018) | +1% (0.0424 ±0.0005) | +2% (0.0232 ±0.0012) | +1% (0.0183 ±0.0008) | -0% (0.4435 ±0.0080) |
| cosine_heads_2 | 4 | +1% (0.1123 ±0.0013) | +0% (0.0421 ±0.0006) | +1% (0.0230 ±0.0019) | +1% (0.0182 ±0.0010) | -0% (0.4441 ±0.0043) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |

**Grouped-query attention under cosine warmup (baseline: heads=4, kv_heads=2)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_heads_mha | 4 | +2% (0.1135 ±0.0018) | +0% (0.0421 ±0.0006) | -2% (0.0224 ±0.0005) | -2% (0.0178 ±0.0005) | -0% (0.4435 ±0.0130) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |

**FFN ratio under cosine warmup (baseline: ffn_dim=256 (2x model dim))**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_ffn_512 | 4 | +1% (0.1130 ±0.0027) | +1% (0.0423 ±0.0006) | +2% (0.0232 ±0.0010) | +0% (0.0181 ±0.0005) | -0% (0.4446 ±0.0061) |
| cosine_ffn_128 | 4 | +0% (0.1118 ±0.0020) | -0% (0.0418 ±0.0005) | -2% (0.0223 ±0.0009) | -2% (0.0177 ±0.0004) | -1% (0.4407 ±0.0022) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |

**Dropout under cosine warmup (baseline: dropout=0.1)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
| cosine_dropout_30 | 4 | -1% (0.1100 ±0.0009) | -2% (0.0410 ±0.0003) | -2% (0.0224 ±0.0006) | -2% (0.0177 ±0.0003) | -1% (0.4427 ±0.0035) |
| cosine_dropout_50 | 4 | -6% (0.1048 ±0.0011) | -6% (0.0395 ±0.0005) | -7% (0.0213 ±0.0014) | -6% (0.0170 ±0.0007) | -6% (0.4170 ±0.0101) |

**Block normalization kind under cosine warmup (baseline: RMSNorm)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_norm_layer | 4 | +0% (0.1119 ±0.0006) | +0% (0.0420 ±0.0003) | +0% (0.0228 ±0.0007) | +0% (0.0181 ±0.0005) | -0% (0.4433 ±0.0043) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
| cosine_norm_batch | 4 | -1% (0.1107 ±0.0016) | -1% (0.0414 ±0.0005) | -5% (0.0217 ±0.0011) | -3% (0.0176 ±0.0006) | -5% (0.4238 ±0.0056) |

**Residual normalization place under cosine warmup (baseline: pre-norm)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_norm_post_layer | 4 | +0% (0.1119 ±0.0018) | +0% (0.0421 ±0.0002) | -4% (0.0218 ±0.0002) | -2% (0.0178 ±0.0004) | +2% (0.4533 ±0.0044) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
| cosine_norm_post | 4 | -0% (0.1112 ±0.0015) | +0% (0.0420 ±0.0006) | -2% (0.0223 ±0.0008) | -2% (0.0178 ±0.0005) | +1% (0.4511 ±0.0039) |

**Input and final normalization under cosine warmup (baseline: input LayerNorm + final LayerNorm)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_norm_no_input | 4 | +6% (0.1182 ±0.0029) | +3% (0.0432 ±0.0005) | +2% (0.0232 ±0.0004) | -2% (0.0178 ±0.0005) | -9% (0.4054 ±0.0106) |
| cosine_norm_all_rms | 4 | +1% (0.1124 ±0.0027) | +0% (0.0421 ±0.0003) | +0% (0.0229 ±0.0010) | -2% (0.0178 ±0.0006) | -1% (0.4425 ±0.0049) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
| cosine_norm_no_final | 4 | -13% (0.0968 ±0.0020) | -12% (0.0368 ±0.0006) | -12% (0.0201 ±0.0007) | -10% (0.0162 ±0.0002) | +46% (0.6498 ±0.0397) |

**BOS token under cosine warmup (baseline: no BOS token)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_bos | 4 | +1% (0.1124 ±0.0030) | -0% (0.0418 ±0.0012) | +2% (0.0232 ±0.0014) | -2% (0.0178 ±0.0009) | -2% (0.4368 ±0.0076) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |

## rq9 — timestamp-delta embeddings

Learned binned deltas concatenated before a dense encoder are best at +5.4%
(0.1177). Reverse timestamp RoPE is +2.4% and inside noise. Plain and log
additions are both lower and high-variance, so bounded bins plus learned fusion
are preferable to adding raw-scale features.

**Timestamp delta and timestamp RoPE under cosine warmup (baseline: no timestamp-delta feature)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| time_bins_concat | 4 | +5% (0.1177 ±0.0016) | +4% (0.0435 ±0.0007) | +3% (0.0234 ±0.0010) | +1% (0.0183 ±0.0007) | -30% (0.3127 ±0.0094) |
| time_rope_reverse | 4 | +3% (0.1144 ±0.0013) | +2% (0.0428 ±0.0007) | +0% (0.0228 ±0.0004) | -1% (0.0180 ±0.0002) | +0% (0.4451 ±0.0047) |
| time_rope | 4 | +2% (0.1134 ±0.0013) | +2% (0.0428 ±0.0006) | +1% (0.0230 ±0.0013) | +1% (0.0183 ±0.0007) | -0% (0.4437 ±0.0035) |
| time_bins_log_rope | 4 | +1% (0.1125 ±0.0013) | -1% (0.0417 ±0.0007) | +3% (0.0235 ±0.0011) | -2% (0.0178 ±0.0011) | -10% (0.4002 ±0.0108) |
| time_log_rope_reverse | 4 | +1% (0.1124 ±0.0024) | +0% (0.0421 ±0.0005) | -0% (0.0227 ±0.0012) | -1% (0.0180 ±0.0008) | -0% (0.4429 ±0.0037) |
| time_log_concat | 4 | +1% (0.1124 ±0.0010) | -2% (0.0413 ±0.0007) | -7% (0.0212 ±0.0010) | -8% (0.0167 ±0.0009) | -31% (0.3055 ±0.0060) |
| time_bins_add | 4 | +1% (0.1123 ±0.0034) | -0% (0.0418 ±0.0008) | +2% (0.0232 ±0.0006) | -2% (0.0178 ±0.0006) | -10% (0.4022 ±0.0099) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
| time_log_rope | 4 | -0% (0.1114 ±0.0002) | +0% (0.0420 ±0.0005) | -1% (0.0225 ±0.0010) | -1% (0.0180 ±0.0008) | +0% (0.4452 ±0.0036) |
| time_plain_add | 4 | -6% (0.1044 ±0.0079) | -8% (0.0386 ±0.0033) | -7% (0.0211 ±0.0019) | -10% (0.0163 ±0.0015) | -58% (0.1864 ±0.0291) |
| time_log_add | 4 | -10% (0.1004 ±0.0083) | -12% (0.0369 ±0.0033) | -11% (0.0202 ±0.0017) | -13% (0.0158 ±0.0016) | -62% (0.1712 ±0.0225) |

## rq10 — per-layer embeddings (Gemma-style)

No. A fresh item table before every layer is 1.2% higher, inside noise, while
tripling the dominant embedding parameters. The extra tables add cost without a
resolved quality gain.

**Per-layer item embeddings under cosine warmup (baseline: one shared item embedding)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| per_layer_embeddings | 4 | +1% (0.1130 ±0.0021) | +1% (0.0423 ±0.0007) | +5% (0.0239 ±0.0015) | +2% (0.0184 ±0.0006) | -3% (0.4323 ±0.0036) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |

## rq11 — online logQ, offline logQ, random, or uncorrected in-batch negatives?

A controlled comparison with 512 negatives and identical architecture,
schedule, learning rates, data, and seeds. Offline logQ uses the exact
positive-item distribution of the cached training windows; random draws
uniformly from known catalog items. Offline logQ is best at +3.7% versus
online, while uniform random is +1.3% and inside noise. Removing logQ costs
12.7%, showing that proposal correction matters; the exact offline distribution
also avoids the streaming estimator's early-run error.

**Negative sampling and logQ under cosine warmup (baseline: 512 in-batch negatives with online logQ)**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| neg_offline_logq | 4 | +4% (0.1220 ±0.0020) | +4% (0.0459 ±0.0004) | +3% (0.0252 ±0.0014) | +4% (0.0198 ±0.0005) | -11% (0.3947 ±0.0042) |
| neg_random | 4 | +1% (0.1192 ±0.0007) | +1% (0.0445 ±0.0004) | +4% (0.0254 ±0.0013) | +1% (0.0192 ±0.0004) | -11% (0.3933 ±0.0027) |
| neg_online_logq | 4 | 0.1177 ±0.0026 | 0.0441 ±0.0008 | 0.0245 ±0.0010 | 0.0190 ±0.0004 | 0.4427 ±0.0054 |
| neg_in_batch_no_logq | 4 | -13% (0.1028 ±0.0019) | -13% (0.0382 ±0.0005) | -17% (0.0203 ±0.0008) | -15% (0.0162 ±0.0006) | +61% (0.7125 ±0.0022) |

## rq2 — best combination for metrics

The best tested transformer combination is embedding LR 0.02 plus the reverse-
position arm at 0.1206 ±0.0010, +8.0% versus the scheduled reference and +2.4%
versus embedding LR 0.02 alone. The timestamp cross does not add to the rate
gain. Offline logQ is a separate rq11 axis and is not duplicated into this
transformer-combination table.

**Final metric candidates**

| variant | runs | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| combo_embedding_position | 4 | +8% (0.1206 ±0.0010) | +6% (0.0447 ±0.0006) | +5% (0.0240 ±0.0010) | +4% (0.0188 ±0.0006) | -2% (0.4341 ±0.0036) |
| embedding_lr_2e2 | 4 | +5% (0.1177 ±0.0026) | +5% (0.0441 ±0.0008) | +7% (0.0245 ±0.0010) | +5% (0.0190 ±0.0004) | -1% (0.4427 ±0.0054) |
| combo_lr_rates | 4 | +3% (0.1152 ±0.0013) | +2% (0.0430 ±0.0004) | +1% (0.0231 ±0.0006) | +1% (0.0182 ±0.0005) | -4% (0.4282 ±0.0081) |
| combo_embedding_time | 4 | +3% (0.1151 ±0.0025) | +4% (0.0435 ±0.0009) | +12% (0.0256 ±0.0010) | +6% (0.0192 ±0.0005) | -7% (0.4125 ±0.0053) |
| lr_cosine_warmup | 4 | 0.1116 ±0.0017 | 0.0420 ±0.0005 | 0.0228 ±0.0009 | 0.0181 ±0.0006 | 0.4451 ±0.0032 |
