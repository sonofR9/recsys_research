# G2 eSASRec on native Yambda-50M

This experiment evaluates the [eSASRec paper](https://arxiv.org/abs/2508.06450), its LiGR and loss components, and the official [RecTools](https://github.com/MTSWebServices/RecTools) recipe on native Yambda-50M. The size-matched empirical bands come from ten restored checkpoints of the unchanged control: 0.003 Recall@100, 0.002 NDCG@100, and 0.06 Coverage@100. They are practical resolution thresholds, not confidence intervals.

## RQ1: What are official and local eSASRec's metrics?

- **Control:** the size-calibrated G1 structure with sampled softmax.
- **Local eSASRec:** shifted-sequence training with local LiGR blocks, 256 uniform negatives, and sampled softmax; fixed-tensor forward, loss, and gradient parity passes against RecTools.
- **Official RecTools:** the three-seed RecTools 0.19.0 LiGR/sampled-softmax mean on the same split, full catalog, seen-item policy, and 3,414-user denominator.

| variant | layer | loss | FFN width | gBCE t | params | epoch s | train s | wall s | targets/s | peak GB | recall@100 | ndcg@100 | coverage@100 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: |
| control | G1 control | sampled softmax | 192 | — | 2222956.000 | 1.383 | 28.527 | 29.886 | 438511.859 | 1.741 | +0.000% (0.115) | +0.000% (0.043) | +0.000% (0.691) |
| **local eSASRec** | LiGR | sampled softmax | 1024 | — | 10876416.000 | 2.359 | 38.685 | 40.544 | 257052.624 | 6.608 | <span style="color: green">+18.262% (0.136)</span> | <span style="color: green">+22.138% (0.053)</span> | <span style="color: red">-30.178% (0.483)</span> |
| official RecTools | RecTools LiGR | sampled softmax | — | — | — | — | — | 454.131 | — | — | <span style="color: red">-3.968% (0.111)</span> | <span style="color: green">+9.445% (0.047)</span> | <span style="color: red">-74.376% (0.177)</span> |

Local eSASRec improves Recall@100 by 0.0211 and NDCG@100 by 0.0096, both beyond their bands, while reducing Coverage@100 by 0.2086. The official and local metric rows are descriptive rather than a metric-reproduction test because RecTools truncates each training session to its last 100 events while local shifted training consumes the full pre-cutoff sequence stream. Exact tensor parity establishes equivalence of the local building blocks under matched inputs. Local LiGR with sampled softmax is therefore the eligible eSASRec recipe and is promoted over the G1-derived control.

## RQ2: What does each pluggable eSASRec component buy?

- **Standard sampled softmax:** official-style SASRec blocks with the paper-width ReLU FFN and sampled softmax.
- **Standard gBCE:** the same standard block and capacity with tuned generalized BCE.
- **Matched standard sampled softmax:** a wider standard ReLU FFN matched to the selected LiGR parameter count.
- **Matched standard gBCE:** the same parameter-matched standard block with tuned generalized BCE.
- **LiGR sampled softmax:** gated attention and FFN residuals with a width-1024 SwiGLU, divisible by 32.
- **LiGR gBCE:** the identical LiGR capacity and negative proposal with tuned generalized BCE.

| loss (standard block, FFN width 256) | recall@100 | ndcg@100 | coverage@100 | gBCE t | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| sampled softmax | 0.128 | 0.049 | 0.426 | — | 43.506 | 280689.561 | 6.576 |
| **gBCE** | <span style="color: green">+2.775% (0.131)</span> | +2.422% (0.050) | <span style="color: red">-22.042% (0.332)</span> | 0.342 | 188.204 | 277675.250 | 6.575 |

| loss (parameter-matched SASRec, FFN width 1792) | recall@100 | ndcg@100 | coverage@100 | gBCE t | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| sampled softmax | 0.125 | 0.049 | 0.408 | — | 33.071 | 272861.339 | 6.608 |
| **gBCE** | <span style="color: green">+5.582% (0.132)</span> | +2.804% (0.051) | -10.183% (0.366) | 0.663 | 202.806 | 272236.531 | 6.608 |

| loss (LiGR, FFN width 1024) | recall@100 | ndcg@100 | coverage@100 | gBCE t | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| **sampled softmax** | 0.136 | 0.053 | 0.483 | — | 40.544 | 257052.624 | 6.608 |
| gBCE | <span style="color: red">-33.891% (0.090)</span> | <span style="color: red">-33.399% (0.035)</span> | <span style="color: green">+50.497% (0.726)</span> | 0.464 | 57.387 | 257361.247 | 6.608 |

| FFN width (standard block, sampled softmax) | recall@100 | ndcg@100 | coverage@100 | params | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| **256** | 0.128 | 0.049 | 0.426 | 9304064.000 | 43.506 | 280689.561 | 6.576 |
| 1792 | <span style="color: red">-2.462% (0.125)</span> | +0.125% (0.049) | -4.248% (0.408) | 10880000.000 | 33.071 | 272861.339 | 6.608 |

| FFN width (standard block, gBCE) | recall@100 | ndcg@100 | coverage@100 | params | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| 256 | 0.131 | 0.050 | 0.332 | 9304064.000 | 188.204 | 277675.250 | 6.575 |
| **1792** | +0.202% (0.132) | +0.498% (0.051) | +10.318% (0.366) | 10880000.000 | 202.806 | 272236.531 | 6.608 |

| block (sampled softmax, parameter-matched) | recall@100 | ndcg@100 | coverage@100 | FFN width | params | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| parameter-matched SASRec | 0.125 | 0.049 | 0.408 | 1792 | 10880000.000 | 33.071 | 272861.339 | 6.608 |
| **LiGR** | <span style="color: green">+9.397% (0.136)</span> | <span style="color: green">+6.968% (0.053)</span> | <span style="color: green">+18.295% (0.483)</span> | 1024 | 10876416.000 | 40.544 | 257052.624 | 6.608 |

| block (gBCE, parameter-matched) | recall@100 | ndcg@100 | coverage@100 | FFN width | params | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| **parameter-matched SASRec** | 0.132 | 0.051 | 0.366 | 1792 | 10880000.000 | 202.806 | 272236.531 | 6.608 |
| LiGR | <span style="color: red">-31.502% (0.090)</span> | <span style="color: red">-30.702% (0.035)</span> | <span style="color: green">+98.213% (0.726)</span> | 1024 | 10876416.000 | 57.387 | 257361.247 | 6.608 |

LiGR beats the parameter-matched standard block under sampled softmax by 0.0117 Recall@100 and 0.0034 NDCG@100, so its gain is not explained by parameter count; the matched stacks differ by only 0.033% in parameters. gBCE helps both standard capacities but sharply reverses under LiGR, losing 0.0462 Recall@100 while increasing coverage by 0.2437. The two losses use the same 256 uniform negatives, and parity, calibration, initialization, masks, parameter counts, convergence, and restored checkpoints all pass their checks. The reversal is therefore treated as a block-by-loss interaction under this protocol, and only LiGR with sampled softmax is promoted.

## RQ3: Does mixed sampling improve coverage without a recall loss?

- **Uniform eSASRec:** the selected LiGR/sampled-softmax recipe with all 256 negatives sampled uniformly.
- **Mixed 0.6 without logQ:** the paper-anchor mixture with 60% uniform and 40% in-batch negatives.
- **Mixed 0.6 with Yi-2019:** the same mixture with fully corrected proposal probabilities.
- **Best tuned mixed trial:** the band-aware best mixed candidate, using 43.6% uniform negatives and Yi-2019 correction.

| variant | uniform fraction | logQ | recall@100 | ndcg@100 | coverage@100 |
| :--- | ---: | :--- | :---: | :---: | ---: |
| **uniform eSASRec** | 1.0 | none | +0.000% (0.136) | +0.000% (0.053) | +0.000% (0.483) |
| mixed sampler | 0.6 | none | <span style="color: red">-13.174% (0.118)</span> | <span style="color: red">-14.134% (0.045)</span> | <span style="color: green">+20.104% (0.580)</span> |
| mixed sampler | 0.6 | Yi-2019 | <span style="color: red">-2.258% (0.133)</span> | -2.503% (0.051) | +4.957% (0.507) |
| best tuned mixed trial | 0.4360874991648083 | Yi-2019 | +0.645% (0.137) | <span style="color: green">+3.932% (0.055)</span> | -8.652% (0.441) |

At the fixed 0.6 mixture, Yi-2019 correction recovers 0.0149 Recall@100 and 0.0061 NDCG@100 relative to no correction, but its remaining recall loss exceeds the 0.003 band and its coverage gain is below 0.06. The best tuned mixed trial keeps recall tied and improves NDCG, but coverage falls rather than improves. No mixed candidate satisfies the pre-approved requirement of non-inferior recall plus a coverage gain beyond its band. Uniform LiGR sampled softmax remains selected.

## Aggregated improvement

- **Baseline:** the recalibrated G1 control.
- **Aggregate:** the selected existing atomic LiGR/sampled-softmax bundle; no duplicate closing run is needed because its standalone and aggregate identities are identical.

| configuration | layer | loss | FFN width | params | wall s | recall@100 | ndcg@100 | coverage@100 |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | :---: | ---: |
| baseline | G1 control | sampled softmax | 192 | 2222956.000 | 29.886 | +0.000% (0.115) | +0.000% (0.043) | +0.000% (0.691) |
| **aggregate** | LiGR | sampled softmax | 1024 | 10876416.000 | 40.544 | <span style="color: green">+18.262% (0.136)</span> | <span style="color: green">+22.138% (0.053)</span> | <span style="color: red">-30.178% (0.483)</span> |

| candidate | qualification | selection | rationale |
| :--- | :--- | :--- | :--- |
| official SASRec block with sampled softmax | qualified | omitted | lower under the band-aware Recall@100, NDCG@100, and wall-time order |
| official SASRec block with gBCE | qualified | omitted | lower under the band-aware Recall@100, NDCG@100, and wall-time order |
| parameter-matched SASRec with sampled softmax | qualified | omitted | lower under the band-aware Recall@100, NDCG@100, and wall-time order |
| parameter-matched SASRec with gBCE | qualified | omitted | lower under the band-aware Recall@100, NDCG@100, and wall-time order |
| LiGR with sampled softmax | qualified | selected | Recall@100 improved beyond its size-matched band |
| LiGR with gBCE | not qualified | omitted | Recall@100 did not qualify for promotion |
| LiGR with mixed sampling | not qualified | omitted | no candidate met the mixed-sampler eligibility rule |

| deterministic selected-recipe reproduction | p50 latency (ms) | p95 latency (ms) | queries/s |
| :--- | ---: | ---: | ---: |
| LiGR with sampled softmax | 3.176 | 3.415 | 80603.214 |

| metric | baseline | aggregate | point gain | percent gain | standalone sum | interaction gap | size-matched band | resolution |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| recall@100 | 0.115 | 0.136 | +0.021 | +18.262% | +0.021 | +0.000 | 0.003 | unresolved |
| ndcg@100 | 0.043 | 0.053 | +0.010 | +22.138% | +0.010 | +0.000 | 0.002 | unresolved |
| coverage@100 | 0.691 | 0.483 | -0.209 | -30.178% | -0.209 | +0.000 | 0.060 | unresolved |

The selected atomic bundle improves Recall@100 and NDCG@100 beyond their native-50M bands, at the cost of a resolved Coverage@100 decrease. Its deterministic reproduction on one A100 measures 3.176 ms p50, 3.415 ms p95, and 80,603 queries/s for a fixed 256-query batch against the full catalog; these are reproduced-recipe weights, not the unavailable original weight object. Because the aggregate reuses one indivisible trained bundle, its standalone sum equals its aggregate gain and the interaction gap is exactly zero and descriptively unresolved. LiGR with sampled softmax is promoted as the G2 future baseline.
