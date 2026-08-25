# G1 — Yambda-50M results

## RQ1 — Does μTransfer work?

| dataset | batch size | embedding LR | deep LR | best/stopped epoch | epoch cap | recall@100 | ndcg@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50M | 1280 | 0.001 | 0.002 | 56/59 | 60 | 0.100 | 0.037 |
| 500M | 1280 | 0.001 | 0.002 | 20/23 | 40 | 0.127 | 0.048 |

| width | common LR | 50M-local LR | common recall@100 | local recall@100 | recall regret |
| --- | --- | --- | --- | --- | --- |
| 16 | 0.032/0.012 | 0.032/0.024 | 0.071 | 0.076 | 6% |
| 32 | 0.032/0.012 | 0.032/0.012 | 0.076 | 0.076 | same run |
| 64 | 0.032/0.012 | 0.032/0.012 | 0.068 | 0.068 | same run |
| 128 | 0.032/0.012 | 0.032/0.012 | 0.065 | 0.065 | same run |
| 256 | 0.032/0.012 | 0.008/0.012 | 0.043 | 0.054 | 20% |

## RQ4 — Does SwiGLU help?

| proxy-selected FFN family | selected width | recall@100 | ndcg@100 | coverage@100 |
| --- | --- | --- | --- | --- |
| GELU | 171 | 0.077 | 0.029 | 0.699 |
| **SwiGLU** | 32 | +2% (0.079) | 0% (0.029) | <span style="color: red">-49% (0.357)</span> |

## RQ6 — Does learning-rate warmup help?

| schedule | no-warmup LR | warmup LR | no-warmup recall@100 | warmup=5% recall@100 | no-warmup ndcg@100 | warmup=5% ndcg@100 |
| --- | --- | --- | --- | --- | --- | --- |
| constant | 0.016/0.006 | 0.008/0.012 | 0.044 | 0% (0.044) | 0.015 | +2% (0.016) |
| cosine | 0.032/0.012 | 0.032/0.006 | 0.063 | <span style="color: red">-27% (0.046)</span> | 0.023 | <span style="color: red">-30% (0.016)</span> |
| **inverse sqrt** | **0.032/0.006** | **0.008/0.012** | **0.045** | **<span style="color: green">+64% (0.073)</span>** | **0.016** | **<span style="color: green">+66% (0.026)</span>** |

## RQ7 — Which position encoding works best: RoPE, ALiBi, learned positions, or combinations?

No reader table is generated from Yambda-50M. These artifacts are correctness diagnostics only and are ineligible for treatment selection or reader claims.

## RQ8 — How do scaling and architecture choices affect metrics?

| dimension | recall@100 | ndcg@100 |
| --- | --- | --- |
| 64 | 0.068 | 0.024 |
| 128 | -3% (0.065) | +2% (0.025) |
| **16** | <span style="color: green">+13% (0.076)</span> | <span style="color: green">+18% (0.029)</span> |
| 256 | <span style="color: red">-20% (0.054)</span> | <span style="color: red">-23% (0.019)</span> |
| 32 | <span style="color: green">+12% (0.076)</span> | <span style="color: green">+17% (0.028)</span> |

| depth | recall@100 | ndcg@100 |
| --- | --- | --- |
| 2 layers | 0.068 | 0.024 |
| **1 layers** | <span style="color: green">+25% (0.085)</span> | <span style="color: green">+36% (0.033)</span> |
| 4 layers | <span style="color: red">-5% (0.064)</span> | -3% (0.023) |

| mha head count | recall@100 | ndcg@100 |
| --- | --- | --- |
| **2Q/2KV** | 0.080 | 0.029 |
| 1Q/1KV | <span style="color: red">-8% (0.073)</span> | <span style="color: red">-11% (0.026)</span> |
| 4Q/4KV | <span style="color: red">-30% (0.056)</span> | <span style="color: red">-37% (0.019)</span> |
| 8Q/8KV | <span style="color: red">-43% (0.045)</span> | <span style="color: red">-47% (0.016)</span> |

| attention grouping | recall@100 | ndcg@100 |
| --- | --- | --- |
| **MHA 2Q/2KV** | 0.080 | 0.029 |
| GQA 2Q/1KV | <span style="color: red">-15% (0.068)</span> | <span style="color: red">-18% (0.024)</span> |

| block normalization kind | recall@100 | ndcg@100 |
| --- | --- | --- |
| LayerNorm | 0.068 | 0.024 |
| **BatchNorm** | <span style="color: green">+18% (0.080)</span> | <span style="color: green">+19% (0.029)</span> |
| RMSNorm | -3% (0.066) | -1% (0.024) |

| residual normalization placement | recall@100 | ndcg@100 |
| --- | --- | --- |
| pre-LayerNorm | 0.068 | 0.024 |
| **post-LayerNorm** | <span style="color: green">+24% (0.084)</span> | <span style="color: green">+33% (0.032)</span> |

| input and final normalization | recall@100 | ndcg@100 |
| --- | --- | --- |
| no input + final LayerNorm | 0.068 | 0.024 |
| **input + final RMSNorm** | <span style="color: green">+22% (0.082)</span> | <span style="color: green">+20% (0.029)</span> |
| input LayerNorm + final LayerNorm | <span style="color: red">-35% (0.044)</span> | <span style="color: red">-35% (0.016)</span> |
| input RMSNorm + final LayerNorm | <span style="color: red">-29% (0.048)</span> | <span style="color: red">-31% (0.017)</span> |
| no input or final norm | <span style="color: green">+6% (0.072)</span> | +3% (0.025) |

| attention window | recall@100 | ndcg@100 |
| --- | --- | --- |
| 50 | 0.068 | 0.024 |
| 10 | <span style="color: red">-34% (0.045)</span> | <span style="color: red">-35% (0.016)</span> |
| 100 | -3% (0.066) | -2% (0.024) |
| 25 | <span style="color: red">-5% (0.064)</span> | -2% (0.024) |
| 75 | 0% (0.068) | 0% (0.024) |
| **full** | 0% (0.068) | -2% (0.024) |

| dropout | recall@100 | ndcg@100 |
| --- | --- | --- |
| 0.1 | 0.068 | 0.024 |
| 0.0 | 0% (0.068) | +2% (0.025) |
| 0.2 | <span style="color: red">-9% (0.062)</span> | <span style="color: red">-8% (0.022)</span> |
| **0.3** | <span style="color: green">+19% (0.081)</span> | <span style="color: green">+26% (0.030)</span> |
| 0.05 | -2% (0.067) | -2% (0.024) |
| 0.5 | <span style="color: green">+16% (0.079)</span> | <span style="color: green">+20% (0.029)</span> |

| bos | recall@100 | ndcg@100 |
| --- | --- | --- |
| disabled | 0.068 | 0.024 |
| **enabled** | <span style="color: green">+15% (0.078)</span> | <span style="color: green">+15% (0.028)</span> |

## RQ9 — Does a timestamp-delta representation improve metrics?

| time representation | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- |
| no time feature | 0.081 | 0.030 | 0.017 | 0.014 | 0.749 |
| 8 log-spaced bins, add | <span style="color: red">-10% (0.073)</span> | <span style="color: red">-15% (0.026)</span> | <span style="color: red">-30% (0.012)</span> | <span style="color: red">-28% (0.010)</span> | <span style="color: red">-65% (0.264)</span> |
| 16 log-spaced bins, add | <span style="color: red">-17% (0.068)</span> | <span style="color: red">-20% (0.024)</span> | <span style="color: red">-29% (0.012)</span> | <span style="color: red">-29% (0.010)</span> | <span style="color: red">-75% (0.186)</span> |
| 32 log-spaced bins, add | <span style="color: red">-9% (0.074)</span> | <span style="color: red">-12% (0.027)</span> | <span style="color: red">-23% (0.013)</span> | <span style="color: red">-21% (0.011)</span> | <span style="color: red">-67% (0.249)</span> |
| 64 log-spaced bins, add | +2% (0.083) | -2% (0.030) | -17% (0.014) | <span style="color: red">-16% (0.011)</span> | <span style="color: red">-48% (0.389)</span> |
| 32 bins + raw reverse RoPE | <span style="color: red">-11% (0.072)</span> | <span style="color: red">-17% (0.025)</span> | <span style="color: red">-29% (0.012)</span> | <span style="color: red">-31% (0.009)</span> | <span style="color: red">-74% (0.192)</span> |
| 32 bins + log forward RoPE | <span style="color: red">-12% (0.072)</span> | <span style="color: red">-16% (0.025)</span> | <span style="color: red">-21% (0.014)</span> | <span style="color: red">-25% (0.010)</span> | <span style="color: red">-76% (0.176)</span> |
| 32 bins, concatenate-and-project | <span style="color: red">-45% (0.045)</span> | <span style="color: red">-47% (0.016)</span> | <span style="color: red">-52% (0.008)</span> | <span style="color: red">-49% (0.007)</span> | <span style="color: red">-98% (0.016)</span> |
| clipped linear delta, add | <span style="color: red">-7% (0.075)</span> | <span style="color: red">-11% (0.027)</span> | <span style="color: red">-26% (0.013)</span> | <span style="color: red">-24% (0.010)</span> | <span style="color: red">-37% (0.471)</span> |
| log delta, add | <span style="color: red">-5% (0.077)</span> | <span style="color: red">-10% (0.027)</span> | -17% (0.014) | <span style="color: red">-19% (0.011)</span> | -1% (0.739) |
| log delta, concatenate-and-project | <span style="color: red">-17% (0.067)</span> | <span style="color: red">-18% (0.025)</span> | <span style="color: red">-24% (0.013)</span> | <span style="color: red">-21% (0.011)</span> | <span style="color: red">-86% (0.107)</span> |
| raw elapsed-time RoPE, forward | +2% (0.083) | +2% (0.031) | -7% (0.016) | -3% (0.013) | +7% (0.803) |
| raw elapsed-time RoPE, reverse | +1% (0.082) | +1% (0.031) | -5% (0.016) | -1% (0.013) | +7% (0.803) |
| **log elapsed-time RoPE, forward** | +3% (0.084) | +1% (0.031) | -4% (0.017) | -3% (0.013) | +3% (0.773) |
| log elapsed-time RoPE, reverse | <span style="color: red">-21% (0.064)</span> | <span style="color: red">-21% (0.024)</span> | <span style="color: red">-31% (0.012)</span> | <span style="color: red">-26% (0.010)</span> | <span style="color: red">-82% (0.137)</span> |

## RQ10 — Do separate item embeddings at every transformer layer help?

| item embeddings | recall@100 | ndcg@100 |
| --- | --- | --- |
| **shared table** | 0.068 | 0.024 |
| per-layer tables | <span style="color: red">-7% (0.063)</span> | <span style="color: red">-4% (0.023)</span> |
