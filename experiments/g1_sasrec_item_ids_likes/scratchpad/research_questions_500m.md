# G1 — Yambda-500M results

## RQ1 — Does μTransfer work?

| dataset | batch size | embedding LR | deep LR | best/stopped epoch | epoch cap | recall@100 | ndcg@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50M | 1280 | 0.001 | 0.002 | 56/59 | 60 | 0.100 | 0.037 |
| 500M | 1280 | 0.001 | 0.002 | 20/23 | 40 | 0.127 | 0.048 |

| width | common LR | 50M-local LR | recall@100 vs local | local recall@100 | ndcg@100 vs local | local ndcg@100 |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.032/0.012 | 0.032/0.024 | 0% (0.120) | 0.120 | 0% (0.045) | 0.045 |
| 32 | 0.032/0.012 | 0.032/0.012 | same run | 0.131 | same run | 0.050 |
| 64 | 0.032/0.012 | 0.032/0.012 | same run | 0.135 | same run | 0.052 |
| 128 | 0.032/0.012 | 0.032/0.012 | same run | 0.134 | same run | 0.051 |
| 256 | 0.032/0.012 | 0.008/0.012 | <span style="color: green">+5% (0.134)</span> | 0.128 | <span style="color: green">+6% (0.051)</span> | 0.048 |

## RQ4 — Does SwiGLU help?

| activation | plain FFN recall@100 | gated FFN recall@100 | plain FFN ndcg@100 | gated FFN ndcg@100 |
| --- | --- | --- | --- | --- |
| ReLU → ReGLU | 0.132 | +2% (0.135) | 0.050 | <span style="color: green">+4% (0.052)</span> |
| GELU → GEGLU | 0.134 | +2% (0.137) | 0.051 | <span style="color: green">+3% (0.053)</span> |
| SiLU → SwiGLU | 0.132 | <span style="color: green">+3% (0.137)</span> | 0.050 | <span style="color: green">+3% (0.052)</span> |

| layers | GELU recall@100 | SwiGLU recall@100 | GELU ndcg@100 | SwiGLU ndcg@100 |
| ---: | --- | --- | --- | --- |
| 2 | 0.134 | +2% (0.137) | 0.051 | <span style="color: green">+2% (0.052)</span> |
| 4 | 0.135 | +1% (0.136) | 0.052 | 0% (0.052) |
| 8 | 0.137 | +1% (0.139) | 0.053 | <span style="color: green">+3% (0.054)</span> |

## RQ5 — Which learning-rate scheduler works best?

| scheduler | optimizer groups scheduled | schedule parameter | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | both | — | 0.141 | 0.054 | 0.031 | 0.023 | 0.718 |
| linear | both | — | <span style="color: green">+3% (0.145)</span> | <span style="color: green">+3% (0.056)</span> | +1% (0.031) | <span style="color: green">+5% (0.024)</span> | -12% (0.635) |
| linear | deep only | — | +1% (0.143) | +1% (0.054) | +2% (0.031) | +4% (0.024) | -6% (0.672) |
| cosine | both | — | +1% (0.144) | +1% (0.054) | 0% (0.031) | +2% (0.024) | -8% (0.660) |
| cosine | deep only | — | +1% (0.143) | +1% (0.055) | 0% (0.031) | +3% (0.024) | -7% (0.668) |
| polynomial | both | — | -1% (0.140) | -1% (0.053) | -4% (0.030) | -1% (0.023) | -10% (0.646) |
| polynomial | deep only | — | <span style="color: red">-3% (0.137)</span> | <span style="color: red">-4% (0.052)</span> | -7% (0.028) | <span style="color: red">-6% (0.022)</span> | -1% (0.707) |
| exponential | both | — | -1% (0.140) | -1% (0.053) | -3% (0.030) | 0% (0.023) | -5% (0.681) |
| exponential | deep only | — | -2% (0.138) | <span style="color: red">-2% (0.053)</span> | -3% (0.030) | -1% (0.023) | -7% (0.669) |
| step | both | — | +1% (0.143) | +2% (0.055) | +1% (0.031) | +3% (0.024) | -3% (0.697) |
| step | deep only | — | +1% (0.142) | +1% (0.054) | 0% (0.031) | +2% (0.024) | -6% (0.675) |
| WSD | both | warmup=0.05, cycles=1 | +2% (0.144) | <span style="color: green">+2% (0.055)</span> | -1% (0.030) | +2% (0.024) | -7% (0.666) |
| WSD | deep only | warmup=0.05, cycles=1 | +2% (0.144) | <span style="color: green">+2% (0.055)</span> | +1% (0.031) | +4% (0.024) | -8% (0.657) |
| inverse sqrt | both | timescale=0.05 | -1% (0.139) | <span style="color: red">-2% (0.053)</span> | -4% (0.029) | -3% (0.022) | +4% (0.749) |
| inverse sqrt | deep only | timescale=0.05 | <span style="color: red">-2% (0.138)</span> | <span style="color: red">-3% (0.052)</span> | -3% (0.030) | -4% (0.022) | +2% (0.730) |
| cosine, warmup 5%, 1 cycle | both | warmup=0.05, cycles=1 | +1% (0.143) | +1% (0.054) | -4% (0.030) | +2% (0.023) | -13% (0.623) |
| **cosine, warmup 5%, 1 cycle** | **deep only** | **warmup=0.05, cycles=1** | **<span style="color: green">+3% (0.146)</span>** | **<span style="color: green">+3% (0.055)</span>** | **+4% (0.032)** | **<span style="color: green">+5% (0.024)</span>** | **-10% (0.647)** |
| cosine, warmup 5%, 2 cycles | both | warmup=0.05, cycles=2 | -2% (0.139) | <span style="color: red">-2% (0.052)</span> | -7% (0.028) | -4% (0.022) | -8% (0.658) |
| cosine, warmup 5%, 2 cycles | deep only | warmup=0.05, cycles=2 | 0% (0.141) | 0% (0.054) | -1% (0.030) | +1% (0.023) | -4% (0.686) |
| cosine, warmup 5%, 4 cycles | both | warmup=0.05, cycles=4 | -2% (0.139) | <span style="color: red">-2% (0.053)</span> | -7% (0.029) | -2% (0.023) | -5% (0.680) |
| cosine, warmup 5%, 4 cycles | deep only | warmup=0.05, cycles=4 | -1% (0.139) | <span style="color: red">-2% (0.053)</span> | -3% (0.030) | -2% (0.023) | 0% (0.720) |
| cosine, tuned warmup | both | warmup=0.05, cycles=1 | <span style="color: red">-2% (0.138)</span> | <span style="color: red">-3% (0.052)</span> | -6% (0.029) | -3% (0.022) | -6% (0.673) |
| cosine, tuned warmup | deep only | warmup=0.0209409463814, cycles=1 | +1% (0.143) | 0% (0.054) | -5% (0.029) | -1% (0.023) | <span style="color: red">-21% (0.569)</span> |

## RQ6 — Does learning-rate warmup help?

| schedule | no-warmup LR | warmup LR | no-warmup recall@100 | warmup=5% recall@100 | no-warmup ndcg@100 | warmup=5% ndcg@100 |
| --- | --- | --- | --- | --- | --- | --- |
| **constant** | **0.016/0.006** | **0.008/0.012** | **0.139** | **0% (0.139)** | **0.054** | **+1% (0.054)** |
| cosine | 0.032/0.012 | 0.032/0.006 | 0.134 | <span style="color: red">-13% (0.117)</span> | 0.051 | <span style="color: red">-15% (0.044)</span> |
| inverse sqrt | 0.032/0.006 | 0.008/0.012 | 0.130 | <span style="color: green">+4% (0.134)</span> | 0.049 | <span style="color: green">+5% (0.052)</span> |

## RQ7 — Which position encoding works best: RoPE, ALiBi, learned positions, or combinations?

### Earlier broad position-encoding comparison

| encoding | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- |
| position none | 0.124 | 0.047 | 0.025 | 0.020 | 0.650 |
| **position alibi** | <span style="color: green">+10% (0.135)</span> | <span style="color: green">+10% (0.052)</span> | <span style="color: green">+13% (0.029)</span> | <span style="color: green">+12% (0.023)</span> | -5% (0.620) |
| position all | <span style="color: green">+6% (0.131)</span> | <span style="color: green">+7% (0.050)</span> | +8% (0.027) | <span style="color: green">+8% (0.022)</span> | -13% (0.567) |
| position learned alibi | <span style="color: green">+9% (0.135)</span> | <span style="color: green">+10% (0.052)</span> | +10% (0.028) | <span style="color: green">+11% (0.023)</span> | -4% (0.621) |
| position learned forward | <span style="color: green">+9% (0.135)</span> | <span style="color: green">+9% (0.052)</span> | +10% (0.028) | <span style="color: green">+11% (0.023)</span> | -11% (0.577) |
| position learned forward reverse | <span style="color: red">-3% (0.120)</span> | <span style="color: red">-5% (0.045)</span> | -10% (0.023) | <span style="color: red">-10% (0.018)</span> | +4% (0.678) |
| position learned reverse | <span style="color: green">+5% (0.130)</span> | <span style="color: green">+4% (0.049)</span> | 0% (0.025) | +1% (0.020) | +3% (0.670) |
| position learned reverse alibi | +1% (0.125) | +1% (0.048) | -3% (0.025) | -1% (0.020) | <span style="color: red">-27% (0.475)</span> |
| position reverse all | <span style="color: red">-4% (0.119)</span> | <span style="color: red">-5% (0.045)</span> | -9% (0.023) | <span style="color: red">-8% (0.019)</span> | <span style="color: red">-31% (0.446)</span> |
| position rope | +1% (0.125) | +1% (0.048) | 0% (0.025) | +1% (0.020) | -4% (0.625) |
| position rope alibi | <span style="color: green">+9% (0.135)</span> | <span style="color: green">+9% (0.052)</span> | +11% (0.028) | <span style="color: green">+10% (0.022)</span> | +1% (0.659) |
| position rope learned | <span style="color: green">+8% (0.134)</span> | <span style="color: green">+9% (0.052)</span> | +11% (0.028) | <span style="color: green">+11% (0.023)</span> | -3% (0.631) |
| position rope learned reverse | <span style="color: red">-4% (0.118)</span> | <span style="color: red">-5% (0.045)</span> | -9% (0.023) | <span style="color: red">-6% (0.019)</span> | <span style="color: red">-41% (0.385)</span> |
| position rope learned reverse alibi | <span style="color: red">-5% (0.117)</span> | <span style="color: red">-6% (0.044)</span> | -12% (0.022) | <span style="color: red">-8% (0.019)</span> | <span style="color: red">-49% (0.332)</span> |
| position rope reverse | +1% (0.125) | +1% (0.048) | 0% (0.025) | +1% (0.021) | -4% (0.625) |
| position rope reverse alibi | <span style="color: green">+8% (0.133)</span> | <span style="color: green">+8% (0.051)</span> | +10% (0.028) | <span style="color: green">+9% (0.022)</span> | -9% (0.591) |
| position rope reverse learned | <span style="color: green">+9% (0.134)</span> | <span style="color: green">+9% (0.052)</span> | +11% (0.028) | <span style="color: green">+11% (0.023)</span> | -3% (0.628) |
| position rope reverse learned alibi | <span style="color: green">+7% (0.132)</span> | <span style="color: green">+7% (0.050)</span> | +8% (0.028) | <span style="color: green">+8% (0.022)</span> | -13% (0.567) |
| position rope reverse learned reverse | <span style="color: green">+6% (0.131)</span> | <span style="color: green">+6% (0.050)</span> | +7% (0.027) | <span style="color: green">+6% (0.021)</span> | -4% (0.623) |

<!-- rq7-reinvestigation-generated:start -->
### Learned-position fusion comparisons

| learned-position treatment | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward additive | 0.134 | 0.051 | 0.028 | 0.022 | 0.647 |
| forward concatenated to item | 0% (0.134) | +1% (0.052) | +2% (0.029) | +3% (0.023) | +3% (0.668) |
| forward + reverse additive | 0% (0.133) | 0% (0.051) | 0% (0.028) | 0% (0.022) | -2% (0.634) |
| forward + reverse concatenated to item | +2% (0.137) | <span style="color: green">+3% (0.053)</span> | +5% (0.029) | <span style="color: green">+6% (0.024)</span> | +1% (0.657) |
| ALiBi + forward additive | 0.136 | 0.052 | 0.028 | 0.022 | 0.662 |
| **ALiBi + forward concatenated to item** | +1% (0.138) | <span style="color: green">+3% (0.053)</span> | +7% (0.030) | <span style="color: green">+8% (0.024)</span> | +3% (0.684) |
| ALiBi + forward + reverse additive | -1% (0.135) | -1% (0.051) | -1% (0.028) | -2% (0.022) | +1% (0.670) |
| ALiBi + forward + reverse concatenated to item | +1% (0.138) | <span style="color: green">+3% (0.053)</span> | +4% (0.029) | <span style="color: green">+6% (0.023)</span> | +4% (0.691) |

### RoPE / ALiBi comparison

| RoPE / ALiBi treatment | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no position encoding | -2% (0.133) | <span style="color: red">-3% (0.050)</span> | -5% (0.027) | <span style="color: red">-6% (0.022)</span> | +8% (0.671) |
| **ALiBi** | 0.135 | 0.052 | 0.029 | 0.023 | 0.624 |
| plain forward RoPE (base 10000) | -1% (0.135) | -1% (0.051) | -2% (0.028) | -3% (0.022) | +5% (0.653) |
| forward RoPE + ALiBi | 0% (0.135) | -1% (0.052) | -2% (0.028) | -3% (0.022) | +5% (0.657) |
<!-- rq7-reinvestigation-generated:end -->

## RQ8 — How do scaling and architecture choices affect metrics?

| dimension | recall@100 | ndcg@100 |
| --- | --- | --- |
| **64** | 0.135 | 0.052 |
| 128 | 0% (0.134) | 0% (0.051) |
| 16 | <span style="color: red">-11% (0.120)</span> | <span style="color: red">-13% (0.045)</span> |
| 256 | <span style="color: red">-5% (0.128)</span> | <span style="color: red">-6% (0.048)</span> |
| 32 | <span style="color: red">-3% (0.131)</span> | <span style="color: red">-3% (0.050)</span> |

| depth | recall@100 | ndcg@100 |
| --- | --- | --- |
| 2 layers | 0.135 | 0.052 |
| 1 layers | -1% (0.134) | -2% (0.051) |
| **4 layers** | +2% (0.137) | <span style="color: green">+2% (0.053)</span> |

| mha head count | recall@100 | ndcg@100 |
| --- | --- | --- |
| **2Q/2KV** | 0.134 | 0.052 |
| 1Q/1KV | <span style="color: red">-3% (0.130)</span> | <span style="color: red">-4% (0.049)</span> |
| 4Q/4KV | <span style="color: red">-5% (0.127)</span> | <span style="color: red">-6% (0.048)</span> |
| 8Q/8KV | <span style="color: red">-16% (0.113)</span> | <span style="color: red">-18% (0.042)</span> |

| attention grouping | recall@100 | ndcg@100 |
| --- | --- | --- |
| MHA 2Q/2KV | 0.134 | 0.052 |
| **GQA 2Q/1KV** | 0% (0.135) | 0% (0.052) |

| block normalization kind | recall@100 | ndcg@100 |
| --- | --- | --- |
| **LayerNorm** | 0.135 | 0.052 |
| BatchNorm | <span style="color: red">-3% (0.131)</span> | <span style="color: red">-4% (0.050)</span> |
| RMSNorm | -1% (0.133) | -1% (0.051) |

| residual normalization placement | recall@100 | ndcg@100 |
| --- | --- | --- |
| pre-LayerNorm | 0.135 | 0.052 |
| **post-LayerNorm** | <span style="color: green">+2% (0.138)</span> | <span style="color: green">+2% (0.053)</span> |

| input and final normalization | recall@100 | ndcg@100 |
| --- | --- | --- |
| no input + final LayerNorm | 0.135 | 0.052 |
| **input + final RMSNorm** | <span style="color: green">+3% (0.138)</span> | <span style="color: green">+3% (0.053)</span> |
| input LayerNorm + final LayerNorm | <span style="color: red">-6% (0.127)</span> | <span style="color: red">-7% (0.048)</span> |
| input RMSNorm + final LayerNorm | <span style="color: red">-8% (0.124)</span> | <span style="color: red">-10% (0.047)</span> |
| no input or final norm | -1% (0.134) | <span style="color: red">-2% (0.051)</span> |

| attention window | recall@100 | ndcg@100 |
| --- | --- | --- |
| 50 | 0.135 | 0.052 |
| 10 | <span style="color: red">-17% (0.112)</span> | <span style="color: red">-19% (0.042)</span> |
| **100** | +1% (0.136) | +1% (0.052) |
| 25 | <span style="color: red">-2% (0.132)</span> | <span style="color: red">-3% (0.050)</span> |
| 75 | -1% (0.133) | -1% (0.051) |
| full | 0% (0.134) | 0% (0.052) |

| dropout | recall@100 | ndcg@100 |
| --- | --- | --- |
| **0.1** | 0.135 | 0.052 |
| 0.0 | <span style="color: red">-3% (0.131)</span> | <span style="color: red">-3% (0.050)</span> |
| 0.2 | -1% (0.133) | -1% (0.051) |
| 0.3 | -2% (0.132) | <span style="color: red">-3% (0.050)</span> |
| 0.05 | -1% (0.133) | -2% (0.051) |
| 0.5 | <span style="color: red">-6% (0.127)</span> | <span style="color: red">-7% (0.048)</span> |

| bos | recall@100 | ndcg@100 |
| --- | --- | --- |
| disabled | 0.135 | 0.052 |
| **enabled** | +1% (0.135) | +2% (0.052) |

| query objective | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard item-state | 0.135 | 0.051 | 0.028 | 0.022 | 0.728 |
| **end-only CLS** | <span style="color: green">+11% (0.149)</span> | <span style="color: green">+19% (0.061)</span> | <span style="color: green">+34% (0.038)</span> | <span style="color: green">+38% (0.031)</span> | <span style="color: red">-39% (0.441)</span> |
| interleaved CLS | -1% (0.133) | <span style="color: red">-2% (0.050)</span> | -6% (0.027) | <span style="color: red">-5% (0.021)</span> | <span style="color: red">-27% (0.532)</span> |

| causal ALiBi retained history length | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | <span style="color: red">-7% (0.124)</span> | <span style="color: red">-8% (0.047)</span> | <span style="color: red">-11% (0.025)</span> | <span style="color: red">-11% (0.020)</span> | <span style="color: red">-37% (0.444)</span> |
| 25 | -2% (0.132) | <span style="color: red">-2% (0.050)</span> | -3% (0.027) | -4% (0.021) | -10% (0.636) |
| 50 | 0% (0.135) | 0% (0.051) | +1% (0.028) | 0% (0.022) | -11% (0.633) |
| 100 | +1% (0.135) | 0% (0.051) | -2% (0.028) | -2% (0.022) | +5% (0.747) |
| 128 | 0.134 | 0.051 | 0.028 | 0.022 | 0.709 |
| 200 | 0% (0.134) | 0% (0.051) | -1% (0.028) | -1% (0.022) | -10% (0.639) |
| 256 | +1% (0.135) | +1% (0.052) | +1% (0.028) | +1% (0.023) | 0% (0.712) |
| **512** | **+1% (0.136)** | **+1% (0.052)** | **+1% (0.028)** | **0% (0.023)** | **-4% (0.679)** |

| reverse-RoPE + ALiBi retained history length | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | <span style="color: red">-7% (0.124)</span> | <span style="color: red">-8% (0.047)</span> | -7% (0.026) | <span style="color: red">-7% (0.020)</span> | <span style="color: red">-27% (0.467)</span> |
| 25 | -2% (0.132) | -2% (0.050) | 0% (0.027) | 0% (0.022) | -2% (0.627) |
| 50 | 0% (0.135) | 0% (0.051) | 0% (0.027) | -1% (0.022) | -3% (0.621) |
| 100 | 0% (0.134) | 0% (0.051) | -1% (0.027) | 0% (0.022) | 0% (0.645) |
| 128 | 0.134 | 0.051 | 0.027 | 0.022 | 0.642 |
| 200 | 0% (0.134) | -1% (0.051) | -1% (0.027) | -2% (0.021) | +13% (0.724) |
| 256 | 0% (0.134) | -1% (0.050) | -1% (0.027) | -3% (0.021) | +11% (0.715) |
| **512** | **0% (0.135)** | **+1% (0.051)** | **+2% (0.028)** | **+2% (0.022)** | **+13% (0.725)** |

## RQ9 — Does a timestamp-delta representation improve metrics?

| time representation | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- |
| no time feature | 0.121 | 0.045 | 0.023 | 0.019 | 0.733 |
| 8 log-spaced bins, add | <span style="color: green">+13% (0.137)</span> | <span style="color: green">+16% (0.052)</span> | <span style="color: green">+23% (0.029)</span> | <span style="color: green">+25% (0.023)</span> | -9% (0.663) |
| 16 log-spaced bins, add | <span style="color: green">+11% (0.135)</span> | <span style="color: green">+14% (0.052)</span> | <span style="color: green">+20% (0.028)</span> | <span style="color: green">+22% (0.023)</span> | <span style="color: red">-21% (0.577)</span> |
| 32 log-spaced bins, add | <span style="color: green">+12% (0.136)</span> | <span style="color: green">+14% (0.052)</span> | <span style="color: green">+22% (0.028)</span> | <span style="color: green">+22% (0.023)</span> | <span style="color: red">-32% (0.497)</span> |
| 64 log-spaced bins, add | <span style="color: green">+11% (0.134)</span> | <span style="color: green">+12% (0.051)</span> | <span style="color: green">+17% (0.027)</span> | <span style="color: green">+17% (0.022)</span> | <span style="color: red">-15% (0.622)</span> |
| **32 bins + raw reverse RoPE** | <span style="color: green">+14% (0.137)</span> | <span style="color: green">+16% (0.052)</span> | <span style="color: green">+23% (0.029)</span> | <span style="color: green">+22% (0.023)</span> | -10% (0.658) |
| 32 bins + log forward RoPE | <span style="color: green">+13% (0.137)</span> | <span style="color: green">+16% (0.053)</span> | <span style="color: green">+25% (0.029)</span> | <span style="color: green">+26% (0.023)</span> | -12% (0.647) |
| 32 bins, concatenate-and-project | +1% (0.122) | +1% (0.046) | +4% (0.024) | +5% (0.019) | <span style="color: red">-44% (0.408)</span> |
| clipped linear delta, add | <span style="color: green">+9% (0.131)</span> | <span style="color: green">+9% (0.049)</span> | <span style="color: green">+13% (0.026)</span> | <span style="color: green">+12% (0.021)</span> | -2% (0.717) |
| log delta, add | <span style="color: green">+7% (0.129)</span> | <span style="color: green">+9% (0.049)</span> | <span style="color: green">+14% (0.027)</span> | <span style="color: green">+14% (0.021)</span> | +3% (0.754) |
| log delta, concatenate-and-project | <span style="color: green">+12% (0.135)</span> | <span style="color: green">+16% (0.052)</span> | <span style="color: green">+23% (0.029)</span> | <span style="color: green">+26% (0.023)</span> | <span style="color: red">-25% (0.547)</span> |
| raw elapsed-time RoPE, forward | <span style="color: green">+5% (0.127)</span> | <span style="color: green">+6% (0.048)</span> | +10% (0.026) | <span style="color: green">+10% (0.020)</span> | +7% (0.782) |
| raw elapsed-time RoPE, reverse | <span style="color: green">+5% (0.126)</span> | <span style="color: green">+6% (0.048)</span> | +10% (0.026) | <span style="color: green">+10% (0.020)</span> | +5% (0.767) |
| log elapsed-time RoPE, forward | <span style="color: red">-3% (0.117)</span> | <span style="color: red">-3% (0.044)</span> | -4% (0.022) | -3% (0.018) | <span style="color: red">-42% (0.424)</span> |
| log elapsed-time RoPE, reverse | <span style="color: red">-11% (0.107)</span> | <span style="color: red">-10% (0.041)</span> | -9% (0.021) | <span style="color: red">-7% (0.017)</span> | -8% (0.677) |

## RQ10 — Do separate item embeddings at every transformer layer help?

### Earlier valid two-layer comparison

| item-feature path | recall@100 | ndcg@100 |
| --- | ---: | ---: |
| input/output item embedding only | 0.140 | 0.053 |
| direct full-width addition before every layer | 0.139 | 0.053 |

### Four-layer reinvestigation

| item-feature path | feature width | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | parameters (M) | median epoch (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Input/output item embedding only** | **—** | **0.138** | **0.053** | **0.029** | **0.023** | **0.648** | **10.3** | **12.70** |
| Direct full-width addition | 64 | <span style="color: red">-11% (0.123)</span> | <span style="color: red">-12% (0.047)</span> | <span style="color: red">-14% (0.025)</span> | <span style="color: red">-14% (0.020)</span> | -3% (0.632) | 50.6 | 12.94 |
| Zero-start concatenated DenseNet residual | 64 | <span style="color: red">-3% (0.133)</span> | <span style="color: red">-3% (0.051)</span> | -2% (0.028) | -3% (0.023) | +4% (0.676) | 50.6 | 13.64 |
| Zero-start Gemma-style PLE | 2 | -1% (0.136) | -1% (0.053) | +0% (0.029) | +1% (0.023) | -8% (0.596) | 11.5 | 13.47 |

## RQ11 — How do online logQ, offline logQ, random, mixed, and uncorrected negatives compare?

### Earlier broad negative-sampling comparison

| negative sampling | negatives | logQ alpha | random fraction | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed in-batch leave-one-out logQ | 2048 | — | — | 0.116 | 0.044 | 0.023 | 0.019 | 0.461 |
| fixed in-batch global-q Yi-2019 | 2048 | — | — | <span style="color: green">+5% (0.122)</span> | <span style="color: green">+5% (0.046)</span> | +8% (0.024) | <span style="color: green">+7% (0.020)</span> | <span style="color: green">+34% (0.617)</span> |
| popularity random global-q Yi-2019 | 512 | — | — | <span style="color: green">+15% (0.133)</span> | <span style="color: green">+16% (0.051)</span> | <span style="color: green">+23% (0.028)</span> | <span style="color: green">+22% (0.023)</span> | <span style="color: green">+29% (0.593)</span> |
| streaming in-batch global-q Yi-2019 | 512 | 0.005 | — | <span style="color: green">+14% (0.133)</span> | <span style="color: green">+15% (0.051)</span> | <span style="color: green">+22% (0.028)</span> | <span style="color: green">+20% (0.022)</span> | +21% (0.556) |
| uncorrected in-batch | 512 | — | — | <span style="color: red">-45% (0.064)</span> | <span style="color: red">-48% (0.023)</span> | <span style="color: red">-53% (0.011)</span> | <span style="color: red">-55% (0.008)</span> | <span style="color: green">+101% (0.925)</span> |
| **uniform random** | 512 | — | — | <span style="color: green">+16% (0.135)</span> | <span style="color: green">+17% (0.052)</span> | <span style="color: green">+23% (0.028)</span> | <span style="color: green">+21% (0.023)</span> | <span style="color: green">+25% (0.577)</span> |
| uniform random + fixed logQ on in-batch negatives | 512 | — | 0.25 | <span style="color: green">+11% (0.129)</span> | <span style="color: green">+12% (0.049)</span> | <span style="color: green">+20% (0.027)</span> | <span style="color: green">+16% (0.021)</span> | <span style="color: green">+36% (0.626)</span> |
| uniform random + streaming logQ on in-batch negatives | 512 | 0.01 | 0.875 | <span style="color: green">+6% (0.123)</span> | <span style="color: green">+6% (0.047)</span> | +11% (0.025) | <span style="color: green">+9% (0.020)</span> | +1% (0.466) |

| homework-matched objective | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- |
| fixed leave-one-out logQ | 0.127 | 0.048 | 0.024 | 0.019 | 0.511 |
| **uniform random** | <span style="color: green">+3% (0.132)</span> | <span style="color: green">+4% (0.050)</span> | +7% (0.026) | <span style="color: green">+6% (0.020)</span> | <span style="color: green">+25% (0.641)</span> |

### Corrected uniform/streaming mixture comparison

| negative sampling | negatives | logQ alpha | uniform fraction | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform catalog | 2048 | — | — | 0.136 | 0.052 | 0.028 | 0.023 | 0.560 |
| streaming in-batch global-q | 1024 | 0.005 | — | -1% (0.134) | +0% (0.052) | +3% (0.029) | +3% (0.023) | +12% (0.628) |
| **popularity catalog global-q** | 2048 | — | — | +1% (0.137) | <span style="color: green">+2% (0.053)</span> | +6% (0.030) | <span style="color: green">+5% (0.024)</span> | <span style="color: green">+18% (0.662)</span> |
| aggregate uniform + streaming global-q | 256 | 0.0025 | 0.75 | +0% (0.136) | +0% (0.052) | +1% (0.028) | +1% (0.023) | <span style="color: green">+21% (0.674)</span> |
