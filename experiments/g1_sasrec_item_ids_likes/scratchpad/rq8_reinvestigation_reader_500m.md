## RQ8 — How do scaling and architecture choices affect metrics?

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
