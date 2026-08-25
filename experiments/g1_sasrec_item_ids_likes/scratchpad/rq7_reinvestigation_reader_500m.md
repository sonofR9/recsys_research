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
