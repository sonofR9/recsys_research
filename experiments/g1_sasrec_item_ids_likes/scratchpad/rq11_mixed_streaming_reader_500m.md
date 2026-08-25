| negative sampling | negatives | logQ alpha | uniform fraction | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform catalog | 2048 | — | — | 0.136 | 0.052 | 0.028 | 0.023 | 0.560 |
| streaming in-batch global-q | 1024 | 0.005 | — | -1% (0.134) | +0% (0.052) | +3% (0.029) | +3% (0.023) | +12% (0.628) |
| **popularity catalog global-q** | 2048 | — | — | +1% (0.137) | <span style="color: green">+2% (0.053)</span> | +6% (0.030) | <span style="color: green">+5% (0.024)</span> | <span style="color: green">+18% (0.662)</span> |
| aggregate uniform + streaming global-q | 256 | 0.0025 | 0.75 | +0% (0.136) | +0% (0.052) | +1% (0.028) | +1% (0.023) | <span style="color: green">+21% (0.674)</span> |
