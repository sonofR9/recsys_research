## RQ13 — Does bounded prefix expansion improve an encoder-decoder?

### Candidate-generation quality

| architecture | prefix expansion | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| encoder-decoder | no expansion | 0.077 | 0.034 | 0.023 | 0.020 | 0.002 |
| encoder-decoder | latest 8 truncated prefixes | <span style="color: green">+49% (0.116)</span> | <span style="color: green">+39% (0.047)</span> | <span style="color: green">+28% (0.029)</span> | <span style="color: green">+24% (0.024)</span> | +4399% (0.101) |
| encoder-decoder | latest 16 truncated prefixes | <span style="color: green">+60% (0.124)</span> | <span style="color: green">+45% (0.049)</span> | <span style="color: green">+25% (0.028)</span> | <span style="color: green">+20% (0.024)</span> | <span style="color: green">+11229% (0.255)</span> |
| encoder-decoder | latest 8 required-length prefixes | <span style="color: green">+20% (0.093)</span> | <span style="color: green">+17% (0.040)</span> | <span style="color: green">+21% (0.027)</span> | <span style="color: green">+15% (0.023)</span> | +630% (0.016) |
| encoder-decoder | latest 16 required-length prefixes | <span style="color: green">+35% (0.105)</span> | <span style="color: green">+26% (0.043)</span> | <span style="color: green">+19% (0.027)</span> | <span style="color: green">+14% (0.022)</span> | +1789% (0.043) |
| encoder-decoder | latest 4 truncated prefixes | <span style="color: green">+31% (0.101)</span> | <span style="color: green">+23% (0.042)</span> | <span style="color: green">+18% (0.027)</span> | <span style="color: green">+12% (0.022)</span> | +2399% (0.056) |
| **encoder-decoder** | **latest 32 truncated prefixes (fitted practical cap)** | **<span style="color: green">+62% (0.125)</span>** | **<span style="color: green">+40% (0.047)</span>** | **+11% (0.025)** | **+4% (0.020)** | **<span style="color: green">+14067% (0.319)</span>** |
| regular decoder-only SASRec | none | <span style="color: green">+74% (0.135)</span> | <span style="color: green">+51% (0.051)</span> | <span style="color: green">+25% (0.028)</span> | <span style="color: green">+14% (0.022)</span> | <span style="color: green">+32263% (0.728)</span> |

### Training efficiency

| architecture | prefix expansion | original users/epoch | expanded examples/epoch | candidate targets/epoch | NTP targets/epoch | input tokens/epoch | selected checkpoint epoch | steady-state targets/s | time through selected checkpoint (train+validation), s | total required training wall (all tuning and boundary artifacts), s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| regular decoder-only SASRec | none | — | 110731 | 0 | 7674702 | 7785433 | 16 / 15 / 20 | 443883.299 | 299.221 | 2530.743 |
| encoder-decoder | no expansion | 75434 | 75434 | 75434 | 0 | 4530971 | 3 | 27944.238 | 8.221 | 193.083 |
| encoder-decoder | latest 8 truncated prefixes | 75434 | 538703 | 538703 | 0 | 34702000 | 5 | 17623.327 | 110.530 | 1878.702 |
| encoder-decoder | latest 16 truncated prefixes | 75434 | 996053 | 996053 | 0 | 66404954 | 7 | 17681.441 | 445.514 | 3438.630 |
| encoder-decoder | latest 8 required-length prefixes | 75434 | 195575 | 195575 | 0 | 20029160 | 4 | 13752.554 | 59.690 | 1263.077 |
| encoder-decoder | latest 16 required-length prefixes | 75434 | 325213 | 325213 | 0 | 36752462 | 5 | 24917.180 | 75.032 | 3069.423 |
| encoder-decoder | latest 4 truncated prefixes | 75434 | 284334 | 284334 | 0 | 17771625 | 6 | 28575.979 | 59.835 | 842.029 |
| **encoder-decoder** | **latest 32 truncated prefixes (fitted practical cap)** | **75434** | **1772396** | **1772396** | **0** | **122550944** | **7** | **15137.767** | **799.969** | **8347.261** |

### Aggregated cap-target evaluation

The selected practical cap is 32. Its full-user Recall@100 is 0.125480; the reader-only 1.10× decoder target is 0.148152. Cap selection used validation Recall@100 only.
