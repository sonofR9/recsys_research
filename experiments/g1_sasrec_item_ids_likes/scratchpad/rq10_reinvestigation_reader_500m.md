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
