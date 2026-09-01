# G1 RQ4 — FFN activation, gating, and depth tuning

Native Yambda-500M, batch 1280, embedding LR 0.064, linear 20-epoch horizon under cap 40. Each table bolds its validation-selected recall@100 row. Plain FFNs use width 192; gated FFNs use width 114 and enable the same 0.1 internal FFN dropout as the plain arms. All arms use μP target/base/delta widths 64/16/32.

### ReLU, 2 layers (plain, width 192)

| embedding LR | deep LR | batch size | best/stopped/horizon epoch | validation recall@100 | final recall@100 | final ndcg@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.064 | 0.006 | 1280 | 12/20/20 | 0.127 | 0.126 | 0.048 |
| **0.064** | **0.012** | **1280** | **14/20/20** | **0.133** | **0.132** | **0.050** |
| 0.064 | 0.024 | 1280 | 14/20/20 | 0.132 | 0.133 | 0.050 |
| 0.064 | 0.048 | 1280 | 14/20/20 | 0.132 | 0.132 | 0.050 |
| 0.064 | 0.096 | 1280 | 14/20/20 | 0.131 | 0.131 | 0.049 |

### GELU, 2 layers (plain, width 192)

| embedding LR | deep LR | batch size | best/stopped/horizon epoch | validation recall@100 | final recall@100 | final ndcg@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.064 | 0.006 | 1280 | 16/20/20 | 0.129 | 0.131 | 0.050 |
| 0.064 | 0.012 | 1280 | 14/20/20 | 0.133 | 0.134 | 0.051 |
| **0.064** | **0.024** | **1280** | **17/20/20** | **0.134** | **0.134** | **0.051** |
| 0.064 | 0.048 | 1280 | 15/20/20 | 0.132 | 0.133 | 0.050 |
| 0.064 | 0.096 | 1280 | 19/20/20 | 0.128 | 0.128 | 0.049 |

### SiLU, 2 layers (plain, width 192)

| embedding LR | deep LR | batch size | best/stopped/horizon epoch | validation recall@100 | final recall@100 | final ndcg@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.064 | 0.006 | 1280 | 14/20/20 | 0.128 | 0.129 | 0.049 |
| 0.064 | 0.012 | 1280 | 14/20/20 | 0.129 | 0.130 | 0.050 |
| **0.064** | **0.024** | **1280** | **16/20/20** | **0.132** | **0.132** | **0.050** |
| 0.064 | 0.048 | 1280 | 15/20/20 | 0.129 | 0.129 | 0.049 |
| 0.064 | 0.096 | 1280 | 17/20/20 | 0.127 | 0.128 | 0.048 |

### ReGLU, 2 layers (gated, width 114)

| embedding LR | deep LR | batch size | best/stopped/horizon epoch | validation recall@100 | final recall@100 | final ndcg@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.064 | 0.006 | 1280 | 14/20/20 | 0.130 | 0.130 | 0.049 |
| 0.064 | 0.012 | 1280 | 17/20/20 | 0.133 | 0.134 | 0.051 |
| **0.064** | **0.024** | **1280** | **16/20/20** | **0.134** | **0.135** | **0.052** |
| 0.064 | 0.048 | 1280 | 14/20/20 | 0.133 | 0.133 | 0.050 |
| 0.064 | 0.096 | 1280 | 17/20/20 | 0.124 | 0.124 | 0.047 |

### GEGLU, 2 layers (gated, width 114)

| embedding LR | deep LR | batch size | best/stopped/horizon epoch | validation recall@100 | final recall@100 | final ndcg@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.064 | 0.006 | 1280 | 12/20/20 | 0.131 | 0.130 | 0.050 |
| **0.064** | **0.012** | **1280** | **16/20/20** | **0.136** | **0.137** | **0.053** |
| 0.064 | 0.024 | 1280 | 13/20/20 | 0.135 | 0.136 | 0.052 |

### SwiGLU, 2 layers (gated, width 114)

| embedding LR | deep LR | batch size | best/stopped/horizon epoch | validation recall@100 | final recall@100 | final ndcg@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.064 | 0.006 | 1280 | 16/20/20 | 0.131 | 0.131 | 0.050 |
| **0.064** | **0.012** | **1280** | **13/20/20** | **0.135** | **0.137** | **0.052** |
| 0.064 | 0.024 | 1280 | 12/20/20 | 0.133 | 0.134 | 0.051 |

### GELU, 4 layers (plain, width 192)

| embedding LR | deep LR | batch size | best/stopped/horizon epoch | validation recall@100 | final recall@100 | final ndcg@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.064 | 0.006 | 1280 | 20/20/20 | 0.132 | 0.132 | 0.050 |
| 0.064 | 0.012 | 1280 | 15/20/20 | 0.135 | 0.134 | 0.051 |
| **0.064** | **0.024** | **1280** | **15/20/20** | **0.136** | **0.135** | **0.052** |
| 0.064 | 0.048 | 1280 | 18/20/20 | 0.129 | 0.130 | 0.050 |
| 0.064 | 0.096 | 1280 | 17/20/20 | 0.130 | 0.131 | 0.050 |

### SwiGLU, 4 layers (gated, width 114)

| embedding LR | deep LR | batch size | best/stopped/horizon epoch | validation recall@100 | final recall@100 | final ndcg@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.064 | 0.006 | 1280 | 16/20/20 | 0.133 | 0.135 | 0.052 |
| **0.064** | **0.012** | **1280** | **12/20/20** | **0.135** | **0.136** | **0.052** |
| 0.064 | 0.024 | 1280 | 16/20/20 | 0.135 | 0.136 | 0.052 |

### GELU, 8 layers (plain, width 192)

| embedding LR | deep LR | batch size | best/stopped/horizon epoch | validation recall@100 | final recall@100 | final ndcg@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.064 | 0.006 | 1280 | 14/20/20 | 0.133 | 0.134 | 0.051 |
| 0.064 | 0.012 | 1280 | 14/20/20 | 0.136 | 0.137 | 0.053 |
| **0.064** | **0.024** | **1280** | **14/20/20** | **0.136** | **0.137** | **0.053** |
| 0.064 | 0.048 | 1280 | 16/20/20 | 0.133 | 0.134 | 0.051 |
| 0.064 | 0.096 | 1280 | 17/20/20 | 0.133 | 0.135 | 0.052 |

### SwiGLU, 8 layers (gated, width 114)

| embedding LR | deep LR | batch size | best/stopped/horizon epoch | validation recall@100 | final recall@100 | final ndcg@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.064 | 0.006 | 1280 | 15/20/20 | 0.135 | 0.137 | 0.053 |
| **0.064** | **0.012** | **1280** | **12/20/20** | **0.139** | **0.139** | **0.054** |
| 0.064 | 0.024 | 1280 | 17/20/20 | 0.138 | 0.139 | 0.054 |
