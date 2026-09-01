# G6: RQ-KMeans semantic IDs in history

## RQ0 — How should SIDs describe history?

| Method (best-G1 baseline) | Recall@100 | Delta Recall@100 | NDCG@100 | MRR@100 | Coverage@100 | SID configuration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Best G1 combination: item IDs | 0.125 | baseline | 0.051 | 0.050 | 0.250 | none |
| Trainable SID event | 0.098 | <span style="color: red">-22.14% (-0.028)</span> | <span style="color: red">-20.29% (0.041)</span> | <span style="color: red">-15.46% (0.042)</span> | <span style="color: red">-38.89% (0.153)</span> | 3 levels × 256 codes; width 32 |
| Item ID + frozen SID event | 0.130 | <span style="color: green">+3.77% (+0.005)</span> | +0.93% (0.052) | +1.42% (0.050) | +7.20% (0.268) | 3 levels × 512 codes; width 128 |
| Item ID + trainable/frozen SID event | 0.127 | +1.27% (+0.002) | <span style="color: red">-4.85% (0.049)</span> | <span style="color: red">-7.32% (0.046)</span> | -0.16% (0.250) | 4 levels × 512 codes; width 32 |
| Trainable SID tokens | 0.108 | <span style="color: red">-13.98% (-0.018)</span> | <span style="color: red">-18.58% (0.042)</span> | <span style="color: red">-21.43% (0.039)</span> | -9.55% (0.226) | 3 levels × 64 codes; width 64 |
| Trainable/frozen SID tokens | 0.096 | <span style="color: red">-23.16% (-0.029)</span> | <span style="color: red">-14.97% (0.044)</span> | +3.97% (0.052) | <span style="color: red">-67.96% (0.080)</span> | 2 levels × 256 codes; width 32 |
| Frozen SID tokens | 0.078 | <span style="color: red">-37.69% (-0.047)</span> | <span style="color: red">-35.24% (0.033)</span> | <span style="color: red">-31.13% (0.034)</span> | <span style="color: red">-96.67% (0.008)</span> | 3 levels × 128 codes; width 64 |
| Interleaved item ID/SID tokens | 0.122 | <span style="color: red">-2.39% (-0.003)</span> | <span style="color: red">-7.72% (0.047)</span> | <span style="color: red">-7.05% (0.046)</span> | +2.02% (0.255) | 3 levels × 32 codes; width 64 |

| Method (original G1 baseline) | Recall@100 | Delta Recall@100 | NDCG@100 | MRR@100 | Coverage@100 | SID configuration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Original G1: item IDs | 0.101 | baseline | 0.037 | 0.030 | 0.593 | none |
| Item ID + frozen SID event | 0.102 | +0.99% (+0.001) | -1.11% (0.036) | -0.30% (0.030) | <span style="color: green">+25.09% (0.742)</span> | 3 levels × 512 codes; width 128 |

| Controlled learned-SID addition | Recall@100 | NDCG@100 | Gate bound | Effective gate |
| :--- | :---: | :---: | :---: | :---: |
| Item ID + frozen SID event (external tuned control) | 0.130 | 0.052 | — | — |
| Learned residual, unbounded | <span style="color: red">-5.90% (0.123)</span> | <span style="color: red">-4.32% (0.049)</span> | unbounded | — |
| Learned residual disabled (shared treatment LR) | <span style="color: red">-7.39% (0.121)</span> | <span style="color: red">-4.29% (0.049)</span> | 0 | 0 |
| Learned residual, bound 0.01 | <span style="color: red">-6.46% (0.122)</span> | <span style="color: red">-6.20% (0.048)</span> | 0.01 | -0.000654967 |
| Learned residual, bound 0.025 | <span style="color: red">-5.99% (0.122)</span> | <span style="color: red">-7.49% (0.048)</span> | 0.025 | -0.00318165 |
| Learned residual, bound 0.05 | **<span style="color: red">-3.39% (0.126)</span>** | <span style="color: red">-10.05% (0.046)</span> | 0.05 | -0.0120591 |
| Learned residual, bound 0.1 | <span style="color: red">-6.15% (0.122)</span> | <span style="color: red">-11.95% (0.046)</span> | 0.1 | -0.0684015 |

| Method (SID retrieval diagnostics) | Exact SID Recall@100 | Prefix L1 | Prefix L2 | Prefix L3 | Prefix L4 | ICR | Collided items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Trainable SID event | 0.106 | 0.558 | 0.152 | 0.106 | — | 14.59% | 24.00% |
| Item ID + frozen SID event | 0.137 | 0.510 | 0.163 | 0.137 | — | 8.36% | 14.62% |
| Item ID + trainable/frozen SID event | 0.130 | 0.517 | 0.159 | 0.134 | 0.130 | 3.98% | 7.25% |
| Trainable SID tokens | 0.174 | 0.782 | 0.356 | 0.174 | — | 52.00% | 69.86% |
| Trainable/frozen SID tokens | 0.142 | 0.525 | 0.142 | — | — | 60.24% | 80.42% |
| Frozen SID tokens | 0.081 | 0.495 | 0.120 | 0.081 | — | 28.11% | 42.43% |
| Interleaved item ID/SID tokens | 0.248 | 0.847 | 0.492 | 0.248 | — | 76.03% | 90.21% |

| Method (SID geometry diagnostics) | p95 load by level | p95 / mean by level | Intra-code cosine by level |
| :--- | :---: | :---: | :---: |
| Trainable SID event | 209 / 216 / 256 | 1.614 / 1.668 / 1.977 | 0.732 / 0.393 / 0.258 |
| Item ID + frozen SID event | 107 / 122 / 152 | 1.653 / 1.884 / 2.348 | 0.776 / 0.392 / 0.245 |
| Item ID + trainable/frozen SID event | 107 / 122 / 152 / 166 | 1.653 / 1.884 / 2.348 / 2.564 | 0.776 / 0.392 / 0.245 / 0.214 |
| Trainable SID tokens | 832 / 734 / 779 | 1.606 / 1.417 / 1.504 | 0.611 / 0.405 / 0.266 |
| Trainable/frozen SID tokens | 209 / 216 | 1.614 / 1.668 | 0.732 / 0.393 |
| Frozen SID tokens | 411 / 411 / 419 | 1.587 / 1.587 / 1.618 | 0.673 / 0.404 / 0.268 |
| Interleaved item ID/SID tokens | 1923 / 1480 / 1551 | 1.654 / 1.429 / 1.497 | 0.527 / 0.375 / 0.269 |

| Method (best-G1 serving cost) | Sequence tokens | Transformer MACs | Tokenizer MACs | Total MACs | Embedding reads |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Best G1 combination: item IDs | 102 | 25.381M | 0.000M | 25.381M | 6,400 |
| Trainable SID event | 102 | 25.381M | 1.024M | 26.405M | 25,600 |
| Item ID + frozen SID event | 102 | 25.381M | 8.192M | 33.573M | 57,600 |
| Item ID + trainable/frozen SID event | 102 | 25.381M | 2.970M | 28.350M | 86,400 |
| Trainable SID tokens | 402 | 161.778M | 1.638M | 163.416M | 25,600 |
| Trainable/frozen SID tokens | 302 | 106.072M | 2.150M | 108.222M | 48,000 |
| Frozen SID tokens | 402 | 161.778M | 4.915M | 166.693M | 51,200 |
| Interleaved item ID/SID tokens | 502 | 227.723M | 2.048M | 229.771M | 32,000 |

| Method (original-G1 serving cost) | Sequence tokens | Transformer MACs | Tokenizer MACs | Total MACs | Embedding reads |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Original G1: item IDs | 100 | 12.390M | 0.000M | 12.390M | 6,400 |
| Item ID + frozen SID event | 100 | 12.390M | 8.192M | 20.582M | 57,600 |

| Target-frequency slice | Control Recall@100 | Semantic Recall@100 | Delta Recall@100 | Users | Targets |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Low | 0.031 | 0.036 | +0.005 | 2,184 | 6,663 |
| Middle | 0.079 | 0.083 | +0.004 | 2,115 | 6,413 |
| High | 0.245 | 0.246 | +0.001 | 2,090 | 6,516 |

| Collision-history slice | Control Recall@100 | Semantic Recall@100 | Delta Recall@100 | Users | Targets |
| :--- | :---: | :---: | :---: | :---: | :---: |
| History has collided base SID | 0.126 | 0.131 | +0.005 | 3,266 | 19,048 |
| No collided base SID in history | 0.107 | 0.109 | +0.002 | 148 | 544 |
