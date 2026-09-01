# G6: RQ-KMeans semantic IDs in history — RQ1–RQ3

## RQ1 — Does content-informed SID initialization outperform random initialization?

| Method | Recall@100 | Delta Recall@100 | NDCG@100 | MRR@100 | Coverage@100 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random initialization | 0.119 | baseline | 0.045 | 0.039 | 0.234 |
| Content-informed PCA initialization | 0.125 | <span style="color: green">+5.04% (0.125)</span> | <span style="color: green">+7.65% (0.048)</span> | <span style="color: green">+17.11% (0.046)</span> | +5.67% (0.247) |

| Method (convergence) | Mean epoch to 95% | Mean normalized Recall AUC |
| :--- | :---: | :---: |
| Random initialization | 7.00 | 0.855 |
| Content-informed PCA initialization | 9.25 | 0.821 |

| Method (SID diagnostics) | Exact SID Recall@100 | Prefix L1 | Prefix L2 | Prefix L3 | Prefix L4 | ICR | Collided items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random initialization | 0.121 | 0.501 | 0.149 | 0.125 | 0.121 | 3.98% | 7.25% |
| Content-informed PCA initialization | 0.128 | 0.503 | 0.157 | 0.131 | 0.128 | 3.98% | 7.25% |

| Tokenizer (intrinsic diagnostics) | p95 load by level | p95 / mean by level | Intra-code cosine by level | Reconstruction MSE by depth | Dead-code fraction by level |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Shared 4 × 512 tokenizer | 107 / 122 / 152 / 166 | 1.653 / 1.884 / 2.348 / 2.564 | 0.776 / 0.392 / 0.245 / 0.214 | 0.232 / 0.153 / 0.115 / 0.091 | 0.00% / 0.00% / 0.00% / 0.00% |

| Tokenizer (collision diagnostics) | Unique base tuples | ICR | Collided items | Suffix symbols | Bucket p50 / p95 / p99 / max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Shared 4 × 512 tokenizer | 31830 | 3.98% | 7.25% | 29 | 1 / 1 / 2 / 28 |

| RQ1 unexpected-result check | Evidence |
| :--- | :---: |
| Initialization hashes and RMS | all checks pass at seeds 43/44/45; max RMS mismatch 1.86e-09 |
| Learned-base-row gradient | nonzero on 4 touched base rows; frozen centroids unchanged |
| 128→32 PCA reconstruction | retained variance 76.0–87.6%; centered MSE 0.000129–0.000585 |
| Frozen-view redundancy | initialized 32D view is a deterministic linear projection of the same frozen 128D centroids |
| Paired LR warm-start erasure | not supported: five fixed-deep-LR AUC deltas are non-monotone |

| Method (serving-cost estimate) | Sequence tokens | Transformer MACs | Tokenizer MACs | Total MACs | Embedding reads |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random initialization | 102 | 25.381M | 2.970M | 28.350M | 86,400 |
| Content-informed PCA initialization | 102 | 25.381M | 2.970M | 28.350M | 86,400 |

| Method (unavailable diagnostics) | Not committed |
| :--- | :---: |
| Both initialization arms | model parameters, artifact bytes, tokenizer fit time, epoch time, peak memory, dedicated full-catalog latency, target/history collision slices |

## RQ2 — What RQ-KMeans setup works best with collision resolution?

| Method | Recall@100 | Delta Recall@100 | NDCG@100 | MRR@100 | Coverage@100 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RQ0 suffix setup (3 levels × 512 shared codes; 20 iterations) | 0.127 | baseline | 0.049 | 0.047 | 0.278 |
| **Selected suffix setup (3 levels × 512 shared codes; 20 iterations)** | 0.127 | +0.00% (0.127) | +0.00% (0.049) | +0.00% (0.047) | +0.00% (0.278) |

| Method (SID diagnostics) | Exact SID Recall@100 | Prefix L1 | Prefix L2 | Prefix L3 | ICR | Collided items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Selected suffix setup | 0.133 | 0.511 | 0.160 | 0.133 | 8.36% | 14.62% |

| Tokenizer (intrinsic diagnostics) | p95 load by level | p95 / mean by level | Intra-code cosine by level | Reconstruction MSE by depth | Dead-code fraction by level |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Selected suffix tokenizer | 107 / 122 / 152 | 1.653 / 1.884 / 2.348 | 0.776 / 0.392 / 0.245 | — | — |

| Tokenizer (collision diagnostics) | Unique base tuples | ICR | Collided items | Suffix symbols | Bucket p50 / p95 / p99 / max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Selected suffix tokenizer | 30378 | 8.36% | 14.62% | 29 | 1 / 2 / 3 / 28 |

| Method (serving-cost estimate) | Sequence tokens | Transformer MACs | Tokenizer MACs | Total MACs | Embedding reads |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Selected suffix setup | 102 | 25.381M | 8.192M | 33.573M | 57,600 |

| Slice diagnostic | Control | Control Recall@100 | Treatment | Treatment Recall@100 | Point delta | Users | Targets |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tail target: low | Best-G1 item-ID baseline | 0.031 | Selected suffix setup | 0.036 | +0.005 | 2184 | 6663 |
| Tail target: middle | Best-G1 item-ID baseline | 0.079 | Selected suffix setup | 0.083 | +0.004 | 2115 | 6413 |
| Tail target: high | Best-G1 item-ID baseline | 0.245 | Selected suffix setup | 0.246 | +0.001 | 2090 | 6516 |
| History has collided SID | Best-G1 item-ID baseline | 0.126 | Selected suffix setup | 0.131 | +0.005 | 3266 | 19048 |
| History has no collided SID | Best-G1 item-ID baseline | 0.107 | Selected suffix setup | 0.109 | +0.002 | 148 | 544 |

| Method (unavailable diagnostics) | Not committed |
| :--- | :---: |
| Selected suffix setup | target-bucket slices, artifact bytes, tokenizer fit time, epoch time, peak memory, dedicated full-catalog latency |

## RQ3 — What RQ-KMeans setup works best without collision resolution?

| Method | Recall@100 | Delta Recall@100 | NDCG@100 | MRR@100 | Coverage@100 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RQ0 suffix setup (3 levels × 512 shared codes; 20 iterations)** | 0.127 | baseline | 0.049 | 0.047 | 0.278 |
| Selected tokenizer without suffix (2 levels × 4096 shared codes; 20 iterations) | 0.127 | +0.40% (0.127) | +1.33% (0.050) | +2.54% (0.049) | -2.16% (0.272) |

| Method (SID diagnostics) | Exact SID Recall@100 | Prefix L1 | Prefix L2 | Prefix L3 | ICR | Collided items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RQ0 suffix setup | 0.133 | 0.511 | 0.160 | 0.133 | 8.36% | 14.62% |
| Selected tokenizer without suffix | 0.132 | 0.250 | 0.132 | — | 6.97% | 12.98% |

| Tokenizer (intrinsic diagnostics) | p95 load by level | p95 / mean by level | Intra-code cosine by level | Reconstruction MSE by depth | Dead-code fraction by level |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RQ0 suffix tokenizer | 107 / 122 / 152 | 1.653 / 1.884 / 2.348 | 0.776 / 0.392 / 0.245 | — | — |
| Selected tokenizer without suffix | 17 / 28 | 2.101 / 3.460 | 0.865 / 0.333 | 0.126 / 0.066 | 0.00% / 0.00% |

| Tokenizer (collision diagnostics) | Unique base tuples | ICR | Collided items | Suffix symbols | Bucket p50 / p95 / p99 / max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RQ0 suffix tokenizer | 30378 | 8.36% | 14.62% | 29 | 1 / 2 / 3 / 28 |
| Selected tokenizer without suffix | 30837 | 6.97% | 12.98% | 0 | 1 / 2 / 2 / 8 |

| Method (serving-cost estimate) | Sequence tokens | Transformer MACs | Tokenizer MACs | Total MACs | Embedding reads |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RQ0 suffix setup | 102 | 25.381M | 8.192M | 33.573M | 57,600 |
| Selected tokenizer without suffix | — | — | — | — | — |

| Slice diagnostic | Control | Control Recall@100 | Treatment | Treatment Recall@100 | Point delta | Users | Targets |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Tail target: low | Best-G1 item-ID baseline | 0.031 | RQ0 suffix setup | 0.036 | +0.005 | 2184 | 6663 |
| Tail target: middle | Best-G1 item-ID baseline | 0.079 | RQ0 suffix setup | 0.083 | +0.004 | 2115 | 6413 |
| Tail target: high | Best-G1 item-ID baseline | 0.245 | RQ0 suffix setup | 0.246 | +0.001 | 2090 | 6516 |
| History has collided SID | Best-G1 item-ID baseline | 0.126 | RQ0 suffix setup | 0.131 | +0.005 | 3266 | 19048 |
| History has no collided SID | Best-G1 item-ID baseline | 0.107 | RQ0 suffix setup | 0.109 | +0.002 | 148 | 544 |

| Method (unavailable diagnostics) | Not committed |
| :--- | :---: |
| Selected tokenizer without suffix | tail, target/history collision slices, model parameters, artifact bytes, tokenizer fit time, epoch time, peak memory, MACs, embedding reads, dedicated full-catalog latency |

## Aggregated improvement

### Native Yambda-50M

| Method | Recall@100 | Delta Recall@100 | NDCG@100 | Delta NDCG@100 | MRR@100 | Delta MRR@100 | Coverage@100 | Delta Coverage@100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Original G1 item-ID baseline | 0.101 | baseline | 0.037 | baseline | 0.030 | baseline | 0.593 | baseline |
| Best-G1 plus terminal SID history | 0.130 | +0.029 (+28.70%) | 0.052 | +0.015 (+41.33%) | 0.050 | +0.020 (+66.84%) | 0.268 | -0.325 (-54.84%) |

| Dataset (component arithmetic) | Metric | Original baseline | Aggregate | Point gain | Percent gain | Best-G1 gain | Terminal SID marginal | Standalone sum | Interaction gap | Interaction band | Interaction resolution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Native 50M | Recall@100 | 0.101 | 0.130 | +0.029 | +28.70% | +0.024 | +0.005 | +0.029 | +0.000 | 0.020 | unresolved |
| Native 50M | NDCG@100 | 0.037 | 0.052 | +0.015 | +41.33% | +0.015 | +0.000 | +0.015 | +0.000 | 0.008 | unresolved |
| Native 50M | MRR@100 | 0.030 | 0.050 | +0.020 | +66.84% | +0.019 | +0.001 | +0.020 | +0.000 | 0.007 | unresolved |
| Native 50M | Coverage@100 | 0.593 | 0.268 | -0.325 | -54.84% | -0.343 | +0.018 | -0.325 | +0.000 | 0.505 | unresolved |
