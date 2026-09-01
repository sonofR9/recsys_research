# G6: RQ-KMeans semantic IDs in history

## Native Yambda-500M rerun status

The approved native-500M RQ0–RQ3 replacement is running. Control batches
`2677d36973e544e8b1140adb64bab437` and
`2d31aee728a3412b8d2d6a84998cf093` are unusable because an unrelated Arcadia
read watchdog sent `SIGTERM` to their workers; their raw artifacts remain in
audit storage. The watchdog is stopped, and clean retry
`241170a4b7a54d4f84adbbc12c537f9a` started all 24 compiler-authenticated
control cells without a failure. The native-50M tables below remain the active
completed evidence until the native-500M dependency chain is selection-resolved.

Native Yambda-50M; all methods retrieve concrete item IDs from the full mapped catalog. RQ0–RQ3 use their approved fixed bands: 0.002 Recall@100, 0.002 NDCG@100, 0.003 MRR@100, and 0.03 Coverage@100.
RQ-KMeans uses one shared codebook size at every residual level; SID retrieval and geometry metrics are diagnostics and never select a treatment.
The generated [RQ0 compact tables](evidence/rq0_reader_native50m.md), [RQ1–RQ3 compact tables](evidence/rq1_plus_reader_native.md), [RQ0 tuning ledger](evidence/rq0_tuning_native50m.md), and [RQ1–RQ3 tuning ledger](evidence/rq1_plus_tuning_native.md) are retained with the audited evidence.

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

- **Trainable SID event:** concatenate learned level embeddings and level tags, then project one token per item with DenseNet.
- **Item ID + frozen SID event:** concatenate the item embedding with frozen RQ-KMeans centroids, then project one token per item with DenseNet.
- **Item ID + trainable/frozen SID event:** add learned SID embeddings to the preceding item-plus-centroid event before its DenseNet projection.
- **Controlled learned-SID addition:** preserve the frozen-SID event path and inject learned SID embeddings through a local zero-initialized residual, optionally bounded by `bound × tanh(raw gate)`.
- **Trainable SID tokens:** expand each item into one learned token per SID level while retaining the same 100 history items.
- **Trainable/frozen SID tokens:** concatenate learned SID embeddings and frozen centroids in every expanded level token.
- **Frozen SID tokens:** use frozen centroids as the expanded level tokens.
- **Interleaved item ID/SID tokens:** emit an item-ID token followed by its learned SID-level tokens for every retained history item.

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

Item ID plus frozen SID event fusion is the RQ0 winner: 3 levels, 512 shared codes per level, and DenseNet width 128 improve Recall@100 by 0.005, beyond the 0.002 band, without a beyond-band NDCG regression.
The gain is largest for low- and middle-frequency targets; the no-collision slice is too small for a strong collision-specific claim.
On the original G1 backbone, the +0.001 Recall@100 bridge remains inside the band, so transfer is unresolved.
The controlled learned-SID addition is rejected: bounding improves Recall@100 from 0.123 to 0.126, but the best bound 0.05 remains below the item-ID-plus-frozen-SID event control's 0.130 Recall and 0.052 NDCG outside both bands.
The zero-bound row disables the residual but is not empirical parity because it retains the treatment embedding LR 0.256 while the independently tuned item-ID-plus-frozen-SID event control uses 0.362; per-run semantic-cache hashes and replay of coverage/SID proxies also remain unavailable, so those diagnostics do not select the treatment.

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

Content-informed PCA initialization raises the four-seed mean Recall@100 by 0.006 and NDCG@100 by 0.003, both beyond the fixed 0.002 bands.
It is slower on every paired confirmation seed: 9.25 versus 7.00 mean epochs to 95%, with lower normalized Recall AUC.
Saved checks find no hash, scale, gradient, reconstruction, or frozen-weight defect; the initialized 32D view is a deterministic projection of frozen centroids already present in the event representation.
Because the approved protocol does not define whether convergence speed overrides a beyond-band final-quality gain, the RQ1 selection remains unresolved pending user validation.

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

RQ2 searches the collision-resolving tokenizer while keeping one shared codebook size at every residual level.
The independent search reselects the RQ0 configuration exactly: 3 levels × 512 shared codes with 20 K-Means iterations.
The duplicate confirmation rows therefore have identical quality and collision diagnostics.
This suffix configuration remains the terminal collision-resolving tokenizer.

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

RQ3 removes the collision-resolution suffix and retrieves through the base SID tuple alone.
Its selected tokenizer uses 2 levels × 4096 shared codes with 20 K-Means iterations.
The +0.0005 Recall@100 change versus the suffix control is inside the 0.002 promotion band, so the RQ0/RQ2 suffix tokenizer remains terminal.
No no-suffix slice evidence was committed, so the report makes no collision-mechanism claim.

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

Best-G1 plus terminal SID history reaches Recall@100 0.130 at 50M.
Relative to the original G1 item-ID baseline, the Recall gain is +0.029 (+28.70%) and the NDCG gain is +0.015 (+41.33%).
The component sums equal the observed gains by construction because the terminal SID treatment is applied directly to each best-G1 baseline; all interaction gaps are zero and unresolved.
