# G3 RQ3 prediction-embedding tuning

## RQ3 — Catalog target

### Learned item-ID

| Trial | Source | Embedding LR | Deep LR | Horizon | Restored epoch | Recall@100 | NDCG@100 | MRR@100 | Coverage@100 | Logged train s | Queue wall s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| rq3_output_learned:01 | reused RQ2 | 0.1064948942 | 0.1258268291 | 25 | 15 | 0.036518 | 0.013462 | 0.012606 | 0.003047 | 42.6 | 132.6 |
| rq3_output_learned:02 | reused RQ2 | 0.1474458978 | 0.03243393933 | 25 | 20 | 0.078386 | 0.027766 | 0.024362 | 0.374713 | 45.4 | 134.7 |
| rq3_output_learned:03 | reused RQ2 | 0.3386444716 | 0.03993056713 | 25 | 24 | 0.072309 | 0.026038 | 0.022128 | 0.336883 | 49.9 | 140.1 |
| rq3_output_learned:04 | reused RQ2 | 0.3041556166 | 0.005733564587 | 40 | 15 | 0.087995 | 0.032486 | 0.029197 | 0.798540 | 52.2 | 92.4 |
| rq3_output_learned:05 | reused RQ2 | 0.3041556166 | 0.0040542424 | 40 | 15 | 0.089751 | 0.032829 | 0.026932 | 0.796338 | 50.8 | 92.2 |
| rq3_output_learned:06 | reused RQ2 | 0.3041556166 | 0.002866782294 | 40 | 25 | 0.083030 | 0.030701 | 0.026957 | 0.871485 | 52.4 | 93.5 |
| rq3_output_learned:07 | reused RQ2 | 0.3041556166 | 0.01450668482 | 40 | 19 | 0.093390 | 0.033027 | 0.028600 | 0.843580 | 53.4 | 92.8 |
| **rq3_output_learned:08** | search | 0.2522734462 | 0.01594176841 | 40 | 20 | 0.094191 | 0.033984 | 0.029410 | 0.823368 | 84.8 | 250.4 |
| rq3_output_learned:09 | search | 0.140585716 | 0.0377207395 | 15 | 15 | 0.069997 | 0.025100 | 0.022531 | 0.162001 | 33.5 | 115.4 |

### Frozen pretrained content

| Trial | Source | Embedding LR | Deep LR | Horizon | Restored epoch | Recall@100 | NDCG@100 | MRR@100 | Coverage@100 | Logged train s | Queue wall s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| rq3_output_frozen_content:01 | search | 0.1064948942 | 0.1258268291 | 25 | 12 | 0.020421 | 0.006006 | 0.003376 | 0.007482 | 51.1 | 340.8 |
| rq3_output_frozen_content:02 | search | 0.1474458978 | 0.03243393933 | 25 | 25 | 0.061499 | 0.021973 | 0.019750 | 0.895529 | 45.8 | 337.2 |
| rq3_output_frozen_content:03 | search | 0.3386444716 | 0.03993056713 | 25 | 22 | 0.055397 | 0.018004 | 0.014060 | 0.679498 | 48.8 | 341.5 |
| rq3_output_frozen_content:04 | search | 0.3041556166 | 0.005733564587 | 40 | 23 | 0.059413 | 0.020849 | 0.017990 | 0.846507 | 75.2 | 518.4 |
| rq3_output_frozen_content:05 | search | 0.3041556166 | 0.0040542424 | 40 | 32 | 0.055325 | 0.020154 | 0.019128 | 0.850278 | 76.3 | 519.4 |
| rq3_output_frozen_content:06 | search | 0.3041556166 | 0.002866782294 | 40 | 39 | 0.054610 | 0.019005 | 0.016360 | 0.839538 | 70.1 | 518.8 |
| rq3_output_frozen_content:07 | search | 0.3041556166 | 0.01450668482 | 40 | 26 | 0.061157 | 0.021107 | 0.018340 | 0.894594 | 72.4 | 1080.7 |
| **rq3_output_frozen_content:08** | search | 0.2522734462 | 0.01594176841 | 40 | 24 | 0.063384 | 0.022139 | 0.019643 | 0.897550 | 70.1 | 938.4 |
| rq3_output_frozen_content:09 | search | 0.140585716 | 0.0377207395 | 15 | 14 | 0.046278 | 0.014995 | 0.010342 | 0.523863 | 29.5 | 400.6 |

### Trainable pretrained content

| Trial | Source | Embedding LR | Deep LR | Horizon | Restored epoch | Recall@100 | NDCG@100 | MRR@100 | Coverage@100 | Logged train s | Queue wall s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| rq3_output_trainable_content:01 | search | 0.1064948942 | 0.1258268291 | 25 | 20 | 0.048723 | 0.017957 | 0.017889 | 0.008628 | 63.9 | 400.5 |
| rq3_output_trainable_content:02 | search | 0.1474458978 | 0.03243393933 | 25 | 23 | 0.072341 | 0.026172 | 0.023951 | 0.485188 | 52.2 | 593.7 |
| rq3_output_trainable_content:03 | search | 0.3386444716 | 0.03993056713 | 25 | 23 | 0.067981 | 0.024004 | 0.020989 | 0.157053 | 54.6 | 598.5 |
| rq3_output_trainable_content:04 | search | 0.3041556166 | 0.005733564587 | 40 | 28 | 0.089992 | 0.031972 | 0.025698 | 0.876011 | 89.9 | 956.5 |
| rq3_output_trainable_content:05 | search | 0.3041556166 | 0.0040542424 | 40 | 25 | 0.090236 | 0.033464 | 0.029412 | 0.840775 | 89.3 | 858.4 |
| rq3_output_trainable_content:06 | search | 0.3041556166 | 0.002866782294 | 40 | 39 | 0.086058 | 0.031157 | 0.027916 | 0.888741 | 75.9 | 782.0 |
| rq3_output_trainable_content:07 | search | 0.3041556166 | 0.01450668482 | 40 | 24 | 0.094572 | 0.034629 | 0.031566 | 0.784059 | 82.1 | 852.3 |
| **rq3_output_trainable_content:08** | search | 0.2522734462 | 0.01594176841 | 40 | 34 | 0.095729 | 0.034336 | 0.030049 | 0.865180 | 87.9 | 767.4 |
| rq3_output_trainable_content:09 | search | 0.140585716 | 0.0377207395 | 15 | 14 | 0.054200 | 0.018974 | 0.016174 | 0.053276 | 34.6 | 332.8 |

### Learned ID + frozen content

| Trial | Source | Embedding LR | Deep LR | Horizon | Restored epoch | Recall@100 | NDCG@100 | MRR@100 | Coverage@100 | Logged train s | Queue wall s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| rq3_output_learned_frozen_content:01 | search | 0.1064948942 | 0.1258268291 | 25 | 12 | 0.035865 | 0.012840 | 0.010926 | 0.003107 | 54.9 | 734.6 |
| rq3_output_learned_frozen_content:02 | search | 0.1474458978 | 0.03243393933 | 25 | 25 | 0.088084 | 0.032059 | 0.028991 | 0.688096 | 60.4 | 741.8 |
| rq3_output_learned_frozen_content:03 | search | 0.3386444716 | 0.03993056713 | 25 | 23 | 0.081869 | 0.029420 | 0.026795 | 0.410070 | 52.4 | 510.2 |
| **rq3_output_learned_frozen_content:04** | search | 0.3041556166 | 0.005733564587 | 40 | 18 | 0.100520 | 0.036951 | 0.034017 | 0.738446 | 84.7 | 969.9 |
| rq3_output_learned_frozen_content:05 | search | 0.3041556166 | 0.0040542424 | 40 | 18 | 0.097587 | 0.035809 | 0.030776 | 0.758145 | 92.9 | 988.3 |
| rq3_output_learned_frozen_content:06 | search | 0.3041556166 | 0.002866782294 | 40 | 22 | 0.089752 | 0.033483 | 0.030187 | 0.803970 | 79.7 | 1060.0 |
| rq3_output_learned_frozen_content:07 | search | 0.3041556166 | 0.01450668482 | 40 | 32 | 0.096963 | 0.033840 | 0.028063 | 0.872270 | 81.2 | 942.7 |
| rq3_output_learned_frozen_content:08 | search | 0.2522734462 | 0.01594176841 | 40 | 25 | 0.100470 | 0.036143 | 0.031181 | 0.831543 | 82.5 | 973.4 |
| rq3_output_learned_frozen_content:09 | search | 0.140585716 | 0.0377207395 | 15 | 15 | 0.063499 | 0.023153 | 0.021574 | 0.130747 | 31.1 | 494.8 |
| rq3_output_learned_frozen_content_deep_lr_lower_boundary:01 | lower-boundary | 0.3041556166 | 0.0020271212 | 40 | 39 | 0.089163 | 0.033230 | 0.030542 | 0.880385 | 55.5 | 81.8 |
| rq3_output_learned_frozen_content_deep_lr_lower_boundary:02 | lower-boundary | 0.3041556166 | 0.001433391147 | 40 | 36 | 0.085975 | 0.032090 | 0.028970 | 0.878816 | 55.6 | 83.0 |
| rq3_output_learned_frozen_content_deep_lr_lower_boundary:03 | lower-boundary | 0.3041556166 | 0.0010135606 | 40 | 34 | 0.081751 | 0.030560 | 0.028220 | 0.880325 | 53.3 | 81.2 |

### Learned ID + trainable content

| Trial | Source | Embedding LR | Deep LR | Horizon | Restored epoch | Recall@100 | NDCG@100 | MRR@100 | Coverage@100 | Logged train s | Queue wall s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| rq3_output_learned_trainable_content:01 | search | 0.1064948942 | 0.1258268291 | 25 | 19 | 0.035225 | 0.012711 | 0.011030 | 0.004495 | 58.2 | 840.4 |
| rq3_output_learned_trainable_content:02 | search | 0.1474458978 | 0.03243393933 | 25 | 23 | 0.081940 | 0.029445 | 0.025834 | 0.619374 | 56.5 | 618.1 |
| rq3_output_learned_trainable_content:03 | search | 0.3386444716 | 0.03993056713 | 25 | 21 | 0.068960 | 0.024229 | 0.021499 | 0.232020 | 62.5 | 569.9 |
| **rq3_output_learned_trainable_content:04** | search | 0.3041556166 | 0.005733564587 | 40 | 18 | 0.096229 | 0.035905 | 0.031233 | 0.707886 | 79.2 | 886.7 |
| rq3_output_learned_trainable_content:05 | search | 0.3041556166 | 0.0040542424 | 40 | 21 | 0.091230 | 0.033921 | 0.029711 | 0.810094 | 78.6 | 900.9 |
| rq3_output_learned_trainable_content:06 | search | 0.3041556166 | 0.002866782294 | 40 | 32 | 0.090867 | 0.033335 | 0.029319 | 0.876523 | 86.6 | 798.0 |
| rq3_output_learned_trainable_content:07 | search | 0.3041556166 | 0.01450668482 | 40 | 32 | 0.094419 | 0.033196 | 0.029229 | 0.864909 | 76.2 | 801.8 |
| rq3_output_learned_trainable_content:08 | search | 0.2522734462 | 0.01594176841 | 40 | 40 | 0.090606 | 0.032131 | 0.028232 | 0.895589 | 79.6 | 720.3 |
| rq3_output_learned_trainable_content:09 | search | 0.140585716 | 0.0377207395 | 15 | 14 | 0.058847 | 0.020843 | 0.017557 | 0.058344 | 30.3 | 550.1 |
| rq3_output_learned_trainable_content_deep_lr_lower_boundary:01 | lower-boundary | 0.3041556166 | 0.0020271212 | 40 | 23 | 0.085934 | 0.032460 | 0.030216 | 0.843460 | 60.2 | 87.9 |
| rq3_output_learned_trainable_content_deep_lr_lower_boundary:02 | lower-boundary | 0.3041556166 | 0.001433391147 | 40 | 24 | 0.087180 | 0.032812 | 0.029942 | 0.837396 | 61.2 | 87.6 |
| rq3_output_learned_trainable_content_deep_lr_lower_boundary:03 | lower-boundary | 0.3041556166 | 0.0010135606 | 40 | 33 | 0.080031 | 0.028605 | 0.024443 | 0.898455 | 60.3 | 87.7 |
