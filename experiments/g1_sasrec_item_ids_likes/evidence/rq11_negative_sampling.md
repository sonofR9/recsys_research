# RQ11 — Historical negative-sampling diagnostic evidence

These rows are audit-only and excluded from the active native-500M RQ11
selection and reader table. Their settings were selected on 50M, and the mixed
objectives do not implement the fully corrected aggregate proposal in the
approved RQ11 plan.

RQ11 compares eight separately tuned negative-sampling families. Yambda-50M
selects each family's learning rates and secondary settings; those settings
transfer unchanged to one native Yambda-500M confirmation. All runs use the
current training-semantics revision, effective batch 1280, the selected
dimension-64 architecture, validation every epoch, recall@100 early stopping
with patience three, and restored best weights. The three 500M treatments that
reached the initial 20-epoch cap were continued with a 40-epoch safety cap and
the original 20-epoch learning-rate horizon. All eight final artifacts are now
selection-resolved.

## Selected Yambda-50M configurations

| negative proposal and correction | embedding LR | deep LR | negatives | logQ alpha | random fraction | recall@100 | ndcg@100 | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed in-batch global q, Yi-2019 | 0.064 | 0.003 | 2,048 | — | — | 0.06481260 | 0.02360984 | 18 |
| fixed in-batch leave-one-out reference | 0.016 | 0.006 | 2,048 | — | — | 0.04475483 | 0.01572838 | 6 |
| streaming in-batch global q, Yi-2019 | 0.016 | 0.024 | 512 | 0.005 | — | 0.06235607 | 0.02202313 | 18 |
| uniform random, uncorrected | 0.032 | 0.012 | 512 | — | — | 0.06770697 | 0.02420040 | 19 |
| **popularity random global q, Yi-2019** | **0.128** | **0.012** | **512** | **—** | **—** | **0.07876568** | **0.02985018** | **20** |
| in-batch, uncorrected | 0.032 | 0.024 | 512 | — | — | 0.04703149 | 0.01536342 | 14 |
| uniform random + streaming-logQ in-batch, negative-only correction | 0.016 | 0.006 | 512 | 0.01 | 0.875 | 0.04510345 | 0.01577537 | 6 |
| uniform random + fixed-logQ in-batch, negative-only correction | 0.032 | 0.024 | 512 | — | 0.25 | 0.07607198 | 0.02736005 | 19 |

## Transferred Yambda-500M results

| negative proposal and correction | embedding LR | deep LR | negatives | logQ alpha | random fraction | recall@100 | ndcg@100 | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed in-batch global q, Yi-2019 | 0.064 | 0.003 | 2,048 | — | — | 0.12170044 | 0.04634026 | 15 |
| fixed in-batch leave-one-out reference | 0.016 | 0.006 | 2,048 | — | — | 0.11638217 | 0.04412851 | 20 |
| streaming in-batch global q, Yi-2019 | 0.016 | 0.024 | 512 | 0.005 | — | 0.13251981 | 0.05078518 | 19 |
| **uniform random, uncorrected** | **0.032** | **0.012** | **512** | **—** | **—** | **0.13460835** | **0.05159223** | **14** |
| popularity random global q, Yi-2019 | 0.128 | 0.012 | 512 | — | — | 0.13338309 | 0.05135822 | 11 |
| in-batch, uncorrected | 0.032 | 0.024 | 512 | — | — | 0.06393058 | 0.02284273 | 15 |
| uniform random + streaming-logQ in-batch, negative-only correction | 0.016 | 0.006 | 512 | 0.01 | 0.875 | 0.12290317 | 0.04684292 | 20 |
| uniform random + fixed-logQ in-batch, negative-only correction | 0.032 | 0.024 | 512 | — | 0.25 | 0.12948204 | 0.04946398 | 15 |

The raw 500M leader is uniform random. Its recall advantage over popularity
random is 0.00123 and over streaming global q is 0.00209; both are within the
shared 500M recall@100 resolution band of 0.00215. The three leading
single-proposal treatments are therefore unresolved relative to one another.
All three outperform fixed global q by much more than that band. The proxy
ranking does not transfer exactly: fixed global q leads streaming global q on
50M, while streaming leads fixed on 500M.

The fixed-logQ mixed treatment is stronger than either pure fixed in-batch
treatment, but weaker than the three leading single-proposal treatments on
500M. Uncorrected in-batch sampling collapses on 500M. Together with the strong
uniform and popularity-random results, this supports proposal diversity and
effective negative-pool size as the likely explanation; it does not show that
logQ correction is generally harmful.

## Matched-grid and tuned-family interpretation

At each of the nine shared points in the initial 3×3 learning-rate grid, the
fixed leave-one-out reference beats otherwise matched uncorrected in-batch
sampling. The recall@100 advantage ranges from 0.00014 to 0.02798. After each
family receives its independent secondary and local tuning, however, the
selected 50M optima reverse: 0.04475483 for leave-one-out and 0.04703149 for
uncorrected in-batch. The matched grid answers the objective comparison at
shared optimizer settings; the independently tuned result answers the best
observed recipe for each family. Neither result licenses a general claim that
logQ is worse.

## Homework-matched objective control

This control keeps the calibrated homework architecture, native data protocol,
effective batch 1,280, 512 negatives, constant schedule, validation cadence,
early stopping, initialization, and all other training settings fixed. It
independently tunes embedding/deep LR for baseline fixed in-batch leave-one-out
logQ and uncorrected uniform-random negatives.

| objective | embedding LR | deep LR | 50M recall@100 | 50M ndcg@100 | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| **fixed in-batch leave-one-out logQ** | **0.001** | **0.002** | **0.10024002** | **0.03746495** | **56** |
| uniform random | 0.002 | 0.004 | 0.09353508 | 0.03517140 | 39 |

Both LR surfaces are cap-resolved and closed on lower and upper transverse
neighbors. Their proxy ranking does not transfer: uniform random is lower on
50M but higher on the native 500M confirmations.

| objective | embedding LR | deep LR | 500M recall@100 | 500M ndcg@100 | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed in-batch leave-one-out logQ | 0.001 | 0.002 | 0.12736188 | 0.04771098 | 20 |
| **uniform random** | **0.002** | **0.004** | **0.13158371** | **0.04963377** | **19** |

The random-minus-logQ difference is +0.00422182 recall@100 (+3.3148%) and
+0.00192279 ndcg@100 (+4.0301%). Both exceed the shared absolute resolution
bands of 0.00215019 and 0.00095122, respectively. These bands are empirical
resolution thresholds, not treatment-specific significance tests.

The selected fixed-logQ configuration is exactly the accepted seed-42 native
500M transfer control
`g1_transfer_selected_native50_af8b8a8133c7_e0p001_d0p002_cap40_ts2_r2_500m`:
the expected homework config has zero top-level metadata differences and zero
transfer-invariant differences from that artifact, whose stopping rule resolved
at best epoch 20 and stopped epoch 23. No duplicate 500M run was needed. Kept
separately as spread context, the ten-repeat unchanged-control mean is
0.12762411 recall@100 and 0.04837984 ndcg@100; it is not the selected single-run
comparison above.

This conventional baseline is not the fixed global-q Yi-2019 row above. That
row uses fully corrected logits, the selected sequence-128/SwiGLU/GQA/window-50
architecture, a linear schedule, independently tuned 0.064/0.003 rates, and
2,048 negatives, so their numeric difference does not isolate logQ correction.
