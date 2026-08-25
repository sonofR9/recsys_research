# RQ1 — μP learning-rate transfer

**The native-50M section below is superseded by
[the transfer study](transfer_study.md).** Its runs pre-date the
annealing-horizon rule and most of them stopped mid-decay, so the regret
columns compare runs that spent different fractions of the same schedule. The
width-256 regret in particular reverses on the reran surface. The rows are kept
as the record of what those artifacts contain, not as a current result.

## Native-50M model-width evidence

The current training-semantics revision 2 surface uses the native 50M cohort,
effective batch 1280, epoch-wise validation, early stopping and restored best
weights. Learning rates are listed as embedding/deep. The control rates
0.032/0.012 are transferred to every width and compared with each width's
independently closed optimum.

Absolute regret is tuned recall minus transferred-LR recall. Relative regret
divides that difference by tuned recall.

| width | transferred LR | transferred recall@100 | tuned LR | tuned recall@100 | absolute regret | relative regret | observed surface |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.032/0.012 | 0.071479 | 0.032/0.024 | 0.076197 | 0.004718 | 6.19% | different grid optimum |
| 32 | 0.032/0.012 | 0.076076 | 0.032/0.012 | 0.076076 | 0 | 0% | same grid optimum |
| 64 | 0.032/0.012 | 0.067707 | 0.032/0.012 | 0.067707 | 0 | 0% | same grid optimum |
| 128 | 0.032/0.012 | 0.065372 | 0.032/0.012 | 0.065372 | 0 | 0% | same grid optimum |
| 256 | 0.032/0.012 | 0.042882 | 0.008/0.012 | 0.053937 | 0.011055 | 20.50% | different grid optimum |

On this single-seed 50M proxy surface, the control rates are also the selected
grid optimum at widths 32, 64 and 128. Widths 16 and 256 select different rates,
with observed absolute/relative regrets of 0.004718/6.19% and 0.011055/20.50%.
This locates a central range where the tested grid transfers exactly, but does
not estimate repeat-to-repeat uncertainty or establish a categorical
width-universal transfer result.

All 50M rows use seed 42 and validation-resolved artifacts under
`generated/logs/`. Shared/oracle artifact pairs are:

| width | shared-rate artifact | local-oracle artifact |
| --- | --- | --- |
| 16 | `g1_rqtune_dimension_16_e32d12_mup_ts2_r2_50m` | `g1_rqtune_dimension_16_e32d24_boundary1_cap40_ts2_r2_50m` |
| 32 | `g1_rqtune_dimension_32_e32d12_capcont_cap40_ts2_r2_50m` | same artifact |
| 64 | `g1_rqtune_architecture_control_e32d12_capcont_cap40_ts2_r2_50m` | same artifact |
| 128 | `g1_rqtune_dimension_128_e32d12_capcont_cap40_ts2_r2_50m` | same artifact |
| 256 | `g1_rqtune_dimension_256_e32d12_mup_ts2_r2_50m` | `g1_rqtune_dimension_256_e8d12_capcont_cap40_ts2_r2_50m` |

## Native-500M common-rate confirmations

Widths 16, 32, 64, 128 and 256 have validation-resolved native-500M
confirmations at the width-32 rates 0.032/0.012. The comparison run reuses each
width's own native-50M selected rates on native 500M. “50M-local LR” therefore
identifies the proxy-selected alternative; it is not a 500M-tuned oracle.

| width | common LR | common best/stopped epoch | common recall@100 | common ndcg@100 | 50M-local LR | local best/stopped epoch | local recall@100 | local ndcg@100 | common − local recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.032/0.012 | 19/22 | 0.120080 | 0.045112 | 0.032/0.024 | 19/22 | 0.120226 | 0.045114 | -0.000146 |
| 32 | 0.032/0.012 | 14/17 | 0.130536 | 0.049940 | 0.032/0.012 | 14/17 | 0.130536 | 0.049940 | 0 |
| 64 | 0.032/0.012 | 14/17 | 0.134608 | 0.051592 | 0.032/0.012 | 14/17 | 0.134608 | 0.051592 | 0 |
| 128 | 0.032/0.012 | 14/17 | 0.134080 | 0.051393 | 0.032/0.012 | 14/17 | 0.134080 | 0.051393 | 0 |
| 256 | 0.032/0.012 | 10/13 | 0.133794 | 0.051344 | 0.008/0.012 | 19/22 | 0.127578 | 0.048388 | +0.006216 |

At width 16, the common and 50M-local choices differ by less than the shared
native-500M recall@100 resolution band of 0.002150. Widths 32, 64 and 128 use
the same run for both roles. At width 256, the common rates outperform the
50M-local choice by 0.006216 recall@100 on this single seed, reversing the
native-50M ordering. This is evidence that the common rates can train all five
tested widths and can be competitive on native 500M. It is not evidence that
0.032/0.012 is the 500M optimum, nor that LR rankings transfer unchanged
between dataset sizes.

All 500M rows use seed 42 and validation-resolved artifacts under
`generated/logs/`:

| width | common-rate artifact | 50M-local-rate artifact |
| --- | --- | --- |
| 16 | `g1_rqtune_rqfinal_dimension_16_e0p032_d0p012_b1280_cap40_ts2_r2_500m` | `g1_rqtune_rqfinal_dimension_16_e0p032_d0p024_b1280_cap40_ts2_r2_500m` |
| 32 | `g1_rqtune_rqfinal_dimension_32_e0p032_d0p012_b1280_ts2_r2_500m` | same artifact |
| 64 | `g1_rqtune_rqfinal_architecture_control_e0p032_d0p012_b1280_ts2_r2_500m` | same artifact |
| 128 | `g1_rqtune_rqfinal_dimension_128_e0p032_d0p012_b1280_ts2_r2_500m` | same artifact |
| 256 | `g1_rqtune_rqfinal_dimension_256_e0p032_d0p012_b1280_ts2_r2_500m` | `g1_rqtune_rqfinal_dimension_256_e0p008_d0p012_b1280_cap40_ts2_r2_500m` |

The recall@100 resolution band is the sample standard deviation
`0.0021501867108938925` from the ten accepted-control repeats, seeds 0–9, in
`scratchpad/baseline_spread_500m.json`. It is an empirical resolution band,
not a confidence interval or a treatment-specific significance test.

## Dataset-size conclusion

This width surface and the one-shot confirmations test performance after
reusing 50M-selected rates on Yambda-500M; they do not test whether the 50M
optimum or LR ranking remains optimal on 500M. The current native-size
confirmation is a separate conventional homework control, not a μP width
treatment: it reuses LR 0.001/0.002 and batch 1280 from native 50M on native
500M, where recall@100 is 0.127362. Selected treatment-specific 50M rates are
also reused unchanged in one-shot native-500M confirmations, without 500M
retuning.
