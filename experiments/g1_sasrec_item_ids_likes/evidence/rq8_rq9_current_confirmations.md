# RQ8/RQ9 confirmation audit

This is an audit note, not a competing result report. Generated tables in
`scratchpad/research_questions_50m.md` and
`scratchpad/research_questions_500m.md` are authoritative; the reader-facing
interpretation is in `README.md`.

## RQ8 — superseded confirmation surface

The two continuations that blocked this question landed as
`g1_rqtune_rqfinal_heads_mha4_e0p008_d0p012_b1280_cap40_ts2_r2_500m` and
`g1_rqtune_rqfinal_heads_mha8_e0p016_d0p003_b1280_cap40_ts2_r2_500m`. That
historical surface completed its declared comparisons, but it is no longer the
active RQ8 evidence: FFN belongs to RQ4, and the query and sequence-length axes
are being retested entirely on native 500M under
`protocol/rq8_reinvestigation_plan.md`. The old artifacts remain available for
audit and do not enter the corrected RQ8 tables.

## RQ9 — timestamp representation

RQ9 is complete for the declared 15-treatment surface. Every row below is the
single native-500M confirmation of its native-50M-selected configuration, uses
effective batch 1280, validates every epoch, and restores the best recall@100
checkpoint. Arms launched before the annealing-horizon rule stopped after three
non-improving epochs; arms launched after it train the full 20-epoch horizon,
so the surface spans 9 to 23 trained epochs. The no-time row is the literal
control. The shared native-500M absolute resolution bands are
0.00215019 recall@100 and 0.00095122 ndcg@100.

| time representation | selected LR | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | --- | --- | --- | --- | --- |
| no time feature | 0.128/0.024 | 0.121 | 0.0453 | 0.0233 | 0.0185 | 0.73 |
| 8 log-spaced bins, add | 0.032/0.024 | +13% (0.137) | +16% (0.0524) | +23% (0.0287) | +25% (0.0231) | -9% (0.66) |
| 16 log-spaced bins, add | 0.032/0.012 | +11% (0.135) | +14% (0.0516) | +20% (0.0280) | +22% (0.0225) | -21% (0.58) |
| 32 log-spaced bins, add | 0.064/0.012 | +12% (0.136) | +14% (0.0518) | +22% (0.0284) | +22% (0.0225) | -32% (0.50) |
| 64 log-spaced bins, add | 0.064/0.024 | +11% (0.134) | +12% (0.0508) | +17% (0.0274) | +17% (0.0216) | -15% (0.62) |
| 32 bins + raw reverse RoPE | 0.064/0.012 | +14% (0.137) | +16% (0.0523) | +23% (0.0288) | +22% (0.0227) | -10% (0.66) |
| 32 bins + log forward RoPE | 0.032/0.024 | +13% (0.137) | +16% (0.0527) | +25% (0.0291) | +26% (0.0233) | -12% (0.65) |
| 32 bins, concatenate-and-project | 0.008/0.012 | +1% (0.122) | +1% (0.0459) | +4% (0.0243) | +5% (0.0194) | -44% (0.41) |
| clipped linear delta, add | 0.128/0.012 | +9% (0.131) | +9% (0.0494) | +13% (0.0263) | +12% (0.0208) | -2% (0.72) |
| log delta, add | 0.128/0.024 | +7% (0.129) | +9% (0.0494) | +14% (0.0267) | +14% (0.0212) | +3% (0.75) |
| log delta, concatenate-and-project | 0.032/0.024 | +12% (0.135) | +16% (0.0525) | +23% (0.0288) | +26% (0.0234) | -25% (0.55) |
| raw elapsed-time RoPE, forward | 0.128/0.024 | +5% (0.127) | +6% (0.0482) | +10% (0.0257) | +10% (0.0204) | +7% (0.78) |
| raw elapsed-time RoPE, reverse | 0.128/0.024 | +5% (0.126) | +6% (0.0480) | +10% (0.0258) | +10% (0.0204) | +5% (0.77) |
| log elapsed-time RoPE, forward | 0.128/0.024 | -3% (0.117) | -3% (0.0439) | -4% (0.0224) | -3% (0.0180) | -42% (0.42) |
| log elapsed-time RoPE, reverse | 0.032/0.012 | -11% (0.107) | -10% (0.0407) | -9% (0.0212) | -7% (0.0171) | -8% (0.68) |

The 32-bin plus raw-reverse combination is only the numerical recall leader:
its +0.000441 margin over 32-bin plus log-forward RoPE, +0.000665 over 8-bin
addition, and +0.002050 over log-delta concatenation are all inside the recall
band, and its NDCG is the lowest of those four. The proxy ranking does not
transfer: log elapsed-time RoPE forward is
the 50M numerical recall leader at 0.084 but falls to 0.117 on 500M, whereas
32-bin plus raw-reverse is 0.072 on 50M and 0.137 on 500M. RQ9 therefore
supports target-regime selection, not promotion from the 50M ranking.
