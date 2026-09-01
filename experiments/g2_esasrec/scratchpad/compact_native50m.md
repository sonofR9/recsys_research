# G2 eSASRec on native Yambda-50M

## RQ1: What are official and local eSASRec's metrics?

| variant | layer | loss | FFN width | gBCE t | uniform | logQ | params | epoch s | train s | wall s | targets/s | peak GB | recall@100 | ndcg@100 | coverage@100 |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: |
| control | G1 control | sampled softmax | 192 | — | — | — | 2222956.000 | 1.383 | 28.527 | 29.886 | 438511.859 | 1.741 | +0.000% (0.115) | +0.000% (0.043) | +0.000% (0.691) |
| ligr sampled softmax | LiGR | sampled softmax | 1024 | — | — | — | 10876416.000 | 2.359 | 38.685 | 40.544 | 257052.624 | 6.608 | <span style="color: green">+18.262% (0.136)</span> | <span style="color: green">+22.138% (0.053)</span> | <span style="color: red">-30.178% (0.483)</span> |
| official rectools | RecTools LiGR | sampled softmax | — | — | — | — | — | — | — | 454.131 | — | — | <span style="color: red">-3.968% (0.111)</span> | <span style="color: green">+9.445% (0.047)</span> | <span style="color: red">-74.376% (0.177)</span> |

## RQ2: What does each pluggable eSASRec component buy?

| loss (standard block, FFN width 256) | recall@100 | ndcg@100 | coverage@100 | gBCE t | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| sampled softmax | 0.128 | 0.049 | 0.426 | — | 43.506 | 280689.561 | 6.576 |
| **gBCE** | <span style="color: green">+2.775% (0.131)</span> | +2.422% (0.050) | <span style="color: red">-22.042% (0.332)</span> | 0.342 | 188.204 | 277675.250 | 6.575 |

| loss (parameter-matched SASRec, FFN width 1792) | recall@100 | ndcg@100 | coverage@100 | gBCE t | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| sampled softmax | 0.125 | 0.049 | 0.408 | — | 33.071 | 272861.339 | 6.608 |
| **gBCE** | <span style="color: green">+5.582% (0.132)</span> | +2.804% (0.051) | -10.183% (0.366) | 0.663 | 202.806 | 272236.531 | 6.608 |

| loss (LiGR, FFN width 1024) | recall@100 | ndcg@100 | coverage@100 | gBCE t | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| **sampled softmax** | 0.136 | 0.053 | 0.483 | — | 40.544 | 257052.624 | 6.608 |
| gBCE | <span style="color: red">-33.891% (0.090)</span> | <span style="color: red">-33.399% (0.035)</span> | <span style="color: green">+50.497% (0.726)</span> | 0.464 | 57.387 | 257361.247 | 6.608 |

| FFN width (standard block, sampled softmax) | recall@100 | ndcg@100 | coverage@100 | params | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| **256** | 0.128 | 0.049 | 0.426 | 9304064.000 | 43.506 | 280689.561 | 6.576 |
| 1792 | <span style="color: red">-2.462% (0.125)</span> | +0.125% (0.049) | -4.248% (0.408) | 10880000.000 | 33.071 | 272861.339 | 6.608 |

| FFN width (standard block, gBCE) | recall@100 | ndcg@100 | coverage@100 | params | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| 256 | 0.131 | 0.050 | 0.332 | 9304064.000 | 188.204 | 277675.250 | 6.575 |
| **1792** | +0.202% (0.132) | +0.498% (0.051) | +10.318% (0.366) | 10880000.000 | 202.806 | 272236.531 | 6.608 |

| block (sampled softmax, parameter-matched) | recall@100 | ndcg@100 | coverage@100 | FFN width | params | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| parameter-matched SASRec | 0.125 | 0.049 | 0.408 | 1792 | 10880000.000 | 33.071 | 272861.339 | 6.608 |
| **LiGR** | <span style="color: green">+9.397% (0.136)</span> | <span style="color: green">+6.968% (0.053)</span> | <span style="color: green">+18.295% (0.483)</span> | 1024 | 10876416.000 | 40.544 | 257052.624 | 6.608 |

| block (gBCE, parameter-matched) | recall@100 | ndcg@100 | coverage@100 | FFN width | params | wall s | targets/s | peak GB |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| **parameter-matched SASRec** | 0.132 | 0.051 | 0.366 | 1792 | 10880000.000 | 202.806 | 272236.531 | 6.608 |
| LiGR | <span style="color: red">-31.502% (0.090)</span> | <span style="color: red">-30.702% (0.035)</span> | <span style="color: green">+98.213% (0.726)</span> | 1024 | 10876416.000 | 57.387 | 257361.247 | 6.608 |

## RQ3: Does mixed sampling improve coverage without a recall loss?

| variant | layer | loss | FFN width | gBCE t | uniform | logQ | params | epoch s | train s | wall s | targets/s | peak GB | recall@100 | ndcg@100 | coverage@100 |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: |
| ligr sampled softmax | LiGR | sampled softmax | 1024 | — | — | — | 10876416.000 | 2.359 | 38.685 | 40.544 | 257052.624 | 6.608 | +0.000% (0.136) | +0.000% (0.053) | +0.000% (0.483) |
| mixed sampler | LiGR | sampled softmax | 1024 | — | 0.6 | none | 10876416.000 | 2.422 | 39.435 | 41.175 | 250268.854 | 4.157 | <span style="color: red">-13.174% (0.118)</span> | <span style="color: red">-14.134% (0.045)</span> | <span style="color: green">+20.104% (0.580)</span> |
| mixed sampler | LiGR | sampled softmax | 1024 | — | 0.6 | yi2019 | 10876416.000 | 2.666 | 43.491 | 45.243 | 227420.304 | 4.157 | <span style="color: red">-2.258% (0.133)</span> | -2.503% (0.051) | +4.957% (0.507) |
| mixed sampler | LiGR | sampled softmax | 1024 | — | 0.4360874991648083 | yi2019 | 10876416.000 | 2.655 | 40.943 | 42.559 | 228308.118 | 4.128 | +0.645% (0.137) | <span style="color: green">+3.932% (0.055)</span> | -8.652% (0.441) |

## Aggregated improvement

| variant | layer | loss | FFN width | gBCE t | uniform | logQ | params | epoch s | train s | wall s | targets/s | peak GB | recall@100 | ndcg@100 | coverage@100 |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: |
| control | G1 control | sampled softmax | 192 | — | — | — | 2222956.000 | 1.383 | 28.527 | 29.886 | 438511.859 | 1.741 | +0.000% (0.115) | +0.000% (0.043) | +0.000% (0.691) |
| ligr sampled softmax | LiGR | sampled softmax | 1024 | — | — | — | 10876416.000 | 2.359 | 38.685 | 40.544 | 257052.624 | 6.608 | <span style="color: green">+18.262% (0.136)</span> | <span style="color: green">+22.138% (0.053)</span> | <span style="color: red">-30.178% (0.483)</span> |

| candidate | qualification | selection | rationale |
| :--- | :--- | :--- | :--- |
| official SASRec block with sampled softmax | qualified | omitted | qualified because Recall@100 improved beyond its size-matched band |
| official SASRec block with gBCE | qualified | omitted | qualified because Recall@100 improved beyond its size-matched band |
| parameter-matched SASRec with sampled softmax | qualified | omitted | qualified because Recall@100 improved beyond its size-matched band |
| parameter-matched SASRec with gBCE | qualified | omitted | qualified because Recall@100 improved beyond its size-matched band |
| LiGR with sampled softmax | qualified | selected | qualified because Recall@100 improved beyond its size-matched band |
| LiGR with gBCE | not qualified | omitted | did not qualify for promotion |
| LiGR with mixed sampling | not qualified | omitted | no candidate met the pre-approved mixed-sampler eligibility rule |

| deterministic selected-recipe reproduction | p50 latency (ms) | p95 latency (ms) | queries/s |
| :--- | ---: | ---: | ---: |
| LiGR with sampled softmax | 3.176 | 3.415 | 80603.214 |

| metric | baseline | aggregate | point gain | percent gain | standalone sum | interaction gap | size-matched band | resolution |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| recall@100 | 0.115 | 0.136 | +0.021 | +18.262% | +0.021 | +0.000 | 0.003 | unresolved |
| ndcg@100 | 0.043 | 0.053 | +0.010 | +22.138% | +0.010 | +0.000 | 0.002 | unresolved |
| coverage@100 | 0.691 | 0.483 | -0.209 | -30.178% | -0.209 | +0.000 | 0.060 | unresolved |
