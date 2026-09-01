# G4: future-item supervision

This experiment tests broader future-positive sets on native Yambda-500M with
the fixed two-layer G1-derived baseline, batch size 512, SwiGLU width 192, and a
15-epoch one-cycle cosine horizon. The embedding LR is fixed; deep LR is the
only tuned field. Every selected run restores epoch 10.

Operational thresholds scale G1's reviewed native-500M relative dispersions to
this control: 0.002586 Recall@100, 0.001210 NDCG@100, and 0.061553
Coverage@100. They are practical single-run bands, not significance tests.

## RQ1: Does a 24-hour future window help?

| variant | recall@100 | ndcg@100 | coverage@100 | horizon | restored epoch |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **next liked item** | **0.153** | **0.062** | **0.458** | **15** | **10** |
| uniform liked event in the next 24 hours | <span style="color: red">−5.7% (0.145)</span> | <span style="color: red">−4.1% (0.059)</span> | <span style="color: red">−27.3% (0.333)</span> | 15 | 10 |

The treatment draws one strictly later liked-event occurrence uniformly from
the next 24 hours and falls back to the next liked item when none exists. It is
eligible on 5,408,817 of 7,674,702 causal prefixes (70.48%) and falls back on
2,265,885 (29.52%). Its mean candidate-set size is 29.871 occurrences.

| target distance | next liked item recall@100 | next liked item ndcg@100 | uniform liked event in the next 24 hours recall@100 | uniform liked event in the next 24 hours ndcg@100 |
| :--- | :---: | :---: | :---: | :---: |
| 0–6 hours | 0.169 | 0.055 | <span style="color: red">−12.2% (0.149)</span> | <span style="color: red">−3.3% (0.054)</span> |
| 6–24 hours | 0.167 | 0.058 | <span style="color: red">−3.5% (0.161)</span> | <span style="color: red">−4.8% (0.056)</span> |
| 1–3 days | 0.161 | 0.056 | <span style="color: red">−6.8% (0.150)</span> | <span style="color: red">−4.4% (0.054)</span> |
| 3–7 days | 0.140 | 0.051 | <span style="color: red">−5.3% (0.132)</span> | <span style="color: red">−4.1% (0.049)</span> |

| final-window event rank | next liked item recall@100 | next liked item ndcg@100 | uniform liked event in the next 24 hours recall@100 | uniform liked event in the next 24 hours ndcg@100 |
| :--- | :---: | :---: | :---: | :---: |
| first final-window event | 0.177 | 0.050 | <span style="color: red">−6.4% (0.166)</span> | <span style="color: red">−5.9% (0.047)</span> |
| events 2–5 | 0.146 | 0.051 | <span style="color: red">−6.0% (0.137)</span> | <span style="color: red">−3.6% (0.049)</span> |
| events 6–10 | 0.116 | 0.041 | <span style="color: red">−4.9% (0.110)</span> | <span style="color: red">−2.5% (0.040)</span> |
| events 11+ | 0.090 | 0.042 | <span style="color: red">−5.0% (0.086)</span> | <span style="color: red">−3.0% (0.041)</span> |

| user activity | next liked item recall@100 | next liked item ndcg@100 | uniform liked event in the next 24 hours recall@100 | uniform liked event in the next 24 hours ndcg@100 |
| :--- | :---: | :---: | :---: | :---: |
| Q1, least active | 0.178 | 0.062 | <span style="color: red">−7.7% (0.164)</span> | <span style="color: red">−4.8% (0.059)</span> |
| Q2 | 0.172 | 0.068 | <span style="color: red">−7.0% (0.160)</span> | <span style="color: red">−7.4% (0.063)</span> |
| Q3 | 0.148 | 0.061 | <span style="color: red">−4.0% (0.142)</span> | <span style="color: red">−2.5% (0.059)</span> |
| Q4, most active | 0.116 | 0.056 | <span style="color: red">−3.2% (0.112)</span> | −1.2% (0.055) |

Recall@100 falls by 0.008823, more than three times the operational band, and
the loss appears in every target-distance and event-rank slice. The 24-hour
objective is inferior and is not promoted.

## RQ2: Does a next-10-liked-events window help?

| variant | recall@100 | ndcg@100 | coverage@100 | horizon | restored epoch |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **next liked item** | **0.153** | **0.062** | **0.458** | **15** | **10** |
| uniform among the next 10 liked events | <span style="color: red">−2.9% (0.149)</span> | −0.6% (0.061) | <span style="color: red">−28.0% (0.330)</span> | 15 | 10 |
| uniform liked event in the next 24 hours | <span style="color: red">−5.7% (0.145)</span> | <span style="color: red">−4.1% (0.059)</span> | <span style="color: red">−27.3% (0.333)</span> | 15 | 10 |

The treatment draws uniformly among the next ten liked-event occurrences. It
is eligible on every causal prefix and has a mean candidate-set size of 9.596
occurrences.

| target distance | next liked item recall@100 | next liked item ndcg@100 | uniform among the next 10 liked events recall@100 | uniform among the next 10 liked events ndcg@100 |
| :--- | :---: | :---: | :---: | :---: |
| 0–6 hours | 0.169 | 0.055 | <span style="color: red">−5.4% (0.160)</span> | <span style="color: red">−3.7% (0.053)</span> |
| 6–24 hours | 0.167 | 0.058 | <span style="color: red">−2.0% (0.164)</span> | −1.7% (0.057) |
| 1–3 days | 0.161 | 0.056 | <span style="color: red">−4.0% (0.155)</span> | −1.1% (0.055) |
| 3–7 days | 0.140 | 0.051 | <span style="color: red">−2.6% (0.136)</span> | −0.7% (0.051) |

| final-window event rank | next liked item recall@100 | next liked item ndcg@100 | uniform among the next 10 liked events recall@100 | uniform among the next 10 liked events ndcg@100 |
| :--- | :---: | :---: | :---: | :---: |
| first final-window event | 0.177 | 0.050 | <span style="color: red">−3.4% (0.171)</span> | −1.9% (0.049) |
| events 2–5 | 0.146 | 0.051 | <span style="color: red">−3.3% (0.141)</span> | −0.8% (0.051) |
| events 6–10 | 0.116 | 0.041 | −1.4% (0.114) | +1.1% (0.041) |
| events 11+ | 0.090 | 0.042 | −0.2% (0.090) | −0.0% (0.042) |

| user activity | next liked item recall@100 | next liked item ndcg@100 | uniform among the next 10 liked events recall@100 | uniform among the next 10 liked events ndcg@100 |
| :--- | :---: | :---: | :---: | :---: |
| Q1, least active | 0.178 | 0.062 | <span style="color: red">−6.8% (0.165)</span> | <span style="color: red">−2.8% (0.060)</span> |
| Q2 | 0.172 | 0.068 | <span style="color: red">−4.6% (0.164)</span> | <span style="color: red">−4.5% (0.065)</span> |
| Q3 | 0.148 | 0.061 | +0.6% (0.149) | <span style="color: green">+2.2% (0.062)</span> |
| Q4, most active | 0.116 | 0.056 | +1.5% (0.118) | <span style="color: green">+3.6% (0.058)</span> |

Recall@100 falls by 0.004388, beyond the operational band, so the treatment is
not non-inferior and is not promoted. Across the three objectives, increasing
mean candidate breadth from 1 to 9.596 to 29.871 corresponds to monotonically
lower Recall@100: 0.153, 0.149, and 0.145. Both treatments lose most in the
least-active quartiles. The evidence supports the tentative interpretation that
uniform sampling from broader future-positive sets dilutes next-item alignment;
the user validation covers the inferior results and control-retention decision,
not this causal explanation.

## RQ3: Can behavior-similar future periods define better positives?

| stage | authenticated static audit | outcome |
| :--- | :--- | :---: |
| pre-selector feasibility | The approved classifier requires a population-sized in-memory fit and has no external-memory fit path. | **stopped** |

The exact approved `HistGradientBoostingClassifier` path materializes the
complete fit matrix and a complete binned matrix and has no external-memory fit
mode. The user-approved “if it is too hard or will run too long, don't try it”
rule therefore stops RQ3 before selector search. It has no downstream quality
result and contributes no aggregate member; no absolute memory or runtime
projection is claimed.

## Aggregated improvement

No treatment qualifies, so the frozen aggregate candidate is the original
next-item baseline and no duplicate aggregate run is launched.

| metric | baseline | aggregate | gain, points | gain | summed non-overlapping component gains, points | interaction gap, points |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Recall@100 | 0.153 | 0.153 | 0.000 | 0.0% | 0.000 | 0.000 |
| NDCG@100 | 0.062 | 0.062 | 0.000 | 0.0% | 0.000 | 0.000 |
| Coverage@100 | 0.458 | 0.458 | 0.000 | 0.0% | 0.000 | 0.000 |

The zero interaction gaps are arithmetic identities because the aggregate has
no treatment members; they are not interaction claims.
