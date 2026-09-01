# G4: future-item supervision

This experiment tests whether broader later-in-canonical-event-order positive sets improve native Yambda-50M retrieval without changing inference. Batch size is fixed at 512; every objective tunes embedding learning rate, deep learning rate, and linear-schedule horizon.
Operational thresholds scale the reviewed native-50M seeds 42–51 calibration's relative dispersions to this control: 0.020259 Recall@100 and 0.008167 NDCG@100. They are practical single-run bands, not significance tests.

## RQ1: Does a 24-hour future window help?

| variant | recall@100 | ndcg@100 | selected horizon | restored epoch |
| :--- | :---: | :---: | :---: | :---: |
| **next liked item** | **0.104** | **0.038** | **25** | **20** |
| uniform liked event in the next 24 hours | −0.8% (0.104) | +1.2% (0.039) | 25 | 14 |

The control predicts the next liked item in canonical event order. The treatment draws one liked-event occurrence uniformly from a strictly later timestamp within 24 hours and falls back to next-item when the window is empty; because ordering within equal timestamps is canonical, 7.09% of realized fallback targets have zero timestamp distance.

| target distance | next-item recall@100 | 24-hour recall@100 | next-item ndcg@100 | 24-hour ndcg@100 |
| :--- | :---: | :---: | :---: | :---: |
| 0–6 hours | 0.091 | +15.1% (0.105) | 0.020 | <span style="color: green">+42.5% (0.029)</span> |
| 6–24 hours | 0.107 | +7.8% (0.115) | 0.034 | +14.3% (0.038) |
| 1–3 days | 0.110 | +1.0% (0.111) | 0.037 | −3.7% (0.035) |
| 3–7 days | 0.095 | +1.5% (0.097) | 0.033 | +2.9% (0.033) |

| user activity | next-item recall@100 | 24-hour recall@100 | next-item ndcg@100 | 24-hour ndcg@100 |
| :--- | :---: | :---: | :---: | :---: |
| Q1, least active | 0.117 | +1.2% (0.118) | 0.039 | +1.4% (0.040) |
| Q2 | 0.116 | −2.5% (0.113) | 0.040 | −2.5% (0.039) |
| Q3 | 0.096 | −3.1% (0.093) | 0.036 | −1.4% (0.035) |
| Q4, most active | 0.088 | +1.5% (0.090) | 0.037 | +7.4% (0.040) |

The treatment is eligible on 67.64% of 606,267 causal prefixes and uses fallback on 32.36%; both arms retain 606,267 target pairs and 20 optimizer steps per epoch. Recall@100 changes by −0.000802, well inside its operational threshold. The isolated 0–6-hour NDCG gain does not overturn the predeclared overall Recall@100 decision. The 24-hour objective is therefore null and is not promoted.

## RQ2: Does a next-10-liked-events window help?

| variant | recall@100 | ndcg@100 | selected horizon | restored epoch |
| :--- | :---: | :---: | :---: | :---: |
| next liked item | 0.104 | 0.038 | 25 | 20 |
| **uniform among next 10 liked events** | **<span style="color: green">+21.8% (0.127)</span>** | **<span style="color: green">+23.3% (0.047)</span>** | **25** | **24** |
| uniform liked event in the next 24 hours | −0.8% (0.104) | +1.2% (0.039) | 25 | 14 |

The treatment draws uniformly among the next ten liked-event occurrences in canonical event order. It is eligible on every causal prefix, retains the same 606,267 target pairs and 20 optimizer steps per epoch, and masks every unique valid positive item; 10.88% of realized draws share the anchor timestamp but occur later in canonical order.

| target distance | next-item recall@100 | 24-hour recall@100 | next-10 recall@100 | next-item ndcg@100 | next-10 ndcg@100 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 0–6 hours | 0.091 | +15.1% (0.105) | <span style="color: green">+37.5% (0.126)</span> | 0.020 | <span style="color: green">+76.2% (0.035)</span> |
| 6–24 hours | 0.107 | +7.8% (0.115) | <span style="color: green">+38.5% (0.148)</span> | 0.034 | <span style="color: green">+37.2% (0.046)</span> |
| 1–3 days | 0.110 | +1.0% (0.111) | +19.2% (0.131) | 0.037 | +13.2% (0.042) |
| 3–7 days | 0.095 | +1.5% (0.097) | <span style="color: green">+22.5% (0.117)</span> | 0.033 | <span style="color: green">+24.6% (0.041)</span> |

| user activity | next-item recall@100 | 24-hour recall@100 | next-10 recall@100 | next-item ndcg@100 | next-10 ndcg@100 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Q1, least active | 0.117 | +1.2% (0.118) | <span style="color: green">+20.0% (0.140)</span> | 0.039 | +15.6% (0.046) |
| Q2 | 0.116 | −2.5% (0.113) | <span style="color: green">+23.1% (0.143)</span> | 0.040 | +17.5% (0.047) |
| Q3 | 0.096 | −3.1% (0.093) | +18.6% (0.114) | 0.036 | <span style="color: green">+27.0% (0.045)</span> |
| Q4, most active | 0.088 | +1.5% (0.090) | <span style="color: green">+26.2% (0.112)</span> | 0.037 | <span style="color: green">+34.3% (0.050)</span> |

Recall@100 improves by 0.023 over next-item, exceeding the 0.020259 operational band, while NDCG@100 improves by 23.3%. It also beats the selected 24-hour model by 22.8% Recall@100. The next-10 objective is supported and is the current fixed-window winner.

## RQ3: Can behavior-similar future periods define better positives?

| selector | test user-balanced ndcg@10 | test auroc |
| :--- | :---: | :---: |
| strongest deterministic | 0.296 | 0.551 |
| **learned** | **+0.4% (0.297)** | **+1.1% (0.557)** |

The deterministic arm is content cosine over 6-hour periods with a 3-day lookahead and minimum one liked event, selected across time, content, and frequency families. The learned classifier uses the same period geometry and combines those signals with causal time, activity, and gap features; both rank the same 5,971,674 test pairs from 120,344 queries and 5,382 users.

The learned-minus-deterministic user-bootstrap mean NDCG@10 difference is 0.001276 with 95% interval [0.000873, 0.001684], so the selector-quality gate passes. Full native-50M five-fold materialization is still waiting for the predeclared quiet-host gate and no downstream recommender run has started. The downstream RQ3 conclusion remains unresolved until materialization, all three 12-trial recommender studies, and any triggered boundary rounds complete.

## Aggregated improvement

The current promotable component is the mutually exclusive next-10 target rule. RQ3 and its materialization gate are incomplete, so aggregate membership is not frozen and no aggregate claim is made yet.
