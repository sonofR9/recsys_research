# G4 native-500M compact results

## RQ1: Does a 24-hour future window help?

| variant | recall@100 | ndcg@100 | coverage@100 | horizon | restored epoch |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **next liked item** | **0.153** | **0.062** | **0.458** | **15** | **10** |
| uniform liked event in the next 24 hours | <span style="color: red">−5.7% (0.145)</span> | <span style="color: red">−4.1% (0.059)</span> | <span style="color: red">−27.3% (0.333)</span> | 15 | 10 |

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

## RQ2: Does a next-10-liked-events window help?

| variant | recall@100 | ndcg@100 | coverage@100 | horizon | restored epoch |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **next liked item** | **0.153** | **0.062** | **0.458** | **15** | **10** |
| uniform among the next 10 liked events | <span style="color: red">−2.9% (0.149)</span> | −0.6% (0.061) | <span style="color: red">−28.0% (0.330)</span> | 15 | 10 |
| uniform liked event in the next 24 hours | <span style="color: red">−5.7% (0.145)</span> | <span style="color: red">−4.1% (0.059)</span> | <span style="color: red">−27.3% (0.333)</span> | 15 | 10 |

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

## RQ3: Can behavior-similar future periods define better positives?

| stage | authenticated static audit | outcome |
| :--- | :--- | :---: |
| pre-selector feasibility | The approved classifier requires a population-sized in-memory fit and has no external-memory fit path. | **stopped** |
