# RQ14 decoder-decoder query memory

## Historical candidate-only comparison

## Candidate-generation quality

| query tokens | cross-attention memory | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **shared CLS** | **four CLS states** | **0.078** | **0.034** | **0.022** | **0.019** | **0.002** |
| distinct CLS_0..3 | four CLS states | 0% (0.079) | <span style="color: green">+3% (0.035)</span> | +11% (0.025) | <span style="color: green">+10% (0.021)</span> | 0% (0.002) |
| shared CLS | history + four CLS states | 0% (0.079) | -2% (0.034) | +9% (0.024) | +1% (0.020) | +36% (0.002) |
| distinct CLS_0..3 | history + four CLS states | +1% (0.079) | +3% (0.035) | +1% (0.023) | +3% (0.020) | +43% (0.002) |

## Training efficiency

| query tokens | cross-attention memory | examples/epoch | candidate targets/epoch | NTP targets/epoch | input tokens/epoch | targets/s | best epoch | processed examples | processed candidate targets | time to checkpoint | total tuning wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **shared CLS** | **four CLS states** | **75,434** | **75,434** | **0** | **4,832,707** | **28,618** | **6** | **452,604** | **452,604** | **0:00:16** | **0:08:52** |
| distinct CLS_0..3 | four CLS states | 75,434 | 75,434 | 0 | 4,832,707 | 28,660 | 4 | 301,736 | 301,736 | 0:00:12 | 0:06:37 |
| shared CLS | history + four CLS states | 75,434 | 75,434 | 0 | 4,832,707 | 27,154 | 3 | 226,302 | 226,302 | 0:00:09 | 0:03:24 |
| distinct CLS_0..3 | history + four CLS states | 75,434 | 75,434 | 0 | 4,832,707 | 27,263 | 4 | 301,736 | 301,736 | 0:00:12 | 0:03:22 |

## NTP-pretrained quality (current decision)

| query slots | second-decoder memory | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **shared CLS** | **four CLS states** | **0.158** | **0.065** | **0.041** | **0.033** | **0.373** |
| distinct CLS_0..3 | four CLS states | 0% (0.159) | 0% (0.065) | -1% (0.041) | -1% (0.033) | +5% (0.393) |
| shared CLS | history + four CLS states | 0% (0.158) | +2% (0.066) | +1% (0.041) | +2% (0.034) | +2% (0.379) |
| distinct CLS_0..3 | history + four CLS states | +1% (0.159) | <span style="color: green">+2% (0.066)</span> | +2% (0.042) | <span style="color: green">+3% (0.034)</span> | -1% (0.370) |

## NTP-pretrained training efficiency

| query slots | second-decoder memory | examples/epoch | candidate targets/epoch | NTP targets/epoch | input tokens/epoch | targets/s | best epoch | processed examples | processed candidate targets | time to checkpoint | 3-cell tuning GPU time |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **shared CLS** | **four CLS states** | **75,434** | **75,434** | **0** | **4,832,707** | **18,200** | **15** | **1,131,510** | **1,131,510** | **0:01:07** | **0:04:12** |
| distinct CLS_0..3 | four CLS states | 75,434 | 75,434 | 0 | 4,832,707 | 25,352 | 18 | 1,357,812 | 1,357,812 | 0:00:56 | 0:03:41 |
| shared CLS | history + four CLS states | 75,434 | 75,434 | 0 | 4,832,707 | 17,505 | 20 | 1,508,680 | 1,508,680 | 0:01:28 | 0:04:17 |
| distinct CLS_0..3 | history + four CLS states | 75,434 | 75,434 | 0 | 4,832,707 | 19,713 | 17 | 1,282,378 | 1,282,378 | 0:01:09 | 0:04:02 |
