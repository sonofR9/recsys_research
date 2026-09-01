## RQ12 — Which decoder-only query-token layout works best?

### Candidate-generation quality

| query objective | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard item-state | 0.135 | 0.051 | 0.028 | 0.022 | 0.728 |
| **end-only CLS** | <span style="color: green">+11% (0.149)</span> | <span style="color: green">+19% (0.061)</span> | <span style="color: green">+34% (0.038)</span> | <span style="color: green">+38% (0.031)</span> | <span style="color: red">-39% (0.441)</span> |
| interleaved CLS | -1% (0.133) | <span style="color: red">-2% (0.050)</span> | -6% (0.027) | <span style="color: red">-5% (0.021)</span> | <span style="color: red">-27% (0.532)</span> |

### Training efficiency

| query objective | examples/epoch | next-item targets/epoch | auxiliary NTP targets/epoch | input tokens/epoch | best epochs (seeds 42 / 43 / 44) | mean steady-state targets/s (epochs 2–20 train only) | mean time through selected checkpoint (train+validation), s | mean full-horizon logged train+validation, s | all required artifacts logged train+validation, s | all required artifacts observed wall (Prepared stage → Final metrics), s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard item-state | 110731 | 7674702 | 0 | 7785433 | 16 / 15 / 20 | 443883.299 | 299.221 | 351.851 | 2464.376 | 2530.743 |
| **end-only CLS** | 110731 | 7674702 | 0 | 7896164 | 18 / 17 / 19 | 418402.556 | 334.426 | 371.595 | 2229.807 | 2321.492 |
| interleaved CLS | 110731 | 7674702 | 0 | 15570866 | 20 / 18 / 17 | 397248.880 | 358.135 | 390.720 | 2346.014 | 2786.721 |
| **all query objectives** | — | — | — | — | — | — | — | — | 7040.198 | 7638.956 |
