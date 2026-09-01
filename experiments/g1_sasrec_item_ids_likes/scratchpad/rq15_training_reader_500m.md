## Candidate-generation quality

| training method | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint scratch, candidate-only | 0.079 | 0.035 | 0.024 | 0.021 | 0.002 |
| NTP pretraining, then candidate-only fine-tuning | <span style="color: green">+100% (0.159)</span> | <span style="color: green">+85% (0.065)</span> | <span style="color: green">+67% (0.041)</span> | <span style="color: green">+60% (0.033)</span> | <span style="color: green">+21243% (0.393)</span> |
| joint scratch, candidate + auxiliary NTP | <span style="color: green">+23% (0.098)</span> | <span style="color: green">+21% (0.043)</span> | <span style="color: green">+18% (0.029)</span> | <span style="color: green">+19% (0.025)</span> | +592% (0.013) |

## Training efficiency

| training method | examples/epoch | input tokens/epoch | candidate targets/epoch | NTP targets/epoch | candidate targets/s | total targets/s | best epoch | processed examples | processed candidate targets | processed NTP targets | fine-tune time | pretraining horizon | cold-start time | total tuning wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint scratch, candidate-only | 75,434 | 4,832,707 | 75,434 | 0 | 24,980 | 24,980 | 4 | 301,736 | 301,736 | 0 | 0:00:12 | — | 0:00:12 | 0:20:18 |
| NTP pretraining, then candidate-only fine-tuning | 110,731 pre + 75,434 fine | 7,785,433 pre + 4,832,707 fine | 75,434 | 0 | 25,352 | 25,352 | 18 | 3,572,432 | 1,357,812 | 153,494,040 | 0:00:56 | 0:05:58 | 0:06:53 | 1:09:31 |
| joint scratch, candidate + auxiliary NTP | 75,434 | 4,832,707 | 75,434 | 4,455,537 | 6,206 | 372,753 | 4 | 301,736 | 301,736 | 17,822,148 | 0:00:50 | — | 0:00:50 | 1:11:23 |
