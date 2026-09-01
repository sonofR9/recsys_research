# RQ15 diagnostics

## Acceptance diagnostics

Acceptance criterion: Adding a pretraining stage should at minimum decrease training time without losing quality, and will most probably improve the main metrics.

| check | result |
| --- | --- |
| quality non inferior | yes |
| cold start faster | no |
| main metrics improved | yes |

## Supervision and learning curves

| stage | candidate targets/epoch | NTP targets/epoch | best/stopped epoch |
| --- | ---: | ---: | ---: |
| scratch candidate only | 75,434 | 0 | 4/20 |
| checkpoint pretraining | 0 | 7,674,702 | 16/20 |
| scratch candidate only | 75,434 | 0 | 4/20 |
| pretrained finetune | 75,434 | 0 | 18/20 |
| auxiliary ntp | 75,434 | 4,455,537 | 4/20 |
