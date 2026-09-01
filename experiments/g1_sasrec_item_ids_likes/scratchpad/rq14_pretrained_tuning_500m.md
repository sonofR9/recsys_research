# RQ14 NTP-pretrained tuning

## shared_cls_only
| source | embedding LR | deep LR | validation recall@100 | validation ndcg@100 | best epoch |
| :--- | ---: | ---: | ---: | ---: | ---: |
| new RQ14 | 0.00025 | 0.000375 | 0.155600 | 0.064400 | 13 |
| **new RQ14** | **0.00025** | **0.00075** | **0.157700** | **0.064400** | **15** |
| new RQ14 | 0.00025 | 0.0015 | 0.156500 | 0.064100 | 16 |

## distinct_cls_only
| source | embedding LR | deep LR | validation recall@100 | validation ndcg@100 | best epoch |
| :--- | ---: | ---: | ---: | ---: | ---: |
| reused RQ15 | 0.00025 | 0.000375 | 0.156600 | 0.064200 | 20 |
| **reused RQ15** | **0.00025** | **0.00075** | **0.158600** | **0.064600** | **18** |
| reused RQ15 | 0.00025 | 0.0015 | 0.157700 | 0.064600 | 11 |

## shared_history
| source | embedding LR | deep LR | validation recall@100 | validation ndcg@100 | best epoch |
| :--- | ---: | ---: | ---: | ---: | ---: |
| new RQ14 | 0.00025 | 0.000375 | 0.156300 | 0.064500 | 15 |
| **new RQ14** | **0.00025** | **0.00075** | **0.158500** | **0.065400** | **20** |
| new RQ14 | 0.00025 | 0.0015 | 0.157600 | 0.064900 | 15 |

## distinct_history
| source | embedding LR | deep LR | validation recall@100 | validation ndcg@100 | best epoch |
| :--- | ---: | ---: | ---: | ---: | ---: |
| new RQ14 | 0.00025 | 0.000375 | 0.157200 | 0.064600 | 19 |
| **new RQ14** | **0.00025** | **0.00075** | **0.159700** | **0.065600** | **17** |
| new RQ14 | 0.00025 | 0.0015 | 0.158700 | 0.065100 | 11 |
