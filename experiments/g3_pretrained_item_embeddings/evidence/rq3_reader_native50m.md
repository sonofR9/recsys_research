# G3 RQ3: Which prediction embedding is best?

| Report protocol | Value |
| :--- | :--- |
| Dataset | native Yambda-50M likes |
| Run policy | one selected run per family; batch 512; seed 42 |
| Band provenance | canonical unchanged native-50M control relative dispersions, scaled to the learned-output reference |
| Operational thresholds | Recall@100 ±0.018, NDCG@100 ±0.007, MRR@100 ±0.007, Coverage@100 ±0.701 |

| RQ3 decision | Result |
| :--- | :--- |
| Scientific / RQ4 winner | Learned ID + frozen content (`rq3_output_learned_frozen_content:04`), Recall@100 0.100520 |
| Gain over learned output | +0.006329 (+6.72%) |
| Recall@100 operational band | ±0.018286 |
| Aggregate promotion | no; gain is inside the operational band |
| Aggregate selection | Learned item-ID (`rq3_output_learned:08`) |

| Prediction embedding | Recall@100 | NDCG@100 | MRR@100 | Coverage@100 |
| :--- | :---: | :---: | :---: | :---: |
| Learned item-ID | 0.094 | 0.034 | 0.029 | 0.823 |
| Frozen pretrained content | <span style="color: red">-32.7% (0.063)</span> | <span style="color: red">-34.9% (0.022)</span> | <span style="color: red">-33.2% (0.020)</span> | +9.0% (0.898) |
| Trainable pretrained content | +1.6% (0.096) | +1.0% (0.034) | +2.2% (0.030) | +5.1% (0.865) |
| **Learned ID + frozen content** | +6.7% (0.101) | +8.7% (0.037) | +15.7% (0.034) | -10.3% (0.738) |
| Learned ID + trainable content | +2.2% (0.096) | +5.7% (0.036) | +6.2% (0.031) | -14.0% (0.708) |

| Prediction embedding (descriptive frequency slices) | Head Recall@100 | Mid Recall@100 | Tail Recall@100 |
| :--- | :---: | :---: | :---: |
| Learned item-ID | 0.129 | 0.035 | 0.016 |
| Frozen pretrained content | -56.8% (0.056) | +83.7% (0.063) | +428.4% (0.083) |
| Trainable pretrained content | +0.2% (0.129) | +28.2% (0.044) | -24.2% (0.012) |
| Learned ID + frozen content | +10.1% (0.142) | +9.9% (0.038) | -20.4% (0.013) |
| Learned ID + trainable content | +7.7% (0.139) | -21.1% (0.027) | -66.5% (0.005) |

| Matched-coordinate target contrast | Pairs | Treatment wins | Mean Recall@100 point delta | Median point delta |
| :--- | :---: | :---: | :---: | :---: |
| Learned item-ID → Learned ID + frozen content | 9 | 7 | +0.005 | +0.007 |
| Learned ID + frozen content → Learned ID + trainable content | 9 | 1 | -0.005 | -0.005 |
| Frozen pretrained content → Trainable pretrained content | 9 | 9 | +0.025 | +0.031 |
| Frozen pretrained content → Learned ID + frozen content | 9 | 9 | +0.031 | +0.035 |
| Learned item-ID → Trainable pretrained content | 9 | 6 | -0.001 | +0.001 |
| Learned item-ID → Frozen pretrained content | 9 | 0 | -0.025 | -0.028 |

| Prediction embedding (selected run) | Restored / horizon | Logged training seconds | Examples/s | Parameters | Peak GPU GB | Full-catalog observed upper bound, s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Learned item-ID | 20 / 40 | 84.8 | 286086 | 4377452 | 4.985 | 0.278 |
| Frozen pretrained content | 24 / 40 | 70.1 | 345987 | 2264108 | 4.996 | 0.330 |
| Trainable pretrained content | 34 / 40 | 87.9 | 276004 | 6507180 | 5.029 | 0.233 |
| Learned ID + frozen content | 18 / 40 | 84.7 | 286296 | 4389740 | 6.514 | 0.337 |
| Learned ID + trainable content | 18 / 40 | 79.2 | 306034 | 8632812 | 6.561 | 0.263 |

| Boundary family | Initial selected deep LR | Added lower probes | Final selected deep LR | Decision |
| :--- | :---: | :---: | :---: | :---: |
| Learned ID + frozen content | 0.005733565 | 0.002027121, 0.001433391, 0.001013561 | 0.005733565 | resolved; no lower probe won |
| Learned ID + trainable content | 0.005733565 | 0.002027121, 0.001433391, 0.001013561 | 0.005733565 | resolved; no lower probe won |

| Efficiency limitation | Status |
| :--- | :--- |
| Exact full-catalog encoding/scoring time | unavailable; the saved upper bound also includes callback, checkpoint restore, and evidence persistence |
| Slice significance | descriptive only; no slice-specific repeat calibration exists |
