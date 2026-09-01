# G3 pretrained item embeddings — native Yambda-50M tables

## RQ1: How does replacing the history item ID with pretrained content affect retrieval?

| Variant (percentage reference: Tied original learned item ID) | Recall@100 | NDCG@100 |
| :--- | :---: | :---: |
| **Tied original learned item ID** | 0.104 | 0.038 |
| Frozen content history | -7.1% (0.097) | -13.6% (0.033) |

| Variant (descriptive slices; percentage reference: Tied original learned item ID) | Head Recall@100 | Mid Recall@100 | Tail Recall@100 |
| :--- | :---: | :---: | :---: |
| **Tied original learned item ID** | 0.144 | 0.037 | 0.020 |
| Frozen content history | -8.0% (0.132) | +6.9% (0.040) | -46.4% (0.011) |
| Evidence status | descriptive only; no slice-specific repeat calibration exists | — | — |

## RQ2: Does concatenating pretrained content with the item-ID embedding help?

| Variant (percentage reference: Tied original learned item ID) | Recall@100 | NDCG@100 |
| :--- | :---: | :---: |
| **Tied original learned item ID** | 0.104 | 0.038 |
| ID + frozen content DenseNet | -10.5% (0.093) | -13.4% (0.033) |

| Variant (descriptive slices; percentage reference: Tied original learned item ID) | Head Recall@100 | Mid Recall@100 | Tail Recall@100 |
| :--- | :---: | :---: | :---: |
| **Tied original learned item ID** | 0.144 | 0.037 | 0.020 |
| ID + frozen content DenseNet | -9.8% (0.130) | -5.8% (0.035) | -38.1% (0.012) |
| Evidence status | descriptive only; no slice-specific repeat calibration exists | — | — |

## RQ3: With concatenated item-ID and pretrained inputs, which catalog target is best?

| Variant (percentage reference: Tied original learned item ID) | Recall@100 | NDCG@100 |
| :--- | :---: | :---: |
| **Tied original learned item ID** | 0.104 | 0.038 |
| Learned item-ID | -9.7% (0.094) | -10.8% (0.034) |
| Frozen pretrained content | <span style="color: red">-39.3% (0.063)</span> | <span style="color: red">-41.9% (0.022)</span> |
| Trainable pretrained content | -8.3% (0.096) | -9.9% (0.034) |
| Learned ID + frozen content | -3.7% (0.101) | -3.1% (0.037) |
| Learned ID + trainable content | -7.8% (0.096) | -5.8% (0.036) |

| Variant (descriptive slices; percentage reference: Tied original learned item ID) | Head Recall@100 | Mid Recall@100 | Tail Recall@100 |
| :--- | :---: | :---: | :---: |
| **Tied original learned item ID** | 0.144 | 0.037 | 0.020 |
| Learned item-ID | -10.2% (0.129) | -7.5% (0.035) | -21.7% (0.016) |
| Frozen pretrained content | -61.2% (0.056) | +69.9% (0.063) | +313.9% (0.083) |
| Trainable pretrained content | -10.0% (0.129) | +18.6% (0.044) | -40.7% (0.012) |
| Learned ID + frozen content | -1.1% (0.142) | +1.6% (0.038) | -37.6% (0.013) |
| Learned ID + trainable content | -3.3% (0.139) | -27.0% (0.027) | -73.8% (0.005) |
| Evidence status | descriptive only; no slice-specific repeat calibration exists | — | — |

## RQ4: Does adding artist and album features improve the metrics?

| Status | Reason |
| :--- | :--- |
| Pending | Authenticated final evidence is not available; RQ4 is unresolved; partial capacity evidence is excluded until an authenticated final selection passes review. |

## RQ5: Does conditioning the item-ID/content mixture on frequency improve tail retrieval?

| Variant (percentage reference: Tied original learned item ID) | Recall@100 | NDCG@100 |
| :--- | :---: | :---: |
| **Tied original learned item ID** | 0.104 | 0.038 |
| Fixed concatenation | -10.5% (0.093) | -13.4% (0.033) |
| Learned global scalar gate | -10.5% (0.093) | -9.4% (0.035) |
| Frequency-conditioned gate | -11.6% (0.092) | -7.9% (0.035) |

| Variant (descriptive slices; percentage reference: Tied original learned item ID) | Head Recall@100 | Mid Recall@100 | Tail Recall@100 |
| :--- | :---: | :---: | :---: |
| **Tied original learned item ID** | 0.144 | 0.037 | 0.020 |
| Fixed concatenation | -9.8% (0.130) | -5.8% (0.035) | -38.1% (0.012) |
| Learned global scalar gate | -8.9% (0.131) | -6.1% (0.035) | -35.0% (0.013) |
| Frequency-conditioned gate | -8.0% (0.132) | -18.8% (0.030) | -74.9% (0.005) |
| Evidence status | descriptive only; no slice-specific repeat calibration exists | — | — |

## RQ6: Does dataset size change the selected treatment's improvement?

| Status | Reason |
| :--- | :--- |
| Pending | Authenticated final evidence is not available; the native-50M/native-500M size comparison is unresolved and has no authenticated final evidence. |

## Aggregated improvement

| Status | Reason |
| :--- | :--- |
| Pending | Authenticated final evidence is not available; the aggregate is unresolved because RQ4 and the size companion are not final; no aggregate metrics are reported. |
