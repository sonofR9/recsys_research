# G3 pretrained item embeddings

This experiment tests frozen pretrained item content in the history input and catalog target. Completed RQ1–RQ3 and RQ5 evidence uses native Yambda-50M likes, batch 512, and one validation-selected run per family.

Operational bands use the canonical unchanged native-50M control relative dispersions, scaled once to the tied original learned-ID baseline; they are practical resolution bands, not significance tests. Operational thresholds — original baseline: Recall@100 ±0.020, NDCG@100 ±0.008, MRR@100 ±0.008, Coverage@100 ±0.669.

## RQ1: How does replacing the history item ID with pretrained content affect retrieval?

Tied original learned item ID — shares one learned table between history and catalog.

Frozen content history — replaces only the history lookup with frozen 128-dimensional content followed by a learned projection.

| Variant (percentage reference: Tied original learned item ID) | Recall@100 | NDCG@100 |
| :--- | :---: | :---: |
| **Tied original learned item ID** | 0.104 | 0.038 |
| Frozen content history | -7.1% (0.097) | -13.6% (0.033) |

| Variant (descriptive slices; percentage reference: Tied original learned item ID) | Head Recall@100 | Mid Recall@100 | Tail Recall@100 |
| :--- | :---: | :---: | :---: |
| **Tied original learned item ID** | 0.144 | 0.037 | 0.020 |
| Frozen content history | -8.0% (0.132) | +6.9% (0.040) | -46.4% (0.011) |
| Evidence status | descriptive only; no slice-specific repeat calibration exists | — | — |

Conclusion: frozen-content history changes Recall@100 by -7.1% and remains inside the original-baseline operational band. It does not improve the original baseline, so the original baseline remains selected. Frequency-slice deltas are descriptive only because no slice-specific repeat calibration exists.

## RQ2: Does concatenating pretrained content with the item-ID embedding help?

Tied original learned item ID — shares one learned table between history and catalog.

ID + frozen content DenseNet — concatenates learned item ID with frozen content and returns to model width through DenseNet.

| Variant (percentage reference: Tied original learned item ID) | Recall@100 | NDCG@100 |
| :--- | :---: | :---: |
| **Tied original learned item ID** | 0.104 | 0.038 |
| ID + frozen content DenseNet | -10.5% (0.093) | -13.4% (0.033) |

| Variant (descriptive slices; percentage reference: Tied original learned item ID) | Head Recall@100 | Mid Recall@100 | Tail Recall@100 |
| :--- | :---: | :---: | :---: |
| **Tied original learned item ID** | 0.144 | 0.037 | 0.020 |
| ID + frozen content DenseNet | -9.8% (0.130) | -5.8% (0.035) | -38.1% (0.012) |
| Evidence status | descriptive only; no slice-specific repeat calibration exists | — | — |

Conclusion: ID/content concatenation changes Recall@100 by -10.5% and remains inside the original-baseline operational band. It does not improve the original baseline, so the original baseline remains selected. Frequency-slice deltas are descriptive only.

## RQ3: With concatenated item-ID and pretrained inputs, which catalog target is best?

Tied original learned item ID — uses the original shared learned-ID input and target.

Learned item-ID — learns the catalog table from random initialization.

Frozen pretrained content — projects the fixed content table to catalog width.

Trainable pretrained content — fine-tunes a content-initialized catalog table before projection.

Learned ID + frozen content — projects their concatenation while keeping content fixed.

Learned ID + trainable content — projects their concatenation while fine-tuning the content-initialized copy. All five targets use the selected concatenated RQ2 history input.

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

Conclusion: learned ID plus frozen content is the raw scientific winner at 0.101 Recall@100, -3.7% versus the original baseline. It wins seven of nine matched coordinates against learned output, while trainable content beats frozen content at all nine matched coordinates, satisfying the authenticated internal ordering required by acceptance. No requested target beats the original baseline, so the original baseline remains selected. Frequency slices are descriptive only.

## RQ4: Does adding artist and album features improve the metrics?

Pending — authenticated final evidence is not available.

| Status | Reason |
| :--- | :--- |
| Pending | Authenticated final evidence is not available; RQ4 is unresolved; partial capacity evidence is excluded until an authenticated final selection passes review. |

## RQ5: Does conditioning the item-ID/content mixture on frequency improve tail retrieval?

Tied original learned item ID — shares one learned table between history and catalog.

Fixed concatenation — uses the selected RQ2 input with content gate fixed at one.

Learned global scalar gate — learns one shared multiplier for the frozen content branch.

Frequency-conditioned gate — maps standardized training-only `log1p(item count)` through a width-8 sigmoid MLP. Gate computation and its p=0.9 initialization stay in FP32 under the outer BF16 training context.

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

Conclusion: the corrected frequency gate fails the explicit fixed/global tail-improvement criterion: its observed tail Recall@100 is -59.4% versus fixed and -61.4% versus global. The fixed/global/frequency variants stay below it; none beats the original baseline on Recall@100. The strongest gate is -10.5% below it, so the original baseline remains selected. Slice deltas are descriptive because no slice-specific repeat calibration exists. The nine earlier BF16-saturated p=0.9999 frequency rows are preserved in bound raw audit evidence but excluded from reader and tuning tables because their gate gradients were zero.

## RQ6: Does dataset size change the selected treatment's improvement?

Pending — authenticated final evidence is not available.

| Status | Reason |
| :--- | :--- |
| Pending | Authenticated final evidence is not available; the native-50M/native-500M size comparison is unresolved and has no authenticated final evidence. |

## Aggregated improvement

Pending — authenticated final evidence is not available.

| Status | Reason |
| :--- | :--- |
| Pending | Authenticated final evidence is not available; the aggregate is unresolved because RQ4 and the size companion are not final; no aggregate metrics are reported. |
