# G3 pretrained item embeddings: active protocol

The approved active protocol is
[`native500m_rerun_plan.md`](native500m_rerun_plan.md). It uses native
Yambda-500M only for RQ1–RQ5, starts from the two-layer form of the selected G1
combination, normalizes every content representation, evaluates RQ4
independently from the common baseline, and fixes RQ5's embedding learning rate
to the selected RQ2 value.

The superseded native-50M and dataset-size protocol is preserved in
[`native50m_historical_plan.md`](native50m_historical_plan.md) for audit only.
It cannot select runs, support claims, or appear in the active report.

The executable closed protocol is versioned under [`native500m/`](native500m/).
