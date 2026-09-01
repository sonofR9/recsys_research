# G1 aggregate dataset-size study

Status: the native-50M MuTransfer aggregate-control calibration and nine added
repeats are complete. Batch 512 with embedding/deep LR
`0.0032610024/0.0253436548` won that six-cell comparison. This is not the
conventional baseline SASRec family: its independently selected batch-1280
`0.001/0.002` run reaches Recall@100 `0.1002400` and remains reported in the
original G1 experiment. Do not use the lower MuTransfer result to claim that
the conventional baseline misses `0.1`.

This separate experiment measures the approved G1 aggregate against a
size-matched control. Native Yambda-50M currently uses the MuTransfer control;
native Yambda-500M reuses the original G1 baseline. Dataset size is the explicit
research axis.

See [protocol/plan.md](protocol/plan.md).

The completed fixed-LR batch diagnostic is preserved in raw audit storage but
is excluded here because its learning-rate/batch confound makes it ineligible
for selection. The corrected comparison tuned both learning rates with the
same three-candidate budget at batch 512 and batch 1280.

## Native-50M MuTransfer aggregate-control calibration

| batch | selected embedding LR | selected deep LR | best/stopped | validation Recall@100 | validation NDCG@100 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **512** | **0.0032610024** | **0.0253436548** | **26/29** | **0.0935** | **0.0346** |
| 1280 | 0.0011832644 | 0.0664081144 | 23/26 | 0.0684 | 0.0259 |

All six runs used the same MuTransfer aggregate-control model, evaluated all
3,414 eligible users, stopped by patience three before the 80-epoch cap, and
restored their best validation checkpoint. Batch 512 is therefore frozen for
that model family, together with its selected learning-rate pair.
