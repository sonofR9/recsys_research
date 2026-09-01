# RQ14 paired-effect diagnostics

| effect | validation recall delta | full recall delta | exceeds +0.003 |
| --- | ---: | ---: | --- |
| distinct minus shared cls only | +0.00030000 | +0.00016960 | no |
| distinct minus shared history | +0.00020000 | +0.00042419 | no |
| history minus cls only shared | +0.00000000 | +0.00024946 | no |
| history minus cls only distinct | -0.00010000 | +0.00050404 | no |

## Bound explanation

The correctness and unexpected-result explanation gates passed for the current artifacts.

Four distinct query-token identities do not produce a resolved Recall@100 gain over one repeated token on either memory surface.

Exposing history plus CLS states does not produce a resolved Recall@100 gain for either shared or distinct query tokens.

The unresolved token and memory effects are consistent with a one-target-per-user supervision bottleneck, not evidence that the added states are inaccessible. RQ13 supplies controlled supporting evidence; it does not prove this mechanism causally for RQ14.
