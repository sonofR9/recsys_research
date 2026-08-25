# RQ4 — FFN family and capacity

RQ4 tunes GELU and SwiGLU as separate model families. Each tested width receives
its own native Yambda-50M learning-rate search; cap hits and learning-rate
boundaries are continued until the selected point is early-stopping-resolved
and surrounded. GELU widths 128, 171, 256, and 384 were tested. SwiGLU widths
16, 32, 64, 96, 128, 171, and 224 were tested after extending the lower-width
boundary. The selected widths, GELU-171 and SwiGLU-32, are interior to their
finite family searches.

All proxy and final runs use physical and effective batch 1280 without gradient
accumulation, model and item-embedding dimension 64, validation every epoch, and
restored best weights. The schedule is linear over a 20-epoch horizon, so the
final runs train that horizon and report the best epoch inside it rather than
stopping on patience. One native Yambda-500M confirmation is run for each family
at its independently selected width and learning rates, plus one width probe
outside the selection.

## Selected Yambda-50M configurations

| FFN family | intermediate width | embedding LR | deep LR | recall@100 | ndcg@100 | best/stopped epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GELU | 171 | 0.128 | 0.012 | 0.07720477 | 0.02868057 | 19/22 |
| **SwiGLU** | **32** | **0.064** | **0.024** | **0.07890102** | **0.02858721** | **20/23** |

These are family-specific optima, not a matched-width or matched-learning-rate
comparison. The proxy selects the recipe transferred to 500M; it does not by
itself establish that SwiGLU is better than GELU.

## Native Yambda-500M confirmations

| FFN family | intermediate width | transferred embedding/deep LR | recall@10 | recall@100 | ndcg@100 | coverage@100 | best/stopped epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GELU | 171 | 0.128/0.012 | 0.02631191 | 0.13094588 | 0.04905266 | 0.71666338 | 15/20 |
| SwiGLU | 32 | 0.064/0.024 | 0.02646215 | 0.13058894 | 0.04979294 | 0.62897107 | 13/20 |
| SwiGLU width probe | 128 | 0.128/0.012 | 0.02750231 | 0.13221816 | 0.05037345 | 0.67427569 | 13/20 |

The width probe is not a third selection; see the section below.

Both selected artifacts spend the same 20-epoch annealing horizon and report their
best epoch within it, so neither family was cut short of the other. GELU's
recall@100 advantage is 0.00035694, within the shared 500M recall@100 resolution
band of 0.00215019. SwiGLU's ndcg@100 advantage is 0.00074028, within the
corresponding 0.00095122 band, and its recall@10 advantage is 0.00015024, within
the 0.00078925 band. The two independently tuned FFN families are therefore
unresolved on the reported ranking-quality metrics. GELU's coverage@100
advantage is 0.08769232, which exceeds the shared coverage@100 band of
0.07109742; this is a material catalog-coverage difference, not evidence of a
recall or NDCG winner.

The earlier pair of confirmations, taken before the annealing-horizon rule, gave
SwiGLU 12 epochs and GELU 18 of the same schedule and reported a 0.00109899
recall@100 gap. Both artifacts remain on disk; the cap-40 relaunches above
supersede them.

## SwiGLU width probe

SwiGLU 32 won the proxy family search over a surface that had not actually
searched the wider widths. Width 128 had 11 learning-rate points, none above
embedding rate 0.064, and most of them truncated before the horizon; its proxy
recall@100 was 0.06120843, last in the family. Nine added points move its
proxy winner to 0.128/0.012 at 0.07600419, an interior maximum bracketed on all
four sides — 0.064/0.012 at 0.07038, 0.256/0.012 at 0.06941, 0.128/0.006 at
0.06707, and 0.128/0.024 at 0.07406. Widths 32, 64, 96, and 128 then sit within
0.003 of each other on the proxy, so the proxy ranking that selected 32 was an
artifact of the missing points rather than a preference.

The 500M probe above confirms 128 rather than selecting it: the report still
transfers only closed family winners, and this run is registered through
`--exploratory-selection` so it is reported beside RQ4 and excluded from it.
Against the GELU selection it gains 0.00127228 recall@100 (inside the 0.00215019
band), 0.00132079 ndcg@100 (through the 0.00095122 band), and 0.00119041
recall@10 (through the 0.00078925 band), and gives up 0.04238769 coverage@100
(inside the 0.07109742 band). It is the best FFN point measured on 500M, and it
removes the coverage argument that made GELU the practical choice at width 32.
Confirming SwiGLU 64 and 96 on 500M would close the family question; until then
the selected-width comparison stands as reported and this probe stands beside it.

## Implementation and selection path

- GELU and SwiGLU are implemented by
  [RegularMLP](../../../dcn/nn/ffn.py#L46) and
  [SwiGLU](../../../dcn/nn/ffn.py#L9), respectively.
- The architecture source exposes the SwiGLU family in
  [variant.py](../configs/variant.py#L315), while
  [rq_tuning_variant.py](../configs/rq_tuning_variant.py#L116) applies the
  independently tuned intermediate width.
- The tested family widths and their configuration overrides are declared in
  the architecture [manifest](../launchers/architecture/manifest.sh#L191).
- The report selector [requires complete family surfaces and rejects a winner
  on a finite width boundary](../analysis/collect.py#L2935), then
  [transfers only the exact selected width, rates, and
  batch](../analysis/collect.py#L2989).
