## RQ5 — Which learning-rate scheduler works best?

| scheduler | optimizer groups scheduled | schedule parameter | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | both | — | 0.141 | 0.054 | 0.031 | 0.023 | 0.718 |
| linear | both | — | <span style="color: green">+3% (0.145)</span> | <span style="color: green">+3% (0.056)</span> | +1% (0.031) | <span style="color: green">+5% (0.024)</span> | -12% (0.635) |
| linear | deep only | — | +1% (0.143) | +1% (0.054) | +2% (0.031) | +4% (0.024) | -6% (0.672) |
| cosine | both | — | +1% (0.144) | +1% (0.054) | 0% (0.031) | +2% (0.024) | -8% (0.660) |
| cosine | deep only | — | +1% (0.143) | +1% (0.055) | 0% (0.031) | +3% (0.024) | -7% (0.668) |
| polynomial | both | — | -1% (0.140) | -1% (0.053) | -4% (0.030) | -1% (0.023) | -10% (0.646) |
| polynomial | deep only | — | <span style="color: red">-3% (0.137)</span> | <span style="color: red">-4% (0.052)</span> | -7% (0.028) | <span style="color: red">-6% (0.022)</span> | -1% (0.707) |
| exponential | both | — | -1% (0.140) | -1% (0.053) | -3% (0.030) | 0% (0.023) | -5% (0.681) |
| exponential | deep only | — | -2% (0.138) | <span style="color: red">-2% (0.053)</span> | -3% (0.030) | -1% (0.023) | -7% (0.669) |
| step | both | — | +1% (0.143) | +2% (0.055) | +1% (0.031) | +3% (0.024) | -3% (0.697) |
| step | deep only | — | +1% (0.142) | +1% (0.054) | 0% (0.031) | +2% (0.024) | -6% (0.675) |
| WSD | both | warmup=0.05, cycles=1 | +2% (0.144) | <span style="color: green">+2% (0.055)</span> | -1% (0.030) | +2% (0.024) | -7% (0.666) |
| WSD | deep only | warmup=0.05, cycles=1 | +2% (0.144) | <span style="color: green">+2% (0.055)</span> | +1% (0.031) | +4% (0.024) | -8% (0.657) |
| inverse sqrt | both | timescale=0.05 | -1% (0.139) | <span style="color: red">-2% (0.053)</span> | -4% (0.029) | -3% (0.022) | +4% (0.749) |
| inverse sqrt | deep only | timescale=0.05 | <span style="color: red">-2% (0.138)</span> | <span style="color: red">-3% (0.052)</span> | -3% (0.030) | -4% (0.022) | +2% (0.730) |
| cosine, warmup 5%, 1 cycle | both | warmup=0.05, cycles=1 | +1% (0.143) | +1% (0.054) | -4% (0.030) | +2% (0.023) | -13% (0.623) |
| **cosine, warmup 5%, 1 cycle** | **deep only** | **warmup=0.05, cycles=1** | **<span style="color: green">+3% (0.146)</span>** | **<span style="color: green">+3% (0.055)</span>** | **+4% (0.032)** | **<span style="color: green">+5% (0.024)</span>** | **-10% (0.647)** |
| cosine, warmup 5%, 2 cycles | both | warmup=0.05, cycles=2 | -2% (0.139) | <span style="color: red">-2% (0.052)</span> | -7% (0.028) | -4% (0.022) | -8% (0.658) |
| cosine, warmup 5%, 2 cycles | deep only | warmup=0.05, cycles=2 | 0% (0.141) | 0% (0.054) | -1% (0.030) | +1% (0.023) | -4% (0.686) |
| cosine, warmup 5%, 4 cycles | both | warmup=0.05, cycles=4 | -2% (0.139) | <span style="color: red">-2% (0.053)</span> | -7% (0.029) | -2% (0.023) | -5% (0.680) |
| cosine, warmup 5%, 4 cycles | deep only | warmup=0.05, cycles=4 | -1% (0.139) | <span style="color: red">-2% (0.053)</span> | -3% (0.030) | -2% (0.023) | 0% (0.720) |
| cosine, tuned warmup | both | warmup=0.05, cycles=1 | <span style="color: red">-2% (0.138)</span> | <span style="color: red">-3% (0.052)</span> | -6% (0.029) | -3% (0.022) | -6% (0.673) |
| cosine, tuned warmup | deep only | warmup=0.0209409463814, cycles=1 | +1% (0.143) | 0% (0.054) | -5% (0.029) | -1% (0.023) | <span style="color: red">-21% (0.569)</span> |
