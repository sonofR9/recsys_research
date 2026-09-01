## RQ13 diagnostic — truncated-prefix selected epochs

| observation | no expansion | truncated 8 | truncated 16 |
| --- | ---: | ---: | ---: |
| expanded examples/epoch | 75434 | 538703 | 996053 |
| selected epoch | 3 | 5 | 7 |
| selected validation recall@100 | 0.0766 | 0.1164 | 0.1233 |
| first epoch at preceding row's selected quality | 3 | 1 | 4 |
| optimizer steps to that point | 177 | 421 | 3116 |
| targets seen to that point | 226302 | 538703 | 3984212 |

| deep LR | truncated-8 best epoch / recall@100 | truncated-16 best epoch / recall@100 |
| ---: | ---: | ---: |
| 0.006 | 6 / 0.1132 | 4 / 0.1219 |
| 0.012 | 5 / 0.1164 | 7 / 0.1233 |
| 0.024 | 6 / 0.1142 | 7 / 0.1128 |

the selected checkpoints are later at caps 8 and 16, but each expanded treatment reached the preceding treatment's selected quality in fewer epochs and then continued to a higher peak; the matched-quality epoch crossings do not imply lower work because larger caps used more optimizer steps and targets

## RQ13 diagnostic — cap-response sensitivity

all 16 independent ±0.003 corners; sensitivity, not a confidence interval

| perturbations at caps 1/4/8/16 | A | B | p | RMSE | fitted recall@100 at cap 32 | fitted target cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| -0.003/-0.003/-0.003/-0.003 | 0.208970 | 0.135598 | 0.156943 | 0.001463 | 0.130259 | 188 |
| -0.003/-0.003/-0.003/+0.003 | 0.480712 | 0.407913 | 0.050000 | 0.001064 | 0.137700 | 64 |
| -0.003/-0.003/+0.003/-0.003 | 0.171380 | 0.098300 | 0.254217 | 0.003803 | 0.130650 | 359 |
| -0.003/-0.003/+0.003/+0.003 | 0.483476 | 0.410388 | 0.051800 | 0.002869 | 0.140528 | 53 |
| -0.003/+0.003/-0.003/-0.003 | 0.137815 | 0.064207 | 0.467387 | 0.000076 | 0.125106 | infinity |
| -0.003/+0.003/-0.003/+0.003 | 0.198911 | 0.125149 | 0.192877 | 0.001099 | 0.134773 | 122 |
| -0.003/+0.003/+0.003/-0.003 | 0.135935 | 0.062545 | 0.542741 | 0.002315 | 0.126401 | infinity |
| -0.003/+0.003/+0.003/+0.003 | 0.170908 | 0.097471 | 0.288741 | 0.001249 | 0.135076 | 186 |
| +0.003/-0.003/-0.003/-0.003 | 0.402625 | 0.323604 | 0.050000 | 0.001754 | 0.130507 | 135 |
| +0.003/-0.003/-0.003/+0.003 | 0.437701 | 0.359855 | 0.050000 | 0.002228 | 0.135100 | 84 |
| +0.003/-0.003/+0.003/-0.003 | 0.230754 | 0.151810 | 0.122857 | 0.004027 | 0.131583 | 160 |
| +0.003/-0.003/+0.003/+0.003 | 0.454258 | 0.376112 | 0.050000 | 0.003341 | 0.137987 | 67 |
| +0.003/+0.003/-0.003/-0.003 | 0.149211 | 0.069623 | 0.317761 | 0.000093 | 0.126065 | infinity |
| +0.003/+0.003/-0.003/+0.003 | 0.435084 | 0.355410 | 0.050000 | 0.000858 | 0.136221 | 79 |
| +0.003/+0.003/+0.003/-0.003 | 0.141580 | 0.062248 | 0.418211 | 0.002464 | 0.126970 | infinity |
| +0.003/+0.003/+0.003/+0.003 | 0.214938 | 0.135567 | 0.156989 | 0.001463 | 0.136258 | 102 |
