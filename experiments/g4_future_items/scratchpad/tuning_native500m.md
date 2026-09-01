# G4 native-500M hyperparameter tuning

Batch size 512, embedding LR 0.0468526465053628, and the 15-epoch one-cycle cosine horizon are fixed. Deep LR is the only tuned field.

## Control: next liked item

| trial | deep lr | horizon | best epoch | validation recall@100 | validation loss |
| :--- | ---: | ---: | ---: | ---: | ---: |
| base-01 | 0.016351873 | 15 | 10 | 0.1507 | 5.8443 |
| base-02 | 0.032703746 | 15 | 10 | 0.1533 | 5.8531 |
| **base-03** | **0.065407491** | **15** | **10** | **0.1536** | **5.8286** |
| upper-r1-01 | 0.13081498 | 15 | 11 | 0.0325 | 7.4714 |
| upper-r1-02 | 0.26162997 | 15 | 13 | 0.0318 | 7.4546 |

## RQ1: 24-hour future window

| trial | deep lr | horizon | best epoch | validation recall@100 | validation loss |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **base-01** | **0.016351873** | **15** | **10** | **0.1460** | **6.1264** |
| base-02 | 0.032703746 | 15 | 14 | 0.1443 | 6.2786 |
| base-03 | 0.065407491 | 15 | 15 | 0.1434 | 6.3174 |
| lower-r1-01 | 0.0040879682 | 15 | 12 | 0.1405 | 6.1989 |
| lower-r1-02 | 0.0081759364 | 15 | 10 | 0.1457 | 6.0730 |

## RQ2: next 10 liked events

| trial | deep lr | horizon | best epoch | validation recall@100 | validation loss |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **base-01** | **0.016351873** | **15** | **10** | **0.1502** | **5.9901** |
| base-02 | 0.032703746 | 15 | 11 | 0.1497 | 6.0373 |
| base-03 | 0.065407491 | 15 | 14 | 0.1433 | 6.1594 |
| lower-r1-01 | 0.0040879682 | 15 | 15 | 0.1481 | 6.1581 |
| lower-r1-02 | 0.0081759364 | 15 | 10 | 0.1502 | 5.9903 |
