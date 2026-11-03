## Round 0: 8 samples per problem from the base model (temperature 0.8)

| | train | test |
|---|---|---|
| Problems | 314 | 500 |
| pass@1 (from the 8 samples) | 32.2 | 31.1 |
| pass@8 | 57.0 | 55.0 |
| Never solved (task 12) | 135 (43.0%) | 225 (45.0%) |
| Solved by all 8 samples | 34 (10.8%) | 39 (7.8%) |
| Samples passing the shown test | 987 | 1539 |
| ...of which fail a hidden test (task 13) | 179 (18.1%) | 294 (19.1%) |
| Samples: pass | 32.2 | 31.1 |
| Samples: no code | 0.0 | 0.0 |
| Samples: syntax | 0.5 | 0.8 |
| Samples: runtime | 13.5 | 13.0 |
| Samples: wrong answer | 53.5 | 54.8 |
| Samples: timeout | 0.4 | 0.4 |

## Training sets (tasks 14, 20, 21)

| Run | Examples | Problems covered | Distinct before cap | Share not passing |
|---|---|---|---|---|
| B_round1 | 521 | 179/314 | 660 | 0.0% |
| C_round1 | 1247 | 314/314 | 2291 | 69.0% |
| D | 314 | 314/314 |  | 0.0% |

## Training cost (task 19)

| Run | Examples | LR | Epochs | Seconds | Peak GPU GB | GPU | Loss first → final |
|---|---|---|---|---|---|---|---|
| D | 314 | 0.0005 | 2 | 49.1 | 8.61 | Tesla T4 | 0.742 → 0.392 |
