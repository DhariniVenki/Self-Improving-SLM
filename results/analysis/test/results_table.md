MBPP test (500 problems), greedy decoding. Percentages; failure columns are shares of all problems. Differences under about 2 points are within run-to-run noise.

| Run | Trained on | pass@1 [95% CI] | vs A | McNemar p (Holm) | pass@8 | Wrong answer | Runtime error | Other |
|---|---|---|---|---|---|---|---|---|
| A | nothing (base model) | 33.6 [29.4, 37.8] |  |  | 55.0 | 54.6 | 11.2 | 0.6 |
| B (round 1) | own passing solutions | 35.2 [31.0, 39.4] | +1.6 [-1.4, +4.8] | 0.748 |  | 53.0 | 10.8 | 1.0 |
| C (round 1) | own solutions, unfiltered | 32.4 [28.4, 36.4] | -1.2 [-4.2, +1.6] | 0.748 |  | 56.8 | 10.4 | 0.4 |
| D | human solutions | 37.2 [33.0, 41.4] | +3.6 [-0.6, +7.8] | 0.342 |  | 51.2 | 10.2 | 1.4 |
