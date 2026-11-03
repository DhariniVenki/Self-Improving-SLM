MBPP heldout (150 problems), greedy decoding. Percentages; failure columns are shares of all problems. Differences under about 2 points are within run-to-run noise.

| Run | Trained on | pass@1 [95% CI] | vs A | McNemar p (Holm) | pass@8 | Wrong answer | Runtime error | Other |
|---|---|---|---|---|---|---|---|---|
| A | nothing (base model) | 48.0 [40.0, 56.0] |  |  | 62.0 | 44.7 | 7.3 | 0.0 |
| B (round 1) | own passing solutions | 45.3 [37.3, 53.3] | -2.7 [-8.0, +2.7] | 0.905 |  | 45.3 | 9.3 | 0.0 |
| C (round 1) | own solutions, unfiltered | 42.0 [34.0, 50.0] | -6.0 [-11.3, -0.7] | 0.196 |  | 49.3 | 8.7 | 0.0 |
| D | human solutions | 44.7 [36.7, 52.7] | -3.3 [-10.0, +3.3] | 0.905 |  | 48.0 | 5.3 | 2.0 |
