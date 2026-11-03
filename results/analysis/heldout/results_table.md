MBPP heldout (150 problems), greedy decoding. Percentages; failure columns are shares of all problems. Differences under about 2 points are within run-to-run noise.

| Run | Trained on | pass@1 [95% CI] | vs A | McNemar p (Holm) | pass@8 | Wrong answer | Runtime error | Other |
|---|---|---|---|---|---|---|---|---|
| A | nothing (base model) | 48.0 [40.0, 56.0] |  |  | 62.0 | 44.7 | 7.3 | 0.0 |
| B (round 1) | own passing solutions | 45.3 [37.3, 53.3] | -2.7 [-8.0, +2.7] | 1.000 |  | 45.3 | 9.3 | 0.0 |
| C (round 1) | own solutions, unfiltered | 42.0 [34.0, 50.0] | -6.0 [-11.3, -0.7] | 0.245 |  | 49.3 | 8.7 | 0.0 |
| D | human solutions | 44.7 [36.7, 52.7] | -3.3 [-10.0, +3.3] | 1.000 |  | 48.0 | 5.3 | 2.0 |

Other comparisons (McNemar, Holm-corrected):

- C (round 1) vs B (round 1): -3.3 points [-8.7, +1.3], only B (round 1) solves 10, only C (round 1) solves 5, p = 1.000
- D vs B (round 1): -0.7 points [-7.3, +6.0], only B (round 1) solves 14, only D solves 13, p = 1.000
- Share of D's gain recovered by B (round 1): undefined, D does not beat A here
- Share of D's gain recovered by C (round 1): undefined, D does not beat A here
