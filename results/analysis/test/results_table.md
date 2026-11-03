MBPP test (500 problems), greedy decoding. Percentages; failure columns are shares of all problems. Differences under about 2 points are within run-to-run noise.

| Run | Trained on | pass@1 [95% CI] | vs A | McNemar p (Holm) | pass@8 | Wrong answer | Runtime error | Other |
|---|---|---|---|---|---|---|---|---|
| A | nothing (base model) | 33.6 [29.4, 37.8] |  |  | 55.0 | 54.6 | 11.2 | 0.6 |
| B (round 1) | own passing solutions | 35.2 [31.0, 39.4] | +1.6 [-1.4, +4.8] | 1.000 |  | 53.0 | 10.8 | 1.0 |
| C (round 1) | own solutions, unfiltered | 32.4 [28.4, 36.4] | -1.2 [-4.2, +1.6] | 1.000 |  | 56.8 | 10.4 | 0.4 |
| D | human solutions | 37.2 [33.0, 41.4] | +3.6 [-0.6, +7.8] | 0.456 |  | 51.2 | 10.2 | 1.4 |

Other comparisons (McNemar, Holm-corrected):

- C (round 1) vs B (round 1): -2.8 points [-5.8, +0.2], only B (round 1) solves 35, only C (round 1) solves 21, p = 0.407
- D vs B (round 1): +2.0 points [-2.4, +6.4], only B (round 1) solves 55, only D solves 65, p = 1.000
- Share of D's gain recovered by B (round 1): 44.4% [-70.5, 266.7] (D's own gain over A is not significant, so this ratio is unstable)
- Share of D's gain recovered by C (round 1): -33.3% [-433.3, 42.9] (D's own gain over A is not significant, so this ratio is unstable)
