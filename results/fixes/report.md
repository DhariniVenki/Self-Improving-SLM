Chosen fix: B32_weighted_lr0.0005 (best heldout score of 6 candidates). Each row averages 3 repeated runs. Test is the split to quote: heldout was used to choose the fix.

### test

| Group | Repeats | Mean pass@1 |
|---|---|---|
| A | 33.6, 33.0, 32.8 | 33.1 |
| B_round1 | 35.2, 34.0, 34.2 | 34.5 |
| B32_weighted_lr0.0005 | 36.4, 35.2, 36.8 | 36.1 |

- B32_weighted_lr0.0005 minus A: +3.0 points [+0.4, +5.6]; McNemar p per repeat (Holm): 0.914, 1.000, 0.220
- B32_weighted_lr0.0005 minus B_round1: +1.7 points [-0.6, +3.9]; McNemar p per repeat (Holm): 1.000, 1.000, 0.914
- B_round1 minus A: +1.3 points [-1.3, +4.0]; McNemar p per repeat (Holm): 1.000, 1.000, 1.000

### heldout

| Group | Repeats | Mean pass@1 |
|---|---|---|
| A | 48.0, 47.3, 48.0 | 47.8 |
| B_round1 | 45.3, 44.7, 48.7 | 46.2 |
| B32_weighted_lr0.0005 | 50.0, 48.0, 46.7 | 48.2 |

- B32_weighted_lr0.0005 minus A: +0.4 points [-4.2, +5.3]; McNemar p per repeat (Holm): 1.000, 1.000, 1.000
- B32_weighted_lr0.0005 minus B_round1: +2.0 points [-2.0, +6.0]; McNemar p per repeat (Holm): 1.000, 1.000, 1.000
- B_round1 minus A: -1.6 points [-6.0, +2.9]; McNemar p per repeat (Holm): 1.000, 1.000, 1.000
