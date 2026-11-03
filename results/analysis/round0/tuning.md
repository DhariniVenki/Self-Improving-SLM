Run D tuning on heldout (150 problems). Base model: 48.0% [40.0, 56.0]. 12 of 12 settings score below the base model. McNemar p against the base model, Holm-corrected over all settings.

| LR | Epochs | heldout pass@1 | vs base | only base / only tuned | p (Holm) | train loss |
|---|---|---|---|---|---|---|
| 5e-05 | 1 | 40.0 | -8.0 | 21 / 9 | 0.513 | 0.690 |
| 5e-05 | 2 | 44.0 | -4.0 | 14 / 8 | 1.000 | 0.637 |
| 5e-05 | 3 | 42.7 | -5.3 | 15 / 7 | 1.000 | 0.604 |
| 0.0001 | 1 | 45.3 | -2.7 | 16 / 12 | 1.000 | 0.664 |
| 0.0001 | 2 | 44.0 | -4.0 | 13 / 7 | 1.000 | 0.600 |
| 0.0001 | 3 | 44.0 | -4.0 | 16 / 10 | 1.000 | 0.554 |
| 0.0002 | 1 | 44.0 | -4.0 | 13 / 7 | 1.000 | 0.643 |
| 0.0002 | 2 | 43.3 | -4.7 | 15 / 8 | 1.000 | 0.558 |
| 0.0002 | 3 | 44.0 | -4.0 | 19 / 13 | 1.000 | 0.491 |
| 0.0005 | 1 | 42.0 | -6.0 | 17 / 8 | 1.000 | 0.629 |
| 0.0005 | 2 | 47.3 | -0.7 | 14 / 13 | 1.000 | 0.511 |
| 0.0005 | 3 | 40.0 | -8.0 | 24 / 12 | 0.718 | 0.412 |

Run D used the setting LR 0.0005, 2 epochs, which scored 47.3% in tuning. Trained and evaluated again, it scored 44.7%: repeating one setting moves heldout pass@1 by 2.7 points.
