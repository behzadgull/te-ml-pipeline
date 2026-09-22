# Ungrouped ladder rerun on the snapfix dataset

Source: `ungrouped_snapfix.zip` (Kaggle), extracted to `kaggle_out/ungrouped_snapfix/`. All 12 runs verified: correct file counts (20/5/10 prediction `.npz` files) and correct row counts (S 185,064 / sigma 182,755 / kappa 121,110 / zT 129,419 -- the snapfix/File A counts, NOT the old ladder_regen_dl counts of 185,844/183,246/121,535/129,851). No STOP condition triggered.

## Item 3: pooled vs per-fold/per-draw R2, per run

| Target | Rung | n | Pooled R2 | Per-fold/draw mean +/- SD | |diff| | Flag |
|---|---|---|---|---|---|---|
| S | random 80/20 | 20 | 0.9582 | 0.9582 +/- 0.0013 | 0.00000 |  |
| S | 5-fold | 5 | 0.9585 | 0.9585 +/- 0.0013 | 0.00000 |  |
| S | 10-fold | 10 | 0.9595 | 0.9595 +/- 0.0014 | 0.00001 |  |
| sigma | random 80/20 | 20 | 0.9150 | 0.9150 +/- 0.0009 | 0.00000 |  |
| sigma | 5-fold | 5 | 0.9152 | 0.9152 +/- 0.0009 | 0.00000 |  |
| sigma | 10-fold | 10 | 0.9174 | 0.9174 +/- 0.0024 | 0.00000 |  |
| kappa | random 80/20 | 20 | 0.9434 | 0.9434 +/- 0.0012 | 0.00000 |  |
| kappa | 5-fold | 5 | 0.9436 | 0.9436 +/- 0.0013 | 0.00001 |  |
| kappa | 10-fold | 10 | 0.9455 | 0.9455 +/- 0.0021 | 0.00002 |  |
| zT | random 80/20 | 20 | 0.9180 | 0.9179 +/- 0.0014 | 0.00001 |  |
| zT | 5-fold | 5 | 0.9181 | 0.9181 +/- 0.0029 | 0.00000 |  |
| zT | 10-fold | 10 | 0.9193 | 0.9193 +/- 0.0032 | 0.00000 |  |

No run's pooled-vs-per-fold-mean difference exceeds 0.0005 (max: 0.00002). **Recommendation: use pooled R2 for Table 1**, consistent with the per-repeat pooled convention composition/chemistry already use, and with how these three rungs were already reported before this rerun -- not a correction, a convention match.

## Item 4: old (ladder_regen_dl) vs new (snapfix) ungrouped cells

| Target | Rung | Old (ladder_regen_dl) | New (snapfix, pooled) | Delta (new-old) |
|---|---|---|---|---|
| S | random 80/20 | 0.9588 | 0.9582 | -0.0006 |
| S | 5-fold | 0.9588 | 0.9585 | -0.0003 |
| S | 10-fold | 0.9595 | 0.9595 | +0.0000 |
| sigma | random 80/20 | 0.9152 | 0.9150 | -0.0002 |
| sigma | 5-fold | 0.9150 | 0.9152 | +0.0002 |
| sigma | 10-fold | 0.9175 | 0.9174 | -0.0001 |
| kappa | random 80/20 | 0.9442 | 0.9434 | -0.0008 |
| kappa | 5-fold | 0.9444 | 0.9436 | -0.0008 |
| kappa | 10-fold | 0.9459 | 0.9455 | -0.0004 |
| zT | random 80/20 | 0.9186 | 0.9180 | -0.0006 |
| zT | 5-fold | 0.9184 | 0.9181 | -0.0003 |
| zT | 10-fold | 0.9196 | 0.9193 | -0.0003 |

All deltas are small (|delta| <= 0.0008), consistent with the row-count difference between the two datasets being a few tenths of a percent and having no material effect on the ungrouped rungs specifically -- unlike the grouped rungs, where the dataset difference is compounded by the SNAP(0.05)/S5 grouping-fix effect on which rows land in which fold.

## Item 5: corrected Table 1, all five rungs from the snapfix dataset

| Target | random 80/20 | 5-fold | 10-fold | composition | chemistry cluster | gap (random - chemistry) |
|---|---|---|---|---|---|---|
| S | 0.9582 | 0.9585 | 0.9595 | 0.8322 +/- 0.0044 | 0.7528 +/- 0.0050 | 0.2054 |
| sigma | 0.9150 | 0.9152 | 0.9174 | 0.7762 +/- 0.0015 | 0.7020 +/- 0.0020 | 0.2130 |
| kappa | 0.9434 | 0.9436 | 0.9455 | 0.8565 +/- 0.0013 | 0.8092 +/- 0.0021 | 0.1342 |
| zT | 0.9180 | 0.9181 | 0.9193 | 0.8178 +/- 0.0009 | 0.7456 +/- 0.0045 | 0.1724 |

Gap range: **0.1342 (kappa) to 0.2130 (sigma)**, mean **0.1812**.
Old gap range (cited in task, ladder_regen_dl ungrouped vs snapfix chemistry -- a cross-dataset comparison): 0.135-0.213, mean 0.182.
New gap range (snapfix throughout, apples-to-apples): 0.1342-0.2130, mean 0.1812 -- nearly unchanged from the old (mismatched-dataset) figure, since the ungrouped side only moved by <=0.0008 per target (see Item 4).
