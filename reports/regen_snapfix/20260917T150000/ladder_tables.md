# Post-snapfix regeneration -- comparison tables

Generated: 20260917T150000. Source: `kaggle_out/regen_snapfix/` (8 runs, extracted from
`regen_postfix (1).zip` in Downloads -- NOT the identically-prefixed `regen_postfix.zip`,
which is the earlier post-bugfix (pre-snapfix) regeneration already tabulated in
`reports/regen_postfix/20260916T161906/`. Disambiguated by progress.log header group counts
(e.g. kappa: 5,908 chemistry_cluster_id groups) matching this session's snapfix CSV
regeneration exactly, not by filename or date alone.

All 8 run directories present and complete (25 repeat*_fold*_predictions.npz each),
397 features, frozen hyperparameters, computed from
`data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv`.

## Run completeness and group counts

| Run | n_features | n_rows (from header) | n_groups | npz count |
|---|---|---|---|---|
| S_composition_full | 397 | 185,064 | 15,214 | 25 |
| S_chemistry_full | 397 | 185,064 | 7,810 | 25 |
| sigma_composition_full | 397 | 182,755 | 14,391 | 25 |
| sigma_chemistry_full | 397 | 182,755 | 7,317 | 25 |
| kappa_composition_full | 397 | 121,110 | 11,762 | 25 |
| kappa_chemistry_full | 397 | 121,110 | 5,908 | 25 |
| zT_composition_full | 397 | 129,419 | 11,401 | 25 |
| zT_chemistry_full | 397 | 129,419 | 5,632 | 25 |

All 8 runs: n_features=397 (expected), all complete (25/25 predictions files each). None missing or incomplete.

## TABLE A -- Five-Way Ladder (grouped rungs NEW/post-snapfix; ungrouped rungs carried over from CLAUDE.md)

| Target | random 80/20 (carried over) | 5-fold (carried over) | 10-fold (carried over) | composition (NEW) | chemistry (NEW) | random-minus-chemistry gap |
|---|---|---|---|---|---|---|
| S | 0.9588 | 0.9588 | 0.9595 | 0.8322 +/- 0.0044 | 0.7528 +/- 0.0050 | 0.2060 |
| sigma (log10) | 0.9152 | 0.9150 | 0.9175 | 0.7762 +/- 0.0015 | 0.7020 +/- 0.0020 | 0.2132 |
| kappa (log10) | 0.9442 | 0.9444 | 0.9459 | 0.8565 +/- 0.0013 | 0.8092 +/- 0.0021 | 0.1350 |
| zT | 0.9186 | 0.9184 | 0.9196 | 0.8178 +/- 0.0009 | 0.7456 +/- 0.0045 | 0.1730 |

Gap range: 0.1350 - 0.2132. Mean gap across 4 targets: 0.1818.

## Old (CLAUDE.md) vs new (post-snapfix), every grouped cell

| Target | Rung | OLD | NEW | delta (new-old) |
|---|---|---|---|---|
| S | composition | 0.8331 +/- 0.0034 | 0.8322 +/- 0.0044 | -0.0009 |
| S | chemistry | 0.8076 +/- 0.0018 | 0.7528 +/- 0.0050 | -0.0548 |
| sigma (log10) | composition | 0.7772 +/- 0.0019 | 0.7762 +/- 0.0015 | -0.0010 |
| sigma (log10) | chemistry | 0.7600 +/- 0.0008 | 0.7020 +/- 0.0020 | -0.0580 |
| kappa (log10) | composition | 0.8562 +/- 0.0010 | 0.8565 +/- 0.0013 | +0.0003 |
| kappa (log10) | chemistry | 0.8460 +/- 0.0011 | 0.8092 +/- 0.0021 | -0.0368 |
| zT | composition | 0.8174 +/- 0.0019 | 0.8178 +/- 0.0009 | +0.0004 |
| zT | chemistry | 0.7968 +/- 0.0030 | 0.7456 +/- 0.0045 | -0.0512 |

## Across-repeat SD, old vs new -- all 8 grouped runs

| Target | Rung | OLD SD | NEW SD | change (new-old) | narrower? |
|---|---|---|---|---|---|
| S | chemistry | 0.0018 | 0.0050 | +0.0032 | no |
| S | composition | 0.0034 | 0.0044 | +0.0010 | no |
| sigma (log10) | chemistry | 0.0008 | 0.0020 | +0.0012 | no |
| sigma (log10) | composition | 0.0019 | 0.0015 | -0.0004 | YES |
| kappa (log10) | chemistry | 0.0011 | 0.0021 | +0.0010 | no |
| kappa (log10) | composition | 0.0010 | 0.0013 | +0.0003 | no |
| zT | chemistry | 0.0030 | 0.0045 | +0.0015 | no |
| zT | composition | 0.0019 | 0.0009 | -0.0010 | YES |

2 of 8 narrowed, 6 of 8 widened. Mean signed change: +0.00086. Mean absolute change: 0.00120.

## Fold-level SD vs across-repeat SD ratio, each new (post-snapfix) run

Ratio = fold_level_std (25 individual outer-fold R2 values) / per_repeat_std (5 per-repeat-pooled R2 values).
A large ratio means most of the fold-to-fold spread washes out when pooled within a repeat --
i.e. repeat-level pooling is doing real variance reduction, consistent with S5 spreading the
largest groups across folds rather than pinning them to one fold index every repeat.

| Target | Rung | fold_level SD | per_repeat SD | ratio |
|---|---|---|---|---|
| S | composition | 0.0173 | 0.0044 | 3.95 |
| S | chemistry | 0.0325 | 0.0050 | 6.48 |
| sigma (log10) | composition | 0.0129 | 0.0015 | 8.52 |
| sigma (log10) | chemistry | 0.0294 | 0.0020 | 14.34 |
| kappa (log10) | composition | 0.0080 | 0.0013 | 6.31 |
| kappa (log10) | chemistry | 0.0172 | 0.0021 | 8.15 |
| zT | composition | 0.0141 | 0.0009 | 15.18 |
| zT | chemistry | 0.0420 | 0.0045 | 9.28 |

### zT chemistry-cluster ratio, explicit callout (task's stated expectation)

Task's cited old figures: across-repeat SD 0.0030, fold-level SD 0.0154 -> ratio 5.13.

**Independently recomputed old-code fold-level SD from `checkpoints/saved_predictions/checkpoints/zT_chemistry/`:
0.01661, not 0.0154 as cited** -- a ~7.5% discrepancy, flagged rather than silently
reconciled. Recomputed old ratio using the verified fold-level SD: 5.54.

New (post-snapfix) zT chemistry: fold_level SD=0.0420, per_repeat SD=0.0045, ratio=9.28.

**Finding, stated plainly: the ratio WIDENED (from ~5.5 old to 9.3 new), it did not narrow.**
This is the opposite of the task's stated expectation ("if S5 is working that ratio should narrow").
Both the fold-level SD (0.0166 -> 0.0420) and the across-repeat SD (0.0030 -> 0.0045) increased in
absolute terms for zT chemistry under the post-snapfix regeneration, but fold-level spread increased
proportionally more than across-repeat spread, so pooling-within-repeat is washing out LESS of the
fold-to-fold variance than it did under the old code, not more. Reported as observed, not interpreted
further -- no conclusion is drawn here per the task's instruction.
