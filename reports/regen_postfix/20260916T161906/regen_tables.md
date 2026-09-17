# Post-fix regeneration -- comparison tables

Generated: 20260916T161906. Source: `kaggle_out/regen_postfix/` (16 runs, extracted from
`regen_postfix.zip`), computed from saved per-row predictions with the same pooling code
used throughout this project (per-repeat pooled R2 from concatenated repeat predictions,
each target's own scale: linear for S/zT, log10 for sigma/kappa).

Note: pooled R2 across all 25 folds equals the arithmetic mean of the 5 per-repeat pooled
R2 values exactly, for every run below -- a mathematical identity here (not a copy-paste),
since each repeat's 5 outer folds are a full, non-overlapping partition of the identical
row set, so per-repeat SS_tot is constant across repeats and pooling distributes evenly.

## TABLE A -- Five-Way Ladder (grouped rungs regenerated; ungrouped rungs carried over)

| Target | random 80/20 (carried over) | 5-fold (carried over) | 10-fold (carried over) | composition (NEW) | chemistry (NEW) | random-minus-chemistry gap |
|---|---|---|---|---|---|---|
| S | 0.9588 | 0.9588 | 0.9595 | 0.8322 +/- 0.0044 | 0.8064 +/- 0.0020 | 0.1524 |
| sigma (log10) | 0.9152 | 0.9150 | 0.9175 | 0.7762 +/- 0.0015 | 0.7579 +/- 0.0012 | 0.1573 |
| kappa (log10) | 0.9442 | 0.9444 | 0.9459 | 0.8565 +/- 0.0013 | 0.8441 +/- 0.0009 | 0.1001 |
| zT | 0.9186 | 0.9184 | 0.9196 | 0.8178 +/- 0.0009 | 0.7979 +/- 0.0024 | 0.1207 |

Gap range: 0.1001 - 0.1573. Mean gap across 4 targets: 0.1326.

## TABLE B -- Descriptor Ablation (all three feature sets regenerated)

| Target | magpie (133) NEW | cbfv (265) NEW | full (397) NEW = new chemistry rung | full-minus-magpie delta |
|---|---|---|---|---|
| S | 0.8028 +/- 0.0019 | 0.8052 +/- 0.0018 | 0.8064 +/- 0.0020 | +0.0036 |
| sigma (log10) | 0.7475 +/- 0.0014 | 0.7564 +/- 0.0019 | 0.7579 +/- 0.0012 | +0.0103 |
| kappa (log10) | 0.8397 +/- 0.0015 | 0.8417 +/- 0.0012 | 0.8441 +/- 0.0009 | +0.0044 |
| zT | 0.7932 +/- 0.0029 | 0.7965 +/- 0.0030 | 0.7979 +/- 0.0024 | +0.0048 |

## Old (CLAUDE.md) vs new, every cell -- both tables

### Table A cells (composition, chemistry)

| Target | Rung | OLD | NEW | delta (new-old) | |delta|>0.02? |
|---|---|---|---|---|---|
| S | composition | 0.8331 +/- 0.0034 | 0.8322 +/- 0.0044 | -0.0009 | no |
| S | chemistry | 0.8076 +/- 0.0018 | 0.8064 +/- 0.0020 | -0.0012 | no |
| sigma (log10) | composition | 0.7772 +/- 0.0019 | 0.7762 +/- 0.0015 | -0.0010 | no |
| sigma (log10) | chemistry | 0.7600 +/- 0.0008 | 0.7579 +/- 0.0012 | -0.0021 | no |
| kappa (log10) | composition | 0.8562 +/- 0.0010 | 0.8565 +/- 0.0013 | +0.0003 | no |
| kappa (log10) | chemistry | 0.8460 +/- 0.0011 | 0.8441 +/- 0.0009 | -0.0019 | no |
| zT | composition | 0.8174 +/- 0.0019 | 0.8178 +/- 0.0009 | +0.0004 | no |
| zT | chemistry | 0.7968 +/- 0.0030 | 0.7979 +/- 0.0024 | +0.0011 | no |

### Table B cells (magpie, cbfv, full)

| Target | Feature set | OLD | NEW | delta (new-old) | |delta|>0.02? |
|---|---|---|---|---|---|
| S | magpie | 0.8038 +/- 0.0031 | 0.8028 +/- 0.0019 | -0.0010 | no |
| S | cbfv | 0.8060 +/- 0.0025 | 0.8052 +/- 0.0018 | -0.0008 | no |
| S | full | 0.8076 +/- 0.0018 | 0.8064 +/- 0.0020 | -0.0012 | no |
| sigma (log10) | magpie | 0.7497 +/- 0.0010 | 0.7475 +/- 0.0014 | -0.0022 | no |
| sigma (log10) | cbfv | 0.7585 +/- 0.0013 | 0.7564 +/- 0.0019 | -0.0021 | no |
| sigma (log10) | full | 0.7600 +/- 0.0008 | 0.7579 +/- 0.0012 | -0.0021 | no |
| kappa (log10) | magpie | 0.8419 +/- 0.0009 | 0.8397 +/- 0.0015 | -0.0022 | no |
| kappa (log10) | cbfv | 0.8436 +/- 0.0013 | 0.8417 +/- 0.0012 | -0.0019 | no |
| kappa (log10) | full | 0.8460 +/- 0.0011 | 0.8441 +/- 0.0009 | -0.0019 | no |
| zT | magpie | 0.7940 +/- 0.0016 | 0.7932 +/- 0.0029 | -0.0008 | no |
| zT | cbfv | 0.7945 +/- 0.0042 | 0.7965 +/- 0.0030 | +0.0020 | no |
| zT | full | 0.7968 +/- 0.0030 | 0.7979 +/- 0.0024 | +0.0011 | no |

**No cell in either table moved by more than 0.02 in either direction.**

## Across-repeat SD, old vs new -- the 8 grouped runs (chemistry + composition, x4 targets)

| Target | Rung | OLD SD | NEW SD | change (new-old) | widened? |
|---|---|---|---|---|---|
| S | chemistry | 0.0018 | 0.0020 | +0.0002 | YES |
| S | composition | 0.0034 | 0.0044 | +0.0010 | YES |
| sigma (log10) | chemistry | 0.0008 | 0.0012 | +0.0004 | YES |
| sigma (log10) | composition | 0.0019 | 0.0015 | -0.0004 | no |
| kappa (log10) | chemistry | 0.0011 | 0.0009 | -0.0002 | no |
| kappa (log10) | composition | 0.0010 | 0.0013 | +0.0003 | YES |
| zT | chemistry | 0.0030 | 0.0024 | -0.0006 | no |
| zT | composition | 0.0019 | 0.0009 | -0.0010 | no |

4 of 8 widened, 4 of 8 narrowed. Mean signed change: -0.00003. Mean absolute change: 0.00050.
