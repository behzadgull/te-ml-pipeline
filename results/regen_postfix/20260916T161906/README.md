# regen_postfix: control run (pre-snapfix cluster ids, corrected fold code)

Do not cite these numbers as results. They exist to test the fold-code correction
on the original cluster identifiers; the published chemistry-cluster rung is
`results/ladder_regen_snapfix/20260917T150000/`.

## What this run is

- 16 Kaggle runs on File A's rows (185,064 / 182,755 / 121,110 / 129,419 for
  S / sigma / kappa / zT): per target, chemistry-cluster CV with the full
  (397), CBFV (265) and MAGPIE (133) feature sets, plus composition CV with
  the full set. 5 repeats x 5 outer folds each, frozen hyperparameters from
  `checkpoints/saved_predictions/checkpoints/frozen_hyperparams/{target}.json`.
  Run dates 2026-09-16 (first log lines 11:09 to 12:50, Kaggle clock).
- It used the **pre-snapfix `chemistry_cluster_id`**: the chemistry-cluster
  runs log 10,679 / 9,964 / 8,215 / 7,944 groups (S / sigma / kappa / zT), the
  same as the 2026-08-22 pre-fix baseline (10,679 for S). The post-snapfix
  runs log 7,810 (S) and 5,632 (zT). The composition runs log
  15,214 / 14,391 / 11,762 / 11,401 composition_id groups, which the snapfix
  did not change.
- It used the **corrected fold code** (S5 `randomized_group_kfold`, commit
  c1c6873, 2026-09-16 15:57 +0500). Evidence: with identical cluster ids, the
  fold-level R2 differs from the pre-fix baseline (S repeat 0 fold 0: 0.8322
  here, 0.8151 pre-fix), so the partitions changed. Not recorded in the run
  config: the code commit. If the Kaggle clock is UTC, the first log line
  (11:09) follows the commit (10:57 UTC) by 12 minutes.

## Role

A control that tests the fold-code correction on the original cluster
identifiers. It has no reverse counterpart (original fold code with the snapped
identifiers), so the split of the total drop below is arithmetic, not causal.
Pooled chemistry-cluster R2 (full feature set, mean of per-repeat pooled R2):

| Target | pre-fix (old ids, old fold code) | this run (old ids, new fold code) | post-fix (new ids, new fold code) |
|---|---|---|---|
| S | 0.8076 | 0.8064 | 0.7528 |
| sigma (log10) | 0.7600 | 0.7579 | 0.7020 |
| kappa (log10) | 0.8460 | 0.8441 | 0.8092 |
| zT | 0.7968 | 0.7979 | 0.7456 |

Under the corrected fold code, the original-identifier rung moves by at most 0.0022
in absolute value and with mixed sign; the remaining change (grouping key plus any
interaction) is 0.035 to 0.056. Recomputed from tracked inputs
by `scripts/grouping_fix_effect.py` (report under
`reports/grouping_fix_effect/`). The aggregate tables for this run are in
`reports/regen_postfix/20260916T161906/`.

## What is here

433 files: per-run `run_config.json`, `progress.log`, per-fold
`repeat{r}_fold{f}.json` (outer R2, hyperparameters, n_train, n_test), and the
Kaggle `environment.txt`. Copied byte-identical from `kaggle_out/regen_postfix/`.

## What is not here, by design

The 400 per-fold `*_predictions.npz` files are not committed.

- 296 are archived off-machine in `kaggle_out_unmatched_20260924.tar.gz`,
  SHA256 `b3952f6c4258458a03b10834525062244e31acb02591a09c221e4fa79bb80bea`.
  Per-file SHA256 (paths as inside the tar) in
  `results/archive_manifests/kaggle_out_unmatched.sha256`.
- Archive location: Google Drive (private),
  https://drive.google.com/file/d/1MNKVq8QbdU5nwcmkieAPfHDj3TjyPpqQ/view?usp=drive_link
  (SHA256 verified after a download round-trip).
- The other 104 are byte-identical to tracked files: the 100 composition-run
  files equal the same runs under `results/ladder_regen_snapfix/20260917T150000/`
  (composition grouping is unaffected by the snapfix), and 4 kappa
  chemistry files equal tracked descriptor-ablation predictions.

## Known gap

The run configs do not record the input dataset's SHA256 or the code commit.
The dataset is identified only by row count and group count (see above).
