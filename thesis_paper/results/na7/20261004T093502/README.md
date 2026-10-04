# NA7: direct versus derived zT under random row-level validation (Kaggle T4 x2)

- **Bundle**: `na7.tar.gz` from Kaggle, SHA256 `6aa494e19ea4b99da963401abd02a78dbeaa3f46946dad00737b3f64950dac60` (equal to the value Kaggle printed and to
  the hash of the downloaded file). This folder is the unpacked bundle; the 55 files in `manifest.json` were each checked against their recorded
  SHA256 after unpacking, with no mismatch. The bundle's own file name is `na7/`; the folder timestamp is the download's modification time, UTC.
- **Code**: commit `b2ad3cf83ff33a15de294bf8db22d8f1d5e40ed5`, `tree_clean` true and `allow_dirty` false in all three session configs.
- **Data**: snapfix CSV, SHA256 `d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489`.
- **Environment**: Python 3.12.3 (uv venv), numpy 1.26.4, pandas 2.2.2, scikit-learn 1.4.2, xgboost 2.0.3, two Tesla T4 (driver 580.178.04).
- **Sessions** (`run_configs/`): `session_01.json` (shard 0 of 2, GPU 0, 0.197 h), `session_01_restored1.json` (shard 1 of 2, GPU 1, 0.183 h), run
  concurrently from 09:08:33Z; `session_03.json` the merge (`--restore-from` both shards, `--max-units 0`, so it computed nothing and wrote
  `results.json`). `status.json`: complete, 25 of 25 units, `accepted_as_result` true.
- **Design**: the 56,088-row all-four-properties-present subset of the grouped direct-vs-derived run; shuffled row-level 5 x 5 KFold (the
  project's `outer_splits("kfold")` with the project's seeding); four XGBoost models, each on its own target's frozen hyperparameters; derived zT
  = S^2 sigma T / kappa from the three component predictions without smearing correction (as in the committed grouped run).
- **Result** (`results.json`, n = 280,440 pooled out-of-fold rows): direct zT R2 0.917, derived 0.880, gap 0.036; the committed chemistry-cluster
  run (`results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda/`) has direct 0.727, derived 0.540, gap 0.186, so the grouped gap is
  5.1 times the random gap (computed in `thesis_paper/scripts/thesis_values.py`, `na7_values()`, which asserts the subset, the folds and the
  provenance fields before printing it).
- The unit predictions (`units/*.npz`) are Git LFS objects. The smear factors in `results.json` (sigma 1.060, kappa 1.014) are this run's own
  out-of-fold values, reported for completeness; they do not enter the derived R2.
