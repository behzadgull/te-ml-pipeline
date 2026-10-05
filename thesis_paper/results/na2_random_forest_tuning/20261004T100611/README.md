# NA2 random forest, first session: S tuning record (INCOMPLETE, not an accepted result)

- **Bundle**: `na2_random_forest.tar.gz`, SHA256 `fe755d4b701ebcb6734ccb1658c69bcc0c5fdd39b78e45cfb83283bdbefcc0dc` (verified); 17 manifest entries re-verified. `status.json`: complete false, 15 of 46
  units, `accepted_as_result` false. Code `b2ad3cf83ff33a15de294bf8db22d8f1d5e40ed5` (clean tree), snapfix CSV `d9fc1e5d...`, Kaggle CPU (4 cores), Python 3.12.3, 10.70 h; stopped by the time budget.
- Only tuning units exist (`units/tune_S_trial000..014.json`): 15 Optuna trials for S, grouped 3-fold chemistry-cluster CV on all S rows, the search space of `src/nested_cv.py`
  (`_random_forest_search_space`). No outer-fold unit exists, so no random-forest test-fold prediction or score was produced.
- **Why it is kept**: as the tuning record behind the design change of `docs/decisions.md` (2026-10-05): the random forest is run with one fixed, literature-based set, no tuning. Its numbers are in
  `../summary/tuning_summary.json` (made by `scripts/na2_tuning_summary.py`).
