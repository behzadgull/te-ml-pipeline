# NA6: carrier-type classifier (p versus n) under chemistry-cluster CV, Kaggle T4

- **Bundle**: `na6.tar.gz`, SHA256 `2bc38410933e38d3ecc2a37315487a0a56394f91e811a9fe7d2b5870812b8cfd` (verified on the downloaded file); this folder is the unpacked bundle; its 55
  manifest entries were re-verified after unpacking (no mismatch). `status.json`: complete, 26 of 26 units (25 folds and the final model), `accepted_as_result` true.
- **Code**: commit `b2ad3cf83ff33a15de294bf8db22d8f1d5e40ed5`, clean tree, `allow_dirty` false. **Data**: snapfix CSV SHA256 `d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489`.
  **Environment**: Python 3.12.3 (uv venv), xgboost 2.0.3, Tesla T4 (GPU 0), 0.152 h, session started 2026-10-04 10:09:16Z (together with the final-models run on GPU 1).
- **Design**: label S > 0 (p-type); the 8 rows with S = 0 are dropped (185,056 rows); XGBClassifier with the S regressor's frozen hyperparameters (not tuned separately); the
  same 5 x 5 chemistry-cluster folds as the rest of the work. `final_classifier.json` (SHA256 `11b20af8138e815ae6aeb04cecd3c92ed62b4ddf48aa15f84a931d0a9b0363c7`) is the classifier
  fitted on all rows.
- Metrics with confidence intervals: `../../na6_metrics/` (computed by `scripts/na6_metrics.py` from the unit predictions).
