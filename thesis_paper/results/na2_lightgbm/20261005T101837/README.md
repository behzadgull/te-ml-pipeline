# NA2 LightGBM, all four targets

- **What**: NA2 LightGBM, all four targets: 20 Optuna trials per target (the XGBoost protocol, 3 inner chemistry-cluster folds), then the 25 chemistry-cluster rung folds with the frozen set (CPU, 4 cores).
- **Bundle**: `na2_lightgbm.tar.gz`, SHA256 `1cf9d675fa8575c4edb162c587c88726f001b95efc0f1cb20b37ec03f522730e` (verified, 2026-10-06); unpacked here; the 291 manifest entries re-verified, no mismatch and no file outside the manifest. `status.json`: complete, 184 of 184 units, `accepted_as_result` true.
- **Code** `7529e7cfb7f14341994d1f57123e11edc75443e4` (clean tree, `--expect-commit` equal), snapfix CSV `d9fc1e5d...`, Python 3.12.3, device cpu, 2.68 h (2026-10-05T10:18:37Z to 2026-10-05T12:59:29Z).
- Per-fold `y_true` and `y_pred` are in `units/*.npz` (model scale: log10 for sigma and kappa); the analyses are `scripts/model_comparison.py` outputs under `results/na2_comparison` and `results/na1_comparison`.
