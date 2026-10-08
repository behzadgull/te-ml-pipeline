# NA2 random forest with the one fixed setting (docs/decisions.md, 2026-10-05), target sigma (log10)

- **What**: NA2 random forest with the one fixed setting (docs/decisions.md, 2026-10-05), target sigma (log10): the 25 chemistry-cluster rung folds, no tuning (CPU, 4 cores).
- **Bundle**: `na2_rf_sigma.tar.gz`, SHA256 `d4fc93f7729bfca756c82cd6c732e062018c9c0a01f2a94b1ec4c542a247c987` (verified, 2026-10-06); unpacked here; the 54 manifest entries re-verified, no mismatch and no file outside the manifest. `status.json`: complete, 25 of 25 units, `accepted_as_result` true.
- **Code** `7529e7cfb7f14341994d1f57123e11edc75443e4` (clean tree, `--expect-commit` equal), snapfix CSV `d9fc1e5d...`, Python 3.12.3, device cpu, 8.43 h (2026-10-05T10:20:59Z to 2026-10-05T18:46:31Z).
- Per-fold `y_true` and `y_pred` are in `units/*.npz` (model scale: log10 for sigma and kappa); the analyses are `scripts/model_comparison.py` outputs under `results/na2_comparison` and `results/na1_comparison`.
