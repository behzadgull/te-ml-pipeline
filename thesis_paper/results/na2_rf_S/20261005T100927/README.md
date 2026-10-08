# NA2 random forest, target S

- **What**: NA2 random forest with the one fixed setting (docs/decisions.md, 2026-10-05), target S: the 25 chemistry-cluster rung folds, no tuning (CPU, 4 cores).
- **Bundle**: `na2_rf_S.tar.gz`, SHA256 `fa6789ed4af5f3b4b67b1f3708222ac36062108174c0157a2fb299fa9fa9d8e9` (verified, 2026-10-06); unpacked here; the 54 manifest entries re-verified, no mismatch and no file outside the manifest. `status.json`: complete, 25 of 25 units, `accepted_as_result` true.
- **Code** `7529e7cfb7f14341994d1f57123e11edc75443e4` (clean tree), snapfix CSV `d9fc1e5d...`, Python 3.12.3, device cpu, 9.79 h (2026-10-05T10:09:27Z to 2026-10-05T19:56:23Z). The frozen setting equals that of the sigma, kappa and zT bundles.
- Per-fold `y_true` and `y_pred` are in `units/*.npz` (model scale); the paired comparison is `scripts/model_comparison.py na2` under `results/na2_comparison`.
