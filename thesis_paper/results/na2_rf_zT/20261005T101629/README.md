# NA2 random forest with the one fixed setting, target zT

- **What**: NA2 random forest with the one fixed setting, target zT: the 25 rung folds, no tuning (CPU, 4 cores).
- **Bundle**: `na2_rf_zT.tar.gz`, SHA256 `96812a5347e9d03de4e23d80d00d92cebe45b144e06c07d23d16fb3186797abf` (verified, 2026-10-06); unpacked here; the 54 manifest entries re-verified, no mismatch and no file outside the manifest. `status.json`: complete, 25 of 25 units, `accepted_as_result` true.
- **Code** `7529e7cfb7f14341994d1f57123e11edc75443e4` (clean tree, `--expect-commit` equal), snapfix CSV `d9fc1e5d...`, Python 3.12.3, device cpu, 6.03 h (2026-10-05T10:16:29Z to 2026-10-05T16:18:05Z).
- Per-fold `y_true` and `y_pred` are in `units/*.npz` (model scale: log10 for sigma and kappa); the analyses are `scripts/model_comparison.py` outputs under `results/na2_comparison` and `results/na1_comparison`.
