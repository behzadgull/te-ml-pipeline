# NA1 nested grouped CV, targets S and kappa

- **What**: NA1 nested grouped CV, targets S and kappa: for each of the 25 outer folds a fresh 20-trial search on the outer training rows only, then the refit and the prediction of the outer test rows (GPU 0).
- **Bundle**: `na1_a.tar.gz`, SHA256 `2ab1404957389e49c7d4a3bf4e22afc8822f154222716de15c9fd0091bb659a8` (verified, 2026-10-06); unpacked here; the 1103 manifest entries re-verified, no mismatch and no file outside the manifest. `status.json`: complete, 1050 of 1050 units, `accepted_as_result` true.
- **Code** `7529e7cfb7f14341994d1f57123e11edc75443e4` (clean tree, `--expect-commit` equal), snapfix CSV `d9fc1e5d...`, Python 3.12.3, device cuda, 7.56 h (2026-10-05T10:35:53Z to 2026-10-05T18:09:03Z).
- Per-fold `y_true` and `y_pred` are in `units/*.npz` (model scale: log10 for sigma and kappa); the analyses are `scripts/model_comparison.py` outputs under `results/na2_comparison` and `results/na1_comparison`.
