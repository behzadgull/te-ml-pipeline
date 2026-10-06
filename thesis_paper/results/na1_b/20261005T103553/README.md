# NA1 nested grouped CV, targets sigma and zT, as na1_a (GPU 1)

- **What**: NA1 nested grouped CV, targets sigma and zT, as na1_a (GPU 1).
- **Bundle**: `na1_b.tar.gz`, SHA256 `9c973ad6e7e4d816a860b7e38329074cc1eaeda60c197e130ad508fd4c663504` (verified, 2026-10-06); unpacked here; the 1103 manifest entries re-verified, no mismatch and no file outside the manifest. `status.json`: complete, 1050 of 1050 units, `accepted_as_result` true.
- **Code** `7529e7cfb7f14341994d1f57123e11edc75443e4` (clean tree, `--expect-commit` equal), snapfix CSV `d9fc1e5d...`, Python 3.12.3, device cuda, 8.21 h (2026-10-05T10:35:53Z to 2026-10-05T18:48:18Z).
- Per-fold `y_true` and `y_pred` are in `units/*.npz` (model scale: log10 for sigma and kappa); the analyses are `scripts/model_comparison.py` outputs under `results/na2_comparison` and `results/na1_comparison`.
