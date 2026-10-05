# NA3 rows: per-row SHAP values for the beeswarm plots, sigma and zT (G4)

- **Bundle**: `na3r_b.tar.gz`, SHA256 `00272f545da60cfde41f197eed6c2f16c3608f5db8e302f6b9fb4c40da5afd85` (verified, 2026-10-05); 23 manifest entries re-verified, no mismatch and no file outside the manifest. `status.json`: complete, 10 of 10 units, `accepted_as_result` true.
- **Code** `7529e7cfb7f14341994d1f57123e11edc75443e4` (clean tree, `--expect-commit` equal), snapfix CSV `d9fc1e5d...e489`, Python 3.12.3, xgboost 2.0.3, Tesla T4, 0.077 h.
- **Design**: repeat 0 of the Paper A chemistry-cluster folds (5 folds), the frozen Paper A hyperparameters; per fold a seeded subsample of 1,000 test rows (`default_rng(seed + fold)`), TreeSHAP of all 397 features, the feature values,
  row positions, measured and predicted values, the SHAP bias. Each unit asserts that SHAP plus bias equals the prediction; the fold R2 of every unit equals the committed rung value (checked on all 10 units).
- Used by `scripts/make_figures_thesis.py` for Figure 11.
