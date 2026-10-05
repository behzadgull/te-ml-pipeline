# NA3: SHAP attribution of the sigma and zT models, chemistry-cluster folds (G3, GPU 1)

- **Bundle**: `na3_b.tar.gz`, SHA256 `7259f608abcadf7a2dc6b1939799c2a5db62046c0a27794a208bfe43de2e12e9` (verified); 103 manifest entries re-verified, no mismatch. `status.json`: complete, 50 of 50 units
  (sigma and zT x 5 repeats x 5 folds), `accepted_as_result` true.
- **Code** `33e9662838ba299a30bc8fccba0caee26d6bc435` (clean tree), snapfix CSV `d9fc1e5d...`, Python 3.12.3, xgboost 2.0.3, Tesla T4, 0.47 h; frozen hyperparameters of Paper A per target; native TreeSHAP
  on a fixed random subsample of 20,000 test rows per fold (`shap_rows`), mean |SHAP| per feature normalised to shares, 397 features. Only the chemistry-cluster arm.
- `results.json`: per target the 20 features with the largest mean |SHAP| (mean over the 25 folds and the SD across folds) and the share of MAGPIE, CBFV and temperature features.
