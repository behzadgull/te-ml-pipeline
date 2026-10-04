# Final models on all rows (no classifier) and JARVIS predictions, Kaggle T4

- **Bundle**: `final_a.tar.gz`, SHA256 `91719958320c4a3c9b387ae670d15e7295a4624c8fe2ab6d42728b374eb63874` (verified); unpacked here; 13 manifest entries re-verified, no mismatch.
  `status.json`: complete, 5 of 5 units (four fits and the JARVIS prediction), `accepted_as_result` true.
- **Code** `b2ad3cf83ff33a15de294bf8db22d8f1d5e40ed5` (clean tree), snapfix CSV `d9fc1e5d...e489`, Python 3.12.3, xgboost 2.0.3, Tesla T4 (GPU 1), 0.036 h.
- **Inputs**: the JARVIS featurised CSV, SHA256 `3c23d5500fa0d3c6fa6a92313ee61e12fb90f93189697966d5de279adfc49c49` (23,218 entries); no classifier and no Materials Project file in this run, so
  the sign of S is the regressor's. Smear factors (frozen, from `results/external_snapfix/20260917T160553/estm_results.json`): sigma 1.3817, kappa 1.0715.
- **Models** (`models/<target>.json`, XGBoost, **not committed**: gitignored and 62 MB; SHA256 `S` f94d4052ede66930592982d1c6f9762457508acf2723741f0597594d1cb5661c, `sigma`
  b76f65aca7a90184d2e34c747f449ea38aa454e3680dd28ea67c516cb1f76684, `kappa` 026b6e6b6080826246bb39bad27e0fabfc5b14f8b5cb8f61f619d8abf4234967, `zT`
  cca78dd976a399e055f072ce47d15a36c6a45dabadc7e9b5cf0d0dfebe7fd426). They are in the bundle; the manifest lists them, so a clone cannot verify those four entries.
- `predictions_jarvis.csv` (SHA256 `a728389eef9fcaa0bf51d3d9d5c88b616e943e417679f94cae4202e099fa06af`) is the input of `scripts/na10_jarvis_analysis.py`. The `results.json` scoring (R2 against the p or the n value, a
  sign-matched R2) is the script's own and is superseded by the analysis script's output (the sign-agreement field there compares with the p value only, which is not a test; see that analysis).
