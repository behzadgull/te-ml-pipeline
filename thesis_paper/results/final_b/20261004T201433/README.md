# Final models with the carrier-type classifier, JARVIS and Materials Project predictions (G3, GPU 0)

- **Bundle**: `final_b.tar.gz`, SHA256 `b1303f7e4c3bbd7eb50e7eb0b4f20ab69f8f8418e3ca3db47c7b2235235ca5b9` (verified, 2026-10-05); unpacked here; the 15 manifest entries re-verified, no mismatch and no file
  outside the manifest. `status.json`: complete, 6 of 6 units (four fits, JARVIS and MP predictions), `accepted_as_result` true.
- **Code** `33e9662838ba299a30bc8fccba0caee26d6bc435` (clean tree, `--expect-commit` equal), snapfix CSV `d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489`, Python 3.12.3, xgboost 2.0.3,
  Tesla T4, 0.035 h.
- **Inputs** (SHA256 in `run_configs/session_01.json`): the classifier `results/na6/20261004T102905/final_classifier.json` (`11b20af8...`), the 409-row MP candidate file (`6386c097...`), the JARVIS
  featurised CSV (`3c23d550...`); smear factors sigma 1.3817, kappa 1.0715 (frozen, `results/external_snapfix/20260917T160553`).
- **Models**: `models/<target>.json` (**not committed**, gitignored). Their SHA256 equal those of `results/final_a/20261004T102855` (S `f94d4052...`, sigma `b76f65aa...`, kappa `026b6e6b...`, zT
  `cca78dd9...`): the same four fitted models, so the only difference between the two runs is the classifier. A clone cannot verify the four manifest entries of the models.
- `predictions_mp.csv`: 409 candidates x 6 temperatures (300 to 800 K in 100 K steps), the regressor's S (`S_reg`), the classifier's sign (`S`, `sign_overridden`), sigma, kappa, direct zT (`zT_direct`) and
  the derived zT. `predictions_jarvis.csv`: the same for the 23,218 JARVIS entries. The `results.json` scoring of the JARVIS S (R2 against the p or n value, sign agreement with the p value) is the Kaggle
  script's own and is superseded by `scripts/na10_jarvis_analysis.py` (the sign-agreement field compares with the p value only).
- Pre-registrations A and B (`docs/decisions.md`, 2026-10-05) were committed before this bundle was unpacked.
