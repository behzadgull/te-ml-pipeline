# NA6: should the carrier-type classifier override the S regressor's sign? Same folds (primary)

Code: `run_config.json` (git head b8e1f0a, tree clean at the start of the run, which took about 75 minutes on the CPU of the maintainer's machine, xgboost 2.0.3; the first unit recorded the
identity in `checkpoints/na6_sign_samefolds/run_start.json`, checked on every resume). The classifier (XGBClassifier, the S regressor's frozen hyperparameters, 397 features, label S > 0) was
refitted on the S regressor's own folds, rebuilt from the snapfix CSV and checked against every saved regressor unit: each fold's classifier is trained on the regressor's training rows (minus
the S = 0 rows) and scored on the regressor's test rows (minus the S = 0 rows), so both rules are scored on exactly one partition. `units/` holds the 25 out-of-fold probability files.
It is a CPU refit, not the Kaggle GPU classifier, so its accuracy can differ slightly from `results/na6`.

| rows | regressor sign accuracy | classifier sign accuracy | paired difference, cluster-bootstrap interval |
|---|---|---|---|
| all (185,056) | 0.941 | 0.949 | +0.0083 [0.0043, 0.0121] |
| \|S\| < 20 uV/K (7,307) | 0.703 | 0.762 | +0.058 [0.033, 0.083] |
| \|S\| >= 20 uV/K (177,749) | 0.951 | 0.957 | +0.0062 [0.0023, 0.0099] |

Decision rule (fixed before the run): the classifier replaces the regressor's sign only if the lower bound of the interval for all rows is above zero. It is (0.0043), so the override is kept.
The gain is small (under one percentage point overall) and mostly in rows with a measured sign close to zero. Discordant rows summed over five repeats: classifier right and regressor wrong
18,009; regressor right and classifier wrong 10,368. The final-model predictions keep both signs (`S_reg` and `S`, with `sign_overridden`), so the regressor's sign remains available.
