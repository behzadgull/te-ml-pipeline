# NA6: classifier sign against the S regressor's sign, same rows, different partitions (secondary)

Code: the commit in `run_config.json` (clean tree). Compares the Kaggle classifier (`results/na6/20261004T102905`, 185,056 rows) with the sign of the S regressor's out-of-fold
predictions (`results/ladder_regen_snapfix/20260917T150000/S_chemistry_full`, 185,064 rows) on the 185,056 rows with S != 0.

The two runs do NOT share a fold partition: the classifier dropped the 8 rows with S = 0, which changes the cluster sizes and so the fold assignment (only 55 to 67 percent of the rows
carry the same fold label, `sign_comparison.json`, `fold_agreement_regressor_vs_classifier`). Each rule is out of fold under its own cluster-grouped 5 x 5 partition, so this is a paired
comparison of rows, not of folds. The same-folds comparison is `../20261004T194811_same_folds` (primary).

Result: sign accuracy of the classifier 0.948 against 0.941 for the regressor; paired difference +0.0074, cluster-bootstrap interval [0.0037, 0.0110].
