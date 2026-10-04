# NA10: final-model Seebeck predictions against JARVIS dft_3d (cross-domain description)

Code: the commit in `run_config.json` (clean tree). Input: `results/final_a/20261004T102855/predictions_jarvis.csv` (SHA256 in `run_config.json`; the regressor's S, no
classifier yet) and the NA10 preparation `results/na10/20261003T133155`. 23,218 JARVIS entries (of 93,902 in dft_3d, those with n- and p-doped Seebeck values at 600 K).

**How the signs were chosen.** Predicted sign = sign of the predicted S (regressor). JARVIS supplies a value for n-type and one for p-type doping; for a semiconductor they
have opposite signs, set by the imposed doping, so JARVIS has no carrier type to compare with. A JARVIS sign is defined only for the 13,123 entries whose two values have the
same sign (metals and semimetals); of those, 3,885 have a reference magnitude of at least 20 uV/K, and sign agreement is computed on these. 10,089 entries have opposite
signs (excluded from the sign analysis), 6 have a zero value. The magnitude reference is mean(|n|, |p|). The earlier `sign_agreement_with_p_value` in the Kaggle run's
results compared the predicted sign with the p value, which is positive for almost every semiconductor, so it measured the share of predicted p-type entries, not agreement.

**Results** (cluster-bootstrap 95% intervals, 1,000 draws): rank correlation of |predicted S| with the JARVIS magnitude 0.23 [0.21, 0.25] for all entries; 0.38 [0.25, 0.48] for
entries whose chemistry cluster is in the training data (n = 1,150) against 0.22 [0.20, 0.24] for the 22,068 unseen; 0.24 [0.17, 0.31] for ABX3 stoichiometry. Sign agreement on
the same-sign subset: 0.504 [0.487, 0.521], i.e. chance (the model predicts p-type for 63% of them, JARVIS has 53% positive); 0.61 [0.53, 0.69] for seen clusters (n = 188),
0.50 for unseen (n = 3,697). Signed Spearman 0.01 and R2 below zero on the same-sign subset. These values are computed, not measured, at one doping and one temperature, for
ideal crystals; the comparison is a cross-domain description, not a validation, and shows no transfer of the sign or the magnitude beyond compositions the model has seen.
