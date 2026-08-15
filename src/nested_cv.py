"""
Nested GroupKFold hyperparameter tuning.

Hyperparameters are tuned only on inner folds nested inside each outer
training fold; outer folds are reserved for reporting only, so tuning
and reporting never share folds. Also provides repeated grouped CV and
a Nadeau-Bengio corrected significance test for comparing ladder rungs
under the small chemistry-cluster count (~15-25 clusters).
"""
