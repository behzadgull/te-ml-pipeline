"""
Five-way, denominator-matched validation-inflation ladder.

Evaluates a single frozen model (hyperparameters tuned once via nested
CV on the chemistry-cluster split) under five evaluation schemes: random
80/20 (repeated ~20x and pooled), 5-fold, 10-fold, composition-level
grouped, and chemistry-cluster grouped. Reports pooled out-of-fold R^2
for every scheme so the schemes are comparable on the same denominator.
"""
