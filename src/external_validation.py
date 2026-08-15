"""
External validation against ESTM and teMatDb.

Computes two separate numbers against the frozen model: source-
deduplicated (no shared DOI with training, tests measurement transfer)
and composition-cluster-deduplicated (no shared chemistry cluster with
training, tests chemistry transfer). Reports dropped-row counts for
each. Each external dataset is touched exactly once, after the model
is fully frozen.
"""
