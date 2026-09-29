"""
Metrics for the Paper B modelling harness (methodology doc, section 8.2).

Every function here operates on already-scored arrays (the target's Paper A
scoring space: log10 for sigma and kappa, linear for S and zT -- see
`transform_target`); nothing here fits a model. Confidence intervals are a
cluster bootstrap over chemistry clusters, computed after the fact from
saved predictions, never during fitting.

No shared imports: this module only uses numpy/pandas, so it has no
dependency on paper_b/SHARED_DEPENDENCIES.md's pinned modules.
"""

import numpy as np

LOG_TRANSFORM_TARGETS = ("sigma", "kappa")  # mirrors src/nested_cv.py's constant of the same name


def transform_target(y, target):
    """log10 for sigma/kappa, unchanged for S/zT -- the Paper A scoring-space convention (section 8's own rule)."""
    return np.log10(y) if target in LOG_TRANSFORM_TARGETS else np.asarray(y, dtype=float)


def r2_score(y_true, y_pred):
    """R^2 = 1 - SS_res / SS_tot. NaN if y_true has zero variance (SS_tot == 0)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def mse(y_true, y_pred):
    """Mean squared error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def skill(y_true, y_pred, baseline_value):
    """
    skill_train = 1 - MSE(model) / MSE(constant baseline). `baseline_value`
    is the mean of the TRAINING targets for the condition being scored
    (section 8.2: "baseline = mean of that condition's training targets"),
    passed in rather than computed here, since the harness knows which rows
    were the training set and this module only sees the test predictions.
    """
    baseline_mse = mse(y_true, np.full_like(np.asarray(y_true, dtype=float), baseline_value))
    if baseline_mse == 0:
        return float("nan")
    return float(1.0 - mse(y_true, y_pred) / baseline_mse)


def within_family_r(y_true, y_pred):
    """Pearson correlation between prediction and target. NaN if either side has zero variance."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def murphy_decomposition(y_true, y_pred):
    """
    Murphy (1988) MSE decomposition: MSE = offset^2 + (s_pred - r*s_true)^2 +
    (1 - r^2) * s_true^2 -- an offset (bias in the mean), a scale term
    (miscalibrated spread) and an unexplained-variance term. Returns a dict
    with the three raw terms, their shares of the MSE (summing to 1 up to
    floating-point error, asserted by the smoke test to 1e-9), and the MSE
    itself, so the decomposition can be checked against `mse()` independently.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    s_true, s_pred = y_true.std(), y_pred.std()
    r = within_family_r(y_true, y_pred)
    r = 0.0 if np.isnan(r) else r
    offset_term = (mean_pred - mean_true) ** 2
    scale_term = (s_pred - r * s_true) ** 2
    unexplained_term = (1 - r ** 2) * s_true ** 2
    total = offset_term + scale_term + unexplained_term
    total_mse = mse(y_true, y_pred)
    shares = {"offset_share": offset_term / total, "scale_share": scale_term / total,
             "unexplained_share": unexplained_term / total} if total > 0 else {
        "offset_share": float("nan"), "scale_share": float("nan"), "unexplained_share": float("nan")}
    return {"offset_term": float(offset_term), "scale_term": float(scale_term), "unexplained_term": float(unexplained_term),
            "decomposition_mse": float(total), "mse_check": float(total_mse), **shares, "r": float(r)}


def pool_metrics(y_true, y_pred, train_mean_baseline, oracle_baseline=None):
    """
    Every section-8.2 point metric for one pooled (y_true, y_pred) array:
    R^2, skill_train (baseline = `train_mean_baseline`), the Murphy shares,
    within-family r, and, if `oracle_baseline` is given (F's own mean,
    computed from these same y_true), the oracle-baseline R^2-equivalent
    skill for comparison.
    """
    out = {"r2": r2_score(y_true, y_pred), "skill_train": skill(y_true, y_pred, train_mean_baseline),
           "within_family_r": within_family_r(y_true, y_pred), "mse": mse(y_true, y_pred),
           "n": int(len(y_true)), **murphy_decomposition(y_true, y_pred)}
    if oracle_baseline is not None:
        out["skill_oracle"] = skill(y_true, y_pred, oracle_baseline)
    return out


def cluster_bootstrap_ci(y_true, y_pred, cluster_ids, metric_fn, n_resamples=1000, seed=0, alpha=0.05):
    """
    95% (or 1-alpha) percentile interval for `metric_fn(y_true, y_pred)` from
    a cluster bootstrap: each resample draws len(unique clusters) clusters
    WITH replacement (a cluster drawn twice contributes its rows twice) and
    recomputes the metric on the resulting pooled rows. Returns (lo, hi,
    resampled values array).
    """
    y_true, y_pred, cluster_ids = np.asarray(y_true), np.asarray(y_pred), np.asarray(cluster_ids)
    unique = np.unique(cluster_ids)
    index_of = {cluster: np.where(cluster_ids == cluster)[0] for cluster in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_of[c] for c in drawn])
        values[i] = metric_fn(y_true[rows], y_pred[rows])
    lo, hi = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), values


def paired_cluster_bootstrap_ci(y_true, y_pred_a, y_pred_b, cluster_ids, metric_fn, n_resamples=1000, seed=0, alpha=0.05):
    """
    95% percentile interval for metric_fn(y_true, y_pred_a) - metric_fn(y_true,
    y_pred_b), using the SAME cluster resample for both conditions each draw
    (paired resamples, section 8.2), for two predictions scored on the same
    rows/clusters (e.g. C0 vs C1, or specialist vs C0).
    """
    y_true, y_pred_a, y_pred_b = np.asarray(y_true), np.asarray(y_pred_a), np.asarray(y_pred_b)
    cluster_ids = np.asarray(cluster_ids)
    unique = np.unique(cluster_ids)
    index_of = {cluster: np.where(cluster_ids == cluster)[0] for cluster in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_of[c] for c in drawn])
        values[i] = metric_fn(y_true[rows], y_pred_a[rows]) - metric_fn(y_true[rows], y_pred_b[rows])
    lo, hi = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), values
