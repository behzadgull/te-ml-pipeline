"""
Nested GroupKFold hyperparameter tuning + repeated grouped CV.

Outer loop: repeated, randomized grouped k-fold over chemistry_cluster_id,
for reporting only. Inner loop: nested inside each outer training fold,
also grouped by chemistry_cluster_id (not random), used only to select
XGBoost hyperparameters via Optuna with MedianPruner. Hyperparameters
are never tuned and reported on the same fold (CLAUDE.md Paper A item 2).

Plain sklearn GroupKFold is fully deterministic given a set of group
labels: it internally sorts on np.unique(groups), which is alphabetical
and independent of row order (verified empirically -- see CLAUDE.md
Grouping Key section -- pre-shuffling row order before calling it
returns the identical fold assignment every time, across 4 seeds
tested). "Repeated grouped CV" therefore cannot be a loop calling
GroupKFold; randomized_group_kfold() below implements the deliberate
randomization instead: shuffle the unique group order, then run the
same greedy descending-group-size bin-packing GroupKFold itself uses.
Since the large majority of chemistry_cluster_id groups are singletons
(70%, all tied at size 1), shuffling their order before the stable sort
changes which ones land in which fold across repeats, while fold
row-counts stay balanced by the same bin-packing logic. This is
verified directly in verify_randomization() below, not assumed.
"""

import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

PROCESSED_DATA_DIR = Path("data/processed")
PROJECT = "ThermoelectricMaterials"

FEATURE_PREFIXES = ("MagpieData", "CBFV_")
GROUP_COL = "chemistry_cluster_id"

N_OUTER_REPEATS = 5  # lower end of CLAUDE.md's 5-10 range; see run_nested_cv's report for the local-compute tradeoff
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_OPTUNA_TRIALS = 20

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_feature_columns(df):
    """All MAGPIE + CBFV feature columns in df (see src/featurization.py)."""
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def load_target_data(target, processed_data_dir=PROCESSED_DATA_DIR, project=PROJECT):
    """
    Load the most recent featurized_<project>_*.csv and filter to rows
    with a non-null `target`, independently of the other three
    properties. Per-target filtering, not the all-four-present subset --
    that subset is reserved for the separate direct-vs-derived zT
    pathway comparison (CLAUDE.md Paper A item 5), not for this.
    """
    candidates = sorted(Path(processed_data_dir).glob(f"featurized_{project}_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No featurized_{project}_*.csv in {processed_data_dir}; run src/featurization.py first"
        )
    df = pd.read_csv(candidates[-1])
    return df[df[target].notna()].reset_index(drop=True)


def randomized_group_kfold(groups, n_splits, rng):
    """
    One randomized grouped k-fold split. Shuffles the unique group order
    with `rng`, then greedily bin-packs groups (sorted by descending
    size, ties broken by the now-randomized order) into the currently-
    smallest fold -- the same algorithm sklearn's GroupKFold uses
    internally, minus its alphabetical, order-independent tie-breaking.
    Yields (train_idx, test_idx) n_splits times.
    """
    groups = np.asarray(groups)
    unique_groups, counts = np.unique(groups, return_counts=True)

    perm = rng.permutation(len(unique_groups))
    unique_groups, counts = unique_groups[perm], counts[perm]

    order = np.argsort(-counts, kind="stable")
    unique_groups, counts = unique_groups[order], counts[order]

    fold_sizes = np.zeros(n_splits, dtype=int)
    group_to_fold = {}
    for group, count in zip(unique_groups, counts):
        fold = int(np.argmin(fold_sizes))
        group_to_fold[group] = fold
        fold_sizes[fold] += count

    fold_assignment = np.array([group_to_fold[g] for g in groups])
    for fold in range(n_splits):
        test_mask = fold_assignment == fold
        yield np.where(~test_mask)[0], np.where(test_mask)[0]


def verify_randomization(groups, n_splits=5, n_seeds=5, watch_groups=None):
    """
    Empirically confirm randomized_group_kfold produces different fold
    compositions across repeats and stays row-balanced -- the same way
    GroupKFold's determinism itself was caught, not assumed. Returns a
    dict report; raises AssertionError if repeats turn out identical
    (which would mean "repeated" grouped CV is not actually repeating).
    """
    groups = np.asarray(groups)
    if watch_groups is None:
        vals, counts = np.unique(groups, return_counts=True)
        watch_groups = vals[np.argsort(-counts)][:3].tolist()

    assignments = []
    fold_size_reports = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        group_to_fold = {}
        fold_sizes = np.zeros(n_splits, dtype=int)
        for fold, (_, test_idx) in enumerate(randomized_group_kfold(groups, n_splits, rng)):
            fold_sizes[fold] = len(test_idx)
            for g in set(groups[test_idx]):
                group_to_fold[g] = fold
        assignments.append(group_to_fold)
        fold_size_reports.append(fold_sizes)

    baseline = assignments[0]
    identical_to_baseline = [
        sum(1 for k in baseline if m.get(k) == baseline[k]) / len(baseline) for m in assignments[1:]
    ]
    watch_report = {g: [m.get(g) for m in assignments] for g in watch_groups}

    all_identical = all(frac == 1.0 for frac in identical_to_baseline)
    if all_identical:
        raise AssertionError(
            "randomized_group_kfold produced IDENTICAL fold assignments across all "
            f"{n_seeds} seeds -- randomization is not working, do not proceed."
        )

    return {
        "n_seeds": n_seeds,
        "fraction_identical_to_seed0": identical_to_baseline,
        "watch_group_fold_by_seed": watch_report,
        "fold_sizes_by_seed": [f.tolist() for f in fold_size_reports],
    }


def _xgb_objective(trial, X, y, groups, n_inner_folds):
    # Full intended search space. Measured per-fit cost on this CPU-only
    # machine (396 features, ~114k-126k row training folds): 57s/fit at
    # max_depth=10/n_estimators=600 vs 3s at depth=3/n_estimators=100 --
    # depth dominates cost. At this range, the full 5-10 repeat design
    # projects to ~24-61 hours locally (measured/projected 2026-08-15,
    # see commit history for the calibration numbers), so this module
    # is intended to run on Kaggle/GPU, not this laptop. A capped range
    # (depth<=7, n_estimators<=350) was
    # used ONLY for local calibration runs and is not applied here --
    # do not narrow this range without an explicit, documented decision.
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }

    inner_gkf = GroupKFold(n_splits=n_inner_folds)
    scores = []
    for step, (inner_train_idx, inner_val_idx) in enumerate(inner_gkf.split(X, groups=groups)):
        model = xgb.XGBRegressor(
            **params, n_jobs=-1, tree_method="hist", random_state=0, objective="reg:squarederror"
        )
        model.fit(X[inner_train_idx], y[inner_train_idx])
        preds = model.predict(X[inner_val_idx])
        scores.append(r2_score(y[inner_val_idx], preds))

        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def tune_hyperparameters(X_train, y_train, groups_train, n_trials=N_OPTUNA_TRIALS, n_inner_folds=N_INNER_FOLDS, seed=0):
    """
    Nested GroupKFold hyperparameter search via Optuna (TPE sampler,
    MedianPruner), grouped by chemistry_cluster_id -- not a random
    split -- entirely inside the outer training fold. Returns
    (best_params, best_inner_cv_r2).
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: _xgb_objective(trial, X_train, y_train, groups_train, n_inner_folds),
        n_trials=n_trials,
    )
    return study.best_params, study.best_value


def run_nested_cv(
    target="zT",
    n_repeats=N_OUTER_REPEATS,
    n_outer_folds=N_OUTER_FOLDS,
    n_inner_folds=N_INNER_FOLDS,
    n_trials=N_OPTUNA_TRIALS,
    seed=0,
):
    """
    Full repeated nested grouped CV for one target. Outer folds
    (repeated, randomized) are for reporting only; hyperparameters are
    tuned fresh inside each outer training fold via a nested, grouped
    (not random) Optuna search, then refit on the full outer-training
    fold and evaluated once on the outer-test fold.
    """
    df = load_target_data(target)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    groups = df[GROUP_COL].to_numpy()

    print(
        f"target={target}: {len(df):,} rows, {len(feature_cols)} features, "
        f"{len(np.unique(groups)):,} chemistry_cluster_id groups"
    )

    results = []
    rng_master = np.random.default_rng(seed)
    t_start = time.perf_counter()

    for repeat in range(n_repeats):
        repeat_rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        for fold, (train_idx, test_idx) in enumerate(randomized_group_kfold(groups, n_outer_folds, repeat_rng)):
            X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            best_params, inner_r2 = tune_hyperparameters(
                X_train, y_train, groups_train,
                n_trials=n_trials, n_inner_folds=n_inner_folds, seed=repeat * 100 + fold,
            )

            model = xgb.XGBRegressor(
                **best_params, n_jobs=-1, tree_method="hist", random_state=0, objective="reg:squarederror"
            )
            model.fit(X_train, y_train)
            outer_r2 = r2_score(y_test, model.predict(X_test))

            results.append(
                {
                    "repeat": repeat, "fold": fold,
                    "n_train": len(train_idx), "n_test": len(test_idx),
                    "outer_r2": outer_r2, "inner_cv_r2": inner_r2,
                    **{f"param_{k}": v for k, v in best_params.items()},
                }
            )
            print(
                f"repeat {repeat} fold {fold}: outer R^2={outer_r2:.4f} "
                f"(inner cv R^2={inner_r2:.4f}), n_train={len(train_idx):,}, n_test={len(test_idx):,}"
            )

    elapsed = time.perf_counter() - t_start
    results_df = pd.DataFrame(results)
    mean_r2 = results_df["outer_r2"].mean()
    std_r2 = results_df["outer_r2"].std()

    print(f"\n=== {target}: repeated grouped CV summary ===")
    print(f"{n_repeats} repeats x {n_outer_folds} outer folds = {len(results_df)} outer evaluations")
    print(f"mean outer R^2 = {mean_r2:.4f}, std = {std_r2:.4f}")
    print(f"total time: {elapsed:.1f}s")

    return results_df


if __name__ == "__main__":
    run_nested_cv(target="zT")
