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

device="cuda" (Kaggle GPU) converts X and y to cupy arrays once, up
front, in run_nested_cv (see _to_device) instead of leaving them as
numpy and letting XGBoost bridge the device mismatch on every single
fit/predict call. See _to_device's docstring for what that mismatch
costs. cupy is optional, not required: if it fails to import (e.g. a
numpy-2.x-only cupy build against this project's pinned numpy 1.26.4),
_to_device warns once and falls back to numpy arrays rather than
raising, so this module still runs -- just without the fix -- on a
Kaggle image where cupy and the pinned numpy don't line up.

--split-strategy selects the OUTER fold scheme, i.e. which rung of
CLAUDE.md's five-way validation-inflation ladder (Paper A item 1) this
run reports: "random" (repeated random 80/20 holdout, ungrouped),
"kfold" (ordinary shuffled K-fold, ungrouped -- pass --n-outer-folds 5
or 10 for those two specific rungs), "composition" (grouped by
composition_id, the looser intermediate rung), "chemistry" (grouped by
chemistry_cluster_id, the frozen strict anchor -- default). See
outer_splits() for the exact per-strategy definition. The INNER
hyperparameter-tuning split inside tune_hyperparameters always groups by
chemistry_cluster_id regardless of --split-strategy, per Paper A item
1's "tune once on the chemistry-cluster split" -- this module still
retunes fresh per outer fold rather than truly freezing one global
model across all five rungs, so it is a building block toward the
ladder, not the full denominator-matched, single-frozen-model
comparison; that pooling/matching step belongs in
src/validation_ladder.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, ShuffleSplit

PROCESSED_DATA_DIR = Path("data/processed")
PROJECT = "ThermoelectricMaterials"

FEATURE_PREFIXES = ("MagpieData", "CBFV_")
GROUP_COL = "chemistry_cluster_id"

SPLIT_STRATEGIES = ("random", "kfold", "composition", "chemistry")
SPLIT_STRATEGY_GROUP_COL = {"composition": "composition_id", "chemistry": GROUP_COL}
RANDOM_HOLDOUT_TEST_SIZE = 0.2  # CLAUDE.md Paper A item 1: "random 80/20"

N_OUTER_REPEATS = 5  # lower end of CLAUDE.md's 5-10 range; see run_nested_cv's report for the local-compute tradeoff
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_OPTUNA_TRIALS = 20

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_feature_columns(df):
    """All MAGPIE + CBFV feature columns in df (see src/featurization.py)."""
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


_CUPY_UNAVAILABLE_WARNED = False


def _to_device(arr, device):
    """
    Move arr onto the GPU as a cupy array when device=='cuda'. Passing a
    host (numpy) array to an XGBRegressor with device='cuda' still runs,
    but XGBoost detects the booster device and the input array's device
    don't match and transparently rebuilds a DMatrix on the GPU on every
    single fit/predict call to bridge the gap (warns "Falling back to
    prediction using DMatrix due to mismatched devices" and re-copies the
    full feature matrix across the PCIe bus each time). Converting once,
    up front, keeps every downstream slice -- X[train_idx], X[test_idx],
    and the repeated per-outer-fold, per-inner-fold, per-Optuna-trial
    slices inside tune_hyperparameters -- resident on the GPU with no
    further host<->device transfers. No-op on cpu.

    If cupy itself fails to import (e.g. a cupy build requiring numpy 2.x
    against this project's pinned numpy 1.26.4 -- seen on Kaggle), this
    is optional, not required: warn once and fall back to returning arr
    unchanged. XGBoost then reproduces the original cross-device
    DMatrix-fallback behavior (correct, just slower) instead of the run
    failing outright.
    """
    global _CUPY_UNAVAILABLE_WARNED
    if device != "cuda":
        return arr
    try:
        import cupy as cp
    except Exception as e:
        if not _CUPY_UNAVAILABLE_WARNED:
            print(
                f"WARNING: device='cuda' requested but cupy is unavailable "
                f"({type(e).__name__}: {e}); falling back to numpy arrays. XGBoost will still "
                f"run on device='cuda' but will rebuild a GPU-side DMatrix on every fit/predict "
                f"call to bridge the host/device gap (correct, just slower -- see _to_device's "
                f"docstring). Install a cupy build compatible with the pinned numpy version to "
                f"restore the fix.",
                flush=True,
            )
            _CUPY_UNAVAILABLE_WARNED = True
        return arr
    return cp.asarray(arr)


def _to_host(arr):
    """Bring a possibly-GPU (cupy) array back to numpy, e.g. for sklearn metrics. No-op for numpy input."""
    return arr.get() if hasattr(arr, "get") else arr


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


def outer_splits(split_strategy, n_rows, group_lookup, n_outer_folds, rng):
    """
    Yield (train_idx, test_idx) for one repeat's worth of outer folds
    under `split_strategy` -- see the module docstring for what each of
    the four CLAUDE.md ladder rungs this implements means:

    - "chemistry" / "composition": randomized_group_kfold over
      group_lookup[split_strategy] (chemistry_cluster_id /
      composition_id respectively) -- n_outer_folds partitions covering
      every row exactly once, each group confined to one fold.
    - "kfold": ordinary shuffled sklearn KFold over all rows, no
      grouping -- the same chemistry cluster or composition can appear
      in both train and test.
    - "random": n_outer_folds independent random 80/20 holdout draws
      (sklearn ShuffleSplit), no grouping and NOT a partition -- test
      sets can overlap across draws within a repeat. This is
      deliberate: CLAUDE.md's ladder repeats the 80/20 split ~20 times
      and pools results, since a single 80/20 split only covers 20% of
      rows (unlike the other, full-coverage rungs).

    group_lookup: {"composition": composition_id array, "chemistry":
    chemistry_cluster_id array}; only the entry matching split_strategy
    is read.
    """
    if split_strategy in ("composition", "chemistry"):
        yield from randomized_group_kfold(group_lookup[split_strategy], n_outer_folds, rng)
        return

    seed = int(rng.integers(0, 2**32 - 1))
    placeholder = np.empty(n_rows)
    if split_strategy == "kfold":
        splitter = KFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    elif split_strategy == "random":
        splitter = ShuffleSplit(n_splits=n_outer_folds, test_size=RANDOM_HOLDOUT_TEST_SIZE, random_state=seed)
    else:
        raise ValueError(f"Unknown split_strategy {split_strategy!r}; must be one of {SPLIT_STRATEGIES}")
    yield from splitter.split(placeholder)


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


def _xgb_objective(trial, X, y, groups, n_inner_folds, device="cpu"):
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
    # split() only needs sample count and `groups` (always a host numpy
    # array -- see run_nested_cv); pass a host placeholder in X's place so
    # sklearn's split-index bookkeeping never touches a GPU-resident X.
    split_placeholder = np.empty(len(groups))
    for step, (inner_train_idx, inner_val_idx) in enumerate(inner_gkf.split(split_placeholder, groups=groups)):
        model = xgb.XGBRegressor(
            **params, n_jobs=-1, tree_method="hist", device=device, random_state=0, objective="reg:squarederror"
        )
        model.fit(X[inner_train_idx], y[inner_train_idx])
        preds = model.predict(X[inner_val_idx])
        scores.append(r2_score(_to_host(y[inner_val_idx]), _to_host(preds)))

        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def tune_hyperparameters(
    X_train, y_train, groups_train, n_trials=N_OPTUNA_TRIALS, n_inner_folds=N_INNER_FOLDS, seed=0, device="cpu"
):
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
        lambda trial: _xgb_objective(trial, X_train, y_train, groups_train, n_inner_folds, device=device),
        n_trials=n_trials,
    )
    return study.best_params, study.best_value


def _fold_checkpoint_path(checkpoint_dir, repeat, fold):
    return checkpoint_dir / f"repeat{repeat}_fold{fold}.json"


def _log_progress(checkpoint_dir, message):
    """
    Append a timestamped line to checkpoint_dir/progress.log, if
    checkpointing is enabled. Kaggle's committed/batch execution mode
    buffers stdout until the process exits, and its log viewer can hide
    that buffered output entirely -- this file is readable at any time
    while a run is still in progress, independent of stdout.
    """
    if checkpoint_dir is None:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(checkpoint_dir / "progress.log", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def _load_checkpoints(checkpoint_dir):
    """Load every repeatN_foldN.json in checkpoint_dir. Returns list of result dicts."""
    records = []
    for path in sorted(checkpoint_dir.glob("repeat*_fold*.json")):
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def _write_run_config(checkpoint_dir, config):
    path = checkpoint_dir / "run_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _check_run_config(checkpoint_dir, config):
    """
    On resume, verify this call's parameters match the run that produced
    the existing checkpoints for every field that affects fold
    composition or data identity (target, split_strategy, seed,
    n_outer_folds). n_repeats
    is allowed to increase (extending an existing run is safe since the
    outer RNG stream is drawn sequentially, one draw per repeat index,
    so repeats 0..old_n_repeats-1 reproduce identically regardless of
    how many more repeats a later call asks for). n_trials/n_inner_folds/
    device only affect hyperparameter search cost and where it runs, not
    fold composition, so mismatches are warned about, not fatal.
    """
    path = checkpoint_dir / "run_config.json"
    if not path.exists():
        _write_run_config(checkpoint_dir, config)
        return

    with open(path, encoding="utf-8") as f:
        existing = json.load(f)

    fatal_keys = ("target", "split_strategy", "seed", "n_outer_folds")
    mismatches = {k: (existing[k], config[k]) for k in fatal_keys if existing.get(k) != config[k]}
    if mismatches:
        raise ValueError(
            f"Resume parameter mismatch in {checkpoint_dir}: {mismatches}. "
            f"target/split_strategy/seed/n_outer_folds must match the run that produced the existing "
            f"checkpoints, since changing any of them changes fold composition for already-checkpointed "
            f"repeats. Use a different checkpoint_dir for a genuinely new run."
        )
    if existing.get("n_repeats", 0) > config["n_repeats"]:
        raise ValueError(
            f"Resume requested n_repeats={config['n_repeats']} but checkpoints already exist for "
            f"n_repeats={existing['n_repeats']} in {checkpoint_dir}. Shrinking n_repeats on resume "
            f"would silently discard completed results; pass n_repeats >= {existing['n_repeats']}."
        )
    for k in ("n_inner_folds", "n_trials", "device"):
        if existing.get(k) != config[k]:
            print(
                f"WARNING: resuming with {k}={config[k]}, but existing checkpoints in "
                f"{checkpoint_dir} were produced with {k}={existing.get(k)}. Fold composition is "
                f"unaffected, but hyperparameter search cost/results may be inconsistent across "
                f"already-completed vs. newly-run folds.",
                flush=True,
            )
    _write_run_config(checkpoint_dir, config)


def run_nested_cv(
    target="zT",
    split_strategy="chemistry",
    n_repeats=N_OUTER_REPEATS,
    n_outer_folds=N_OUTER_FOLDS,
    n_inner_folds=N_INNER_FOLDS,
    n_trials=N_OPTUNA_TRIALS,
    seed=0,
    device="cpu",
    checkpoint_dir=None,
):
    """
    Full repeated nested CV for one target, outer folds built under
    `split_strategy` (one of SPLIT_STRATEGIES; see the module docstring
    and outer_splits() for what each rung is). Outer folds (repeated,
    randomized) are for reporting only; hyperparameters are tuned fresh
    inside each outer training fold via a nested Optuna search grouped
    by chemistry_cluster_id (always, regardless of split_strategy -- see
    module docstring), then refit on the full outer-training fold and
    evaluated once on the outer-test fold.

    device: passed straight to xgb.XGBRegressor (with tree_method="hist"
    fixed) -- "cpu" (default) or "cuda" for GPU-accelerated fits, e.g.
    on Kaggle.

    checkpoint_dir: directory to write one JSON file per completed
    (repeat, fold) -- {"repeat", "fold", "split_strategy", "n_train",
    "n_test", "outer_r2", "inner_cv_r2", "param_*"} -- immediately after
    that fold finishes, plus a run_config.json recording the parameters
    that must match on resume. Defaults to
    checkpoints/nested_cv/<target>/<split_strategy>/. On startup, any
    (repeat, fold) with an existing checkpoint file is skipped and its
    saved result reused instead of recomputed -- this is how a run
    resumes after a Kaggle session is killed mid-run. Resuming REQUIRES
    the same target/split_strategy/seed/n_outer_folds as the original
    call (n_repeats may be increased to extend the run); a mismatch
    raises rather than silently producing an inconsistent result set.
    Pass checkpoint_dir=False to disable checkpointing entirely (nothing
    written, nothing skipped).
    """
    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"split_strategy={split_strategy!r} must be one of {SPLIT_STRATEGIES}")

    if checkpoint_dir is False:
        checkpoint_dir = None
    elif checkpoint_dir is None:
        checkpoint_dir = Path("checkpoints") / "nested_cv" / target / split_strategy
    else:
        checkpoint_dir = Path(checkpoint_dir)

    completed = {}
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _check_run_config(
            checkpoint_dir,
            {
                "target": target, "split_strategy": split_strategy, "seed": seed, "n_outer_folds": n_outer_folds,
                "n_repeats": n_repeats, "n_inner_folds": n_inner_folds,
                "n_trials": n_trials, "device": device,
            },
        )
        for record in _load_checkpoints(checkpoint_dir):
            completed[(record["repeat"], record["fold"])] = record
        if completed:
            msg = f"Resuming: {len(completed)} outer fold(s) already checkpointed in {checkpoint_dir}"
            print(msg, flush=True)
            _log_progress(checkpoint_dir, msg)

    df = load_target_data(target)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)

    # chemistry_groups is used for two independent purposes: (a) the
    # inner hyperparameter-tuning split, always, regardless of
    # split_strategy; (b) the outer split itself, only when
    # split_strategy == "chemistry". Both stay on host: grouping/index
    # bookkeeping only, never fed to XGBoost.
    chemistry_groups = df[GROUP_COL].to_numpy()
    group_lookup = {"chemistry": chemistry_groups, "composition": df["composition_id"].to_numpy()}

    # Convert once, up front, not per-fold -- see _to_device docstring for
    # why this is the fix for the cross-device DMatrix-fallback warning.
    X = _to_device(X, device)
    y = _to_device(y, device)

    if split_strategy in ("composition", "chemistry"):
        group_col = SPLIT_STRATEGY_GROUP_COL[split_strategy]
        n_groups = len(np.unique(group_lookup[split_strategy]))
        split_desc = f"split_strategy={split_strategy} ({n_groups:,} {group_col} groups)"
    else:
        split_desc = f"split_strategy={split_strategy} (ungrouped)"
    start_msg = (
        f"target={target}: {len(df):,} rows, {len(feature_cols)} features, "
        f"{split_desc}, device={device}"
    )
    print(start_msg, flush=True)
    _log_progress(checkpoint_dir, start_msg)

    results = list(completed.values())
    rng_master = np.random.default_rng(seed)
    t_start = time.perf_counter()

    for repeat in range(n_repeats):
        repeat_rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        fold_iter = outer_splits(split_strategy, len(df), group_lookup, n_outer_folds, repeat_rng)
        for fold, (train_idx, test_idx) in enumerate(fold_iter):
            if (repeat, fold) in completed:
                msg = f"repeat {repeat} fold {fold}: skipping, already checkpointed"
                print(msg, flush=True)
                _log_progress(checkpoint_dir, msg)
                continue

            X_train, y_train, groups_train = X[train_idx], y[train_idx], chemistry_groups[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            best_params, inner_r2 = tune_hyperparameters(
                X_train, y_train, groups_train,
                n_trials=n_trials, n_inner_folds=n_inner_folds, seed=repeat * 100 + fold, device=device,
            )

            model = xgb.XGBRegressor(
                **best_params, n_jobs=-1, tree_method="hist", device=device,
                random_state=0, objective="reg:squarederror",
            )
            model.fit(X_train, y_train)
            outer_r2 = r2_score(_to_host(y_test), _to_host(model.predict(X_test)))

            record = {
                "repeat": repeat, "fold": fold, "split_strategy": split_strategy,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "outer_r2": outer_r2, "inner_cv_r2": inner_r2,
                **{f"param_{k}": v for k, v in best_params.items()},
            }

            if checkpoint_dir is not None:
                with open(_fold_checkpoint_path(checkpoint_dir, repeat, fold), "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)

            results.append(record)
            fold_msg = (
                f"repeat {repeat} fold {fold}: outer R^2={outer_r2:.4f} "
                f"(inner cv R^2={inner_r2:.4f}), n_train={len(train_idx):,}, n_test={len(test_idx):,}"
            )
            print(fold_msg, flush=True)
            _log_progress(checkpoint_dir, fold_msg)

    elapsed = time.perf_counter() - t_start
    results_df = pd.DataFrame(results).sort_values(["repeat", "fold"]).reset_index(drop=True)
    mean_r2 = results_df["outer_r2"].mean()
    std_r2 = results_df["outer_r2"].std()

    summary_lines = [
        f"=== {target}: repeated grouped CV summary ===",
        f"{n_repeats} repeats x {n_outer_folds} outer folds = {len(results_df)} outer evaluations",
        f"mean outer R^2 = {mean_r2:.4f}, std = {std_r2:.4f}",
        f"this call's new compute time: {elapsed:.1f}s",
    ]
    print("\n" + "\n".join(summary_lines), flush=True)
    for line in summary_lines:
        _log_progress(checkpoint_dir, line)

    return results_df


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Repeated nested grouped CV for one thermoelectric property target.",
    )
    parser.add_argument("--target", default="zT", help="Target property column (default: zT)")
    parser.add_argument(
        "--split-strategy", default="chemistry", choices=SPLIT_STRATEGIES,
        help="Outer-fold scheme, one rung of CLAUDE.md's five-way validation ladder (default: chemistry). "
        "Use --split-strategy kfold with --n-outer-folds 5 or 10 for those two rungs.",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help='XGBoost device, passed with tree_method="hist" (default: cpu; use cuda on Kaggle/GPU)',
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory for per-fold checkpoint JSON files "
        "(default: checkpoints/nested_cv/<target>/<split_strategy>/)",
    )
    parser.add_argument("--n-repeats", type=int, default=N_OUTER_REPEATS, help=f"Outer repeats (default: {N_OUTER_REPEATS})")
    parser.add_argument("--n-outer-folds", type=int, default=N_OUTER_FOLDS, help=f"Outer folds per repeat (default: {N_OUTER_FOLDS})")
    parser.add_argument("--n-inner-folds", type=int, default=N_INNER_FOLDS, help=f"Inner tuning folds (default: {N_INNER_FOLDS})")
    parser.add_argument("--n-trials", type=int, default=N_OPTUNA_TRIALS, help=f"Optuna trials per outer fold (default: {N_OPTUNA_TRIALS})")
    parser.add_argument("--seed", type=int, default=0, help="Master RNG seed (default: 0)")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    results_df = run_nested_cv(
        target=args.target,
        split_strategy=args.split_strategy,
        n_repeats=args.n_repeats,
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
        n_trials=args.n_trials,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(results_df, flush=True)
    return results_df


if __name__ == "__main__":
    main()
