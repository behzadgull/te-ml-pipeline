"""
Nested GroupKFold hyperparameter tuning + repeated grouped CV.

Outer loop: repeated, randomized grouped k-fold over chemistry_cluster_id,
for reporting only. Inner loop: nested inside each outer training fold,
also grouped by chemistry_cluster_id (not random), used only to select
hyperparameters via Optuna with MedianPruner. Hyperparameters are never
tuned and reported on the same fold (CLAUDE.md Paper A item 2).

--model selects the model family (MODEL_TYPES: "xgboost" default,
"lightgbm", "random_forest", "ridge"), all driven through the exact same
nested-CV machinery, chemistry-cluster grouping, split_strategy rungs,
frozen-hyperparameter mode, and pooled-OOF-R^2 reporting below -- only
the search space (see MODEL_REGISTRY) and the model constructor differ,
so results are directly comparable across model families (a Figure 1
model-comparison table/plot, and CLAUDE.md Paper B item 4's "two model
families... bracket capacity" requirement, both read off the same
results_df schema regardless of --model). ridge is deliberately wrapped
in a StandardScaler pipeline -- MAGPIE/CBFV/temperature features span
wildly different scales, and an unscaled Ridge fit would be measuring
that, not real model capacity. Only xgboost has a GPU path in this
module (see _to_device); --device cuda with any other --model prints a
one-time warning and runs that model on CPU regardless.

sigma and kappa are trained and evaluated on log10-transformed targets
(LOG_TRANSFORM_TARGETS), decided 2026-08-20: both span multiple orders
of magnitude (sigma ~10^3-10^6+ S/m, kappa ~0.05-25 W/mK), so raw-scale
squared-error loss is dominated by the largest-magnitude samples and
effectively ignores relative error on low-conductivity/low-kappa
materials; log10 converts this to a relative-error objective. S and zT
stay on linear/raw scale (S can be negative; neither spans orders of
magnitude the same way). This also makes the noise-floor comparison
(CLAUDE.md Paper A item 3, which requires log-space R^2) internally
consistent rather than needing a post-hoc conversion. Every R^2 this
module reports for sigma/kappa is therefore in log10 space, not linear
scale -- results_df.attrs["target_scale"] and the "target_scale"
checkpoint/record column make this explicit rather than ambiguous.

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
outer_splits() for the exact per-strategy definition.

Two-step workflow for the actual five-way ladder (CLAUDE.md Paper A item
1: "tune hyperparameters ONCE on the chemistry-cluster split via nested
CV, freeze. Evaluate the identical frozen model under all five schemes
... using pooled out-of-fold R^2"):

  1. tune_once(target=..., model_type=...) -- ONE chemistry-cluster-grouped
     hyperparameter search on the full dataset, saved to disk (default
     checkpoints/frozen_hyperparams/<target>_<model_type>.json -- one
     frozen file per model, hyperparameters aren't comparable across
     model families). This constitutes a complete nested-CV tuning pass
     on its own: tune_hyperparameters()'s Optuna search already
     validates every trial via an inner chemistry-cluster-grouped
     GroupKFold.
  2. run_nested_cv(..., frozen_hyperparams_path=<that file>) -- once per
     rung (five calls, varying only split_strategy/n_outer_folds). Every
     outer fold reuses the frozen hyperparameters unchanged instead of
     retuning via tune_hyperparameters() -- split_strategy is then the
     ONLY varying factor across the five runs, as the ladder requires.
     Without frozen_hyperparams_path, run_nested_cv still retunes fresh
     per outer fold (the original, pre-ladder behavior) -- useful on its
     own, but not the frozen-model ladder comparison.

Every run_nested_cv() call, regardless of frozen/retuned mode, computes
and reports POOLED out-of-fold R^2 -- all held-out (y_true, y_pred) pairs
across every outer fold and repeat, concatenated once and scored with a
single r2_score() call -- as the PRIMARY ladder metric (results_df.attrs
["pooled_r2"]), alongside the mean/std of per-fold R^2 as a secondary
diagnostic. Per-fold predictions are saved to
<checkpoint_dir>/repeatN_foldN_predictions.npz so pooling stays correct
across a resumed run, not just folds computed in the current process.

n_repeats defaults to None, resolved per split_strategy rather than one
fixed number for all five rungs: N_OUTER_REPEATS_GROUPED (5) for
composition/chemistry, where CLAUDE.md's group-composition-effect
justification applies (see the Grouping Key section's "Repeated grouped
CV is still required" note); N_OUTER_REPEATS_UNGROUPED (1) for
random/kfold, which have no group-composition confound to average away
-- see the Grouping Key section for the full reasoning. Pass --n-repeats
explicitly to override either default.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROCESSED_DATA_DIR = Path("data/processed")
PROJECT = "ThermoelectricMaterials"
FROZEN_HYPERPARAMS_DIR = Path("checkpoints") / "frozen_hyperparams"

FEATURE_PREFIXES = ("MagpieData", "CBFV_")
TEMPERATURE_COL = "temperature_bin"
GROUP_COL = "chemistry_cluster_id"

SPLIT_STRATEGIES = ("random", "kfold", "composition", "chemistry")
SPLIT_STRATEGY_GROUP_COL = {"composition": "composition_id", "chemistry": GROUP_COL}
GROUPED_SPLIT_STRATEGIES = tuple(SPLIT_STRATEGY_GROUP_COL.keys())  # ("composition", "chemistry")
RANDOM_HOLDOUT_TEST_SIZE = 0.2  # CLAUDE.md Paper A item 1: "random 80/20"

MODEL_TYPES = ("xgboost", "lightgbm", "random_forest", "ridge")

# sigma spans ~10^3-10^6+ S/m, kappa ~0.05-25 W/mK -- both multiple
# orders of magnitude. Raw-scale squared-error loss is then dominated by
# the largest-magnitude samples and effectively ignores relative error
# on low-conductivity/low-kappa materials; log10 converts this to a
# relative-error objective. S (can be negative, -1000 to 1000 uV/K) and
# zT (0-4, not multiple orders of magnitude) stay on raw scale. Decided
# 2026-08-20, see CLAUDE.md Paper A item 3 -- this also makes the
# noise-floor comparison (which requires log-space R^2) internally
# consistent instead of needing a post-hoc conversion. Positivity for
# log10 is guaranteed by data_cleaning.py's step 1 bounds (sigma >= 10,
# kappa >= 0.05), not re-checked here.
LOG_TRANSFORM_TARGETS = ("sigma", "kappa")


def _target_scale(target):
    """"log10" for LOG_TRANSFORM_TARGETS, else "linear"."""
    return "log10" if target in LOG_TRANSFORM_TARGETS else "linear"


def _transform_target(y, target):
    """Apply the frozen per-target scale decision (see LOG_TRANSFORM_TARGETS) to a raw y array."""
    return np.log10(y) if target in LOG_TRANSFORM_TARGETS else y

# Deliberate per-strategy defaults, not one number for all five ladder
# rungs -- see the module docstring and CLAUDE.md's Grouping Key section
# for why grouped and ungrouped strategies need different repeat counts.
N_OUTER_REPEATS_GROUPED = 5  # lower end of CLAUDE.md's 5-10 range for composition/chemistry
N_OUTER_REPEATS_UNGROUPED = 1  # random/kfold: no group-composition confound to average away
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_OPTUNA_TRIALS = 20

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_feature_columns(df):
    """
    All MAGPIE + CBFV feature columns in df (see src/featurization.py),
    plus TEMPERATURE_COL. Temperature is a per-row model input like any
    other -- each row is already one formula at one temperature_bin, so
    withholding it discards real signal. CLAUDE.md's frozen "temperature
    axis cut" (Paper A item 8) only rules out building a dedicated
    temperature-extrapolation experiment; it was previously misread as
    excluding temperature from the feature set entirely, which this
    function did until 2026-08-18 (found by comparing against the
    thesis's own feature list, which does include T_K).
    """
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)] + [TEMPERATURE_COL]


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
    if split_strategy in GROUPED_SPLIT_STRATEGIES:
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


def verify_repeat_seed_parity(seed=0, n_repeats=3, n_outer_folds=3, n_rows=200):
    """
    Empirically confirm that run_nested_cv()'s per-repeat outer-fold RNG
    seeding depends ONLY on `seed` (and n_repeats), not on split_strategy
    -- required for the composition-vs-chemistry Nadeau-Bengio pairing
    (CLAUDE.md Paper A item 1) to be valid: "repeat i" must land at the
    same point in the RNG stream in both rungs, so paired differences
    are comparing "the same random draw index" even though composition
    and chemistry group by different columns and can never share an
    actual row partition.

    Runs two REAL run_nested_cv() calls -- split_strategy="composition"
    and "chemistry", identical seed, tiny synthetic in-memory data
    (load_target_data patched so no CSV I/O), frozen ridge
    hyperparameters (cheap), checkpointing disabled -- with
    np.random.default_rng wrapped to record every seed it's constructed
    from, in call order. Compares the two recorded sequences (index 0 is
    rng_master itself, seeded from `seed` in both calls by construction;
    indices 1..n_repeats are the per-repeat repeat_rng seeds actually
    exercising run_nested_cv's real code path, not a manual
    re-derivation of it).

    Returns the shared sequence (list of ints) if the two rungs match;
    raises AssertionError with both sequences if they diverge -- would
    mean the NB-test pairing assumption is false and item 4 cannot
    treat composition/chemistry repeats as paired.
    """
    import tempfile
    from unittest.mock import patch

    rng = np.random.default_rng(12345)
    synthetic_df = pd.DataFrame(
        {
            "MagpieData_test": rng.normal(size=n_rows),
            TEMPERATURE_COL: rng.choice([300.0, 325.0, 350.0], size=n_rows),
            "dummy_target": rng.normal(size=n_rows),
            "composition_id": rng.integers(0, 20, size=n_rows),
            GROUP_COL: rng.integers(0, 40, size=n_rows),
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        frozen_path = Path(tmp_dir) / "dummy_ridge.json"
        with open(frozen_path, "w", encoding="utf-8") as f:
            json.dump(
                {"target": "dummy_target", "target_scale": "linear", "model_type": "ridge",
                 "inner_cv_r2": 0.0, "best_params": {"alpha": 1.0}},
                f,
            )

        recorded = {"composition": [], "chemistry": []}

        def _run_one(split_strategy):
            seeds_seen = []
            real_default_rng = np.random.default_rng

            def _recording_default_rng(s=None):
                seeds_seen.append(s)
                return real_default_rng(s)

            with patch("src.nested_cv.load_target_data", return_value=synthetic_df.copy()), \
                 patch("numpy.random.default_rng", side_effect=_recording_default_rng):
                run_nested_cv(
                    target="dummy_target", model_type="ridge", split_strategy=split_strategy,
                    n_repeats=n_repeats, n_outer_folds=n_outer_folds, seed=seed,
                    checkpoint_dir=False, frozen_hyperparams_path=frozen_path,
                )
            recorded[split_strategy] = seeds_seen

        _run_one("composition")
        _run_one("chemistry")

    comp_seeds, chem_seeds = recorded["composition"], recorded["chemistry"]
    if comp_seeds != chem_seeds:
        raise AssertionError(
            f"Per-repeat RNG seed sequences diverged between split_strategy='composition' "
            f"({comp_seeds}) and 'chemistry' ({chem_seeds}) for the same seed={seed} -- the "
            f"composition-vs-chemistry Nadeau-Bengio pairing assumption does NOT hold."
        )
    return comp_seeds


def nadeau_bengio_test(scores_a, scores_b, n_train, n_test):
    """
    Nadeau & Bengio (2003) corrected paired t-test for k repeated-CV
    scores from two rungs/methods. A naive paired t-test assumes
    independent samples; repeated-CV repeats share overlapping training
    data (e.g. the same rows land in different folds' training sets
    across repeats), which understates the true variance and inflates
    the false-positive rate. The correction inflates the variance
    estimate by (1/k + n_test/n_train) instead of the naive test's 1/k
    alone -- the test/train size ratio approximates the correlation
    that overlap induces.

    scores_a, scores_b: paired per-repeat scores, e.g. two
    run_nested_cv() calls' results_df.attrs["per_repeat_r2"] values for
    the SAME seed -- pairing requires "repeat i" to mean the same point
    in the RNG stream in both arrays, verified for this project by
    verify_repeat_seed_parity(), not assumed. n_train/n_test:
    representative SINGLE-FOLD training/test set sizes (e.g. the mean
    n_train/n_test across one rung's outer folds) -- not the whole
    dataset size and not summed across a repeat's folds.

    Returns a dict: k (number of paired repeats), mean_diff, var_diff,
    the corrected t-statistic, degrees of freedom (k-1), and a two-sided
    p-value (scipy.stats.t).
    """
    from scipy import stats

    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if scores_a.shape != scores_b.shape:
        raise ValueError(
            f"scores_a and scores_b must be paired (same shape): {scores_a.shape} vs {scores_b.shape}"
        )
    k = len(scores_a)
    if k < 2:
        raise ValueError(f"Nadeau-Bengio test needs at least 2 paired repeats, got k={k}")

    d = scores_a - scores_b
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))
    correction = 1.0 / k + n_test / n_train
    denom = np.sqrt(correction * var_d)

    if denom == 0:
        t_stat = 0.0 if mean_d == 0 else float("inf") * np.sign(mean_d)
    else:
        t_stat = mean_d / denom

    df = k - 1
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df))) if np.isfinite(t_stat) else 0.0

    return {
        "k": k, "mean_diff": mean_d, "var_diff": var_d,
        "n_train": n_train, "n_test": n_test,
        "t_statistic": float(t_stat), "df": df, "p_value": p_value,
    }


def _xgb_search_space(trial):
    # Full intended search space. Measured per-fit cost on this CPU-only
    # machine (396 features -- now 397 with temperature_bin added
    # 2026-08-18, negligible cost difference -- ~114k-126k row training
    # folds): 57s/fit at
    # max_depth=10/n_estimators=600 vs 3s at depth=3/n_estimators=100 --
    # depth dominates cost. At this range, the full 5-10 repeat design
    # projects to ~24-61 hours locally (measured/projected 2026-08-15,
    # see commit history for the calibration numbers), so this module
    # is intended to run on Kaggle/GPU, not this laptop. A capped range
    # (depth<=7, n_estimators<=350) was
    # used ONLY for local calibration runs and is not applied here --
    # do not narrow this range without an explicit, documented decision.
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def _build_xgb_model(params, device):
    return xgb.XGBRegressor(
        **params, n_jobs=-1, tree_method="hist", device=device, random_state=0, objective="reg:squarederror"
    )


def _lightgbm_search_space(trial):
    # Same depth/estimator/regularization scale as xgboost's space, plus
    # num_leaves (LightGBM's primary complexity control, leaf-wise growth
    # rather than xgboost's level-wise) and min_child_samples in place of
    # min_child_weight.
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def _build_lightgbm_model(params, device):
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "model_type='lightgbm' requires the lightgbm package (pip install lightgbm, "
            "pinned in requirements.txt)."
        ) from e
    # No GPU path for lightgbm in this module -- see module docstring;
    # `device` is accepted for interface parity with the other builders
    # but intentionally unused here.
    return lgb.LGBMRegressor(**params, n_jobs=-1, random_state=0, verbosity=-1)


def _random_forest_search_space(trial):
    # Bagged trees, not boosted -- structurally different from
    # xgboost/lightgbm (CLAUDE.md Paper B item 4's "bracket capacity"
    # framing: still a high-capacity tree ensemble, but no sequential
    # residual-fitting, no learning rate).
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
    }


def _build_random_forest_model(params, device):
    return RandomForestRegressor(**params, n_jobs=-1, random_state=0)


def _ridge_search_space(trial):
    # The constrained/lower-capacity model CLAUDE.md Paper B item 4 asks
    # the high-capacity GBDT to be bracketed against -- linear in the
    # features, one regularization knob.
    return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}


def _build_ridge_model(params, device):
    # StandardScaler first: MAGPIE/CBFV/temperature features span wildly
    # different scales (see module docstring) -- an unscaled Ridge fit
    # would be measuring that, not real model capacity.
    return make_pipeline(StandardScaler(), Ridge(**params, random_state=0))


MODEL_REGISTRY = {
    "xgboost": {"search_space": _xgb_search_space, "build": _build_xgb_model, "supports_gpu": True},
    "lightgbm": {"search_space": _lightgbm_search_space, "build": _build_lightgbm_model, "supports_gpu": False},
    "random_forest": {
        "search_space": _random_forest_search_space, "build": _build_random_forest_model, "supports_gpu": False,
    },
    "ridge": {"search_space": _ridge_search_space, "build": _build_ridge_model, "supports_gpu": False},
}


_NON_XGB_GPU_WARNED = False


def _resolve_device_for_model(model_type, device):
    """
    Only xgboost has a GPU path in this module (see module docstring).
    Downgrades device to "cpu" for any other model_type when
    device=="cuda" was requested, printing a one-time warning instead of
    silently ignoring the request or letting an unsupported device kwarg
    fail deep inside model construction.
    """
    global _NON_XGB_GPU_WARNED
    if device != "cuda" or MODEL_REGISTRY[model_type]["supports_gpu"]:
        return device
    if not _NON_XGB_GPU_WARNED:
        print(
            f"WARNING: device='cuda' requested but model_type={model_type!r} has no GPU path in "
            f"this module (only 'xgboost' does) -- running on CPU instead.",
            flush=True,
        )
        _NON_XGB_GPU_WARNED = True
    return "cpu"


def _objective(trial, model_type, X, y, groups, n_inner_folds, device="cpu"):
    registry = MODEL_REGISTRY[model_type]
    params = registry["search_space"](trial)

    inner_gkf = GroupKFold(n_splits=n_inner_folds)
    scores = []
    # split() only needs sample count and `groups` (always a host numpy
    # array -- see run_nested_cv); pass a host placeholder in X's place so
    # sklearn's split-index bookkeeping never touches a GPU-resident X.
    split_placeholder = np.empty(len(groups))
    for step, (inner_train_idx, inner_val_idx) in enumerate(inner_gkf.split(split_placeholder, groups=groups)):
        model = registry["build"](params, device)
        model.fit(X[inner_train_idx], y[inner_train_idx])
        preds = model.predict(X[inner_val_idx])
        scores.append(r2_score(_to_host(y[inner_val_idx]), _to_host(preds)))

        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def tune_hyperparameters(
    X_train, y_train, groups_train, model_type="xgboost",
    n_trials=N_OPTUNA_TRIALS, n_inner_folds=N_INNER_FOLDS, seed=0, device="cpu"
):
    """
    Nested GroupKFold hyperparameter search via Optuna (TPE sampler,
    MedianPruner), grouped by chemistry_cluster_id -- not a random
    split -- entirely inside the outer training fold. Search space and
    model constructor come from MODEL_REGISTRY[model_type]. Returns
    (best_params, best_inner_cv_r2).
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"model_type={model_type!r} must be one of {MODEL_TYPES}")
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: _objective(trial, model_type, X_train, y_train, groups_train, n_inner_folds, device=device),
        n_trials=n_trials,
    )
    return study.best_params, study.best_value


def tune_once(
    target="zT",
    model_type="xgboost",
    n_trials=N_OPTUNA_TRIALS,
    n_inner_folds=N_INNER_FOLDS,
    seed=0,
    device="cpu",
    output_path=None,
):
    """
    Tune model_type's hyperparameters ONCE via chemistry-cluster-grouped
    CV on the full dataset, then save the result to disk -- CLAUDE.md
    Paper A item 1: "tune hyperparameters ONCE on the chemistry-cluster
    split via nested CV, freeze." tune_hyperparameters()'s Optuna search
    already validates every trial via an inner chemistry-cluster-grouped
    GroupKFold (n_inner_folds), so a single call here on the FULL
    dataset (not one outer fold's training subset) is a complete
    nested-CV tuning pass on its own. Each of the five ladder rungs then
    supplies the "outer" half -- its own fold scheme, evaluated with
    these frozen hyperparameters via
    run_nested_cv(model_type=model_type, frozen_hyperparams_path=output_path),
    never retuning.

    output_path defaults to FROZEN_HYPERPARAMS_DIR/<target>_<model_type>.json
    -- hyperparameters are model-specific, so each model_type gets its
    own frozen file even for the same target. For target in
    LOG_TRANSFORM_TARGETS ("sigma", "kappa"), y is log10-transformed
    before tuning (see LOG_TRANSFORM_TARGETS docstring), so inner_cv_r2
    is computed in log space for those two targets, linear scale for
    S/zT -- result["target_scale"] records which.
    Returns the saved result dict (target, target_scale, model_type,
    tuning params, inner_cv_r2, best_params, n_rows, n_features).
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"model_type={model_type!r} must be one of {MODEL_TYPES}")
    device = _resolve_device_for_model(model_type, device)

    df = load_target_data(target)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    target_scale = _target_scale(target)
    y = _transform_target(y, target)
    groups = df[GROUP_COL].to_numpy()

    X = _to_device(X, device)
    y = _to_device(y, device)

    print(
        f"tune_once: target={target} (scale={target_scale}), model_type={model_type}, {len(df):,} rows, "
        f"{len(feature_cols)} features, {len(np.unique(groups)):,} chemistry_cluster_id groups, "
        f"n_trials={n_trials}, n_inner_folds={n_inner_folds}, device={device}",
        flush=True,
    )

    best_params, inner_cv_r2 = tune_hyperparameters(
        X, y, groups, model_type=model_type, n_trials=n_trials, n_inner_folds=n_inner_folds, seed=seed, device=device
    )

    result = {
        "target": target,
        "target_scale": target_scale,
        "model_type": model_type,
        "n_trials": n_trials,
        "n_inner_folds": n_inner_folds,
        "seed": seed,
        "device": device,
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "inner_cv_r2": inner_cv_r2,
        "best_params": best_params,
    }

    output_path = Path(output_path) if output_path else FROZEN_HYPERPARAMS_DIR / f"{target}_{model_type}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"tune_once: inner CV R^2 = {inner_cv_r2:.4f}", flush=True)
    print(f"tune_once: saved frozen hyperparameters to {output_path}", flush=True)
    return result


def _load_frozen_hyperparams(path, expected_model_type=None):
    """
    Load a tune_once() JSON output. Returns (best_params, inner_cv_r2).
    Raises ValueError if expected_model_type is given and doesn't match
    the file's model_type -- hyperparameters are model-specific, so
    silently reusing e.g. xgboost params to construct a Ridge model
    would otherwise crash confusingly deep inside model construction
    instead of with a clear message here.
    """
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    file_model_type = result.get("model_type", "xgboost")  # files predating --model default to xgboost
    if expected_model_type is not None and file_model_type != expected_model_type:
        raise ValueError(
            f"{path} was tuned for model_type={file_model_type!r}, but this run requested "
            f"model_type={expected_model_type!r}. Hyperparameters are model-specific -- generate "
            f"a separate frozen file per model with --tune-once --model {expected_model_type}."
        )
    return result["best_params"], result["inner_cv_r2"]


def _fold_checkpoint_path(checkpoint_dir, repeat, fold):
    return checkpoint_dir / f"repeat{repeat}_fold{fold}.json"


def _predictions_path(checkpoint_dir, repeat, fold):
    return checkpoint_dir / f"repeat{repeat}_fold{fold}_predictions.npz"


def _save_predictions(checkpoint_dir, repeat, fold, y_true, y_pred):
    """
    Persist one outer fold's held-out (y_true, y_pred) pair (host numpy
    arrays) so pooled out-of-fold R^2 can be reconstructed correctly
    across a resumed run -- not just from folds computed in the current
    process. No-op if checkpointing is disabled.
    """
    if checkpoint_dir is None:
        return
    np.savez(_predictions_path(checkpoint_dir, repeat, fold), y_true=y_true, y_pred=y_pred)


def _load_predictions(checkpoint_dir, repeat, fold):
    """
    Load one outer fold's saved (y_true, y_pred) pair. Raises
    FileNotFoundError for a checkpoint written before pooled-R^2 support
    existed (no predictions file saved) -- callers should warn and
    exclude that fold from pooling rather than crash the whole run.
    """
    data = np.load(_predictions_path(checkpoint_dir, repeat, fold))
    return data["y_true"], data["y_pred"]


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
    composition, data identity, or which model/scale a checkpoint's
    results belong to (target, target_scale, model_type, split_strategy,
    seed, n_outer_folds). model_type and target_scale are fatal (not
    warn-only) even though neither affects fold COMPOSITION, specifically
    so pointing two different models -- or resuming a sigma/kappa
    checkpoint_dir predating the log10-transform decision (2026-08-20,
    see LOG_TRANSFORM_TARGETS) -- at the same checkpoint_dir can never
    silently reuse cached predictions/outer_r2 computed under a
    different model or a different target scale. That would silently
    corrupt a multi-model comparison or mix log-space and linear-scale
    predictions in one pooled R^2, rather than just cost search-quality
    like the warn-only fields below. n_repeats is
    allowed to increase (extending an existing run is safe since the
    outer RNG stream is drawn sequentially, one draw per repeat index,
    so repeats 0..old_n_repeats-1 reproduce identically regardless of
    how many more repeats a later call asks for). n_trials/n_inner_folds/
    device/frozen_hyperparams_path only affect hyperparameter search
    cost, where it runs, or which hyperparameters get used -- not fold
    composition -- so mismatches are warned about, not fatal.
    """
    path = checkpoint_dir / "run_config.json"
    if not path.exists():
        _write_run_config(checkpoint_dir, config)
        return

    with open(path, encoding="utf-8") as f:
        existing = json.load(f)

    fatal_keys = ("target", "target_scale", "model_type", "split_strategy", "seed", "n_outer_folds")
    # existing.get(), not existing[], on BOTH sides: a checkpoint_dir predating the
    # introduction of a given fatal key (e.g. target_scale, added 2026-08-20) simply
    # lacks that key entirely -- must still raise the clean ValueError below, not a
    # bare KeyError from indexing a missing key.
    mismatches = {k: (existing.get(k), config[k]) for k in fatal_keys if existing.get(k) != config[k]}
    if mismatches:
        raise ValueError(
            f"Resume parameter mismatch in {checkpoint_dir}: {mismatches}. "
            f"target/target_scale/model_type/split_strategy/seed/n_outer_folds must match the run "
            f"that produced the existing checkpoints, since changing any of them changes fold "
            f"composition or which model/scale the checkpointed results belong to. Use a different "
            f"checkpoint_dir for a genuinely new run."
        )
    if existing.get("n_repeats", 0) > config["n_repeats"]:
        raise ValueError(
            f"Resume requested n_repeats={config['n_repeats']} but checkpoints already exist for "
            f"n_repeats={existing['n_repeats']} in {checkpoint_dir}. Shrinking n_repeats on resume "
            f"would silently discard completed results; pass n_repeats >= {existing['n_repeats']}."
        )
    for k in ("n_inner_folds", "n_trials", "device", "frozen_hyperparams_path"):
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
    model_type="xgboost",
    split_strategy="chemistry",
    n_repeats=None,
    n_outer_folds=N_OUTER_FOLDS,
    n_inner_folds=N_INNER_FOLDS,
    n_trials=N_OPTUNA_TRIALS,
    seed=0,
    device="cpu",
    checkpoint_dir=None,
    frozen_hyperparams_path=None,
):
    """
    Full repeated nested CV for one target, one model_type (MODEL_TYPES;
    see module docstring and MODEL_REGISTRY for the per-model search
    space/constructor), outer folds built under `split_strategy` (one of
    SPLIT_STRATEGIES; see the module docstring and outer_splits() for
    what each rung is).

    n_repeats: defaults to None, resolved per split_strategy rather than
    one fixed number for all five ladder rungs -- N_OUTER_REPEATS_GROUPED
    (5) for composition/chemistry (CLAUDE.md's group-composition-effect
    justification), N_OUTER_REPEATS_UNGROUPED (1) for random/kfold (no
    such confound). Pass explicitly to override. See module docstring.

    frozen_hyperparams_path: path to a tune_once() JSON output for this
    SAME model_type (mismatches raise, see _load_frozen_hyperparams).
    When given, every outer fold reuses those hyperparameters unchanged
    instead of calling tune_hyperparameters() -- this is what makes
    split_strategy the ONLY varying factor across the five ladder rungs
    (CLAUDE.md Paper A item 1). When None (default), hyperparameters are
    tuned fresh inside each outer training fold via a nested Optuna
    search grouped by chemistry_cluster_id (always, regardless of
    split_strategy), then refit on the full outer-training fold and
    evaluated once on the outer-test fold -- the original, pre-ladder
    behavior, still useful on its own.

    device: "cpu" (default) or "cuda" for GPU-accelerated fits, e.g. on
    Kaggle -- only meaningful for model_type="xgboost" (see module
    docstring and _resolve_device_for_model); silently downgraded to
    "cpu" (with a one-time warning) for every other model_type.

    checkpoint_dir: directory to write one JSON file per completed
    (repeat, fold) -- {"repeat", "fold", "split_strategy", "model_type",
    "hyperparam_source", "n_train", "n_test", "outer_r2", "inner_cv_r2",
    "param_*"} -- plus a same-named *_predictions.npz holding that
    fold's held-out (y_true, y_pred) arrays (used to reconstruct pooled
    out-of-fold R^2 correctly across a resumed run), immediately after
    that fold finishes, plus a run_config.json recording the parameters
    that must match on resume. Defaults to
    checkpoints/nested_cv/<target>/<model_type>/<split_strategy>/. On
    startup, any (repeat, fold) with an existing checkpoint file is
    skipped and its saved result (and predictions) reused instead of
    recomputed -- this is how a run resumes after a Kaggle session is
    killed mid-run. Resuming REQUIRES the same
    target/model_type/split_strategy/seed/n_outer_folds as the original
    call (n_repeats may be increased to extend the run); a mismatch
    raises rather than silently producing an inconsistent result set
    (model_type is fatal, not warn-only, specifically so pointing two
    different models at the same checkpoint_dir can never silently reuse
    one model's cached predictions under another model's label). Pass
    checkpoint_dir=False to disable checkpointing entirely (nothing
    written, nothing skipped).

    For target in LOG_TRANSFORM_TARGETS ("sigma", "kappa"), y is
    log10-transformed before training/evaluation (see
    LOG_TRANSFORM_TARGETS docstring for rationale) -- every R^2 in
    results_df and its .attrs is then computed in log10 space, not
    linear/raw scale, for those two targets; S/zT stay linear.
    results_df.attrs["target_scale"] ("log10" or "linear") and the
    "target_scale" checkpoint column record which for every row, so this
    is never ambiguous downstream.

    Returns results_df with extra metadata attached via .attrs:
    "pooled_r2" and "pooled_n" -- the PRIMARY ladder metric (CLAUDE.md
    Paper A item 1), a single R^2 computed from every held-out
    prediction across all outer folds and repeats concatenated together,
    distinct from the per-fold "outer_r2" column's macro-averaged
    mean/std (secondary/diagnostic; the two can diverge, especially for
    split_strategy="random" where held-out sets overlap and vary in
    size across draws) -- plus "model_type", "split_strategy",
    "target_scale", and "hyperparam_source" for labeling a multi-model
    comparison figure.
    """
    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"split_strategy={split_strategy!r} must be one of {SPLIT_STRATEGIES}")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"model_type={model_type!r} must be one of {MODEL_TYPES}")

    device = _resolve_device_for_model(model_type, device)

    if n_repeats is None:
        n_repeats = N_OUTER_REPEATS_GROUPED if split_strategy in GROUPED_SPLIT_STRATEGIES else N_OUTER_REPEATS_UNGROUPED

    if frozen_hyperparams_path is not None:
        frozen_params, frozen_inner_r2 = _load_frozen_hyperparams(frozen_hyperparams_path, expected_model_type=model_type)
    else:
        frozen_params, frozen_inner_r2 = None, None

    if checkpoint_dir is False:
        checkpoint_dir = None
    elif checkpoint_dir is None:
        checkpoint_dir = Path("checkpoints") / "nested_cv" / target / model_type / split_strategy
    else:
        checkpoint_dir = Path(checkpoint_dir)

    target_scale = _target_scale(target)

    completed = {}
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _check_run_config(
            checkpoint_dir,
            {
                "target": target, "target_scale": target_scale, "model_type": model_type,
                "split_strategy": split_strategy,
                "seed": seed, "n_outer_folds": n_outer_folds,
                "n_repeats": n_repeats, "n_inner_folds": n_inner_folds,
                "n_trials": n_trials, "device": device,
                "frozen_hyperparams_path": str(frozen_hyperparams_path) if frozen_hyperparams_path else None,
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
    y = _transform_target(y, target)

    # chemistry_groups is used for two independent purposes: (a) the
    # inner hyperparameter-tuning split, always, regardless of
    # split_strategy; (b) the outer split itself, only when
    # split_strategy == "chemistry". Both stay on host: grouping/index
    # bookkeeping only, never fed to the model.
    chemistry_groups = df[GROUP_COL].to_numpy()
    group_lookup = {"chemistry": chemistry_groups, "composition": df["composition_id"].to_numpy()}

    # Convert once, up front, not per-fold -- see _to_device docstring for
    # why this is the fix for the cross-device DMatrix-fallback warning.
    # No-op for model_type != "xgboost" since device was already resolved
    # to "cpu" above.
    X = _to_device(X, device)
    y = _to_device(y, device)

    if split_strategy in GROUPED_SPLIT_STRATEGIES:
        group_col = SPLIT_STRATEGY_GROUP_COL[split_strategy]
        n_groups = len(np.unique(group_lookup[split_strategy]))
        split_desc = f"split_strategy={split_strategy} ({n_groups:,} {group_col} groups)"
    else:
        split_desc = f"split_strategy={split_strategy} (ungrouped)"
    hyperparam_desc = (
        f"frozen ({frozen_hyperparams_path})" if frozen_params is not None else "retuned per outer fold"
    )
    start_msg = (
        f"target={target} (scale={target_scale}): model_type={model_type}, {len(df):,} rows, "
        f"{len(feature_cols)} features, {split_desc}, n_repeats={n_repeats}, "
        f"hyperparameters={hyperparam_desc}, device={device}"
    )
    print(start_msg, flush=True)
    _log_progress(checkpoint_dir, start_msg)

    results = list(completed.values())
    # Bucketed by repeat, not one flat list: lets us report a pooled
    # R^2 PER REPEAT (pool only that repeat's n_outer_folds folds), not
    # just one R^2 pooled across every repeat and fold together. The
    # flat, all-repeats pooled_r2/pooled_n below are still computed --
    # just derived from these buckets rather than a separate list --
    # since r2_score doesn't care about concatenation order.
    pooled_by_repeat = {r: {"y_true": [], "y_pred": []} for r in range(n_repeats)}
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
                if checkpoint_dir is not None:
                    try:
                        y_true_saved, y_pred_saved = _load_predictions(checkpoint_dir, repeat, fold)
                        pooled_by_repeat[repeat]["y_true"].append(y_true_saved)
                        pooled_by_repeat[repeat]["y_pred"].append(y_pred_saved)
                    except FileNotFoundError:
                        warn_msg = (
                            f"WARNING: repeat {repeat} fold {fold} has a checkpoint but no saved "
                            f"predictions file (checkpointed before pooled-R^2 support was added) -- "
                            f"excluded from this run's pooled out-of-fold R^2, which will undercount."
                        )
                        print(warn_msg, flush=True)
                        _log_progress(checkpoint_dir, warn_msg)
                continue

            X_train, y_train, groups_train = X[train_idx], y[train_idx], chemistry_groups[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if frozen_params is not None:
                best_params, inner_r2 = frozen_params, frozen_inner_r2
            else:
                best_params, inner_r2 = tune_hyperparameters(
                    X_train, y_train, groups_train, model_type=model_type,
                    n_trials=n_trials, n_inner_folds=n_inner_folds, seed=repeat * 100 + fold, device=device,
                )

            model = MODEL_REGISTRY[model_type]["build"](best_params, device)
            model.fit(X_train, y_train)
            y_test_host = _to_host(y_test)
            y_pred_host = _to_host(model.predict(X_test))
            outer_r2 = r2_score(y_test_host, y_pred_host)

            _save_predictions(checkpoint_dir, repeat, fold, y_test_host, y_pred_host)
            pooled_by_repeat[repeat]["y_true"].append(y_test_host)
            pooled_by_repeat[repeat]["y_pred"].append(y_pred_host)

            record = {
                "repeat": repeat, "fold": fold, "model_type": model_type, "split_strategy": split_strategy,
                "target_scale": target_scale,
                "hyperparam_source": "frozen" if frozen_params is not None else "retuned_per_fold",
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

    # Per-repeat pooled R^2: pool only the n_outer_folds folds belonging
    # to ONE repeat, giving n_repeats separate R^2 values -- CLAUDE.md's
    # bare all-repeats pooled_r2 below has no spread to report; this is
    # what a mean+-SD ladder entry, and the Nadeau-Bengio paired test
    # between two rungs (see nadeau_bengio_test), are actually computed
    # from.
    per_repeat_r2 = {}
    for repeat in range(n_repeats):
        yt = pooled_by_repeat[repeat]["y_true"]
        yp = pooled_by_repeat[repeat]["y_pred"]
        if yt:
            per_repeat_r2[repeat] = float(r2_score(np.concatenate(yt), np.concatenate(yp)))

    if per_repeat_r2:
        per_repeat_r2_values = np.array([per_repeat_r2[r] for r in sorted(per_repeat_r2)])
        per_repeat_r2_mean = float(per_repeat_r2_values.mean())
        per_repeat_r2_std = (
            float(per_repeat_r2_values.std(ddof=1)) if len(per_repeat_r2_values) > 1 else float("nan")
        )
    else:
        per_repeat_r2_values = np.array([])
        per_repeat_r2_mean = float("nan")
        per_repeat_r2_std = float("nan")

    # Flat, all-repeats pooled R^2: same underlying predictions as
    # per_repeat_r2 above, just concatenated across every repeat instead
    # of kept separate -- r2_score doesn't depend on concatenation
    # order, so this is unaffected by the per-repeat bucketing.
    all_y_true = [arr for bucket in pooled_by_repeat.values() for arr in bucket["y_true"]]
    all_y_pred = [arr for bucket in pooled_by_repeat.values() for arr in bucket["y_pred"]]
    if all_y_true:
        pooled_true_arr = np.concatenate(all_y_true)
        pooled_pred_arr = np.concatenate(all_y_pred)
        pooled_r2 = float(r2_score(pooled_true_arr, pooled_pred_arr))
        pooled_n = int(len(pooled_true_arr))
    else:
        pooled_r2 = float("nan")
        pooled_n = 0

    results_df.attrs["pooled_r2"] = pooled_r2
    results_df.attrs["pooled_n"] = pooled_n
    results_df.attrs["per_repeat_r2"] = per_repeat_r2
    results_df.attrs["per_repeat_r2_mean"] = per_repeat_r2_mean
    results_df.attrs["per_repeat_r2_std"] = per_repeat_r2_std
    results_df.attrs["model_type"] = model_type
    results_df.attrs["split_strategy"] = split_strategy
    results_df.attrs["target_scale"] = target_scale
    results_df.attrs["hyperparam_source"] = "frozen" if frozen_params is not None else "retuned_per_fold"

    r2_scale_note = (
        f"computed in log10 space (target={target} trained on log10-transformed y, "
        f"see LOG_TRANSFORM_TARGETS)" if target_scale == "log10" else "computed in linear (raw) space"
    )
    per_repeat_str = ", ".join(f"{per_repeat_r2[r]:.4f}" for r in sorted(per_repeat_r2))
    summary_lines = [
        f"=== {target} ({model_type}, {split_strategy}, scale={target_scale}): repeated CV summary ===",
        f"{n_repeats} repeats x {n_outer_folds} outer folds = {len(results_df)} outer evaluations",
        f"R^2 below {r2_scale_note}.",
        f"POOLED out-of-fold R^2 = {pooled_r2:.4f} (n={pooled_n:,} held-out predictions) "
        f"-- all repeats combined into one number",
        f"PER-REPEAT pooled R^2 = {per_repeat_r2_mean:.4f} +/- {per_repeat_r2_std:.4f} "
        f"(n={len(per_repeat_r2_values)} repeats: [{per_repeat_str}]) "
        f"-- PRIMARY ladder metric (mean +/- across-repeat SD), CLAUDE.md Paper A item 1",
        f"mean outer R^2 = {mean_r2:.4f}, std = {std_r2:.4f} (per-fold macro-average, secondary/diagnostic)",
        f"hyperparameters: {hyperparam_desc}",
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
        "--model", dest="model_type", default="xgboost", choices=MODEL_TYPES,
        help="Model family (default: xgboost). Same nested-CV machinery, chemistry-cluster "
        "grouping, and pooled-OOF-R^2 reporting for every choice -- only the hyperparameter "
        "search space and model constructor differ (see MODEL_REGISTRY). Only xgboost has a "
        "GPU path in this module.",
    )
    parser.add_argument(
        "--split-strategy", default="chemistry", choices=SPLIT_STRATEGIES,
        help="Outer-fold scheme, one rung of CLAUDE.md's five-way validation ladder (default: chemistry). "
        "Use --split-strategy kfold with --n-outer-folds 5 or 10 for those two rungs.",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help='Device for model_type="xgboost" (tree_method="hist", device=...); every other '
        "--model runs on CPU regardless, with a one-time warning (default: cpu; use cuda on "
        "Kaggle/GPU)",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory for per-fold checkpoint JSON files "
        "(default: checkpoints/nested_cv/<target>/<model_type>/<split_strategy>/)",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=None,
        help="Outer repeats. Default depends on --split-strategy: "
        f"{N_OUTER_REPEATS_GROUPED} for grouped strategies (composition/chemistry, to average "
        f"away GroupKFold's fold-composition effect), {N_OUTER_REPEATS_UNGROUPED} for ungrouped "
        "strategies (random/kfold, no group-composition confound to average away) -- see "
        "CLAUDE.md's Grouping Key section. Pass explicitly to override.",
    )
    parser.add_argument("--n-outer-folds", type=int, default=N_OUTER_FOLDS, help=f"Outer folds per repeat (default: {N_OUTER_FOLDS})")
    parser.add_argument("--n-inner-folds", type=int, default=N_INNER_FOLDS, help=f"Inner tuning folds (default: {N_INNER_FOLDS})")
    parser.add_argument("--n-trials", type=int, default=N_OPTUNA_TRIALS, help=f"Optuna trials per outer fold (default: {N_OPTUNA_TRIALS})")
    parser.add_argument("--seed", type=int, default=0, help="Master RNG seed (default: 0)")
    parser.add_argument(
        "--frozen-hyperparams", default=None,
        help="Path to a tune_once() JSON output for this SAME --model. When given, every outer "
        "fold reuses those hyperparameters unchanged instead of retuning -- makes split_strategy "
        "the ONLY varying factor across the five ladder rungs (CLAUDE.md Paper A item 1). "
        "Generate with --tune-once --model <same model>.",
    )
    parser.add_argument(
        "--tune-once", action="store_true",
        help="Run ONLY the one-time chemistry-cluster-grouped hyperparameter search for --model "
        "and save it to --frozen-hyperparams-out, then exit -- does not run the ladder itself. "
        "Run this once per target PER MODEL, then pass its output to --frozen-hyperparams.",
    )
    parser.add_argument(
        "--frozen-hyperparams-out", default=None,
        help="Output path for --tune-once "
        "(default: checkpoints/frozen_hyperparams/<target>_<model>.json)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.tune_once:
        return tune_once(
            target=args.target,
            model_type=args.model_type,
            n_trials=args.n_trials,
            n_inner_folds=args.n_inner_folds,
            seed=args.seed,
            device=args.device,
            output_path=args.frozen_hyperparams_out,
        )

    results_df = run_nested_cv(
        target=args.target,
        model_type=args.model_type,
        split_strategy=args.split_strategy,
        n_repeats=args.n_repeats,
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
        n_trials=args.n_trials,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        frozen_hyperparams_path=args.frozen_hyperparams,
    )
    print(results_df, flush=True)
    print(
        f"\npooled out-of-fold R^2 = {results_df.attrs.get('pooled_r2'):.4f} "
        f"(n={results_df.attrs.get('pooled_n'):,}, scale={results_df.attrs.get('target_scale')})",
        flush=True,
    )
    return results_df


if __name__ == "__main__":
    main()
