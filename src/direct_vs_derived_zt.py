"""
Direct-vs-derived zT pathway (CLAUDE.md Paper A item 5).

Compares two ways of getting a zT prediction, on the identical all-four-
properties-present subset (S, sigma, kappa, zT all non-null) and the
identical chemistry-cluster grouped CV splits for both pathways:

  (a) DIRECT: predict zT straight from features.
  (b) DERIVED: predict S, sigma, kappa separately (sigma/kappa in log10
      space, matching nested_cv.py's LOG_TRANSFORM_TARGETS), then
      combine via zT_derived = (S_pred/1e6)^2 * sigma_pred * T / kappa_pred
      -- the same formula data_cleaning.py's step6_zt_self_consistency
      uses (Snyder & Toberer 2008's zT identity; S converted from uV/K to
      V/K, T = the row's temperature_bin, which holds actual Kelvin
      values, not a bin index).

Both pathways, and all three component models, reuse ONE frozen
hyperparameter set from a single chemistry-cluster tune_once() run on
zT -- deliberately, so the comparison isolates "does going through three
intermediate models and combining formulaically lose accuracy," without
also confounding it with each property getting its own independently-
tuned model. Every fit uses the full search space (see nested_cv.py's
_xgb_search_space); nothing here caps it for local-runtime convenience.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.nested_cv import (
    FROZEN_HYPERPARAMS_DIR,
    GROUP_COL,
    MODEL_REGISTRY,
    N_OUTER_FOLDS,
    N_OUTER_REPEATS_GROUPED,
    PROCESSED_DATA_DIR,
    PROJECT,
    _load_frozen_hyperparams,
    get_feature_columns,
    randomized_group_kfold,
    tune_once,
)

CHECKPOINT_DIR = Path("checkpoints") / "direct_vs_derived_zt"

# S/zT stay linear; sigma/kappa are trained in log10 space, matching
# nested_cv.py's LOG_TRANSFORM_TARGETS -- the derived-zT formula below
# converts sigma_pred/kappa_pred back to linear (10**pred) before
# combining.
COMPONENT_KEYS = ("zT_direct", "S", "sigma_log10", "kappa_log10")


def load_all_four_subset(processed_data_dir=PROCESSED_DATA_DIR, project=PROJECT):
    """
    Load the most recent featurized_<project>_*.csv, filtered to rows
    where S, sigma, kappa, AND zT are all non-null -- the subset CLAUDE.md
    Paper A item 5 requires both the direct and derived pathway be
    evaluated on identically, so training-set-size doesn't confound the
    comparison.
    """
    candidates = sorted(Path(processed_data_dir).glob(f"featurized_{project}_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No featurized_{project}_*.csv in {processed_data_dir}; run src/featurization.py first"
        )
    df = pd.read_csv(candidates[-1])
    subset = df.dropna(subset=["S", "sigma", "kappa", "zT"]).reset_index(drop=True)
    return subset, candidates[-1]


def get_or_tune_zt_hyperparams(model_type="xgboost", device="cpu", **tune_kwargs):
    """
    Load zT's frozen hyperparameters from FROZEN_HYPERPARAMS_DIR if
    already tuned (e.g. by a prior nested_cv.py --tune-once --target zT
    run); otherwise run tune_once(target="zT", ...) now and save it
    there, so a later `nested_cv.py --frozen-hyperparams` run for the
    production five-way ladder can reuse the identical file rather than
    silently duplicating the search. Returns (best_params, inner_cv_r2).
    """
    path = FROZEN_HYPERPARAMS_DIR / f"zT_{model_type}.json"
    if path.exists():
        print(f"Reusing existing frozen zT hyperparameters: {path}", flush=True)
        return _load_frozen_hyperparams(path, expected_model_type=model_type)

    print(f"No frozen zT hyperparameters found at {path}; running tune_once now.", flush=True)
    tune_once(target="zT", model_type=model_type, device=device, output_path=path, **tune_kwargs)
    return _load_frozen_hyperparams(path, expected_model_type=model_type)


def _fold_path(repeat, fold):
    return CHECKPOINT_DIR / f"repeat{repeat}_fold{fold}.npz"


def _fit_predict(X_train, y_train, X_test, params, model_type, device):
    model = MODEL_REGISTRY[model_type]["build"](params, device)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_direct_vs_derived(
    model_type="xgboost",
    n_repeats=N_OUTER_REPEATS_GROUPED,
    n_outer_folds=N_OUTER_FOLDS,
    seed=0,
    device="cpu",
    checkpoint_dir=CHECKPOINT_DIR,
):
    """
    Run the full direct-vs-derived comparison: N_OUTER_REPEATS_GROUPED
    repeats x N_OUTER_FOLDS chemistry-cluster-grouped outer folds (same
    repeat count/randomization CLAUDE.md's Grouping Key section requires
    for any grouped CV rung), fitting four models per fold (direct zT,
    S, sigma_log10, kappa_log10) with ONE shared frozen hyperparameter
    set from zT's tune_once. Each fold's four (y_true, y_pred) arrays are
    checkpointed to checkpoint_dir immediately, so an interrupted run
    resumes instead of restarting; pass checkpoint_dir=None to disable.

    Returns a dict: pooled R^2 + n for each of the four component models
    plus "zT_derived" (the combined S^2*sigma*T/kappa prediction scored
    against actual zT), subset size/cluster count, and the frozen
    hyperparameters used.
    """
    subset, source_path = load_all_four_subset()
    feature_cols = get_feature_columns(subset)
    X = subset[feature_cols].to_numpy(dtype=np.float64)
    groups = subset[GROUP_COL].to_numpy()
    T = subset["temperature_bin"].to_numpy(dtype=np.float64)

    y = {
        "zT_direct": subset["zT"].to_numpy(dtype=np.float64),
        "S": subset["S"].to_numpy(dtype=np.float64),
        "sigma_log10": np.log10(subset["sigma"].to_numpy(dtype=np.float64)),
        "kappa_log10": np.log10(subset["kappa"].to_numpy(dtype=np.float64)),
    }
    y_zt_actual = subset["zT"].to_numpy(dtype=np.float64)

    best_params, inner_cv_r2 = get_or_tune_zt_hyperparams(model_type=model_type, device=device)

    n_groups = len(np.unique(groups))
    print(
        f"Subset: {len(subset):,} rows from {source_path}, {n_groups:,} chemistry clusters, "
        f"{len(feature_cols)} features. Frozen zT hyperparameters (inner CV R^2={inner_cv_r2:.4f}): "
        f"{best_params}",
        flush=True,
    )

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pooled = {k: {"y_true": [], "y_pred": []} for k in COMPONENT_KEYS}
    derived_true, derived_pred = [], []

    rng_master = np.random.default_rng(seed)
    t_start = time.perf_counter()
    fold_num, total_folds = 0, n_repeats * n_outer_folds

    for repeat in range(n_repeats):
        repeat_rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        fold_iter = randomized_group_kfold(groups, n_outer_folds, repeat_rng)
        for fold, (train_idx, test_idx) in enumerate(fold_iter):
            fold_num += 1
            fold_path = _fold_path(repeat, fold) if checkpoint_dir is not None else None

            if fold_path is not None and fold_path.exists():
                data = np.load(fold_path)
                for key in COMPONENT_KEYS:
                    pooled[key]["y_true"].append(data[f"{key}_true"])
                    pooled[key]["y_pred"].append(data[f"{key}_pred"])
                derived_true.append(data["zT_derived_true"])
                derived_pred.append(data["zT_derived_pred"])
                print(f"[{fold_num}/{total_folds}] repeat {repeat} fold {fold}: skipping, checkpointed", flush=True)
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            T_test = T[test_idx]

            fold_preds = {}
            for key in COMPONENT_KEYS:
                fold_preds[key] = _fit_predict(X_train, y[key][train_idx], X_test, best_params, model_type, device)
                pooled[key]["y_true"].append(y[key][test_idx])
                pooled[key]["y_pred"].append(fold_preds[key])

            sigma_pred = 10.0 ** fold_preds["sigma_log10"]
            kappa_pred = 10.0 ** fold_preds["kappa_log10"]
            zt_derived_pred = ((fold_preds["S"] / 1.0e6) ** 2) * sigma_pred * T_test / kappa_pred
            zt_derived_true = y_zt_actual[test_idx]
            derived_true.append(zt_derived_true)
            derived_pred.append(zt_derived_pred)

            if fold_path is not None:
                save_kwargs = {"zT_derived_true": zt_derived_true, "zT_derived_pred": zt_derived_pred}
                for key in COMPONENT_KEYS:
                    save_kwargs[f"{key}_true"] = y[key][test_idx]
                    save_kwargs[f"{key}_pred"] = fold_preds[key]
                np.savez(fold_path, **save_kwargs)

            elapsed = time.perf_counter() - t_start
            print(
                f"[{fold_num}/{total_folds}] repeat {repeat} fold {fold}: "
                f"n_train={len(train_idx):,}, n_test={len(test_idx):,}, elapsed={elapsed:.0f}s",
                flush=True,
            )

    results = {}
    for key in COMPONENT_KEYS:
        yt = np.concatenate(pooled[key]["y_true"])
        yp = np.concatenate(pooled[key]["y_pred"])
        results[key] = {"pooled_r2": float(r2_score(yt, yp)), "n": int(len(yt))}

    derived_true_arr = np.concatenate(derived_true)
    derived_pred_arr = np.concatenate(derived_pred)
    results["zT_derived"] = {
        "pooled_r2": float(r2_score(derived_true_arr, derived_pred_arr)),
        "n": int(len(derived_true_arr)),
    }

    results["subset_n_rows"] = len(subset)
    results["subset_n_chemistry_clusters"] = n_groups
    results["model_type"] = model_type
    results["n_repeats"] = n_repeats
    results["n_outer_folds"] = n_outer_folds
    results["frozen_hyperparams_source"] = "zT tune_once (shared across all four models)"
    results["frozen_hyperparams_inner_cv_r2"] = inner_cv_r2
    results["best_params"] = best_params

    return results


def report(results):
    """Print the direct-vs-derived comparison: subset size, both pooled R^2 values, and component diagnostics."""
    print()
    print(f"Subset: {results['subset_n_rows']:,} rows, {results['subset_n_chemistry_clusters']:,} chemistry clusters")
    print(f"Model: {results['model_type']}, frozen hyperparameters from {results['frozen_hyperparams_source']}")
    print()
    print(f"{'Pathway':<20}{'Pooled R^2':>12}{'n':>10}")
    print("-" * 42)
    print(f"{'DIRECT zT':<20}{results['zT_direct']['pooled_r2']:>12.4f}{results['zT_direct']['n']:>10,}")
    print(f"{'DERIVED zT':<20}{results['zT_derived']['pooled_r2']:>12.4f}{results['zT_derived']['n']:>10,}")
    print()
    print("Component models feeding the derived pathway (diagnostic, not the headline comparison):")
    for key, label in [("S", "S (linear)"), ("sigma_log10", "sigma (log10)"), ("kappa_log10", "kappa (log10)")]:
        print(f"  {label:<18}{results[key]['pooled_r2']:>10.4f}  (n={results[key]['n']:,})")


def main():
    results = run_direct_vs_derived()
    report(results)
    out_path = CHECKPOINT_DIR / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
