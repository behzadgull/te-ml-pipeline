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
hyperparameter set: zT's CANONICAL frozen hyperparameters
(checkpoints/saved_predictions/checkpoints/frozen_hyperparams/zT.json --
the same file every confirmed result in CLAUDE.md cites) -- deliberately,
so the comparison isolates "does going through three intermediate models
and combining formulaically lose accuracy," without also confounding it
with each property getting its own independently-tuned model. This module
CONSUMES that canonical file; it does not tune. (A prior version loaded
from nested_cv.py's ephemeral tune_once cache instead, which had silently
drifted to a stale, pre-File-A hyperparameter set -- see
load_zt_frozen_hyperparams's docstring.) Every fit uses the full search
space the canonical file was tuned with (see nested_cv.py's
_xgb_search_space); nothing here caps it for local-runtime convenience.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.nested_cv import (
    GROUP_COL,
    MODEL_REGISTRY,
    N_OUTER_FOLDS,
    N_OUTER_REPEATS_GROUPED,
    PROCESSED_DATA_DIR,
    PROJECT,
    _load_frozen_hyperparams,
    get_feature_columns,
    randomized_group_kfold,
)

CHECKPOINT_DIR = Path("checkpoints") / "direct_vs_derived_zt"

# Duplicated from src.external_validation.FROZEN_HYPERPARAMS_DIR rather than
# imported from there: importing it here would create a circular import
# (external_validation -> backtransform_check -> direct_vs_derived_zt ->
# external_validation). Keep this value in sync with that module's constant
# if it ever changes -- both must point at the same canonical directory.
CANONICAL_FROZEN_HYPERPARAMS_DIR = Path("checkpoints/saved_predictions/checkpoints/frozen_hyperparams")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# S/zT stay linear; sigma/kappa are trained in log10 space, matching
# nested_cv.py's LOG_TRANSFORM_TARGETS -- the derived-zT formula below
# converts sigma_pred/kappa_pred back to linear (10**pred) before
# combining.
COMPONENT_KEYS = ("zT_direct", "S", "sigma_log10", "kappa_log10")

# Each component model's own canonical frozen hyperparameters, used when
# hyperparams_mode="per_target". The default, "zt_shared", reproduces the
# published comparison (results/direct_vs_derived_snapfix/20260918T200058/):
# zT's set applied to all four models.
COMPONENT_FROZEN_JSON = {
    "zT_direct": "zT.json",
    "S": "S.json",
    "sigma_log10": "sigma.json",
    "kappa_log10": "kappa.json",
}
HYPERPARAMS_MODES = ("zt_shared", "per_target")


def _git_state():
    """
    Return {"git_head", "tree_clean", "git_source"} for the code that is
    running (CLAUDE.md standing rule: every run config records the code
    commit and whether the working tree was clean). Uses git when the
    working directory is a repository; otherwise falls back to the
    TE_GIT_HEAD / TE_TREE_CLEAN environment variables (for a code copy
    without .git, e.g. an uploaded Kaggle dataset), recorded as
    git_source="env". Raises if neither is available: an unrecorded
    code identity is what the rule exists to prevent.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        return {"git_head": head, "tree_clean": porcelain.strip() == "", "git_source": "git"}
    except (OSError, subprocess.CalledProcessError):
        head, clean = os.environ.get("TE_GIT_HEAD"), os.environ.get("TE_TREE_CLEAN")
        if head and clean in ("true", "false"):
            return {"git_head": head, "tree_clean": clean == "true", "git_source": "env"}
        raise RuntimeError(
            "Cannot record the code identity: not a git repository, and TE_GIT_HEAD / "
            "TE_TREE_CLEAN (true|false) are not set."
        )


def _run_provenance(dataset_path, hyperparams_paths):
    """Dataset SHA256 and size, code commit and tree state, and each hyperparameter file's SHA256."""
    dataset_path = Path(dataset_path)
    state = _git_state()
    return {
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "dataset_bytes": dataset_path.stat().st_size,
        **state,
        "hyperparams_sha256": {str(k): _sha256_file(v) for k, v in hyperparams_paths.items()},
    }


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


def load_zt_frozen_hyperparams(model_type="xgboost"):
    """
    Load zT's CANONICAL frozen hyperparameters
    (checkpoints/saved_predictions/checkpoints/frozen_hyperparams/zT.json)
    -- the same file every confirmed result in CLAUDE.md cites, produced
    once by `nested_cv.py --tune-once --target zT`. This function is a
    consumer of that canonical file, not a producer: a missing file raises
    FileNotFoundError rather than silently tuning a fresh, uncommitted
    substitute and writing it into an untracked cache.

    That silent-tune-and-cache path was the source of a prior bug: this
    module used to load from nested_cv.py's ephemeral tune_once cache
    (checkpoints/frozen_hyperparams/zT_xgboost.json) instead of this
    canonical file. That cache held a stale hyperparameter set tuned
    before File A became canonical (129,188 rows, File B's row count, not
    File A's 129,419) -- silently different from every other confirmed
    result in this project, undetected until an explicit audit compared
    the two paths.

    Returns (best_params, inner_cv_r2, path).
    """
    path = CANONICAL_FROZEN_HYPERPARAMS_DIR / "zT.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical zT frozen hyperparameters not found at {path}. This script consumes the "
            f"canonical frozen hyperparameters and does not tune -- run "
            f"`python src/nested_cv.py --tune-once --target zT` first if this file is genuinely missing."
        )
    print(f"Loading canonical frozen zT hyperparameters: {path}", flush=True)
    best_params, inner_cv_r2 = _load_frozen_hyperparams(path, expected_model_type=model_type)
    return best_params, inner_cv_r2, path


def load_component_frozen_hyperparams(model_type="xgboost"):
    """
    Load each component model's OWN canonical frozen hyperparameters
    (S.json, sigma.json, kappa.json, zT.json in the canonical directory),
    for hyperparams_mode="per_target". Returns {component_key:
    (best_params, inner_cv_r2, path)}. Consumer only: a missing file
    raises, as in load_zt_frozen_hyperparams.
    """
    out = {}
    for key, fname in COMPONENT_FROZEN_JSON.items():
        path = CANONICAL_FROZEN_HYPERPARAMS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Canonical frozen hyperparameters for {key} not found at {path}.")
        best_params, inner_cv_r2 = _load_frozen_hyperparams(path, expected_model_type=model_type)
        out[key] = (best_params, inner_cv_r2, path)
    return out


def _fold_path(repeat, fold, checkpoint_dir=CHECKPOINT_DIR):
    return Path(checkpoint_dir) / f"repeat{repeat}_fold{fold}.npz"


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
    hyperparams_mode="zt_shared",
):
    """
    Run the full direct-vs-derived comparison: N_OUTER_REPEATS_GROUPED
    repeats x N_OUTER_FOLDS chemistry-cluster-grouped outer folds (same
    repeat count/randomization CLAUDE.md's Grouping Key section requires
    for any grouped CV rung), fitting four models per fold (direct zT,
    S, sigma_log10, kappa_log10) with ONE shared frozen hyperparameter
    set: zT's canonical frozen hyperparameters. Each fold's four (y_true, y_pred) arrays are
    checkpointed to checkpoint_dir immediately, so an interrupted run
    resumes instead of restarting; pass checkpoint_dir=None to disable.

    Returns a dict: pooled R^2 + n for each of the four component models
    plus "zT_derived" (the combined S^2*sigma*T/kappa prediction scored
    against actual zT), subset size/cluster count, and the frozen
    hyperparameters used.

    hyperparams_mode: "zt_shared" (default, the published comparison) fits
    all four models with zT's canonical frozen hyperparameters;
    "per_target" fits each model with its own canonical frozen file
    (S.json, sigma.json, kappa.json, zT.json). Everything else (subset,
    folds, seeds, back-transform) is identical. run_config.json, written
    to checkpoint_dir, records the dataset SHA256, code commit,
    tree_clean, and each hyperparameter file's SHA256; a resumed run
    whose recorded inputs differ raises instead of mixing checkpoints.
    """
    if hyperparams_mode not in HYPERPARAMS_MODES:
        raise ValueError(f"hyperparams_mode={hyperparams_mode!r} must be one of {HYPERPARAMS_MODES}")
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

    if hyperparams_mode == "zt_shared":
        zt_params, zt_inner_cv_r2, zt_path = load_zt_frozen_hyperparams(model_type=model_type)
        params_by_key = {key: zt_params for key in COMPONENT_KEYS}
        hyperparams_paths = {"zt_shared": zt_path}
        hyperparams_source = "zT canonical frozen hyperparameters (shared across all four models)"
        inner_cv_r2 = zt_inner_cv_r2
        best_params = zt_params
    else:
        loaded = load_component_frozen_hyperparams(model_type=model_type)
        params_by_key = {key: loaded[key][0] for key in COMPONENT_KEYS}
        hyperparams_paths = {key: loaded[key][2] for key in COMPONENT_KEYS}
        hyperparams_source = "per-target canonical frozen hyperparameters (each model uses its own target's file)"
        inner_cv_r2 = {key: loaded[key][1] for key in COMPONENT_KEYS}
        best_params = params_by_key

    n_groups = len(np.unique(groups))
    print(
        f"Subset: {len(subset):,} rows from {source_path}, {n_groups:,} chemistry clusters, "
        f"{len(feature_cols)} features. Hyperparameters ({hyperparams_source}): {best_params}",
        flush=True,
    )

    provenance = _run_provenance(source_path, hyperparams_paths)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "hyperparams_mode": hyperparams_mode,
            "model_type": model_type,
            "seed": seed,
            "n_repeats": n_repeats,
            "n_outer_folds": n_outer_folds,
            "device": device,
            "subset_n_rows": len(subset),
            "subset_n_chemistry_clusters": n_groups,
            "provenance": provenance,
        }
        config_path = checkpoint_dir / "run_config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                previous = json.load(f)
            # device and code identity may legitimately differ on resume; the inputs may not.
            keys = ("hyperparams_mode", "model_type", "seed", "n_repeats", "n_outer_folds",
                    "subset_n_rows", "subset_n_chemistry_clusters")
            drift = [k for k in keys if previous.get(k) != run_config[k]]
            for k in ("dataset_sha256", "hyperparams_sha256"):
                if previous.get("provenance", {}).get(k) != provenance[k]:
                    drift.append(f"provenance.{k}")
            if drift:
                raise ValueError(
                    f"{config_path} was written by a run with different inputs ({', '.join(drift)}); "
                    f"use a fresh checkpoint_dir instead of mixing checkpoints."
                )
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

    print(
        f"REGENERATION GATE -- checkpoint_dir={checkpoint_dir}, hyperparams_mode={hyperparams_mode}, "
        f"hyperparams_sha256={provenance['hyperparams_sha256']}, "
        f"dataset_sha256={provenance['dataset_sha256']}, git_head={provenance['git_head']}, "
        f"tree_clean={provenance['tree_clean']}, subset_n_rows={len(subset):,}, "
        f"subset_n_chemistry_clusters={n_groups:,}. "
        f"A rerun against different inputs must print a different value here.",
        flush=True,
    )

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
            fold_path = _fold_path(repeat, fold, checkpoint_dir) if checkpoint_dir is not None else None

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
                fold_preds[key] = _fit_predict(X_train, y[key][train_idx], X_test, params_by_key[key], model_type, device)
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
    results["hyperparams_mode"] = hyperparams_mode
    results["frozen_hyperparams_source"] = hyperparams_source
    results["frozen_hyperparams_inner_cv_r2"] = inner_cv_r2
    results["best_params"] = best_params
    results["provenance"] = provenance

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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Direct-vs-derived zT comparison (CLAUDE.md Paper A item 5).")
    parser.add_argument("--hyperparams", choices=HYPERPARAMS_MODES, default="zt_shared",
                        help="zt_shared: zT's frozen set for all four models (published run); "
                             "per_target: each model's own frozen file")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR),
                        help="per-fold checkpoints and run_config.json (use a fresh directory per configuration)")
    args = parser.parse_args(argv)

    results = run_direct_vs_derived(
        device=args.device, checkpoint_dir=args.checkpoint_dir, hyperparams_mode=args.hyperparams
    )
    report(results)
    out_path = Path(args.checkpoint_dir) / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
