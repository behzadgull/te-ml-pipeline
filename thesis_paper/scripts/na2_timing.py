"""
Run-time estimate for the NA2 CPU sessions on Kaggle (4 cores): one fixed-setting random-forest fit and LightGBM trials, measured here with 4 threads and calibrated against a real Kaggle
measurement.

Calibration. The first random-forest session (results/na2_random_forest_tuning/<stamp>, Kaggle CPU, 4 cores, S) recorded the wall time of tuning trial 12 (600 trees, depth 30, 3 inner
folds): that exact configuration is timed here on the same kind of inner split (GroupKFold(3) of all S rows, first split: fit and predict), with a few trees and scaled to its tree count,
and the ratio Kaggle / local is the calibration factor. Nothing is scored: only seconds are recorded, no R2 or prediction is kept.
Estimates. For every target, the fixed-setting random-forest fit (RF_FIXED of kaggle/na2_trees.py) is timed on the first outer training fold of the rung (repeat 0, fold 0) with a few trees
and scaled to 500 trees (tree-wise cost is linear), then multiplied by the calibration factor. LightGBM: three configurations of its search space (smallest, middle and largest, n_estimators
100 / 350 / 600) are timed on the S inner split and on the S outer training fold; a trial costs three inner fits, a rung unit one outer fit.

Usage (from the repository root):  python thesis_paper/scripts/na2_timing.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import run_record as rr  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

DATASET = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
TRIAL_RECORD = "thesis_paper/results/na2_random_forest_tuning/20261004T100611/units/tune_S_trial012.json"
RF_FIXED = {"n_estimators": 500, "max_features": 1 / 3, "min_samples_leaf": 5, "max_depth": None, "min_samples_split": 2, "bootstrap": True}
N_JOBS = 4
TREES_TIMED = (4, 12)  # two tree counts; the per-tree time is their difference
LGB = {"smallest": dict(n_estimators=100, max_depth=3, num_leaves=15, learning_rate=0.1, subsample=0.5, colsample_bytree=0.5, min_child_samples=50, reg_lambda=1.0, reg_alpha=1.0),
       "middle": dict(n_estimators=350, max_depth=8, num_leaves=63, learning_rate=0.05, subsample=0.75, colsample_bytree=0.75, min_child_samples=20, reg_lambda=1.0, reg_alpha=1.0),
       "largest": dict(n_estimators=600, max_depth=12, num_leaves=255, learning_rate=0.02, subsample=1.0, colsample_bytree=1.0, min_child_samples=5, reg_lambda=0.01, reg_alpha=0.01)}


def timed_fit_predict(make, X, y, tr, te):
    """Seconds of fit + predict."""
    t0 = time.perf_counter()
    m = make()
    m.fit(X[tr], y[tr])
    m.predict(X[te])
    return time.perf_counter() - t0


def per_tree_seconds(params, X, y, tr, te):
    """Seconds per tree of a random forest, from two tree counts (fit + predict)."""
    t = []
    for n in TREES_TIMED:
        t.append(timed_fit_predict(lambda: RandomForestRegressor(**{**params, "n_estimators": n}, n_jobs=N_JOBS, random_state=0), X, y, tr, te))
    return (t[1] - t[0]) / (TREES_TIMED[1] - TREES_TIMED[0]), t


def main():
    """Entry point."""
    prov = rr.provenance({"dataset": DATASET, "kaggle_trial_record": TRIAL_RECORD}, __file__)
    head = pd.read_csv(REPO / DATASET, nrows=2)
    feat = ncv.get_feature_columns(head)
    df = pd.read_csv(REPO / DATASET, usecols=[ncv.GROUP_COL, "S", "sigma", "kappa", "zT", *feat])
    out = {"n_jobs": N_JOBS, "trees_timed": TREES_TIMED, "targets": {}}
    # calibration on S
    s = df[df["S"].notna()].reset_index(drop=True)
    X = s[feat].to_numpy(np.float64)
    y = s["S"].to_numpy(np.float64)
    g = s[ncv.GROUP_COL].to_numpy()
    tr, te = next(iter(GroupKFold(n_splits=3).split(np.empty(len(g)), groups=g)))
    trial = json.loads((REPO / TRIAL_RECORD).read_text(encoding="utf-8"))
    p12 = {**trial["params"]}
    per_tree, raw = per_tree_seconds(p12, X, y, tr, te)
    local_trial_seconds = 3 * per_tree * p12["n_estimators"]
    calib = trial["seconds"] / local_trial_seconds
    out["calibration"] = {"trial": 12, "params": p12, "kaggle_trial_seconds": trial["seconds"], "local_seconds_per_tree_fit_predict": per_tree, "raw_seconds": raw,
                          "n_train_inner": int(len(tr)), "local_estimated_trial_seconds": local_trial_seconds, "factor_kaggle_over_local": calib}
    for target in ("S", "sigma", "kappa", "zT"):
        d = df[df[target].notna()].reset_index(drop=True)
        Xt = d[feat].to_numpy(np.float64)
        yt = ncv._transform_target(d[target].to_numpy(np.float64), target)
        gt = d[ncv.GROUP_COL].to_numpy()
        rng = np.random.default_rng(np.random.default_rng(0).integers(0, 2**32 - 1))
        tro, teo = next(iter(ncv.outer_splits("chemistry", len(d), {"chemistry": gt}, 5, rng)))
        pt, rawt = per_tree_seconds(RF_FIXED, Xt, yt, tro, teo)
        rf_local = pt * RF_FIXED["n_estimators"]
        out["targets"][target] = {"n_rows": int(len(d)), "n_train_outer": int(len(tro)), "rf_fixed_local_seconds_per_tree": pt, "rf_fixed_local_seconds_per_fit": rf_local,
                                  "rf_fixed_kaggle_seconds_per_fit": rf_local * calib, "rf_fixed_kaggle_hours_for_25_fits": 25 * rf_local * calib / 3600}
    # LightGBM on S (inner split and outer training fold)
    lg = {}
    for name, p in LGB.items():
        ti = timed_fit_predict(lambda: lgb.LGBMRegressor(**p, n_jobs=N_JOBS, random_state=0, verbosity=-1), X, y, tr, te)
        d = s
        rng = np.random.default_rng(np.random.default_rng(0).integers(0, 2**32 - 1))
        tro, teo = next(iter(ncv.outer_splits("chemistry", len(d), {"chemistry": g}, 5, rng)))
        to = timed_fit_predict(lambda: lgb.LGBMRegressor(**p, n_jobs=N_JOBS, random_state=0, verbosity=-1), X, y, tro, teo)
        lg[name] = {"params": p, "local_inner_fit_seconds": ti, "local_outer_fit_seconds": to}
    out["lightgbm_S_local"] = lg
    out["lightgbm_note"] = "local seconds with 4 threads, not calibrated: the LightGBM factor Kaggle / local was not measured; the random-forest factor is reported for reference"
    d = REPO / "thesis_paper" / "results" / "na2_timing" / rr.utc_stamp()
    d.mkdir(parents=True, exist_ok=True)
    rr.write_json(d / "timing.json", out)
    rr.write_json(d / "run_config.json", {**prov, "cpu_count": __import__("os").cpu_count(), "lightgbm": lgb.__version__})
    print("wrote", d)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
