"""
Paired comparison of models scored on the SAME chemistry-cluster folds (5 repeats x 5 folds), with chemistry-cluster bootstrap intervals.

  na2:  XGBoost (the committed Paper A rung, frozen hyperparameters) against LightGBM (tuned, na2_trees.py) and the random forest (one fixed setting, na2_trees.py --fixed-hyperparams), per target.
  na1:  the frozen-hyperparameter rung of Paper A (tuned once on all rows, so the evaluation folds were seen by the tuning) against the NESTED rung of na1_nested_cv.py (re-tuned inside every outer
        training fold), per target. Difference frozen minus nested = the optimism of the non-nested result (positive: the earlier numbers were optimistic).

Method. The folds are rebuilt here from the snapfix CSV (same seed, same fold code) so that the chemistry cluster of every held-out row is known; every unit of every model is checked against the rebuild
(identical measured values in the same order). The pooled out-of-fold R2 of a repeat is computed from per-cluster sums, so that a bootstrap draw (clusters resampled with replacement, every row of a drawn
cluster in every repeat carried along, n_boot draws, seed 0) gives the R2 of each model and repeat on the same resampled data; the statistic is the mean over the 5 repeats and the difference of two models'
means is paired. Intervals: the 2.5th and 97.5th percentiles of the bootstrap distribution. Also reported: the per-repeat values, their mean and SD, and the paired per-repeat differences with their
t interval (df 4); the repeats re-split the same clusters, so the cluster bootstrap is the interval for "which chemistries are in the data", the t interval only for the fold assignment.

Usage (from the repository root):  python thesis_paper/scripts/model_comparison.py na2   or   python thesis_paper/scripts/model_comparison.py na1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import run_record as rr  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

DATASET = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
DATASET_SHA256 = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"
LADDER = "results/ladder_regen_snapfix/20260917T150000"
RESULTS = "thesis_paper/results"
TARGETS = ("S", "sigma", "kappa", "zT")
N_REPEATS, N_FOLDS, SEED = 5, 5, 0
BUNDLES = {  # analysis -> {model: {target: results folder (a run folder under thesis_paper/results)}}
    "na2": {"lightgbm": {t: f"{RESULTS}/na2_lightgbm/20261005T101837" for t in TARGETS},
            "random_forest": {"S": f"{RESULTS}/na2_rf_S/20261005T100927", "sigma": f"{RESULTS}/na2_rf_sigma/20261005T102059", "kappa": f"{RESULTS}/na2_rf_kappa/20261005T101358", "zT": f"{RESULTS}/na2_rf_zT/20261005T101629"}},
    "na1": {"nested": {"S": f"{RESULTS}/na1_a/20261005T103553", "kappa": f"{RESULTS}/na1_a/20261005T103553", "sigma": f"{RESULTS}/na1_b/20261005T103553", "zT": f"{RESULTS}/na1_b/20261005T103553"}},
}


def load_preds(analysis, model, target, run, r, f):
    """y_true, y_pred of one unit of a model."""
    if analysis == "na1":
        u = np.load(REPO / run / "units" / f"rung_{target}_r{r}_f{f}.npz")
    else:
        u = np.load(REPO / run / "units" / f"rung_{target}_repeat{r}_fold{f}.npz")
    return u["y_true"], u["y_pred"]


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", choices=["na1", "na2"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--stacking-dir", default=None, help="na2 only: the output folder of scripts/na2_stacking_analysis.py (units of the nested stack's outer predictions)")
    args = ap.parse_args()
    bundles = {m: dict(per) for m, per in BUNDLES[args.analysis].items()}
    if args.stacking_dir:
        assert args.analysis == "na2"
        bundles["stacking"] = {t: args.stacking_dir for t in TARGETS}
    base = "xgboost_frozen"
    inputs = {"dataset": DATASET}
    for model, per in bundles.items():
        for run in sorted(set(per.values())):
            inputs[f"{model}:{run}/manifest"] = f"{run}/manifest.json"
    prov = rr.provenance(inputs, __file__)
    assert prov["inputs"]["dataset"]["sha256"] == DATASET_SHA256, "not the snapfix CSV"
    out = {"analysis": args.analysis, "n_boot": args.n_boot, "reference": base, "targets": {}}
    rng_boot = np.random.default_rng(SEED)
    tq = stats.t.ppf(0.975, N_REPEATS - 1)
    for target in TARGETS:
        models = {base: None}
        models.update({m: per[target] for m, per in bundles.items() if target in per})
        df = pd.read_csv(REPO / DATASET, usecols=[ncv.GROUP_COL, target])
        df = df[df[target].notna()].reset_index(drop=True)
        y = ncv._transform_target(df[target].to_numpy(dtype=np.float64), target)
        codes, uniq = pd.factorize(df[ncv.GROUP_COL].to_numpy())
        nc = len(uniq)
        rng_master = np.random.default_rng(SEED)
        yy, preds, cl = [], {m: [] for m in models}, []
        for r in range(N_REPEATS):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            order, parts = [], {m: [] for m in models}
            for f, (_, te) in enumerate(ncv.outer_splits("chemistry", len(df), {"chemistry": df[ncv.GROUP_COL].to_numpy()}, N_FOLDS, rng)):
                order.append(te)
                for m, run in models.items():
                    if m == base:
                        u = np.load(REPO / LADDER / f"{target}_chemistry_full" / f"repeat{r}_fold{f}_predictions.npz")
                        yt, yp = u["y_true"], u["y_pred"]
                    else:
                        yt, yp = load_preds(args.analysis, m, target, run, r, f)
                    assert len(yt) == len(te) and np.allclose(yt, y[te]), f"{target} {m} repeat {r} fold {f}: the rebuilt fold differs from the saved one"
                    parts[m].append(np.asarray(yp, dtype=float))
            order = np.concatenate(order)
            yy.append(y[order])
            cl.append(codes[order])
            for m in models:
                preds[m].append(np.concatenate(parts[m]))
        # per-cluster sums, per repeat: n, sum y, sum y^2, SSE of each model
        S = {"n": np.zeros((N_REPEATS, nc)), "sy": np.zeros((N_REPEATS, nc)), "syy": np.zeros((N_REPEATS, nc))}
        SSE = {m: np.zeros((N_REPEATS, nc)) for m in models}
        for r in range(N_REPEATS):
            S["n"][r] = np.bincount(cl[r], minlength=nc)
            S["sy"][r] = np.bincount(cl[r], weights=yy[r], minlength=nc)
            S["syy"][r] = np.bincount(cl[r], weights=yy[r] ** 2, minlength=nc)
            for m in models:
                SSE[m][r] = np.bincount(cl[r], weights=(yy[r] - preds[m][r]) ** 2, minlength=nc)

        if args.analysis == "na2":  # the unweighted mean of the three base models: no fitted meta-learner, so no leak to worry about
            assert all(b in models for b in (base, "lightgbm", "random_forest"))
            preds["mean_of_three"] = [np.mean([preds[b][r] for b in (base, "lightgbm", "random_forest")], axis=0) for r in range(N_REPEATS)]
            models["mean_of_three"] = None
            SSE["mean_of_three"] = np.zeros((N_REPEATS, nc))
            for r in range(N_REPEATS):
                SSE["mean_of_three"][r] = np.bincount(cl[r], weights=(yy[r] - preds["mean_of_three"][r]) ** 2, minlength=nc)

        def r2s(w):
            n, sy, syy = S["n"] @ w, S["sy"] @ w, S["syy"] @ w
            sst = syy - sy ** 2 / n
            return {m: 1 - (SSE[m] @ w) / sst for m in models}  # per repeat

        full = r2s(np.ones(nc))
        boots = {m: [] for m in models}
        for _ in range(args.n_boot):
            res = r2s(np.bincount(rng_boot.integers(0, nc, nc), minlength=nc).astype(float))
            for m in models:
                boots[m].append(float(res[m].mean()))
        pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        t_out = {"n_rows": int(len(df)), "n_clusters": int(nc), "models": {}, "differences_vs_reference": {}}
        for m in models:
            v = full[m]
            t_out["models"][m] = {"per_repeat_r2": v.tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "ci95_cluster_bootstrap_of_mean": pct(boots[m])}
        ref = np.array(boots[base])
        for m in models:
            if m == base:
                continue
            d_rep = full[m] - full[base]
            d_boot = np.array(boots[m]) - ref
            t_out["differences_vs_reference"][m] = {"mean_diff": float(d_rep.mean()), "ci95_cluster_bootstrap": pct(d_boot), "per_repeat_diff": d_rep.tolist(),
                                                    "ci95_t_across_repeats": [float(d_rep.mean() - tq * d_rep.std(ddof=1) / np.sqrt(N_REPEATS)), float(d_rep.mean() + tq * d_rep.std(ddof=1) / np.sqrt(N_REPEATS))],
                                                    "share_of_bootstrap_draws_model_above_reference": float((d_boot > 0).mean())}
        if args.analysis == "na2" and "stacking" in models:  # the stack against the best single model of the target (chosen on these same folds, which favours the single model), and against the plain mean
            singles = [base, "lightgbm", "random_forest"]
            best = max(singles, key=lambda m: float(full[m].mean()))
            t_out["best_single"] = best
            t_out["extra_pairs"] = {}
            for a, b in (("stacking", best), ("mean_of_three", best), ("stacking", "mean_of_three")):
                d_rep = full[a] - full[b]
                d_boot = np.array(boots[a]) - np.array(boots[b])
                t_out["extra_pairs"][f"{a}_minus_{b}"] = {"mean_diff": float(d_rep.mean()), "ci95_cluster_bootstrap": pct(d_boot), "per_repeat_diff": d_rep.tolist(),
                                                          "share_of_bootstrap_draws_first_above_second": float((d_boot > 0).mean())}
        if args.analysis == "na1":  # fold level: how often the nested fold R2 is below the frozen one, and the inner-CV R2 of the best trials
            run = bundles["nested"][target]
            nested_fold, frozen_fold, inner = [], [], []
            for r in range(N_REPEATS):
                for f in range(N_FOLDS):
                    meta = json.loads((REPO / run / "units" / f"rung_{target}_r{r}_f{f}.json").read_text(encoding="utf-8"))["meta"]
                    nested_fold.append(meta["outer_r2"]); frozen_fold.append(meta["committed_frozen_outer_r2"]); inner.append(meta["best_inner_cv_r2"])
            nf, ff = np.array(nested_fold), np.array(frozen_fold)
            t_out["fold_level"] = {"n_folds": int(len(nf)), "frozen_minus_nested_mean": float((ff - nf).mean()), "folds_nested_below_frozen": int((nf < ff).sum()),
                                   "mean_best_inner_cv_r2": float(np.mean(inner)), "mean_nested_outer_r2": float(nf.mean()), "mean_frozen_outer_r2": float(ff.mean())}
        out["targets"][target] = t_out
        print(target, {m: round(t_out["models"][m]["mean"], 4) for m in models}, {m: round(d["mean_diff"], 4) for m, d in t_out["differences_vs_reference"].items()}, flush=True)
    d = REPO / "thesis_paper" / "results" / f"{args.analysis}_comparison" / rr.utc_stamp()
    d.mkdir(parents=True, exist_ok=True)
    rr.write_json(d / "comparison.json", out)
    rr.write_json(d / "run_config.json", {**prov, "n_repeats": N_REPEATS, "n_folds": N_FOLDS, "seed": SEED})
    print("wrote", d)


if __name__ == "__main__":
    main()
