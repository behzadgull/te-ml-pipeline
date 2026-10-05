"""
NA1: nested grouped CV. How optimistic is the chemistry-cluster R2 of Paper A, whose hyperparameters were tuned once on all rows and so also saw the rows of every evaluation fold?

For each target (S, sigma, kappa, zT) and each outer fold (repeat r, fold f) of the chemistry-cluster rung (the same folds as Paper A: same seed, same randomized group k-fold; the fold sizes are checked
against the committed run):
  1. tune on the outer TRAINING rows only: one Optuna search of --n-trials trials (TPE, median pruner, src.nested_cv's XGBoost search space and objective), each trial scored by --n-inner-folds-fold
     chemistry-cluster GroupKFold inside the training rows; every trial is a checkpointed unit (a killed session resumes the same search; the TPE sampler is reseeded with the number of trials
     done, so a resumed search is valid but not bit-identical to an uninterrupted one);
  2. refit the best set on the outer training rows and predict the outer test rows: one unit per fold with y_true and y_pred.
Targets sigma and kappa are trained and scored in log10 space, as in Paper A. The result is, per target, the pooled out-of-fold R2 of each repeat under nested tuning, set against the committed
Paper A value (hyperparameters frozen from the one tuning on all rows), and their paired difference per repeat.

Smoke mode: ~3,000 rows, 2 trials, 1 repeat x 3 folds, a tiny search space, CPU.

Usage:
    python thesis_paper/scripts/kaggle/na1_nested_cv.py --targets S,kappa --out-dir /kaggle/working/na1_a --expect-commit <sha> --device cuda --time-budget-hours 10.5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import harness as H  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

LADDER_DIR = "results/ladder_regen_snapfix/20260917T150000"
LADDER_METRICS = "reports/regen_snapfix/20260917T150000/ladder_metrics.json"
TARGETS = ("S", "sigma", "kappa", "zT")
TARGET_ROWS = {"S": 185064, "sigma": 182755, "kappa": 121110, "zT": 129419}


def smoke_space(trial):
    """A tiny XGBoost search space for the plumbing test."""
    return {"n_estimators": trial.suggest_int("n_estimators", 10, 30, step=10), "max_depth": trial.suggest_int("max_depth", 2, 3),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True)}


def space_distributions(space):
    """The Optuna distributions of a search space, read off a throw-away trial."""
    t = optuna.create_study().ask()
    space(t)
    return dict(t.distributions)


def tune_fold(sess, target, r, f, X_tr, y_tr, g_tr, args):
    """Run (or resume) the tuning trials of one outer fold; returns (best params, best inner R2, tuning seconds)."""
    space = ncv.MODEL_REGISTRY["xgboost"]["search_space"]
    dists = space_distributions(space)
    prefix = f"tune_{target}_r{r}_f{f}_"
    done = sorted(u for u in sess._all_units() if u.startswith(prefix))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed + 1000 * r + 10 * f + len(done)),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    secs = 0.0
    for u in done:
        meta, _ = sess.load(u)
        secs += meta["seconds"]
        if meta["state"] == "complete":
            study.add_trial(optuna.trial.create_trial(params=meta["params"], distributions=dists, value=meta["value"]))
        else:
            study.add_trial(optuna.trial.create_trial(params=meta["params"], distributions=dists, state=optuna.trial.TrialState.PRUNED, value=None))
    n_done = len(done)
    while n_done < args.n_trials:
        if sess.out_of_time():
            return None
        trial = study.ask()
        t0 = time.perf_counter()
        try:
            value = ncv._objective(trial, "xgboost", X_tr, y_tr, g_tr, args.n_inner_folds, device=args.device)
            study.tell(trial, value)
            meta = {"state": "complete", "value": float(value)}
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            meta = {"state": "pruned", "value": None}
        meta.update({"params": dict(trial.params), "seconds": time.perf_counter() - t0, "trial_number": n_done})
        sess.save(f"{prefix}trial{n_done:02d}", meta)
        secs += meta["seconds"]
        n_done += 1
    complete = [sess.load(f"{prefix}trial{k:02d}")[0] for k in range(args.n_trials)]
    complete = [m for m in complete if m["state"] == "complete"]
    best = max(complete, key=lambda m: m["value"])
    return best["params"], best["value"], secs


def main():
    ap = argparse.ArgumentParser()
    H.add_common_args(ap)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--n-trials", type=int, default=ncv.N_OPTUNA_TRIALS)
    ap.add_argument("--n-inner-folds", type=int, default=ncv.N_INNER_FOLDS)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.n_trials, args.n_repeats, args.n_folds, args.device = 2, 1, 3, "cpu"
        ncv.MODEL_REGISTRY["xgboost"] = {**ncv.MODEL_REGISTRY["xgboost"], "search_space": smoke_space}
    targets = [t for t in args.targets.split(",") if t]
    sess = H.Session("na1_nested_cv", args, {"targets": targets, "n_trials": args.n_trials, "n_inner_folds": args.n_inner_folds, "n_repeats": args.n_repeats,
                                             "n_folds": args.n_folds, "seed": args.seed, "smoke": args.smoke})
    df_all = H.load_frame(sess)
    units_total = len(targets) * args.n_repeats * args.n_folds * (args.n_trials + 1)
    stop = False
    for target in targets:
        df = df_all[df_all[target].notna()].reset_index(drop=True)
        if args.smoke:
            df = H.smoke_subset(df, ncv.GROUP_COL, 3000)
        elif len(df) != TARGET_ROWS[target]:
            raise SystemExit(f"{target} has {len(df)} rows, expected {TARGET_ROWS[target]}")
        cols = ncv.get_feature_columns(df)
        X = df[cols].to_numpy(dtype=np.float64)
        y = ncv._transform_target(df[target].to_numpy(dtype=np.float64), target)
        groups = df[ncv.GROUP_COL].to_numpy()
        Xd, yd = ncv._to_device(X, args.device), ncv._to_device(y, args.device)
        rng_master = np.random.default_rng(args.seed)
        for r in range(args.n_repeats):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            for f, (tr, te) in enumerate(ncv.outer_splits("chemistry", len(df), {"chemistry": groups}, args.n_folds, rng)):
                uid = f"rung_{target}_r{r}_f{f}"
                if sess.done(uid):
                    continue
                if sess.out_of_time():
                    print("time budget reached before", uid, flush=True)
                    stop = True
                    break
                ref = None
                if not args.smoke:
                    ref = json.loads((H.repo_root() / LADDER_DIR / f"{target}_chemistry_full" / f"repeat{r}_fold{f}.json").read_text(encoding="utf-8"))
                    assert len(te) == ref["n_test"] and len(tr) == ref["n_train"], "fold assignment differs from the committed XGBoost rung"
                X_tr, y_tr = Xd[tr], yd[tr]
                tuned = tune_fold(sess, target, r, f, X_tr, y_tr, groups[tr], args)
                if tuned is None:
                    print("time budget reached during the tuning of", uid, flush=True)
                    stop = True
                    break
                params, inner_r2, tune_secs = tuned
                t0 = time.perf_counter()
                model = ncv._build_xgb_model(params, args.device)
                model.fit(X_tr, y_tr)
                pred = ncv._to_host(model.predict(Xd[te])).astype(float)
                r2 = H.r2(y[te], pred)
                sess.save(uid, {"target": target, "repeat": r, "fold": f, "n_train": int(len(tr)), "n_test": int(len(te)), "outer_r2": r2, "best_inner_cv_r2": inner_r2,
                                "best_params": params, "committed_frozen_outer_r2": None if ref is None else ref["outer_r2"], "tuning_seconds": tune_secs,
                                "fit_seconds": time.perf_counter() - t0}, {"y_true": y[te], "y_pred": pred})
                print(f"{uid}: nested R2 {r2:.4f} (frozen {None if ref is None else round(ref['outer_r2'], 4)}), inner {inner_r2:.4f}, tuning {tune_secs:.0f}s", flush=True)
                del X_tr, y_tr
            if stop:
                break
        if stop:
            break

    results = None
    if all(sess.done(f"rung_{t}_r{r}_f{f}") for t in targets for r in range(args.n_repeats) for f in range(args.n_folds)):
        results = {}
        for target in targets:
            per_repeat, committed = [], []
            if not args.smoke:
                lad = json.loads((H.repo_root() / LADDER_METRICS).read_text(encoding="utf-8"))["runs"][f"{target}_chemistry_full"]
                committed = lad["per_repeat_r2"]
            for r in range(args.n_repeats):
                ys, ps = [], []
                for f in range(args.n_folds):
                    _, a = sess.load(f"rung_{target}_r{r}_f{f}")
                    ys.append(a["y_true"]); ps.append(a["y_pred"])
                per_repeat.append(H.r2(np.concatenate(ys), np.concatenate(ps)))
            inner = [sess.load(f"rung_{target}_r{r}_f{f}")[0]["best_inner_cv_r2"] for r in range(args.n_repeats) for f in range(args.n_folds)]
            results[target] = {"nested_per_repeat_r2": per_repeat, "nested_mean": float(np.mean(per_repeat)), "nested_sd": float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else None,
                               "frozen_per_repeat_r2": committed, "frozen_minus_nested_per_repeat": [c - n for c, n in zip(committed, per_repeat)],
                               "mean_best_inner_cv_r2": float(np.mean(inner))}
        print(json.dumps(results, indent=1), flush=True)
    sess.finish(units_total, results)


if __name__ == "__main__":
    main()
