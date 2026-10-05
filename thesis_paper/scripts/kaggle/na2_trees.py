"""
NA2: random forest and LightGBM against the committed XGBoost chemistry-cluster result, on the Paper A data (CPU; run once per --model).

For each target (S, sigma, kappa, zT):
  1. tune: one Optuna search of --n-trials trials scored by 3-fold chemistry-cluster CV on all rows of the target (src.nested_cv's search space and
     objective for the model, exactly what tune_once does for XGBoost), saved in tune_once's JSON format as frozen/<target>_<model>.json. Every
     trial is a unit, so a killed session resumes the same search (the TPE sampler is reseeded with seed + trials done, so a resumed search is valid
     but not bit-identical to an uninterrupted one);
  2. rung: the chemistry-cluster rung, n_repeats x n_folds (5 x 5) outer folds with those frozen hyperparameters, the same fold assignment as the
     XGBoost rung (same seed, same random group k-fold), one unit per fold holding y_true and y_pred (the format of the committed Paper A
     predictions), so stacking can be built from them (na2_stacking.py, run locally where the committed XGBoost predictions are).
Targets sigma and kappa are trained and scored in log10 space, as in Paper A. The first full-size unit of the rung is checked against the committed
XGBoost fold sizes (n_test must match), which proves the fold assignment is the same.

--fixed-hyperparams (random_forest only; design change recorded in docs/decisions.md, 2026-10-05): no tuning. Every target uses the one pre-stated set RF_FIXED (the
software defaults for regression of Probst, Wright & Boulesteix 2019: a third of the features per split, node size 5, 500 trees, bootstrap sampling); step 1 is skipped and
each target has only its 25 rung units. The set is part of the run's identity.

Smoke mode: ~3,000 rows, 2 trials, 1 repeat x 3 folds (with --fixed-hyperparams: 20 trees).

Usage:
    python thesis_paper/scripts/kaggle/na2_trees.py --model random_forest --out-dir /kaggle/working/na2_rf --expect-commit <sha> --time-budget-hours 10.5
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
RF_FIXED = {"n_estimators": 500, "max_features": 1 / 3, "min_samples_leaf": 5, "max_depth": None, "min_samples_split": 2, "bootstrap": True}
TARGETS = ("S", "sigma", "kappa", "zT")
TARGET_ROWS = {"S": 185064, "sigma": 182755, "kappa": 121110, "zT": 129419}


def space_distributions(space):
    """The Optuna distributions of a search space, read off a throw-away trial."""
    t = optuna.create_study().ask()
    space(t)
    return dict(t.distributions)


def tune_target(sess, model, target, X, y, groups, args):
    """Run (or resume) the tuning trials of one target; returns the frozen-hyperparameter dict or None if out of time."""
    space = ncv.MODEL_REGISTRY[model]["search_space"]
    dists = space_distributions(space)
    prefix = f"tune_{target}_"
    done_units = sorted(u for u in sess._all_units() if u.startswith(prefix) and u[len(prefix):].startswith("trial"))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed + len(done_units)),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    for u in done_units:
        meta, _ = sess.load(u)
        if meta["state"] == "complete":
            study.add_trial(optuna.trial.create_trial(params=meta["params"], distributions=dists, value=meta["value"]))
        else:
            study.add_trial(optuna.trial.create_trial(params=meta["params"], distributions=dists, state=optuna.trial.TrialState.PRUNED, value=None))
    n_done = len(done_units)
    while n_done < args.n_trials:
        if sess.out_of_time():
            print(f"time budget reached during tuning of {target} after {n_done} trials", flush=True)
            return None
        trial = study.ask()
        t0 = time.perf_counter()
        try:
            value = ncv._objective(trial, model, X, y, groups, args.n_inner_folds, device="cpu")
            study.tell(trial, value)
            meta = {"state": "complete", "value": float(value)}
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            meta = {"state": "pruned", "value": None}
        meta.update({"params": dict(trial.params), "seconds": time.perf_counter() - t0, "trial_number": n_done})
        sess.save(f"{prefix}trial{n_done:03d}", meta)
        n_done += 1
        print(f"{target} trial {n_done}/{args.n_trials}: {meta['state']} value {meta['value']} ({meta['seconds']:.0f}s)", flush=True)
    complete = [u for u in done_units + [f"{prefix}trial{k:03d}" for k in range(len(done_units), n_done)] if sess.load(u)[0]["state"] == "complete"]
    best_u = max(complete, key=lambda u: sess.load(u)[0]["value"])
    bm, _ = sess.load(best_u)
    frozen = {"target": target, "target_scale": ncv._target_scale(target), "model_type": model, "n_trials": args.n_trials, "n_inner_folds": args.n_inner_folds,
              "seed": args.seed, "device": "cpu", "n_rows": int(len(y)), "n_features": int(X.shape[1]), "inner_cv_r2": bm["value"], "best_params": bm["params"]}
    fdir = sess.out / "frozen"
    fdir.mkdir(exist_ok=True)
    (fdir / f"{target}_{model}.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    sess.save(f"{prefix}best", frozen)
    return frozen


def main():
    ap = argparse.ArgumentParser()
    H.add_common_args(ap)
    ap.add_argument("--model", required=True, choices=["random_forest", "lightgbm"])
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--n-trials", type=int, default=ncv.N_OPTUNA_TRIALS)
    ap.add_argument("--n-inner-folds", type=int, default=ncv.N_INNER_FOLDS)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-hyperparams", action="store_true", help="random_forest only: no tuning, use RF_FIXED for every target")
    args = ap.parse_args()
    if args.fixed_hyperparams and args.model != "random_forest":
        raise SystemExit("--fixed-hyperparams is defined for random_forest only (LightGBM keeps its tuning)")
    if args.smoke:
        args.n_trials, args.n_repeats, args.n_folds, args.device = 2, 1, 3, "cpu"
    fixed = dict(RF_FIXED, n_estimators=20) if (args.fixed_hyperparams and args.smoke) else (dict(RF_FIXED) if args.fixed_hyperparams else None)
    targets = [t for t in args.targets.split(",") if t]
    sess = H.Session(f"na2_{args.model}", args, {"model": args.model, "targets": targets, "n_trials": args.n_trials, "n_inner_folds": args.n_inner_folds,
                                                 "n_repeats": args.n_repeats, "n_folds": args.n_folds, "seed": args.seed, "smoke": args.smoke,
                                                 "fixed_hyperparams": fixed})
    df_all = H.load_frame(sess)
    units_total = 0
    for target in targets:
        units_total += (0 if fixed else args.n_trials + 1) + args.n_repeats * args.n_folds
        df = df_all[df_all[target].notna()].reset_index(drop=True)
        if args.smoke:
            df = H.smoke_subset(df, ncv.GROUP_COL, 3000)
        elif len(df) != TARGET_ROWS[target]:
            raise SystemExit(f"{target} has {len(df)} rows, expected {TARGET_ROWS[target]}")
        cols = ncv.get_feature_columns(df)
        X = df[cols].to_numpy(dtype=np.float64)
        y = ncv._transform_target(df[target].to_numpy(dtype=np.float64), target)
        groups = df[ncv.GROUP_COL].to_numpy()
        frozen_path = sess.out / "frozen" / f"{target}_{args.model}.json"
        if fixed:
            frozen = {"target": target, "target_scale": ncv._target_scale(target), "model_type": args.model, "tuned": False, "n_trials": 0, "seed": args.seed, "device": "cpu",
                      "n_rows": int(len(y)), "n_features": int(X.shape[1]), "inner_cv_r2": None, "best_params": fixed}
            frozen_path.parent.mkdir(exist_ok=True)
            frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        elif frozen_path.exists() and sess.done(f"tune_{target}_best"):
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        else:
            frozen = tune_target(sess, args.model, target, X, y, groups, args)
            if frozen is None:
                break
        build = ncv.MODEL_REGISTRY[args.model]["build"]
        rng_master = np.random.default_rng(args.seed)
        stop = False
        for r in range(args.n_repeats):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            for f, (tr, te) in enumerate(ncv.outer_splits("chemistry", len(df), {"chemistry": groups}, args.n_folds, rng)):
                uid = f"rung_{target}_repeat{r}_fold{f}"
                if sess.done(uid):
                    continue
                if sess.out_of_time():
                    print("time budget reached before", uid, flush=True)
                    stop = True
                    break
                if r == 0 and f == 0 and not args.smoke:
                    ref = json.loads((H.repo_root() / LADDER_DIR / f"{target}_chemistry_full" / "repeat0_fold0.json").read_text(encoding="utf-8"))
                    assert len(te) == ref["n_test"] and len(tr) == ref["n_train"], "fold assignment differs from the committed XGBoost rung"
                t0 = time.perf_counter()
                model = build(frozen["best_params"], "cpu")
                model.fit(X[tr], y[tr])
                pred = np.asarray(model.predict(X[te]), dtype=float)
                sess.save(uid, {"repeat": r, "fold": f, "n_train": int(len(tr)), "n_test": int(len(te)), "outer_r2": H.r2(y[te], pred), "seconds": time.perf_counter() - t0},
                          {"y_true": y[te], "y_pred": pred})
                print(f"{uid}: R2 {H.r2(y[te], pred):.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
            if stop:
                break
        if stop:
            break

    results = None
    complete = len([u for u in sess._all_units()]) >= units_total
    if complete:
        results = {}
        for target in targets:
            per_repeat = []
            for r in range(args.n_repeats):
                ys, ps = [], []
                for f in range(args.n_folds):
                    _, a = sess.load(f"rung_{target}_repeat{r}_fold{f}")
                    ys.append(a["y_true"]); ps.append(a["y_pred"])
                per_repeat.append(H.r2(np.concatenate(ys), np.concatenate(ps)))
            results[target] = {"per_repeat_r2": per_repeat, "mean": float(np.mean(per_repeat)), "sd": float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else None,
                               "inner_cv_r2": json.loads((sess.out / "frozen" / f"{target}_{args.model}.json").read_text(encoding="utf-8"))["inner_cv_r2"]}
        print(json.dumps(results, indent=1), flush=True)
    sess.finish(units_total, results)


if __name__ == "__main__":
    main()
