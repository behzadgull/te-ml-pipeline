"""
NA2, nested stacking: the meta-learner of a stack of XGBoost, LightGBM and the random forest, trained on INNER out-of-fold base predictions only (CPU).

For each target and each outer fold (repeat r, fold f) of the chemistry-cluster rung (the same folds as Paper A: same seed, same randomized group k-fold; the fold sizes are checked against
the committed XGBoost run):
  1. inside the outer TRAINING rows, split by chemistry cluster into --n-inner-folds inner folds (GroupKFold, as the tuning does: src.nested_cv._objective);
  2. for each of the three base models (XGBoost with the frozen Paper A hyperparameters of the target, LightGBM with the frozen tuned set of results/na2_lightgbm, the random forest with the one
     fixed set of docs/decisions.md, 2026-10-05) fit on the inner training rows and predict the inner validation rows, so that every outer training row gets one prediction from a model that never
     saw it (its own inner fold) and, since the outer test rows are not in X_tr at all, from a model that never saw any outer test row either;
  3. fit the meta-learner (sklearn Ridge, alpha 1, positive=True, intercept, as in na2_stacking.py) on those inner out-of-fold predictions against the outer training labels;
  4. save its coefficients and intercept as the unit. The stack's prediction of the outer test rows is made locally by applying them to the base models' committed outer-fold predictions (which are
     fits on the full outer training rows with the same hyperparameters), by scripts/na2_stacking_analysis.py; the outer test labels never enter this script's fits.
The leak-freeness is asserted in the code (assert_nested): outer train and test rows and clusters are disjoint; the inner validation folds partition the outer training rows exactly once, with
disjoint rows and clusters between each inner training set and its validation fold; the meta-learner is fitted on exactly len(outer train) rows built only from inner validation predictions.
Targets sigma and kappa are trained and scored in log10 space, as in Paper A. XGBoost runs on the CPU here (the Paper A outer fits ran on a GPU); that only changes the inner predictions
the weights are learned from, not the outer predictions.

Smoke mode: ~3,000 rows, 1 repeat x 3 folds, 20 trees for the forest.

Usage:
    python thesis_paper/scripts/kaggle/na2_stacking_nested.py --targets S --repeats 0,1 --out-dir /kaggle/working/na2_stk_S_a --expect-commit <sha> --time-budget-hours 11
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import harness as H  # noqa: E402
from na2_trees import LADDER_DIR, RF_FIXED, TARGET_ROWS, TARGETS  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

XGB_FROZEN = "checkpoints/saved_predictions/checkpoints/frozen_hyperparams/{t}.json"
LGBM_FROZEN = "thesis_paper/results/na2_lightgbm/20261005T101837/frozen/{t}_lightgbm.json"
BASES = ("xgboost", "lightgbm", "random_forest")


def assert_nested(tr, te, groups, inner, oof_rows):
    """The leak-freeness of the meta-learner's training data, as assertions (see the module docstring)."""
    assert len(np.intersect1d(tr, te)) == 0, "outer train and test rows overlap"
    assert len(np.intersect1d(groups[tr], groups[te])) == 0, "an outer test cluster is in the outer training rows"
    seen = np.zeros(len(tr), dtype=int)
    for itr, iva in inner:
        assert len(np.intersect1d(itr, iva)) == 0, "an inner training row is in its own validation fold"
        assert len(np.intersect1d(groups[tr][itr], groups[tr][iva])) == 0, "an inner validation cluster is in its own inner training set"
        assert itr.max() < len(tr) and iva.max() < len(tr), "an inner index points outside the outer training rows"
        seen[iva] += 1
    assert (seen == 1).all(), "the inner validation folds do not partition the outer training rows exactly once"
    assert oof_rows == len(tr), "the meta-learner is not fitted on exactly the outer training rows"


def main():
    ap = argparse.ArgumentParser()
    H.add_common_args(ap)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--repeats", default="0,1,2,3,4", help="comma-separated repeats this session computes")
    ap.add_argument("--n-inner-folds", type=int, default=ncv.N_INNER_FOLDS)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=5, help="repeats of the whole design (the outer-fold RNG is advanced through all of them)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.n_repeats, args.n_folds, args.device, args.repeats = 1, 3, "cpu", "0"
    targets = [t for t in args.targets.split(",") if t]
    repeats = [int(x) for x in args.repeats.split(",") if x != ""]
    assert all(0 <= r < args.n_repeats for r in repeats)
    rf_params = dict(RF_FIXED, n_estimators=20) if args.smoke else dict(RF_FIXED)
    sess = H.Session("na2_stacking_nested", args, {"targets": targets, "repeats": repeats, "n_inner_folds": args.n_inner_folds, "n_repeats": args.n_repeats, "n_folds": args.n_folds,
                                                   "seed": args.seed, "smoke": args.smoke, "rf_params": rf_params, "meta": {"alpha": 1.0, "positive": True}})
    df_all = H.load_frame(sess)
    units_total = len(targets) * len(repeats) * args.n_folds
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
        params = {"xgboost": ncv._load_frozen_hyperparams(H.repo_root() / XGB_FROZEN.format(t=target), "xgboost")[0],
                  "lightgbm": ncv._load_frozen_hyperparams(H.repo_root() / LGBM_FROZEN.format(t=target), "lightgbm")[0], "random_forest": rf_params}
        build = {m: ncv.MODEL_REGISTRY[m]["build"] for m in BASES}
        rng_master = np.random.default_rng(args.seed)
        for r in range(args.n_repeats):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))  # advanced for every repeat, so that repeat r has the committed fold assignment
            if r not in repeats:
                continue
            for f, (tr, te) in enumerate(ncv.outer_splits("chemistry", len(df), {"chemistry": groups}, args.n_folds, rng)):
                uid = f"stack_{target}_repeat{r}_fold{f}"
                if sess.done(uid):
                    continue
                if sess.out_of_time():
                    print("time budget reached before", uid, flush=True)
                    stop = True
                    break
                if not args.smoke:
                    ref = json.loads((H.repo_root() / LADDER_DIR / f"{target}_chemistry_full" / f"repeat{r}_fold{f}.json").read_text(encoding="utf-8"))
                    assert len(te) == ref["n_test"] and len(tr) == ref["n_train"], "fold assignment differs from the committed XGBoost rung"
                t0 = time.perf_counter()
                X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
                inner = list(GroupKFold(n_splits=args.n_inner_folds).split(np.empty(len(tr)), groups=g_tr))
                oof = np.full((len(tr), len(BASES)), np.nan)
                secs = {}
                for j, m in enumerate(BASES):
                    t1 = time.perf_counter()
                    for itr, iva in inner:
                        model = build[m](params[m], "cpu")
                        model.fit(X_tr[itr], y_tr[itr])
                        oof[iva, j] = np.asarray(model.predict(X_tr[iva]), dtype=float)
                    secs[m] = time.perf_counter() - t1
                assert not np.isnan(oof).any(), "an outer training row has no inner out-of-fold prediction"
                assert_nested(tr, te, groups, inner, oof.shape[0])
                meta = Ridge(alpha=1.0, positive=True).fit(oof, y_tr)
                inner_r2 = {m: H.r2(y_tr, oof[:, j]) for j, m in enumerate(BASES)}
                sess.save(uid, {"target": target, "repeat": r, "fold": f, "n_train": int(len(tr)), "n_test": int(len(te)), "coef_xgboost_lightgbm_random_forest": list(map(float, meta.coef_)),
                                "intercept": float(meta.intercept_), "inner_oof_r2": inner_r2, "meta_in_sample_r2": H.r2(y_tr, meta.predict(oof)), "mean3_inner_oof_r2": H.r2(y_tr, oof.mean(axis=1)),
                                "base_seconds": secs, "seconds": time.perf_counter() - t0, "nested_assertions": "passed"})
                print(f"{uid}: weights {np.round(meta.coef_, 3).tolist()} intercept {meta.intercept_:.3f}, inner OOF R2 {({m: round(v, 4) for m, v in inner_r2.items()})} ({time.perf_counter() - t0:.0f}s)", flush=True)
            if stop:
                break
        if stop:
            break
    results = None
    if all(sess.done(f"stack_{t}_repeat{r}_fold{f}") for t in targets for r in repeats for f in range(args.n_folds)):
        results = {t: {"mean_weights_xgboost_lightgbm_random_forest": np.mean([sess.load(f"stack_{t}_repeat{r}_fold{f}")[0]["coef_xgboost_lightgbm_random_forest"]
                                                                                for r in repeats for f in range(args.n_folds)], axis=0).tolist()} for t in targets}
        print(json.dumps(results, indent=1), flush=True)
    sess.finish(units_total, results)


if __name__ == "__main__":
    main()
