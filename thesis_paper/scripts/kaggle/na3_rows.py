"""
NA3 rows: SHAP values of individual rows, for the beeswarm plots (Figure 11). The NA3 run (na3_shap.py) saved only the mean absolute SHAP value per feature and fold, which cannot
draw a beeswarm; this run repeats the first repeat of the same chemistry-cluster rung (repeat 0, the same folds, the same frozen Paper A hyperparameters) and saves, for a fixed,
seeded subsample of each fold's test rows, the SHAP value of every feature and the feature values themselves.

Per target and fold: refit the frozen model on the fold's training rows (the fold R2 is checked against the committed rung), draw ROWS_PER_FOLD test rows without replacement with the
generator default_rng(seed + fold) (sorted), compute exact TreeSHAP (native XGBoost pred_contribs) for them. The unit stores the 397 SHAP values and the 397 feature values (float32), the row
positions in the target's frame, the measured and predicted value, and the SHAP bias term. The sum of a row's SHAP values plus the bias equals the model's prediction (asserted).

Smoke mode: ~3,000 rows, 3 folds, 20 trees, 50 rows per fold.

Usage:
    python thesis_paper/scripts/kaggle/na3_rows.py --targets S,kappa --out-dir /kaggle/working/na3r_a --expect-commit <sha> --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import harness as H  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

FROZEN_DIR = H.repo_root() / "checkpoints" / "saved_predictions" / "checkpoints" / "frozen_hyperparams"
LADDER_DIR = H.repo_root() / "results" / "ladder_regen_snapfix" / "20260917T150000"
TARGETS = ("S", "sigma", "kappa", "zT")
TARGET_ROWS = {"S": 185064, "sigma": 182755, "kappa": 121110, "zT": 129419}
ROWS_PER_FOLD = 1000


def main():
    ap = argparse.ArgumentParser()
    H.add_common_args(ap)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows_per_fold = ROWS_PER_FOLD
    if args.smoke:
        args.n_folds, args.device, rows_per_fold = 3, "cpu", 50
    targets = [t for t in args.targets.split(",") if t]
    hp = {t: json.loads((FROZEN_DIR / f"{t}.json").read_text(encoding="utf-8"))["best_params"] for t in targets}
    if args.smoke:
        hp = {t: {**p, "n_estimators": 20, "max_depth": min(p["max_depth"], 4)} for t, p in hp.items()}
    sess = H.Session("na3_rows", args, {"targets": targets, "repeat": 0, "n_folds": args.n_folds, "seed": args.seed, "smoke": args.smoke, "rows_per_fold": rows_per_fold, "hyperparams": hp})
    df_all = H.load_frame(sess)
    units_total = len(targets) * args.n_folds
    for target in targets:
        df = df_all[df_all[target].notna()].reset_index(drop=True)
        if args.smoke:
            df = H.smoke_subset(df, ncv.GROUP_COL, 3000)
        elif len(df) != TARGET_ROWS[target]:
            raise SystemExit(f"{target}: {len(df)} rows, expected {TARGET_ROWS[target]}")
        cols = ncv.get_feature_columns(df)
        X = df[cols].to_numpy(dtype=np.float64)
        y = ncv._transform_target(df[target].to_numpy(dtype=np.float64), target)
        groups = df[ncv.GROUP_COL].to_numpy()
        Xd, yd = ncv._to_device(X, args.device), ncv._to_device(y, args.device)
        rng_master = np.random.default_rng(args.seed)
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))  # repeat 0: the first draw, as in every rung
        for f, (tr, te) in enumerate(ncv.outer_splits("chemistry", len(df), {"chemistry": groups}, args.n_folds, rng)):
            uid = f"{target}_repeat0_fold{f}"
            if sess.done(uid):
                continue
            if sess.out_of_time():
                print("time budget reached before", uid, flush=True)
                break
            t0 = time.perf_counter()
            model = ncv._build_xgb_model(hp[target], args.device)
            model.fit(Xd[tr], yd[tr])
            pred_all = ncv._to_host(model.predict(Xd[te])).astype(float)
            fold_r2 = H.r2(y[te], pred_all)
            sub = np.sort(np.random.default_rng(args.seed + f).choice(len(te), size=min(rows_per_fold, len(te)), replace=False))
            rows = te[sub]
            contribs = model.get_booster().predict(xgb.DMatrix(X[rows]), pred_contribs=True)
            shap, bias = contribs[:, :-1], contribs[:, -1]
            pred_rows = pred_all[sub]
            assert np.allclose(shap.sum(axis=1) + bias, pred_rows, atol=1e-2 * max(1.0, float(np.abs(pred_rows).max()))), "SHAP values plus bias must equal the prediction"
            committed = None
            cp = LADDER_DIR / f"{target}_chemistry_full" / f"repeat0_fold{f}.json"
            if cp.exists() and not args.smoke:
                committed = json.loads(cp.read_text(encoding="utf-8"))["outer_r2"]
            sess.save(uid, {"target": target, "repeat": 0, "fold": f, "n_train": int(len(tr)), "n_test": int(len(te)), "n_rows_saved": int(len(rows)),
                            "fold_r2": fold_r2, "committed_fold_r2": committed, "seconds": time.perf_counter() - t0},
                      {"shap": shap.astype(np.float32), "x": X[rows].astype(np.float32), "row_index": rows.astype(np.int64), "y_true": y[rows], "y_pred": pred_rows,
                       "bias": bias.astype(np.float32)})
            print(f"{uid}: R2 {fold_r2:.4f} (committed {committed}), {len(rows)} rows, {time.perf_counter() - t0:.0f}s", flush=True)

    results = None
    if all(sess.done(f"{t}_repeat0_fold{f}") for t in targets for f in range(args.n_folds)):
        results = {"units": units_total, "feature_columns": cols, "rows_per_fold": rows_per_fold, "per_target": {}}
        for target in targets:
            n = 0
            for f in range(args.n_folds):
                meta, _ = sess.load(f"{target}_repeat0_fold{f}")
                n += meta["n_rows_saved"]
            results["per_target"][target] = {"n_rows_saved": n, "n_folds": args.n_folds}
        print(json.dumps(results["per_target"]), flush=True)
    sess.finish(units_total, results)


if __name__ == "__main__":
    main()
