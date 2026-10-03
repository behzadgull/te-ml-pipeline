"""
NA3: SHAP attribution under chemistry-cluster CV for S, sigma, kappa and zT (GPU).

For each target and each of the 5 repeats x 5 folds (same fold assignment as the committed XGBoost rung): fit the target's frozen XGBoost
model on the training fold, compute exact TreeSHAP (XGBoost's native pred_contribs, bias column dropped) on a fixed seeded random subsample of
the test fold (20,000 rows, as in scripts/shap_attribution_zT.py; the whole fold if smaller), and keep per-feature mean |SHAP|, its share of the
total, and gain importance. One unit per (target, repeat, fold), with the repeat and fold in the key and the meta (the committed Paper A npz
for zT lost the repeat). Each fold's own R2 is recorded next to the committed value as a sanity check (a different GPU may differ slightly).

Results: per target, mean |SHAP| over the 25 folds with fold SD, the top 20 features, the top-5 table, and the share of the total in MAGPIE, CBFV
and temperature features. sigma and kappa are in log10 units, S in microV/K, zT unitless.

Smoke mode: ~3,000 rows, 1 repeat x 3 folds, 20 trees, 300-row SHAP subsample.

Usage:
    python thesis_paper/scripts/kaggle/na3_shap.py --out-dir /kaggle/working/na3 --expect-commit <sha> --device cuda --time-budget-hours 10.5
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
SHAP_ROWS = 20000


def group_of(col):
    if col == ncv.TEMPERATURE_COL:
        return "temperature"
    return "magpie" if col.startswith("MagpieData") else "cbfv"


def main():
    ap = argparse.ArgumentParser()
    H.add_common_args(ap)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    shap_rows = SHAP_ROWS
    if args.smoke:
        args.n_repeats, args.n_folds, args.device, shap_rows = 1, 3, "cpu", 300
    targets = [t for t in args.targets.split(",") if t]
    hp = {t: json.loads((FROZEN_DIR / f"{t}.json").read_text(encoding="utf-8"))["best_params"] for t in targets}
    if args.smoke:
        hp = {t: {**p, "n_estimators": 20, "max_depth": min(p["max_depth"], 4)} for t, p in hp.items()}
    sess = H.Session("na3_shap", args, {"targets": targets, "n_repeats": args.n_repeats, "n_folds": args.n_folds, "seed": args.seed, "smoke": args.smoke,
                                        "shap_rows": shap_rows, "hyperparams": hp, "arm": "chemistry"})
    df_all = H.load_frame(sess)
    units_total = len(targets) * args.n_repeats * args.n_folds
    stop = False
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
        for r in range(args.n_repeats):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            for f, (tr, te) in enumerate(ncv.outer_splits("chemistry", len(df), {"chemistry": groups}, args.n_folds, rng)):
                uid = f"{target}_repeat{r}_fold{f}"
                if sess.done(uid):
                    continue
                if sess.out_of_time():
                    print("time budget reached before", uid, flush=True)
                    stop = True
                    break
                t0 = time.perf_counter()
                model = ncv._build_xgb_model(hp[target], args.device)
                model.fit(Xd[tr], yd[tr])
                pred = ncv._to_host(model.predict(Xd[te])).astype(float)
                fold_r2 = H.r2(y[te], pred)
                sub_rng = np.random.default_rng(args.seed)
                pos = np.sort(sub_rng.choice(len(te), size=min(shap_rows, len(te)), replace=False))
                booster = model.get_booster()
                contribs = booster.predict(xgb.DMatrix(X[te][pos]), pred_contribs=True)[:, :-1]
                mean_abs = np.abs(contribs).mean(axis=0)
                gain = booster.get_score(importance_type="gain")
                gain_arr = np.array([gain.get(f"f{i}", 0.0) for i in range(len(cols))])
                committed = None
                cp = LADDER_DIR / f"{target}_chemistry_full" / f"repeat{r}_fold{f}.json"
                if cp.exists() and not args.smoke:
                    committed = json.loads(cp.read_text(encoding="utf-8"))["outer_r2"]
                sess.save(uid, {"target": target, "repeat": r, "fold": f, "n_train": int(len(tr)), "n_test": int(len(te)), "n_shap_rows": int(len(pos)),
                                "fold_r2": fold_r2, "committed_fold_r2": committed, "seconds": time.perf_counter() - t0},
                          {"mean_abs_shap": mean_abs, "shap_share": mean_abs / mean_abs.sum(), "gain": gain_arr, "gain_share": gain_arr / max(gain_arr.sum(), 1e-12),
                           "shap_subsample_positions": pos, "test_idx": te.astype(np.int64)})
                print(f"{uid}: R2 {fold_r2:.4f} (committed {committed}) {time.perf_counter() - t0:.0f}s", flush=True)
            if stop:
                break
        if stop:
            break

    results = None
    if all(sess.done(f"{t}_repeat{r}_fold{f}") for t in targets for r in range(args.n_repeats) for f in range(args.n_folds)):
        results = {"units": units_total, "feature_columns": cols, "per_target": {}}
        for target in targets:
            arrs = [sess.load(f"{target}_repeat{r}_fold{f}")[1]["mean_abs_shap"] for r in range(args.n_repeats) for f in range(args.n_folds)]
            A = np.array(arrs)
            mean, sd = A.mean(axis=0), A.std(axis=0, ddof=1) if len(A) > 1 else np.zeros(A.shape[1])
            order = np.argsort(-mean)
            share_by_group = {}
            for g in ("magpie", "cbfv", "temperature"):
                idx = [i for i, c in enumerate(cols) if group_of(c) == g]
                per_fold = A[:, idx].sum(axis=1) / A.sum(axis=1)
                share_by_group[g] = {"mean": float(per_fold.mean()), "sd": float(per_fold.std(ddof=1)) if len(per_fold) > 1 else None}
            results["per_target"][target] = {"top20": [{"feature": cols[i], "mean_abs_shap": float(mean[i]), "fold_sd": float(sd[i])} for i in order[:20]],
                                             "share_by_group": share_by_group, "n_folds": int(len(A))}
        print(json.dumps({t: results["per_target"][t]["top20"][:5] for t in targets}, indent=1), flush=True)
    sess.finish(units_total, results)


if __name__ == "__main__":
    main()
