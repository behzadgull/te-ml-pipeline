"""
NA7: direct versus component-wise (derived) zT under RANDOM (row-level) validation, to test the thesis statement that the advantage of direct
prediction is visible only under chemistry-grouped validation.

Design: exactly the committed per-target run (results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda) except for the split. Same
56,088-row subset (S, sigma, kappa and zT all present), same four XGBoost models (direct zT, S, log10 sigma, log10 kappa) each with its own
target's frozen hyperparameters, same 5 repeats x 5 folds and seed 0, same derived zT = S^2 sigma T / kappa. The folds are shuffled row-level
5-fold CV (src.nested_cv.outer_splits "kfold"), so the same chemistry cluster can sit in training and test, as in the random rung of the ladder.
Everything is reported next to the committed grouped result so the comparison is one table.

Units: one per (repeat, fold); each holds the four models' held-out predictions. Smoke mode (--smoke) uses about 3,000 rows, 1 repeat x 3
folds and 20 trees; its numbers mean nothing.

Usage (repository root):
    python thesis_paper/scripts/kaggle/na7_random_dvd.py --out-dir /kaggle/working/na7 --expect-commit <sha> --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import harness as H  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

FROZEN_DIR = H.repo_root() / "checkpoints" / "saved_predictions" / "checkpoints" / "frozen_hyperparams"
COMPONENTS = {"zT_direct": "zT", "S": "S", "sigma_log10": "sigma", "kappa_log10": "kappa"}
GROUPED_RESULT = "results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda/results.json"


def main():
    ap = argparse.ArgumentParser()
    H.add_common_args(ap)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default=None, help="'i/n': this process computes the units whose index (repeat-major order) is i modulo n, for splitting the "
                    "work over two GPUs. Not part of the run identity: the shards are merged by a final session with --restore-from a,b")
    args = ap.parse_args()
    if args.smoke:
        args.n_repeats, args.n_folds, args.device = 2, 3, "cpu"
    hp = {key: json.loads((FROZEN_DIR / f"{t}.json").read_text(encoding="utf-8")) for key, t in COMPONENTS.items()}
    params = {k: dict(v["best_params"]) for k, v in hp.items()}
    if args.smoke:
        params = {k: {**v, "n_estimators": 20, "max_depth": min(v["max_depth"], 4)} for k, v in params.items()}
    sess = H.Session("na7_random_dvd", args, {"n_repeats": args.n_repeats, "n_folds": args.n_folds, "seed": args.seed, "smoke": args.smoke,
                                              "hyperparams": params, "split": "kfold_shuffled_rows"})
    df = H.load_frame(sess)
    sub = df.dropna(subset=["S", "sigma", "kappa", "zT"]).reset_index(drop=True)
    if args.smoke:
        sub = H.smoke_subset(sub, ncv.GROUP_COL, 3000)
    elif len(sub) != 56088:
        raise SystemExit(f"the all-four-present subset has {len(sub)} rows, expected 56088")
    cols = ncv.get_feature_columns(sub)
    X = sub[cols].to_numpy(dtype=np.float64)
    T = sub["temperature_bin"].to_numpy(dtype=np.float64)
    y = {"zT_direct": sub["zT"].to_numpy(float), "S": sub["S"].to_numpy(float),
         "sigma_log10": np.log10(sub["sigma"].to_numpy(float)), "kappa_log10": np.log10(sub["kappa"].to_numpy(float))}
    zt_true = sub["zT"].to_numpy(float)
    Xd = ncv._to_device(X, args.device)
    all_units = [(r, f) for r in range(args.n_repeats) for f in range(args.n_folds)]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        assert 0 <= i < n, f"--shard {args.shard}"
        units = [u for k, u in enumerate(all_units) if k % n == i]
    else:
        units = list(all_units)
    rng_master = np.random.default_rng(args.seed)
    splits = {}
    for r in range(args.n_repeats):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        splits[r] = list(ncv.outer_splits("kfold", len(sub), {}, args.n_folds, rng))
    for r, f in units:
        uid = f"repeat{r}_fold{f}"
        if sess.done(uid):
            continue
        if sess.out_of_time():
            print("time budget reached: stopping before", uid, flush=True)
            break
        tr, te = splits[r][f]
        arrays = {}
        pred = {}
        for key in COMPONENTS:
            yk = ncv._to_device(y[key][tr], args.device)
            model = ncv._build_xgb_model(params[key], args.device)
            model.fit(Xd[tr] if args.device == "cuda" else X[tr], yk)
            pred[key] = ncv._to_host(model.predict(Xd[te] if args.device == "cuda" else X[te])).astype(float)
            arrays[f"{key}_true"], arrays[f"{key}_pred"] = y[key][te], pred[key]
        derived = ((pred["S"] / 1.0e6) ** 2) * (10.0 ** pred["sigma_log10"]) * T[te] / (10.0 ** pred["kappa_log10"])
        arrays["zT_derived_true"], arrays["zT_derived_pred"] = zt_true[te], derived
        sess.save(uid, {"repeat": r, "fold": f, "n_train": int(len(tr)), "n_test": int(len(te)),
                        "fold_r2": {k: H.r2(arrays[f"{k}_true"], arrays[f"{k}_pred"]) for k in list(COMPONENTS) + ["zT_derived"]}}, arrays)
        print(f"{uid}: direct R2 {H.r2(zt_true[te], pred['zT_direct']):.4f}, derived R2 {H.r2(zt_true[te], derived):.4f}", flush=True)

    results = None
    if all(sess.done(f"repeat{r}_fold{f}") for r, f in all_units):
        units = all_units
        pooled = {k: ([], []) for k in list(COMPONENTS) + ["zT_derived"]}
        for r, f in units:
            _, a = sess.load(f"repeat{r}_fold{f}")
            for k in pooled:
                pooled[k][0].append(a[f"{k}_true"])
                pooled[k][1].append(a[f"{k}_pred"])
        res = {}
        for k, (yt, yp) in pooled.items():
            yt, yp = np.concatenate(yt), np.concatenate(yp)
            res[k] = {"pooled_r2": H.r2(yt, yp), "mae": float(np.mean(np.abs(yt - yp))), "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))), "n": int(len(yt))}
        # Duan smearing of the log10 back-transform, factors from this run's own out-of-fold residuals (as in the grouped run)
        yt_s = np.concatenate(pooled["sigma_log10"][0]); yp_s = np.concatenate(pooled["sigma_log10"][1])
        yt_k = np.concatenate(pooled["kappa_log10"][0]); yp_k = np.concatenate(pooled["kappa_log10"][1])
        res["smear_factors"] = {"sigma": float(np.mean(10.0 ** (yt_s - yp_s))), "kappa": float(np.mean(10.0 ** (yt_k - yp_k)))}
        res["gap_direct_minus_derived"] = res["zT_direct"]["pooled_r2"] - res["zT_derived"]["pooled_r2"]
        res["subset_n_rows"], res["n_repeats"], res["n_folds"] = int(len(sub)), args.n_repeats, args.n_folds
        gp = Path(H.repo_root() / GROUPED_RESULT)
        if gp.exists() and not args.smoke:
            g = json.loads(gp.read_text(encoding="utf-8"))
            res["grouped_run_for_comparison"] = {"path": GROUPED_RESULT, "zT_direct": g["zT_direct"]["pooled_r2"], "zT_derived": g["zT_derived"]["pooled_r2"],
                                                 "gap": g["zT_direct"]["pooled_r2"] - g["zT_derived"]["pooled_r2"]}
        results = res
        print(json.dumps({k: res[k] for k in ("zT_direct", "zT_derived", "gap_direct_minus_derived")}, indent=1), flush=True)
    sess.finish(len(units), results)


if __name__ == "__main__":
    main()
