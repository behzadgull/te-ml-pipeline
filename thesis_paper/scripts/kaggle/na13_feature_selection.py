"""
NA13: the thesis's explicit feature-selection pipeline against the full 397-feature set, under chemistry-cluster CV (selection on the CPU, the XGBoost fits on --device).

The thesis describes a three-step selection (Pearson filtering, LassoCV, mutual-information ranking) that "reduced the features to 25-44 per target"
and reports it as worse than using all features (Table 3). Its code is not in the repository, and its selection was probably made on all data. Here
the selection is performed INSIDE each outer training fold, so that the comparison is leak-free:
  1. Pearson filter: among the training fold's features, drop one of each pair with |r| > 0.95 (the later column of the pair);
  2. LassoCV on the surviving, standardised features (3-fold chemistry-cluster inner folds, 20 alphas), keep the non-zero coefficients;
  3. mutual-information ranking (sklearn mutual_info_regression, 20,000-row subsample, seed 0) of the survivors; keep the top k, where k is the
     number of features the thesis reports for the target (S 25, sigma 44, kappa 39, zT 32) or all survivors if fewer.
The model is the target's frozen XGBoost (unchanged hyperparameters, as in the ablation). Paired design (docs/decisions.md, pre-registration 2026-10-09): in every unit the SAME
frozen XGBoost is fitted twice on the same outer training rows, on all 397 features and on the selected columns, on the same machine and the same --device, and both are scored on the same
outer test rows; the unit records r2_all397, r2_selected and their difference, and both sets of predictions. The same 5 x 5 folds as the rest of the work, one unit per (target, repeat, fold),
with the selected feature names stored. Results: pooled per-repeat R2 of both fits and their paired difference (mean and SD over the repeats), the per-fold difference with its spread, the
committed 397-feature rung value reported alongside (it ran on cuda on Kaggle; it is not used for the claim), and how often each feature was selected.
The k per target are the thesis's own counts (from its Table 3); they are constants here, recorded in the run's parameters.

Smoke mode: ~3,000 rows, 1 repeat x 3 folds, 20 trees, 3 alphas, device forced to cpu.

Usage:
    python thesis_paper/scripts/kaggle/na13_feature_selection.py --targets S --device cuda --out-dir ~/runs/na13_S --expect-commit <sha>
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import harness as H  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

FROZEN_DIR = H.repo_root() / "checkpoints" / "saved_predictions" / "checkpoints" / "frozen_hyperparams"
LADDER = H.repo_root() / "reports" / "regen_snapfix" / "20260917T150000" / "ladder_metrics.json"
TARGETS = ("S", "sigma", "kappa", "zT")
TARGET_ROWS = {"S": 185064, "sigma": 182755, "kappa": 121110, "zT": 129419}
K_THESIS = {"S": 25, "sigma": 44, "kappa": 39, "zT": 32}  # features kept by the thesis's selection (its Table 3)
PEARSON_MAX = 0.95
MI_ROWS = 20000
LASSO_MAX_ITER = 20000  # the smallest max_iter of the pre-registered ladder at which no fold warned (sklearn default 2000 did not converge); docs/decisions.md 2026-10-07, Result 2026-10-08


def pearson_keep(Xtr):
    """Boolean mask: greedily drop the later column of every pair with |r| above PEARSON_MAX (constant columns are dropped)."""
    sd = Xtr.std(axis=0)
    ok = sd > 0
    Z = np.zeros_like(Xtr)
    Z[:, ok] = (Xtr[:, ok] - Xtr[:, ok].mean(axis=0)) / sd[ok]
    corr = np.abs(Z.T @ Z / len(Z))
    keep = ok.copy()
    for i in range(corr.shape[0]):
        if not keep[i]:
            continue
        dup = np.where((corr[i, i + 1:] > PEARSON_MAX) & keep[i + 1:])[0] + i + 1
        keep[dup] = False
    return keep


def select(Xtr, ytr, groups_tr, k, n_alphas, seed=0, max_iter=LASSO_MAX_ITER):
    """
    Indices of the selected columns (steps 1 to 3) and a dict of what happened: survivors of the Pearson filter, non-zero Lasso coefficients (the data-driven size, before k is imposed), the chosen alpha,
    and the number of ConvergenceWarnings raised by the LassoCV fits (the inner path fits and the final refit). Everything is computed from the rows passed in, which are the outer training rows only.
    """
    keep1 = np.where(pearson_keep(Xtr))[0]
    Xs = StandardScaler().fit_transform(Xtr[:, keep1])
    lasso = LassoCV(cv=list(GroupKFold(n_splits=3).split(Xs, groups=groups_tr)), n_alphas=n_alphas, random_state=seed, n_jobs=-1, max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lasso.fit(Xs, ytr)
    keep2 = keep1[np.abs(lasso.coef_) > 1e-12]
    n_warnings = int(sum(issubclass(c.category, ConvergenceWarning) for c in caught))
    # LassoCV fits one path per inner fold and then refits once at the chosen alpha on all the rows it was given; that refit is the last warning-raising fit and its dual gap is `dual_gap_`.
    # sklearn warns when the gap exceeds tol times the centred sum of squares of y, so the refit's convergence is recorded separately from the inner paths'.
    # (`dual_gap_` is stored divided by the number of rows, as lasso_path returns it; multiplied back it is the number in sklearn's warning message.)
    final_tolerance = float(lasso.tol) * float(np.sum((ytr - ytr.mean()) ** 2))
    final_gap = float(lasso.dual_gap_) * len(ytr)
    final_converged = bool(final_gap <= final_tolerance)
    info = {"n_after_pearson": int(len(keep1)), "n_after_lasso": int(len(keep2)), "lasso_alpha": float(lasso.alpha_), "lasso_alpha_index": int(np.argmin(np.abs(lasso.alphas_ - lasso.alpha_))),
            "lasso_alphas": [float(x) for x in lasso.alphas_],
            "n_convergence_warnings": n_warnings, "n_convergence_warnings_inner_paths": n_warnings - int(not final_converged),
            "final_refit_converged": final_converged, "final_refit_dual_gap": final_gap, "final_refit_tolerance": final_tolerance, "final_refit_n_iter": int(lasso.n_iter_),
            "lasso_max_iter": int(max_iter)}
    if len(keep2) <= k:
        return keep2, info
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(MI_ROWS, len(Xtr)), replace=False)
    mi = mutual_info_regression(Xtr[idx][:, keep2], ytr[idx], random_state=seed)
    return keep2[np.argsort(-mi)[:k]], info


def main():
    ap = argparse.ArgumentParser()
    H.add_common_args(ap)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lasso-max-iter", type=int, default=LASSO_MAX_ITER)
    args = ap.parse_args()
    n_alphas = 20
    if args.smoke:
        args.n_repeats, args.n_folds, args.device, n_alphas = 1, 3, "cpu", 3
    targets = [t for t in args.targets.split(",") if t]
    hp = {t: json.loads((FROZEN_DIR / f"{t}.json").read_text(encoding="utf-8"))["best_params"] for t in targets}
    if args.smoke:
        hp = {t: {**p, "n_estimators": 20, "max_depth": min(p["max_depth"], 4)} for t, p in hp.items()}
    sess = H.Session("na13_feature_selection", args, {"targets": targets, "n_repeats": args.n_repeats, "n_folds": args.n_folds, "seed": args.seed, "smoke": args.smoke,
                                                      "k": K_THESIS, "pearson_max": PEARSON_MAX, "n_alphas": n_alphas, "mi_rows": MI_ROWS, "lasso_max_iter": args.lasso_max_iter, "device": args.device,
                                                      "paired_all397": True, "hyperparams": hp})
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
                sel, info = select(X[tr], y[tr], groups[tr], K_THESIS[target], n_alphas, args.seed, args.lasso_max_iter)
                assert len(sel) > 0, f"{uid}: the selection kept no feature (LassoCV set every coefficient to zero)"
                t1 = time.perf_counter()
                model = ncv._build_xgb_model(hp[target], args.device)
                model.fit(X[tr][:, sel], y[tr])
                pred = np.asarray(model.predict(X[te][:, sel]), dtype=float)
                sec_selected = time.perf_counter() - t1
                t2 = time.perf_counter()
                model_all = ncv._build_xgb_model(hp[target], args.device)  # the paired reference: the same frozen model on all 397 features, same rows, same device
                model_all.fit(X[tr], y[tr])
                pred_all = np.asarray(model_all.predict(X[te]), dtype=float)
                sec_all = time.perf_counter() - t2
                r2_selected, r2_all397 = H.r2(y[te], pred), H.r2(y[te], pred_all)
                sess.save(uid, {"target": target, "repeat": r, "fold": f, "n_train": int(len(tr)), "n_test": int(len(te)), **info, "device": args.device,
                                "n_selected": int(len(sel)), "selected": [cols[i] for i in sel], "outer_r2": r2_selected, "r2_selected": r2_selected, "r2_all397": r2_all397,
                                "delta_r2_selected_minus_all397": r2_selected - r2_all397, "seconds_selected_fit": sec_selected, "seconds_all397_fit": sec_all,
                                "seconds": time.perf_counter() - t0},
                          {"y_true": y[te], "y_pred": pred, "y_pred_all397": pred_all})
                print(f"{uid}: {len(sel)} selected ({info['n_after_pearson']} after Pearson, {info['n_after_lasso']} after Lasso, alpha {info['lasso_alpha']:.3g}, {info['n_convergence_warnings']} convergence warnings, final refit {'converged' if info['final_refit_converged'] else 'NOT converged'}), "
                      f"R2 selected {r2_selected:.4f}, all 397 {r2_all397:.4f}, difference {r2_selected - r2_all397:+.4f} [{args.device}] ({time.perf_counter() - t0:.0f}s)", flush=True)
            if stop:
                break
        if stop:
            break

    results = None
    if all(sess.done(f"{t}_repeat{r}_fold{f}") for t in targets for r in range(args.n_repeats) for f in range(args.n_folds)):
        committed = json.loads(LADDER.read_text(encoding="utf-8"))["runs"] if not args.smoke else {}
        results = {}
        for target in targets:
            per_repeat, per_repeat_all, counts, metas = [], [], {}, []
            for r in range(args.n_repeats):
                ys, ps, pa = [], [], []
                for f in range(args.n_folds):
                    meta, a = sess.load(f"{target}_repeat{r}_fold{f}")
                    ys.append(a["y_true"]); ps.append(a["y_pred"]); pa.append(a["y_pred_all397"])
                    metas.append(meta)
                    for name in meta["selected"]:
                        counts[name] = counts.get(name, 0) + 1
                per_repeat.append(H.r2(np.concatenate(ys), np.concatenate(ps)))
                per_repeat_all.append(H.r2(np.concatenate(ys), np.concatenate(pa)))
            paired = [a - b for a, b in zip(per_repeat, per_repeat_all)]
            full = committed.get(f"{target}_chemistry_full", {})
            n_folds_total = len(metas)

            def spread(key):
                v = np.array([m[key] for m in metas], dtype=float)
                return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "min": float(v.min()), "max": float(v.max())}

            results[target] = {"device": args.device,
                               "selected_per_repeat_r2": per_repeat, "selected_mean": float(np.mean(per_repeat)),
                               "selected_sd": float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else None,
                               "all397_same_machine_per_repeat_r2": per_repeat_all, "all397_same_machine_mean": float(np.mean(per_repeat_all)),
                               "all397_same_machine_sd": float(np.std(per_repeat_all, ddof=1)) if len(per_repeat_all) > 1 else None,
                               "paired_difference_selected_minus_all397_per_repeat": paired, "paired_difference_mean": float(np.mean(paired)),
                               "paired_difference_sd": float(np.std(paired, ddof=1)) if len(paired) > 1 else None,
                               "delta_r2_per_fold": spread("delta_r2_selected_minus_all397"), "n_folds_selected_above_all397": int(sum(m["delta_r2_selected_minus_all397"] > 0 for m in metas)),
                               "seconds_selected_fit": spread("seconds_selected_fit"), "seconds_all397_fit": spread("seconds_all397_fit"),
                               "committed_397_rung_mean_reported_alongside_not_used_for_the_claim": full.get("per_repeat_r2_mean"),
                               "committed_minus_selected_cross_device_not_the_claim": (full["per_repeat_r2_mean"] - float(np.mean(per_repeat))) if full else None,
                               "k_thesis": K_THESIS[target], "n_folds": n_folds_total,
                               "n_after_pearson": spread("n_after_pearson"), "n_after_lasso": spread("n_after_lasso"), "n_selected": spread("n_selected"),
                               "lasso_alpha": spread("lasso_alpha"), "n_convergence_warnings": spread("n_convergence_warnings"),
                               "n_convergence_warnings_inner_paths": spread("n_convergence_warnings_inner_paths"),
                               "n_folds_with_warnings": int(sum(m["n_convergence_warnings"] > 0 for m in metas)),
                               "n_folds_final_refit_not_converged": int(sum(not m["final_refit_converged"] for m in metas)),
                               "lasso_alpha_grid_position": spread("lasso_alpha_index"), "alpha_grid_size": len(metas[0]["lasso_alphas"]),
                               "n_folds_alpha_at_grid_minimum": int(sum(m["lasso_alpha_index"] == len(m["lasso_alphas"]) - 1 for m in metas)),
                               "share_folds_alpha_at_grid_minimum": float(np.mean([m["lasso_alpha_index"] == len(m["lasso_alphas"]) - 1 for m in metas])),
                               "selection_frequency": dict(sorted(((n, c / n_folds_total) for n, c in counts.items()), key=lambda kv: (-kv[1], kv[0]))),
                               "selected_in_every_fold": sorted(n for n, c in counts.items() if c == n_folds_total)}
        n_all = sum(results[t]["n_folds"] for t in targets)
        n_min = sum(results[t]["n_folds_alpha_at_grid_minimum"] for t in targets)
        results["_overall"] = {"units": n_all, "units_with_chosen_alpha_at_grid_minimum": n_min, "share": n_min / n_all}  # report-only, decisions.md 2026-10-07
        print(json.dumps({t: {k: v for k, v in results[t].items() if k != "selection_frequency"} for t in results}, indent=1), flush=True)
    sess.finish(units_total, results)


if __name__ == "__main__":
    main()
