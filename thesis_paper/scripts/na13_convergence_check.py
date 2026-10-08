"""
NA13 convergence diagnostic, pre-registered in docs/decisions.md (2026-10-07, "NA13 before the Kaggle run"), written before any result of it existed.

For repeat 0 of the chemistry-cluster rung (the same folds as the rest of the work), every fold of each --targets target, and each max_iter of the ladder (default 2000, 20000, 200000), it runs
the selection of scripts/kaggle/na13_feature_selection.py (`select`, imported, not reimplemented) on the outer training rows only and records:
  - the chosen LassoCV alpha (value and position in the descending alpha grid), the number of ConvergenceWarnings of the LassoCV fits, the Lasso non-zero count, the final selected set and the
    fold R2 of the target's frozen XGBoost on it;
  - an independent dual-gap check of every inner-fold path fit on the same alpha grid (sklearn.linear_model.lasso_path on the centred inner training rows; a fit has converged when its dual gap
    is at most tol x the centred sum of squares of y, sklearn's own criterion), giving the positions of the non-converged alphas relative to the chosen one; the number of such fits is compared with the
    number of warnings the same path call raises, which validates the criterion;
  - the number of Pearson survivors, the largest |r| among them and the condition number of their standardised matrix.
Once per target, over all rows: the pairs of the 397 columns with |r| > 0.9999 and, for each MAGPIE `avg_dev <property>` column, its best-matching CBFV `dev` column and their correlation.
`--summarize <run dirs>` merges runs (one per target set, for example) and evaluates the criteria A to F of the pre-registration.

Usage (repository root):
    python thesis_paper/scripts/na13_convergence_check.py --targets kappa,zT --max-iters 2000,20000
    python thesis_paper/scripts/na13_convergence_check.py --summarize thesis_paper/results/na13_convergence/<utc>,<utc>
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kaggle"))
sys.path.insert(0, str(REPO))
import run_record as rr  # noqa: E402
import na13_feature_selection as na13  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

DATASET = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
DATASET_SHA256 = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"
TARGETS = ("S", "sigma", "kappa", "zT")
TOL = 1e-4  # sklearn's default, the LassoCV setting of na13_feature_selection.py
ALPHA_GRID_STEP = 3.0 / 19  # log10(1 / eps) / (n_alphas - 1) for eps 1e-3 and 20 alphas
SEED = 0
RESULTS = "thesis_paper/results/na13_convergence"


def dual_gap_check(Xs, ytr, groups_tr, alphas, max_iter):
    """Per inner fold: the grid positions of the alphas whose path fit did not converge, and how many warnings that path call raised."""
    out = []
    for itr, _ in GroupKFold(n_splits=3).split(Xs, groups=groups_tr):
        Xc = Xs[itr] - Xs[itr].mean(axis=0)
        yc = ytr[itr] - ytr[itr].mean()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            _, _, gaps = lasso_path(Xc, yc, alphas=alphas, max_iter=max_iter, tol=TOL, precompute=True)
        nonconv = np.where(gaps * len(yc) > TOL * float(yc @ yc))[0]  # lasso_path returns the dual gap divided by n_samples; sklearn compares gap with tol * y.y
        out.append({"nonconverged_alpha_positions": nonconv.tolist(), "n_nonconverged_by_gap": int(len(nonconv)),
                    "n_warnings_of_the_path_call": int(sum(issubclass(c.category, ConvergenceWarning) for c in caught))})
    return out


def final_refit_check(Xs, ytr, alpha, max_iter):
    """Independent check of LassoCV's final refit: a plain Lasso at the chosen alpha on all the rows it was given (precompute off, as LassoCV does for its 'auto' setting); the number of warnings it raises."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = Lasso(alpha=alpha, max_iter=max_iter, tol=TOL, precompute=False, fit_intercept=True).fit(Xs, ytr)
    return {"n_warnings": int(sum(issubclass(c.category, ConvergenceWarning) for c in caught)), "dual_gap": float(model.dual_gap_) * len(ytr), "tolerance": TOL * float(np.sum((ytr - ytr.mean()) ** 2)),
            "n_iter": int(np.max(model.n_iter_))}


def collinearity_overview(X, cols):
    """Exact-duplicate column pairs and the avg_dev / dev correspondence over all rows of a target."""
    sd = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    C = np.abs(Z.T @ Z / len(Z))
    np.fill_diagonal(C, 0.0)
    iu = np.triu_indices(len(cols), 1)
    dup = [(cols[i], cols[j], float(C[i, j])) for i, j in zip(*iu) if C[i, j] > 0.9999]
    mag = [i for i, c in enumerate(cols) if c.startswith("MagpieData avg_dev ")]
    cb = [j for j, c in enumerate(cols) if c.startswith("CBFV_dev_")]
    pairs = []
    for i in mag:
        j = max(cb, key=lambda jj: C[i, jj])
        pairs.append({"magpie": cols[i], "cbfv_best_match": cols[j], "abs_r": float(C[i, j]), "magpie_index": i, "cbfv_index": j})
    return {"n_columns": len(cols), "n_constant_columns": int((sd == 0).sum()), "exact_duplicate_pairs_abs_r_gt_0.9999": [{"a": a, "b": b, "abs_r": r} for a, b, r in dup],
            "avg_dev_to_dev_best_matches": pairs}


def run(args):
    out = rr.new_run_dir("na13_convergence")
    prov = rr.provenance({"dataset": DATASET}, __file__)
    assert prov["inputs"]["dataset"]["sha256"] == DATASET_SHA256, "not the snapfix CSV"
    ladder = [int(x) for x in args.max_iters.split(",")]
    hp_dir = REPO / "checkpoints" / "saved_predictions" / "checkpoints" / "frozen_hyperparams"
    df_all = pd.read_csv(REPO / DATASET)
    res = {"max_iters": ladder, "repeat": 0, "targets": {}}
    for target in args.targets.split(","):
        df = df_all[df_all[target].notna()].reset_index(drop=True)
        assert len(df) == na13.TARGET_ROWS[target]
        cols = ncv.get_feature_columns(df)
        X = df[cols].to_numpy(dtype=np.float64)
        y = ncv._transform_target(df[target].to_numpy(dtype=np.float64), target)
        groups = df[ncv.GROUP_COL].to_numpy()
        hp = json.loads((hp_dir / f"{target}.json").read_text(encoding="utf-8"))["best_params"]
        t_out = {"collinearity_all_rows": collinearity_overview(X, cols), "folds": {}}
        rng_master = np.random.default_rng(SEED)
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))  # repeat 0
        for f, (tr, te) in enumerate(ncv.outer_splits("chemistry", len(df), {"chemistry": groups}, 5, rng)):
            if args.folds and f not in args.folds:
                continue
            Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
            keep1 = np.where(na13.pearson_keep(Xtr))[0]
            Xs = StandardScaler().fit_transform(Xtr[:, keep1])
            C = np.abs(Xs.T @ Xs / len(Xs))
            ev = np.linalg.eigvalsh(Xs.T @ Xs / len(Xs))
            fold = {"n_train": int(len(tr)), "n_after_pearson": int(len(keep1)), "max_abs_r_among_survivors": float(np.max(C - np.diag(np.diag(C)))),
                    "condition_number_of_survivors": float(np.sqrt(ev.max() / max(ev.min(), 1e-300))), "min_eigenvalue": float(ev.min()),
                    "dropped_by_pearson_avg_dev_dev_pairs": None, "settings": {}}
            kept = set(keep1.tolist())
            fold["dropped_by_pearson_avg_dev_dev_pairs"] = [{"magpie_kept": p["magpie_index"] in kept, "cbfv_kept": p["cbfv_index"] in kept, "abs_r": p["abs_r"]}
                                                           for p in t_out["collinearity_all_rows"]["avg_dev_to_dev_best_matches"]]
            for M in ladder:
                t0 = time.perf_counter()
                sel, info = na13.select(Xtr, ytr, gtr, na13.K_THESIS[target], 20, SEED, M)
                assert info["n_after_pearson"] == len(keep1)
                model = ncv._build_xgb_model(hp, "cpu")
                model.fit(Xtr[:, sel], ytr)
                r2 = 1.0 - float(np.sum((y[te] - model.predict(X[te][:, sel])) ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
                gap = dual_gap_check(Xs, ytr, gtr, np.array(info["lasso_alphas"]), M)
                final = final_refit_check(Xs, ytr, info["lasso_alpha"], M)
                n_inner = int(sum(g["n_nonconverged_by_gap"] for g in gap))
                fold["settings"][str(M)] = {**info, "selected": [cols[i] for i in sel], "outer_r2": r2, "inner_path_checks": gap,
                                            "n_nonconverged_by_gap_total": n_inner, "final_refit_independent_check": final,
                                            "final_refit_agrees_with_select": bool((final["n_warnings"] == 0) == info["final_refit_converged"]),
                                            "warnings_equal_inner_gap_plus_final_refit": bool(info["n_convergence_warnings"] == n_inner + final["n_warnings"]),
                                            "criterion_matches_path_warnings": all(g["n_nonconverged_by_gap"] == g["n_warnings_of_the_path_call"] for g in gap),
                                            "seconds": time.perf_counter() - t0}
                print(f"{target} fold {f} max_iter {M}: alpha {info['lasso_alpha']:.4g} (position {info['lasso_alpha_index']}), warnings {info['n_convergence_warnings']}, "
                      f"non-converged by gap {n_inner} + final refit {final['n_warnings']}, after Lasso {info['n_after_lasso']}, selected {len(sel)}, R2 {r2:.4f} "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
            t_out["folds"][str(f)] = fold
        res["targets"][target] = t_out
        (out / "diagnostic.json").write_text(json.dumps(res, indent=1), encoding="utf-8")  # rewritten after every target
    rr.write_json(out / "run_config.json", {**prov, "targets": args.targets, "max_iters": ladder, "folds": args.folds})
    print("wrote", out)


def jaccard(a, b):
    """Jaccard similarity of two collections."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def summarize(dirs):
    """Evaluate the pre-registered criteria A to F on one or more diagnostic runs."""
    runs = [json.loads((Path(d) / "diagnostic.json").read_text(encoding="utf-8")) for d in dirs]
    ladder = runs[0]["max_iters"]
    assert all(r["max_iters"] == ladder for r in runs)
    base = str(ladder[0])
    folds = [(t, f, fold) for r in runs for t, tt in r["targets"].items() for f, fold in tt["folds"].items()]
    out = {"dirs": list(dirs), "n_folds": len(folds), "ladder": ladder, "A": {}, "B_C_D": {}, "E": {}, "F": {}}
    adopted = None
    for M in map(str, ladder):
        warn = [fold["settings"][M]["n_convergence_warnings"] for _, _, fold in folds]
        gaps = [fold["settings"][M]["n_nonconverged_by_gap_total"] for _, _, fold in folds]
        finals = [fold["settings"][M]["final_refit_independent_check"]["n_warnings"] for _, _, fold in folds]
        ok = sum(warn) == 0 and sum(gaps) == 0 and sum(finals) == 0
        out["A"][M] = {"folds_with_warnings": int(sum(w > 0 for w in warn)), "total_warnings": int(sum(warn)), "folds_with_nonconverged_fits_by_gap": int(sum(g > 0 for g in gaps)),
                       "total_nonconverged_fits_by_gap": int(sum(gaps)),
                       "folds_with_final_refit_not_converged (independent check)": int(sum(f > 0 for f in finals)),
                       "warnings_equal_inner_gap_plus_final_refit_in_every_fold": all(fold["settings"][M]["warnings_equal_inner_gap_plus_final_refit"] for _, _, fold in folds),
                       "final_refit_check_agrees_with_select_in_every_fold": all(fold["settings"][M]["final_refit_agrees_with_select"] for _, _, fold in folds),
                       "criterion_matches_path_warnings_in_every_fold": all(fold["settings"][M]["criterion_matches_path_warnings"] for _, _, fold in folds),
                       "holds": bool(ok)}
        if ok and adopted is None:
            adopted = M
    out["A"]["adopted_max_iter"] = adopted
    for M in map(str, ladder[1:]):
        jac, dn, dr2, dalpha, at_min_both = [], [], [], [], []
        for _, _, fold in folds:
            o, n = fold["settings"][base], fold["settings"][M]
            last = len(o["lasso_alphas"]) - 1
            at_min_both.append(o["lasso_alpha_index"] == last and n["lasso_alpha_index"] == last)
            dalpha.append(abs(np.log10(n["lasso_alpha"]) - np.log10(o["lasso_alpha"])))
            jac.append(jaccard(o["selected"], n["selected"]))
            dn.append(abs(n["n_after_lasso"] - o["n_after_lasso"]) / max(o["n_after_lasso"], 1))
            dr2.append(n["outer_r2"] - o["outer_r2"])
        # B is counted only on the folds where the chosen alpha is not the grid minimum under both settings (decisions.md 2026-10-07, criterion B); the others are reported separately
        counted = [d for d, e in zip(dalpha, at_min_both) if not e]
        b_ok = int(sum(d <= ALPHA_GRID_STEP + 1e-12 for d in counted))
        b_need = int(np.ceil(0.9 * len(counted)))
        out["B_C_D"][M] = {"B_folds_counted": len(counted), "B_folds_with_alpha_at_grid_minimum_under_both_settings (not counted, no stability claim)": int(sum(at_min_both)),
                           "B_alpha_within_one_grid_step_folds": b_ok, "B_threshold_folds": b_need,
                           "B_max_abs_log10_alpha_change_counted_folds": float(max(counted)) if counted else None,
                           "B_max_abs_log10_alpha_change_all_folds_for_information": float(max(dalpha)),
                           "C_jaccard_median": float(np.median(jac)), "C_jaccard_min": float(min(jac)),
                           "C_n_after_lasso_relative_change_median": float(np.median(dn)), "D_max_abs_r2_difference": float(np.max(np.abs(dr2))),
                           "D_mean_abs_r2_difference": float(np.mean(np.abs(dr2))), "D_mean_r2_difference_new_minus_old": float(np.mean(dr2)),
                           "B_holds": (bool(b_ok >= b_need) if counted else None),
                           "C_holds": bool(np.median(jac) >= 0.8 and min(jac) >= 0.6 and np.median(dn) <= 0.10),
                           "D_holds": bool(np.max(np.abs(dr2)) <= 0.005 and np.mean(np.abs(dr2)) <= 0.002)}
    pos_below, pos_at_or_above, n_nonconv = 0, 0, 0
    per_fold = []
    for t, f, fold in folds:
        c = fold["settings"][base]["lasso_alpha_index"]
        idx = [i for g in fold["settings"][base]["inner_path_checks"] for i in g["nonconverged_alpha_positions"]]
        n_nonconv += len(idx)
        pos_below += sum(i > c for i in idx)
        pos_at_or_above += sum(i <= c for i in idx)
        per_fold.append({"target": t, "fold": int(f), "chosen_position": c, "nonconverged_positions": sorted(idx)})
    out["E2_alpha_position"] = {}
    for M in map(str, ladder):
        pos_all = [fold["settings"][M]["lasso_alpha_index"] for _, _, fold in folds]
        size = len(folds[0][2]["settings"][M]["lasso_alphas"])
        out["E2_alpha_position"][M] = {"grid_size": size, "per_fold": [{"target": t, "fold": int(f), "position": fold["settings"][M]["lasso_alpha_index"]} for t, f, fold in folds],
                                      "folds_at_grid_minimum (position grid_size - 1)": int(sum(p == size - 1 for p in pos_all)), "share_at_grid_minimum": float(np.mean([p == size - 1 for p in pos_all]))}
    out["E"] = {"setting": base, "nonconverged_inner_fits": n_nonconv, "below_chosen_alpha (smaller alpha, higher position)": pos_below, "at_or_above_chosen_alpha": pos_at_or_above, "per_fold": per_fold}
    for r in runs:
        for t, tt in r["targets"].items():
            c = tt["collinearity_all_rows"]
            fs = tt["folds"].values()
            out["F"][t] = {"exact_duplicate_pairs": len(c["exact_duplicate_pairs_abs_r_gt_0.9999"]),
                           "avg_dev_dev_pairs_abs_r_ge_0.99": int(sum(p["abs_r"] >= 0.99 for p in c["avg_dev_to_dev_best_matches"])),
                           "avg_dev_dev_pairs_total": len(c["avg_dev_to_dev_best_matches"]),
                           "avg_dev_dev_min_abs_r": float(min(p["abs_r"] for p in c["avg_dev_to_dev_best_matches"])),
                           "pairs_with_both_columns_surviving_pearson_per_fold_max": int(max(sum(p["magpie_kept"] and p["cbfv_kept"] for p in fold["dropped_by_pearson_avg_dev_dev_pairs"]) for fold in fs)),
                           "survivors_min_max": [int(min(f["n_after_pearson"] for f in fs)), int(max(f["n_after_pearson"] for f in fs))],
                           "max_abs_r_among_survivors": float(max(f["max_abs_r_among_survivors"] for f in fs)),
                           "condition_number_survivors_min_max": [float(min(f["condition_number_of_survivors"] for f in fs)), float(max(f["condition_number_of_survivors"] for f in fs))]}
    return out


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--max-iters", default="2000,20000,200000")
    ap.add_argument("--folds", default="", help="comma list of fold indices of repeat 0 (default all five)")
    ap.add_argument("--summarize", default=None, help="comma list of run directories to evaluate instead of running")
    args = ap.parse_args()
    args.folds = [int(x) for x in args.folds.split(",") if x != ""]
    if args.summarize:
        s = summarize(args.summarize.split(","))
        d = Path(args.summarize.split(",")[0])
        (d / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in s.items() if k != "E"}, indent=1))
        print(json.dumps({k: v for k, v in s["E"].items() if k != "per_fold"}, indent=1))
        return
    run(args)


if __name__ == "__main__":
    main()
