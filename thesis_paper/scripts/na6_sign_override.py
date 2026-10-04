"""
NA6 follow-up: should the carrier-type classifier override the S regressor's sign? Both are scored on the same rows, each out of fold under its own chemistry-cluster 5 x 5 partition (the two partitions differ, see below; na6_sign_samefolds.py repeats the comparison on one partition).

  (a) regressor sign: the sign of the out-of-fold prediction of the S regressor (results/ladder_regen_snapfix/<stamp>/S_chemistry_full, frozen hyperparameters, 397 features);
  (b) classifier sign: the p-type decision (probability > 0.5) of the XGBClassifier of results/na6/<stamp>, same features, same folds.
The true sign is that of the measured S (the 8 rows with S = 0 are in neither score: the classifier never saw them, and a sign is not defined for them).

The regressor ran on 185,064 rows and the classifier on 185,056, so the two fold sets are rebuilt here from the snapfix CSV with the same seed and fold code and compared row by row
(the number of rows whose fold differs is reported); every unit of both runs is checked against its rebuild (same labels). Paired comparison on the common rows:
  - per repeat: accuracy of (a), of (b), and the difference b - a; mean over the 5 repeats and a t interval (df = 4) of the per-repeat differences;
  - cluster bootstrap of the paired difference (chemistry clusters resampled with replacement, all rows of a drawn cluster carried along, difference computed per repeat and
    averaged over repeats; n_boot draws, seed 0); the 2.5th and 97.5th percentiles;
  - discordant rows (b right and a wrong, a right and b wrong), summed over repeats;
  - strata by the size of the true S (< 20 uV/K, where the sign is close to the measurement noise, and >= 20 uV/K).
Decision rule (recorded in docs/decisions.md): the classifier's sign replaces the regressor's only if the bootstrap interval of the paired accuracy difference (b - a) lies above zero;
otherwise the regressor's sign is used. The magnitude of the difference is reported so that "clearly better" can be judged, not only its sign.

Usage (from the repository root):
    python thesis_paper/scripts/na6_sign_override.py --na6-dir thesis_paper/results/na6/20261004T102905 --regressor-dir results/ladder_regen_snapfix/20260917T150000/S_chemistry_full
"""

import argparse
import hashlib
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
N_REPEATS, N_FOLDS, SEED = 5, 5, 0
SMALL_S = 20.0  # uV/K: below this the sign of the measured S is within the measurement noise (same threshold as the NA10 analysis)


def fold_ids(groups, n):
    """fold_ids[r][i] = the test fold of row i in repeat r, rebuilt with the harness's seeds."""
    rng_master = np.random.default_rng(SEED)
    out = np.full((N_REPEATS, n), -1)
    splits = {}
    for r in range(N_REPEATS):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        splits[r] = list(ncv.outer_splits("chemistry", n, {"chemistry": groups}, N_FOLDS, rng))
        for f, (_, te) in enumerate(splits[r]):
            out[r, te] = f
    assert (out >= 0).all()
    return out, splits


def paired_comparison(ok_a, ok_b, abs_s, codes, n_boot):
    """
    Paired comparison of two sign rules on the same rows. ok_a, ok_b: boolean (n_repeats, n_rows), True where the rule gives the right sign; abs_s: |true S| per row;
    codes: chemistry-cluster code per row. Returns the per-stratum results (accuracy of each rule, the paired difference b - a with its cluster-bootstrap and across-repeat
    intervals, and the discordant rows).
    """
    small = abs_s < SMALL_S
    nc = int(codes.max()) + 1
    masks = {"all": np.ones(len(codes), bool), "abs_S_lt_20": small, "abs_S_ge_20": ~small}
    tq = stats.t.ppf(0.975, N_REPEATS - 1)
    rng = np.random.default_rng(SEED)
    cnt = {name: np.zeros((N_REPEATS, 3, nc)) for name in masks}
    for name, m in masks.items():
        for r in range(N_REPEATS):
            cnt[name][r, 0] = np.bincount(codes, weights=m.astype(float), minlength=nc)
            cnt[name][r, 1] = np.bincount(codes, weights=(m & ok_a[r]).astype(float), minlength=nc)
            cnt[name][r, 2] = np.bincount(codes, weights=(m & ok_b[r]).astype(float), minlength=nc)
    draws = [np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(float) for _ in range(n_boot)]
    out_strata = {}
    for name, m in masks.items():
        a = np.array([ok_a[r][m].mean() for r in range(N_REPEATS)])
        b = np.array([ok_b[r][m].mean() for r in range(N_REPEATS)])
        d = b - a
        boot_d, boot_a, boot_b = [], [], []
        for w in draws:
            tot = cnt[name][:, 0] @ w
            ra, rb = (cnt[name][:, 1] @ w) / tot, (cnt[name][:, 2] @ w) / tot
            boot_a.append(ra.mean()); boot_b.append(rb.mean()); boot_d.append((rb - ra).mean())
        pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        b_only = int(sum(((~ok_a[r]) & ok_b[r] & m).sum() for r in range(N_REPEATS)))
        a_only = int(sum((ok_a[r] & (~ok_b[r]) & m).sum() for r in range(N_REPEATS)))
        out_strata[name] = {"n_rows": int(m.sum()),
                            "regressor_sign_accuracy": {"mean": float(a.mean()), "ci95_cluster_bootstrap": pct(boot_a), "per_repeat": a.tolist()},
                            "classifier_sign_accuracy": {"mean": float(b.mean()), "ci95_cluster_bootstrap": pct(boot_b), "per_repeat": b.tolist()},
                            "difference_classifier_minus_regressor": {"mean": float(d.mean()), "ci95_cluster_bootstrap": pct(boot_d),
                                                                      "ci95_t_across_repeats": [float(d.mean() - tq * d.std(ddof=1) / np.sqrt(N_REPEATS)),
                                                                                                float(d.mean() + tq * d.std(ddof=1) / np.sqrt(N_REPEATS))],
                                                                      "per_repeat": d.tolist()},
                            "discordant_rows_summed_over_repeats": {"classifier_right_regressor_wrong": b_only, "regressor_right_classifier_wrong": a_only}}
    return out_strata


def decide(out_strata):
    """The decision rule: the classifier only if the lower cluster-bootstrap bound of the paired accuracy difference (all rows) is above zero."""
    return "classifier" if out_strata["all"]["difference_classifier_minus_regressor"]["ci95_cluster_bootstrap"][0] > 0 else "regressor"


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--na6-dir", required=True)
    ap.add_argument("--regressor-dir", required=True)
    ap.add_argument("--dataset", default=str(REPO / DATASET))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    na6, reg = Path(args.na6_dir), Path(args.regressor_dir)
    prov = rr.provenance({"dataset": str(Path(args.dataset).resolve()), "na6_results": str((na6 / "results.json").resolve().relative_to(REPO))}, __file__)
    assert prov["inputs"]["dataset"]["sha256"] == DATASET_SHA256, "not the snapfix CSV"
    reg_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(reg.glob("repeat*_fold*_predictions.npz"))}
    assert len(reg_files) == N_REPEATS * N_FOLDS

    full = pd.read_csv(args.dataset, usecols=["S", ncv.GROUP_COL])
    full = full[full["S"].notna()].reset_index(drop=True)  # the regressor's 185,064 rows (load_target_data("S"))
    keep = (full["S"] != 0).to_numpy()
    s_all = full["S"].to_numpy(float)
    g_all = full[ncv.GROUP_COL].to_numpy()
    folds_reg, splits_reg = fold_ids(g_all, len(full))
    sub = np.where(keep)[0]  # position of every classifier row in the regressor's row order
    g_sub = g_all[sub]
    folds_clf, splits_clf = fold_ids(g_sub, len(sub))

    # regressor out-of-fold predictions, one per row per repeat
    pred_reg = np.full((N_REPEATS, len(full)), np.nan)
    for r in range(N_REPEATS):
        for f, (_, te) in enumerate(splits_reg[r]):
            u = np.load(reg / f"repeat{r}_fold{f}_predictions.npz")
            assert len(u["y_pred"]) == len(te) and np.array_equal(u["y_true"], s_all[te]), f"regressor repeat {r} fold {f}: the rebuilt fold differs from the saved one"
            pred_reg[r, te] = u["y_pred"]
    assert not np.isnan(pred_reg).any()
    # classifier out-of-fold probabilities
    y = (s_all[sub] > 0).astype(int)
    proba = np.full((N_REPEATS, len(sub)), np.nan)
    for r in range(N_REPEATS):
        for f, (_, te) in enumerate(splits_clf[r]):
            u = np.load(na6 / "units" / f"repeat{r}_fold{f}.npz")
            assert len(u["y_true"]) == len(te) and np.array_equal(u["y_true"].astype(int), y[te]), f"classifier repeat {r} fold {f}: the rebuilt fold differs from the saved one"
            proba[r, te] = u["proba"]
    assert not np.isnan(proba).any()

    same_fold = [float((folds_reg[r, sub] == folds_clf[r]).mean()) for r in range(N_REPEATS)]
    # fold-label equality is too strict (labels can be permuted); compare the partitions: rows that share a fold in one split must share it in the other
    same_partition = []
    for r in range(N_REPEATS):
        pair = pd.crosstab(folds_reg[r, sub], folds_clf[r]).to_numpy()
        same_partition.append(float(pair.max(axis=1).sum() / len(sub)))  # share of rows whose regressor fold maps to the classifier fold holding most of it

    ok_a = (np.sign(pred_reg[:, sub]) == np.where(y == 1, 1, -1)[None, :])
    ok_b = ((proba > 0.5) == (y == 1)[None, :])
    codes, uniques = pd.factorize(g_sub)
    nc = len(uniques)
    out_strata = paired_comparison(ok_a, ok_b, np.abs(s_all[sub]), codes, args.n_boot)
    decision = decide(out_strata)
    out = {"n_rows": int(len(sub)), "n_clusters": int(nc), "n_regressor_rows": int(len(full)), "rows_with_S_zero_excluded": int((~keep).sum()),
           "n_repeats": N_REPEATS, "n_boot": args.n_boot, "small_S_threshold_uV_per_K": SMALL_S,
           "fold_agreement_regressor_vs_classifier": {"share_of_rows_with_identical_fold_label_per_repeat": same_fold,
                                                       "share_of_rows_whose_regressor_fold_maps_to_the_classifier_fold_holding_most_of_it": same_partition},
           "strata": out_strata, "decision_rule": "classifier replaces the regressor's sign only if the lower 95% cluster-bootstrap bound of the paired accuracy difference (classifier - regressor) is above 0",
           "decision": decision, "regressor_files_sha256": reg_files,
           "regressor_dir": str(reg), "na6_dir": str(na6)}
    d_out = REPO / "thesis_paper" / "results" / "na6_sign_override" / rr.utc_stamp()
    d_out.mkdir(parents=True, exist_ok=True)
    rr.write_json(d_out / "sign_comparison.json", out)
    rr.write_json(d_out / "run_config.json", {**prov, "seed": SEED})
    print("wrote", d_out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("regressor_files_sha256",)}, indent=1))


if __name__ == "__main__":
    main()
