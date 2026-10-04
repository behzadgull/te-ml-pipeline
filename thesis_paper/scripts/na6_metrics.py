"""
NA6 metrics with confidence intervals: the carrier-type classifier (p versus n) of results/na6/<stamp>, scored from its saved out-of-fold probabilities.

The classifier ran on Kaggle (scripts/kaggle/na6_classifier.py) and saved, per outer fold, the held-out labels and probabilities but not the chemistry-cluster ids of
the held-out rows. The folds are rebuilt here from the snapfix CSV with the same seeds and the same fold code, and every unit is checked against the rebuild
(same number of test rows, identical labels), so the cluster id of every held-out row is known.

Two kinds of interval, both reported:
  - across repeats: the mean and a t interval (df = 4) of the per-repeat pooled metric (5 repeats of 5 grouped folds), as for the rest of the work;
  - cluster bootstrap: chemistry clusters are resampled with replacement (n_boot draws), every row of a drawn cluster in every repeat is carried along, and each metric
    is computed per repeat and averaged over the 5 repeats; the 2.5th and 97.5th percentiles are the interval. This is the interval for "which chemistries are in the
    data", which the across-repeat interval does not capture (the repeats re-split the same clusters).
Metrics: accuracy, balanced accuracy, precision, recall and F1 of the p class, precision and recall of the n class, ROC AUC (AUC: bootstrap on repeat 0 only, with
weights, tied probabilities count half; the number of exact ties is reported). The pooled accuracy is checked against results/na6/<stamp>/results.json.

Usage (from the repository root):
    python thesis_paper/scripts/na6_metrics.py --na6-dir thesis_paper/results/na6/20261004T102905
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import run_record as rr  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

DATASET = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
DATASET_SHA256 = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"
N_REPEATS, N_FOLDS, SEED = 5, 5, 0


def counts_to_metrics(tp, fp, tn, fn):
    """Metrics from confusion counts (arrays allowed): positive class = p-type (1)."""
    n = tp + fp + tn + fn
    rec_p, rec_n = tp / (tp + fn), tn / (tn + fp)
    prec_p, prec_n = tp / (tp + fp), tn / (tn + fn)
    return {"accuracy": (tp + tn) / n, "balanced_accuracy": (rec_p + rec_n) / 2, "precision_p": prec_p, "recall_p": rec_p,
            "f1_p": 2 * prec_p * rec_p / (prec_p + rec_p), "precision_n": prec_n, "recall_n": rec_n}


def weighted_auc(y, order, starts, w):
    """ROC AUC with row weights `w`; `order` sorts the rows by score, `starts` are the indices (in sorted order) where each distinct score begins; tied scores count half."""
    ys, ws = y[order], w[order]
    neg, pos = ws * (1 - ys), ws * ys
    neg_g, pos_g = np.add.reduceat(neg, starts), np.add.reduceat(pos, starts)
    below = np.cumsum(neg_g) - neg_g
    return float((pos_g * (below + 0.5 * neg_g)).sum() / (pos_g.sum() * neg_g.sum()))


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--na6-dir", required=True)
    ap.add_argument("--dataset", default=str(REPO / DATASET))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-boot-auc", type=int, default=500)
    args = ap.parse_args()
    na6 = Path(args.na6_dir)
    res = json.loads((na6 / "results.json").read_text(encoding="utf-8"))
    prov = rr.provenance({"dataset": str(Path(args.dataset).resolve()), "na6_results": str((na6 / "results.json").resolve().relative_to(REPO)),
                          "na6_manifest": str((na6 / "manifest.json").resolve().relative_to(REPO))}, __file__)
    assert prov["inputs"]["dataset"]["sha256"] == DATASET_SHA256, "not the snapfix CSV"

    df = pd.read_csv(args.dataset, usecols=["S", ncv.GROUP_COL])
    df = df[df["S"].notna() & (df["S"] != 0)].reset_index(drop=True)
    assert len(df) == res["n_rows"], (len(df), res["n_rows"])
    y = (df["S"].to_numpy() > 0).astype(int)
    groups = df[ncv.GROUP_COL].to_numpy()
    codes, uniques = pd.factorize(groups)
    nc = len(uniques)
    rng_master = np.random.default_rng(SEED)
    proba = np.full((N_REPEATS, len(df)), np.nan)
    for r in range(N_REPEATS):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        for f, (tr, te) in enumerate(ncv.outer_splits("chemistry", len(df), {"chemistry": groups}, N_FOLDS, rng)):
            u = np.load(na6 / "units" / f"repeat{r}_fold{f}.npz")
            assert len(u["y_true"]) == len(te) and np.array_equal(u["y_true"].astype(int), y[te]), f"repeat {r} fold {f}: the rebuilt fold differs from the saved one"
            proba[r, te] = u["proba"]
    assert not np.isnan(proba).any()
    pred = proba > 0.5

    def cm(r, w=None):
        w = np.ones(len(y)) if w is None else w
        t, p = y == 1, pred[r]
        return (float((w * (t & p)).sum()), float((w * (~t & p)).sum()), float((w * (~t & ~p)).sum()), float((w * (t & ~p)).sum()))

    per_repeat = {k: [] for k in ("accuracy", "balanced_accuracy", "precision_p", "recall_p", "f1_p", "precision_n", "recall_n", "roc_auc")}
    for r in range(N_REPEATS):
        m = counts_to_metrics(*cm(r))
        for k in m:
            per_repeat[k].append(float(m[k]))
        per_repeat["roc_auc"].append(float(roc_auc_score(y, proba[r])))
    assert abs(np.mean(per_repeat["accuracy"]) - res["accuracy_pooled"]) < 1e-9, "accuracy differs from results.json"
    tq = stats.t.ppf(0.975, N_REPEATS - 1)
    across = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "ci95_t": [float(np.mean(v) - tq * np.std(v, ddof=1) / np.sqrt(N_REPEATS)),
                                                                                       float(np.mean(v) + tq * np.std(v, ddof=1) / np.sqrt(N_REPEATS))],
                  "per_repeat": v} for k, v in per_repeat.items()}

    # cluster bootstrap: per-cluster confusion counts per repeat, then multinomial cluster weights
    cc = np.zeros((N_REPEATS, 4, nc))
    for r in range(N_REPEATS):
        t, p = y == 1, pred[r]
        for i, mask in enumerate((t & p, ~t & p, ~t & ~p, t & ~p)):
            cc[r, i] = np.bincount(codes, weights=mask.astype(float), minlength=nc)
    rng = np.random.default_rng(SEED)
    boots = {k: [] for k in ("accuracy", "balanced_accuracy", "precision_p", "recall_p", "f1_p", "precision_n", "recall_n")}
    for _ in range(args.n_boot):
        w = np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(float)
        ms = [counts_to_metrics(*(cc[r] @ w)) for r in range(N_REPEATS)]
        for k in boots:
            boots[k].append(float(np.mean([m[k] for m in ms])))
    order = np.argsort(proba[0], kind="stable")
    sorted_scores = proba[0][order]
    starts = np.concatenate([[0], np.where(np.diff(sorted_scores) != 0)[0] + 1])
    n_ties = int(len(proba[0]) - len(starts))
    auc_b = []
    for _ in range(args.n_boot_auc):
        w = np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(float)[codes]
        auc_b.append(weighted_auc(y, order, starts, w))
    boot = {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in boots.items()}
    boot["roc_auc_repeat0"] = {"mean": float(np.mean(auc_b)), "ci95": [float(np.percentile(auc_b, 2.5)), float(np.percentile(auc_b, 97.5))], "n_boot": args.n_boot_auc,
                               "exact_ties_in_repeat0_probabilities": n_ties}
    # errors by chemistry cluster size (is the accuracy carried by large clusters?)
    sizes = np.bincount(codes)
    sz = sizes[codes]
    by_size = {}
    for name, lo, hi in (("singleton_cluster", 1, 1), ("2_to_9_rows", 2, 9), ("10_to_99_rows", 10, 99), ("100_or_more_rows", 100, 10**9)):
        mask = (sz >= lo) & (sz <= hi)
        by_size[name] = {"n_rows": int(mask.sum()), "accuracy": float(np.mean([(pred[r][mask] == (y[mask] == 1)).mean() for r in range(N_REPEATS)]))}
    out = {"n_rows": int(len(y)), "n_clusters": int(nc), "share_p_type": float(y.mean()), "n_repeats": N_REPEATS, "n_folds": N_FOLDS, "n_boot": args.n_boot,
           "across_repeats": across, "cluster_bootstrap": boot, "accuracy_by_cluster_size": by_size,
           "confusion_matrix_all_repeats_rows_true_cols_pred_n_p": res["confusion_matrix_rows_true_cols_pred_n_p"],
           "check": "pooled accuracy equals results.json accuracy_pooled; every unit equals the rebuilt fold"}
    d = REPO / "thesis_paper" / "results" / "na6_metrics" / rr.utc_stamp()
    d.mkdir(parents=True, exist_ok=True)
    rr.write_json(d / "metrics.json", out)
    rr.write_json(d / "run_config.json", {**prov, "na6_dir": str(na6.resolve().relative_to(REPO)), "seed": SEED})
    print("wrote", d)
    print(json.dumps({"across_repeats": {k: {"mean": v["mean"], "ci95_t": v["ci95_t"]} for k, v in across.items()}, "cluster_bootstrap": boot, "by_size": by_size}, indent=1))


if __name__ == "__main__":
    main()
