"""
NA6: carrier-type classifier (p versus n) under chemistry-cluster CV (GPU), and the final classifier for the screening.

Label: sign of the Seebeck coefficient (1 = p-type, S > 0; 0 = n-type, S < 0; the 8 rows with S = 0 are dropped). Features: the same 397.
Model: XGBClassifier with the S regressor's frozen hyperparameters (the thesis says "the same features" and gives no separate tuning; a separate
search is not run here, and that choice is recorded). Folds: the same chemistry-cluster 5 x 5 as the rest of the work.

Units: one per (repeat, fold) with held-out labels and probabilities, plus one "final" unit: the classifier fitted on all rows, saved as
final_classifier.json for na_final_models.py (its sign decision overrides the regressor's when they disagree).
Results: accuracy (pooled and per repeat, mean and SD), precision, recall, F1, ROC AUC, the pooled confusion matrix, and the class balance.

Smoke mode: ~3,000 rows, 1 repeat x 3 folds, 20 trees.

Usage:
    python thesis_paper/scripts/kaggle/na6_classifier.py --out-dir /kaggle/working/na6 --expect-commit <sha> --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import harness as H  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

FROZEN_S = H.repo_root() / "checkpoints" / "saved_predictions" / "checkpoints" / "frozen_hyperparams" / "S.json"


def build(params, device):
    return xgb.XGBClassifier(**params, n_jobs=-1, tree_method="hist", device=device, random_state=0, objective="binary:logistic", eval_metric="logloss")


def main():
    ap = argparse.ArgumentParser()
    H.add_common_args(ap)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.n_repeats, args.n_folds, args.device = 1, 3, "cpu"
    params = dict(json.loads(FROZEN_S.read_text(encoding="utf-8"))["best_params"])
    if args.smoke:
        params = {**params, "n_estimators": 20, "max_depth": min(params["max_depth"], 4)}
    sess = H.Session("na6_classifier", args, {"n_repeats": args.n_repeats, "n_folds": args.n_folds, "seed": args.seed, "smoke": args.smoke, "hyperparams": params,
                                              "label": "S>0", "hyperparams_source": "S regressor frozen set"})
    df = H.load_frame(sess)
    df = df[df["S"].notna() & (df["S"] != 0)].reset_index(drop=True)
    if args.smoke:
        df = H.smoke_subset(df, ncv.GROUP_COL, 3000)
    elif len(df) != 185064 - 8:
        raise SystemExit(f"{len(df)} rows with S != 0, expected {185064 - 8}")
    cols = ncv.get_feature_columns(df)
    X = df[cols].to_numpy(dtype=np.float64)
    y = (df["S"].to_numpy() > 0).astype(int)
    groups = df[ncv.GROUP_COL].to_numpy()
    Xd = ncv._to_device(X, args.device)
    units = [(r, f) for r in range(args.n_repeats) for f in range(args.n_folds)]
    rng_master = np.random.default_rng(args.seed)
    splits = {}
    for r in range(args.n_repeats):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        splits[r] = list(ncv.outer_splits("chemistry", len(df), {"chemistry": groups}, args.n_folds, rng))
    for r, f in units:
        uid = f"repeat{r}_fold{f}"
        if sess.done(uid):
            continue
        if sess.out_of_time():
            print("time budget reached before", uid, flush=True)
            break
        tr, te = splits[r][f]
        t0 = time.perf_counter()
        model = build(params, args.device)
        model.fit(Xd[tr], y[tr])
        proba = np.asarray(model.predict_proba(Xd[te])[:, 1], dtype=float)
        sess.save(uid, {"repeat": r, "fold": f, "n_train": int(len(tr)), "n_test": int(len(te)), "accuracy": float(accuracy_score(y[te], proba > 0.5)),
                        "seconds": time.perf_counter() - t0}, {"y_true": y[te], "proba": proba})
        print(f"{uid}: accuracy {accuracy_score(y[te], proba > 0.5):.4f}", flush=True)
    if not sess.done("final") and not sess.out_of_time() and all(sess.done(f"repeat{r}_fold{f}") for r, f in units):
        model = build(params, args.device)
        model.fit(Xd, y)
        model.save_model(str(sess.out / "final_classifier.json"))
        sess.save("final", {"n_rows": int(len(y)), "model_file": "final_classifier.json", "feature_columns": cols})
    results = None
    if all(sess.done(u) for u in [f"repeat{r}_fold{f}" for r, f in units] + ["final"]):
        yt, pp, per_rep = [], [], []
        for r in range(args.n_repeats):
            ys, ps = [], []
            for f in range(args.n_folds):
                _, a = sess.load(f"repeat{r}_fold{f}")
                ys.append(a["y_true"]); ps.append(a["proba"])
            per_rep.append(float(accuracy_score(np.concatenate(ys), np.concatenate(ps) > 0.5)))
            yt += ys; pp += ps
        yt, pp = np.concatenate(yt), np.concatenate(pp)
        pred = pp > 0.5
        results = {"n_rows": int(len(y)), "share_p_type": float(y.mean()), "n_p": int(y.sum()), "n_n": int((1 - y).sum()),
                   "accuracy_pooled": float(accuracy_score(yt, pred)), "accuracy_per_repeat": per_rep, "accuracy_repeat_mean": float(np.mean(per_rep)),
                   "accuracy_repeat_sd": float(np.std(per_rep, ddof=1)) if len(per_rep) > 1 else None,
                   "precision_p": float(precision_score(yt, pred)), "recall_p": float(recall_score(yt, pred)), "f1_p": float(f1_score(yt, pred)),
                   "precision_n": float(precision_score(1 - yt, 1 - pred)), "recall_n": float(recall_score(1 - yt, 1 - pred)),
                   "roc_auc": float(roc_auc_score(yt, pp)), "confusion_matrix_rows_true_cols_pred_n_p": confusion_matrix(yt, pred).tolist()}
        print(json.dumps(results, indent=1), flush=True)
    sess.finish(len(units) + 1, results)


if __name__ == "__main__":
    main()
