"""
NA6 sign override, the same-folds version. na6_sign_override.py found that the Kaggle classifier's folds are not the S regressor's folds (the classifier dropped the 8 rows with S = 0, which changes
the cluster sizes and so the fold assignment: only about 55 to 67 percent of the rows share a fold label), so its comparison pairs the same rows but not the same partitions. Here the
classifier is refitted, with the S regressor's frozen hyperparameters and the 397 features, on the S regressor's own folds (results/ladder_regen_snapfix/<stamp>/S_chemistry_full, rebuilt from the
snapfix CSV and checked against every saved unit): each fold's classifier is trained on the regressor's training rows (minus the S = 0 rows) and scored on the regressor's test rows (minus the S = 0
rows), so both rules are evaluated on exactly the same partition of the rows. On the CPU (no GPU here), one fit per (repeat, fold), checkpointed per unit outside the repository tree
(checkpoints/na6_sign_samefolds/) and resumable; the paired comparison is the one of na6_sign_override.py (paired_comparison, decision rule in decide()).

The run's identity (code commit, tree state, script and dataset SHA256) is recorded when the first unit starts and checked on every resume.

Usage (from the repository root):
    python thesis_paper/scripts/na6_sign_samefolds.py --regressor-dir results/ladder_regen_snapfix/20260917T150000/S_chemistry_full
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import run_record as rr  # noqa: E402
import na6_sign_override as so  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

FROZEN_S = REPO / "checkpoints" / "saved_predictions" / "checkpoints" / "frozen_hyperparams" / "S.json"
CKPT = REPO / "checkpoints" / "na6_sign_samefolds"


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--regressor-dir", required=True)
    ap.add_argument("--dataset", default=str(REPO / so.DATASET))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-units", type=int, default=None, help="stop after this many new fits (for timing)")
    args = ap.parse_args()
    reg = Path(args.regressor_dir)
    prov = rr.provenance({"dataset": str(Path(args.dataset).resolve()), "frozen_hyperparams": str(FROZEN_S.relative_to(REPO))}, __file__)
    assert prov["inputs"]["dataset"]["sha256"] == so.DATASET_SHA256, "not the snapfix CSV"
    CKPT.mkdir(parents=True, exist_ok=True)
    start = CKPT / "run_start.json"
    if start.exists():
        old = json.loads(start.read_text(encoding="utf-8"))
        assert old["git_head"] == prov["git_head"] and old["script_sha256"] == prov["script_sha256"], "resuming with different code than the run started with"
        prov = old
    else:
        assert prov["tree_clean"], f"working tree is not clean: {prov['dirty_files']}"
        rr.write_json(start, prov)
    params = dict(json.loads(FROZEN_S.read_text(encoding="utf-8"))["best_params"])

    head = pd.read_csv(args.dataset, nrows=2)
    feat = ncv.get_feature_columns(head)
    assert len(feat) == 397
    df = pd.read_csv(args.dataset, usecols=["S", ncv.GROUP_COL, *feat])
    df = df[df["S"].notna()].reset_index(drop=True)
    s_all = df["S"].to_numpy(float)
    g_all = df[ncv.GROUP_COL].to_numpy()
    X = df[feat].to_numpy(dtype=np.float64)
    del df
    _, splits = so.fold_ids(g_all, len(s_all))
    keep = s_all != 0
    y = (s_all > 0).astype(int)
    done = 0
    for r in range(so.N_REPEATS):
        for f, (tr, te) in enumerate(splits[r]):
            path = CKPT / f"repeat{r}_fold{f}.npz"
            u = np.load(reg / f"repeat{r}_fold{f}_predictions.npz")
            assert len(u["y_pred"]) == len(te) and np.array_equal(u["y_true"], s_all[te]), f"repeat {r} fold {f}: the rebuilt fold differs from the regressor's"
            tr_k, te_k = tr[keep[tr]], te[keep[te]]
            if path.exists():
                z = np.load(path)
                assert np.array_equal(z["rows"], te_k)
                continue
            if args.max_units is not None and done >= args.max_units:
                continue
            t0 = time.perf_counter()
            m = xgb.XGBClassifier(**params, n_jobs=-1, tree_method="hist", device="cpu", random_state=0, objective="binary:logistic", eval_metric="logloss")
            m.fit(X[tr_k], y[tr_k])
            proba = m.predict_proba(X[te_k])[:, 1]
            np.savez(path, rows=te_k, proba=proba)
            done += 1
            print(f"repeat{r}_fold{f}: {time.perf_counter() - t0:.0f}s, accuracy {((proba > 0.5) == (y[te_k] == 1)).mean():.4f}", flush=True)
    if not all((CKPT / f"repeat{r}_fold{f}.npz").exists() for r in range(so.N_REPEATS) for f in range(so.N_FOLDS)):
        print("not all units done yet; rerun to continue")
        return

    sub = np.where(keep)[0]
    pos = np.full(len(s_all), -1)
    pos[sub] = np.arange(len(sub))
    pred_reg = np.full((so.N_REPEATS, len(s_all)), np.nan)
    proba = np.full((so.N_REPEATS, len(sub)), np.nan)
    for r in range(so.N_REPEATS):
        for f, (_, te) in enumerate(splits[r]):
            pred_reg[r, te] = np.load(reg / f"repeat{r}_fold{f}_predictions.npz")["y_pred"]
            z = np.load(CKPT / f"repeat{r}_fold{f}.npz")
            proba[r, pos[z["rows"]]] = z["proba"]
    assert not np.isnan(pred_reg).any() and not np.isnan(proba).any()
    ok_a = np.sign(pred_reg[:, sub]) == np.where(y[sub] == 1, 1, -1)[None, :]
    ok_b = (proba > 0.5) == (y[sub] == 1)[None, :]
    codes, uniques = pd.factorize(g_all[sub])
    strata = so.paired_comparison(ok_a, ok_b, np.abs(s_all[sub]), codes, args.n_boot)
    out = {"n_rows": int(len(sub)), "n_clusters": int(len(uniques)), "n_repeats": so.N_REPEATS, "n_boot": args.n_boot, "small_S_threshold_uV_per_K": so.SMALL_S,
           "design": "the classifier refitted on the S regressor's own folds (same partition for both rules), CPU, frozen S hyperparameters, 397 features",
           "strata": strata, "decision_rule": "classifier replaces the regressor's sign only if the lower 95% cluster-bootstrap bound of the paired accuracy difference (all rows) is above 0",
           "decision": so.decide(strata), "regressor_dir": str(reg)}
    d_out = REPO / "thesis_paper" / "results" / "na6_sign_override" / (rr.utc_stamp() + "_same_folds")
    (d_out / "units").mkdir(parents=True, exist_ok=True)
    for p in sorted(CKPT.glob("repeat*_fold*.npz")):
        shutil.copy2(p, d_out / "units" / p.name)
    rr.write_json(d_out / "sign_comparison_same_folds.json", out)
    rr.write_json(d_out / "run_config.json", {**prov, "seed": so.SEED, "xgboost": xgb.__version__, "device": "cpu", "hyperparameters": params})
    print("wrote", d_out)
    print(json.dumps({"all": strata["all"], "decision": out["decision"]}, indent=1))


if __name__ == "__main__":
    main()
