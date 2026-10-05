"""
NA2, stacking (local, CPU, minutes): a stacked model over the XGBoost, random forest and LightGBM out-of-fold predictions.

The three base models were scored on identical chemistry-cluster 5 x 5 folds (checked here: y_true is bit-identical across the three for every fold).
For each target and repeat, a non-negative ridge meta-learner (sklearn Ridge, alpha 1, positive=True, intercept) is fitted on the base models'
out-of-fold predictions of four folds and predicts the fifth (cross-fitted across the five outer folds of that repeat), so the stack's prediction for a
row never comes from a meta-learner that saw that row's label. Also reported: the simple mean of the three, and each base model on its own.
Pooled per-repeat R2 (mean and SD over the 5 repeats) is compared with XGBoost's committed value.

Caveat, stated in the output: the base predictions of the four meta-training folds were produced by base models that were trained on rows
that include the held-out fold's rows, so the stack is not fully nested; a stack whose gain over XGBoost is small is therefore a weak result
either way, and a large gain would have to be rechecked with nested stacking before it is claimed. The meta-learner has 4 coefficients, so the
optimism is expected to be tiny, but it is not zero.

Inputs: --rf-dir and --lgbm-dir are the na2_trees.py output bundles (a directory or the .tar.gz); each is verified (status complete, manifest SHA256,
identity fields) before use. Either may be several bundles separated by commas, when the model ran as one session per target (the random forest does):
every target must then be covered by exactly one bundle, and the bundles must share the dataset SHA256 and the code commit. The XGBoost OOF predictions are the committed results/ladder_regen_snapfix/20260917T150000/<target>_chemistry_full/
(Git LFS; must be present locally).

Usage:
    python thesis_paper/scripts/kaggle/na2_stacking.py --rf-dir <dir|tar.gz> --lgbm-dir <dir|tar.gz> [--out-dir results/na2_stacking/<utc>]
"""

import argparse
import json
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

LADDER_DIR = H.repo_root() / "results" / "ladder_regen_snapfix" / "20260917T150000"
TARGETS = ("S", "sigma", "kappa", "zT")


def open_bundle(path, tmp):
    p = Path(path)
    if p.is_file():
        with tarfile.open(p) as tf:
            tf.extractall(tmp / p.stem)
        sub = tmp / p.stem
        inner = [d for d in sub.iterdir() if d.is_dir()]
        p = inner[0] if len(inner) == 1 and not (sub / "status.json").exists() else sub
    st = json.loads((p / "status.json").read_text(encoding="utf-8"))
    if not st.get("complete"):
        raise SystemExit(f"{p}: session is not complete")
    man = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    for rel, sha in man["files"].items():
        if H.sha256_file(p / rel) != sha:
            raise SystemExit(f"{p / rel} does not match its manifest SHA256")
    cfg = json.loads(sorted((p / "run_configs").glob("session_*.json"))[-1].read_text(encoding="utf-8"))
    return p, cfg


def open_parts(arg, tmp, targets=TARGETS):
    """The bundles of one model (comma-separated paths) as ({target: units directory}, config of the first bundle); every target is covered by exactly one bundle."""
    parts = [open_bundle(x, tmp) for x in str(arg).split(",") if x]
    cfgs = [c for _, c in parts]
    assert len({c["dataset_sha256"] for c in cfgs}) == 1 and len({c["git_head"] for c in cfgs}) == 1, "the bundles of one model differ in dataset or code commit"
    where = {}
    for p, _ in parts:
        for t in targets:
            if (p / "units" / f"rung_{t}_repeat0_fold0.npz").exists():
                assert t not in where, f"target {t} is in two bundles"
                where[t] = p / "units"
    missing = [t for t in targets if t not in where]
    if missing:
        raise SystemExit(f"no bundle covers the targets {missing}")
    return where, cfgs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-dir", required=True)
    ap.add_argument("--lgbm-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    rf, rf_cfg = open_parts(args.rf_dir, tmp)
    lg, lg_cfg = open_parts(args.lgbm_dir, tmp)
    assert rf_cfg["dataset_sha256"] == lg_cfg["dataset_sha256"], "random forest and LightGBM ran on different datasets"
    ladder = json.loads((H.repo_root() / "reports" / "regen_snapfix" / "20260917T150000" / "ladder_metrics.json").read_text(encoding="utf-8"))["runs"]
    out = Path(args.out_dir) if args.out_dir else H.repo_root() / "thesis_paper" / "results" / "na2_stacking" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for target in TARGETS:
        per = {k: [] for k in ("xgboost", "random_forest", "lightgbm", "mean3", "stack")}
        weights = []
        for r in range(args.n_repeats):
            Y, P = [], []
            for f in range(args.n_folds):
                x = np.load(LADDER_DIR / f"{target}_chemistry_full" / f"repeat{r}_fold{f}_predictions.npz")
                a = np.load(rf[target] / f"rung_{target}_repeat{r}_fold{f}.npz")
                b = np.load(lg[target] / f"rung_{target}_repeat{r}_fold{f}.npz")
                assert np.array_equal(x["y_true"], a["y_true"]) and np.array_equal(x["y_true"], b["y_true"]), f"{target} r{r} f{f}: y_true differs between models"
                Y.append(x["y_true"])
                P.append(np.column_stack([x["y_pred"], a["y_pred"], b["y_pred"]]))
            stack_pred = []
            for f in range(args.n_folds):
                tr = [g for g in range(args.n_folds) if g != f]
                meta = Ridge(alpha=1.0, positive=True).fit(np.vstack([P[g] for g in tr]), np.concatenate([Y[g] for g in tr]))
                stack_pred.append(meta.predict(P[f]))
                weights.append(list(map(float, meta.coef_)) + [float(meta.intercept_)])
            y_all, P_all = np.concatenate(Y), np.vstack(P)
            per["xgboost"].append(H.r2(y_all, P_all[:, 0]))
            per["random_forest"].append(H.r2(y_all, P_all[:, 1]))
            per["lightgbm"].append(H.r2(y_all, P_all[:, 2]))
            per["mean3"].append(H.r2(y_all, P_all.mean(axis=1)))
            per["stack"].append(H.r2(y_all, np.concatenate(stack_pred)))
        committed = ladder[f"{target}_chemistry_full"]["per_repeat_r2_mean"]
        assert abs(np.mean(per["xgboost"]) - committed) < 1e-9, f"{target}: recomputed XGBoost {np.mean(per['xgboost'])} differs from committed {committed}"
        w = np.array(weights)
        results[target] = {k: {"per_repeat_r2": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in per.items()}
        results[target]["stack_minus_xgboost_mean"] = results[target]["stack"]["mean"] - results[target]["xgboost"]["mean"]
        results[target]["meta_weights_mean_xgb_rf_lgbm_intercept"] = list(map(float, w.mean(axis=0)))
        results[target]["meta_weights_sd"] = list(map(float, w.std(axis=0, ddof=1)))
        print(target, {k: round(results[target][k]["mean"], 4) for k in per}, "stack-xgb", round(results[target]["stack_minus_xgboost_mean"], 4), flush=True)

    (out / "stacking_results.json").write_text(json.dumps({"results": results, "caveat": "not fully nested: see module docstring",
                                                           "inputs": {"rf_dir": str(args.rf_dir), "lgbm_dir": str(args.lgbm_dir), "dataset_sha256": rf_cfg["dataset_sha256"],
                                                                      "rf_git_head": rf_cfg["git_head"], "lgbm_git_head": lg_cfg["git_head"]}},
                                                          indent=2), encoding="utf-8")
    state = H.git_state()
    (out / "run_config.json").write_text(json.dumps({"script": Path(__file__).name, "script_sha256": H.sha256_file(Path(__file__)), "git_head": state[0], "tree_clean": state[1],
                                                     "args": vars(args)}, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
