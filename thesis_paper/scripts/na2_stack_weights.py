"""
NA2, nested stacking: what the fitted meta-learner weights look like, and how alike the three base models' outer-fold predictions are.

The stacking section of the paper does not interpret the weights; this script produces the two facts that justify saying so:
  1. the non-negative ridge weights of XGBoost, LightGBM and the random forest in the 25 outer folds of every target (mean, SD, minimum, maximum, and the
     range of each weight over the folds), read from the 100 `stack_<target>_repeat<r>_fold<f>.json` units of the ten committed bundles `results/na2_stk_*`;
  2. the Pearson correlation of each pair of base models' outer-fold predictions within every fold (in the scored space of the target: log10 for sigma
     and kappa), read from the same committed predictions that `scripts/model_comparison.py na2` uses.
It computes nothing new about any model: no refit, no bootstrap. Every weight unit is checked against the SHA256 recorded by
`na2_stacking_analysis.py` in `results/na2_stacking_preds/<stamp>/run_config.json`, so it is the same set of fits whose outer predictions Table 7 scores.

Usage (from the repository root):
    python thesis_paper/scripts/na2_stack_weights.py [--out-dir thesis_paper/results/na2_stack_weights/<utc>]
Output: summary.json and run_config.json (inputs, script SHA256, git state, SHA256 of every unit read).
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import model_comparison as mc  # noqa: E402
import run_record as rr  # noqa: E402

STACKING_PREDS = "thesis_paper/results/na2_stacking_preds/20261009T190106"
TARGETS = mc.TARGETS
MODELS = ("xgboost", "lightgbm", "random_forest")
N_REPEATS, N_FOLDS = mc.N_REPEATS, mc.N_FOLDS


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    cfg = json.loads((REPO / STACKING_PREDS / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["tree_clean"] and cfg["stack_dataset_sha256"] == mc.DATASET_SHA256, "the stacking predictions were not made from a clean tree on the snapfix CSV"
    bundles = [REPO / Path(p.replace("\\", "/")) for p in cfg["stack_bundles"]]
    unit_sha = {}
    weights = {t: [] for t in TARGETS}
    for t in TARGETS:
        for r in range(N_REPEATS):
            for f in range(N_FOLDS):
                name = f"stack_{t}_repeat{r}_fold{f}"
                hits = [b / "units" / f"{name}.json" for b in bundles if (b / "units" / f"{name}.json").exists()]
                assert len(hits) == 1, f"{name}: found in {len(hits)} bundles, expected exactly one"
                p = hits[0]
                sha = rr.sha256_file(p)
                assert sha == cfg["weight_unit_sha256"][name], f"{name}: SHA256 differs from the one na2_stacking_analysis.py recorded"
                unit_sha[str(p.relative_to(REPO)).replace("\\", "/")] = sha
                m = json.loads(p.read_text(encoding="utf-8"))
                m = m.get("meta", m)
                assert m["nested_assertions"] == "passed" and (m["target"], m["repeat"], m["fold"]) == (t, r, f)
                weights[t].append(m["coef_xgboost_lightgbm_random_forest"])
    out = {"analysis": "na2_stack_weights", "stack_predictions_run": STACKING_PREDS, "n_folds_per_target": N_REPEATS * N_FOLDS, "targets": {}}
    rf_runs = mc.BUNDLES["na2"]["random_forest"]
    lg_runs = mc.BUNDLES["na2"]["lightgbm"]
    for t in TARGETS:
        W = np.array(weights[t], dtype=float)
        assert W.shape == (N_REPEATS * N_FOLDS, 3) and (W >= 0).all()
        pred = {m: [] for m in MODELS}
        for r in range(N_REPEATS):
            for f in range(N_FOLDS):
                srcs = {"xgboost": REPO / mc.LADDER / f"{t}_chemistry_full" / f"repeat{r}_fold{f}_predictions.npz",
                        "lightgbm": REPO / lg_runs[t] / "units" / f"rung_{t}_repeat{r}_fold{f}.npz",
                        "random_forest": REPO / rf_runs[t] / "units" / f"rung_{t}_repeat{r}_fold{f}.npz"}
                yt = None
                for m, p in srcs.items():
                    u = np.load(p)
                    unit_sha[str(p.relative_to(REPO)).replace("\\", "/")] = rr.sha256_file(p)
                    yt = u["y_true"] if yt is None else yt
                    assert np.allclose(u["y_true"], yt), f"{t} {m} repeat {r} fold {f}: y_true differs between the base models"
                    pred[m].append(np.asarray(u["y_pred"], dtype=float))
        corr = {}
        for a, b in itertools.combinations(MODELS, 2):
            c = np.array([np.corrcoef(pred[a][i], pred[b][i])[0, 1] for i in range(N_REPEATS * N_FOLDS)])
            corr[f"{a}__{b}"] = {"min": float(c.min()), "mean": float(c.mean()), "max": float(c.max())}
        out["targets"][t] = {
            "weights": {m: {"mean": float(W[:, i].mean()), "sd": float(W[:, i].std(ddof=1)), "min": float(W[:, i].min()), "max": float(W[:, i].max()),
                            "range_over_folds": float(W[:, i].max() - W[:, i].min())} for i, m in enumerate(MODELS)},
            "weight_sum": {"min": float(W.sum(1).min()), "max": float(W.sum(1).max())},
            "pairwise_prediction_correlation_per_fold": corr,
            "min_pairwise_prediction_correlation": float(min(v["min"] for v in corr.values())),
        }
    allw = np.concatenate([np.array(weights[t], dtype=float).ravel() for t in TARGETS])
    out["overall"] = {"weight_min": float(allw.min()), "weight_max": float(allw.max()), "n_weights": int(allw.size),
                      "min_pairwise_prediction_correlation": float(min(out["targets"][t]["min_pairwise_prediction_correlation"] for t in TARGETS))}
    prov = rr.provenance({"stacking_preds_run_config": f"{STACKING_PREDS}/run_config.json", "stacking_preds_manifest": f"{STACKING_PREDS}/manifest.json"}, __file__)
    d = Path(args.out_dir) if args.out_dir else REPO / "thesis_paper" / "results" / "na2_stack_weights" / rr.utc_stamp()
    d.mkdir(parents=True, exist_ok=True)
    rr.write_json(d / "summary.json", out)
    rr.write_json(d / "run_config.json", {**prov, "n_units_read": len(unit_sha), "unit_sha256": unit_sha})
    print("wrote", d)
    for t in TARGETS:
        x = out["targets"][t]
        print(t, "weight mean", [round(x["weights"][m]["mean"], 3) for m in MODELS], "min", [round(x["weights"][m]["min"], 3) for m in MODELS],
              "max", [round(x["weights"][m]["max"], 3) for m in MODELS], "min corr", round(x["min_pairwise_prediction_correlation"], 4))


if __name__ == "__main__":
    main()
