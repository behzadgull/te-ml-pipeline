"""
Summary of the first random-forest session's S tuning record (results/na2_random_forest_tuning/<stamp>/units/tune_S_trial*.json): how many trials were completed and pruned, the range of the inner-CV R2
of the completed ones with no trial excluded, the same range restricted to deep trees (max_depth >= 15; stated as a second view, not an exclusion), the search-space location of the trials, and the
inner-CV R2 of the frozen XGBoost S model, which was obtained with the same objective (grouped 3-fold chemistry-cluster CV on all S rows) for comparison. No test-fold quantity is involved.

Usage (from the repository root):  python thesis_paper/scripts/na2_tuning_summary.py
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

RUN = "thesis_paper/results/na2_random_forest_tuning/20261004T100611"
XGB_S = "checkpoints/saved_predictions/checkpoints/frozen_hyperparams/S.json"
DEEP = 15


def main():
    """Entry point."""
    files = sorted(glob.glob(str(REPO / RUN / "units" / "tune_S_trial*.json")))
    prov = rr.provenance({"xgboost_S_frozen": XGB_S, **{f"trial_{i:03d}": str(Path(f).relative_to(REPO)) for i, f in enumerate(files)}}, __file__)
    trials = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    trials = [t.get("meta", t) for t in trials]
    done = [t for t in trials if t["state"] == "complete"]
    v = np.array([t["value"] for t in done])
    deep = np.array([t["value"] for t in done if t["params"]["max_depth"] >= DEEP])
    xg = json.loads((REPO / XGB_S).read_text(encoding="utf-8"))
    out = {"run": RUN, "n_trials": len(trials), "n_complete": len(done), "n_pruned": sum(t["state"] == "pruned" for t in trials),
           "inner_cv_r2_all_complete": {"min": float(v.min()), "max": float(v.max()), "range": float(v.max() - v.min()), "mean": float(v.mean()), "sd": float(v.std(ddof=1))},
           "deep_trees_max_depth_ge": DEEP, "n_deep": int(len(deep)),
           "inner_cv_r2_deep": {"min": float(deep.min()), "max": float(deep.max()), "range": float(deep.max() - deep.min())},
           "shallowest_trial": {"max_depth": min(t["params"]["max_depth"] for t in done), "inner_cv_r2": float(min(done, key=lambda t: t["params"]["max_depth"])["value"])},
           "best_trial": {"trial": max(done, key=lambda t: t["value"])["trial_number"], "inner_cv_r2": float(v.max()), "params": max(done, key=lambda t: t["value"])["params"]},
           "trial_hours_total": float(sum(t["seconds"] for t in trials) / 3600),
           "xgboost_S_frozen_inner_cv_r2": xg["inner_cv_r2"], "xgboost_S_n_trials": xg["n_trials"],
           "gap_xgboost_minus_best_random_forest_trial": float(xg["inner_cv_r2"] - v.max())}
    d = REPO / "thesis_paper" / "results" / "na2_random_forest_tuning" / "summary"
    d.mkdir(parents=True, exist_ok=True)
    rr.write_json(d / "tuning_summary.json", out)
    rr.write_json(d / "run_config.json", prov)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
