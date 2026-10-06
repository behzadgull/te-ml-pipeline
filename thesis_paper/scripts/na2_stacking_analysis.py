"""
NA2, nested stacking, step 2 (local, seconds): apply the meta-learner weights learned by scripts/kaggle/na2_stacking_nested.py to the base models' outer-fold predictions.

For every target, repeat and outer fold, the weights (non-negative ridge coefficients for XGBoost, LightGBM, random forest, and an intercept) were fitted on INNER out-of-fold predictions inside
that outer training set only (asserted in the Kaggle script), so the stacked prediction of the outer test rows is
        y_stack = w_xgb * p_xgb + w_lgbm * p_lgbm + w_rf * p_rf + intercept
with p_* the committed outer-fold predictions of the three base models (fits on the full outer training rows with the same frozen hyperparameters). No outer test label enters the weights.
Output: <out-dir>/units/rung_<target>_repeat<r>_fold<f>.npz with y_true and y_pred (the format scripts/model_comparison.py reads), manifest.json (SHA256 of every file), run_config.json
(inputs, script SHA256, git state). Then:  python thesis_paper/scripts/model_comparison.py na2 --stacking-dir <out-dir>

Usage (from the repository root):
    python thesis_paper/scripts/na2_stacking_analysis.py --stack-dirs <bundle dir or .tar.gz>,<...> [--out-dir thesis_paper/results/na2_stacking_preds/<utc>]
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kaggle"))
import harness as H  # noqa: E402
import model_comparison as mc  # noqa: E402
import na2_stacking as st  # noqa: E402

N_REPEATS, N_FOLDS = 5, 5


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-dirs", required=True, help="the na2_stacking_nested.py bundles (directories or .tar.gz), comma-separated; together they must cover every target, repeat and fold once")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    parts = [st.open_bundle(x, tmp) for x in args.stack_dirs.split(",") if x]
    cfgs = [c for _, c in parts]
    assert len({c["dataset_sha256"] for c in cfgs}) == 1 and len({c["git_head"] for c in cfgs}) == 1, "the bundles differ in dataset or code commit"
    assert all(c["tree_clean"] and not c["smoke"] for c in cfgs), "a bundle is from a dirty tree or a smoke run"
    assert len({json.dumps({k: v for k, v in c["params"].items() if k not in ("targets", "repeats")}, sort_keys=True) for c in cfgs}) == 1, "the bundles differ in design parameters"
    where = {}
    for p, _ in parts:
        for u in (p / "units").glob("stack_*.json"):
            assert u.stem not in where, f"unit {u.stem} is in two bundles"
            where[u.stem] = u
    out = Path(args.out_dir) if args.out_dir else REPO / "thesis_paper" / "results" / "na2_stacking_preds" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    (out / "units").mkdir(parents=True, exist_ok=True)
    meta_sha = {}
    for target in mc.TARGETS:
        for r in range(N_REPEATS):
            for f in range(N_FOLDS):
                uid = f"stack_{target}_repeat{r}_fold{f}"
                assert uid in where, f"missing {uid}"
                w = json.loads(where[uid].read_text(encoding="utf-8"))["meta"]
                assert w["nested_assertions"] == "passed" and (w["target"], w["repeat"], w["fold"]) == (target, r, f)
                coef, icpt = np.array(w["coef_xgboost_lightgbm_random_forest"]), w["intercept"]
                x = np.load(REPO / mc.LADDER / f"{target}_chemistry_full" / f"repeat{r}_fold{f}_predictions.npz")
                yt_l, p_l = mc.load_preds("na2", "lightgbm", target, mc.BUNDLES["na2"]["lightgbm"][target], r, f)
                yt_r, p_r = mc.load_preds("na2", "random_forest", target, mc.BUNDLES["na2"]["random_forest"][target], r, f)
                assert np.allclose(x["y_true"], yt_l, rtol=0, atol=1e-6) and np.allclose(x["y_true"], yt_r, rtol=0, atol=1e-6), f"{uid}: y_true differs between models"
                assert len(x["y_true"]) == w["n_test"], f"{uid}: the fold size differs from the one the weights were fitted for"
                pred = np.column_stack([x["y_pred"], p_l, p_r]) @ coef + icpt
                np.savez_compressed(out / "units" / f"rung_{target}_repeat{r}_fold{f}.npz", y_true=x["y_true"], y_pred=pred)
                meta_sha[uid] = H.sha256_file(where[uid])
    files = {str(p.relative_to(out)).replace("\\", "/"): H.sha256_file(p) for p in sorted((out / "units").glob("*.npz"))}
    (out / "manifest.json").write_text(json.dumps({"files": files}, indent=1), encoding="utf-8")
    state = H.git_state()
    (out / "run_config.json").write_text(json.dumps({"script": Path(__file__).name, "script_sha256": H.sha256_file(Path(__file__)), "git_head": state[0], "tree_clean": state[1],
                                                     "stack_bundles": [str(p) for p, _ in parts], "stack_dataset_sha256": cfgs[0]["dataset_sha256"], "stack_code_commit": cfgs[0]["git_head"],
                                                     "weight_unit_sha256": meta_sha}, indent=1), encoding="utf-8")
    print("wrote", out, len(files), "units")


if __name__ == "__main__":
    main()
