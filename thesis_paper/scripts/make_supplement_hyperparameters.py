"""
Supplementary Table S2: the hyperparameters of the two comparison models of Section 3.1.2, generated from the committed frozen-hyperparameter files of the NA2 runs (nothing is typed here).
Part A: LightGBM, tuned once per target by the XGBoost protocol (results/na2_lightgbm/<run>/frozen/<target>_lightgbm.json), with the search space read from src/nested_cv.py
(_lightgbm_search_space, parsed with ast). Part B: the random forest, one fixed setting for every target (results/na2_rf_<target>/<run>/frozen/<target>_random_forest.json); the script asserts that
the four files hold the same setting. Writes thesis_paper/paper/supplement_hyperparameters.md.

Usage (repository root):  python thesis_paper/scripts/make_supplement_hyperparameters.py
"""

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

OUT = REPO / "thesis_paper" / "paper" / "supplement_hyperparameters.md"
LGBM_DIR = "thesis_paper/results/na2_lightgbm/20261005T101837/frozen"
RF_DIR = {"S": "na2_rf_S/20261005T100927", "sigma": "na2_rf_sigma/20261005T102059", "kappa": "na2_rf_kappa/20261005T101358", "zT": "na2_rf_zT/20261005T101629"}
TARGETS = ("S", "sigma", "kappa", "zT")
HEAD = {"S": "S", "sigma": "*σ*", "kappa": "*κ*", "zT": "zT"}
LGBM_KEYS = ("n_estimators", "max_depth", "num_leaves", "learning_rate", "subsample", "colsample_bytree", "min_child_samples", "reg_lambda", "reg_alpha")
RF_KEYS = ("n_estimators", "max_features", "min_samples_leaf", "max_depth", "min_samples_split", "bootstrap")


def fmt(x):
    """Integers as they are, floats to three significant figures, None as 'none (no limit)'."""
    if x is None:
        return "none (no limit)"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{x:.3g}"


def lightgbm_ranges():
    """The search ranges of _lightgbm_search_space in src/nested_cv.py, read with ast: {name: 'low to high' (log scale / step)}."""
    tree = ast.parse((REPO / "src" / "nested_cv.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_lightgbm_search_space")
    out = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("suggest_int", "suggest_float"):
            name, lo, hi = (ast.literal_eval(a) for a in node.args[:3])
            kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            s = f"{lo:g} to {hi:g}"
            if kw.get("log"):
                s += " (log scale)"
            if "step" in kw:
                s += f" (step {kw['step']})"
            out[name] = s
    assert set(out) == set(LGBM_KEYS), out
    return out


def main():
    """Write the supplement table."""
    lg = {t: json.loads((REPO / LGBM_DIR / f"{t}_lightgbm.json").read_text(encoding="utf-8")) for t in TARGETS}
    rf = {t: json.loads((REPO / "thesis_paper" / "results" / RF_DIR[t] / "frozen" / f"{t}_random_forest.json").read_text(encoding="utf-8")) for t in TARGETS}
    assert all(lg[t]["model_type"] == "lightgbm" and lg[t]["n_trials"] == lg["S"]["n_trials"] and lg[t]["n_inner_folds"] == lg["S"]["n_inner_folds"] for t in TARGETS)
    assert all(rf[t]["model_type"] == "random_forest" and rf[t]["tuned"] is False for t in TARGETS)
    assert all(rf[t]["best_params"] == rf["S"]["best_params"] for t in TARGETS), "the random-forest setting differs between targets"
    rng = lightgbm_ranges()
    inputs = {f"lightgbm_{t}": f"{LGBM_DIR}/{t}_lightgbm.json" for t in TARGETS}
    inputs.update({f"random_forest_{t}": f"thesis_paper/results/{RF_DIR[t]}/frozen/{t}_random_forest.json" for t in TARGETS})
    inputs["nested_cv"] = "src/nested_cv.py"
    prov = rr.provenance(inputs, __file__)
    rr.write_json(REPO / "thesis_paper" / "reports" / "supplement_hyperparameters" / "run_config.json", prov)
    n_tr, n_in = lg["S"]["n_trials"], lg["S"]["n_inner_folds"]
    out = [f"**Supplementary Table [[stbl:hyper]]:** Hyperparameters of the comparison models of Section 3.1.2. Part A: LightGBM, tuned once per target on all rows of the target by a {n_tr}-trial Optuna search scored with {n_in}-fold chemistry-cluster "
           "cross-validation (the XGBoost protocol), then frozen; the search ranges are those of the code. Part B: the random forest, which is not tuned: one setting, fixed in advance, is used for all four targets (Section 2.3). "
           "The target scale is log~10~ for *σ* and *κ* and linear for S and zT.", "",
           "*Part A: LightGBM (tuned, frozen).*", "",
           "| **Parameter** | **S** | ***σ*** | ***κ*** | **zT** | **Search range** |", "|---------------------|-----------|-----------|-----------|-----------|------------------|"]
    for k in LGBM_KEYS:
        out.append(f"| {k} | " + " | ".join(fmt(lg[t]["best_params"][k]) for t in TARGETS) + f" | {rng[k]} |")
    out.append("| inner-CV R^2^ of the chosen set | " + " | ".join(fmt(lg[t]["inner_cv_r2"]) for t in TARGETS) + " | |")
    out.append("| rows of the target | " + " | ".join(f"{lg[t]['n_rows']:,}" for t in TARGETS) + " | |")
    out += ["", "*Part B: random forest (fixed setting, not tuned; the same for S, σ, κ and zT).*", "", "| **Parameter** | **Value** |", "|---------------------|-----------|"]
    for k in RF_KEYS:
        out.append(f"| {k} | {fmt(rf['S']['best_params'][k])} |")
    out += ["", "Generated by `scripts/make_supplement_hyperparameters.py` from the frozen files of `results/na2_lightgbm` and `results/na2_rf_*` and from `src/nested_cv.py` "
            "(the SHA256 of every input is in `reports/supplement_hyperparameters/run_config.json`)."]
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
