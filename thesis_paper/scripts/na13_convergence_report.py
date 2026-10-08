"""
NA13 convergence diagnostic: the report of the 20-fold run, written into a new results folder.

`na13_convergence_check.summarize` (the diagnostic's own summary code, pre-registered criteria A to G in docs/decisions.md, 2026-10-07 entry and amendments) is applied to the run folders given in --runs and its
output is written unchanged as `summary.json`. Next to it this script writes what summarize() does not give, all computed from the same `diagnostic.json` files:
  per_fold.csv   one row per outer fold: chosen alpha position and warnings at each setting, whether the final refit converged, the accounting check, the Jaccard similarity of the selected sets, the Lasso
                 non-zero counts, the fold R2 at each setting and its difference
  report.json    the counts quoted in the decisions record: folds at the grid minimum under both settings, folds with an identical chosen position, folds that break C and D, how many chosen alphas lie inside
                 the non-converged range
  run_config.json  provenance: the SHA256 of every input file, the code commit and tree state, this script's SHA256
  README.md      a short text generated from the numbers above

Usage (repository root):
    python thesis_paper/scripts/na13_convergence_report.py --runs <dir>,<dir>,<dir>,<dir>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kaggle"))
sys.path.insert(0, str(REPO))
import run_record as rr  # noqa: E402
import na13_convergence_check as c  # noqa: E402

ORDER = ("S", "sigma", "kappa", "zT")


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of diagnostic run folders (relative to the repository root)")
    args = ap.parse_args()
    dirs = [d for d in args.runs.split(",") if d]
    inputs = {}
    for d in dirs:
        inputs[f"{Path(d).name}/diagnostic.json"] = f"{d}/diagnostic.json"
        inputs[f"{Path(d).name}/run_config.json"] = f"{d}/run_config.json"
    out = rr.new_run_dir("na13_convergence")
    prov = rr.provenance(inputs, __file__)
    summary = c.summarize([str(REPO / d) for d in dirs])
    summary["dirs"] = dirs  # repository-relative, instead of the absolute paths summarize() received
    rr.write_json(out / "summary.json", summary)
    run_cfgs = [json.loads((REPO / d / "run_config.json").read_text(encoding="utf-8")) for d in dirs]
    runs = [json.loads((REPO / d / "diagnostic.json").read_text(encoding="utf-8")) for d in dirs]
    rows = []
    for r in runs:
        for t, tt in r["targets"].items():
            for f, fold in tt["folds"].items():
                rows.append((t, int(f), fold))
    rows.sort(key=lambda x: (ORDER.index(x[0]), x[1]))
    last = None
    table, breaks_c, breaks_d, excluded, identical, inside = [], [], [], [], 0, 0
    for t, f, fold in rows:
        o, n = fold["settings"]["2000"], fold["settings"]["20000"]
        last = len(o["lasso_alphas"]) - 1
        jac = c.jaccard(o["selected"], n["selected"])
        dr2 = n["outer_r2"] - o["outer_r2"]
        both_min = o["lasso_alpha_index"] == last and n["lasso_alpha_index"] == last
        nonconv = [i for g in o["inner_path_checks"] for i in g["nonconverged_alpha_positions"]]
        identical += int(o["lasso_alpha_index"] == n["lasso_alpha_index"])
        inside += int(bool(nonconv) and o["lasso_alpha_index"] >= min(nonconv))
        if jac < 0.6:
            breaks_c.append({"target": t, "fold": f, "jaccard": jac})
        if abs(dr2) > 0.005:
            breaks_d.append({"target": t, "fold": f, "delta_r2": dr2})
        if both_min:
            excluded.append({"target": t, "fold": f})
        table.append({"target": t, "fold": f, "alpha_position_2000": o["lasso_alpha_index"], "alpha_position_20000": n["lasso_alpha_index"], "abs_delta_log10_alpha": abs(np.log10(n["lasso_alpha"]) - np.log10(o["lasso_alpha"])),
                      "at_grid_minimum_under_both": both_min, "warnings_2000": o["n_convergence_warnings"], "warnings_20000": n["n_convergence_warnings"],
                      "inner_nonconverged_by_gap_2000": o["n_nonconverged_by_gap_total"], "inner_nonconverged_by_gap_20000": n["n_nonconverged_by_gap_total"],
                      "final_refit_converged_2000": o["final_refit_converged"], "final_refit_converged_20000": n["final_refit_converged"],
                      "warnings_equal_inner_plus_refit_2000": o["warnings_equal_inner_gap_plus_final_refit"], "warnings_equal_inner_plus_refit_20000": n["warnings_equal_inner_gap_plus_final_refit"],
                      "jaccard_selected": jac, "n_after_lasso_2000": o["n_after_lasso"], "n_after_lasso_20000": n["n_after_lasso"], "n_selected_2000": len(o["selected"]), "n_selected_20000": len(n["selected"]),
                      "outer_r2_2000": o["outer_r2"], "outer_r2_20000": n["outer_r2"], "delta_r2_20000_minus_2000": dr2})
    with open(out / "per_fold.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    ab = [abs(x["delta_r2_20000_minus_2000"]) for x in table]
    report = {"n_folds": len(table), "targets": sorted({x["target"] for x in table}), "alpha_grid_size": last + 1,
              "alpha_position_identical_folds": identical, "alpha_position_changed_folds": [{"target": x["target"], "fold": x["fold"], "from": x["alpha_position_2000"], "to": x["alpha_position_20000"]} for x in table if x["alpha_position_2000"] != x["alpha_position_20000"]],
              "folds_at_grid_minimum_under_both_settings": excluded, "folds_breaking_C_jaccard_below_0.60": breaks_c, "folds_breaking_D_abs_delta_r2_above_0.005": breaks_d,
              "mean_abs_delta_r2": float(np.mean(ab)), "max_abs_delta_r2": float(np.max(ab)), "mean_signed_delta_r2": float(np.mean([x["delta_r2_20000_minus_2000"] for x in table])),
              "folds_with_chosen_alpha_inside_the_nonconverged_range_at_2000": inside, "run_configs": [{"dir": d, "git_head": r["git_head"], "tree_clean": r["tree_clean"], "targets": r["targets"], "max_iters": r["max_iters"], "script_sha256": r["script_sha256"]} for d, r in zip(dirs, run_cfgs)]}
    rr.write_json(out / "report.json", report)
    rr.write_json(out / "run_config.json", {**prov, "runs": dirs})
    A, B = summary["A"], summary["B_C_D"]["20000"]
    (out / "README.md").write_text(
        "# NA13 convergence diagnostic: report of the 20-fold run (2000 and 20,000 iterations)\n\n"
        f"Generated by `scripts/na13_convergence_report.py` from the four run folders {', '.join(Path(d).name for d in dirs)} (targets {', '.join(report['targets'])}; "
        f"each `run_config.json`: commit {run_cfgs[0]['git_head'][:7]}, tree_clean {all(r['tree_clean'] for r in run_cfgs)}). `summary.json` is the output of `na13_convergence_check.summarize`, unchanged except that its `dirs` entry lists the repository-relative folder names; "
        "`per_fold.csv` and `report.json` hold the per-fold values and counts; `run_config.json` records the inputs' SHA256 and the code state of THIS report "
        f"(commit {prov['git_head'][:7]}, tree_clean {prov['tree_clean']}{'' if prov['tree_clean'] else ', dirty files: ' + ', '.join(prov['dirty_files'])}).\n\n"
        f"- **A**: at 2000 iterations {A['2000']['folds_with_warnings']} of {summary['n_folds']} folds warn ({A['2000']['total_warnings']} warnings; {A['2000']['total_nonconverged_fits_by_gap']} non-converged inner-path fits; "
        f"final refit not converged in {A['2000']['folds_with_final_refit_not_converged (independent check)']} folds); at 20,000 {A['20000']['folds_with_warnings']} folds warn, so `adopted_max_iter` is {A['adopted_max_iter']}. "
        f"The warning accounting (warnings = inner-path non-converged + final refit) holds in every fold and setting: {A['2000']['warnings_equal_inner_gap_plus_final_refit_in_every_fold'] and A['20000']['warnings_equal_inner_gap_plus_final_refit_in_every_fold']}.\n"
        f"- **B**: {B['B_folds_counted']} folds counted, {len(excluded)} excluded (chosen alpha at the grid minimum under both settings); {B['B_alpha_within_one_grid_step_folds']} of {B['B_folds_counted']} within one grid step (threshold {B['B_threshold_folds']}), holds: {B['B_holds']}. "
        f"The chosen alpha position is identical in {identical} of {len(table)} folds; changed: {', '.join(str(x['target']) + ' fold ' + str(x['fold']) + ' (' + str(x['from']) + ' to ' + str(x['to']) + ')' for x in report['alpha_position_changed_folds'])}.\n"
        f"- **C**: Jaccard median {B['C_jaccard_median']:.3f}, minimum {B['C_jaccard_min']:.4f}, `n_after_lasso` relative change median {B['C_n_after_lasso_relative_change_median']:.3f}; holds: {B['C_holds']}; "
        f"folds below 0.60: {', '.join(str(x['target']) + ' fold ' + str(x['fold']) for x in breaks_c) or 'none'}.\n"
        f"- **D**: maximum |dR2| {B['D_max_abs_r2_difference']:.4f}, mean |dR2| {B['D_mean_abs_r2_difference']:.4f}, mean signed dR2 {B['D_mean_r2_difference_new_minus_old']:+.4f}; holds: {B['D_holds']}; "
        f"folds above 0.005: {', '.join(str(x['target']) + ' fold ' + str(x['fold']) for x in breaks_d) or 'none'}.\n"
        f"- **E, F, G (report-only)**: `summary.json` keys `E`, `F`, `E2_alpha_position`; the chosen alpha lies inside the non-converged range at 2000 iterations in {inside} of {len(table)} folds.\n",
        encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
