"""
The number list of the nested-stacking (Table 7, Section 4.1.3) and explicit-feature-selection (Table 7b, Section 4.1.5) drafts: every number the new or edited
sentences and the two tables take from a generated value, with the value, its source file and the field it is read from.

paper.md holds the generated markers; this script lists them (inline markers of the listed passages, and the keys of Tables 7 and 7b through
audit_numbers.block_traces) and maps each key to its source by pattern. A key the mapping does not know prints as "?" and the script exits non-zero, so a new
key cannot be listed without a source. The source folders are read from thesis_values.py's constants, so a rerun of an analysis changes one place.

Usage (from the repository root):  python thesis_paper/scripts/number_list_na2_na13.py   ->  thesis_paper/reports/number_list_na2_na13.csv
"""

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import audit_numbers as an  # noqa: E402
import make_thesis_values as mtv  # noqa: E402
import thesis_values as tv  # noqa: E402

paper = (REPO / "thesis_paper/paper/paper.md").read_text(encoding="utf-8").replace(chr(13) + chr(10), chr(10))
v = tv.values()
v.update(mtv.extra_values(v))

C = f"{tv.NA2_CMP}/comparison.json"
CR = f"{tv.NA2_CMP}/run_config.json"
W = f"{tv.NA2_STACK_WEIGHTS}/summary.json"
A = f"{tv.NA13_ANALYSIS}/analysis.json"
F = f"{tv.NA13_ANALYSIS}/selection_frequency.csv"
T = f"{tv.NA13_TEMP}/summary.json"
CV = f"{tv.NA13_CONV}/summary.json"
CVR = f"{tv.NA13_CONV}/report.json"
P = "thesis_paper/results/na13_{S,sigma,kappa,zT}/<stamp>/run_configs/session_01.json"
MODELS = {"xgb": "xgboost_frozen", "lgbm": "lightgbm", "rf": "random_forest", "m3": "mean_of_three"}
TT = r"(S|sigma|kappa|zT)"


def source(k):  # noqa: C901
    m = re.fullmatch(rf"a2_stk_{TT}", k)
    if m: return C, f"targets.{m[1]}.models.stacking.mean"
    m = re.fullmatch(rf"a2_stk_{TT}_sd", k)
    if m: return C, f"targets.{m[1]}.models.stacking.sd"
    m = re.fullmatch(rf"a2_stk_{TT}_(d|lo|hi)", k)
    if m: return C, f"targets.{m[1]}.differences_vs_reference.stacking." + {"d": "mean_diff", "lo": "ci95_cluster_bootstrap[0]", "hi": "ci95_cluster_bootstrap[1]"}[m[2]]
    m = re.fullmatch(rf"a2_stkbest_{TT}_(d|lo|hi)", k)
    if m: return C, f"targets.{m[1]}.extra_pairs.stacking_minus_<targets.{m[1]}.best_single>." + {"d": "mean_diff", "lo": "ci95_cluster_bootstrap[0]", "hi": "ci95_cluster_bootstrap[1]"}[m[2]]
    m = re.fullmatch(rf"a2_stkbest_{TT}_name", k)
    if m: return C, f"targets.{m[1]}.best_single (shown as a name)"
    m = re.fullmatch(rf"a2_stkmean_{TT}_(d|lo|hi)", k)
    if m: return C, f"targets.{m[1]}.extra_pairs.stacking_minus_mean_of_three." + {"d": "mean_diff", "lo": "ci95_cluster_bootstrap[0]", "hi": "ci95_cluster_bootstrap[1]"}[m[2]] + " (four decimals)"
    if k == "a2_stk_dmin": return C, "min over targets of differences_vs_reference.stacking.mean_diff"
    if k == "a2_stk_dmax": return C, "max over targets of differences_vs_reference.stacking.mean_diff"
    if k == "a2_stkmean_amax": return C, "max over targets of |extra_pairs.stacking_minus_mean_of_three.mean_diff| (four decimals)"
    if k == "a2_w_min": return W, "overall.weight_min"
    if k == "a2_w_max": return W, "overall.weight_max"
    if k == "a2_w_corr_min": return W, "overall.min_pairwise_prediction_correlation (three decimals)"
    if k == "a2_w_range_min": return W, "min over targets of (max over models of targets.<t>.weights.<m>.range_over_folds)"
    if k == "a2_w_range_max": return W, "max over targets of (max over models of targets.<t>.weights.<m>.range_over_folds)"
    if k == "a2_nfolds": return W, "n_folds_per_target"
    if k == "a2_nfits": return W, "n_folds_per_target x 4 targets"
    m = re.fullmatch(rf"a2_(xgb|lgbm|rf|m3)_{TT}(_sd)?", k)
    if m: return C, f"targets.{m[2]}.models.{MODELS[m[1]]}." + ("sd" if m[3] else "mean")
    m = re.fullmatch(rf"a2_(lgbm|rf|m3)_{TT}_(d|lo|hi)", k)
    if m: return C, f"targets.{m[2]}.differences_vs_reference.{MODELS[m[1]]}." + {"d": "mean_diff", "lo": "ci95_cluster_bootstrap[0]", "hi": "ci95_cluster_bootstrap[1]"}[m[3]]
    if k in ("a2_n_top", "a2_n_excl", "a2_n_diffs", "a2_maxgap") or re.fullmatch(r"a2_(lgbm|rf|m3)_(dmin|dmax|n)", k):
        return C, "counts or extremes over targets of differences_vs_reference.*.mean_diff and ci95_cluster_bootstrap (thesis_values.na2_na1_values; unchanged value)"
    if k == "n_repeats": return C, "len(targets.S.models.xgboost_frozen.per_repeat_r2)"
    if k == "n_folds": return CR, "n_folds"
    if k in ("optuna_trials", "inner_folds", "n_feat"): return "paper_a_values.py (Paper A frozen hyperparameters / ladder metrics)", "pre-existing key, value unchanged"
    # NA13
    m = re.fullmatch(rf"na13_{TT}_(all|all_sd|sel|sel_sd)", k)
    if m: return A, f"targets.{m[1]}.pooled_per_repeat_r2." + {"all": "all397_mean", "all_sd": "all397_sd", "sel": "selected_mean", "sel_sd": "selected_sd"}[m[2]]
    m = re.fullmatch(rf"na13_{TT}_(d|lo|hi)", k)
    if m: return A, f"targets.{m[1]}.paired_difference_selected_minus_all397." + {"d": "mean", "lo": "ci95_cluster_bootstrap[0]", "hi": "ci95_cluster_bootstrap[1]"}[m[2]]
    m = re.fullmatch(rf"na13_{TT}_k", k)
    if m: return A, f"targets.{m[1]}.counts_over_the_25_folds.k_thesis"
    m = re.fullmatch(rf"na13_{TT}_nl(_min|_max)?", k)
    if m: return A, f"targets.{m[1]}.counts_over_the_25_folds.n_after_lasso." + {None: "mean", "_min": "min", "_max": "max"}[m[2]]
    m = re.fullmatch(rf"na13_{TT}_np", k)
    if m: return A, f"targets.{m[1]}.counts_over_the_25_folds.n_after_pearson.mean"
    if k in ("na13_np_min", "na13_np_max"): return A, "min / max over targets of counts_over_the_25_folds.n_after_pearson.mean"
    if k in ("na13_ratio_min", "na13_ratio_max"): return A, "min / max over targets of counts_over_the_25_folds.n_after_lasso.mean / k_thesis"
    m = re.fullmatch(rf"na13_{TT}_gridmin", k)
    if m: return A, f"targets.{m[1]}.alpha_grid_position.units_at_grid_minimum"
    if k in ("na13_gridmin_units", "na13_units", "na13_gridmin_pct"): return A, "overall_100_units." + {"na13_gridmin_units": "units_with_chosen_alpha_at_grid_minimum", "na13_units": "units", "na13_gridmin_pct": "share (x100)"}[k]
    if k == "na13_nfolds": return A, "targets.S.per_fold_delta_r2.n_folds"
    if k == "na13_maxdiff": return A, "max over targets of committed_rung_alongside_not_used_for_the_claim.all397_this_machine_minus_committed_per_fold.max_abs"
    if k in ("na13_warn_n", "na13_warn_target", "na13_warn_repeat", "na13_warn_fold"): return A, "targets.*.convergence.units_with_warnings[0] (warnings / target / repeat / fold)"
    if k in ("na13_pearson", "na13_alphas", "na13_mi_rows", "na13_maxiter"): return P, "params." + {"na13_pearson": "pearson_max", "na13_alphas": "n_alphas", "na13_mi_rows": "mi_rows", "na13_maxiter": "lasso_max_iter"}[k]
    m = re.fullmatch(rf"na13_{TT}_temp(_sd|_d|_share|_below)?", k)
    if m:
        f = {None: "pooled_per_repeat_r2.selected_plus_temperature_mean", "_sd": "pooled_per_repeat_r2.selected_plus_temperature (sample SD of the 5 per-repeat values)",
             "_d": "pooled_per_repeat_r2.selected_plus_temperature_mean minus analysis.json all397_mean", "_share": "(selected_plus_temperature_mean - selected_mean) / (all397_mean - selected_mean) x 100, means from summary and analysis",
             "_below": "n_folds (25) minus n_folds_selected_plus_temperature_above_all397"}[m[2]]
        return T, f"targets.{m[1]}.{f}"
    if k == "na13_conv_lo": return CV, "min(ladder)"
    if k == "na13_conv_folds": return CV, "n_folds"
    if k == "na13_conv_warn_2000": return CV, "A.2000.total_warnings"
    if k == "na13_conv_warn_20000": return CV, "A.20000.total_warnings"
    if k == "na13_conv_warn_folds_2000": return CV, "A.2000.folds_with_warnings"
    if k == "na13_conv_jac_med": return CV, "B_C_D.20000.C_jaccard_median"
    if k == "na13_conv_jac_min": return CV, "B_C_D.20000.C_jaccard_min"
    if k == "na13_conv_d_mean": return CV, "B_C_D.20000.D_mean_abs_r2_difference"
    if k == "na13_conv_d_n": return CVR, "len(folds_breaking_D_abs_delta_r2_above_0.005)"
    if k == "na13_conv_d_thr": return CVR, "the threshold in the key name folds_breaking_D_abs_delta_r2_above_<thr>"
    return "?", "?"


def cut(start, end):
    i = paper.index(start)
    return paper[i:paper.index(end, i) + len(end)]


text_sections = {
    "Introduction/contributions bullet": cut("shown to depend little on the learner", "on every target;"),
    "3.3": cut("no explicit feature selection is applied", "(Section 4.1.5)."),
    "3.4": cut("and no single model differed from it", "(Section 4.1.3)."),
    "4.1.3 text": cut("a small gain from averaging. The stack", "not that no learner could do better."),
    "4.1.5 text": paper[paper.index("### 4.1.5"):paper.index("<!-- BEGIN TABLE 7b -->")],
    "Conclusions (RQ1)": cut("A nested stack of XGBoost, LightGBM and a random forest", "(RQ1)"),
}
rows, seen = [], set()
for where, txt in text_sections.items():
    for k in dict.fromkeys(re.findall(r"<!--v:(\w+)-->", txt)):
        rows.append((where, k, v[k], *source(k)))
for where, txt in text_sections.items():
    for k in dict.fromkeys(re.findall(r"<!--c:(\w+)-->", txt)):
        rows.append((where, k, "0.0062", "docs/decisions.md (2026-10-09 pre-registration, item 3); registry row na13_platform_delta, class CITED", "internal record; run output not committed"))
tr = an.block_traces(paper)
for n, label in (("7", "Table 7"), ("7b", "Table 7b")):
    for k in dict.fromkeys(key for (_, _, key) in tr[n]):
        rows.append((label, k, v[k], *source(k)))
out = REPO / "thesis_paper" / "reports" / "number_list_na2_na13.csv"
with open(out, "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["where", "key", "value", "source_file", "field"])
    w.writerows(rows)
unknown = [r[1] for r in rows if r[3] == "?"]
print("rows", len(rows), "unknown", unknown, "->", out)
sys.exit(1 if unknown else 0)
