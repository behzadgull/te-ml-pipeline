"""
Every number the thesis paper's prose and tables take from Paper A, as formatted strings, read from the committed artifacts
through paper_a_values.py (never typed). `values()` returns {key: text}; make_thesis_values.py writes them into paper.md.

Naming: T is S, sigma, kappa or zT. R2 values are given to three decimals in prose and tables; standard deviations are
across-repeat SDs of per-repeat pooled R2 (grouped rungs) or per-fold/per-draw SDs (ungrouped rungs), as in Paper A.
"""

import math
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_a_values as pav  # noqa: E402

T4 = pav.TARGETS
REPO_ROOT = Path(__file__).resolve().parents[2]
NAME = {"S": "S", "sigma": "σ", "kappa": "κ", "zT": "zT"}


def _r3(x):
    return f"{x:.3f}"


def _n(x):
    return f"{int(x):,}"


def values():
    """All Paper A-derived values, with assertions that tie related numbers together."""
    j = pav._json
    v = {}
    ladder, ung, abl = j(pav.LADDER), j(pav.UNGROUPED), j(pav.ABLATION)
    funnel = {int(s["step"].split("_")[0]): s["rows"] for s in j(pav.FUNNEL)["funnel"]}
    grp = j(pav.GROUPING)
    raw = j(pav.RAWMETA)

    # ---- data
    v["n_clean"] = _n(funnel[11])
    for t in T4:
        v[f"n_{t}"] = _n(ladder["runs"][f"{t}_chemistry_full"]["n_rows_header"])
    v["n_feat"] = pav.get("features.full")[0].split(" ")[0]
    v["n_magpie"] = pav.get("features.magpie")[0].split(" ")[0]
    v["n_cbfv"] = pav.get("features.cbfv")[0].split(" ")[0]
    assert int(v["n_magpie"]) + int(v["n_cbfv"]) + 1 == int(v["n_feat"])
    v["n_clusters"] = _n(grp["total_chemistry_clusters"]["new"])
    v["n_formulas"] = _n(grp["unique_formulas"]["new"])
    v["n_parent"] = _n(grp["parent_system_groups"]["new"])
    v["n_featurized"] = _n(grp["n_rows_total"])
    f = raw["files"]
    v["raw_papers"] = _n(f["papers"]["counted_row_count"])
    v["raw_samples"] = _n(f["samples"]["counted_row_count"])
    v["raw_curves"] = _n(f["curves"]["counted_row_count"])
    snap = re.match(r"(\d{4}-\d{2}-\d{2})", raw["upstream_db_snapshot"]).group(1)
    v["snapshot_date"] = datetime.strptime(snap, "%Y-%m-%d").strftime("%d %B %Y").lstrip("0")
    for k in range(1, 12):
        v[f"n_step{k}"] = _n(funnel[k])
    for k in range(5, 12):
        r = funnel[k - 1] - funnel[k]
        v[f"rem_step{k}"] = _n(r)
        v[f"remp_step{k}"] = f"{100 * r / funnel[k - 1]:.1f}"
    for t in T4:
        n = ladder["runs"][f"{t}_chemistry_full"]["n_rows_header"]
        v[f"cov_{t}"] = f"{100 * n / grp['n_rows_total']:.1f}"

    # ---- cleaning constants (read from src/data_cleaning.py)
    c = pav._constants({"TEMP_MIN_K", "TEMP_MAX_K", "TEMP_BIN_WIDTH_K", "MAD_THRESHOLD", "MIN_TEMP_COVERAGE", "SMOOTHNESS_WINDOW",
                        "MULTI_SOURCE_CV_THRESHOLDS", "ZT_SELF_CONSISTENCY_MAX_REL_ERROR", "PROPERTY_BOUNDS"})
    v["t_min"], v["t_max"], v["bin_width"] = (str(c[k]) for k in ("TEMP_MIN_K", "TEMP_MAX_K", "TEMP_BIN_WIDTH_K"))
    v["b_sigma_exp"] = str(round(math.log10(c["PROPERTY_BOUNDS"]["sigma"][1])))
    v["t_second"] = str(c["TEMP_MIN_K"] + c["TEMP_BIN_WIDTH_K"])
    assert len({c["MULTI_SOURCE_CV_THRESHOLDS"][t] for t in ("S", "kappa", "zT")}) == 1  # prose: one CV threshold for S, kappa and zT
    v["bins"] = str((c["TEMP_MAX_K"] - c["TEMP_MIN_K"]) // c["TEMP_BIN_WIDTH_K"] + 1)
    v["mad_thr"] = f"{c['MAD_THRESHOLD']:g}"
    v["mad_sigma"] = f"{c['MAD_THRESHOLD'] / 1.4826:.1f}"
    v["zt_rel_err"] = f"{100 * c['ZT_SELF_CONSISTENCY_MAX_REL_ERROR']:.0f}"
    v["cv_S"], v["cv_sigma"], v["cv_kappa"], v["cv_zT"] = (f"{c['MULTI_SOURCE_CV_THRESHOLDS'][t]:g}" for t in T4)
    v["min_temps"] = str(c["MIN_TEMP_COVERAGE"])
    v["smooth_window"] = str(c["SMOOTHNESS_WINDOW"])
    b = c["PROPERTY_BOUNDS"]
    v["b_S"] = f"{b['S'][0]:g}".replace("-", "−")
    v["b_S_hi"] = f"{b['S'][1]:g}"
    v["b_sigma_lo"] = f"{b['sigma'][0]:g}"
    v["b_kappa_lo"] = f"{b['kappa'][0]:g}"
    v["b_kappa_hi"] = f"{b['kappa'][1]:g}"
    v["b_zT_hi"] = f"{b['zT'][1]:g}"

    # ---- validation ladder
    gaps = {}
    for t in T4:
        chem = ladder["runs"][f"{t}_chemistry_full"]
        rand = ung["per_run"][t]["random"]["pooled_r2"]
        v[f"chem_{t}"] = _r3(chem["per_repeat_r2_mean"])
        v[f"chem_sd_{t}"] = _r3(chem["per_repeat_r2_std"])
        v[f"foldsd_{t}"] = _r3(chem["fold_level_std"])
        v[f"rand_{t}"] = _r3(rand)
        v[f"k5_{t}"] = _r3(ung["per_run"][t]["kfold5"]["pooled_r2"])
        v[f"k10_{t}"] = _r3(ung["per_run"][t]["kfold10"]["pooled_r2"])
        v[f"comp_{t}"] = _r3(ung["composition_snapfix"][t]["mean"])
        gaps[t] = rand - chem["per_repeat_r2_mean"]
        v[f"gap_{t}"] = f"+{gaps[t]:.3f}"
        spread = [ung["per_run"][t][k]["pooled_r2"] for k in ("random", "kfold5", "kfold10")]
        v[f"spread_{t}"] = f"{max(spread) - min(spread):.3f}"
    lo, hi = min(gaps, key=gaps.get), max(gaps, key=gaps.get)
    v["gap_lo"], v["gap_lo_name"] = f"{gaps[lo]:.3f}", NAME[lo]
    v["gap_hi"], v["gap_hi_name"] = f"{gaps[hi]:.3f}", NAME[hi]
    v["spread_max"] = f"{max(float(v[f'spread_{t}']) for t in T4):.3f}"
    v["foldsd_min"] = f"{min(float(v[f'foldsd_{t}']) for t in T4):.3f}"
    v["foldsd_max"] = f"{max(float(v[f'foldsd_{t}']) for t in T4):.3f}"
    ab = abl
    for t in T4:
        for k in ("magpie", "cbfv"):
            v[f"abl_{t}_{k}"] = _r3(ab[t][k]["per_repeat_r2_mean"])
        v[f"abl_delta_{t}"] = f"{ab['deltas_full_minus_magpie'][t]:.3f}"
        assert abs(ab[t]["full"]["per_repeat_r2_mean"] - ladder["runs"][f"{t}_chemistry_full"]["per_repeat_r2_mean"]) < 1e-9
    dmax = max(T4, key=lambda t: ab["deltas_full_minus_magpie"][t])
    assert dmax == "sigma"  # prose: the largest descriptor-ablation gain is for sigma
    v["abl_delta_max"] = f"{ab['deltas_full_minus_magpie'][dmax]:.3f}"
    v["n_magpie_t"] = str(int(v["n_magpie"]) + 1)
    order = sorted(T4, key=lambda t: -float(v[f"chem_{t}"]))
    v["rank_order"] = ", ".join(NAME[t] for t in order)
    assert v["rank_order"] == "κ, S, zT, σ", v["rank_order"]  # the prose names this order
    assert (lo, hi) == ("kappa", "sigma"), (lo, hi)  # the prose names the smallest and largest gap
    v["repeats"] = str(ladder["runs"]["S_chemistry_full"]["n_repeats_cfg"])
    v["outer_folds"] = str(ladder["runs"]["S_chemistry_full"]["n_outer_folds_cfg"])
    v["rand_draws"] = str(ung["per_run"]["S"]["random"]["n_files"])
    v["n_pooled_chem_zT"] = _n(ladder["runs"]["zT_chemistry_full"]["pooled_n"])

    # ---- hyperparameters
    hp = {t: j(pav.HYPER.format(t)) for t in T4}
    assert len({(h["n_trials"], h["n_inner_folds"]) for h in hp.values()}) == 1
    v["optuna_trials"] = str(hp["S"]["n_trials"])
    v["inner_folds"] = str(hp["S"]["n_inner_folds"])
    v["search_text"] = pav.get("hyper.search")[0].split(": ", 1)[1]

    # ---- external validation
    e = j(pav.ESTM)
    a, bb = e["dedup_a_source_doi"], e["dedup_b_chemistry_cluster"]
    ins = {k: j(pav.INSUPPORT)[k] for k in ("a", "b")}
    v["estm_scope"] = _n(a["n_dropped"] + a["n_surviving"])
    v["estm_a_n"], v["estm_b_n"] = _n(a["n_surviving"]), _n(bb["n_surviving"])
    v["estm_a_drop"], v["estm_b_drop"] = _n(a["n_dropped"]), _n(bb["n_dropped"])
    v["estm_dois"] = _n(a["unique_estm_dois_overlapping_training"])
    v["estm_b_clusters"] = _n(bb["unique_estm_clusters_overlapping_training"])
    v["estm_a_insup_n"], v["estm_b_insup_n"] = _n(ins["a"]["n_in_support"]), _n(ins["b"]["n_in_support"])
    v["estm_a_ood"], v["estm_b_ood"] = f"{100 * ins['a']['ood_row_fraction']:.0f}", f"{100 * ins['b']['ood_row_fraction']:.0f}"
    for t in T4:
        key = "zT_direct" if t == "zT" else t
        internal = ins["a"]["properties"][t]["r2_internal_chemistry"]
        assert abs(internal - float(ladder["runs"][f"{t}_chemistry_full"]["per_repeat_r2_mean"])) < 1e-9
        for lab, d, k in (("a", a, "a"), ("b", bb, "b")):
            r2 = d["results"][key]["r2"]
            v[f"estm_{lab}_{t}"] = _r3(r2)
            v[f"estm_{lab}_insup_{t}"] = _r3(ins[k]["properties"][t]["r2_in_support"])
            v[f"estm_{lab}_drop_{t}"] = f"{r2 - internal:+.3f}".replace("-", "−")
    v["estm_a_zT_derived"], v["estm_b_zT_derived"] = _r3(a["results"]["zT_derived"]["r2"]), _r3(bb["results"]["zT_derived"]["r2"]).replace("-", "−")

    es = {t: bb["results"]["zT_direct" if t == "zT" else t]["r2"] for t in T4}
    assert max(es, key=es.get) == "kappa" and min(es, key=es.get) == "sigma"  # prose: kappa transfers best, sigma worst
    drops = {t: es[t] - float(ins["b"]["properties"][t]["r2_internal_chemistry"]) for t in T4}
    assert min(drops, key=drops.get) == "sigma"  # prose: sigma has the largest drop

    # ---- direct vs derived
    d, bt = j(pav.DVD), j(pav.DVD_BT)
    v["dvd_direct"], v["dvd_derived"] = _r3(d["zT_direct"]["pooled_r2"]), _r3(d["zT_derived"]["pooled_r2"])
    v["dvd_gap"] = _r3(d["zT_direct"]["pooled_r2"] - d["zT_derived"]["pooled_r2"])
    v["dvd_S"], v["dvd_sigma"], v["dvd_kappa"] = _r3(d["S"]["pooled_r2"]), _r3(d["sigma_log10"]["pooled_r2"]), _r3(d["kappa_log10"]["pooled_r2"])
    v["dvd_rows"], v["dvd_clusters"] = _n(d["subset_n_rows"]), _n(d["subset_n_chemistry_clusters"])
    v["dvd_duan"] = f"{bt['duan_effect_on_derived_r2']:+.3f}".replace("-", "−")
    assert abs(bt["duan_effect_on_derived_r2"]) < 0.01  # prose: the smearing correction is too small to matter
    weakest = min(("S", "sigma_log10", "kappa_log10"), key=lambda k: d[k]["pooled_r2"])
    assert weakest == "sigma_log10"

    # ---- SHAP: the zT top five and the shares come from NA3 (thesis_paper/results/na3_b, 25 folds); see na3_values()
    coarse = {g["group"]: g["chemistry_mean_share"] for g in j(pav.SHAP)["coarse_group_comparison"]}
    for k in ("magpie", "cbfv", "temperature"):
        v[f"shap_share_{k}"] = f"{100 * coarse[k]:.0f}"
    return v


if __name__ == "__main__":
    for k, val in values().items():
        print(k, "=", val)


# ---------------------------------------------------------------------------------------------------------------------------
# Results of the new analyses (thesis_paper/results/<analysis>/<UTC stamp>/results.json). The run folder is named explicitly, never
# globbed, and each file is pinned in SHARED_DEPENDENCIES.md.
# ---------------------------------------------------------------------------------------------------------------------------
NA_RUNS = {
    "na4": "thesis_paper/results/na4/20261003T131903",
    "na5": "thesis_paper/results/na5/20261003T131813",
    "na8": "thesis_paper/results/na8/20261003T131938",
    "na9": "thesis_paper/results/na9/20261003T132121",
}


# NA7 ran on Kaggle (T4 x2, two shards merged); its folder is the unpacked bundle: results.json, status.json, manifest.json (SHA256 of every
# unit file), run_configs/session_*.json and the 50 unit files. It has no run_config.json, so it is not in NA_RUNS.
NA7_RUN = "thesis_paper/results/na7/20261004T093502"
NA7_FILES = [f"{NA7_RUN}/results.json", f"{NA7_RUN}/status.json", f"{NA7_RUN}/manifest.json"]
NA10_ANALYSIS = "thesis_paper/results/na10_analysis/20261004T190444"
NA10_CLF = "thesis_paper/results/na10_analysis/20261005T054659"
NA6_METRICS = "thesis_paper/results/na6_metrics/20261004T135638"
NA6_SIGN = "thesis_paper/results/na6_sign_override/20261004T194811_same_folds"
NA11_FILTER = "thesis_paper/results/na11/20261004T135055"
NA11_SENS = "thesis_paper/results/na11_sensitivity/20261004T135205"
NA11_NOVELTY = "thesis_paper/results/na11_candidate_novelty/20261004T185101"
NA3_A = "thesis_paper/results/na3_a/20261004T201639"
NA3_B = "thesis_paper/results/na3_b/20261004T201433"
NA2_TUNING = "thesis_paper/results/na2_random_forest_tuning/summary"
NA11_LIMITS = "thesis_paper/results/na11_shortlist_limits/20261005T060020"
NA11_LIMITS_CSV = "thesis_paper/docs/shortlist_thermal_limits.csv"
NA11_RANKED = "thesis_paper/results/na11_ranked/20261005T053426"
GROUPED_DVD = "results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda/results.json"


def na_path(name):
    """Repository-relative path of an analysis' results.json."""
    return f"{NA_RUNS[name]}/results.json"


def na_json(name):
    """Parsed results.json of an analysis, with its run_config checked for the fixed provenance fields."""
    cfg = pav._json(f"{NA_RUNS[name]}/run_config.json")
    assert {"inputs", "git_head", "tree_clean", "script_sha256"} <= set(cfg), name
    return pav._json(na_path(name))


def na7_values():
    """NA7: direct versus derived zT under shuffled row-level 5 x 5 KFold, against the committed grouped run. The gap ratio is computed here."""
    st, r = pav._json(f"{NA7_RUN}/status.json"), pav._json(f"{NA7_RUN}/results.json")
    cfg = [pav._json(f"{NA7_RUN}/run_configs/{n}") for n in ("session_01.json", "session_01_restored1.json", "session_03.json")]
    assert st["complete"] and st["accepted_as_result"] and st["units_done"] == st["units_total"] == 25
    assert all(c["tree_clean"] and not c["allow_dirty"] and not c["smoke"] for c in cfg) and len({c["git_head"] for c in cfg}) == 1
    g = pav._json(GROUPED_DVD)
    assert r["subset_n_rows"] == g["subset_n_rows"], "NA7 and the grouped run must use the same subset"
    assert r["n_repeats"] == 5 and r["n_folds"] == 5
    gap_random = r["zT_direct"]["pooled_r2"] - r["zT_derived"]["pooled_r2"]
    gap_grouped = g["zT_direct"]["pooled_r2"] - g["zT_derived"]["pooled_r2"]
    assert abs(gap_random - r["gap_direct_minus_derived"]) < 1e-12 and abs(gap_grouped - r["grouped_run_for_comparison"]["gap"]) < 1e-12
    assert 0 < gap_random < gap_grouped
    return {"na7_direct": _r3(r["zT_direct"]["pooled_r2"]), "na7_derived": _r3(r["zT_derived"]["pooled_r2"]),
            "na7_mae_direct": f"{r['zT_direct']['mae']:.3f}", "na7_mae_derived": f"{r['zT_derived']['mae']:.3f}",
            "na7_rmse_direct": f"{r['zT_direct']['rmse']:.3f}", "na7_rmse_derived": f"{r['zT_derived']['rmse']:.3f}",
            "na7_gap": _r3(gap_random), "na7_grouped_gap": _r3(gap_grouped), "na7_ratio": f"{gap_grouped / gap_random:.1f}",
            "na7_sigma": _r3(r["sigma_log10"]["pooled_r2"]), "na7_S": _r3(r["S"]["pooled_r2"]), "na7_kappa": _r3(r["kappa_log10"]["pooled_r2"]),
            "na7_n": _n(r["zT_direct"]["n"])}


def na10_values():
    """NA10: the JARVIS comparison, from the committed analysis (counts per step, Spearman and sign agreement by stratum with cluster-bootstrap intervals)."""
    a = pav._json(f"{NA10_ANALYSIS}/analysis.json")
    cfg = pav._json(f"{NA10_ANALYSIS}/run_config.json")
    assert cfg["tree_clean"] and a["predicted_sign_source"] == "regressor"
    c, st = a["counts_per_step"], a["strata"]
    assert c["predicted"] == c["featurised"] == c["with_seebeck_n_and_p"]
    assert c["same_sign_in_metals"] + c["same_sign_in_semiconductors"] == c["same_sign_entries"]
    assert c["same_sign_ref_ge_20_in_metals"] + c["same_sign_ref_ge_20_in_semiconductors"] == c["same_sign_with_reference_ge_20_uV_per_K"]
    assert c["same_sign_entries"] + c["opposite_sign_entries"] + c["neither"] == c["predicted"]
    assert a["gap_metal_threshold_eV"] == 0.05 and a["min_reference_uV_per_K"] == 20.0
    v = {"jv_total": _n(c["jarvis_entries_total"]), "jv_both": _n(c["with_seebeck_n_and_p"]), "jv_same": _n(c["same_sign_entries"]), "jv_opp": _n(c["opposite_sign_entries"]),
         "jv_metal": _n(c["metal_or_semimetal_gap_lt_0.05"]), "jv_semi": _n(c["semiconductor_gap_ge_0.05"]),
         "jv_same_metal": _n(c["same_sign_in_metals"]), "jv_same_semi": _n(c["same_sign_in_semiconductors"]), "jv_opp_semi": _n(c["opposite_sign_in_semiconductors"]),
         "jv_ref_n": _n(c["same_sign_with_reference_ge_20_uV_per_K"]), "jv_ref_metal": _n(c["same_sign_ref_ge_20_in_metals"]), "jv_ref_semi": _n(c["same_sign_ref_ge_20_in_semiconductors"]),
         "jv_cluster_seen": _n(c["cluster_seen"]), "jv_n_clusters": _n(st["all"]["n_clusters"]),
         "jv_metal_share_ref": f"{100 * c['same_sign_ref_ge_20_in_metals'] / c['same_sign_with_reference_ge_20_uV_per_K']:.0f}"}

    def put(key, stratum, sign=False):
        s = st[stratum]
        if sign:
            s = s["sign_analysis_same_sign_entries_with_reference_ge_20"]
            val, ci, n = s["sign_agreement"], s["sign_agreement_ci95_cluster_bootstrap"], s["n"]
        else:
            val, ci, n = s["spearman_abs_pred_vs_mean_abs_jarvis"], s["spearman_ci95_cluster_bootstrap"], s["n"]
        v[f"{key}"], v[f"{key}_lo"], v[f"{key}_hi"], v[f"{key}_n"] = f"{val:.2f}", f"{ci[0]:.2f}", f"{ci[1]:.2f}", _n(n)

    for key, stratum in (("jv_rho_all", "all"), ("jv_rho_metal", "metal_or_semimetal"), ("jv_rho_semi", "semiconductor"), ("jv_rho_mseen", "metal_cluster_seen"),
                         ("jv_rho_munseen", "metal_cluster_unseen"), ("jv_rho_seen", "cluster_seen"), ("jv_rho_unseen", "cluster_unseen"), ("jv_rho_abx3", "abx3_stoichiometry")):
        put(key, stratum)
    for key, stratum in (("jv_sign_all", "all"), ("jv_sign_metal", "metal_or_semimetal"), ("jv_sign_semi", "semiconductor"), ("jv_sign_mseen", "metal_cluster_seen"),
                         ("jv_sign_munseen", "metal_cluster_unseen")):
        put(key, stratum, sign=True)
    # the same analysis with the classifier's sign (the screening configuration)
    ac = pav._json(f"{NA10_CLF}/analysis.json")
    assert pav._json(f"{NA10_CLF}/run_config.json")["tree_clean"] and ac["predicted_sign_source"] == "classifier" and ac["counts_per_step"] == a["counts_per_step"]
    for key, stratum in (("jvc_all", "all"), ("jvc_semi", "semiconductor"), ("jvc_mseen", "metal_cluster_seen"), ("jvc_munseen", "metal_cluster_unseen")):
        sc = ac["strata"][stratum]["sign_analysis_same_sign_entries_with_reference_ge_20"]
        v[key], v[f"{key}_lo"], v[f"{key}_hi"] = f"{sc['sign_agreement']:.2f}", f"{sc['sign_agreement_ci95_cluster_bootstrap'][0]:.2f}", f"{sc['sign_agreement_ci95_cluster_bootstrap'][1]:.2f}"
    assert ac["strata"]["all"]["spearman_abs_pred_vs_mean_abs_jarvis"] == st["all"]["spearman_abs_pred_vs_mean_abs_jarvis"]  # |S| is unchanged by the sign override
    s_all = st["all"]["sign_analysis_same_sign_entries_with_reference_ge_20"]
    v["jv_pred_pos"], v["jv_jarvis_pos"] = f"{100 * s_all['share_predicted_positive']:.0f}", f"{100 * s_all['share_jarvis_positive']:.0f}"
    assert "sign_analysis_same_sign_entries_with_reference_ge_20" not in st["semiconductor_cluster_seen"]
    return v


def na6_values():
    """NA6: the carrier-type classifier under chemistry-cluster CV (accuracy and the like with cluster-bootstrap intervals) and its sign accuracy against the S regressor's on the same folds."""
    m = pav._json(f"{NA6_METRICS}/metrics.json")
    sg = pav._json(f"{NA6_SIGN}/sign_comparison_same_folds.json")
    cfg = pav._json(f"{NA6_SIGN}/run_config.json")
    assert cfg["tree_clean"] and sg["decision"] == "classifier" and sg["n_rows"] == m["n_rows"]
    ac, bs = m["across_repeats"], m["cluster_bootstrap"]
    (tn, fp), (fn, tp) = m["confusion_matrix_all_repeats_rows_true_cols_pred_n_p"]
    assert tn + fp + fn + tp == m["n_repeats"] * m["n_rows"]
    v = {"cl_n": _n(m["n_rows"]), "cl_clusters": _n(m["n_clusters"]), "cl_pshare": f"{100 * m['share_p_type']:.1f}",
         "cl_tn": _n(tn), "cl_fp": _n(fp), "cl_fn": _n(fn), "cl_tp": _n(tp),
         "cl_single_n": _n(m["accuracy_by_cluster_size"]["singleton_cluster"]["n_rows"]), "cl_single_acc": f"{m['accuracy_by_cluster_size']['singleton_cluster']['accuracy']:.2f}",
         "cl_small_acc": f"{m['accuracy_by_cluster_size']['2_to_9_rows']['accuracy']:.2f}"}
    for key, k in (("acc", "accuracy"), ("bacc", "balanced_accuracy"), ("prec_p", "precision_p"), ("rec_p", "recall_p"), ("prec_n", "precision_n"), ("rec_n", "recall_n")):
        v[f"cl_{key}"], v[f"cl_{key}_lo"], v[f"cl_{key}_hi"] = _r3(ac[k]["mean"]), _r3(bs[k]["ci95"][0]), _r3(bs[k]["ci95"][1])
    auc = bs["roc_auc_repeat0"]
    v["cl_auc"], v["cl_auc_lo"], v["cl_auc_hi"] = _r3(ac["roc_auc"]["mean"]), _r3(auc["ci95"][0]), _r3(auc["ci95"][1])
    st = sg["strata"]
    d, small = st["all"], st["abs_S_lt_20"]
    v["cl_sign_clf"], v["cl_sign_reg"] = _r3(d["classifier_sign_accuracy"]["mean"]), _r3(d["regressor_sign_accuracy"]["mean"])
    dd = d["difference_classifier_minus_regressor"]
    v["cl_sign_diff"], v["cl_sign_diff_lo"], v["cl_sign_diff_hi"] = f"{dd['mean']:.3f}", f"{dd['ci95_cluster_bootstrap'][0]:.3f}", f"{dd['ci95_cluster_bootstrap'][1]:.3f}"
    v["cl_sign_small_diff"] = f"{small['difference_classifier_minus_regressor']['mean']:.3f}"
    v["cl_sign_small_n"] = _n(small["n_rows"])
    v["cl_sign_big_diff"] = f"{st['abs_S_ge_20']['difference_classifier_minus_regressor']['mean']:.3f}"
    return v


def na11_values():
    """NA11: counts after each screening filter, the sensitivity lists, and how many candidates lie in chemistry clusters of the training data."""
    c = pav._json(f"{NA11_FILTER}/counts.json")
    sens = pav._json(f"{NA11_SENS}/counts.json")
    nov = pav._json(f"{NA11_NOVELTY}/counts.json")
    for d in (f"{NA11_FILTER}/run_config.json", f"{NA11_SENS}/run_config.json", f"{NA11_NOVELTY}/run_config.json"):
        assert pav._json(d)["tree_clean"], d
    s3, s2 = c["sensitivity_S3_e_hull_0.10"], c["sensitivity_S2_final_cap_x1.5"]
    lists = nov["lists"]
    assert lists["main"]["n"] == c["f6_toxic_free_ranked_list"] == sens["main_list_size"] and lists["E_hull_0.10"]["n"] == s3["f6_toxic_free_ranked_list"]
    assert lists["gap_cap_0.9"]["n"] == s2["f6_toxic_free_ranked_list"] and lists["E_hull_0.10_cap_0.9"]["n"] == s3["S4_with_cap_0.9_f6"]
    assert nov["seen_compounds_in_main"] == ["BaSnO3", "CaMnO3", "CsSnI3", "CsSnI3", "LaCoO3", "LaNiO3", "LaRhO3", "YCoO3", "YCoO3"], "the compounds named in the text"
    main, e10, cap = lists["main"], nov["added_by_relaxation_relative_to_main"]["E_hull_0.10"], nov["added_by_relaxation_relative_to_main"]["gap_cap_0.9"]
    assert main["cluster_seen_any"] + main["cluster_unseen_any"] == main["n"] and e10["cluster_seen_any"] == 0
    return {"mp_f0": _n(c["f0_ehull_le_0.05"]), "mp_f1": _n(c["f1_gap_0.1_to_3.0"]), "mp_f2": _n(c["f2_lead_and_radioactive_free"]), "mp_f3": _n(c["f3_abx3"]),
            "mp_anti": _n(c["f3_anti_perovskite"]), "mp_f4": _n(c["f4_connectivity_perovskite_type"]), "mp_sg_rule": _n(c["f3_space_group_rule"]),
            "mp_sg_not_conn": _n(c["space_group_rule_but_not_connectivity"]), "mp_conn_not_sg": _n(c["f4_not_in_space_group_rule"]),
            "mp_f5": _n(c["f5_gap_le_0.6"]), "mp_f6": _n(c["f6_toxic_free_ranked_list"]),
            "mp_s3_f4": _n(s3["f4_connectivity_perovskite_type"]), "mp_s3_f6": _n(s3["f6_toxic_free_ranked_list"]), "mp_s2_f6": _n(s2["f6_toxic_free_ranked_list"]),
            "mp_s4_f6": _n(s3["S4_with_cap_0.9_f6"]), "mp_surv": _n(sens["n_surviving_every_e_hull_tightening"]), "mp_stable": _n(sens["main_list_stable_only_E_hull_0"]),
            "mp_main_in_e10": _n(sens["main_list_members_at_e_hull_0.10"]), "mp_tl": _n(len(sens["added_by_allowing_Tl_As_Be"])),
            "nv_main_seen": _n(main["cluster_seen_any"]), "nv_main_unseen": _n(main["cluster_unseen_any"]), "nv_main_clusters": _n(main["unique_clusters"]),
            "nv_e10_added": _n(e10["n"]), "nv_cap_added": _n(cap["n"]), "nv_cap_added_seen": _n(cap["cluster_seen_any"]),
            "nv_all": _n(lists["all_perovskite_type"]["n"]), "nv_all_seen": _n(lists["all_perovskite_type"]["cluster_seen_any"]), "nv_all_unseen": _n(lists["all_perovskite_type"]["cluster_unseen_any"])}


EXTRA_FILES = [  # thesis-paper results read by the value hooks of the later analyses, pinned in SHARED_DEPENDENCIES.md
    f"{NA6_METRICS}/metrics.json", f"{NA6_METRICS}/run_config.json", f"{NA6_SIGN}/sign_comparison_same_folds.json", f"{NA6_SIGN}/run_config.json",
    f"{NA10_ANALYSIS}/analysis.json", f"{NA10_ANALYSIS}/run_config.json", f"{NA10_CLF}/analysis.json", f"{NA10_CLF}/run_config.json",
    f"{NA11_FILTER}/counts.json", f"{NA11_FILTER}/run_config.json", f"{NA11_SENS}/counts.json", f"{NA11_SENS}/run_config.json",
    f"{NA11_NOVELTY}/counts.json", f"{NA11_NOVELTY}/run_config.json",
    f"{NA2_TUNING}/tuning_summary.json", f"{NA2_TUNING}/run_config.json",
    f"{NA11_LIMITS}/shortlist_limits.csv", f"{NA11_LIMITS}/summary.json", f"{NA11_LIMITS}/run_config.json", NA11_LIMITS_CSV,
    f"{NA11_RANKED}/summary.json", f"{NA11_RANKED}/ranked_main_30.csv", f"{NA11_RANKED}/shortlist.csv", f"{NA11_RANKED}/run_config.json",
    *[f"{r}/{f}" for r in (NA3_A, NA3_B) for f in ("results.json", "status.json", "run_configs/session_01.json")],
]


def na3_per_target():
    """NA3 (G3): SHAP attribution of the four models under chemistry-cluster folds, S and kappa from the first bundle, sigma and zT from the second."""
    out = {}
    for run in (NA3_A, NA3_B):
        st, r, cfg = pav._json(f"{run}/status.json"), pav._json(f"{run}/results.json"), pav._json(f"{run}/run_configs/session_01.json")
        assert st["complete"] and st["accepted_as_result"] and st["units_done"] == st["units_total"] == 50
        assert cfg["tree_clean"] and not cfg["allow_dirty"] and not cfg["smoke"] and cfg["params"]["arm"] == "chemistry" and cfg["params"]["shap_rows"] == 20000
        assert cfg["params"]["n_repeats"] == 5 and cfg["params"]["n_folds"] == 5
        for t, d in r["per_target"].items():
            assert d["n_folds"] == 25 and len(d["top20"]) == 20
            out[t] = {**d, "feature_columns": r["feature_columns"], "shap_rows": cfg["params"]["shap_rows"]}
    assert set(out) == set(T4), sorted(out)
    return out


def shap_display_name(raw):
    """A feature's column name as shown in the paper (the column name itself, in code font)."""
    return raw


def na3_values():
    """NA3: top features and the MAGPIE / CBFV / temperature shares per target (all from the Kaggle G3 bundles)."""
    na3 = na3_per_target()
    v = {"shap_rows": _n(na3["zT"]["shap_rows"]), "shap_folds": _n(na3["zT"]["n_folds"])}
    cols = na3["zT"]["feature_columns"]
    n_mag, n_cbfv = sum(c.startswith("MagpieData") for c in cols), sum(c.startswith("CBFV_") for c in cols)
    assert n_mag + n_cbfv + 1 == len(cols) == 397
    v["shap_n_magpie"], v["shap_n_cbfv"] = _n(n_mag), _n(n_cbfv)
    ratios = {}
    for i in range(1, 6):
        v[f"t9_rank_{i}"] = str(i)
    for t in T4:
        d = na3[t]
        for i, f in enumerate(d["top20"][:5], 1):
            v[f"t9_{t}_{i}"] = f"`{shap_display_name(f['feature'])}` ({f['mean_abs_shap']:.1f})" if t == "S" else f"`{shap_display_name(f['feature'])}` ({f['mean_abs_shap']:.3f})"
        v[f"shap_{t}_1"], v[f"shap_{t}_2"], v[f"shap_{t}_3"] = (f"`{shap_display_name(f['feature'])}`" for f in d["top20"][:3])
        v[f"shap_{t}_1_val"], v[f"shap_{t}_2_val"], v[f"shap_{t}_3_val"] = (f"{f['mean_abs_shap']:.1f}" if t == "S" else f"{f['mean_abs_shap']:.3f}" for f in d["top20"][:3])
        names = [f["feature"] for f in d["top20"]]
        v[f"shap_{t}_temp_rank"] = str(names.index("temperature_bin") + 1)
        sh = d["share_by_group"]
        for g in ("magpie", "cbfv", "temperature"):
            v[f"shap_share_{t}_{g}"] = f"{100 * sh[g]['mean']:.0f}"
        assert abs(sum(sh[g]["mean"] for g in sh) - 1) < 1e-4
        assert max(sh, key=lambda g: sh[g]["mean"]) == "cbfv"  # prose: CBFV features carry the largest share in every model
        ratios[t] = (sh["cbfv"]["mean"] / n_cbfv) / (sh["magpie"]["mean"] / n_mag)
        v[f"shap_ratio_{t}"] = f"{ratios[t]:.2f}"
    assert max(T4, key=lambda t: float(v[f"shap_share_{t}_temperature"])) == "zT" and min(T4, key=lambda t: float(v[f"shap_share_{t}_temperature"])) == "S"  # prose
    v["shap_cbfv_min"], v["shap_cbfv_max"] = (f"{min(float(v[f'shap_share_{t}_cbfv']) for t in T4):.0f}", f"{max(float(v[f'shap_share_{t}_cbfv']) for t in T4):.0f}")
    v["shap_mag_min"], v["shap_mag_max"] = (f"{min(float(v[f'shap_share_{t}_magpie']) for t in T4):.0f}", f"{max(float(v[f'shap_share_{t}_magpie']) for t in T4):.0f}")
    v["shap_ratio_min"], v["shap_ratio_max"] = f"{min(ratios.values()):.2f}", f"{max(ratios.values()):.2f}"
    # the zT result must equal the earlier committed zT run (same frozen hyperparameters, same folds): shares to the integer the paper printed, and the prose about temperature first and NdValence third
    old = {g["group"]: g["chemistry_mean_share"] for g in pav._json(pav.SHAP)["coarse_group_comparison"]}
    for g in ("magpie", "cbfv", "temperature"):
        assert f"{100 * old[g]:.0f}" == v[f"shap_share_zT_{g}"], g
    for tt in ("S", "sigma", "kappa"):  # prose of the conclusion: the largest mean |SHAP| of S, sigma and kappa belongs to a CBFV statistic; temperature is second for kappa
        assert na3[tt]["top20"][0]["feature"].startswith("CBFV_"), tt
    assert v["shap_kappa_temp_rank"] == "2"
    assert na3["zT"]["top20"][0]["feature"] == "temperature_bin" and na3["zT"]["top20"][2]["feature"].endswith("NdValence")
    return v


def _formula_md(f):
    """A formula with pandoc subscripts: BaZrS3 -> BaZrS~3~."""
    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"~\1~", f)


def _sigma_md(x):
    """A conductivity as 'm x 10^k^'."""
    e = int(math.floor(math.log10(x)))
    return f"{x / 10 ** e:.1f} × 10^{e}^"


def na11_ranked_values():
    """NA11: the ranked list (Table 11), the shortlist of the pre-registered rule A and the sign-override fractions, from the committed ranked-list run."""
    import pandas as pd

    summ = pav._json(f"{NA11_RANKED}/summary.json")
    cfg = pav._json(f"{NA11_RANKED}/run_config.json")
    assert cfg["tree_clean"]
    d = pd.read_csv(REPO_ROOT / NA11_RANKED / "ranked_main_30.csv")
    sl = pd.read_csv(REPO_ROOT / NA11_RANKED / "shortlist.csv")
    assert len(d) == 30 and set(sl["material_id"]) == {"mp-3163", "mp-614013", "mp-1288145", "mp-5163"}
    d = d.sort_values(["cluster_seen_any", "rank_in_group_main_30"], ascending=[False, True]).reset_index(drop=True)
    assert d["cluster_seen_any"].iloc[:9].all() and not d["cluster_seen_any"].iloc[9:].any()
    v = {}
    for i, r in d.iterrows():
        k = i + 1
        sgn = "−" if r["S_at_max"] < 0 else ""
        v[f"t11_{k}_group"] = "seen" if r["cluster_seen_any"] else "unseen"
        v[f"t11_{k}_rank"] = str(int(r["rank_in_group_main_30"]))
        v[f"t11_{k}_name"] = _formula_md(r["formula"]) + (" †" if r["shortlist"] else "") + (" ‡" if (r["formula"] == "CsSnI3" and r["T_at_zT_max"] > 724.15) else "") + f" ({r['material_id']})"
        v[f"t11_{k}_ehull"] = f"{r['ehull']:.3f}"
        v[f"t11_{k}_stab"] = "on hull" if r["stability"] == "on the hull" else "metastable"
        v[f"t11_{k}_gap"] = f"{r['gap']:.2f}"
        v[f"t11_{k}_T"] = str(int(r["T_at_zT_max"]))
        v[f"t11_{k}_S"] = f"{sgn}{abs(r['S_at_max']):.0f}"
        v[f"t11_{k}_sigma"] = _sigma_md(r["sigma_at_max"])
        v[f"t11_{k}_kappa"] = f"{r['kappa_at_max']:.2f}"
        v[f"t11_{k}_zT"] = f"{r['zT_max']:.2f}"
        v[f"t11_{k}_lit"] = "yes" if r["previously_studied_as_thermoelectric"] == "yes" else "no evidence"
    ov, ovm, ovs = summ["override_all_409"], summ["override_main_30"], summ["override_shortlist"]
    assert ovm["entries"] == 30 and ovm["predictions"] == 180
    v.update({"ov_main_k": _n(ovm["predictions_overridden"]), "ov_main_n": _n(ovm["predictions"]), "ov_main_share": f"{100 * ovm['share_predictions_overridden']:.0f}",
              "ov_main_entries": _n(ovm["entries_with_any_override"]), "ov_all_k": _n(ov["predictions_overridden"]), "ov_all_n": _n(ov["predictions"]),
              "ov_all_share": f"{100 * ov['share_predictions_overridden']:.0f}", "ov_all_entries": _n(ov["entries_with_any_override"]), "ov_all_cands": _n(ov["entries"]),
              "ov_sl_k": _n(ovs["predictions_overridden"])})
    sl = sl.sort_values("shortlist_order")
    for i, r in enumerate(sl.itertuples(), 1):
        v[f"sl_{i}_name"] = _formula_md(r.formula)
        v[f"sl_{i}_zT"] = f"{r.zT_max:.2f}"
        v[f"sl_{i}_T"] = str(int(r.T_at_zT_max))
        v[f"sl_{i}_ehull"] = f"{r.ehull:.3f}"
    assert list(sl["formula"]) == summ["shortlist_order"]
    v["sl_n"] = _n(len(sl))
    assert all(v[f"sl_{i}_T"] == v["sl_1_T"] for i in range(1, 5)) and set(summ["main_30"]["T_at_max_counts"]) == {"800"}
    v["mp_tmax_main"] = str(max(int(k) for k in summ["main_30"]["T_at_max_counts"]))
    v["mp_tmax_all_800"] = _n(summ["all_409"]["T_at_max_counts"]["800"])
    v["mp_tmax_all"] = _n(sum(summ["all_409"]["T_at_max_counts"].values()))
    v["mp_main_on_hull"] = _n(summ["main_30"]["on_the_hull"])
    v["mp_main_ptype"] = _n(summ["main_30"]["p_type_at_max"])
    v["mp_zt_all_max"] = f"{summ['all_409']['zT_max_range'][1]:.2f}"
    v["mp_zt_main_max"] = f"{summ['main_30']['zT_max_range'][1]:.2f}"
    v["mp_zt_main_min"] = f"{summ['main_30']['zT_max_range'][0]:.2f}"
    lit = d["previously_studied_as_thermoelectric"] == "yes"
    v["lit_main_yes"], v["lit_main_no"] = _n(lit.sum()), _n((~lit).sum())
    v["lit_main_seen_yes"] = _n((lit & d["cluster_seen_any"]).sum())
    v["lit_main_unseen_yes"] = _n((lit & ~d["cluster_seen_any"]).sum())
    return v


def na2_design_values():
    """NA2: the random-forest design change: the first (tuned, stopped) session's S tuning record against the frozen XGBoost model, from the committed tuning summary."""
    t = pav._json(f"{NA2_TUNING}/tuning_summary.json")
    assert pav._json(f"{NA2_TUNING}/run_config.json")["tree_clean"]
    assert t["n_trials"] == t["n_complete"] + t["n_pruned"] and t["xgboost_S_n_trials"] == 20
    return {"rf_trials_done": _n(t["n_trials"]), "rf_complete": _n(t["n_complete"]), "rf_pruned": _n(t["n_pruned"]), "rf_hours": f"{t['trial_hours_total']:.1f}",
            "rf_min": _r3(t["inner_cv_r2_all_complete"]["min"]), "rf_max": _r3(t["inner_cv_r2_all_complete"]["max"]),
            "rf_deep_min": _r3(t["inner_cv_r2_deep"]["min"]), "rf_deep_max": _r3(t["inner_cv_r2_deep"]["max"]), "rf_deep_range": _r3(t["inner_cv_r2_deep"]["range"]),
            "rf_n_deep": _n(t["n_deep"]), "rf_shallow_depth": _n(t["shallowest_trial"]["max_depth"]), "rf_shallow": _r3(t["shallowest_trial"]["inner_cv_r2"]),
            "xgb_inner": _r3(t["xgboost_S_frozen_inner_cv_r2"])}


def _safe_eval(node):
    """Value of a constant expression of numbers, strings, None, booleans, tuples, lists and dicts, allowing + - * / between numbers (e.g. 1 / 3)."""
    import ast
    import operator

    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
    if isinstance(node, ast.BinOp) and type(node.op) in ops:
        return ops[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        x = _safe_eval(node.operand)
        return -x if isinstance(node.op, ast.USub) else x
    if isinstance(node, ast.Dict):
        return {_safe_eval(k): _safe_eval(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.Tuple, ast.List)):
        return type(node).__name__ == "Tuple" and tuple(map(_safe_eval, node.elts)) or list(map(_safe_eval, node.elts))
    return ast.literal_eval(node)


def na11_limits_values():
    """NA11: the thermal limits of the shortlist (sourced) and the secondary 600 K view."""
    import pandas as pd

    assert pav._json(f"{NA11_LIMITS}/run_config.json")["tree_clean"]
    d = pd.read_csv(REPO_ROOT / NA11_LIMITS / "shortlist_limits.csv")
    sm = pav._json(f"{NA11_LIMITS}/summary.json")
    assert sm["order_primary"] == sm["order_within_limit"] == sm["order_at_secondary_T"], "the prose says the three orders agree"
    d = d.sort_values("rank_primary").reset_index(drop=True)
    v = {"lim_secondary_T": str(sm["secondary_T_K"])}
    for i, r in d.iterrows():
        k = i + 1
        v[f"lim_{k}_name"] = _formula_md(r["formula"])
        ul, cl = r["unconditional_limit_K"], r["conditional_limit_K"]
        if pd.notna(ul) and ul <= 800:
            v[f"lim_{k}_text"] = f"melts at {ul:.0f} K"
        elif pd.notna(ul):
            v[f"lim_{k}_text"] = f"decomposes at {ul:.0f} K in air (none at or below 800 K)"
        elif pd.notna(cl):
            v[f"lim_{k}_text"] = f"none in air; reduced above {cl:.0f} K in a reducing atmosphere"
        else:
            v[f"lim_{k}_text"] = "none found at or below 800 K"
        v[f"lim_{k}_range"] = f"{int(r['grid_max_reported_K'])}"
        v[f"lim_{k}_zt"] = f"{r['zT_max_within_limit']:.2f}"
        v[f"lim_{k}_zt_T"] = str(int(r["T_at_zT_max_within_limit"]))
        v[f"lim_{k}_zt600"] = f"{r['zT_at_secondary_T']:.2f}"
        v[f"lim_{k}_rank1"], v[f"lim_{k}_rank2"] = str(int(r["rank_primary"])), str(int(r["rank_at_secondary_T"]))
    c = d[d["formula"] == "CsSnI3"].iloc[0]
    v["cs_limit_K"] = f"{c['unconditional_limit_K']:.0f}"
    v["cs_grid_max"] = str(int(c["grid_max_reported_K"]))
    v["cs_zt_limit"], v["cs_zt_primary"] = f"{c['zT_max_within_limit']:.2f}", f"{c['zT_max_primary_all_grid']:.2f}"
    v["cs_primary_T"] = str(int(c["T_at_zT_max_primary"]))
    v["lco_cond_K"] = f"{d[d['formula'] == 'LaCoO3'].iloc[0]['conditional_limit_K']:.0f}"
    v["lim_order"] = ", ".join(_formula_md(f) for f in sm["order_at_secondary_T"])
    assert not bool(c["primary_value_inside_limit"]) and all(bool(x) for x in d[d["formula"] != "CsSnI3"]["primary_value_inside_limit"])
    return v


def config_value(pointer):
    """Value of a module-level constant of a committed file, for DESIGN markers: 'path:NAME', 'path:NAME[key]' (dict key or tuple index), optionally
    followed by '*k' or '/k'. The value is formatted with :g, so 0.5 * 100 reads 50."""
    import ast

    m = re.fullmatch(r"([^:]+):(\w+)(?:\[([^\]]+)\])?(?:([*/])([\d.]+))?", pointer)
    assert m, f"bad config pointer {pointer!r}"
    path, name, sub, op, k = m.groups()
    tree = ast.parse((Path(__file__).resolve().parents[2] / path).read_text(encoding="utf-8"))
    val = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            val = _safe_eval(node.value)
    assert val is not None, f"{name} not found in {path}"
    if sub is not None:
        val = val[ast.literal_eval(sub)]
    if op:
        val = val * float(k) if op == "*" else val / float(k)
    return f"{val:g}"


def table_values(base):
    """Cell values of Tables 2, 4 and 6 that the block functions used to read straight from the artifacts. They are keys of the value dictionary so
    that every number in a generated block is traceable (audit_numbers.py); the formatting is unchanged."""
    minus = "−"
    v = {}

    def f(x, dp):
        return f"{x:,.{dp}f}".replace("-", minus)

    n4 = na_json("na4")["per_target"]
    dps = {"S": 1, "sigma": 0, "kappa": 2, "zT": 3}
    for t in T4:
        d, dp = n4[t], dps[t]
        v[f"t2_{t}_n"], v[f"t2_{t}_cov"] = f"{d['n']:,}", f"{100 * d['coverage']:.1f}%"
        v[f"t2_{t}_mean"], v[f"t2_{t}_median"], v[f"t2_{t}_sd"] = f(d["mean"], dp), f(d["median"], dp), f(d["std"], dp)
        if t == "sigma":
            e = int(math.floor(math.log10(d["max"])))
            assert e == 6, "the table caption units assume a maximum of order 10^6"
            v[f"t2_{t}_min"], v[f"t2_{t}_max"] = f"{d['min']:,.0f}", f"{d['max'] / 10 ** e:.2f} × 10^{e}^"
        elif t == "S":
            v[f"t2_{t}_min"], v[f"t2_{t}_max"] = f(d["min"], 0), f(d["max"], 1)
        else:
            v[f"t2_{t}_min"], v[f"t2_{t}_max"] = f(d["min"], dp), f(d["max"], dp)
    hp_files = {t: pav._json(pav.HYPER.format(t))["best_params"] for t in T4}
    rng = dict(re.findall(r"(\w+) ([^;]+)", base["search_text"].replace("; ", ";").replace(";", "; ")))
    for k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha"):
        v[f"t4_range_{k}"] = rng[k].replace("..", "–").strip().rstrip(";")
        for t in T4:
            x = hp_files[t][k]
            v[f"t4_{t}_{k}"] = f"{x:.4g}" if isinstance(x, float) else str(x)
    lad = pav._json(pav.LADDER)["runs"]
    for t in T4:
        v[f"comp_sd_{t}"] = f"{lad[f'{t}_composition_full']['per_repeat_r2_std']:.3f}"
    return v


def new_analysis_values():
    """Formatted values from the NA4, NA5, NA8 and NA9 results."""
    v = {}
    base = values()
    n4, n5, n8, n9 = (na_json(k) for k in ("na4", "na5", "na8", "na9"))
    # NA5: cleaning diagnostics; the rerun was accepted only because it reproduced the committed funnel (asserted in the script, and here)
    funnel = {int(s["step"].split("_")[0]): s["rows"] for s in pav._json(pav.FUNNEL)["funnel"]}
    assert {int(k): x for k, x in n5["rows_after_step"].items()} == funnel
    v["na5_points_all"] = _n(n5["points_all_curves"])
    v["na5_points_target"] = _n(n5["points_target_curves_before_range_filter"])
    v["na5_zt6_before"] = f"{n5['zT_before_step6']['mean']:.3f}"
    v["na5_zt6_after"] = f"{n5['zT_after_step6']['mean']:.3f}"
    v["na5_zt_final"] = f"{n5['zT_final']['mean']:.3f}"
    v["na5_spikes"] = _n(n5["step11_spikes_set_to_nan_total"])
    v["na5_pre2005"] = _n(n5["extra_filters_on_final_rows"]["published_before_2005"])
    v["na5_lt2props"] = _n(n5["extra_filters_on_final_rows"]["fewer_than_2_properties"])
    # NA4: dataset statistics
    pt = n4["per_target"]
    for t in T4:
        assert pt[t]["n"] == int(base[f"n_{t}"].replace(",", "")), t
    v["na4_zt3"] = _n(n4["zT_above_3"]["n"])
    v["na4_zt3_pct"] = f"{100 * n4['zT_above_3']['share']:.3f}"
    v["na4_zt_mean"] = f"{pt['zT']['mean']:.3f}"
    v["na4_S_p_share"] = f"{100 * n4['S_p_type']['share_positive']:.1f}"
    v["na4_S_mean"], v["na4_S_median"] = f"{pt['S']['mean']:.1f}", f"{pt['S']['median']:.1f}"
    s = n4["rows_per_temperature_bin"]["zT_summary"]
    v["na4_sparse_bin"], v["na4_sparse_n"] = s["sparsest_bin"], _n(s["sparsest_n"])
    v["na4_dense_bin"], v["na4_dense_n"] = s["densest_bin"], _n(s["densest_n"])
    v["na4_max_bin"] = max(n4["rows_per_temperature_bin"]["zT"], key=int)
    fr = n4["family_rows"]
    v["na4_pt_share"] = f"{100 * fr['perovskite_titanate']['S']['share']:.1f}"
    v["na4_mn_share"] = f"{100 * fr['manganite']['S']['share']:.1f}"
    v["na4_co_share"] = f"{100 * fr['cobaltite']['S']['share']:.1f}"
    # NA8: error metrics (S in microV/K; sigma and kappa in log10 units)
    for t in T4:
        d = n8["ladder"][t]
        assert abs(d["r2_mean"] - float(pav._json(pav.LADDER)["runs"][f"{t}_chemistry_full"]["per_repeat_r2_mean"])) < 1e-9
        dp = 1 if t == "S" else 3
        v[f"mae_{t}"], v[f"rmse_{t}"] = f"{d['mae_mean']:.{dp}f}", f"{d['rmse_mean']:.{dp}f}"
        v[f"na8_slope_{t}"] = f"{d['repeat0_calibration']['slope_pred_on_true']:.2f}"
    v["mae_dvd_direct"], v["mae_dvd_derived"] = f"{n8['dvd']['direct']['mae']:.3f}", f"{n8['dvd']['derived']['mae']:.3f}"
    c = n8["ladder"]["zT"]["repeat0_calibration"]
    v["na8_zt_top_true"], v["na8_zt_top_pred"] = f"{c['top5pct_true_mean']:.2f}", f"{c['top5pct_pred_mean']:.2f}"
    for t in T4:
        assert float(v[f"na8_slope_{t}"]) < 1.0  # prose: predictions are compressed towards the mean
    # NA9: ESTM counts
    assert n9["estm_rows_in_scope"] == int(base["estm_scope"].replace(",", ""))
    v["na9_formulas"], v["na9_seen"], v["na9_unseen"] = _n(n9["estm_unique_formulas"]), _n(n9["formulas_seen_in_training"]), _n(n9["formulas_unseen_in_training"])
    v["na9_clusters"], v["na9_clusters_seen"] = _n(n9["estm_unique_clusters"]), _n(n9["clusters_seen_in_training"])
    assert n9["clusters_seen_in_training"] == int(base["estm_b_clusters"].replace(",", ""))
    v.update(table_values(base))
    v7 = na7_values()
    g = pav._json(GROUPED_DVD)  # the grouped run behind Table 10: its R2 values must be the ones printed there (the gap is taken from unrounded values)
    assert _r3(g["zT_direct"]["pooled_r2"]) == base["dvd_direct"] and _r3(g["zT_derived"]["pooled_r2"]) == base["dvd_derived"]
    v.update(v7)
    v.update(na10_values())
    v.update(na6_values())
    v.update(na11_values())
    v.update(na3_values())
    v.update(na11_ranked_values())
    v.update(na11_limits_values())
    v.update(na2_design_values())
    return v
