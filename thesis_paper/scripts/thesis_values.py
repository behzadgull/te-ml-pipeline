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

    # ---- SHAP (zT; the five folds kept in the committed npz)
    top = pav.get("shap.zT.top5")[0].split(" (zT,")[0].split("; ")
    names = [re.sub(r"\s*\(([0-9.]+)\)$", "", x) for x in top]
    vals = [re.search(r"\(([0-9.]+)\)$", x).group(1) for x in top]
    for i, (nm, val) in enumerate(zip(names, vals), 1):
        v[f"shap_zT_{i}"], v[f"shap_zT_{i}_val"] = nm, val
    assert names[0] == "temperature_bin" and names[2].endswith("NdValence")  # prose: temperature first, NdValence third
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


def na_path(name):
    """Repository-relative path of an analysis' results.json."""
    return f"{NA_RUNS[name]}/results.json"


def na_json(name):
    """Parsed results.json of an analysis, with its run_config checked for the fixed provenance fields."""
    cfg = pav._json(f"{NA_RUNS[name]}/run_config.json")
    assert {"inputs", "git_head", "tree_clean", "script_sha256"} <= set(cfg), name
    return pav._json(na_path(name))


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
    return v
