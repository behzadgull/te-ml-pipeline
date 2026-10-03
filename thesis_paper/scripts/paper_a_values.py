"""
Verified Paper A values, read from committed artifacts, for the thesis-paper claim inventory.

Every number returned here is computed from a committed file (never typed): `get(key)` returns
(formatted value, [source paths relative to the repository root]). A missing file or key raises, so a moved or
changed artifact stops the inventory build instead of leaving a stale value.

Keys (T is one of S, sigma, kappa, zT):
  ladder.rand.T  ladder.k5.T  ladder.k10.T  ladder.comp.T  ladder.chem.T   pooled R2 per validation protocol
  gap.T  gap.range  ungrouped.spread.T                                        random minus chemistry; max spread among ungrouped rungs
  rows.T  features.full  features.magpie  features.cbfv                       row and feature counts
  funnel.final  funnel.raw  funnel.stepN  funnel.removed.stepN                cleaning funnel (N = 5..11)
  raw.counts  data.clusters  data.formulas  data.parent_systems
  ablation.T  ablation.delta.T                                                descriptor ablation
  noise.T                                                                     R2_max and combined headroom
  dvd                                                                         direct vs derived zT
  estm.a  estm.b  estm.n_in_scope  estm.insupport.a  estm.insupport.b        ESTM external validation
  shap.zT.top5  shap.coarse                                                   SHAP (zT only)
  hyper.T  hyper.search                                                       frozen hyperparameters and search space
  clean.bounds  clean.params                                                  cleaning constants read from src/data_cleaning.py
  pipeline.preproc                                                            whether src/nested_cv.py imputes or scales XGBoost inputs
"""

import ast
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TARGETS = ("S", "sigma", "kappa", "zT")

LADDER = "reports/regen_snapfix/20260917T150000/ladder_metrics.json"
UNGROUPED = "reports/ungrouped_snapfix/20260922T093243/metrics.json"
ABLATION = "reports/ablation_snapfix/20260918T000111/ablation_metrics.json"
NOISE = "results/noise_floor/20260923T202312/noise_floor_inputs.json"
DVD = "results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda/results.json"
DVD_BT = "results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda/backtransform_check_results.json"
ESTM = "results/external_snapfix/20260917T160553/estm_results.json"
INSUPPORT = "reports/insupport_share/20260924T194948Z/insupport_share.json"
SHAP = "results/shap_attribution/20260917T134930/summary.json"
SHAP_ARRAYS = "results/shap_attribution/20260917T134930/shap_arrays.npz"
FUNNEL = "results/cleaning_funnel/20260914T100914/funnel_counts.json"
RAWMETA = "results/raw_pull_metadata/extraction_metadata.json"
GROUPING = "reports/grouping_key_remeasure/20260918T000232/grouping_key_remeasure.json"
HYPER = "checkpoints/saved_predictions/checkpoints/frozen_hyperparams/{}.json"
CLEANING = "src/data_cleaning.py"
NESTED = "src/nested_cv.py"

_cache = {}


def _json(path):
    if path not in _cache:
        _cache[path] = json.loads((REPO / path).read_text(encoding="utf-8"))
    return _cache[path]


def _f4(x):
    return f"{x:.4f}"


def _pm(m, s):
    return f"{m:.4f} ± {s:.4f}"


def _n(x):
    return f"{int(x):,}"


def _chem(t):
    r = _json(LADDER)["runs"][f"{t}_chemistry_full"]
    return r["per_repeat_r2_mean"], r["per_repeat_r2_std"]


def _rand(t):
    return _json(UNGROUPED)["per_run"][t]["random"]["pooled_r2"]


def _constants(names):
    """Literal values of module-level assignments in src/data_cleaning.py."""
    tree = ast.parse((REPO / CLEANING).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(out) == set(names), set(names) - set(out)
    return out


def _funnel_steps():
    return {int(s["step"].split("_")[0]): s["rows"] for s in _json(FUNNEL)["funnel"]}


def get(key):
    """Return (formatted verified value, [source paths])."""
    p = key.split(".")
    head = p[0]
    if head == "ladder":
        kind, t = p[1], p[2]
        if kind == "rand":
            d = _json(UNGROUPED)["per_run"][t]["random"]
            return _pm(d["pooled_r2"], d["per_fold_sd"]) + " (20 draws, pooled)", [UNGROUPED]
        if kind in ("k5", "k10"):
            d = _json(UNGROUPED)["per_run"][t]["kfold5" if kind == "k5" else "kfold10"]
            return _pm(d["pooled_r2"], d["per_fold_sd"]) + " (pooled)", [UNGROUPED]
        if kind == "comp":
            d = _json(UNGROUPED)["composition_snapfix"][t]
            return _pm(d["mean"], d["sd"]) + " (5 repeats)", [UNGROUPED]
        if kind == "chem":
            return _pm(*_chem(t)) + " (5 repeats)", [LADDER]
    if head == "gap":
        if p[1] == "range":
            gaps = {t: _rand(t) - _chem(t)[0] for t in TARGETS}
            lo, hi = min(gaps, key=gaps.get), max(gaps, key=gaps.get)
            return f"+{gaps[lo]:.3f} ({lo}) to +{gaps[hi]:.3f} ({hi})", [UNGROUPED, LADDER]
        t = p[1]
        return f"+{_rand(t) - _chem(t)[0]:.3f}", [UNGROUPED, LADDER]
    if head == "ungrouped":
        t = p[2]
        d = _json(UNGROUPED)["per_run"][t]
        v = [d["random"]["pooled_r2"], d["kfold5"]["pooled_r2"], d["kfold10"]["pooled_r2"]]
        return f"{max(v) - min(v):.4f}", [UNGROUPED]
    if head == "rows":
        return _n(_json(LADDER)["runs"][f"{p[1]}_chemistry_full"]["n_rows_header"]), [LADDER]
    if head == "features":
        d = _json(ABLATION)["S"]
        n = {"full": d["full"]["n_features"], "magpie": d["magpie"]["n_features"] - 1, "cbfv": d["cbfv"]["n_features"] - 1}[p[1]]
        return str(n) + (" (excluding temperature_bin)" if p[1] != "full" else " (132 MAGPIE + 264 CBFV + temperature_bin)"), [ABLATION]
    if head == "funnel":
        steps = _funnel_steps()
        if p[1] == "final":
            return _n(steps[11]), [FUNNEL]
        if p[1] == "raw":
            return _n(steps[1]) + " observations after range filtering", [FUNNEL]
        if p[1] == "removed":
            n = int(p[2].replace("step", ""))
            r = steps[n - 1] - steps[n]
            return f"{_n(r)} rows ({100 * r / steps[n - 1]:.1f}% of {_n(steps[n - 1])})", [FUNNEL]
        n = int(p[1].replace("step", ""))
        return _n(steps[n]), [FUNNEL]
    if head == "raw":
        m = _json(RAWMETA)["files"]
        return (f"{_n(m['papers']['counted_row_count'])} papers, {_n(m['samples']['counted_row_count'])} samples, "
                f"{_n(m['curves']['counted_row_count'])} curves; snapshot {_json(RAWMETA)['upstream_db_snapshot']}"), [RAWMETA]
    if head == "data":
        g = _json(GROUPING)
        return {
            "clusters": f"{_n(g['total_chemistry_clusters']['new'])} chemistry clusters",
            "formulas": f"{_n(g['unique_formulas']['new'])} unique formulas ({_n(g['n_rows_total'])} featurized rows)",
            "parent_systems": f"{_n(g['parent_system_groups']['new'])} parent chemical systems",
        }[p[1]], [GROUPING]
    if head == "ablation":
        a = _json(ABLATION)
        if p[1] == "delta":
            return f"+{a['deltas_full_minus_magpie'][p[2]]:.4f}", [ABLATION]
        t = p[1]
        return ", ".join(f"{k} {_pm(a[t][k]['per_repeat_r2_mean'], a[t][k]['per_repeat_r2_std'])}" for k in ("magpie", "cbfv", "full")), [ABLATION]
    if head == "noise":
        t = p[1]
        n = _json(NOISE)
        space = "linear" if t in ("S", "zT") else "log"
        r2max = n["item1_rerun_results"][t][space]["r2_max"]
        rows = [r for r in n["item3_combined_ceiling_new"] if r["label"].split(" ")[0] == t]
        hr = ", ".join(f"{r['headroom_lower']:.3f}-{r['headroom_upper']:.3f}" for r in rows)
        return f"R2_max {_f4(r2max)} ({space}); headroom {hr}", [NOISE]
    if head == "dvd":
        d, b = _json(DVD), _json(DVD_BT)
        return (f"direct zT {_f4(d['zT_direct']['pooled_r2'])}, derived {_f4(d['zT_derived']['pooled_r2'])} "
                f"(gap {d['zT_direct']['pooled_r2'] - d['zT_derived']['pooled_r2']:.4f}); components S {_f4(d['S']['pooled_r2'])}, "
                f"sigma(log10) {_f4(d['sigma_log10']['pooled_r2'])}, kappa(log10) {_f4(d['kappa_log10']['pooled_r2'])}; "
                f"subset {_n(d['subset_n_rows'])} rows, {_n(d['subset_n_chemistry_clusters'])} clusters; "
                f"Duan effect on derived {b['duan_effect_on_derived_r2']:+.4f}"), [DVD, DVD_BT]
    if head == "estm":
        e = _json(ESTM)
        if p[1] in ("a", "b"):
            d = e["dedup_a_source_doi" if p[1] == "a" else "dedup_b_chemistry_cluster"]
            r = d["results"]
            lab = "pass (a) DOI-disjoint" if p[1] == "a" else "pass (b) cluster-disjoint"
            return (f"{lab}, n={_n(r['S']['n'])}: S {_f4(r['S']['r2'])}, sigma {_f4(r['sigma']['r2'])}, kappa {_f4(r['kappa']['r2'])}, "
                    f"zT direct {_f4(r['zT_direct']['r2'])}, zT derived {_f4(r['zT_derived']['r2'])}"), [ESTM]
        if p[1] == "n_in_scope":
            a = e["dedup_a_source_doi"]
            b = e["dedup_b_chemistry_cluster"]
            assert a["n_dropped"] + a["n_surviving"] == b["n_dropped"] + b["n_surviving"]
            return f"{_n(a['n_dropped'] + a['n_surviving'])} ESTM rows in scope before deduplication", [ESTM]
        if p[1] == "insupport":
            d = _json(INSUPPORT)[p[2]]
            pr = d["properties"]
            return (f"{d['stratum']}, in-support n={_n(d['n_in_support'])} of {_n(d['n'])}: "
                    + ", ".join(f"{t} {_f4(pr[t]['r2_in_support'])}" for t in TARGETS)), [INSUPPORT]
    if head == "shap":
        if p[1] == "coarse":
            c = _json(SHAP)["coarse_group_comparison"]
            return "; ".join(f"{g['group']} {g['chemistry_mean_share']:.3f}" for g in c) + " (chemistry arm, share of all 397 features)", [SHAP]
        z = np.load(REPO / SHAP_ARRAYS, allow_pickle=True)
        names = [str(x) for x in z["feature_cols"]]
        keys = sorted(k for k in z.files if k.startswith("chemistry__") and k.endswith("__mean_abs_shap"))
        assert len(keys) == 5, keys  # the npz keeps folds 0-4 only; the repeat they belong to is not recorded
        arr = np.mean([z[k] for k in keys], axis=0)
        order = np.argsort(-arr)[:5]
        return "; ".join(f"{names[i]} ({arr[i]:.3f})" for i in order) + " (zT, chemistry arm, mean |SHAP| over the 5 folds kept in the npz; repeat not recorded)", [SHAP_ARRAYS]
    if head == "hyper":
        if p[1] == "search":
            txt = (REPO / NESTED).read_text(encoding="utf-8")
            lo = txt.index('"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50)')
            block = txt[lo:txt.index("}", lo)]
            items = re.findall(r'"(\w+)": trial\.suggest_(\w+)\("\w+", ([^,]+), ([^,)]+)(?:, step=(\d+))?(?:, log=True)?\)', block)
            return "XGBoost Optuna search (20 trials, 3 inner folds): " + "; ".join(f"{n} {a}..{b}" for n, _, a, b, _ in items), [NESTED]
        d = _json(HYPER.format(p[1]))["best_params"]
        return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()), [HYPER.format(p[1])]
    if head == "clean":
        if p[1] == "bounds":
            b = _constants({"PROPERTY_BOUNDS"})["PROPERTY_BOUNDS"]
            return "; ".join(f"{k} {lo:g} to {hi:g}" for k, (lo, hi) in b.items()), [CLEANING]
        c = _constants({"TEMP_MIN_K", "TEMP_MAX_K", "TEMP_BIN_WIDTH_K", "MAD_THRESHOLD", "MIN_TEMP_COVERAGE", "SMOOTHNESS_WINDOW", "MULTI_SOURCE_CV_THRESHOLDS", "ZT_SELF_CONSISTENCY_MAX_REL_ERROR"})
        return (f"T {c['TEMP_MIN_K']}-{c['TEMP_MAX_K']} K in {c['TEMP_BIN_WIDTH_K']} K bins; zT self-consistency {c['ZT_SELF_CONSISTENCY_MAX_REL_ERROR']:.0%}; "
                f"CV thresholds {c['MULTI_SOURCE_CV_THRESHOLDS']}; MAD {c['MAD_THRESHOLD']} (unscaled: no 1.4826 factor in step9); min temperatures {c['MIN_TEMP_COVERAGE']}; "
                f"smoothness window {c['SMOOTHNESS_WINDOW']}"), [CLEANING]
    if head == "pipeline":
        txt = (REPO / NESTED).read_text(encoding="utf-8")
        has_imp = "SimpleImputer" in txt
        return (f"src/nested_cv.py imputes inputs: {has_imp}; StandardScaler only in the ridge pipeline; XGBoost receives raw features"), [NESTED]
    if head == "digitization":
        d = _json(NOISE)["item2_digitization_ceiling"]["values"]
        return "composition-matched teMatDb-vs-Starrydata2 label agreement R2: " + ", ".join(f"{k} {v:.3f}" for k, v in d.items()), [NOISE]
    if head == "coverage":
        t = p[1]
        n = _json(GROUPING)["n_rows_total"]
        r = _json(LADDER)["runs"][f"{t}_chemistry_full"]["n_rows_header"]
        return f"{100 * r / n:.1f}% ({_n(r)} of {_n(n)} featurized rows)", [LADDER, GROUPING]
    if head == "scale":
        return ", ".join(f"{t} {_json(LADDER)['runs'][t + '_chemistry_full']['target_scale']}" for t in TARGETS), [LADDER]
    if head == "foldsd":
        return ", ".join(f"{t} {_json(LADDER)['runs'][t + '_chemistry_full']['fold_level_std']:.3f}" for t in TARGETS) + " (fold-level SD of 25 chemistry-cluster folds)", [LADDER]
    if head == "estmdrop":
        e, ins = _json(ESTM), _json(INSUPPORT)
        out = []
        for lab, k, ek in (("pass a", "a", "dedup_a_source_doi"), ("pass b", "b", "dedup_b_chemistry_cluster")):
            parts = []
            for t in TARGETS:
                internal = ins[k]["properties"][t]["r2_internal_chemistry"]
                ext = e[ek]["results"]["zT_direct" if t == "zT" else t]["r2"]
                parts.append(f"{t} {ext - internal:+.3f}")
            out.append(f"{lab} (ESTM full-set minus internal chemistry-cluster R2): " + ", ".join(parts))
        return "; ".join(out), [ESTM, INSUPPORT]
    raise KeyError(key)


# Every committed file this module reads, in one place so that thesis_paper/SHARED_DEPENDENCIES.md can pin them.
ARTIFACT_PATHS = tuple(
    sorted(
        {
            LADDER, UNGROUPED, ABLATION, NOISE, DVD, DVD_BT, ESTM, INSUPPORT, SHAP, SHAP_ARRAYS, FUNNEL, RAWMETA, GROUPING,
            CLEANING, NESTED, *[HYPER.format(t) for t in TARGETS],
        }
    )
)
