"""
NA12: the parts of the pre-registered report that summary.json of the comparison run did not hold, computed from the run's own outputs (docs/decisions.md, pre-registration B item 4 and the freeze entry of 2026-10-19). No rerun of the comparison:
this script reads comparison.csv and the committed prediction files, and adds one block, "added_after_run", to summary.json (the file as the run wrote it is kept as summary_as_run.json).

1. log10 comparison (pre-registered, B item 4): for sigma and kappa, per paper and level, the number compared, the median absolute log10 ratio and the signed median (log10 of the prediction minus log10 of the
   literature value, from comparison.csv), the same for the neighbouring-bin run, and the overall summary of the independent papers (papers weigh equally: the median over papers of each paper's median).
   The relative-error summaries of summary.json are recomputed from comparison.csv and must equal those of the run (asserted).
2. Training-source papers (by DOI): the number of rows of the training table (the snapfix CSV, SHA256 checked) whose DOI is the paper's, in total, per property and as distinct compositions; and how many of
   the paper's compared values have a training row of the same DOI and composition.
3. NOT PRE-REGISTERED, added after the result, context only: per property, the median absolute relative error and, for sigma and kappa, the median absolute and signed log10 ratio of all committed chemistry-cluster
   out-of-fold predictions (the rung of the ladder, 5 repeats x 5 folds, every prediction counted), for comparison with the errors of NA12.

Usage (repository root):  python thesis_paper/scripts/na12_comparison_report.py --run-dir thesis_paper/results/na12_comparison/<UTC>
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import na12_literature_comparison as cmp  # noqa: E402
import run_record as rr  # noqa: E402

LEVELS = ("exact", "cluster")
LOG = ("sigma", "kappa")
NAMES = {"10.1006/jssc.1995.1384": "Ohtaki et al. 1995", "10.1007/s11664-009-0666-x": "Shibasaki et al. 2009", "10.1021/acsaem.2c01936": "Sebastia-Luna et al. 2022",
         "10.1039/d0ce00702a": "Rajasekaran et al. 2020", "10.1063/1.1847723": "Ohta et al. 2005", "10.1080/14786435.2016.1263404": "Singh and Pandey 2016", "10.1109/ict.2006.331291": "Flahaut et al. 2006"}


def load_result(run_dir):
    res = pd.read_csv(Path(run_dir) / "comparison.csv", dtype={"paper": str})
    for c in ("in_window", "cluster_only", "doi_in_training"):
        res[c] = res[c].astype(bool)
    for level in LEVELS:
        for f in ("inside_band",):
            for pre in (level, f"{level}_alt"):
                col = f"{pre}_{f}"
                res[col] = res[col].map(lambda x: x if isinstance(x, bool) else (True if x == "True" else (False if x == "False" else None))).astype(object)
    return res


def log_block(g):
    """Per property (sigma, kappa) and level: n, median |log10 ratio|, signed median."""
    out = {}
    for t in LOG:
        for level in LEVELS:
            s = g[(g["target"] == t) & (g[f"{level}_n_rows"].astype(float) > 0)]
            v = s[f"{level}_log10_ratio"].astype(float).dropna()
            out[f"{t}_{level}"] = {"n_compared": int(len(s)), "n_with_log10_ratio": int(len(v)), "median_abs_log10_ratio": float(v.abs().median()) if len(v) else None,
                                   "median_signed_log10_ratio": float(v.median()) if len(v) else None}
    return out


def log_summary(res):
    by_paper = {}
    for paper, g in res.groupby("paper"):
        by_paper[paper] = {"training_source": bool(g["doi_in_training"].all()), "by_property_and_level": log_block(g)}
    indep = {k: v for k, v in by_paper.items() if not v["training_source"]}
    overall = {"n_papers": len(indep), "papers": sorted(indep), "by_property_and_level": {}}
    for t in LOG:
        for level in LEVELS:
            k = f"{t}_{level}"
            a = [v["by_property_and_level"][k]["median_abs_log10_ratio"] for v in indep.values() if v["by_property_and_level"][k]["n_with_log10_ratio"]]
            s = [v["by_property_and_level"][k]["median_signed_log10_ratio"] for v in indep.values() if v["by_property_and_level"][k]["n_with_log10_ratio"]]
            overall["by_property_and_level"][k] = {"n_papers_with_comparisons": len(a), "median_over_papers_of_median_abs_log10_ratio": float(np.median(a)) if a else None,
                                                   "median_over_papers_of_signed_median_log10_ratio": float(np.median(s)) if s else None}
    return {"by_paper": by_paper, "overall_independent_papers": overall}


def check_against_run(res, summary):
    """The relative-error summaries recomputed from comparison.csv must equal those of the run."""
    again = cmp.summarise(res)
    for paper, b in summary["by_paper"].items():
        for k, v in b["by_property_and_level"].items():
            w = again["by_paper"][paper]["by_property_and_level"][k]
            assert v["n_compared"] == w["n_compared"] and v["n_inside_band"] == w["n_inside_band"], (paper, k, v, w)
            if v["median_abs_rel_error"] is not None:
                assert abs(v["median_abs_rel_error"] - w["median_abs_rel_error"]) < 1e-9 * max(1.0, abs(v["median_abs_rel_error"])), (paper, k)
    alt = cmp.swap_in_alt(res)
    assert summary["edge_sensitivity"]["neighbouring_bin_overall_independent_papers"]["n_papers"] == cmp.summarise(alt)["overall_independent_papers"]["n_papers"]
    return alt


def training_rows(dataset, res):
    """The rows of the training table of each training-source paper (by DOI)."""
    df = pd.read_csv(dataset, usecols=["DOI", "composition_id", *cmp.TARGETS])
    df["doi_n"] = df["DOI"].dropna().map(cmp.norm_doi)
    out = {}
    for paper, g in res.groupby("paper"):
        if not bool(g["doi_in_training"].all()):
            continue
        sub = df[df["doi_n"] == cmp.norm_doi(paper)]
        out[paper] = {"name": NAMES.get(paper, paper), "rows_in_training_table": int(len(sub)), "distinct_compositions": int(sub["composition_id"].nunique()),
                      "rows_with_a_value_of": {t: int(sub[t].notna().sum()) for t in cmp.TARGETS},
                      "compared_values_with_a_training_row_of_the_same_doi_and_composition": int(g["doi_and_composition_in_training"].astype(bool).sum()), "n_values": int(len(g))}
    return out


def rung_context():
    """All committed chemistry-cluster out-of-fold predictions (every repeat and fold), per property: median |relative error|, and for sigma and kappa the median |log10 ratio| and the signed median."""
    out = {}
    for t in cmp.TARGETS:
        d = REPO / cmp.LADDER / f"{t}_chemistry_full"
        yt, yp = [], []
        for p in sorted(d.glob("repeat*_fold*_predictions.npz")):
            u = np.load(p)
            yt.append(u["y_true"])
            yp.append(u["y_pred"])
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        lin_t, lin_p = (10.0 ** yt, 10.0 ** yp) if t in cmp.LOG_TARGETS else (yt, yp)
        ok = lin_t != 0
        rel = np.abs((lin_p[ok] - lin_t[ok]) / np.abs(lin_t[ok]))
        rec = {"n_predictions": int(len(yt)), "n_with_nonzero_true_value": int(ok.sum()), "median_abs_rel_error": float(np.median(rel)),
               "fraction_inside_alleno_band": float(np.mean(rel <= cmp.read_uncertainties()[t]))}
        if t in cmp.LOG_TARGETS:
            r = yp - yt
            rec.update({"median_abs_log10_ratio": float(np.median(np.abs(r))), "median_signed_log10_ratio": float(np.median(r))})
        out[t] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default=str(REPO / cmp.DATASET))
    args = ap.parse_args()
    d = Path(args.run_dir)
    prov = rr.provenance({"comparison_csv": str(d / "comparison.csv"), "summary_json": str(d / "summary.json"), "dataset": args.dataset, "comparison_script": "thesis_paper/scripts/na12_literature_comparison.py"}, __file__)
    assert prov["inputs"]["dataset"]["sha256"] == cmp.DATASET_SHA256, "not the snapfix CSV"
    summ_path, orig_path = d / "summary.json", d / "summary_as_run.json"
    if not orig_path.exists():
        shutil.copyfile(summ_path, orig_path)
    summary = json.loads(orig_path.read_text(encoding="utf-8"))  # always start from the file as the run wrote it
    res = load_result(d)
    alt = check_against_run(res, summary)
    block = {"provenance": {**{k: prov[k] for k in ("git_head", "tree_clean", "script", "script_sha256")}, "inputs": prov["inputs"], "summary_as_run_sha256": rr.sha256_file(orig_path)},
             "log10_comparison": {"headline": log_summary(res), "neighbouring_bin": {"overall_independent_papers": log_summary(alt)["overall_independent_papers"]},
                                  "definition": "log10 of the prediction minus log10 of the literature value (sigma and kappa), per value from comparison.csv; per paper and level the median of |.| and the signed median over the compared values; "
                                                "overall: the median over the independent papers of each paper's median (papers weigh equally)"},
             "training_source_rows": training_rows(args.dataset, res),
             "NOT_PRE_REGISTERED_context_rung_errors": {"note": "added after the result; every committed chemistry-cluster out-of-fold prediction counted (5 repeats x 5 folds), not averaged per bin as the NA12 predictions are",
                                                        "by_property": rung_context()}}
    summary["added_after_run"] = block
    rr.write_json(summ_path, summary)
    rr.write_json(d / "report_run_config.json", prov)
    print(json.dumps(block["training_source_rows"], indent=1))
    print(json.dumps(block["NOT_PRE_REGISTERED_context_rung_errors"], indent=1))
    print(json.dumps(block["log10_comparison"]["headline"]["overall_independent_papers"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
